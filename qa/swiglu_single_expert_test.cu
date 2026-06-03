/***************************************************************************************************
 * Standalone single-expert numeric test for the SwiGLU-fused MoE up-proj SM100 kernel.
 *   A[M,I] = silu(X·W1[0:I]ᵀ) · (X·W1[I:2I]ᵀ),  X=[M,d], W1=[2I,d] (concat: gate||up), no permute.
 *
 * Build (on a Blackwell box):
 *   nvcc -std=c++17 -arch=sm_100a --expt-relaxed-constexpr \
 *        -I<repo>/3rdparty/cutlass/include \
 *        -I<repo>/3rdparty/cutlass/tools/util/include \
 *        -I<repo>/transformer_engine/common \
 *        qa/swiglu_single_expert_test.cu -o swiglu_test && ./swiglu_test
 *
 * Shapes are hard-coded to the P1 step-2 single-expert case: M=256, I=128, d=128.
 **************************************************************************************************/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Compile-time tunables for autotuning sweeps. Override with -DTILEM=.. -DTILEN=.. -DKSTAGES=.. -DACCSTAGES=..
// Defaults = the tuned config. Kernel constraints: ACCSTAGES*2*TILEN<=512 (pow2), TILEN<=256.
#ifndef TILEM
#define TILEM 256
#endif
#ifndef TILEN
#define TILEN 64
#endif
#ifndef TILEK
#define TILEK 16
#endif
#ifndef KSTAGES
#define KSTAGES 16
#endif
#ifndef ACCSTAGES
#define ACCSTAGES 2
#endif
// Override the tunables while keeping ClusterM=2, MinBlocks=1.
#define LAUNCH_GROUPED(...) \
  transformer_engine::grouped_gemm_swiglu::LaunchSwiGluGrouped<Element, ElementOut, TILEM, TILEN, TILEK, KSTAGES, 2, 1, ACCSTAGES>(__VA_ARGS__)

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"
#include "gemm/cutlass_grouped_gemm_swiglu.cuh"

using Element = cutlass::bfloat16_t;     // bf16 in
using ElementOut = cutlass::bfloat16_t;  // bf16 out

#define CHECK_CUDA(call)                                                                    \
  do {                                                                                      \
    cudaError_t _e = (call);                                                                \
    if (_e != cudaSuccess) {                                                                \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);  \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

static float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

int main(int argc, char** argv) {
  // Grouped MoE: argv = (Me I d [G]). User shape (G,M,N,K)=(32,16384,512,2048) ⇒ Me=512 I=512 d=2048 G=32.
  // Me=tokens/expert, I(=N)=ffn half-hidden / output width, d(=K)=model dim, G=#experts. M=G*Me total tokens.
  const int Me = (argc > 1) ? std::atoi(argv[1]) : 512;
  const int I = (argc > 2) ? std::atoi(argv[2]) : 512;
  const int d = (argc > 3) ? std::atoi(argv[3]) : 2048;
  const int G = (argc > 4) ? std::atoi(argv[4]) : 32;
  // VARLEN-M mode (env SWIGLU_VARLEN=<comma sizes>): build UNEVEN, TileM-aligned experts to exercise the
  // per-m-tile expert table + CLC. The env value is a per-expert token-count PATTERN cycled across G
  // experts, e.g. SWIGLU_VARLEN=1024,2048,3072 (each rounded up to a TILEM multiple). SWIGLU_VARLEN=1 =>
  // default 256,512,768. Unset => UNIFORM (every expert = Me).
  const char* vl = std::getenv("SWIGLU_VARLEN");
  const bool varlen = (vl != nullptr);
  const int kTM = TILEM;                  // experts are aligned to TileM (token-rounding, like SonicMoE)
  std::vector<int> expert_ntok(G);
  std::vector<int> mtile_expert_h;        // [num_m_tiles] global m-tile -> expert id
  std::vector<int> e_of_token;            // [M] token -> expert id (host reference)
  int M = 0;
  if (varlen) {
    std::vector<int> pat;
    char buf[256];
    std::strncpy(buf, vl, 255);
    buf[255] = 0;
    for (char* tok = std::strtok(buf, ","); tok; tok = std::strtok(nullptr, ",")) {
      int v = std::atoi(tok);
      if (v > 0) pat.push_back(((v + kTM - 1) / kTM) * kTM);  // round up to TILEM multiple
    }
    if (pat.empty()) { pat = {kTM, 2 * kTM, 3 * kTM}; }       // SWIGLU_VARLEN=1 => default pattern
    for (int e = 0; e < G; ++e) expert_ntok[e] = pat[e % pat.size()];
  } else {
    for (int e = 0; e < G; ++e) expert_ntok[e] = Me;
  }
  for (int e = 0; e < G; ++e) {
    M += expert_ntok[e];
    for (int t = 0; t < expert_ntok[e] / kTM; ++t) mtile_expert_h.push_back(e);
    for (int r = 0; r < expert_ntok[e]; ++r) e_of_token.push_back(e);
  }
  // Host fp32 ref is O(Mref*I*d): full check for small varlen (validates ALL experts incl. high indices),
  // else first 512 rows. (Run varlen with small I/d/G so Mref==M.)
  const int Mref = varlen ? (M <= 8192 ? M : 2048) : ((M < 512) ? M : 512);
  const int twoI = 2 * I;
  const int W1rows = G * twoI;            // stacked per-expert weights [G*2I, d]
  std::printf("GROUPED G=%d Me=%d (M=%d) I=%d d=%d  varlen=%d num_m_tiles=%zu\n", G, Me, M, I, d,
              (int)varlen, mtile_expert_h.size());

  int dev = 0;
  CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  std::printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  if (prop.major < 10) {
    std::printf("WARNING: this kernel requires SM100 (Blackwell). Running anyway may fail.\n");
  }

  // ---- host random inputs (fp32 master), then convert to bf16 ----
  std::srand(1234);
  std::vector<float> hX_f(M * d), hW1_f((size_t)W1rows * d);
  auto rnd = []() { return (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; };
  for (auto& v : hX_f) v = rnd();
  for (auto& v : hW1_f) v = rnd();

  std::vector<Element> hX(M * d), hW1((size_t)W1rows * d);
  for (size_t i = 0; i < (size_t)M * d; ++i) hX[i] = static_cast<Element>(hX_f[i]);
  for (size_t i = 0; i < (size_t)W1rows * d; ++i) hW1[i] = static_cast<Element>(hW1_f[i]);

  // ---- device buffers ----
  Element *dX = nullptr, *dW1 = nullptr;
  ElementOut* dA = nullptr;
  CHECK_CUDA(cudaMalloc(&dX, sizeof(Element) * (size_t)M * d));
  CHECK_CUDA(cudaMalloc(&dW1, sizeof(Element) * (size_t)W1rows * d));
  CHECK_CUDA(cudaMalloc(&dA, sizeof(ElementOut) * (size_t)M * I));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), sizeof(Element) * (size_t)M * d, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW1, hW1.data(), sizeof(Element) * (size_t)W1rows * d, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dA, 0, sizeof(ElementOut) * (size_t)M * I));

  // VARLEN-M: upload the per-m-tile expert table (device); nullptr in uniform mode.
  int* d_mte = nullptr;
  if (varlen) {
    CHECK_CUDA(cudaMalloc(&d_mte, sizeof(int) * mtile_expert_h.size()));
    CHECK_CUDA(cudaMemcpy(d_mte, mtile_expert_h.data(), sizeof(int) * mtile_expert_h.size(),
                          cudaMemcpyHostToDevice));
  }
  const int* mte_arg = varlen ? d_mte : nullptr;
  const int M_arg = varlen ? M : 0;

  // ---- launch (grouped) ----
  cudaError_t st = LAUNCH_GROUPED(
      dX, dW1, dA, G, Me, I, d, /*stream=*/0, dev, /*sm_count=*/prop.multiProcessorCount, mte_arg, M_arg);
  if (st != cudaSuccess) {
    std::printf("Launch returned: %s\n", cudaGetErrorString(st));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<ElementOut> hA(M * I);
  CHECK_CUDA(cudaMemcpy(hA.data(), dA, sizeof(ElementOut) * M * I, cudaMemcpyDeviceToHost));

  // ---- fp32 host reference: gate = X @ W1[:I].T, up = X @ W1[I:].T, A = silu(gate)*up ----
  // W1 is RowMajor [2I,d]: row r at offset r*d. gate rows = [0..I), up rows = [I..2I).
  // A is RowMajor [M,I]: A[m,n] at offset m*I + n.
  std::vector<float> ref(Mref * I);
  for (int m = 0; m < Mref; ++m) {
    const int e = varlen ? e_of_token[m] : (m / Me);  // expert of token m → weights W1[e*2I .. (e+1)*2I)
    const size_t gbase = (size_t)(e * twoI) * d;       // gate rows [e*2I, e*2I+I)
    const size_t ubase = (size_t)(e * twoI + I) * d;   // up   rows [e*2I+I, e*2I+2I)
    for (int n = 0; n < I; ++n) {
      float gate = 0.0f, up = 0.0f;
      for (int kk = 0; kk < d; ++kk) {
        float x = static_cast<float>(hX[(size_t)m * d + kk]);
        gate += x * static_cast<float>(hW1[gbase + (size_t)n * d + kk]);   // W1[e*2I + n, :]
        up += x * static_cast<float>(hW1[ubase + (size_t)n * d + kk]);     // W1[e*2I + I + n, :]
      }
#if defined(SWIGLU_DEBUG_RAW_GATE)
      ref[m * I + n] = gate;            // ISOLATION: compare against raw gate = X·W1[0:I]ᵀ
#elif defined(SWIGLU_DEBUG_RAW_UP)
      ref[m * I + n] = up;              // ISOLATION: compare against raw up = X·W1[I:2I]ᵀ
#else
      ref[m * I + n] = silu_f(gate) * up;
#endif
    }
  }

  // Overall + per-row-half error (leader CTA rows [0,128) vs follower rows [128,256)).
  double max_abs = 0.0, max_rel = 0.0;
  double max_abs_lo = 0.0, max_abs_hi = 0.0;  // lo = rows[0,M/2), hi = rows[M/2,M)
  int n_close = 0;                            // count of |err|<5e-2
  int n_fail = 0;                             // bf16 criterion: fails BOTH abs AND rel tol
  for (int m = 0; m < Mref; ++m) {
    for (int n = 0; n < I; ++n) {
      float r = ref[m * I + n];
      float got = static_cast<float>(hA[m * I + n]);
      float abs_err = std::fabs(got - r);
      float rel_err = abs_err / (std::fabs(r) + 1e-6f);
      if (abs_err > max_abs) max_abs = abs_err;
      if (rel_err > max_rel) max_rel = rel_err;
      if (m < Mref / 2) { if (abs_err > max_abs_lo) max_abs_lo = abs_err; }
      else             { if (abs_err > max_abs_hi) max_abs_hi = abs_err; }
      if (abs_err < 5e-2f) ++n_close;
      // bf16 elementwise tolerance: OK if absolute OR relative is small. SwiGLU outputs reach ~|24|,
      // where bf16 rounding gives abs~0.3 at rel~1% — a pure-abs threshold wrongly flags those.
      if (abs_err > 5e-2f && rel_err > 5e-2f) ++n_fail;
    }
  }

  std::printf("[correctness checked over first %d of %d rows]\n", Mref, M);
  std::printf("max_abs_err = %.6f   max_rel_err = %.6f\n", max_abs, max_rel);
  std::printf("elements within 5e-2: %d / %d (%.1f%%)\n", n_close, Mref * I,
              100.0 * n_close / (Mref * I));
  // Coordinate-independent checksums: a PERMUTATION (read OK, scatter coord wrong) preserves
  // both sum and sum-of-squares; WRONG VALUES (GEMM/read wrong) do not.
  double sum_got = 0, sum_ref = 0, sq_got = 0, sq_ref = 0;
  for (int j = 0; j < Mref * I; ++j) {
    double g = static_cast<float>(hA[j]), r = ref[j];
    sum_got += g; sum_ref += r; sq_got += g * g; sq_ref += r * r;
  }
  std::printf("CHECKSUM  sum: got=% .3f ref=% .3f | sumsq: got=% .3f ref=% .3f\n",
              sum_got, sum_ref, sq_got, sq_ref);
  // Row-0 dump (got vs ref) to cross-reference the kernel's T0 printf (datapath-0 = row 0).
  std::printf("ROW0 ref:");
  for (int n = 0; n < 16; ++n) std::printf(" %6.3f", ref[0 * I + n]);
  std::printf("\nROW0 got:");
  for (int n = 0; n < 16; ++n) std::printf(" %6.3f", static_cast<float>(hA[0 * I + n]));
  std::printf("\nCOL0 ref:");
  for (int m = 0; m < 16; ++m) std::printf(" %6.3f", ref[m * I + 0]);
  std::printf("\n");
  // Sample dump: got vs ref at a few in-range (m,n).
  int sm[] = {0, 1, 32, 64, 96, 120, 126, 127};
  for (int si = 0; si < 8; ++si) {
    int m = sm[si];
    if (m >= M) continue;
    std::printf("  (m=%3d,n=0) got=% .4f ref=% .4f | (n=1) got=% .4f ref=% .4f\n", m,
                static_cast<float>(hA[m * I + 0]), ref[m * I + 0],
                static_cast<float>(hA[m * I + 1]), ref[m * I + 1]);
  }
  // bf16 in/out + fp32 accumulate: every element must pass abs-OR-rel tol (5e-2). Large-magnitude
  // SwiGLU outputs only meet the relative bound, so a pure-abs test is wrong.
  bool pass = (n_fail == 0);
  std::printf("n_fail (abs>5e-2 AND rel>5e-2): %d / %d\n", n_fail, Mref * I);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  // ---- perf timing (fused grouped kernel) ----
  const int warm = 10, iters = 100;
  for (int it = 0; it < warm; ++it)
    LAUNCH_GROUPED(
        dX, dW1, dA, G, Me, I, d, 0, dev, prop.multiProcessorCount, mte_arg, M_arg);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; ++it)
    LAUNCH_GROUPED(
        dX, dW1, dA, G, Me, I, d, 0, dev, prop.multiProcessorCount, mte_arg, M_arg);
  cudaEventRecord(e1);
  CHECK_CUDA(cudaEventSynchronize(e1));
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  double s = ms / 1e3 / iters;
  double flops = 4.0 * double(M) * I * d;  // gate + up GEMMs over all tokens, each 2*M*I*d
  // Fused 1·IT HBM model: read X[M,d] + W1[G*2I,d] once, write A[M,I] once (bf16).
  double bytes = 2.0 * (double(M) * d + 2.0 * double(W1rows) * d + double(M) * I);
  std::printf("FUSED  shape(M=%d,I=%d,d=%d)  %.4f ms/iter  %.1f TFLOPS  %.0f GB/s\n", M, I, d,
              ms / iters, flops / s / 1e12, bytes / s / 1e9);

  cudaFree(dX);
  cudaFree(dW1);
  cudaFree(dA);
  return pass ? 0 : 1;
}
