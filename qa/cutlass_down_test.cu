/***************************************************************************************************
 * Standalone numeric + perf test for the MoE DOWN-projection SM100 grouped GEMM kernel.
 *   per expert e:  Y_e[Me, N] = X_e[Me, K] @ W_e[N, K]^T   (row-major X and Y; W [N,K] used Tᵀ)
 *   PLAIN GEMM — NO SwiGLU, NO gate/up, NO silu. fp32 host reference oracle (silu-free).
 *
 * Build (on a Blackwell box):
 *   nvcc -std=c++17 -arch=sm_100a --expt-relaxed-constexpr \
 *        -I<repo>/3rdparty/cutlass/include \
 *        -I<repo>/3rdparty/cutlass/tools/util/include \
 *        -I<repo>/transformer_engine/common \
 *        qa/cutlass_down_test.cu -o down_test && ./down_test
 *
 * argv = (Me K N [G])   defaults Me=768 K=512 N=2048 G=32  -> M = G*Me = 24576.
 *   Me = tokens/expert (uniform; TileM-aligned).  K (= I) = contraction = down-proj input width.
 *   N  (= H) = output width.  G = #experts.  M = G*Me total tokens.
 * Override tile knobs: -DTILEM=.. -DTILEN=.. -DTILEK=.. -DKSTAGES=.. -DACCSTAGES=..
 * Kernel constraints: ACCSTAGES*TILEN <= 512 and pow2 (TILEN in {16,32,64,128,256}); TILEN<=256.
 **************************************************************************************************/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Compile-time tunables for autotuning sweeps (mirror swiglu_single_expert_test.cu).
#ifndef TILEM
#define TILEM 256
#endif
#ifndef TILEN
#define TILEN 64
#endif
#ifndef TILEK
#define TILEK 64
#endif
#ifndef KSTAGES
#define KSTAGES 8
#endif
#ifndef ACCSTAGES
#define ACCSTAGES 2
#endif
#ifndef CLUSTERM
#define CLUSTERM 2
#endif

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"
#include "gemm/cutlass_grouped_gemm_down_v2.cuh"

// Override the tunables while keeping MinBlocks=1.
#define LAUNCH_DOWN(...)                                                                       \
  transformer_engine::grouped_gemm_down::LaunchDownGemmGroupedV2<Element, ElementOut, TILEM,   \
                                                                 TILEN, TILEK, KSTAGES,        \
                                                                 CLUSTERM, 1, ACCSTAGES>(__VA_ARGS__)

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

int main(int argc, char** argv) {
  // Down-proj real shape (G,M,N,K) = (32, 24576, 2048, 512): Me=768 K=512 N=2048 G=32.
  const int Me = (argc > 1) ? std::atoi(argv[1]) : 768;
  const int K = (argc > 2) ? std::atoi(argv[2]) : 512;   // contraction (== I, down-proj input width)
  const int N = (argc > 3) ? std::atoi(argv[3]) : 2048;  // output width (== H)
  const int G = (argc > 4) ? std::atoi(argv[4]) : 32;
  const int kTM = TILEM;  // experts are TileM-aligned (token-rounding, like SonicMoE)

  // UNIFORM, TileM-aligned experts: each expert owns Me tokens (Me must be a TILEM multiple for the
  // m_tile_expert table to be exact). Build the per-m-tile expert table + token->expert map.
  const int Me_aligned = ((Me + kTM - 1) / kTM) * kTM;
  const int M = G * Me_aligned;
  std::vector<int> mtile_expert_h;  // [num_m_tiles] global m-tile -> expert id
  std::vector<int> e_of_token(M);   // [M] token -> expert id (host reference)
  for (int e = 0; e < G; ++e) {
    for (int t = 0; t < Me_aligned / kTM; ++t) mtile_expert_h.push_back(e);
    for (int r = 0; r < Me_aligned; ++r) e_of_token[(size_t)e * Me_aligned + r] = e;
  }
  const int Wrows = G * N;  // stacked per-expert weights [G*N, K]
  const int Mref = (M < 512) ? M : 512;  // fp32 ref over the first ~512 rows
  std::printf("DOWN GROUPED G=%d Me=%d (M=%d) N=%d K=%d  num_m_tiles=%zu\n", G, Me_aligned, M, N, K,
              mtile_expert_h.size());

  int dev = 0;
  CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  std::printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  if (prop.major < 10)
    std::printf("WARNING: this kernel requires SM100 (Blackwell). Running anyway may fail.\n");

  // ---- host random inputs (fp32 master), then convert to bf16 ----
  std::srand(1234);
  std::vector<float> hX_f((size_t)M * K), hW_f((size_t)Wrows * K);
  auto rnd = []() { return (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; };
  for (auto& v : hX_f) v = rnd();
  for (auto& v : hW_f) v = rnd();

  std::vector<Element> hX((size_t)M * K), hW((size_t)Wrows * K);
  for (size_t i = 0; i < (size_t)M * K; ++i) hX[i] = static_cast<Element>(hX_f[i]);
  for (size_t i = 0; i < (size_t)Wrows * K; ++i) hW[i] = static_cast<Element>(hW_f[i]);

  // ---- device buffers ----
  Element *dX = nullptr, *dW = nullptr;
  ElementOut* dY = nullptr;
  CHECK_CUDA(cudaMalloc(&dX, sizeof(Element) * (size_t)M * K));
  CHECK_CUDA(cudaMalloc(&dW, sizeof(Element) * (size_t)Wrows * K));
  CHECK_CUDA(cudaMalloc(&dY, sizeof(ElementOut) * (size_t)M * N));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), sizeof(Element) * (size_t)M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW, hW.data(), sizeof(Element) * (size_t)Wrows * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dY, 0, sizeof(ElementOut) * (size_t)M * N));

  // Per-m-tile expert table (device). Required because Me may be a TileM multiple but the kernel
  // still keys W blocks by m-tile; pass it explicitly (it is the canonical varlen-safe path).
  int* d_mte = nullptr;
  CHECK_CUDA(cudaMalloc(&d_mte, sizeof(int) * mtile_expert_h.size()));
  CHECK_CUDA(cudaMemcpy(d_mte, mtile_expert_h.data(), sizeof(int) * mtile_expert_h.size(),
                        cudaMemcpyHostToDevice));

  // ---- launch ----
  cudaError_t st =
      LAUNCH_DOWN(dX, dW, dY, G, N, K, M, /*stream=*/0, dev,
                  /*sm_count=*/prop.multiProcessorCount, d_mte, /*Me=*/Me_aligned);
  if (st != cudaSuccess) std::printf("Launch returned: %s\n", cudaGetErrorString(st));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<ElementOut> hY((size_t)M * N);
  CHECK_CUDA(cudaMemcpy(hY.data(), dY, sizeof(ElementOut) * (size_t)M * N, cudaMemcpyDeviceToHost));

  // ---- fp32 host reference: Y[m,n] = sum_k X[m,k] * W[e*N + n, k]   (silu-free!) ----
  // W is RowMajor [G*N, K]: row r at offset r*K. expert e's rows = [e*N, (e+1)*N).
  // Y is RowMajor [M, N]: Y[m,n] at offset m*N + n.
  std::vector<float> ref((size_t)Mref * N);
  for (int m = 0; m < Mref; ++m) {
    const int e = e_of_token[m];
    const size_t wbase = (size_t)(e * N) * K;  // expert e's W block base
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        float x = static_cast<float>(hX[(size_t)m * K + kk]);
        acc += x * static_cast<float>(hW[wbase + (size_t)n * K + kk]);  // W[e*N + n, :]
      }
      ref[(size_t)m * N + n] = acc;
    }
  }

  // ---- error analysis (bf16 in/out + fp32 accumulate; abs-OR-rel tol 5e-2) ----
  double max_abs = 0.0, max_rel = 0.0;
  int n_close = 0, n_fail = 0;
  for (int m = 0; m < Mref; ++m) {
    for (int n = 0; n < N; ++n) {
      float r = ref[(size_t)m * N + n];
      float got = static_cast<float>(hY[(size_t)m * N + n]);
      float abs_err = std::fabs(got - r);
      float rel_err = abs_err / (std::fabs(r) + 1e-6f);
      if (abs_err > max_abs) max_abs = abs_err;
      if (rel_err > max_rel) max_rel = rel_err;
      if (abs_err < 5e-2f) ++n_close;
      if (abs_err > 5e-2f && rel_err > 5e-2f) ++n_fail;  // bf16: fail only if BOTH abs AND rel exceed
    }
  }

  std::printf("[correctness checked over first %d of %d rows]\n", Mref, M);
  std::printf("max_abs_err = %.6f   max_rel_err = %.6f\n", max_abs, max_rel);
  std::printf("elements within 5e-2: %d / %d (%.1f%%)\n", n_close, Mref * N,
              100.0 * n_close / ((double)Mref * N));
  // Coordinate-independent checksums (a permutation preserves sum + sum-of-squares).
  double sum_got = 0, sum_ref = 0, sq_got = 0, sq_ref = 0;
  for (size_t j = 0; j < (size_t)Mref * N; ++j) {
    double g = static_cast<float>(hY[j]), r = ref[j];
    sum_got += g; sum_ref += r; sq_got += g * g; sq_ref += r * r;
  }
  std::printf("CHECKSUM  sum: got=% .3f ref=% .3f | sumsq: got=% .3f ref=% .3f\n", sum_got, sum_ref,
              sq_got, sq_ref);
  // Row-0 dump + a few samples.
  std::printf("ROW0 ref:");
  for (int n = 0; n < 16; ++n) std::printf(" %7.3f", ref[n]);
  std::printf("\nROW0 got:");
  for (int n = 0; n < 16; ++n) std::printf(" %7.3f", static_cast<float>(hY[n]));
  std::printf("\n");
  int sm[] = {0, 1, 32, 64, 96, 120, 126, 127};
  for (int si = 0; si < 8; ++si) {
    int m = sm[si];
    if (m >= Mref) continue;
    std::printf("  (m=%3d,n=0) got=% .4f ref=% .4f | (n=1) got=% .4f ref=% .4f\n", m,
                static_cast<float>(hY[(size_t)m * N + 0]), ref[(size_t)m * N + 0],
                static_cast<float>(hY[(size_t)m * N + 1]), ref[(size_t)m * N + 1]);
  }
  bool pass = (n_fail == 0);
  std::printf("n_fail (abs>5e-2 AND rel>5e-2): %d / %d\n", n_fail, Mref * N);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  // ---- perf timing (50 warmup + 200 iters) ----
  const int warm = 50, iters = 200;
  for (int it = 0; it < warm; ++it)
    LAUNCH_DOWN(dX, dW, dY, G, N, K, M, 0, dev, prop.multiProcessorCount, d_mte, Me_aligned);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; ++it)
    LAUNCH_DOWN(dX, dW, dY, G, N, K, M, 0, dev, prop.multiProcessorCount, d_mte, Me_aligned);
  cudaEventRecord(e1);
  CHECK_CUDA(cudaEventSynchronize(e1));
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  double s = ms / 1e3 / iters;
  double flops = 2.0 * double(M) * N * K;  // ONE GEMM over all tokens: 2*M*N*K
  // HBM model: read X[M,K] + W[G*N,K] once, write Y[M,N] once (bf16).
  double bytes = 2.0 * (double(M) * K + double(Wrows) * K + double(M) * N);
  std::printf("DOWN  shape(M=%d,N=%d,K=%d)  %.4f ms/iter  %.1f TFLOPS  %.0f GB/s\n", M, N, K,
              ms / iters, flops / s / 1e12, bytes / s / 1e9);

  cudaFree(dX);
  cudaFree(dW);
  cudaFree(dY);
  cudaFree(d_mte);
  return pass ? 0 : 1;
}
