/***************************************************************************************************
 * Standalone E2E test: MoE grouped forward = UP-proj(+SwiGLU) THEN DOWN-proj, composed and TIMED
 * inside ONE cudaGraph.
 *
 *   Stage 1 (UP):   A[M,I] = SwiGLU( X[M,d] @ W1[G*2I, d]^T )   via LaunchSwiGluGroupedV2
 *                   (W1 host-permuted to the V2 gran-G interleaved layout, GLU_G=1, exactly like
 *                    swiglu_single_expert_test.cu's SWIGLU_V2 setup block)
 *   Stage 2 (DOWN): Y[M,d] = A[M,I] @ W2[G*d, I]^T              via LaunchDownGemmGroupedV2
 *                   (W2 plain [G*d, I]; CUTLASS N = d = 2048 (output width H), K = I = 512)
 *
 *   A is the OUTPUT of stage 1 AND the INPUT of stage 2 — the SAME device buffer dA.
 *
 *   Real 4K-MoE shape:  d = 2048 (model dim = up-K = down-N = H), I = 512 (ffn half = up-out = down-K),
 *                       Me = 768 tokens/expert, G = 32  ->  M = G*Me = 24576.
 *
 * Build (on a Blackwell SM100 box) — SAME -I flags as the other qa tests; note the V2 up-proj needs
 * the SWIGLU_V2 header, which is included unconditionally here:
 *   nvcc -std=c++17 -arch=sm_100a --expt-relaxed-constexpr \
 *        -I<repo>/3rdparty/cutlass/include \
 *        -I<repo>/3rdparty/cutlass/tools/util/include \
 *        -I<repo>/transformer_engine/common \
 *        qa/cutlass_e2e_cudagraph_test.cu -o e2e_test && ./e2e_test 768 512 2048 32
 *
 * argv = (Me I d [G])   defaults Me=768 I=512 d=2048 G=32  -> M = G*Me = 24576.
 *
 * Tile knobs (defaults = the known-best per-stage configs):
 *   UP  : -DTILEM_UP -DTILEN_UP -DTILEK_UP -DKSTAGES_UP -DACCSTAGES_UP
 *         default  TileM=256 TileN=64  TileK=64 KStages=8 AccStages=2   (the "1141" config)
 *   DOWN: -DTILEM_DN -DTILEN_DN -DTILEK_DN -DKSTAGES_DN -DACCSTAGES_DN
 *         default  TileM=128 TileN=256 TileK=64 KStages=8 AccStages=2   (the "740" config)
 *   GLU_G defaults to 1 (lowest R3 de-interleave risk); pass -DGLU_G=.. to sweep.
 **************************************************************************************************/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ===================== UP-proj (SwiGLU V2) tile knobs — the "1141" config =====================
#ifndef TILEM_UP
#define TILEM_UP 256
#endif
#ifndef TILEN_UP
#define TILEN_UP 64
#endif
#ifndef TILEK_UP
#define TILEK_UP 64
#endif
#ifndef KSTAGES_UP
#define KSTAGES_UP 8
#endif
#ifndef ACCSTAGES_UP
#define ACCSTAGES_UP 2
#endif

// ===================== DOWN-proj tile knobs — the "740" config ================================
#ifndef TILEM_DN
#define TILEM_DN 128
#endif
#ifndef TILEN_DN
#define TILEN_DN 256
#endif
#ifndef TILEK_DN
#define TILEK_DN 64
#endif
#ifndef KSTAGES_DN
#define KSTAGES_DN 8
#endif
#ifndef ACCSTAGES_DN
#define ACCSTAGES_DN 2
#endif

#ifndef CLUSTERM
#define CLUSTERM 2
#endif

// GLU_G controls the V2 up-proj gate/up interleave granularity (and the host permute below). The V2
// header #defines GLU_G=1 if unset, so set it BEFORE including the header to keep host+device in sync.
#ifndef GLU_G
#define GLU_G 1
#endif

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"
// The up-proj V2 header pulls in the V1 SwiGLU machinery (cutlass_grouped_gemm_swiglu.cuh) it depends
// on; the down header also includes V1. Include order is fine (both are #pragma once).
#include "gemm/cutlass_grouped_gemm_swiglu_v2.cuh"
#include "gemm/cutlass_grouped_gemm_down_v2.cuh"

using Element = cutlass::bfloat16_t;     // bf16 in
using ElementOut = cutlass::bfloat16_t;  // bf16 out (A and Y)

#define CHECK_CUDA(call)                                                                    \
  do {                                                                                      \
    cudaError_t _e = (call);                                                                \
    if (_e != cudaSuccess) {                                                                \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);  \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

static float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

// ----- the two launcher wrappers, bound to the per-stage tile knobs (mirror the sibling tests) -----
//
// UP:  LaunchSwiGluGroupedV2<Element,ElementOut, TILEM_UP,TILEN_UP,TILEK_UP,KSTAGES_UP, CLUSTERM,1,ACCSTAGES_UP>
//        (X, W1_il, A, G, Me, I, d, stream, dev, sm_count, m_tile_expert, M_varlen,
//         m_gather_idx=nullptr, T_src=0, prob=nullptr)
// DOWN:LaunchDownGemmGroupedV2<Element,ElementOut, TILEM_DN,TILEN_DN,TILEK_DN,KSTAGES_DN, CLUSTERM,1,ACCSTAGES_DN>
//        (A, W2, Y, G, N=d, K=I, M_total, stream, dev, sm_count, m_tile_expert, Me)
#define LAUNCH_UP(...)                                                                          \
  transformer_engine::grouped_gemm_swiglu::LaunchSwiGluGroupedV2<                               \
      Element, ElementOut, TILEM_UP, TILEN_UP, TILEK_UP, KSTAGES_UP, CLUSTERM, 1, ACCSTAGES_UP>(\
      __VA_ARGS__)

#define LAUNCH_DOWN(...)                                                                        \
  transformer_engine::grouped_gemm_down::LaunchDownGemmGroupedV2<                               \
      Element, ElementOut, TILEM_DN, TILEN_DN, TILEK_DN, KSTAGES_DN, CLUSTERM, 1, ACCSTAGES_DN>(\
      __VA_ARGS__)

int main(int argc, char** argv) {
  // Grouped MoE E2E: argv = (Me I d [G]). Real 4K-MoE -> Me=768 I=512 d=2048 G=32, M=G*Me=24576.
  //   Me = tokens/expert (uniform; rounded up to a TileM multiple for BOTH stages).
  //   I  = ffn half-hidden = up-proj output width = down-proj contraction (K).
  //   d  = model dim = up-proj contraction (K) = down-proj output width (N = H).
  const int Me_in = (argc > 1) ? std::atoi(argv[1]) : 768;
  const int I = (argc > 2) ? std::atoi(argv[2]) : 512;
  const int d = (argc > 3) ? std::atoi(argv[3]) : 2048;
  const int G = (argc > 4) ? std::atoi(argv[4]) : 32;

  // ---- TileM-alignment. The two stages may use DIFFERENT TileM (UP=256, DOWN=128). The
  // m_tile_expert table is keyed by m-tile, so it is DIFFERENT per stage (different num_m_tiles).
  // We build BOTH tables. Me is rounded up to lcm-friendly: the max of the two TileMs so that BOTH
  // tables are exact (every expert spans a whole number of m-tiles in either tiling).  M is identical
  // for both stages (only the m-tile DECOMPOSITION differs).  <<E2E
  const int kTM_up = TILEM_UP;
  const int kTM_dn = TILEM_DN;
  const int kTM_max = (kTM_up > kTM_dn) ? kTM_up : kTM_dn;
  const int Me = ((Me_in + kTM_max - 1) / kTM_max) * kTM_max;  // multiple of BOTH TileMs
  const int M = G * Me;

  const int twoI = 2 * I;
  const int W1rows = G * twoI;  // up weights stacked [G*2I, d]
  const int W2rows = G * d;     // down weights stacked [G*d, I]   (N=d output rows per expert)

  // Per-stage m_tile_expert tables.
  std::vector<int> mte_up_h, mte_dn_h;
  std::vector<int> e_of_token(M);
  for (int e = 0; e < G; ++e) {
    for (int t = 0; t < Me / kTM_up; ++t) mte_up_h.push_back(e);
    for (int t = 0; t < Me / kTM_dn; ++t) mte_dn_h.push_back(e);
    for (int r = 0; r < Me; ++r) e_of_token[(size_t)e * Me + r] = e;
  }

  std::printf(
      "E2E GROUPED  G=%d Me=%d (M=%d)  I=%d d=%d  |  UP[TM=%d TN=%d TK=%d KS=%d AS=%d GLU_G=%d] "
      "DOWN[TM=%d TN=%d TK=%d KS=%d AS=%d]\n",
      G, Me, M, I, d, TILEM_UP, TILEN_UP, TILEK_UP, KSTAGES_UP, ACCSTAGES_UP, GLU_G, TILEM_DN,
      TILEN_DN, TILEK_DN, KSTAGES_DN, ACCSTAGES_DN);
  std::printf("  num_m_tiles up=%zu down=%zu (M-tile decomposition differs; M is shared)\n",
              mte_up_h.size(), mte_dn_h.size());

  int dev = 0;
  CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  std::printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  if (prop.major < 10)
    std::printf("WARNING: this kernel requires SM100 (Blackwell). Running anyway may fail.\n");

  // ---- host random fp32 masters -> bf16 (same rnd as the two sibling tests) ----
  std::srand(1234);
  std::vector<float> hX_f((size_t)M * d), hW1_f((size_t)W1rows * d), hW2_f((size_t)W2rows * I);
  auto rnd = []() { return (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; };
  for (auto& v : hX_f) v = rnd();
  for (auto& v : hW1_f) v = rnd();
  for (auto& v : hW2_f) v = rnd();

  std::vector<Element> hX((size_t)M * d), hW1((size_t)W1rows * d), hW2((size_t)W2rows * I);
  for (size_t i = 0; i < (size_t)M * d; ++i) hX[i] = static_cast<Element>(hX_f[i]);
  for (size_t i = 0; i < (size_t)W1rows * d; ++i) hW1[i] = static_cast<Element>(hW1_f[i]);
  for (size_t i = 0; i < (size_t)W2rows * I; ++i) hW2[i] = static_cast<Element>(hW2_f[i]);

  // ---- device buffers. dA is BOTH stage1 output AND stage2 input (same buffer).  <<E2E ----
  Element *dX = nullptr, *dW1 = nullptr, *dW2 = nullptr;
  ElementOut *dA = nullptr, *dY = nullptr;
  CHECK_CUDA(cudaMalloc(&dX, sizeof(Element) * (size_t)M * d));
  CHECK_CUDA(cudaMalloc(&dW1, sizeof(Element) * (size_t)W1rows * d));
  CHECK_CUDA(cudaMalloc(&dW2, sizeof(Element) * (size_t)W2rows * I));
  CHECK_CUDA(cudaMalloc(&dA, sizeof(ElementOut) * (size_t)M * I));  // A[M,I] (stage1 out / stage2 in)
  CHECK_CUDA(cudaMalloc(&dY, sizeof(ElementOut) * (size_t)M * d));  // Y[M,d] (stage2 out)
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), sizeof(Element) * (size_t)M * d, cudaMemcpyHostToDevice));

  // ---- UP W1: host-permute to the V2 gran-G interleaved layout (VERBATIM from
  // swiglu_single_expert_test.cu's #if defined(SWIGLU_V2) block). The fp32 ref below reads the PLAIN
  // hW1 as the oracle, so the interleave only changes the DEVICE input.  <<E2E ----
  {
    const int kOutN_h = TILEN_UP;      // output n-tile == V1 TileN
    const int kMmaN_h = 2 * TILEN_UP;  // interleaved MMA width
    const int Gint = GLU_G;            // interleave gran (must match the GLU_G the V2 kernel compiled with)
    std::vector<Element> hW1_il((size_t)W1rows * d);
    for (int e = 0; e < G; ++e)
      for (int t = 0; t < I / kOutN_h; ++t)
        for (int c = 0; c < kOutN_h; ++c) {
          int g = c / Gint, j = c % Gint;
          size_t dst_gate = (size_t)(e * twoI + t * kMmaN_h + (2 * g * Gint + j));
          size_t dst_up = (size_t)(e * twoI + t * kMmaN_h + (2 * g * Gint + Gint + j));
          size_t src_gate = (size_t)(e * twoI + t * kOutN_h + c);    // plain gate row
          size_t src_up = (size_t)(e * twoI + I + t * kOutN_h + c);  // plain up   row
          std::memcpy(&hW1_il[dst_gate * d], &hW1[src_gate * d], sizeof(Element) * d);
          std::memcpy(&hW1_il[dst_up * d], &hW1[src_up * d], sizeof(Element) * d);
        }
    CHECK_CUDA(cudaMemcpy(dW1, hW1_il.data(), sizeof(Element) * (size_t)W1rows * d,
                          cudaMemcpyHostToDevice));
  }
  // ---- DOWN W2: plain [G*d, I] (VERBATIM upload from cutlass_down_test.cu; no permute).  <<E2E ----
  CHECK_CUDA(cudaMemcpy(dW2, hW2.data(), sizeof(Element) * (size_t)W2rows * I, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dA, 0, sizeof(ElementOut) * (size_t)M * I));
  CHECK_CUDA(cudaMemset(dY, 0, sizeof(ElementOut) * (size_t)M * d));

  // ---- per-stage m_tile_expert tables (device) ----
  int *d_mte_up = nullptr, *d_mte_dn = nullptr;
  CHECK_CUDA(cudaMalloc(&d_mte_up, sizeof(int) * mte_up_h.size()));
  CHECK_CUDA(cudaMalloc(&d_mte_dn, sizeof(int) * mte_dn_h.size()));
  CHECK_CUDA(cudaMemcpy(d_mte_up, mte_up_h.data(), sizeof(int) * mte_up_h.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_mte_dn, mte_dn_h.data(), sizeof(int) * mte_dn_h.size(),
                        cudaMemcpyHostToDevice));

  const int sm = prop.multiProcessorCount;

  // =============================================================================================
  // §1 · CORRECTNESS (eager, no graph): run stage1 then stage2 ONCE, then compose the fp32 ref.
  //   stage1 oracle: A_ref[m,n] = silu(X·W1gate^T) * (X·W1up^T)  using PLAIN hW1  (like swiglu test)
  //   stage2 oracle: Y_ref[m,p] = sum_n A_ref[m,n] * W2[e*d + p, n]                (silu-free, down test)
  //   A flows into Y: we compute A_ref in fp32 first, then feed it to the Y_ref GEMM.
  // =============================================================================================
  {
    cudaError_t s1 = LAUNCH_UP(dX, dW1, dA, G, Me, I, d, /*stream=*/0, dev, sm, d_mte_up,
                               /*M_varlen=*/M, /*m_gather_idx=*/nullptr, /*T_src=*/0,
                               /*prob=*/nullptr);
    if (s1 != cudaSuccess) std::printf("UP launch returned: %s\n", cudaGetErrorString(s1));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t s2 = LAUNCH_DOWN(dA, dW2, dY, G, /*N=*/d, /*K=*/I, /*M_total=*/M, /*stream=*/0, dev,
                                 sm, d_mte_dn, /*Me=*/Me);
    if (s2 != cudaSuccess) std::printf("DOWN launch returned: %s\n", cudaGetErrorString(s2));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // fp32 reference over the first Mref rows of Y.
  const int Mref = (M < 256) ? M : 256;
  std::vector<ElementOut> hY((size_t)Mref * d);
  // copy only the first Mref rows of Y back (Y is row-major [M,d]).
  CHECK_CUDA(cudaMemcpy(hY.data(), dY, sizeof(ElementOut) * (size_t)Mref * d, cudaMemcpyDeviceToHost));

  std::vector<float> A_ref((size_t)Mref * I);  // stage1 fp32 result for the first Mref rows
  for (int m = 0; m < Mref; ++m) {
    const int e = e_of_token[m];                 // expert of token m
    const size_t gbase = (size_t)(e * twoI) * d;       // gate rows [e*2I, e*2I+I)
    const size_t ubase = (size_t)(e * twoI + I) * d;   // up   rows [e*2I+I, e*2I+2I)
    for (int n = 0; n < I; ++n) {
      float gate = 0.0f, up = 0.0f;
      for (int kk = 0; kk < d; ++kk) {
        float x = static_cast<float>(hX[(size_t)m * d + kk]);
        gate += x * static_cast<float>(hW1[gbase + (size_t)n * d + kk]);  // W1[e*2I + n, :]
        up += x * static_cast<float>(hW1[ubase + (size_t)n * d + kk]);    // W1[e*2I + I + n, :]
      }
      // ROUND to bf16: the device writes A in bf16 (stage1 output dtype), so the stage2 oracle must
      // consume the SAME bf16-rounded A -- else the 2-stage fp32 ref diverges from the device by the
      // amplified bf16 intermediate rounding (was the ~2% n_fail; a precision artifact, not a bug).
      A_ref[(size_t)m * I + n] = static_cast<float>(static_cast<Element>(silu_f(gate) * up));
    }
  }
  std::vector<float> Y_ref((size_t)Mref * d);  // stage2 fp32 result = A_ref · W2^T (silu-free)
  for (int m = 0; m < Mref; ++m) {
    const int e = e_of_token[m];
    const size_t wbase = (size_t)(e * d) * I;  // W2 expert block base: rows [e*d, (e+1)*d), each I wide
    for (int p = 0; p < d; ++p) {
      float acc = 0.0f;
      for (int n = 0; n < I; ++n)
        acc += A_ref[(size_t)m * I + n] * static_cast<float>(hW2[wbase + (size_t)p * I + n]);
      Y_ref[(size_t)m * d + p] = acc;
    }
  }

  double max_abs = 0.0, max_rel = 0.0;
  int n_close = 0, n_fail = 0;
  for (int m = 0; m < Mref; ++m) {
    for (int p = 0; p < d; ++p) {
      float r = Y_ref[(size_t)m * d + p];
      float got = static_cast<float>(hY[(size_t)m * d + p]);
      float abs_err = std::fabs(got - r);
      float rel_err = abs_err / (std::fabs(r) + 1e-6f);
      if (abs_err > max_abs) max_abs = abs_err;
      if (rel_err > max_rel) max_rel = rel_err;
      if (abs_err < 5e-2f) ++n_close;
      if (abs_err > 5e-2f && rel_err > 5e-2f) ++n_fail;  // bf16: fail only if BOTH abs AND rel exceed
    }
  }
  std::printf("[E2E correctness checked over first %d of %d rows of Y]\n", Mref, M);
  std::printf("max_abs_err = %.6f   max_rel_err = %.6f\n", max_abs, max_rel);
  std::printf("elements within 5e-2: %d / %d (%.1f%%)\n", n_close, Mref * d,
              100.0 * n_close / ((double)Mref * d));
  std::printf("ROW0 Y_ref:");
  for (int p = 0; p < 12; ++p) std::printf(" %7.3f", Y_ref[p]);
  std::printf("\nROW0 Y_got:");
  for (int p = 0; p < 12; ++p) std::printf(" %7.3f", static_cast<float>(hY[p]));
  std::printf("\n");
  bool pass = (n_fail == 0);
  std::printf("n_fail (abs>5e-2 AND rel>5e-2): %d / %d\n", n_fail, Mref * d);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  // =============================================================================================
  // §2 · EAGER perf: 20 warmup + 100 iters of {stage1; stage2;}  (single default-stream timeline).
  // =============================================================================================
  const int warm_eager = 20, iters_eager = 100;
  for (int it = 0; it < warm_eager; ++it) {
    LAUNCH_UP(dX, dW1, dA, G, Me, I, d, 0, dev, sm, d_mte_up, M, nullptr, 0, nullptr);
    LAUNCH_DOWN(dA, dW2, dY, G, d, I, M, 0, dev, sm, d_mte_dn, Me);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CHECK_CUDA(cudaEventCreate(&e0));
  CHECK_CUDA(cudaEventCreate(&e1));
  CHECK_CUDA(cudaEventRecord(e0));
  for (int it = 0; it < iters_eager; ++it) {
    LAUNCH_UP(dX, dW1, dA, G, Me, I, d, 0, dev, sm, d_mte_up, M, nullptr, 0, nullptr);
    LAUNCH_DOWN(dA, dW2, dY, G, d, I, M, 0, dev, sm, d_mte_dn, Me);
  }
  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));
  float ms_eager = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms_eager, e0, e1));
  const double ms_eager_iter = ms_eager / iters_eager;

  // =============================================================================================
  // §3 · CUDA-GRAPH perf.
  //   The launchers do host-side work that is NOT safe to capture: cudaFuncSetAttribute,
  //   cudaOccupancyMaxActiveClusters, and (fallback) query_device_multiprocessor_count. We pass an
  //   explicit sm_count (=prop.multiProcessorCount) so the fallback device query is avoided, and we
  //   WARM UP the two launches on the capture stream a few times BEFORE BeginCapture so the one-time
  //   attribute set has happened (the attribute set is idempotent, but it issues a runtime call that
  //   should not land inside the capture).  See the deliverable report's ranked risk list — if any of
  //   these host calls still trip the capture, hoist them out of the launchers (the human will).
  // =============================================================================================
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup ON the capture stream (runs cudaFuncSetAttribute / cudaOccupancyMaxActiveClusters once).
  for (int it = 0; it < 5; ++it) {
    LAUNCH_UP(dX, dW1, dA, G, Me, I, d, stream, dev, sm, d_mte_up, M, nullptr, 0, nullptr);
    LAUNCH_DOWN(dA, dW2, dY, G, d, I, M, stream, dev, sm, d_mte_dn, Me);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Capture the two launches into ONE graph. ThreadLocal mode so a stray runtime call inside a
  // launcher (if any) does not abort the whole-process capture; it will instead error this stream.
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  cudaError_t up_st = LAUNCH_UP(dX, dW1, dA, G, Me, I, d, stream, dev, sm, d_mte_up, M, nullptr, 0,
                                nullptr);
  cudaError_t dn_st = LAUNCH_DOWN(dA, dW2, dY, G, d, I, M, stream, dev, sm, d_mte_dn, Me);
  cudaError_t cap_end = cudaStreamEndCapture(stream, &graph);
  if (up_st != cudaSuccess)
    std::printf("UP launch (during capture) returned: %s\n", cudaGetErrorString(up_st));
  if (dn_st != cudaSuccess)
    std::printf("DOWN launch (during capture) returned: %s\n", cudaGetErrorString(dn_st));
  if (cap_end != cudaSuccess) {
    std::printf(
        "cudaStreamEndCapture FAILED: %s\n  -> a launcher made a capture-unsafe runtime call inside "
        "the capture region (likely cudaFuncSetAttribute / cudaOccupancyMaxActiveClusters). Hoist "
        "that call out of the launcher (see report) and retry.\n",
        cudaGetErrorString(cap_end));
    std::exit(1);
  }
  CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  const int warm_graph = 20, iters_graph = 200;
  for (int it = 0; it < warm_graph; ++it) {
    CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  cudaEvent_t g0, g1;
  CHECK_CUDA(cudaEventCreate(&g0));
  CHECK_CUDA(cudaEventCreate(&g1));
  CHECK_CUDA(cudaEventRecord(g0, stream));
  for (int it = 0; it < iters_graph; ++it) CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
  CHECK_CUDA(cudaEventRecord(g1, stream));
  CHECK_CUDA(cudaEventSynchronize(g1));
  float ms_graph = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms_graph, g0, g1));
  const double ms_graph_iter = ms_graph / iters_graph;

  // =============================================================================================
  // §4 · Report.  FLOP = up(2*M*2I*d) + down(2*M*d*I) = 4*M*I*d + 2*M*d*I = 6*M*I*d.
  // =============================================================================================
  const double flop = 4.0 * double(M) * I * d + 2.0 * double(M) * d * I;  // = 6*M*I*d
  const double tf_eager = flop / (ms_eager_iter / 1e3) / 1e12;
  const double tf_graph = flop / (ms_graph_iter / 1e3) / 1e12;
  std::printf("E2E up+down  eager %.4f ms (%.1f TF) | graph %.4f ms (%.1f TF)  | speedup %.3fx\n",
              ms_eager_iter, tf_eager, ms_graph_iter, tf_graph, ms_eager_iter / ms_graph_iter);

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(g0);
  cudaEventDestroy(g1);
  cudaFree(dX);
  cudaFree(dW1);
  cudaFree(dW2);
  cudaFree(dA);
  cudaFree(dY);
  cudaFree(d_mte_up);
  cudaFree(d_mte_dn);
  return pass ? 0 : 1;
}
