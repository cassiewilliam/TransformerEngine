/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// V2b path ENABLED: read CONTIGUOUS un-permuted W1 (Muon-safe); the 5D-TMA box geometry produces the
// gran-8 gate/up interleave on the fly (NO host permute). Matches the V1 marshalling's W1 layout.
#define SWIGLU_V2_5D_TMA

// SonicMoE F2 — fused up-proj + SwiGLU grouped GEMM, V2 C-API definition (SM100 / Blackwell).
//
// V2 = the MegaMoE gran-8 gate/up INTERLEAVE design: ONE wide GEMM (N=2*TileN) into ONE interleaved
// accumulator + a de-interleave epilogue (silu(gate)*up by column), replacing V1's two W1-TMAs / two
// GEMMs / two accs. See cutlass_grouped_gemm_swiglu_v2.cuh for the change-spec + the open risks.
//
// This is the ONLY translation unit that instantiates the heavy CUTLASS SwiGLU-V2 kernel
// (cutlass_grouped_gemm_swiglu_v2.cuh : LaunchSwiGluGroupedV2). It exposes the CUTLASS-free entry
// declared in cutlass_grouped_gemm_swiglu_v2.h, mirroring the V1 split
// (cutlass_grouped_gemm_swiglu.cuh / .cu). The kernel is a 2-SM tcgen05 warp-specialized schedule, so
// this .cu is compiled for SM100-specific archs ONLY (compute_100a/103a) — see the dedicated
// COMPILE_OPTIONS in common/CMakeLists.txt; it must NOT be built for pre-Blackwell archs (the tcgen05
// / cta_group::2 PTX is invalid there).
//
// W1 LAYOUT CONTRACT (the V2-vs-V1 delta the caller must honor):
//   * DEFAULT (V2a, compilable): W1 is HOST-PERMUTED to gran-G interleaved per output-tile, i.e. for
//     each expert the gate||up rows are reordered so a single box-N=2*TileN tile holds them interleaved
//     at the de-interleave granularity. W1 is still [G*2I, d] row-major (same SHAPE as V1), only the
//     row ORDER within each expert differs. The upstream marshalling must produce this permuted W1.
//   * V2b (SWIGLU_V2_5D_TMA build): W1 is the CONTIGUOUS, UN-permuted [G*2I, d] (Muon-safe); the 5D
//     TMA box geometry produces the gran-8 interleave on the fly (NO host permute). OFF by default.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
// IMPORTANT: include the cute-heavy kernel header FIRST (like cutlass_grouped_gemm_swiglu.cu does), so
// cute's group_modes/copy/tma_partition resolve cleanly BEFORE any TE common headers pulled by the .h
// (reversed order pollutes the namespace -> "no instance of group_modes" / "no operator()" errors).
#include "cutlass_grouped_gemm_swiglu_v2.cuh"  // LaunchSwiGluGroupedV2<Element, ElementOut> (templated)
#include "cutlass_grouped_gemm_swiglu_v2.h"    // the clean C-API declaration (signature match)

// Exported via the version-script whitelist (see the .h / libtransformer_engine.version) so the
// pytorch extension can resolve this symbol across the .so boundary. Signature is IDENTICAL to
// cutlass_grouped_swiglu (V1) so the pytorch binding marshalling is a copy-paste.
void cutlass_grouped_swiglu_v2(const void *X, const void *W1, void *A, int G, int Me, int I, int d,
                               const int *m_tile_expert, int M_varlen, const float *prob,
                               transformer_engine::DType dtype, int device, int math_sm_count,
                               cudaStream_t stream) {
  using namespace transformer_engine;

  // SM100 (Blackwell, CC 10.x) only: the kernel uses the 2-SM (cta_group::2) tcgen05 schedule. The
  // upstream dispatcher should already gate on NVTE_USE_FUSED_MOE + is_blackwell, but assert here so a
  // mis-routed call fails loudly instead of launching invalid PTX.
  int sm_major = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(sm_major == 10,
             "cutlass_grouped_swiglu_v2 requires an SM100 (Blackwell) device; the fused SwiGLU-V2 "
             "grouped-GEMM kernel is a 2-SM tcgen05 schedule with no pre-Blackwell path.");

  // VARLEN-M token packing: m_tile_expert != nullptr => uneven (TileM-aligned) experts, total tokens
  // = M_varlen. nullptr => uniform M = G*Me (M_varlen unused). The .cuh launcher reads this identically.
  //
  // CONFIG (first-cut default): mirror V1's tuned tile knobs but TileN_=64 so kMmaN=2*TileN=128 and
  // kTmemCols=AccStages*kMmaN=256<=512 (the V2 acc-pow2 constraint; TileN=128 -> kMmaN=256 -> 512 also
  // fits at AccStages=2). TileK=16/kStages=16/ClusterM=2/MinBlocks=1/AccStages=2. Autotune on the box.
  cudaError_t status = cudaErrorInvalidValue;
  if (dtype == DType::kBFloat16) {
    status = grouped_gemm_swiglu::LaunchSwiGluGroupedV2<cutlass::bfloat16_t, cutlass::bfloat16_t, 256,
                                                        64, 16, 16, 2, 1, 2>(
        reinterpret_cast<const cutlass::bfloat16_t *>(X),
        reinterpret_cast<const cutlass::bfloat16_t *>(W1),
        reinterpret_cast<cutlass::bfloat16_t *>(A), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen, /*d_m_gather_idx=*/nullptr, /*T_src=*/0, /*d_prob=*/prob);
  } else if (dtype == DType::kFloat16) {
    status = grouped_gemm_swiglu::LaunchSwiGluGroupedV2<cutlass::half_t, cutlass::half_t, 256, 64, 16,
                                                        16, 2, 1, 2>(
        reinterpret_cast<const cutlass::half_t *>(X), reinterpret_cast<const cutlass::half_t *>(W1),
        reinterpret_cast<cutlass::half_t *>(A), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen, /*d_m_gather_idx=*/nullptr, /*T_src=*/0, /*d_prob=*/prob);
  } else {
    NVTE_ERROR("cutlass_grouped_swiglu_v2: only BF16 and FP16 are supported.");
  }
  NVTE_CHECK_CUDA(status);
}
