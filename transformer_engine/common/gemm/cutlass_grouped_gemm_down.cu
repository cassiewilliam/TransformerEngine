/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE F2 — fused-MoE DOWN-projection (FC2) grouped GEMM, C-API definition (SM100 / Blackwell).
//
// This is the ONLY translation unit that instantiates the heavy CUTLASS down-proj kernel
// (cutlass_grouped_gemm_down_v2.cuh : LaunchDownGemmGroupedV2). It exposes the CUTLASS-free entry
// declared in cutlass_grouped_gemm_down.h, mirroring the F2 up-proj split
// (cutlass_grouped_gemm_swiglu.cuh / .cu). The kernel is a 2-SM tcgen05 warp-specialized schedule, so
// this .cu is compiled for SM100-specific archs ONLY (compute_100a/103a) — see the dedicated
// COMPILE_OPTIONS in common/CMakeLists.txt; it must NOT be built for pre-Blackwell archs (the tcgen05
// / cta_group::2 PTX is invalid there).

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
// IMPORTANT: include the cute-heavy kernel header FIRST (like cutlass_grouped_gemm_swiglu.cu does), so
// cute's group_modes/copy/tma_partition resolve cleanly BEFORE any TE common headers pulled by the .h
// (reversed order pollutes the namespace -> "no instance of group_modes" / "no operator()" errors).
#include "cutlass_grouped_gemm_down_v2.cuh"  // LaunchDownGemmGroupedV2<Element, ElementOut> (templated)
#include "cutlass_grouped_gemm_down.h"       // the clean C-API declaration (signature match)

// Exported via the version-script whitelist (see the .h / libtransformer_engine.version) so the
// pytorch extension can resolve this symbol across the .so boundary.
void cutlass_grouped_down(const void *A, const void *W2, void *Y, int G, int N, int K, int M,
                          const int *m_tile_expert, transformer_engine::DType dtype, int device,
                          int math_sm_count, cudaStream_t stream) {
  using namespace transformer_engine;

  // SM100 (Blackwell, CC 10.x) only: the kernel uses the 2-SM (cta_group::2) tcgen05 schedule. The
  // upstream dispatcher should already gate on NVTE_USE_SONIC_DOWN_KERNEL + is_blackwell, but assert
  // here so a mis-routed call fails loudly instead of launching invalid PTX.
  int sm_major = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(sm_major == 10,
             "cutlass_grouped_down requires an SM100 (Blackwell) device; the fused down-proj "
             "grouped-GEMM kernel is a 2-SM tcgen05 schedule with no pre-Blackwell path.");

  // VARLEN-M token packing: m_tile_expert != nullptr => uneven (TileM=256-aligned) experts, total
  // tokens = M. nullptr => uniform M (Me = M/G per expert). The .cuh launcher reads this identically.
  // CONFIG: TileM=256, TileN=128, TileK=64, kStages=8, ClusterM=2, MinBlocks=1, AccStages=2 — the
  // validated 726 TFLOP/s PASS config that REUSES the up-proj's TileM=256 m_tile_expert table.
  // Args to LaunchDownGemmGroupedV2: (X=A, W=W2, Y, G, N=H, K=I, M_total=M, stream, device,
  //                                   sm_count=math_sm_count, d_m_tile_expert=m_tile_expert, Me=0).
  cudaError_t status = cudaErrorInvalidValue;
  if (dtype == DType::kBFloat16) {
    status = grouped_gemm_down::LaunchDownGemmGroupedV2<cutlass::bfloat16_t, cutlass::bfloat16_t, 256,
                                                        128, 64, 8, 2, 1, 2>(
        reinterpret_cast<const cutlass::bfloat16_t *>(A),
        reinterpret_cast<const cutlass::bfloat16_t *>(W2),
        reinterpret_cast<cutlass::bfloat16_t *>(Y), G, N, K, M, stream, device, math_sm_count,
        m_tile_expert, /*Me=*/0);
  } else if (dtype == DType::kFloat16) {
    status = grouped_gemm_down::LaunchDownGemmGroupedV2<cutlass::half_t, cutlass::half_t, 256, 128, 64,
                                                        8, 2, 1, 2>(
        reinterpret_cast<const cutlass::half_t *>(A), reinterpret_cast<const cutlass::half_t *>(W2),
        reinterpret_cast<cutlass::half_t *>(Y), G, N, K, M, stream, device, math_sm_count,
        m_tile_expert, /*Me=*/0);
  } else {
    NVTE_ERROR("cutlass_grouped_down: only BF16 and FP16 are supported.");
  }
  NVTE_CHECK_CUDA(status);
}
