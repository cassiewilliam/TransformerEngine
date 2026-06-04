/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE F2 — fused up-proj + SwiGLU grouped GEMM, C-API definition (SM100 / Blackwell).
//
// This is the ONLY translation unit that instantiates the heavy CUTLASS SwiGLU kernel
// (cutlass_grouped_gemm_swiglu.cuh : LaunchSwiGluGrouped). It exposes the CUTLASS-free entry declared
// in cutlass_grouped_gemm_swiglu.h, mirroring the F0 split (cutlass_grouped_gemm.cuh / .cu). The
// kernel is a 2-SM tcgen05 warp-specialized schedule, so this .cu is compiled for SM100-specific
// archs ONLY (compute_100a/103a) — see the dedicated COMPILE_OPTIONS in common/CMakeLists.txt; it
// must NOT be built for pre-Blackwell archs (the tcgen05 / cta_group::2 PTX is invalid there).

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass_grouped_gemm_swiglu.cuh"  // LaunchSwiGluGrouped<Element, ElementOut> (templated)
#include "cutlass_grouped_gemm_swiglu.h"    // the clean C-API declaration (signature match)

// Exported via the version-script whitelist (see the .h / libtransformer_engine.version) so the
// pytorch extension can resolve this symbol across the .so boundary.
void cutlass_grouped_swiglu(const void *X, const void *W1, void *A, int G, int Me, int I, int d,
                            const int *m_tile_expert, int M_varlen,
                            transformer_engine::DType dtype, int device, int math_sm_count,
                            cudaStream_t stream) {
  using namespace transformer_engine;

  // SM100 (Blackwell, CC 10.x) only: the kernel uses the 2-SM (cta_group::2) tcgen05 schedule. The
  // upstream dispatcher should already gate on NVTE_USE_SONIC_MOE + is_blackwell, but assert here so a
  // mis-routed call fails loudly instead of launching invalid PTX.
  int sm_major = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(sm_major == 10,
             "cutlass_grouped_swiglu requires an SM100 (Blackwell) device; the fused SwiGLU "
             "grouped-GEMM kernel is a 2-SM tcgen05 schedule with no pre-Blackwell path.");

  // VARLEN-M token packing: m_tile_expert != nullptr => uneven (TileM-aligned) experts, total tokens
  // = M_varlen. nullptr => uniform M = G*Me (M_varlen unused). The .cuh launcher reads this identically.
  cudaError_t status = cudaErrorInvalidValue;
  if (dtype == DType::kBFloat16) {
    status = grouped_gemm_swiglu::LaunchSwiGluGrouped<cutlass::bfloat16_t, cutlass::bfloat16_t>(
        reinterpret_cast<const cutlass::bfloat16_t *>(X),
        reinterpret_cast<const cutlass::bfloat16_t *>(W1),
        reinterpret_cast<cutlass::bfloat16_t *>(A), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen);
  } else if (dtype == DType::kFloat16) {
    status = grouped_gemm_swiglu::LaunchSwiGluGrouped<cutlass::half_t, cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t *>(X), reinterpret_cast<const cutlass::half_t *>(W1),
        reinterpret_cast<cutlass::half_t *>(A), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen);
  } else {
    NVTE_ERROR("cutlass_grouped_swiglu: only BF16 and FP16 are supported.");
  }
  NVTE_CHECK_CUDA(status);
}
