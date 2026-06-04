/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE B1 — fused SwiGLU-backward grouped GEMM, C-API definition (SM100 / Blackwell).
//
// M2 (DownProj GEMM + dAct fused): dA = dY · W2ᵀ in TMEM, then the dswiglu epilogue reads the SAVED h
// [M,2I] and writes dY1[M,2I] = dgate||dup. (dA = dY·W2 via W2ᵀ-transposed B reuses the forward structure,
// avoiding expert-in-K.) The h pointer is passed in the dGrad slot of LaunchDSwiGluGrouped (the kernel
// reads params.dGrad as the saved h). dprob (the colvec-reduce) is M2b — IGNORED here for now.
// Only this .cu instantiates the heavy CUTLASS kernel; SM100-only arch flags.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass_grouped_gemm_dswiglu.cuh"  // LaunchDSwiGluGrouped<Element, ElementOut> (templated)
#include "cutlass_grouped_gemm_dswiglu.h"    // the clean C-API declaration (signature match)

// Exported via the version-script whitelist. Signature: (dY, W2, h, dY1, dprob, G, Me, I, d, mte,
// M_varlen, prob, dtype, dev, smc, stream). M2a ignores dprob (M2b adds the colvec-reduce).
void cutlass_grouped_dswiglu(const void *dY, const void *W2, const void *h, void *dY1, float *dprob,
                             int G, int Me, int I, int d, const int *m_tile_expert, int M_varlen,
                             const float *prob, transformer_engine::DType dtype, int device,
                             int math_sm_count, cudaStream_t stream) {
  using namespace transformer_engine;

  int sm_major = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(sm_major == 10,
             "cutlass_grouped_dswiglu requires an SM100 (Blackwell) device; the kernel is a 2-SM "
             "tcgen05 schedule with no pre-Blackwell path.");

  // M3 (h-prefetch via TMA) was a NET LOSS (312us vs 282 scattered) — TMA-load contends with the dY1
  // store-TMA engine + smem round-trip; the scattered LSU read parallelizes the store-TMA + overlaps the
  // compute. Reverted to the scattered-read baseline (default kStages=16). See b1-backward-kernel-design.
  // M2b: dprob (router-prob grad OUTPUT [M] fp32, caller pre-zeroed) is now wired as the col-reduce output.
  cudaError_t status = cudaErrorInvalidValue;
  if (dtype == DType::kBFloat16) {
    status = grouped_gemm_dswiglu::LaunchDSwiGluGrouped<cutlass::bfloat16_t, cutlass::bfloat16_t>(
        reinterpret_cast<const cutlass::bfloat16_t *>(dY),
        reinterpret_cast<const cutlass::bfloat16_t *>(W2),
        /*dGrad=saved h*/ reinterpret_cast<const cutlass::bfloat16_t *>(h),
        reinterpret_cast<cutlass::bfloat16_t *>(dY1), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen, /*d_m_gather_idx=*/nullptr, /*T_src=*/0, /*d_prob=*/prob,
        /*d_dprob=*/dprob);
  } else if (dtype == DType::kFloat16) {
    status = grouped_gemm_dswiglu::LaunchDSwiGluGrouped<cutlass::half_t, cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t *>(dY), reinterpret_cast<const cutlass::half_t *>(W2),
        /*dGrad=saved h*/ reinterpret_cast<const cutlass::half_t *>(h),
        reinterpret_cast<cutlass::half_t *>(dY1), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen, /*d_m_gather_idx=*/nullptr, /*T_src=*/0, /*d_prob=*/prob,
        /*d_dprob=*/dprob);
  } else {
    NVTE_ERROR("cutlass_grouped_dswiglu: only BF16 and FP16 are supported.");
  }
  NVTE_CHECK_CUDA(status);
}
