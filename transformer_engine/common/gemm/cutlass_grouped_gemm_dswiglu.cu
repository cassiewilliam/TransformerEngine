/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE B1 — fused SwiGLU-backward grouped GEMM, C-API definition (SM100 / Blackwell).
//
// MILESTONE M1a (GEMM-only): dA = dY · W2ᵀ → dA[M, I] (the BW DownProj GEMM, single accumulator,
// passthrough epilogue). The kernel reuses the forward (X·W1ᵀ) structure by feeding W2 TRANSPOSED
// (W2ᵀ [G·I, d], expert in the N-dim), so it AVOIDS the expert-in-K complexity. The h / dprob / prob
// arguments are part of the M2 (dswiglu) signature and are IGNORED here (the epilogue is passthrough).
// Only this .cu instantiates the heavy CUTLASS kernel (LaunchDSwiGluGrouped); SM100-only arch flags.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass_grouped_gemm_dswiglu.cuh"  // LaunchDSwiGluGrouped<Element, ElementOut> (templated)
#include "cutlass_grouped_gemm_dswiglu.h"    // the clean C-API declaration (signature match)

// Exported via the version-script whitelist so the pytorch extension can resolve this symbol.
// Signature matches the .h / binding (dY, W2, h, dY1, dprob, ...); M1a ignores h/dprob/prob.
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

  // M1a: dA = dY · W2ᵀ passthrough. dY → X slot, W2ᵀ → W1 slot, dA → dY1 slot. dGrad/prob = nullptr
  // (the passthrough epilogue reads neither); h / dprob are M2-only and ignored.
  (void)h;
  (void)dprob;
  (void)prob;
  cudaError_t status = cudaErrorInvalidValue;
  if (dtype == DType::kBFloat16) {
    status = grouped_gemm_dswiglu::LaunchDSwiGluGrouped<cutlass::bfloat16_t, cutlass::bfloat16_t>(
        reinterpret_cast<const cutlass::bfloat16_t *>(dY),
        reinterpret_cast<const cutlass::bfloat16_t *>(W2),
        /*dGrad=*/static_cast<const cutlass::bfloat16_t *>(nullptr),
        reinterpret_cast<cutlass::bfloat16_t *>(dY1), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen, /*d_m_gather_idx=*/nullptr, /*T_src=*/0, /*d_prob=*/nullptr);
  } else if (dtype == DType::kFloat16) {
    status = grouped_gemm_dswiglu::LaunchDSwiGluGrouped<cutlass::half_t, cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t *>(dY), reinterpret_cast<const cutlass::half_t *>(W2),
        /*dGrad=*/static_cast<const cutlass::half_t *>(nullptr),
        reinterpret_cast<cutlass::half_t *>(dY1), G, Me, I, d, stream, device, math_sm_count,
        m_tile_expert, M_varlen, /*d_m_gather_idx=*/nullptr, /*T_src=*/0, /*d_prob=*/nullptr);
  } else {
    NVTE_ERROR("cutlass_grouped_dswiglu: only BF16 and FP16 are supported.");
  }
  NVTE_CHECK_CUDA(status);
}
