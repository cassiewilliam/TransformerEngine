/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE F2 — fused up-proj + SwiGLU grouped GEMM, CLEAN C-API declaration.
//
// This header is intentionally CUTLASS-free (no cute/, no .cuh): the heavy templated kernel lives in
// cutlass_grouped_gemm_swiglu.cuh and is compiled ONLY inside cutlass_grouped_gemm_swiglu.cu (nvcc,
// SM100). Host-side callers — the pytorch binding (gemm.cpp, compiled by the host C++ compiler) and
// the grouped-GEMM dispatcher (cublaslt_gemm.cu) — include THIS header to call the entry by symbol,
// exactly as they call nvte_multi_tensor_gemm / cutlass_grouped_gemm. The signature uses only void*
// device pointers, ints, a cudaStream_t and transformer_engine::DType, so it pulls in no device code.

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_SWIGLU_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_SWIGLU_H_

#include <cuda_runtime_api.h>

#include "../common.h"  // transformer_engine::DType

// SonicMoE F2 fused up-projection + SwiGLU (Blackwell / SM100 only).
//
//   A[m, :I]  =  silu(gate[m]) * up[m]
//   where  [gate || up][m, :2I]  =  X[m, :d] @ W1[e(m)]^T        (per-expert grouped GEMM)
//
// The gate||up product is NEVER materialized to global memory — the SwiGLU is fused in the kernel
// epilogue and only the activated A[M, I] is written out (the route#2 / zero-Muon design). This
// replaces a TE up-GroupedLinear (out=2I) followed by a separate SwiGLU activation with a single
// pass, and is gated upstream by NVTE_USE_FUSED_MOE.
//
// Layout / dtype contract (mirrors LaunchSwiGluGrouped in the .cuh):
//   X   : [M, d]      row-major, BF16 or FP16   (permuted/grouped tokens; M = sum of per-expert tokens)
//   W1  : [G*2I, d]   row-major, same dtype as X (per-expert gate||up weights stacked over experts)
//   A   : [M, I]      row-major, same dtype as X (fused SwiGLU output)
//
// Token packing — two modes (matches the kernel's varlen-M hook):
//   * UNIFORM  : m_tile_expert == nullptr. Every expert has exactly Me tokens; M == G*Me.
//   * VARLEN-M : m_tile_expert != nullptr. Device array of length ceil(M_varlen / TileM) mapping each
//                global m-tile (TileM=256 rows) to its expert id; M == M_varlen. Each expert's token
//                count MUST be a multiple of TileM (pad upstream — SonicMoE token-rounding). Me is then
//                unused for token counting but still passed for the descriptor's per-expert stride.
//
// SM100 ONLY: the kernel is a 2-SM (cta_group::2) tcgen05 warp-specialized schedule. The caller is
// responsible for arch gating (NVTE_USE_FUSED_MOE + is_blackwell); this entry NVTE_CHECKs SM major==10.
//
// EXPORT: this symbol is whitelisted in transformer_engine/common/libtransformer_engine.version (the
// linker version-script; TE marks everything else `local`). That is required so the pytorch extension
// (a separate .so) can resolve it — the F0 cutlass_grouped_gemm stays local since it is only called
// within the common lib.
// prob (optional, fp32 [M] grouped-row order): per-token router gate; the epilogue scales A[m,:] *=
// prob[m] (the real MoE SwiGLU). nullptr => no gating (byte-identical to the ungated path).
void cutlass_grouped_swiglu(const void *X, const void *W1, void *A, int G, int Me, int I, int d,
                            const int *m_tile_expert, int M_varlen, const float *prob,
                            transformer_engine::DType dtype, int device, int math_sm_count,
                            cudaStream_t stream);

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_SWIGLU_H_
