/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE B1 — fused SwiGLU-backward grouped GEMM, CLEAN C-API declaration (single-GEMM design).
//
// This header is intentionally CUTLASS-free (no cute/, no .cuh): the heavy templated kernel lives in
// cutlass_grouped_gemm_dswiglu.cuh and is compiled ONLY inside cutlass_grouped_gemm_dswiglu.cu (nvcc,
// SM100). Host-side callers (the pytorch binding gemm.cpp) include THIS header to call the entry by
// symbol. The signature uses only void* device pointers, ints, a cudaStream_t and
// transformer_engine::DType, so it pulls in no device code. Mirrors cutlass_grouped_gemm_swiglu.h.

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_DSWIGLU_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_DSWIGLU_H_

#include <cuda_runtime_api.h>

#include "../common.h"  // transformer_engine::DType

// SonicMoE B1 fused SwiGLU backward (Blackwell / SM100 only) — the REFERENCE single-GEMM design.
//
//   dA[m, :I]      =  dY[m, :d] @ W2[e(m)]            (FC2 dgrad; per-expert grouped GEMM, K = d)
//   grad[m, i]     =  dA[m, i] * prob[m]
//   dY1[m, i]      =  grad * up[m,i] * silu'(gate[m,i])             (gate-half of dh, cols [0, I))
//   dY1[m, I + i]  =  grad * silu(gate[m,i])                        (up-half  of dh, cols [I, 2I))
//   dprob[m]      +=  dA[m, i] * silu(gate[m,i]) * up[m,i]          (col-reduced over i)
//   where [gate || up][m, :2I] = h[m, :2I] is the SwiGLU input SAVED by the forward.
//
// One GEMM (dA = dY·W2) computes dA into TMEM (never to HBM); the epilogue reads the saved SwiGLU input
// h, applies the SwiGLU backward, writes dY1, and column-reduces dprob. This is the reference
// (backward_grouped_mlp.py fc2_dactivation) design: NO recompute, NO dual GEMM. It replaces the per-op
// fallback (separate cuBLAS dA-GEMM + dswiglu kernel + swiglu-for-dprob).
//
// Layout / dtype contract (mirrors LaunchDSwiGluGrouped in the .cuh):
//   dY    : [M, d]    row-major, BF16 or FP16  (grad of the MoE output; FC2-dgrad input, K = d)
//   W2    : [G*d, I]  row-major, same dtype     (per-expert FC2 weight [d, I] stacked; B-operand, N = I)
//   h     : [M, 2I]   row-major, same dtype     (SAVED SwiGLU input gate||up)
//   dY1   : [M, 2I]   row-major, same dtype     (OUTPUT: dgate||dup = grad wrt h)
//   dprob : [M]       fp32                       (OUTPUT: router-prob grad; caller zeroes it first;
//                                                 nullptr => not produced)
//
// Token packing — UNIFORM (m_tile_expert == nullptr, M = G*Me) or VARLEN-M (m_tile_expert != nullptr,
// length ceil(M_varlen/256), each expert's tokens a multiple of TileM=256), identical to the forward.
//
// SM100 ONLY (2-SM tcgen05). EXPORT: whitelisted in libtransformer_engine.version. prob (optional,
// fp32 [M]): per-token router gate; nullptr => grad = dA (ungated).
void cutlass_grouped_dswiglu(const void *dY, const void *W2, const void *h, void *dY1, float *dprob,
                             int G, int Me, int I, int d, const int *m_tile_expert, int M_varlen,
                             const float *prob, transformer_engine::DType dtype, int device,
                             int math_sm_count, cudaStream_t stream);

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_DSWIGLU_H_
