/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE F2 — fused up-proj + SwiGLU grouped GEMM, V2 CLEAN C-API declaration.
//
// V2 collapses V1's "2 W1-TMAs (gate,up) -> 2 GEMMs(N=TileN) -> 2 TMEM accs" into ONE wide
// GEMM(N=2*TileN) -> ONE interleaved acc -> de-interleave epilogue (silu(gate)*up by column).
//
// This header is intentionally CUTLASS-free (no cute/, no .cuh): the heavy templated kernel lives in
// cutlass_grouped_gemm_swiglu_v2.cuh and is compiled ONLY inside cutlass_grouped_gemm_swiglu_v2.cu
// (nvcc, SM100). Host-side callers — the pytorch binding (gemm.cpp, host C++ compiler) and any
// dispatcher — include THIS header to call the entry by symbol. The signature uses only void* device
// pointers, ints, a cudaStream_t and transformer_engine::DType, so it pulls in no device code.

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_SWIGLU_V2_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_SWIGLU_V2_H_

#include <cuda_runtime_api.h>

#include "../common.h"  // transformer_engine::DType

// SonicMoE F2 fused up-projection + SwiGLU — V2 (gran-8 gate/up interleave). Blackwell / SM100 only.
//
//   A[m, :I]  =  silu(gate[m]) * up[m]
//   where  [gate || up][m, :2I]  =  X[m, :d] @ W1[e(m)]^T        (per-expert grouped GEMM)
//
// SIGNATURE: IDENTICAL to cutlass_grouped_swiglu (V1) so the pytorch marshalling is a copy-paste; the
// ONLY behavioral difference is the W1 LAYOUT CONTRACT below.
//
// Layout / dtype contract (mirrors LaunchSwiGluGroupedV2 in the .cuh):
//   X   : [M, d]      row-major, BF16 or FP16   (permuted/grouped tokens; M = sum of per-expert tokens)
//   W1  : [G*2I, d]   row-major, same dtype as X. SHAPE is identical to V1, but the row ORDER differs:
//         * DEFAULT (V2a): per expert, the gate||up rows are HOST-PERMUTED to gran-G interleaved order
//           (so a single box-N=2*TileN tile holds them interleaved). Upstream must produce this layout.
//         * V2b (SWIGLU_V2_5D_TMA build): W1 is the CONTIGUOUS, UN-permuted [G*2I, d] (Muon-safe); the
//           5D TMA box geometry produces the interleave on the fly. OFF by default.
//   A   : [M, I]      row-major, same dtype as X (fused SwiGLU output; the [M,2I] intermediate is never
//                     materialized — the de-interleave + silu*mul happen inside the epilogue).
//
// Token packing — two modes (matches the kernel's varlen-M hook), IDENTICAL to V1:
//   * UNIFORM  : m_tile_expert == nullptr. Every expert has exactly Me tokens; M == G*Me.
//   * VARLEN-M : m_tile_expert != nullptr. Device int32 array of length ceil(M_varlen / TileM) mapping
//                each global m-tile (TileM=256 rows) to its expert id; M == M_varlen. Each expert's
//                token count MUST be a multiple of TileM (pad upstream).
//
// prob (optional, fp32 [M] grouped-row order): per-token router gate; the epilogue scales A[m,:] *=
// prob[m]. nullptr => no gating.
//
// SM100 ONLY: the kernel is a 2-SM (cta_group::2) tcgen05 warp-specialized schedule. The caller is
// responsible for arch gating; this entry NVTE_CHECKs SM major == 10.
//
// EXPORT: whitelist this symbol in transformer_engine/common/libtransformer_engine.version (the linker
// version-script; TE marks everything else `local`) so the pytorch extension (a separate .so) resolves
// it — exactly as cutlass_grouped_swiglu (V1) is whitelisted.
void cutlass_grouped_swiglu_v2(const void *X, const void *W1, void *A, int G, int Me, int I, int d,
                               const int *m_tile_expert, int M_varlen, const float *prob,
                               transformer_engine::DType dtype, int device, int math_sm_count,
                               cudaStream_t stream);

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_SWIGLU_V2_H_
