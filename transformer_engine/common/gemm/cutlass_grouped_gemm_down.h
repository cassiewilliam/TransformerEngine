/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// SonicMoE F2 — fused-MoE DOWN-projection (FC2) grouped GEMM, CLEAN C-API declaration.
//
// This header is intentionally CUTLASS-free (no cute/, no .cuh): the heavy templated kernel lives in
// cutlass_grouped_gemm_down_v2.cuh and is compiled ONLY inside cutlass_grouped_gemm_down.cu (nvcc,
// SM100). Host-side callers — the pytorch binding (gemm.cpp, compiled by the host C++ compiler) —
// include THIS header to call the entry by symbol, exactly as they call cutlass_grouped_swiglu. The
// signature uses only void* device pointers, ints, a cudaStream_t and transformer_engine::DType, so
// it pulls in no device code. Mirrors cutlass_grouped_gemm_swiglu.h.

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_DOWN_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_DOWN_H_

#include <cuda_runtime_api.h>

#include "../common.h"  // transformer_engine::DType

// SonicMoE F2 fused-MoE down-projection (FC2) grouped GEMM (Blackwell / SM100 only).
//
//   Y[m, :N]  =  A[m, :K] @ W2[e(m)]^T        (per-expert plain grouped GEMM, NO activation)
//
// This is the FC2 of the fused MoE MLP: A is the up-proj+SwiGLU output [M, I] (so K = I, the ffn half),
// W2 is the FC2 weight [G*H, I] (per-expert [H, I] blocks stacked over experts), and Y is the model-dim
// output [M, H] (so N = H = d). It reuses the SAME TileM=256 m_tile_expert table the fused up-proj
// builds (both kernels m-tile in blocks of 256 rows).
//
// Layout / dtype contract (mirrors LaunchDownGemmGroupedV2 in cutlass_grouped_gemm_down_v2.cuh):
//   A   : [M, K]   row-major, BF16 or FP16   (up-proj output; M = sum of per-expert tokens, K = I)
//   W2  : [G*N, K] row-major, same dtype as A (per-expert [N, K] weight blocks stacked; N = H)
//   Y   : [M, N]   row-major, same dtype as A (down-proj output)
//
// Token packing — two modes (matches the kernel's varlen-M hook, IDENTICAL to cutlass_grouped_swiglu):
//   * UNIFORM  : m_tile_expert == nullptr. Every expert has exactly M/G tokens.
//   * VARLEN-M : m_tile_expert != nullptr. Device array of length ceil(M / 256) mapping each global
//                m-tile (TileM=256 rows) to its expert id. Each expert's token count MUST be a
//                multiple of TileM=256 (pad upstream). This is the SAME table the up-proj uses.
//
// SM100 ONLY: the kernel is a 2-SM (cta_group::2) tcgen05 warp-specialized schedule. The caller is
// responsible for arch gating (NVTE_USE_FUSED_MOE + is_blackwell); this entry NVTE_CHECKs SM major==10.
//
// EXPORT: this symbol is whitelisted in transformer_engine/common/libtransformer_engine.version (the
// linker version-script) so the pytorch extension (a separate .so) can resolve it across the boundary.
void cutlass_grouped_down(const void *A, const void *W2, void *Y, int G, int N, int K, int M,
                          const int *m_tile_expert, transformer_engine::DType dtype, int device,
                          int math_sm_count, cudaStream_t stream);

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_GROUPED_GEMM_DOWN_H_
