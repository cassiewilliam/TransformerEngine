# Graph-safe fused grouped GEMM (`NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM`) — B300 validation

## What the flag does

`NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1` routes `te.GroupedLinear` through the
**cublasLt grouped GEMM via `GroupedTensor` metadata** path
([`grouped_linear.py`](../transformer_engine/pytorch/module/grouped_linear.py) `_use_fused_grouped_gemm`).
It is **CUDA-graph-safe**: split sizes may live in a CUDA tensor, so the path performs no host-side
checks on the splits (host syncs would break graph capture). It is gated on:

- compute capability `>= (10, 0)` (Blackwell; B300 is `(10, 3)`),
- `activation_dtype ∈ {bf16, fp16}` (or MXFP8 with `MXFP8Quantizer` for all of in/weight/grad),
- not `debug` / `cpu_offloading` / `fp8_calibration` / `save_original_input` / `backward_override`,
- no output quantizers.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA B300 SXM6 AC, compute capability **10.3 (sm_103)** |
| TransformerEngine | `2.17.0.dev0` (branch `feat/b300_groupgemm_v2`), built `NVTE_CUDA_ARCHS=100` (→ sm_100a + sm_103a cubins) |
| torch / CUDA | 2.12.0a0 / 13.2 |
| date | 2026-06-04 |

## Result — all fused-path tests PASS on B300

`pytest tests/pytorch/test_grouped_linear.py -k "fused_path_cuda_graph_safe or grouped_tensor_path_matches_legacy or single_grouped_bias_delay_wgrad"`

**17 passed, 0 failed** (52 unrelated CUTLASS/FP8 cases skipped):

| test | variants | result |
|---|---|---|
| `test_grouped_linear_fused_path_cuda_graph_safe` | `[True/False-bf16]`, `[True/False-mxfp8]` | **4 PASS** |
| `test_grouped_linear_grouped_tensor_path_matches_legacy` | bf16 + mxfp8 × flag combos | **12 PASS** |
| `test_grouped_linear_grouped_tensor_path_single_grouped_bias_delay_wgrad` | — | **1 PASS** |

Key takeaways:

- **CUDA-graph capture works** with the fused grouped-tensor path (`make_graphed_callables`), bf16 and
  MXFP8, with and without bias — this is the "graph-safe" guarantee.
- The fused grouped-tensor path is **numerically equivalent to the legacy per-expert path**
  (`matches_legacy`), bf16 and MXFP8.
- One benign warning under capture (`AccumulateGrad` stream-mismatch hint); does not affect correctness.

## Related: CUTLASS grouped GEMM on B300 (separate flag, `NVTE_USE_CUTLASS_GROUPED_GEMM`)

While validating the above we found and fixed a B300 dispatch bug in the *other* grouped-GEMM path:
the runtime guard `is_blackwell = (sm_arch == 100)` ([`cublaslt_gemm.cu`](../transformer_engine/common/gemm/cublaslt_gemm.cu))
excluded B300 (sm_arch **103**), so `NVTE_USE_CUTLASS_GROUPED_GEMM=1` silently fell back to cuBLAS on
B300. Fixed to accept the whole Blackwell CC 10.x family. After the fix:

- `test_grouped_linear_accuracy_cutlass`: **48 passed** on B300 (the test's Hopper-only skip was
  widened to Hopper SM90 + Blackwell SM100/SM103). The `cutlass::arch::Sm100` tag + sm_103a cubin is
  correct on B300 — **no separate `Sm103` arch tag needed** in `cutlass_grouped_gemm.cuh`.
- Direct grouped-linear forward correctness on sm_103: `rel_err = 0.0014` vs fp32 reference (matches cuBLAS).
