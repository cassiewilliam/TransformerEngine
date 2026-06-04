# B300 End2End MFU baseline — CopassV4 Case 7 (MoE grouped GEMM)

**Canonical baseline for comparing MoE grouped-GEMM optimizations.** New optimizations should be
measured against the **cuBLAS** row below, same config / same build / same clock regime.

## Setup (fixed across all rows)

| item | value |
|---|---|
| workload | CopassV4 Case 7: GBS 3200, MBS 4, seq 4096, EP8, 256 experts, topK 12, MoE-FFN 512, fused MHC, HybridEP permute-fusion (32 SM), bf16, Muon |
| machine | 8× NVIDIA B300 SXM6 (compute capability **10.3 / sm_103**), driver 580.159.04 |
| TransformerEngine | `2.17.0.dev0` (branch `feat/b300_groupgemm_v2`), built `NVTE_CUDA_ARCHS=100` (sm_100a + sm_103a cubins) |
| clocks | default boost (sustained **2032 MHz** over the 110 s steps → naturally frequency-stable) |
| FLOPS metric | Megatron `--log-throughput` `TFLOP/s/GPU` (= tokens/s × 26.99e9 / 8 / 1e12); MFU = TFLOP/GPU ÷ 2250 |
| statistic | median of steady-state iters 5–10 (post CUDA-graph capture); each row's steady steps are stable to ±0.2 TFLOP |

## End2End results (all TE 2.17, apples-to-apples)

| grouped-GEMM path | flag | TFLOP/s/GPU | MFU | step | tokens/s | vs baseline |
|---|---|--:|--:|--:|--:|--:|
| **cuBLAS multi-stream (BASELINE)** | `NVTE_USE_CUTLASS_GROUPED_GEMM=0` | **397.3** | **17.66%** | 111.3 s | 117,766 | — |
| CUTLASS varlen grouped GEMM | `NVTE_USE_CUTLASS_GROUPED_GEMM=1` | 412.9 | 18.35% | 107.1 s | 122,360 | **+3.9%** |
| Fused GroupedTensor (graph-safe) | `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1` | 395.1 | 17.56% | 111.9 s | 117,128 | −0.6% |

Notes:
- **CUTLASS varlen** is the throughput winner (+3.9%). Only works on B300 after the sm_103 dispatch fix
  (see below) — before it, `NVTE_USE_CUTLASS_GROUPED_GEMM=1` silently fell back to cuBLAS.
- **Fused GroupedTensor** is built for **CUDA-graph safety** (split sizes in CUDA tensors, no host sync),
  not raw speed — slightly below cuBLAS at End2End; enable it when you need to graph-capture the MoE
  grouped GEMM (passes all `test_grouped_linear_fused_path_cuda_graph_safe`).
- Reference: the earlier TE **2.14** cuBLAS run measured 399.0 / 17.73% (the "17.75%" figure). TE 2.17
  cuBLAS is ~0.4% lower; use the TE 2.17 cuBLAS row above as the in-build baseline.

## Numerical correctness — CUTLASS == cuBLAS

Same TE 2.17 build, same seed/data, only the grouped-GEMM backend differs:

| iter | cuBLAS | CUTLASS | |Δ loss| |
|--:|--:|--:|--:|
| 1–4 | 12.60919 … 12.58496 | identical | **0** (bit-identical) |
| 5 | 12.52726 | 12.52724 | 2e-5 |
| 7 | 12.33256 | 12.33257 | 1e-5 |
| 8 | 12.19933 | 12.19928 | 5e-5 |

Bit-identical for the first 4 steps; thereafter diverge only at the **~1e-5** level (bf16 GEMM
accumulation-order rounding). Isolated kernel: `test_grouped_linear_accuracy_cutlass` **48 passed**,
direct fwd rel_err 0.0014 vs fp32. → CUTLASS path is numerically equivalent.

## The sm_103 dispatch fix (why CUTLASS was off on B300)

`cublaslt_gemm.cu` gated `is_blackwell = (sm_arch() == 100)`; B300 reports `sm_arch == 103`
(= 10·major+minor), so the guard failed and CUTLASS silently fell back to cuBLAS. Fixed to accept the
whole Blackwell CC 10.x family (`sm >= 100 && sm < 110`). The `cutlass::arch::Sm100` tag + sm_103a cubin
is correct on B300 — no separate `Sm103` arch tag needed. (The repo's CUTLASS accuracy test was
Hopper-only `!= (9,0)`, which is why this was never caught; widened to Hopper + Blackwell.)

## How to reproduce a row

```bash
# in te_build container (TE 2.17, deps: transformers, emerging_optimizers, deep_ep/HybridEP, cuda.tile)
# baseline:
NVTE_USE_CUTLASS_GROUPED_GEMM=0 bash run_b300_case_7.sh
# CUTLASS opt:
NVTE_USE_CUTLASS_GROUPED_GEMM=1 bash run_b300_case_7.sh
# graph-safe fused:
NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1 bash run_b300_case_7.sh
# read steady-state (iters 5+): grep "throughput per GPU" <log>; MFU = TFLOP_GPU / 2250
```
