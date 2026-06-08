# B300 (sm_103) validation — latest commit (on-device wgrad CUTLASS, all-6-GEMM grouped-tensor path)

**Commit** `6a5e69fe` (SonicMoE: on-device wgrad CUTLASS in the grouped-tensor path — all 6 MoE GEMMs CUTLASS, host pointer-loop + D2H sync removed) + uncommitted sm103 guard fixes.
**HW/SW** 8×B300 SXM6 (compute capability **10.3 / sm_103**), TE `2.17.0.dev0`, built `NVTE_CUDA_ARCHS=100` (sm_100a + sm_103a cubins), bf16. Date 2026-06-05.

## TL;DR

| check | result |
|---|---|
| Build on sm_103a | ✅ BUILD_EXIT=0, IMPORT_OK (new on-device wgrad + sm103-guarded swiglu/dswiglu compile) |
| Correctness | ✅ **65 passed, 0 failed** (`grouped_tensor_path_matches_legacy`, `accuracy_cutlass`, `fused_path_cuda_graph_safe`, `single_grouped_bias_delay_wgrad`) + fused-MoE E2E all-gradient **DROP-IN PASS** |
| Forward perf | ✅ **fused 512 TF/s @ Case-7 shape, 1.82× vs best unfused** |
| Backward perf @ Case-7 | ✅ **on-device wgrad (graphsafe/grouped-tensor) is the FASTEST fwd+bwd** — NOT slower (anomaly below was the e2e test's *different* shape) |

## 4-version performance @ Case-7 shape (H=2048, I=512, G=32, M=24576, locked 1800 MHz)

**Forward (full MoE up+SwiGLU+down):**

| version | TFLOP/s | ms |
|---|--:|--:|
| **fused** (swiglu kernel + CUTLASS dn) | **513.3** | 0.301 |
| graphsafe cuBLAS13.4 (grouped-tensor) | 282.0 | 0.548 |
| cutlass (CUTLASS_F0 per-op) | 255.1 | 0.606 |
| non-graphsafe cuBLAS (legacy merged) | 108.5 | 1.425 |

**fwd+bwd (grouped-GEMM up-proj + down-proj, summed):**

| version | up fwd+bwd | down fwd+bwd | **total** |
|---|--:|--:|--:|
| **graphsafe / grouped-tensor (new on-device wgrad)** | 0.952 ms (324.8 TF/s) | 1.166 ms | **2.118 ms ← fastest** |
| cutlass (CUTLASS_F0) | 1.223 ms (252.8) | 1.218 ms | 2.441 ms |
| non-graphsafe cuBLAS | 2.054 ms (150.6) | 1.239 ms | 3.293 ms |
| fused (dswiglu) | — | — | ~2.02 ms (user CUTLASSd bench: BWD 1.76 ms) |

→ **At the Case-7 shape the new on-device-wgrad grouped-tensor path has the fastest fwd+bwd** (2.12 ms; matches the local `CUTLASSd FWD+BWD 2.02 ms` at boost), beating CUTLASS-F0 (2.44 ms) and cuBLAS (3.29 ms). **So the on-device wgrad is faster, not slower** — the "shouldn't be slower" anomaly below is confirmed **shape-specific to the e2e test (d=512/I=2048)**, not a kernel regression. Likely the wgrad tile schedule is poor for that swapped shape (K=Me=768, weight dims d512×I2048); worth shape-aware tile selection if that shape matters.

| Backward perf | ⚠️ **anomaly — see below; on-device wgrad is NOT faster than cuBLAS at the e2e test shape, and the full bwd is ~100× the fwd.** |

## Forward (grouped_gemm_backends_bench, real Case-7 shape H=2048 / I=512, clocks locked 1800 MHz)

| path | ms/iter | TFLOP/s |
|---|--:|--:|
| **fused DIRECT (swiglu kernel) + CUTLASS dn** | **0.3017** | **512.4** |
| unfused MERGED+silu (graph-safe cuBLAS 13.4) | 0.5477 | 282.3 |
| unfused MERGED (CUTLASS_F0) | 0.6138 | 251.9 |
| unfused MERGED (legacy cuBLAS) | 1.4433 | 107.1 |
| unfused 2-separate (legacy cuBLAS) | 2.0838 | 74.2 |

→ fused forward **1.82×** over the best unfused. (At boost ~2032 MHz this scales to ~585 TF/s, matching the local `CUTLASSd FWD 0.2643 ms / 585 TF/s` reading.)

## Backward — the "shouldn't be slower" anomaly

`te_fused_moe_e2e_test.py --all`, **e2e-test shape d=512 / I=2048** (NOTE: swapped vs Case-7), G=8, M=6144, locked 1800 MHz:

| run (down-proj wgrad backend) | fused fwd+bwd | unfused fwd+bwd | fwd ratio | fwd+bwd ratio |
|---|--:|--:|--:|--:|
| **old** — cuBLAS wgrad (`NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0`) | 36.2 ms | **22.3 ms** | 1.57× | **0.62×** |
| **new** — on-device CUTLASS wgrad (`…=1`) | 36.1 ms | **36.4 ms** | 1.56× | 1.01× |

Two anomalies, both contrary to expectation ("不应该更慢"):

1. **On-device CUTLASS wgrad is SLOWER than cuBLAS at this shape.** Switching the unfused down-proj wgrad cuBLAS→on-device-CUTLASS moved fwd+bwd **22.3 → 36.4 ms (+14 ms)**. The whole point of the commit (drop the host pointer-loop + D2H sync, all-CUTLASS) is to be ≥ cuBLAS — here it regresses.
2. **The full MoE backward is ~100× the forward** (fwd 0.35 ms vs fwd+bwd ~36 ms → bwd ~35.6 ms). Normal bwd ≈ 2–3× fwd. ~36 ms for a single small MoE layer (G=8, M=6144) is pathological and is the same ~36 ms for the *fused* path regardless of wgrad backend — i.e. there is a large bwd cost that is **not** the wgrad GEMM.

**Contrast:** the dedicated `CUTLASSd` bench at the **Case-7 shape (H=2048/I=512)** reads **BWD 1.76 ms, FWD+BWD 2.02 ms** — i.e. the on-device wgrad *is* fast there. So the slowdown is **not** intrinsic to the kernel; it is specific to the e2e-test shape (d=512/I=2048) and/or the e2e test's backward construction.

### Hypotheses (why it shouldn't be slower / where the time goes)
- **Shape sensitivity of the wgrad tile config.** wgrad is `dW = Xᵀ·dY`, K=Me=768 (ragged), M/N = weight dims. At Case-7 (H2048/I512) the dims tile well; at the e2e shape (d512/I2048) the on-device wgrad likely lands on a poor tile/partial-K schedule. Candidate: autotune / pick tile by shape (cf. [[moe-backward-bottleneck]] — wgrad at 3–6% of peak).
- **The ~36 ms (100× fwd) is backward-bound on something other than the wgrad GEMM** (it's ~constant across wgrad backends). Likely the e2e test's autograd backward: recompute of h / dswiglu, per-expert Python/op overhead, or a sync. Needs an nsys capture of the e2e backward to attribute (kernel vs op/host).
- Possible the e2e down-proj does not actually route to the new on-device wgrad (dtype/shape eligibility in `_use_fused_grouped_gemm`), so the +14 ms came from a different path change — verify with `NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1` / profiling.

### Suggested next steps
1. nsys-profile the e2e backward at d=512/I=2048 → attribute the ~36 ms (is it the wgrad kernel, dswiglu recompute, or host/op overhead?).
2. Compare the on-device wgrad **kernel time** at the two shapes (H2048/I512 vs d512/I2048) directly (kernel-only, no autograd) to confirm the shape-tuning gap.
3. If shape-tuned: add tile-config selection / autotune for the wgrad GEMM (the same lever as the varlen-K grouped GEMM in [[moe-backward-bottleneck]]).

## NVTE_USE_FUSED_MOE MFU — mcore-style MoE FFN (qa/moe_4backends_fwdbwd.py, te.ops framework, Case-7 shape, locked 1800 MHz)

MoE FFN = GroupedLinear(up,2I) → ScaledSwiGLU → GroupedLinear(down). FWD flop 154.6 G; FWD+BWD = 3× (463.8 G); peak 2250 TF/s.

| backend (MODE) | FWD TF/s | FWD MFU | FWD+BWD | BWD | FWD+BWD MFU |
|---|--:|--:|--:|--:|--:|
| legacy cuBLAS (non-graphsafe) | 284.7 | 12.7% | 2.865 ms | 2.32 ms | 7.2% |
| graphsafe cuBLAS 13.4 | 253.0 | 11.2% | 3.428 ms | 2.82 ms | 6.0% |
| cutlass (F0) | 186.1 | 8.3% | **2.552 ms** | **1.72 ms** | **8.1%** |
| **fused (NVTE_USE_FUSED_MOE=1)** | **397.1** | **17.6%** | 3.068 ms | 2.68 ms | 6.7% |

- **fused wins FORWARD decisively (397 TF/s, 17.6% MFU)** — the fused up+SwiGLU kernel.
- **fused FWD+BWD MFU (6.7%) is NOT best** — its **backward (2.68 ms) is the bottleneck** (cutlass bwd 1.72 ms → cutlass wins fwd+bwd at 8.1%).
- The bench's `fused` mode keeps the **down-proj on graph-safe cuBLAS** for the backward; wiring the new on-device-wgrad grouped-tensor path into the fused mode's down-proj (it had the fastest per-GEMM fwd+bwd above) is the obvious next lever.
- Note: compass-mcore *training* routes to the te.ops fused MoE **only via `--use-transformer-engine-op-fuser`** (its `experts.py` `fused_moe` is flashinfer/inference-only). See the E2E result below.

## Case 7 End2End MFU with NVTE_USE_FUSED_MOE (full training, from scratch, boost)

Wired the SonicMoE fused MoE into compass-mcore training via the TE op-fuser, adapted for the **non-interleaved (plain)** model:
- TE: `validate_grouped_mlp_dims(..., allow_plain_glu=True)` — the CUTLASS kernel reads plain gate||up, so `glu_interleave_size=None` is valid (re-verified: te_fused_moe_e2e interleave=None → all-gradient DROP-IN PASS).
- compass-mcore gate `_is_fused_impl_supported`: accept `NVTE_USE_FUSED_MOE` + plain interleave (`None`/32).
- Run: `--use-transformer-engine-op-fuser` + `NVTE_USE_FUSED_MOE=1`, no interleave, from scratch. Smoke: fused activates (0 "not available"), **no hyper-connections×op-fuser conflict**, loss sane.

Steady-state (iters 5-8 all 386.7-386.9 TFLOP/GPU; loss 12.61→12.20 healthy):

| config (Case 7 E2E) | TFLOP/GPU | MFU |
|---|--:|--:|
| **CUTLASS (module path, all-6-GEMM CUTLASS)** | 412.9 | **18.35% ← best** |
| cuBLAS (module path) | 397.3 | 17.66% |
| **fused MoE (NVTE_USE_FUSED_MOE, op-fuser)** | ~386.8 | **17.19% (regression)** |

**Result: fused MoE E2E (17.19%) is BELOW both cuBLAS and CUTLASS — a net regression.** Integration is correct (activates, no conflict, loss healthy) but NOT an optimization. Root cause: the op-fuser fuses only **up+SwiGLU**; the **down-proj + ALL backward GEMMs fall back to graph-safe cuBLAS**, so it forfeits the optimized CUTLASS path (on-device wgrad, all-6-GEMM CUTLASS) that the CUTLASS baseline has. The fused forward win (1.57×) covers only ~half the MoE forward, MoE is ~4.6% of E2E (nsys) → the diluted forward gain is outweighed by the slower fused backward + op-framework overhead. **The actual optimization for Case 7 is CUTLASS (18.35%)**; making fused competitive requires wiring the on-device-wgrad CUTLASS into the op-fuser's down-proj + backward.

## Reproduce
```bash
# build (te_build container, TE 2.17 src at /data1/min.yang/te_b300_src)
NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=100 MAX_JOBS=48 pip install --no-build-isolation -e .
# correctness
python -m pytest tests/pytorch/test_grouped_linear.py -k "grouped_tensor_path_matches_legacy or accuracy_cutlass or fused_path_cuda_graph_safe"
# forward perf (Case-7 shape)
CUDA_VISIBLE_DEVICES=0 python qa/grouped_gemm_backends_bench.py
# backward (e2e, on-device wgrad)
CUDA_VISIBLE_DEVICES=0 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1 python qa/te_fused_moe_e2e_test.py --all
```

---

## 5-backend CUDA-graph fwd+bwd validation (fresh 26.04 container, 2026-06-08)

Per the dispatch map (`docs/grouped_gemm_dispatch_map.html` §2 "五条主线"), the mcore op-fuser **expert-GEMM path** (`te_ops` `GroupedLinear → ScaledSwiGLU → GroupedLinear` — exactly what mcore's TEGroupedMLP runs under `use_te_op_fuser`) was benchmarked across all 5 grouped-GEMM backends, **captured in a CUDA graph (fwd+bwd)** and timed (warmup 5+20, then 100 graph replays, CUDA events).

**Env (fresh, isolated from the 26.05 build):** container `sonic-moe-2604` = `nvcr.io/nvidia/pytorch:26.04-py3` (**privileged** — nsys CUPTI needs `perf_event_open`, blocked by default seccomp), **TE 2.17.0.dev0** rebuilt `NVTE_CUDA_ARCHS=100` in a separate source tree `te_build_2604` (no clobber of the 26.05 `.so`), **CUDA 13.2 / cuBLAS 13.4.0.1** (≥13.3 ⇒ graph-safe grouped GEMM valid), **quack-kernels 0.5.0**, bf16. Shape = ragged Case-7 `G=32 D=2048 I=512 Me=768 M=24576`, K[min/avg/max]=256/768/2048 (imbal 2.7×), empty B200 (auto-picked). Bench: `qa/moe_cg_5way.py`.

| # | backend (MODE) | dispatch path | CUDA-graph | ms/iter | TFLOP/s |
|---|---|---|---|--:|--:|
| ① | **multistream** (legacy cuBLAS) | list → `multi_stream_cublas_gemm` | ❌ **GRAPH_FAIL** | — | — |
| ② | **gt_cublas** (graph-safe cuBLASLt) | grouped-tensor → `cublasLtMatmul` (nvjet) | ✅ | 1.0785 | 430 |
| ③ | **cutlass_list** (CUTLASS·multi_tensor) | list → `cutlass_grouped_gemm` | ❌ **GRAPH_FAIL** | — | — |
| ④ | **cutlass_gt** (CUTLASS·group-tensor) | grouped-tensor → `cutlass_grouped_gemm_device_ptrs` | ✅ | 1.0433 | 445 |
| ⑤ | **fused** (SonicMoE / QuACK) | `quack gemm_gated` + `Sm100DownGemmKernelV2` + `Sm100DSwiGluKernel` | ✅ | **0.9206** | **504** |

→ **fused is fastest: 0.9206 ms = 1.17× vs the graph-safe cuBLAS baseline (②), 1.13× vs CUTLASS group-tensor (④).** cutlass_gt is 1.03× over cuBLAS.

**Key finding — ① and ③ are NOT CUDA-graph-capturable.** Both raise `Cannot copy between CPU and CUDA tensors during CUDA graph capture` — the **list / multi_tensor path does a host-pointer setup / D2H sync during capture**, so it is fundamentally not graph-safe. Only the **grouped-tensor (on-device) path** (②④) and the **fused op** (⑤) capture cleanly. This confirms the dispatch-map insight: graph-safe ⟺ grouped-tensor path.

> Note: `gt_cublas` is the **default** behaviour of `te_ops.GroupedLinear` on SM100+bf16 (`GroupedLinearFusibleOp._is_graph_safe_path_supported` checks only SM≥10.0+dtype, **no env flag**) — *not* something set by an env var. `multistream`/`cutlass_list` are forced by monkeypatching `_is_graph_safe_path_supported→False`. (The **module** `GroupedLinear` is the opposite: graph-safe is gated OFF by default, needs `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1`.)

### Per-kernel nsys (CUDA-graph mode, `--cuda-graph-trace=node`, 30 replays)

Captured in **cudagraph mode** (not eager), saved locally at `qa/nsys_2604/cg_{gt_cublas,cutlass_gt,fused}.nsys-rep` (openable in Nsight). The GEMM kernels confirm each backend's dispatch:

| backend | expert GEMM kernels (per-kernel) | activation/epilogue |
|---|---|---|
| **gt_cublas** | `nvjet_sm100_tst_128x256_64x6_2cta` — NNT (wgrad) 137µs, TNT (dgrad) 108/91µs, NTT (fwd) 98µs, NNT 63µs | `ScaledSwiGLU` elementwise ~75µs ×105 + `cast_fp8_gated` |
| **cutlass_gt** | `cutlass::device_kernel<GemmUniversal<GroupProblemShape>>` ×3 (fwd/dgrad/wgrad) ~84–91µs + `cutlass_pack_device_args` / `splits_to_offsets` | same `ScaledSwiGLU` elementwise + `cast_fp8_gated` |
| **fused** | `quackgemm_actGemmGatedSm100` (fwd up) 90µs · `Sm100DownGemmKernelV2` (down) 69µs · `Sm100DSwiGluKernel` (B2 bwd) **145µs** · `GemmUniversal` (dgrad/wgrad) 91/106/101µs | **fused into the kernels** — the standalone SwiGLU elementwise (~75µs×105) is gone |

- **Where fused's 1.17× comes from:** the up+SwiGLU and down are fused, so the standalone `ScaledSwiGLU` elementwise (~75µs ×105 instances in ②④) disappears, and the up-proj uses QuACK `gemm_gated`.
- **fused's largest single kernel is `Sm100DSwiGluKernel` (B2 bwd dswiglu) at 145µs** — the backward dswiglu remains the heaviest expert kernel (consistent with `moe_backend_optimization_record.md` §6: dswiglu is the bwd ceiling).
- A backend-invariant `vectorized_elementwise CUDAFunctor_add` (~2210 inst, 8.5 ms, ~20%) appears identically in all three — it is the autograd grad-accumulation in `.backward()`, not the GEMM path.

### Reproduce (5-backend cudagraph + nsys)
```bash
# in the 26.04 privileged container (sonic-moe-2604), TE at /data1/min.yang/te_build_2604
for m in multistream gt_cublas cutlass_list cutlass_gt fused; do
  MODE=$m CUDA_VISIBLE_DEVICES=<idle> python /data1/min.yang/moe_cg_5way.py   # GRAPH_OK,<mode>,<ms>,<TF>
done
# nsys (cudagraph mode) for the graph-safe configs:
MODE=fused NSYS_REPLAYS=30 CUDA_VISIBLE_DEVICES=<idle> \
  nsys profile --cuda-graph-trace=node -t cuda -o cg_fused -f true python /data1/min.yang/moe_cg_5way.py
```

---

## 6-NVTX aligned per-operator profile (real op speedup, 2026-06-08)

To compare backends **per logical operator**, the MoE expert fwd+bwd was decomposed into **6 explicit grouped-GEMM phases**, each wrapped in an aligned NVTX range so nsys attributes kernels to the *same* operator across backends. Explicit orchestration (not autograd) is required because the 6 phases cross op boundaries (up-GEMM+swiglu) and split dgrad vs wgrad. Validated vs torch autograd (gt_cublas/cutlass_gt rel ~4e-3). Bench: `qa/moe_nvtx_ops.py`. Empty B200 (GPU auto-pick), 26.04 privileged container, 30 iters, NVTX via `nsys -t cuda,nvtx` + `nvtx_kern_sum`.

The 6 phases:
1. `NVTX_FWD_UP_SWIGLU_GEMM`  — h = X@W1ᵀ ; A = swiglu(h)
2. `NVTX_FWD_DOWN_GEMM`       — Y = A@W2ᵀ
3. `NVTX_BWD_DOWN_DSWILU_GEMM`— dA = dY@W2 ; dh = dswiglu(dA,h)
4. `NVTX_BWD_UP_GEMM`         — dX = dh@W1
5. `NVTX_DWGRAD_DOWN_GEMM`    — dW2 = Aᵀ@dY
6. `NVTX_DWGRAD_UP_GEMM`      — dW1 = Xᵀ@dh

**Per-phase main kernel time (per-iter µs; nsys `nvtx_kern_sum`, total/38 inst):**

| # | NVTX phase | gt_cublas (cuBLAS nvjet) | cutlass_gt (CUTLASS GemmUniversal) | fused (SonicMoE native kernels) |
|---|---|--:|--:|--:|
| ① | FWD_UP_SWIGLU | 119.3 (nvjet TNT) + swiglu | 105.7 + swiglu | **Sm100SwiGlu 133.5** (up+swiglu fused) **+ h-recompute 101.9** |
| ② | FWD_DOWN | 75.4 (TNT) | 64.2 | **Sm100DownGemmKernelV2 68.4** |
| ③ | BWD_DOWN_DSWIGLU | 74.8 (nvjet NNT) + dswiglu | 63.6 + dswiglu | **Sm100DSwiGlu 143.6** (dgrad+dswiglu, ONE kernel) |
| ④ | BWD_UP | 113.2 (NNT) | 105.8 | 100.3 |
| ⑤ | DWGRAD_DOWN | 71.2 (NTT) | 71.5 | 70.3 |
| ⑥ | DWGRAD_UP | 123.1 (NTT) | 112.5 | 109.0 |
| | **6-GEMM sum (GEMM only)** | **577.0** | **523.3** | — |

**Findings:**
- **CUTLASS grouped GEMM is 1.10× over cuBLAS nvjet across the 6 GEMMs** (per-phase 1.07–1.18×; down-wgrad ⑤ is a tie). Transpose suffix confirms the phase: fwd=TNT, dgrad=NNT, wgrad=NTT.
- **fused fuses the activation into the GEMM kernels** — ① `Sm100SwiGlu` (up+SwiGLU, no separate swiglu elementwise) and ③ `Sm100DSwiGlu` (down-dgrad + dswiglu in ONE kernel). ④⑤⑥ use the same CUTLASS `GemmUniversal` as cutlass_gt (FUSED_MOE satisfies the CUTLASS gate), and run ~equal/slightly faster.
- **fused pays two penalties:** ① a **~102 µs h-recompute** GEMM (the up kernel doesn't emit/save h; B2 needs plain h — see `forward_fused_moe.py:368` Phase-0 recompute, the [[h-emit Phase-1 TODO]]); ③ the **B2 `Sm100DSwiGlu` at 143.6 µs is store-bound** (heavier than the bare dgrad GEMM 63.6 µs; consistent with `moe_backend_optimization_record.md §6`).

**Caveats (do not affect kernel timing):**
- gt_cublas/cutlass_gt phases ① and ③ use a **torch reference swiglu/dswiglu** (several elementwise kernels) — *not* the optimized activation kernel; only the GEMM kernel times are representative.
- fused phase ① measures **`te_cutlass_grouped_swiglu` (Sm100SwiGlu, CUTLASS V1)**, not the QuACK `gemm_gated` used by the op-fuser headline path (≈97 µs, see the cudagraph/eager nsys above).
- fused B2 `te_cutlass_grouped_dswiglu` emits **interleaved** dY1 (like QuACK preact h) → the standalone bench's dX/dW1 are numerically off (rel ~1.4); the **real model consumes that layout correctly** and the **kernel time is valid**. Eliminating the interleave→plain mismatch is the open [[gemm_dgated / TMA-4D]] item.

**Artifacts:** `qa/nsys_2604/ops6_{gt_cublas,cutlass_gt,fused}.nsys-rep` (openable in Nsight; aligned NVTX ranges), plus the earlier `cg_*`/`eager_*` (cudagraph + eager, full NVTX). Reproduce: `MODE=<cfg> NVTX=1 CUDA_VISIBLE_DEVICES=<idle> python qa/moe_nvtx_ops.py` (add `CHECK=1` for torch-validation, or wrap in `nsys -t cuda,nvtx`).

### Items 1+2 resolved — QuACK gemm_gated(store_preact) ↔ gemm_dgated matched pair (2026-06-08)

The two fused-path penalties (① ~102 µs h-recompute, ③ B2 store-bound + plain-vs-interleaved h mismatch) are **both** fixed by using QuACK's *matched pair*: `gemm_gated(store_preact=True)` emits the pre-activation h (interleaved [gate₀,up₀,gate₁,up₁,…], `concat_layout=False` convention — see `sonicmoe/functional/__init__.py:414`), and `gemm_dgated` consumes **that same interleaved h** directly (does dA'=dY₂·W2 + dswiglu + dprob col-reduce in one kernel).

**Validation (standalone, Case-7 shape, empty B200):** the pair is self-consistent — `a_prime` (gemm_dgated's recomputed s·swiglu(h)) vs `s·A` (gemm_gated's own postact) = **3.85e-3** (bf16 noise) ⇒ the h emit→read round-trip is correct. (A vs a *plain* torch ref is ~1.41 only because the standalone harness doesn't model QuACK's gather/cu_seqlens order; the real pipeline's conventions are consistent. This is exactly why the earlier `NVTE_QUACK_EMIT_H` experiment failed: B2 `te_cutlass_grouped_dswiglu` reads **plain** h, QuACK stores **interleaved** — the matched `gemm_dgated` is the correct consumer.)

**Perf (Case-7 G=32/M=24576/I=512/D=2048, empty card):**

| op | current fused | QuACK matched pair | gain |
|---|--:|--:|--:|
| ① fwd up+SwiGLU (+h) | Sm100SwiGlu 133.5 µs **+ h-recompute 101.9 µs** = 235 µs | **gemm_gated 93.2 µs** (h stored, no recompute) | **−142 µs** |
| ③ bwd down-dgrad+dswiglu | B2 Sm100DSwiGlu 143.6 µs | **gemm_dgated 90.9 µs** | **1.58× (−53 µs)** |

→ ~**190 µs/iter** off the fused MoE expert (eliminate the forward h-recompute + faster backward dswiglu), correctness-safe. Bench: `qa/quack_pair_check.py` (self-consistency), `qa/quack_pair_perf.py` (timing).

**Remaining (integration, not yet shipped):** wire `gemm_gated(store_preact=True)` into `forward_fused_moe.py` (drop the `:368` Phase-0 recompute) + a `gemm_dgated` path into `backward_fused_moe.py` (replace B2), with the up-proj dgrad/wgrad consuming the **interleaved** dh via QuACK `gemm` (the SonicMoE `functional/backward.py` pattern). Validate end-to-end via `te_fused_moe_e2e_test.py --all` (all-gradient drop-in). Pure-Python + QuACK-JIT (no C++ rebuild).

### Integration landed — gemm_dgated is the DEFAULT fused backward (2026-06-08)

The QuACK matched pair is now the **default** fused-MoE path (`NVTE_USE_FUSED_MOE=1`), gated by `NVTE_QUACK_EMIT_H` (**default `1`**; set `0` to fall back to the CUTLASS B2 + recompute-h):
- **forward** (`forward_fused_moe.py`): `gemm_gated(store_preact=True)` stores the interleaved pre-activation h → the Phase-0 `X@W1ᵀ` recompute is skipped.
- **backward** (`backward_fused_moe.py`): `gemm_dgated` consumes that interleaved h natively (dA'=dY₂·W2 + dswiglu + dprob col-reduce in ONE kernel); its interleaved dY1 is de-interleaved (gran-1) to plain so the existing up-dgrad/wgrad — and **dW1 — stay plain (Muon-safe)**.

**Correctness — `te_fused_moe_e2e_test.py` (default, no env), all-gradient drop-in vs torch:**

| tensor | n_fail | max_abs | result |
|---|--:|--:|---|
| y (fwd) | 0 | 0.00001 | PASS |
| d_input | 0 | 0.00001 | PASS |
| **d_fc1_weight (dW1)** | 0 | **0.00005** | **PASS** (plain → Muon-safe) |
| d_fc2_weight | 0 | 0.00005 | PASS |
| d_prob | 0 | 0.00002 | PASS |

→ **DROP-IN PASS** (and *more* accurate than the B2 path: d_input 0.00001 vs 0.00474, dW1 0.00005 vs 0.01799).

**Perf — cudagraph fwd+bwd TFLOP/s (same run; box was busy so abs ms is high — read relative):**

| backend | ms | TFLOP/s | vs cuBLAS |
|---|--:|--:|--:|
| gt_cublas | 1.2653 | 367 | 1.00× |
| cutlass_gt | 1.1996 | 387 | 1.05× |
| fused — old (B2 + recompute) | 1.0573 | 439 | 1.20× |
| **fused — new (gemm_dgated, DEFAULT)** | **0.9728** | **477** | **1.30×** |

→ new default fused: **1.30× vs cuBLAS, 1.23× vs CUTLASS, 1.09× vs the old B2 fused** (−84.5 µs). A residual `cat` de-interleave kernel (~10–30 µs) remains; STEP 2 ([[Option B]]) removes it by switching up-dgrad/wgrad to QuACK `gemm` (zero-copy, `concat_layout=("out",)` → plain dW1).

### Step 2 (opt-in) — zero-copy QuACK gemm for up-dgrad + wgrads (2026-06-08)

Benchmark (real Case-7 ragged, empty B200): **QuACK `gemm` is 1.2–1.3× faster than the CUTLASS grouped GEMM (GGT) on every grouped GEMM**, bit-identical vs torch (`qa/quack_wgrad_bench.py`):

| op | CUTLASS GGT | QuACK gemm | speedup | note |
|---|--:|--:|--:|---|
| FWD_DOWN | 90.6 µs | 72.0 µs | 1.26× | but dedicated `Sm100DownGemmKernelV2`=68 µs already wins → **keep Sm100Down** |
| BWD_UP (dgrad) | 113.2 µs | 88.8 µs | 1.27× | switch to QuACK |
| DWgrad Up (dW1) | 120.5 µs | 94.9 µs | 1.27× | switch to QuACK |
| DWgrad Down (dW2) | 89.2 µs | 73.6 µs | 1.21× | switch to QuACK |

**Integration (`NVTE_QUACK_ZEROCOPY=1`, default 0):** keep dY1 **interleaved** (no de-interleave `cat`); up-dgrad = QuACK `gemm(dY1, W1, concat_layout=("B",))`; up/down-wgrad = QuACK `gemm(x.T, dY, cu_seqlens_k, concat_layout=("out",))` → **plain dW (Muon-safe)**, written into the per-expert dW via a transposed view (no extra transpose). `_compute_grouped_wgrad` falls back to CUTLASS GGT (with a plain de-interleave) when `accumulate` (Megatron main_grad) / delayed-wgrad / single-grouped-weight — so **main_grad fusion stays correct**.

**Correctness — e2e (NVTE_QUACK_ZEROCOPY=1):** all-gradient drop-in **PASS** (d_input/d_fc1_weight/d_fc2_weight/d_prob `n_fail 0`, dW1 plain max_abs 5e-5 → the interleaved-dY→plain-dW via `concat_layout=("out",)` is correct).

**Perf — cudagraph fwd+bwd (same run):**

| backend | ms | TFLOP/s | vs cuBLAS |
|---|--:|--:|--:|
| gt_cublas | 1.2579 | 369 | 1.00× |
| cutlass_gt | 1.1930 | 389 | 1.05× |
| fused — Step 1 (gemm_dgated, default) | 0.9660 | 480 | 1.30× |
| **fused — Step 2 (zero-copy QuACK, opt-in)** | **0.8350** | **556** | **1.51×** |

→ Step 2 is **1.157× over Step 1 (−131 µs)**, **1.51× vs cuBLAS / 1.43× vs CUTLASS**. **Default stays Step 1** (main_grad-safe, validated); Step 2 is opt-in (`NVTE_QUACK_ZEROCOPY=1`) — validated for the non-main_grad path; the accumulate/main_grad fallback de-interleave is defensive but not yet exercised by a Megatron main_grad e2e.
