# F2 — Fuse SwiGLU (gated activation) into the up-proj grouped-GEMM epilogue (Blackwell SM100, FP16/BF16)

Status: design / research only. No build, no GPU, no source edits performed.
Target: `transformer_engine/common/gemm/cutlass_grouped_gemm.cuh` SM100 path (tcgen05 2-CTA), FP16/BF16 in, FP32 accumulate.

---

## 0. Problem statement (recap)

MoE up-projection grouped GEMM computes `H = Xg @ W1`, `W1 ∈ [in, 2*inter]`, so `H = [tokens, 2*inter]` laid out as **concatenated halves** `[gate || up]` along N (gate = `H[:, :inter]`, up = `H[:, inter:2*inter]`). SwiGLU then computes `A = silu(H[:, :inter]) * H[:, inter:2*inter]` → `[tokens, inter]`.

Confirmed that TE uses the **concatenated-halves** convention (not interleaved):
- `transformer_engine/common/cast/dispatch/gated.cuh:35` — `const size_t cols = input.flat_last_dim() / 2;`
- `transformer_engine/common/cast/fp8/gated_fp8.cuh` uses two TMA descriptors `tensor_map_input_act` (gate, first half) and `tensor_map_input_gate` (up/"gate" multiplier, second half), reading `act_elt` and `gate_elt` from the two halves (lines 50–172). I.e. element `n` of the activation half pairs with element `n+inter` of the up half.
- `transformer_engine/common/activation/swiglu.cu:23` `nvte_swiglu` → `gated_act_fn<fp32, Empty, silu<...>>`.

Today this runs as a **separate kernel** after the GEMM: full O(tokens·2·inter) HBM read of `H` + O(tokens·inter) write of `A`. F2 wants the up-proj GEMM epilogue to emit `A` directly (and optionally also store `H` for backward), so gate/up never fully land in HBM.

---

## 1. Feasibility verdict

**The hard part is real and not solvable by a plain per-element EVT epilogue.** In CUTLASS the GEMM partitions the N dimension into tiles. With the current SM100 config (`cutlass_grouped_gemm.cuh:130`, `TileShape = Shape<_256,_256,_64>`, N-tile = 256) and `inter` typically ≫ 256, gate element `n` and up element `n+inter` live in **different N-tiles** that are produced by different CTAs / at different times. A standard EVT epilogue (`cutlass::epilogue::fusion::*` built by `CollectiveBuilder`) only sees the accumulator fragment of **one output tile**; it cannot read the paired element from another N-tile out of the accumulator. So `silu(gate)*up` cannot be done purely "in registers" inside one un-modified grouped GEMM.

### What CUTLASS actually provides for gated/GLU activation

I searched `3rdparty/cutlass` (CUTLASS 4.2). Findings:

| Construct | Where | Pairs across N-tile? | Usable on SM100 CollectiveBuilder? |
|---|---|---|---|
| `cutlass::epilogue::thread::SiLu` (functor) | `include/cutlass/epilogue/thread/activation.h:452` | n/a (per-element) | yes (as an `ActivationFn`) — but it's just silu, not gated |
| `LeftSiLUAndMul` (true GLU epilogue, **two accumulators**) | `examples/45_dual_gemm/thread/left_silu_and_mul.h:66` | **No** — it takes `(lhs, rhs)` = two *separate GEMM accumulators* (gate-GEMM + up-GEMM) and returns `silu(lhs)*rhs` | **No.** Example 45 (`dual_gemm`) is **SM80-only** legacy 2.x API (`examples/45_dual_gemm/dual_gemm.cu:133` `arch::Sm80`). It is a *dual* GEMM (two B matrices, two accumulators fused in shared epilogue), not a single 2N-wide GEMM. |
| EVT fusion **operations** (the new CollectiveBuilder path) | `include/cutlass/epilogue/fusion/operations.hpp` | — | yes |
| `LinCombEltAct` (`D = act(alpha*acc + beta*C)`) | `operations.hpp:131`; sm90 callback `sm90_callbacks_tma_warpspecialized.hpp:285` `Sm90LinCombEltAct` | per-element only | yes |
| `Sm90AuxLoad` (TMA-load an extra tensor into the epilogue) | `include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp:213` | reads an *external HBM tensor* tile-aligned to the output tile | yes (sm100 reuses sm90 nodes) |
| `Sm90Compute<Fn>` tree node (unary `SiLu`, binary `multiplies`) | `sm90_visitor_compute_tma_warpspecialized.hpp:91`, tree visitor at `:410` | per-element | yes |

**There is NO `Gated`/`GLU`/`SiLuAndMul` fusion node in the CollectiveBuilder EVT library** (`operations.hpp` contains only `ScaledAcc`, `LinearCombination`, `LinCombEltAct`, `LinCombPerRow/ColBias[EltAct][Aux]`, block-scale variants — no gated op). Confirmed: `grep -rli "gated|glu|AndMul|GatedActivation" include/` → no hits in `epilogue/fusion/`. The only true GLU epilogue in the whole tree (`LeftSiLUAndMul`) is the **two-accumulator dual-GEMM** pattern and is SM80-only.

Also note the SM100 epilogue uses **Sm90-prefixed EVT nodes**: `sm100_callbacks_tma_warpspecialized.hpp:45` `#include ".../sm90_callbacks_tma_warpspecialized.hpp"` and builds with `Sm90EVT`, `Sm90Compute`, `Sm90AuxLoad`. So any EVT tree we write targets the `Sm90*` nodes even under `ArchTag = Sm100`.

### TE's own precedent (cuBLASLt) only does *non-gated* fused activation

`transformer_engine/common/gemm/cublaslt_gemm.cu` wires GELU as a fused epilogue via `cublasLtEpilogue_t`:
- `CUBLASLT_EPILOGUE_GELU_AUX` / `..._GELU_AUX_BIAS` (lines 654, 679), writing the pre-activation matrix to an **auxiliary output** `pre_gelu_out` (`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`, line 660/683, `..._AUX_LD` line 662/685).
- This is **plain GELU on a single N output** (output N == aux N). cuBLASLt has **no gated/SwiGLU epilogue** — GLU pairs two N-halves and cuBLASLt's epilogue is strictly per-element. So even TE's existing fused-activation path cannot express SwiGLU; it only demonstrates the "write a second aux tensor alongside D" mechanic that F2's "also store H for backward" needs.

**Verdict:** A *single* grouped GEMM that emits SwiGLU(`A`) directly via a stock CUTLASS epilogue is **not supported** by CUTLASS 4.2 — no gated EVT node exists, and EVT cannot pair across N-tiles. Three viable paths exist (Section 2); the recommended one is a **custom EVT tree using `Sm90AuxLoad` to read the matching up-tile from HBM** (clean, minimal kernel surgery) OR, for max performance and to actually avoid the HBM round-trip, an **interleaved-weight / tile-paired custom epilogue**. Path C (dual-GEMM-style two-accumulator GLU) is rejected for SM100 because that API path is SM80-only here.

---

## 2. Concrete approaches

### Option A (recommended first cut) — custom EVT epilogue: `silu(acc) * AuxLoad(up_tile)`

**Idea.** Run the up-proj GEMM with N restricted to the **gate half only** (N = `inter`), so the accumulator tile *is* the gate. Read the matching up element from HBM via `Sm90AuxLoad` pointing at the up half of `H`, multiply by `silu(gate)`, and store `A`. Because aux is loaded tile-aligned to the **same (m, n) output coordinates**, and gate-tile `n` must pair with up element at the same `n` *within the up half*, the aux tensor's base pointer is simply `H_up = W1·X` offset, OR we make the GEMM compute both halves but only over a layout where the aux pointer is `D_gate_base + inter` per row.

There are two concrete sub-variants:

- **A1 (two GEMMs, fuse in epilogue via aux).** Compute the up half `U = X @ W1[:, inter:]` first into HBM (a normal grouped GEMM, output `[tokens, inter]`). Then run the gate GEMM `G = X @ W1[:, :inter]` whose epilogue does `A = silu(G) * AuxLoad(U)`. This is essentially the dual-GEMM split but expressed as two ordinary SM100 grouped GEMMs, the second carrying a fused EVT. **Cost:** still writes `U` to HBM (O(tokens·inter)) and reads it back once in the epilogue — so it removes the *separate activation kernel's* `H` read+`A` write but adds a `U` round-trip. Net HBM traffic ≈ break-even-to-better vs today, and it removes a kernel launch + improves overlap. Simplest to land.

- **A2 (one 2N GEMM + aux re-read).** Keep the single GEMM producing the full `H=[tokens, 2*inter]` to HBM (as today), but **also** in the same epilogue, for the *gate* tiles only, `AuxLoad` the corresponding up tile and store `A`. Problem: in one launch the up tile for gate-tile `n` may not be computed yet (different CTA). So A2 needs the up tile to already be in HBM → reduces to "GEMM writes H, then a *second pass* (or the activation kernel) reads it". Not a real fusion. **Reject A2.**

So Option A in practice means **A1**: two SM100 grouped GEMMs, the gate GEMM carries the fused SwiGLU EVT epilogue that aux-loads the up result.

**EVT type (targets `Sm90*` nodes, valid under `ArchTag=Sm100`):**
```cpp
namespace fe = cutlass::epilogue::fusion;
using namespace cute;

// gate accumulator = acc (from this GEMM); up tile = aux loaded from HBM (U)
// A = silu(alpha*acc) * U     (beta unused; bias optional)
using ElementC      = /* cutlass::half_t or bfloat16_t */;
using ElementCompute = float;     // fp32 accumulate
using StrideAux     = cute::Stride<int64_t, cute::_1, int64_t>; // RowMajor [M,N,L]

using SwiGLUEvt =
  fe::Sm90EVT<                                              // multiply
    fe::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute,
                    cutlass::FloatRoundStyle::round_to_nearest>,
    fe::Sm90EVT<                                            // silu(alpha*acc)
      fe::Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute,
                      cutlass::FloatRoundStyle::round_to_nearest>,
      fe::Sm90EVT<                                          // alpha * acc
        fe::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute,
                        cutlass::FloatRoundStyle::round_to_nearest>,
        fe::Sm90ScalarBroadcast<ElementCompute>,           // alpha
        fe::Sm90AccFetch                                   // acc (gate)
      >
    >,
    fe::Sm90AuxLoad<                                        // U (up half) from HBM
      /*Stages=*/ EpiStages, /*EpilogueTile=*/ EpiTile,
      /*Element=*/ ElementC, /*StrideMNL=*/ StrideAux,
      /*SmemLayoutAtom=*/ SmemLayoutAtomAux, /*CopyOpS2R=*/ CopyOpS2R>
  >;
```
The `Sm90AuxLoad` template appears at `sm90_visitor_load_tma_warpspecialized.hpp:213` with `Arguments{ Element const* ptr_aux; Element null_default; StrideMNL dAux; }` (line ~241). You pass `ptr_aux = U_base[group]` per expert and `dAux = stride of U`.

Rather than hand-instantiate `EpiStages/EpiTile/SmemLayoutAtomAux/CopyOpS2R`, prefer building this through the existing `CollectiveBuilder` by passing a **custom `FusionOperation`** as the last template arg (the slot currently holding `cutlass::epilogue::fusion::LinearCombination<...>` at `cutlass_grouped_gemm.cuh:98`). CUTLASS lets you pass either a stock `FusionOperation` *or* a fully-formed `Sm90*EVT` callback type there; the builder will route the latter through `FusionCallbacks` (see `callbacks.hpp:58`, "handle custom EVTs"). Using a stock op is cleaner if one matched — but none does, so we pass the explicit `SwiGLUEvt` tree (the builder fills Stages/EpiTile for us when we give the *operation* form; for the raw-callback form we mirror what `Sm90LinCombEltAct` does).

**Pros:** no kernel-internal surgery; reuses the exact CollectiveBuilder path already in `cutlass_grouped_gemm.cuh`; keeps `ArchTag=Sm100` + 2-SM schedule. **Cons:** still one HBM round-trip of `U`; alignment constraint (`Sm90AuxLoad` static-asserts `Alignment*sizeof_bits % 128 == 0`, line 215 — fine for 8×BF16/16-byte).

### Option B (recommended for peak perf) — interleaved-weight, paired-in-tile custom epilogue

**Idea.** Store `W1` so that gate column `n` and up column `n` are **adjacent in N** (interleave to `[g0,u0,g1,u1,…]` instead of `[g0…g_{inter-1}, u0…u_{inter-1}]`). Then for a contiguous N-tile of width 256, the tile holds 128 (gate,up) **pairs**, both in the *same* accumulator fragment. A thin custom epilogue (or an EVT `Sm90Compute` that operates on even/odd lanes) computes `silu(g)*u` entirely in registers — **no HBM round-trip at all**, and output `A` N-tile is half the input N-tile. This is the genuine SonicMoE-style fusion.

**Cost:** requires `W1` to be physically interleaved (a one-time weight reshape/permutation at load time, or a transformed copy). The backward `H` (if needed) is then also interleaved and must be de-interleaved for `nvte_dswiglu` — or the backward grad-activation kernel taught the interleaved layout. The grouped GEMM N stride/output writeback also changes (output stride halves; deinterleave-aware store). This touches the epilogue store path, not just the fusion functor, so it's more invasive than A.

**Pros:** truly removes the HBM round-trip → the SonicMoE win. **Cons:** weight-layout change ripples to (a) checkpoint/load, (b) backward dgrad/wgrad layout, (c) any code assuming concatenated halves (`gated.cuh:35`). Bigger blast radius.

### Option C (rejected) — dual-GEMM two-accumulator GLU

`examples/45_dual_gemm` + `LeftSiLUAndMul`. Rejected: that path is **SM80 legacy 2.x** (`dual_gemm.cu:133`), not the SM100 CollectiveBuilder/tcgen05 path the teammate built, and not a grouped/ptr-array kernel. Porting it to SM100 grouped would be a from-scratch kernel.

### Recommendation

Land **Option A1** first (low risk, reuses current kernel structure, gives a correct fused-activation grouped GEMM and removes the separate activation kernel launch + the `H` read / `A` write of the activation pass). Then, if profiling shows the `U` round-trip dominates, do **Option B** (interleaved weights) for the real HBM-traffic win. Keep both behind the same `fuse_swiglu` flag so the Python side is unchanged.

---

## 3. Draft code

### 3.1 Epilogue type — diff against the current `LinearCombination` (in `GemmGivenSchedule`)

Current (`cutlass_grouped_gemm.cuh:94-98`):
```cpp
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;
```

Proposed — parameterize `GemmGivenSchedule` (and `ScheduleConfig`) on an `enum class EpiKind { kLinear, kSwiGLU }` (or a `bool kFuseSwiGLU`), and select the fusion op:
```cpp
  // Reuse the gate accumulator + aux-loaded up tile. Stride for the aux (up) tensor.
  using StrideAux = cutlass::detail::TagToStrideC_t<LayoutC>;   // RowMajor [M,N,L]

  // silu(alpha*acc) * AuxLoad(U)
  using SwiGLUFusion =
    cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies, ElementC, ElementAccumulator,
          cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::Sm90Compute<
              cutlass::epilogue::thread::SiLu, ElementAccumulator, ElementAccumulator,
              cutlass::FloatRoundStyle::round_to_nearest>,
          cutlass::epilogue::fusion::Sm90LinearCombination<     // alpha*acc (+0*C)
              ElementAccumulator, ElementAccumulator, ElementC, ElementAccumulator>>,
      cutlass::epilogue::fusion::Sm90AuxLoad<
          /*Stages=*/0 /*auto*/, /*EpilogueTile=*/cutlass::epilogue::collective::EpilogueTileAuto,
          /*Element=*/ElementC, StrideAux,
          /*SmemLayoutAtom + CopyOp -> builder-default*/>>;

  using FusionOp = std::conditional_t<
      kFuseSwiGLU, SwiGLUFusion,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule,
      FusionOp>::CollectiveOp;
```
Notes / things to verify at build time (Section 5):
- The `Sm90AuxLoad` Stages/EpilogueTile/SmemLayoutAtom/CopyOpS2R are normally filled by the builder when you pass the **operation form**. The cleanest is to add a *named* fusion operation struct (mirroring `operations.hpp` `LinCombEltAct`) e.g. `SwiGLUAuxUp<ElementOutput, ElementCompute>` and a matching `FusionCallbacks` specialization in a TE-local header that expands to the `Sm90EVT` tree above — exactly how `Sm90LinCombEltAct` is registered at `sm90_callbacks_tma_warpspecialized.hpp:285`. Then the slot in `CollectiveBuilder` is just `SwiGLUAuxUp<ElementC, ElementAccumulator>` and the builder fills the rest.
- Aux pointer + stride are per-group (grouped GEMM is ptr-array), so the aux argument must be a **pointer array** `ElementC** ptr_U` and `StrideAux* dU`, threaded like the existing `ptr_C`/`ldc` arrays.

### 3.2 `MakeArguments` — thread the aux (up) pointers/strides into `epilogue.thread`

Current `MakeArguments` (`cutlass_grouped_gemm.cuh:147-187`) sets only `fusion_args.alpha/beta`. For the SwiGLU fusion the `epilogue.thread` Arguments gains the AuxLoad sub-args:
```cpp
  // when kFuseSwiGLU:
  fusion_args.alpha = alpha;                       // scales the gate accumulator
  // AuxLoad node args (names per Sm90AuxLoad::Arguments at
  // sm90_visitor_load_tma_warpspecialized.hpp:241):
  fusion_args.aux.ptr_aux       = nullptr;         // set per-group below
  fusion_args.aux.null_default  = ElementC(0);
  fusion_args.aux.dAux          = /* packed stride of U: {inter, _1, 0} */;
  // For ptr-array, CUTLASS exposes a *_ptr_array form analogous to alpha_ptr_array.
```
(The exact nested member name is whatever the named `SwiGLUAuxUp` op exposes; with `Sm90AuxLoad` it is `.thread.<child>.ptr_aux`. Build-time introspection needed — Section 5, OQ-3.)

### 3.3 Launch path — new flag through `CutlassGroupedGemm` and the `.cu` dispatch

Add `bool fuse_swiglu` (and `ElementC** ptr_U`, `StrideC* stride_U`) to:
- `template <bool trans_a, bool trans_b, typename Element, bool kSm100, bool kFuseSwiGLU=false> void CutlassGroupedGemm(...)` — `cutlass_grouped_gemm.cuh:232`. Allocate one more `ptr_size` block in the host/device workspace for `ptr_U`, and one more `ldd_size` for `stride_U` (bump `param_workspace_size` at line 254 accordingly). Fill `ptr_U_host[i] = up_half_base[i]` and `ldu_host[i] = inter`.
- The C entry point `cutlass_grouped_gemm(...)` (`cutlass_grouped_gemm.cu:88`) gains a `bool fuse_swiglu` param (and the up-tensor array). Its `run`/`dispatch` lambdas (lines 106-128) add a `kFuseSwiGLU` axis: `run(tag, sm100_tag, swiglu_tag)`.
- New explicit instantiations in `cutlass_grouped_gemm.cu` for `<…, kSm100=true, kFuseSwiGLU=true>` for `half_t` and `bfloat16_t`, NN/NT/TN layouts (mirror lines 46-63).
- Declaration update at `cutlass_grouped_gemm.cuh:579`:
  `void cutlass_grouped_gemm(..., bool fuse_swiglu, const NVTETensor* U /*up*/, ...);`

For **Option A1** the caller passes: GEMM B = `W1[:, :inter]` (gate weight) so the GEMM N = `inter`; `D[i]` = activated `A` output; `U[i]` = the up result `X@W1[:, inter:]` precomputed by a first ordinary grouped GEMM. For a single-GEMM convenience wrapper you can launch both internally (up GEMM with `LinearCombination` → `U` HBM, then gate GEMM with `SwiGLUFusion` reading `U`).

Keep everything FP16/BF16 in, FP32 accumulate (`ElementAccumulator = float`, unchanged at `cutlass_grouped_gemm.cuh:81`). No fp8/fp4.

### 3.4 Optional: also store H (pre-activation) for backward

SwiGLU backward needs the pre-activation `H=[gate||up]`. If the model needs it (training), keep the up GEMM's `U` in HBM (already there in A1) and additionally store the gate `G` via a second EVT store node (`Sm90AuxStore`), or simply run the up-proj as the un-fused GEMM producing full `H` plus a *separate* SwiGLU only when fusion is disabled. Simplest training story: gate `G` is recoverable from `A` and `U` only non-trivially, so for training either (a) store `G` via an aux-store node, or (b) keep the unfused path for training and use F2 for inference. Decide per OQ-5.

---

## 4. Integration + test plan

### 4.1 GroupedLinear opt-in

`transformer_engine/pytorch/module/grouped_linear.py` is a *single* GroupedLinear (one weight). The SwiGLU fusion belongs to the **MLP** that chains fc1 (up-proj, `2*inter` out) → activation → fc2. The fused-op MLP lives at `transformer_engine/pytorch/ops/fused/forward_grouped_mlp.py` (`_GroupedMLP`, which already composes `(fc1, activation, fc2)` and has GLU awareness: `is_glu_activation(activation)`, and a cuDNN fused path `_cudnn_act_func = "swiglu"|"geglu"`, lines 121-141). That is the right integration seam:
- Add a CUTLASS-SM100 fused branch alongside the existing cuDNN-FE fused branch. When `arch==SM100`, dtype∈{bf16,fp16}, and `is_glu_activation(activation)` with SiLU, route fc1+activation to the new `cutlass_grouped_gemm(..., fuse_swiglu=true, ...)` instead of (grouped GEMM → `nvte_swiglu`).
- Plumb a flag from the op (e.g. `NVTE_FUSE_SWIGLU_EPILOGUE` env or an op ctor arg) down through the pytorch C++ extension that calls `cutlass_grouped_gemm`. The fc1 weight must be presented as gate-half / up-half (A1) or interleaved (Option B).
- Backward: when fused, dgrad uses `nvte_dswiglu` over the stored `H` as today (Section 3.4 governs whether `H` is available). Wgrad path is unchanged (`CutlassGroupedGemmWgrad`).

### 4.2 Correctness validation

Reference = current TE behavior: separate grouped GEMM + `torch`/`nvte_swiglu`. Add a test in `tests/pytorch/test_grouped_linear.py` mirroring `test_grouped_linear_accuracy_cutlass` (line 383):
```python
# A = silu(H[:, :inter]) * H[:, inter:]
def ref_swiglu(x, w1):           # w1: [in, 2*inter]
    h = x @ w1
    g, u = h.chunk(2, dim=-1)
    return torch.nn.functional.silu(g) * u
```
- For each expert/group and `m_split`, compare fused output `A` to `ref_swiglu`, using `dtype_tols` (test file line 125; bf16 rtol=1.6e-2). SwiGLU adds a multiply + sigmoid in fp32 compute, so reuse the bf16 tolerance already used for cutlass tests (`rtol=1e-3, atol=1e-3` at line 367, possibly loosen to bf16 row of `dtype_tols`).
- Test matrix: `num_gemms ∈ {3,6}` (existing), `inter` both divisible and not divisible by N-tile (256) to exercise predication; `dtype ∈ {bf16, fp16}`; include an empty-group (m=0) case (the kernel already special-cases empties for wgrad — verify fwd path handles aux pointers for empties).
- Backward (if F2 covers training): finite-diff / autograd compare of dgrad vs reference `torch.autograd` through `ref_swiglu`.
- Gate the test on SM100 only (`pytest.mark.skipif` on compute capability, like the existing cutlass test at line 373-376).

### 4.3 Performance measurement

- Microbench: time (a) `[grouped GEMM (2N) → nvte_swiglu]` vs (b) `[fused F2]` for representative MoE shapes (tokens per expert ~ 2k–8k, in=4096, inter=4096/14336, num_experts=8). Measure kernel time + HBM bytes (ncu `dram__bytes`). Expect F2-B to cut the activation pass's `H` read (O(2·inter)) + `A` write (O(inter)) → near-zero extra traffic; F2-A1 to remove the separate launch and the `H` read but keep one `U` round-trip.
- Compare against the existing cuDNN-FE fused MLP path (`grouped_gemm_activation_kernel`, `forward_grouped_mlp.py:78`) as a second baseline.
- End-to-end: a `GroupedLinear`/MoE-MLP forward step, tokens/s and peak memory.

---

## 5. Risks / open questions for the human

1. **OQ-1 (which option).** A1 (aux-load, low risk, one `U` round-trip) vs B (interleaved weights, true HBM win, bigger blast radius). Is the goal an inference-time speedup (A1 is enough) or matching SonicMoE's HBM savings (need B)? My recommendation: ship A1, measure, then B if needed.
2. **OQ-2 (Sm90AuxLoad on grouped/ptr-array SM100).** I confirmed `Sm90AuxLoad` exists (`sm90_visitor_load_tma_warpspecialized.hpp:213`) and that SM100 callbacks reuse `Sm90*` nodes (`sm100_callbacks_tma_warpspecialized.hpp:45`), but I did **not** verify the *grouped/ptr-array* epilogue supports a per-group aux **pointer array** with `KernelPtrArrayTmaWarpSpecialized2SmSm100` + `PtrArrayTmaWarpSpecialized2Sm`. Must confirm the AuxLoad node accepts the `ptr_array` form (analogous to `alpha_ptr_array` in `MakeArguments`, line 166) under the 2-SM grouped schedule, or whether aux is single-pointer-only. **This is the single biggest unknown.**
3. **OQ-3 (fusion Arguments wiring).** The nested member path to set `ptr_aux`/`dAux` (and any `*_ptr_array`) under the custom `Sm90EVT` tree is template-introspection-dependent. Cleanest fix: register a **named** `FusionOperation` (`SwiGLUAuxUp`) + `FusionCallbacks` specialization in a TE header (mirror `Sm90LinCombEltAct` at `sm90_callbacks_tma_warpspecialized.hpp:285`) so `decltype(arguments.epilogue.thread)` has a flat, documented `Arguments` struct. Confirm whether to upstream this as a real fusion op or keep TE-local.
4. **OQ-4 (alignment / inter divisibility).** `Sm90AuxLoad` static-asserts `Alignment*sizeof_bits % 128 == 0` (line 215). With BF16/FP16 and 16B alignment (`AlignmentC = 8`) this holds, but `inter` not divisible by the N-tile (256) needs predicated aux loads at the N edge — verify the node predicates correctly, and that `U`'s row stride = `inter` (not padded) is honored.
5. **OQ-5 (backward / store H).** Does F2 need to serve **training**? If yes, decide how `H=[gate||up]` is preserved for `nvte_dswiglu`: keep `U` in HBM (free in A1) + aux-store `G`, or restrict F2 to inference and keep the unfused path for training. Backward grad-activation also assumes concatenated halves (`gated.cuh:35`) — Option B's interleaving would require teaching `nvte_dswiglu` the interleaved layout.
6. **OQ-6 (W1 layout / interleaving cost, Option B only).** Interleaving `W1` to `[g,u]` pairs is a one-time weight transform but ripples to checkpoint load, fc2 input expectations, and the backward layout. Quantify the load-time reshape cost and confirm no other code assumes the concatenated-halves layout of `H`.
7. **OQ-7 (dual launch overhead, Option A1).** A1 needs two grouped-GEMM launches (up, then gate+fuse). Confirm the saved activation-kernel launch + `H` read outweighs the second GEMM launch + `U` round-trip for the target shapes; if not, A1 isn't worth it and we must go straight to B.
8. **OQ-8 (clamped/scaled SwiGLU variants).** TE has `clamped_swiglu` (`swiglu.cu:38`) and GeGLU. F2 as drafted only covers plain SiLU-GLU. Decide whether to parameterize the activation functor (swap `SiLu` for `GELU`/clamped) — straightforward for GELU (functor exists at `activation.h:574`), more work for clamped (needs limit/alpha params in the compute node).
9. **OQ-9 (epilogue smem budget).** Adding `Sm90AuxLoad` consumes extra epilogue shared memory; the mainloop stage count is carved out from it (`StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>`, `cutlass_grouped_gemm.cuh:103`). With TileShape 256×256 the aux smem may reduce mainloop stages and cost throughput — verify stage count after adding the node.

---

### Appendix — key file:line references used

- TE SM100 grouped GEMM + epilogue slot: `transformer_engine/common/gemm/cutlass_grouped_gemm.cuh:94-98` (epilogue builder), `:117-139` (ScheduleConfig, 2-SM schedule, TileShape 256×256×64), `:147-187` (MakeArguments / fusion_args), `:232-349` (CutlassGroupedGemm launch), `:579` (C decl).
- TE dispatch + instantiations: `transformer_engine/common/gemm/cutlass_grouped_gemm.cu:46-63` (SM100 instantiations), `:88-137` (`cutlass_grouped_gemm` entry, dispatch lambdas).
- TE concatenated-halves SwiGLU: `common/cast/dispatch/gated.cuh:35,114`; `common/cast/fp8/gated_fp8.cuh:50-172`; `common/activation/swiglu.cu:23-46`.
- TE cuBLASLt fused-activation precedent (GELU-only, aux output): `common/gemm/cublaslt_gemm.cu:441,654,660,679,683`.
- CUTLASS GLU constructs: `epilogue/thread/activation.h:452` (`SiLu`); `examples/45_dual_gemm/thread/left_silu_and_mul.h:66` (`LeftSiLUAndMul`, two-acc, SM80); `examples/45_dual_gemm/dual_gemm.cu:133` (`arch::Sm80`).
- CUTLASS EVT (CollectiveBuilder path): `epilogue/fusion/operations.hpp` (no gated node); `sm90_callbacks_tma_warpspecialized.hpp:58,285` (`Sm90EVT`, `Sm90LinCombEltAct`); `sm90_visitor_load_tma_warpspecialized.hpp:62,213,241` (`Sm90AccFetch`, `Sm90AuxLoad`, its `Arguments`); `sm90_visitor_compute_tma_warpspecialized.hpp:91,410` (`Sm90Compute` unary/binary tree); `sm100_callbacks_tma_warpspecialized.hpp:45` (SM100 reuses Sm90 nodes); `callbacks.hpp:58` (custom-EVT routing).
- Python integration seam + test: `pytorch/ops/fused/forward_grouped_mlp.py:110-141` (`_GroupedMLP`, GLU/cuDNN path); `pytorch/module/grouped_linear.py`; `tests/pytorch/test_grouped_linear.py:125,367,373-383` (tols, cutlass test, SM-gating).
