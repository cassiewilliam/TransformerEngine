# F2 route#2 · P1 — dual-accumulator gated grouped GEMM (SwiGLU-fused up-proj), SM100 tcgen05

> **Scope of this file.** A *structurally-correct, hand-integrate-and-compile-iterate* skeleton for the
> fused up-proj: `A[T,I] = silu(gate)·up`, where `gate = X·W1[0:I]ᵀ`, `up = X·W1[I:2I]ᵀ`, with `W1`
> kept in its standard concatenated `[2I,d]` layout (no weight permute). All citations are real
> `file:line` in this repo's bundled **CUTLASS 4.2.0** (`3rdparty/cutlass/`, verified via
> `CHANGELOG.md:5` "4.2.0 (2025-09-15)"). Every uncertain spot is marked `// VERIFY` with a reason.
> This is a draft for hand-integration — it is **not** expected to compile as-is.

Base reference in our tree: `transformer_engine/common/gemm/cutlass_grouped_gemm.cuh`
(SM100 path: `ScheduleConfig` lines 117-139 — `KernelPtrArrayTmaWarpSpecialized2SmSm100`,
`PtrArrayTmaWarpSpecialized2Sm`, TileShape `256×256×64`, ClusterShape `2×1×1`,
`GroupProblemShape<Shape<int,int,int>>`, host loop `CutlassGroupedGemm` lines 232-349).

---

## 0. The structural pattern extracted from FMHA (attention logic stripped)

From `examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`, the
reusable "two MMAs + two TMEM accumulators in one warp-specialized persistent kernel" skeleton is:

1. **Two collective MMAs built from the GEMM CollectiveBuilder** (`Sm100`, `OpClassTensorOp`),
   `CollectiveMmaQK` / `CollectiveMmaPV` — file lines 90-105. Each is just used as a *type provider*:
   it gives `TiledMma`, `SmemLayoutA/B`, `make_fragment_A/B`, and `partition_fragment_C`. The FMHA
   author does **not** run the collective's own `load()`/`mma()` loop — a custom `mma()` (lines
   257-511) issues `cute::gemm(...)` atoms directly into hand-placed TMEM tensors.
2. **Two TMEM accumulators as offset views of one `partition_fragment_C`** — lines 293-304:
   ```
   Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));   // acc #1 (S=QKᵀ)
   Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{})); // acc #2 (O=PV)
   tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);          // explicit TMEM col offset
   tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
   ```
   The TMEM column map is a hand-authored enum `TmemAllocation` (lines 121-134): `S0=0, S1=128,
   O0=256, O1=384, kEnd=512`. **This is the load-bearing trick for two accumulators** — allocate the
   full 512-col TMEM once, then carve named sub-ranges by adding a byte/col offset to
   `accumulator.data()`.
3. **Warp specialization roles** in the kernel (`kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`
   lines 53-72): `Load`(1 warp), `MMA`(1 warp), `Softmax0/1`(8), `Correction`(4), `Epilogue`(1). The
   MMA warp owns TMEM allocation (`tmem_allocator.allocate(...Sm100TmemCapacityColumns...)` line 539)
   and issues both MMAs; consumer warps read the accumulators.
4. **One pipeline per producer→consumer edge** (lines 145-171): `PipelineTmaUmmaAsync` (load→mma),
   `PipelineUmmaAsync` (mma→consumer), `PipelineAsync` (consumer→epilogue). For a *plain* gated GEMM
   we collapse almost all of these — see §1.

**What we delete vs FMHA:** softmax (rowmax/exp2/rowsum), the online-rescale correction warpgroup,
the `P` (probabilities) re-quantization in TMEM, the LSE output. Our second MMA is **independent of**
the first (both read the *same* X-tile; they do not chain like `P=softmax(S)` → `O=P·V`). That makes
our dependency graph strictly simpler than FMHA's.

---

## 1. Construction decision — **(a) dual-collective custom kernel**, NOT (b) gathered single-B

**Decision: (a) two `CollectiveBuilder` mainloop *type-providers* (CollectiveMmaGate / CollectiveMmaUp)
composed in a custom SM100 warp-specialized kernel, à la FMHA — but greatly simplified because the two
MMAs are independent and share the A operand.**

### Why not (b) "single GEMM, gather both row-blocks into one wide B-tile + pairing epilogue"

Option (b) = make `N = 2I`, TMA-load B rows `[j0..j0+BN-1]` (gate) and `[I+j0..I+j0+BN-1]` (up) into
one accumulator of width `2·BN`, then in the epilogue pair column `c` with column `c+BN`:
`A[:,c] = silu(acc[:,c])·acc[:,c+BN]`.

It is *seductive* (reuses stock grouped GEMM + a small epilogue), but it does not actually work cleanly
in CUTLASS 4.2:

- **No pairwise/GLU fusion node exists.** `epilogue/fusion/operations.hpp` only ships single-source
  elementwise-activation nodes (`LinCombEltAct`, `LinCombEltActBlockScaleFactor`); confirmed by
  `grep` — there is `epilogue/thread/activation.h:452 struct SiLu` and `linear_combination_silu.h`,
  but **nothing that multiplies two halves of one accumulator**. A GLU epilogue that reads
  `acc[:,c]` and `acc[:,c+BN]` together is not expressible through the standard EVT
  (`Sm100` visitor tree in `epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp`) without a
  *custom* visitor that cross-reads columns — i.e. you write custom code either way.
- **The "gather" B-load is the harder TMA.** Pulling rows `j0` and `I+j0` into one *contiguous* B-tile
  requires either a non-affine TMA box or two TMA copies into adjacent smem halves — which is exactly
  the "two B descriptors" of option (a), just hidden inside one mainloop you'd have to fork anyway
  (the stock `sm100_mma_array_warpspecialized.hpp:646-648` issues exactly one `copy(...tma_load_b...)`).
- **Accumulator width.** `N=2I` per CTA tile is identical TMEM pressure to two `N=I` accumulators, so
  (b) buys nothing on TMEM while losing the clean "two independent MMAs" structure FMHA already proves.

### Why (a) is the right shape for *our* problem (and simpler than FMHA)

- FMHA already demonstrates two `partition_fragment_C` accumulators coexisting in 512-col TMEM
  (lines 293-304) — our G1 evidence.
- Our two MMAs are **independent and share A** (X-tile). So unlike FMHA we need **one load pipeline**
  (X shared + gate-B + up-B in the same producer commit) and **one mma→epilogue pipeline**. No
  softmax/correction warps, no `OrderedSequenceBarrier`. The "second TMA descriptor" is the only real
  new mechanism, and ex.75 + our base already build per-group ptr-array B descriptors (G2 evidence).
- It keeps `W1` in `[2I,d]`: the up B-operand is **the same ptr-array shifted by `I` rows**
  (`ptr_B_up[i] = ptr_B_gate[i] + I*row_stride`), with an **identical `StrideB`**. Zero weight permute.

> **Pragmatic fallback noted for the integrator.** A *much* faster route to first numerics — if the
> custom warp-specialized pipeline proves too costly to debug — is to keep the **stock** grouped GEMM
> mainloop entirely and instead instantiate it **twice conceptually but fused via a custom Sm100
> epilogue visitor** that holds *two* accumulator TMEM tiles. But CUTLASS's standard kernel only owns
> *one* mainloop, so two accumulators still force a custom kernel. Hence (a) is unavoidable for the
> true single-pass fusion; (b)/EVT-only cannot express it. If P1 must de-risk first, build the
> **non-fused** two-GEMM-then-EVT-SiLU·mul (`route A1` in the plan) for a numeric oracle, then port to
> (a).

---

## 2. Concrete CUTLASS C++ draft — `cutlass_grouped_gemm_swiglu.cuh`

This is the new header to hand-integrate next to `cutlass_grouped_gemm.cuh`. It is intentionally a
**skeleton**: the type config + host setup are the most load-bearing/verified parts; the custom
mainloop/epilogue device code is structurally laid out with `// VERIFY` at each uncertain CUTLASS call.

```cpp
/***************************************************************************************************
 * F2 route#2 — SwiGLU-fused MoE up-proj grouped GEMM (SM100 tcgen05, dual TMEM accumulator).
 * A[T,I] = silu(X·W1[0:I]ᵀ) · (X·W1[I:2I]ᵀ),  W1 kept standard [2I,d] (no permute).
 * DRAFT skeleton — hand-integrate + compile-iterate. Patterned on:
 *   examples/77_blackwell_fmha/.../sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp  (dual acc)
 *   examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu                 (ptr-array host)
 *   transformer_engine/common/gemm/cutlass_grouped_gemm.cuh                         (our base)
 **************************************************************************************************/
#pragma once

#include <transformer_engine/transformer_engine.h>
#include <cub/cub.cuh>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"
#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"           // cute::TMEM::Allocator2Sm, Sm100TmemCapacityColumns
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"                 // PipelineTmaUmmaAsync, PipelineUmmaAsync, PipelineAsync
#include "cutlass/epilogue/thread/activation.h"          // cutlass::epilogue::thread::SiLu
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace transformer_engine {
namespace grouped_gemm_swiglu {

using namespace cute;

using ProblemShapeType = cute::Shape<int, int, int>;                       // (M=tokens, N=I, K=d)
using ProblemShape     = cutlass::gemm::GroupProblemShape<ProblemShapeType>;

// -----------------------------------------------------------------------------------------------
// 2.1 · Type config — reuse the base SM100 2-SM schedule (cutlass_grouped_gemm.cuh:117-139).
//      We instantiate TWO collective mainloops with IDENTICAL types; they differ only at runtime
//      by which ptr_B array (gate vs up) they consume.  X (=A operand) is shared.
// -----------------------------------------------------------------------------------------------
template <typename Element_ /*bf16/fp16*/, typename ElementOut_ /*bf16/fp16*/>
struct SwiGluConfig {
  using Element    = Element_;                              // X and W1 element
  using ElementAcc = float;                                 // fp32 accumulate (required)
  using ElementOut = ElementOut_;                           // A[T,I] element

  using ArchTag      = cutlass::arch::Sm100;
  using OpClass      = cutlass::arch::OpClassTensorOp;

  // X[T,d] row-major (token-major, K=d contiguous). W1[2I,d] row-major: each output row is d-contig,
  // so the "up" block starts exactly I rows below "gate" — see §2.3 stride arithmetic.
  using LayoutX  = cutlass::layout::RowMajor;               // A operand  (M,K)=(tokens,d)
  using LayoutW1 = cutlass::layout::RowMajor;               // B operand  (N,K)=(I,   d)   // VERIFY: B as (N,K) row-major == each W1 output-row d-contiguous; matches base LayoutB packed({k,n}) usage (cutlass_grouped_gemm.cuh:304)
  using LayoutA  = cutlass::layout::RowMajor;               // output A[T,I]

  static constexpr int AlignX  = 128 / cutlass::sizeof_bits<Element>::value;     // 8 for bf16
  static constexpr int AlignW1 = 128 / cutlass::sizeof_bits<Element>::value;     // 8
  static constexpr int AlignA  = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

  // FINAL base schedule (cutlass_grouped_gemm.cuh:120-133): 2-SM / 2-CTA tcgen05, cluster 2x1x1.
  // NOTE the N here is 128 (NOT the base 256): two fp32 accumulators must fit 512-col TMEM with
  // double-buffering — see §3 hardest-point #1.  // VERIFY: N=128 vs 256 — TMEM budget, not perf yet.
  using TileShape    = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using ClusterShape = cute::Shape<cute::_2,   cute::_1,   cute::_1>;

  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;     // only for type-providers

  // Two collective MMA *type-providers* (same type; used twice at runtime). We reuse the gemm
  // CollectiveBuilder exactly like FMHA reuses it for CollectiveMmaQK/PV (mainloop hpp:90-105),
  // and like our base builds CollectiveMainloop (cutlass_grouped_gemm.cuh:100-105).
  // StageCount is a placeholder; the builder re-derives it (FMHA comment, mainloop hpp:95).
  using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass,
      Element,  LayoutX*,  AlignX,        // A operand = X  (ptr-array: LayoutX* per group)
      Element,  LayoutW1*, AlignW1,       // B operand = W1 row-block (ptr-array)
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      KernelSchedule>::CollectiveOp;
  // We will instantiate CollectiveMma's TiledMma twice and place its accumulator at two TMEM offsets.

  using TiledMma   = typename CollectiveMma::TiledMma;
  using SmemLayoutX  = typename CollectiveMma::SmemLayoutA;   // X smem
  using SmemLayoutW1 = typename CollectiveMma::SmemLayoutB;   // W1 row-block smem (per gate/up)

  // Strides used by host (per-group). Element ptrs are arrays; stride is shared across groups
  // (uniform d, I) but we still pass per-group like the base does (cutlass_grouped_gemm.cuh:303-305).
  using StrideX  = cutlass::detail::TagToStrideA_t<LayoutX>;    // VERIFY: exact alias — base uses Gemm::GemmKernel::InternalStrideA; here we have no GemmUniversal so derive from layout tag (TagToStrideA_t in cutlass/gemm/gemm.h)
  using StrideW1 = cutlass::detail::TagToStrideB_t<LayoutW1>;   // VERIFY: same — TagToStrideB_t
  using StrideA  = cutlass::detail::TagToStrideC_t<LayoutA>;    // VERIFY: output stride
};

// -----------------------------------------------------------------------------------------------
// 2.2 · TMEM column map for the two accumulators (mirrors FMHA TmemAllocation, mainloop hpp:121-134).
//      512 cols total (Sm100TmemCapacityColumns). With N=128, one fp32 acc = 128 cols.
//      ACC0 = gate, ACC1 = up.  NO double-buffer here (single-stage acc) — see §3 #1.
// -----------------------------------------------------------------------------------------------
enum class SwiGluTmem : uint32_t {
  kAccCols = 128,                 // == TileShape N  // VERIFY: tcgen05 fp32 acc col count == N for this MMA atom (FMHA kSizeS=128 for QK N=128, mainloop hpp:122)
  ACC_GATE = 0,
  ACC_UP   = ACC_GATE + kAccCols, // 128
  kEnd     = ACC_UP   + kAccCols, // 256  (<= 512, room to spare; lets us keep N=128 safely)
};

// -----------------------------------------------------------------------------------------------
// 2.3 · The custom kernel.  Warp roles trimmed to: Load (1 warp), MMA (1 warp), Epilogue (N warps).
//      Compared to FMHA we drop Softmax0/1 + Correction entirely.
// -----------------------------------------------------------------------------------------------
template <typename Config>
struct Sm100SwiGluGroupedKernel {
  using Element    = typename Config::Element;
  using ElementAcc = typename Config::ElementAcc;
  using ElementOut = typename Config::ElementOut;
  using TileShape  = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  using TiledMma   = typename Config::TiledMma;

  using TmemAllocator = cute::TMEM::Allocator2Sm;   // 2-SM schedule -> Allocator2Sm
                                                    // (selected automatically in stock kernel at
                                                    //  sm100_gemm_array_tma_warpspecialized.hpp:187-188)

  // --- pipelines (one per edge; far fewer than FMHA) ---
  // X + gate-B + up-B all loaded by the Load warp, consumed by the MMA warp:
  using PipelineLoad = cutlass::PipelineTmaUmmaAsync<
      /*Stages=*/3, typename TiledMma::AtomThrShapeMNK>;          // VERIFY: AtomThrShapeMNK member name on TiledMma (FMHA uses CollectiveMmaQK::AtomThrShapeMNK, mainloop hpp:147)
  // MMA -> Epilogue (both accumulators ready):
  using PipelineEpi  = cutlass::PipelineUmmaAsync</*Stages=*/2>;  // VERIFY: stage count; mma->epi like FMHA PipelineO (mainloop hpp:165)

  struct SharedStorage {
    // X smem (shared by both MMAs) + two W1 row-block smem buffers (gate, up).
    struct TensorStorage {
      cute::array_aligned<Element, cute::cosize_v<typename Config::SmemLayoutX>>  smem_x;
      cute::array_aligned<Element, cute::cosize_v<typename Config::SmemLayoutW1>> smem_w1_gate;
      cute::array_aligned<Element, cute::cosize_v<typename Config::SmemLayoutW1>> smem_w1_up;
      // + epilogue A-store smem (omitted; mirror FMHA epilogue TensorStorage::smem_o)
    } tensors;
    struct PipelineStorage {
      alignas(16) typename PipelineLoad::SharedStorage load;
      alignas(16) typename PipelineEpi::SharedStorage  epi;
    } pipelines;
    uint32_t tmem_base_ptr;
  };

  // Host-facing args: TWO ptr_B arrays (gate, up).  ptr_A (=X) shared.
  struct Arguments {
    ProblemShape problem_shape;                         // per-group (M=tokens, N=I, K=d)
    const Element** ptr_X;                              // [num_groups]
    typename Config::StrideX*  stride_X;
    const Element** ptr_W1_gate;                        // [num_groups]  = W1_base
    const Element** ptr_W1_up;                          // [num_groups]  = W1_base + I*row_stride  (§2.5)
    typename Config::StrideW1* stride_W1;               // SHARED stride for gate & up (identical)
    ElementOut**    ptr_A;                              // [num_groups] output
    typename Config::StrideA*  stride_A;
    cutlass::KernelHardwareInfo hw_info;
  };
  using Params = Arguments;                             // VERIFY: real kernels translate args->params via
                                                        // to_underlying_arguments (build TMA descs there)

  static constexpr int NumWarps = 8;                    // VERIFY: 1 load + 1 mma + ~4 epi + slack
  static constexpr int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  using ArchTag = cutlass::arch::Sm100;

  // ---- device entry ----
  CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem);
    int warp_idx = cutlass::canonical_warp_idx_sync();   // FMHA kernel hpp:260
    enum { LOAD = 0, MMA = 1, EPI = 2 };
    int role = (warp_idx == 0) ? LOAD : (warp_idx == 1) ? MMA : EPI;

    // ---- pipeline construction (producer/consumer roles set per warp; FMHA kernel hpp:284-391) ----
    typename PipelineLoad::Params lp;
    lp.role = (role == LOAD) ? PipelineLoad::ThreadCategory::Producer
                             : PipelineLoad::ThreadCategory::Consumer;
    lp.is_leader = (role == LOAD) && cute::elect_one_sync();
    lp.transaction_bytes = /* bytes(X-tile)+bytes(gateB)+bytes(upB) */ 0;  // VERIFY: sum of 3 TMA box bytes
                                                                           // (FMHA sets TransactionBytesLoadQ/K, mainloop hpp:173-176)
    PipelineLoad pipeline_load(ss.pipelines.load, lp, ClusterShape{});
    PipelineEpi  pipeline_epi (ss.pipelines.epi,  /*params*/{} /*VERIFY*/);

    TmemAllocator tmem_allocator{};
    __syncthreads();

    typename PipelineLoad::PipelineState load_prod = cutlass::make_producer_start_state<PipelineLoad>();
    typename PipelineLoad::PipelineState load_cons;
    typename PipelineEpi::PipelineState  epi_prod  = cutlass::make_producer_start_state<PipelineEpi>();
    typename PipelineEpi::PipelineState  epi_cons;

    // ====================================================================================
    // LOAD warp: per group, per output-tile, K-loop: TMA X-tile + gate row-block + up row-block.
    // X uses ptr_X; gate uses ptr_W1_gate; up uses ptr_W1_up (= +I*row_stride, built host-side §2.5).
    // ====================================================================================
    if (role == LOAD) {
      cutlass::arch::warpgroup_reg_set</*NumRegsOther*/32>();   // VERIFY: reg budget
      // for each group g (grouped tile scheduler) and each (m0,j0) tile:
      //   for k_tile in K/BK:
      //     pipeline_load.producer_acquire(load_prod);
      //     if (elect_one) {
      //       copy(tma_x .with(barrier,mcast),  gX (_,k_tile),  sX (_,write_stage));      // shared X
      //       copy(tma_wg.with(barrier,mcast),  gWg(_,k_tile),  sWg(_,write_stage));      // gate B
      //       copy(tma_wu.with(barrier,mcast),  gWu(_,k_tile),  sWu(_,write_stage));      // up   B
      //     }                                                                            // VERIFY: 2-SM
      //     pipeline_load.producer_commit(load_prod); ++load_prod;                       // multicast mask
      // (pattern: sm100_mma_array_warpspecialized.hpp:646-648 issues ONE tma_load_b; we issue X + 2×B.)
    }

    // ====================================================================================
    // MMA warp: allocate full TMEM once; build two accumulator views; K-loop issues TWO MMAs.
    // ====================================================================================
    else if (role == MMA) {
      cutlass::arch::warpgroup_reg_set</*NumRegsOther*/32>();
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &ss.tmem_base_ptr);  // FMHA kernel hpp:539
      __syncwarp();

      TiledMma tiled_mma;                                   // FMHA mainloop hpp:274
      // Two accumulator TMEM views at fixed column offsets (FMHA mainloop hpp:293-304):
      Tensor acc = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));   // (MMA,M,N) tmem frag
      Tensor acc_gate = acc; acc_gate.data() = ss.tmem_base_ptr + uint32_t(SwiGluTmem::ACC_GATE);
      Tensor acc_up   = acc; acc_up.data()   = ss.tmem_base_ptr + uint32_t(SwiGluTmem::ACC_UP);
      // VERIFY: acc.data() base — FMHA adds enum to data().get(); the stock kernel instead calls
      //         set_tmem_offsets(tmem_storage, tmem_base_ptr) (sm100_gemm_array...:875). Pick one and
      //         be consistent; the "+offset on a copied Tensor" form is the FMHA idiom and is fine.

      // X / W1 smem fragments (FMHA mainloop hpp:281-287):
      Tensor sX  = make_tensor(make_smem_ptr(ss.tensors.smem_x.data()),       typename Config::SmemLayoutX{});
      Tensor sWg = make_tensor(make_smem_ptr(ss.tensors.smem_w1_gate.data()), typename Config::SmemLayoutW1{});
      Tensor sWu = make_tensor(make_smem_ptr(ss.tensors.smem_w1_up.data()),   typename Config::SmemLayoutW1{});
      Tensor tCrX  = tiled_mma.make_fragment_A(sX);
      Tensor tCrWg = tiled_mma.make_fragment_B(sWg);
      Tensor tCrWu = tiled_mma.make_fragment_B(sWu);

      // K-loop (per output tile). Both MMAs are INDEPENDENT and share tCrX:
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;         // first k zeroes both accs
      // for k_tile ... :
      //   pipeline_load.consumer_wait(load_cons);
      //   int rs = load_cons.index();
      //   CUTLASS_PRAGMA_UNROLL for (k_block ...) {
      //     cute::gemm(tiled_mma, tCrX(_,_,k_block,rs), tCrWg(_,_,k_block,rs), acc_gate);   // FMHA mainloop hpp:726
      //     cute::gemm(tiled_mma, tCrX(_,_,k_block,rs), tCrWu(_,_,k_block,rs), acc_up);
      //     tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      //   }
      //   pipeline_load.consumer_release(load_cons); ++load_cons;
      // After K-loop: signal epilogue both accs ready:
      //   pipeline_epi.producer_acquire(epi_prod); pipeline_epi.producer_commit(epi_prod); ++epi_prod;
      // VERIFY: with a SINGLE tiled_mma issuing into two distinct TMEM tiles, accumulate_ flag is shared;
      //         confirm zero-init applies independently per acc on the first k_block (FMHA uses
      //         gemm_zero_acc / gemm_reset_zero_acc helpers, mainloop hpp:334/427 — may need those wrappers).
      tmem_allocator.release_allocation_lock();             // FMHA kernel hpp via mainloop; stock :928
    }

    // ====================================================================================
    // EPILOGUE warp(s): TMEM-load both accs -> silu(gate)*up in fp32 -> cast -> TMA-store A.
    // ====================================================================================
    else { // EPI
      cutlass::arch::warpgroup_reg_set</*NumRegsOther*/32>();
      // pipeline_epi.consumer_wait(epi_cons);
      // Build TMEM->reg loads for BOTH accumulators (FMHA correction_epilogue, mainloop hpp:777-866):
      //   using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;                        // mainloop hpp:539
      //   auto tiled_load = make_tmem_copy(TMEM_LOAD{}, acc_gate_view);
      //   copy(tiled_load, tTMEM_gate, rGate);   copy(tiled_load, tTMEM_up, rUp);  // two loads
      //   CUTLASS_PRAGMA_UNROLL for (i ...) {
      //     float g = rGate(i), u = rUp(i);
      //     float a = cutlass::epilogue::thread::SiLu<float>{}(g) * u;          // activation.h:452
      //     rOut(i) = static_cast<ElementOut>(a);
      //   }
      //   // store rOut -> smem (NumericArrayConverter) -> TMA store to A[T,I]   (FMHA store, epilogue hpp:199-213)
      // pipeline_epi.consumer_release(epi_cons); ++epi_cons;
      // tmem_allocator.free(ss.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);  // FMHA kernel hpp:625
      // VERIFY: who frees TMEM (MMA vs last EPI warp) — FMHA frees in correction/epilogue warp; pick one owner.
    }
  }
};

// -----------------------------------------------------------------------------------------------
// 2.4 · Host driver — mirrors CutlassGroupedGemm (cutlass_grouped_gemm.cuh:232-349) but builds a
//      SECOND ptr_B array (up) at +I*row_stride and sets problem N = I.
// -----------------------------------------------------------------------------------------------
template <typename Element, typename ElementOut>
void CutlassGroupedGemmSwiGlu(const NVTETensor* X,       // [num] each (tokens_g, d)
                              const NVTETensor* W1,      // [num] each (2I, d)  standard concat
                              NVTETensor* A,             // [num] each (tokens_g, I)
                              NVTETensor* workspace,
                              int I /*half hidden*/, int num_groups,
                              cudaStream_t stream, int device, int math_sm_count) {
  using Config = SwiGluConfig<Element, ElementOut>;
  using StrideX  = typename Config::StrideX;
  using StrideW1 = typename Config::StrideW1;
  using StrideA  = typename Config::StrideA;

  // host staging (same workspace partitioning idiom as base; we need ONE EXTRA ptr array for up).
  // problem_sizes_host[g] = (M=tokens_g, N=I, K=d).
  // ptr_X_host[g]       = X[g].dptr
  // ptr_W1_gate_host[g] = W1[g].dptr
  // ptr_W1_up_host[g]   = W1[g].dptr + (int64_t)I * row_stride_elems   (§2.5 — THE +I offset)
  // ptr_A_host[g]       = A[g].dptr
  // stride_W1_host[g]   = make_cute_packed_stride(StrideW1{}, {I, d, 1});  // N=I rows  // VERIFY: arg order {N,K,L}
  //   ^ base uses LayoutB::packed({k,n}).stride(0) for a scalar ld; here we want the cute stride tuple
  //     (ex.75:484 stride_B = make_cute_packed_stride(StrideB{}, {N,K,1})).
  //
  // for (g) {
  //   const auto* w1 = convertNVTETensorCheck(W1[g]);
  //   const int   d  = w1->data.shape[1];
  //   const int64_t row_stride_elems = d;   // RowMajor [2I,d]: consecutive output rows are d apart  // VERIFY
  //   Element* w1_base = reinterpret_cast<Element*>(w1->data.dptr);
  //   ptr_W1_gate_host[g] = w1_base;
  //   ptr_W1_up_host[g]   = w1_base + (int64_t)I * row_stride_elems;       // +I rows -> up block
  //   ...
  // }
  // cudaMemcpyAsync host staging -> device workspace (base :308).
  //
  // typename Sm100SwiGluGroupedKernel<Config>::Arguments args{ ... };
  // Launch via cluster (ClusterLauncher::launch, device/fmha.hpp:217-222) with grid = #tiles,
  // block = MaxThreadsPerBlock, cluster = ClusterShape, smem = sizeof(SharedStorage).
  // VERIFY: smem may exceed 48KB -> cudaFuncSetAttribute MaxDynamicSharedMemorySize (fmha.hpp:171-181).
}

}  // namespace grouped_gemm_swiglu
}  // namespace transformer_engine
```

### 2.5 · The `+I*row_stride` offset arithmetic (load-bearing — keep it scalar, build it host-side)

`W1` is RowMajor `[2I, d]`: output row `r` lives at element offset `r*d` (the K=d dimension is
contiguous). The B operand is `(N,K) = (I, d)`. The gate block is rows `[0..I-1]`; the up block is
rows `[I..2I-1]`, i.e. the **same tensor shifted down by `I` rows**:

```
ptr_W1_up = ptr_W1_gate + I * d                  // d == row_stride in elements for RowMajor [2I,d]
stride_W1_up == stride_W1_gate                   // IDENTICAL — up is just a base-pointer shift
N_problem = I                                    // each GEMM produces I output channels (= width of A)
```

This is exactly the ptr-array idiom of ex.75 (`ptr_B_host[i] = block_B.get() + offset_B[i]`,
`75_blackwell_grouped_gemm.cu:520`) and our base (`ptr_B_host[i] = inputB->data.dptr`,
`cutlass_grouped_gemm.cuh:300`) — we simply add a constant `I*d` element offset for the up array, and
the `StrideB` is the same one the base already computes (`LayoutB::packed({k,n})`,
`cutlass_grouped_gemm.cuh:304`). **No permute, no second weight tensor.** This is the whole zero-Muon
argument made concrete.

---

## 3. The 5 hardest correctness points

### #1 — TMEM allocation & partition for two fp32 accumulators (the real budget)
**Type/call:** `cute::TMEM::Allocator2Sm` (`cute/arch/tmem_allocator_sm100.hpp:117`), `allocate(512, &tmem_base_ptr)` (`:135`), `Sm100TmemCapacityColumns = 512` (`:120`). Two accumulators built as offset views of `partition_fragment_C(tiled_mma, ...)` (FMHA `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:293-304`); column map enum like `TmemAllocation` (`:121-134`).
**The subtlety P0's napkin math missed:** the stock SM100 GEMM kernel double-buffers the accumulator for mainloop/epilogue overlap — `AccTensor` has an `ACC_PIPE` mode of `AccumulatorPipelineStageCount` (`sm100_mma_array_warpspecialized.hpp:477-478` "ACC_PIPE=2 so we can double buffer"). With the **base N=256**, *one* double-buffered acc already = `256×2 = 512` cols = **all** of TMEM. Two accumulators at N=256 with ACC_PIPE=2 would need 1024 cols → does not fit. **Therefore the draft uses N=128 and a single accumulator stage** (`2×128 = 256 ≤ 512`), trading the mainloop/epilogue acc-overlap for fitting two accs. // VERIFY at compile: that a single-stage acc is acceptable for the custom kernel (FMHA itself does not ACC_PIPE its S/O accs — they are plain fragments, mainloop hpp:293-304, which is the precedent we follow).

### #2 — Warp-specialization roles & TMEM ownership
**Type/call:** `cutlass::canonical_warp_idx_sync()` (FMHA kernel `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:260`); role enum + `warp_idx_to_WarpRole` (`:53-72`); `warpgroup_reg_set<...>()` per role (`:437,469,523`). **The MMA warp must own `tmem_allocator.allocate`** (`:539`) and only it issues `cute::gemm`; exactly one warp frees (`tmem_allocator.free`, `:625`). Producer/consumer `role` is stamped into each pipeline's `Params` per warp (`:284-391`). For us: Load=Producer of `PipelineLoad`, MMA=Consumer of `PipelineLoad`+Producer of `PipelineEpi`, Epi=Consumer of `PipelineEpi`. // VERIFY: number of epilogue warps and their `consumer_arv_count` must match the named-barrier arrive counts (FMHA sets `consumer_arv_count = NumWarps* NumThreadsPerWarp`, `:319,332,345`).

### #3 — Pipeline/barriers between the 2 MMAs and the epilogue
**Type/call:** `cutlass::PipelineTmaUmmaAsync<Stages, AtomThrShapeMNK>` for load→mma (FMHA `:145-154`); `cutlass::PipelineUmmaAsync<Stages>` for mma→epilogue (FMHA `:159,165`). **Key simplification vs FMHA:** our two MMAs are *independent* (both consume the same X-tile, write disjoint TMEM), so a *single* `producer_commit` on `PipelineLoad` after loading X+gateB+upB is enough, and a *single* `PipelineEpi` commit after **both** `cute::gemm`s of the K-loop finish. No `OrderedSequenceBarrier`, no per-acc S0/S1 pipes, no correction pipe. // VERIFY: the epilogue must `consumer_wait` on `PipelineEpi` *after both* accs are committed — emit the commit once, after the K-loop, not per-MMA (otherwise epilogue could read a half-finished `acc_up`). Also `cutlass::arch::fence_view_async_tmem_store()` before the epilogue reads TMEM (FMHA `:663,1021`).

### #4 — The 2nd TMA descriptor under ptr-array + 2-SM
**Type/call:** B-load TMA atom built by `make_tma_atom_B_sm100<...>` (`sm100_mma_array_warpspecialized.hpp:279-286`); per-group address swap via `tensormaps_replace_global_address(... ptr_B[next_batch])` (`:778-779`). We need **two** such B descriptors (gate, up) — duplicate the single-B machinery: a second `tma_load_b`, a second smem buffer (`smem_w1_up`), a second `copy(...)` in the load loop (the stock loop issues exactly one at `:646-648`). Under 2-SM, the TMA is multicast across the 2 CTAs; the same `mcast_mask_b` and `AtomThrShapeMNK` apply to both descriptors since gate/up share `StrideB` and N. // VERIFY: with `KernelPtrArrayTmaWarpSpecialized2SmSm100`, both descriptors must be device-modified per group (ex.75 does device-side tensormap update, file header `:38-41`); our `to_underlying_arguments` must construct *two* `TMA_B` and the kernel must update *both* on group change. This is the single genuinely-new mechanism vs the base.

### #5 — `+I*row_stride` arithmetic against the packed `StrideB`
**Type/call:** `cutlass::make_cute_packed_stride(StrideB{}, {N,K,1})` (ex.75 `75_blackwell_grouped_gemm.cu:484`); base computes the scalar ld via `LayoutB::packed({k,n}).stride(0)` (`cutlass_grouped_gemm.cuh:304`). For RowMajor `[2I,d]` the row stride is `d` **elements**, so `ptr_up = ptr_gate + (int64_t)I*d`. **The pointer offset is in elements of `Element` (bf16), the `StrideB` is unchanged.** Pitfalls to verify: (a) `I*d` must be `int64_t` (W1 can be large — `I*d` overflows int32 for big experts); (b) the stride tuple arg order for `make_cute_packed_stride` is `{N,K,L}` with `N=I` (not `2I`) — the descriptor sees only the `I`-row sub-block; (c) alignment: `ptr_up = ptr_gate + I*d` must stay 16-byte/128-bit TMA-aligned, which holds iff `I*d*sizeof(Element)` is a multiple of 16 — true for bf16 when `I*d % 8 == 0` (always, since `d` is a multiple of 8 for our shapes). // VERIFY all three at integration; (a) is the most likely silent bug.

---

## 4. Minimal single-expert / uniform-shape compile test (do this FIRST)

Goal: get the *type config + dual accumulator + dual MMA + silu·mul epilogue* to **compile and run one
expert** before any grouped/varlen complexity. Instantiate in this order, each step compiling before
the next:

1. **Types only.** Instantiate `SwiGluConfig<bf16,bf16>` and force the compiler to build
   `Config::CollectiveMma`, `Config::TiledMma`, `Config::SmemLayoutX/W1`. Compile a `static_assert` on
   `cute::cosize_v<SmemLayoutW1>` and on `partition_fragment_C(TiledMma{}, take<0,2>(TileShape{}))`'s
   shape. **This alone validates G1 N=128 fits** (no runtime needed).
2. **Single-group, uniform shape, fixed pointers.** Drop the grouped tile scheduler. Hard-code
   `num_groups = 1`, `M=256, N=I=128, K=d=128`. Build the kernel with a trivial 1-tile launch
   (grid = 1 cluster of 2 CTAs). `ptr_X/ptr_W1_gate/ptr_W1_up/ptr_A` are single device pointers;
   `ptr_W1_up = ptr_W1_gate + I*d`. This exercises: TMEM alloc, two `cute::gemm`, the epilogue
   silu·mul, the A-store — **the entire novel core** without ptr-array/tensormap-swap machinery.
3. **Numeric oracle.** Compare against a trivial host/`torch` reference:
   `A = silu(X @ W1[:I].T) * (X @ W1[I:].T)` in fp32. Use one expert, contiguous tensors. This is the
   P1 acceptance gate ("数值对齐参考" in the plan, `F2_route2_complete_plan.md:64`).
4. **Then add ptr-array + per-group tensormap update** (two B descriptors, hardest-point #4) → uniform
   multi-group → varlen tokens (P2). Only after step 3 passes.

**First file to get compiling:** a `.cu` that includes `cutlass_grouped_gemm_swiglu.cuh` and does only
step 1+2, built for `sm_100a` (the example gates on `CUTLASS_ARCH_MMA_SM100_SUPPORTED`,
`75_blackwell_grouped_gemm.cu:100`). Keep it out of the TE build until step 3 passes.

---

## Appendix · exact file:line evidence index

| Claim | File:line |
|---|---|
| CUTLASS 4.2.0 in tree | `3rdparty/cutlass/CHANGELOG.md:5` |
| Base SM100 schedule (2-SM, 256×256×64, cluster 2×1×1) | `transformer_engine/common/gemm/cutlass_grouped_gemm.cuh:120-133` |
| Base host ptr-array loop, `ptr_B_host[i]=dptr`, packed strides | `cutlass_grouped_gemm.cuh:287-306` |
| FMHA two CollectiveBuilder MMAs | `examples/77_.../sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:90-105` |
| FMHA two TMEM accumulators via `partition_fragment_C` + offset | same file `:293-304` |
| FMHA TMEM column-map enum (512 total) | same file `:121-134` |
| FMHA `cute::gemm` into TMEM acc; `accumulate_=Zero/One` | same file `:334,726-730` |
| FMHA load→mma / mma→consumer pipelines | same file `:145-168` |
| FMHA epilogue: TMEM-load acc → convert → store | same file `:777-866`, epilogue hpp `:199-213` |
| FMHA kernel warp roles + reg_set + TMEM alloc/free | `kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:53-72,539,625` |
| `Allocator2Sm` / 512 cols / allocate / free | `cute/arch/tmem_allocator_sm100.hpp:117-181` |
| Stock kernel auto-selects Allocator1Sm/2Sm; ACC double-buffer | `gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp:187-188`; `gemm/collective/sm100_mma_array_warpspecialized.hpp:477-478` |
| Stock single B-load TMA copy (we add a 2nd) | `gemm/collective/sm100_mma_array_warpspecialized.hpp:646-648,778-779` |
| ex.75 ptr-array host (`ptr_B`, `make_cute_packed_stride`) | `examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu:484,520,623` |
| `SiLu` activation (fp32) | `cutlass/epilogue/thread/activation.h:452-460` |
| No GLU/pairwise fusion node (only single-source act) | `cutlass/epilogue/fusion/operations.hpp` (grep: only `LinCombEltAct*`) |
| Cluster launch (grid/cluster/smem, dyn-smem attr) | `examples/77_.../device/fmha.hpp:171-181,217-222` |

---

## P1 Step-2 device kernel — implementation notes

Implemented in `transformer_engine/common/gemm/cutlass_grouped_gemm_swiglu.cuh`
(`Sm100SwiGluKernel<Config>` device `operator()` + `LaunchSwiGluSingleExpert<>` host launcher) and
exercised by `qa/swiglu_single_expert_test.cu`. Single-expert only: `num_groups=1`, plain single
pointers `X,W1,A`, uniform `M=256, N=I=128, K=d=128`, one CTA-cluster = one output tile.

### Builder / schedule chosen
**`KernelTmaWarpSpecialized2SmSm100` (NON-ptr-array), 2-SM / cluster `2×1×1`, TileShape `256×128×64`.**
Rationale: the non-ptr-array single-tensor collective
`include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp` builds its TMA atoms from plain
`(ptr,stride)` in `to_underlying_arguments` (`:352-415`) with **no per-group tensormap swap** — exactly
what single-expert needs. We use `Config::CollectiveMma` purely as a **type + TMA-descriptor provider**
(à la FMHA): we call `CollectiveMma::to_underlying_arguments(...)` twice host-side (once with
`ptr_B=W1_gate`, once with `ptr_B=W1_up=W1+I*d`) and keep `tma_load_a`(=X) + both `tma_load_b`. The
device kernel then hand-rolls Load / MMA / Epilogue mirroring the collective's own `load_init`
(`:494-544`), `load` (`:585-626`), `mma_init` (`:548-576`), `mma` (`:648-708`). This diverges from the
draft's original `PtrArrayTmaWarpSpecialized2Sm` plan (§2.1) — chosen because it removes the entire
device-side tensormap-update mechanism (hardest-point #4) for step-2, and re-adds it only at P2.

Warp roles (256 threads = 8 warps): **warp0=Load** (TMA producer, 1 warp), **warp1=MMA** (UMMA +
TMEM owner, 1 warp), **warps4-7=Epilogue** (warpgroup-1, TMEM→reg→silu·mul→global store, 4 warps),
rest Empty. `cutlass::canonical_warp_idx_sync()` + `warp_role()`.

### How each of the 5 hardest points was resolved (actual CUTLASS calls)
1. **Dual-acc TMEM budget.** `cute::TMEM::Allocator2Sm` (2-SM ⇒ size(AtomThrShapeMNK)==2),
   `tmem_allocator.allocate(Sm100TmemCapacityColumns/*512*/, &ss.tmem_base_ptr)` in the **MMA warp**
   (`tmem_allocator_sm100.hpp:135`). Two **single-stage** accumulators built as offset views of one
   `partition_fragment_C(tiled_mma, take<0,2>(TileShape{}))` (the collective itself uses
   `take<0,2>(TileShape{})` at `sm100_mma_warpspecialized.hpp:453`): `acc_gate.data() = tmem_base +
   uint32_t(SwiGluTmem::ACC_GATE/*0*/)`, `acc_up.data() = tmem_base + 128`. N=128 ⇒ 2×128=256 ≤ 512.
   No ACC_PIPE double-buffer (FMHA precedent — its S/O accs are plain fragments, `:293-304`).
2. **Warp-spec roles & TMEM ownership.** MMA warp owns `allocate`; exactly one Epilogue warp
   (`canonical_warp_idx_sync()==4`) calls `tmem_allocator.free(...)`. `warpgroup_reg_dealloc<40>()` on
   Load/MMA, `warpgroup_reg_alloc<160>()` on Epilogue.
3. **Pipelines.** `PipelineLoad = PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShapeMNK>`
   (load→mma; one `producer_commit`/stage after X+gateB+upB; `transaction_bytes =
   TmaTransactionBytes + bits_to_bytes(2-SM × cosize(take<0,3>(SmemLayoutW1)) × 16)` to add the 2nd B
   box). `PipelineEpi = PipelineUmmaAsync<1, AtomThrShapeMNK>` (mma→epi; **single** `producer_commit`
   AFTER the whole K-loop so the epilogue never sees a half-finished `acc_up`). The MMA→epi commit uses
   the 2-SM `umma_arrive_multicast_2x1SM` automatically (`sm100_pipeline.hpp:263`). Epilogue calls
   `cutlass::arch::fence_view_async_tmem_store()` before TMEM-loads (`arch/barrier.h:897`).
4. **2nd TMA descriptor.** Built host-side via a second `to_underlying_arguments` call with
   `ptr_B=W1_up`; device Load warp issues **three** `copy(tma.with(*bar,mcast), gXY(_,k), sXY(_,wr))`
   (X + gate-B + up-B) into three smem buffers (`smem_x`, `smem_w1_gate`, `smem_w1_up`). gate/up share
   `StrideW1`, `SmemLayoutW1`, and the B-multicast mask, so this is the single-B machinery duplicated.
5. **`+I*row_stride` offset.** `W1_up = W1 + (int64_t)I*d` (RowMajor `[2I,d]` ⇒ row stride = d
   elements; `int64_t` to avoid int32 overflow for large experts). `StrideW1` identical for gate & up;
   `make_cute_packed_stride(StrideW1{}, {I,d,1})` (N=I rows, `util/packed_stride.hpp:78`). The 16-byte
   TMA alignment of `W1_up` holds because `I*d` is a multiple of 8 (d%8==0).

### Epilogue chosen: direct global store (not TMA store)
After `consumer_wait`, load both accs TMEM→reg with `make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{},
tAcc_gate)` (FMHA `:539,545`), compute `silu(gate)*up` in fp32 via
`cutlass::epilogue::thread::SiLu<float>` (`activation.h:452`), cast to bf16, and write each owned
element straight to global `A` using the **identity-coordinate tensor**
`tiled_mma.get_slice(0).partition_C(make_identity_tensor(take<0,2>(TileShape{})))` to get `(row,col)`
(FMHA `:798-800`, row/col via `get<0>/get<1>`). This avoids building an O-smem layout + `SM90_TMA_STORE`
descriptor + `partition_S/D` — the single biggest source of epilogue compile errors — at a small perf
cost that is irrelevant for the step-2 numeric gate.

### Launch
`cutlass::ClusterLauncher::make_cluster_launch_config(grid=1cluster, cluster=2×1×1, block=256, smem)` +
`cudaLaunchKernelExC` on `cutlass::device_kernel<Sm100SwiGluKernel<Config>>`
(`device_kernel.h:118` needs `Params`, `MaxThreadsPerBlock`, `MinBlocksPerMultiprocessor`,
`operator()(params,smem)` — all provided). Sets `cudaFuncAttributeMaxDynamicSharedMemorySize` when
smem ≥ 48 KB and `cudaFuncAttributeNonPortableClusterSizeAllowed` (matches `ClusterLauncher::init`).

### Ranked most-likely first-compile failures (fix-fast list for the integrator)
1. **2-SM accumulator row mapping in the epilogue.** `partition_C(make_identity_tensor((256,128)))`
   yields intra-tile rows 0..255, but each CTA's TMEM physically holds only its 128 datapaths. If the
   coordinate→global-row mapping is off by the CTA half, the bottom 128 rows of `A` will be wrong (or
   double-written). May need `+ 128*block_rank_in_cluster` on the row, or to restrict each CTA's
   epilogue to its own M-half. **This is the #1 thing to verify against output numerics.**
2. **`PipelineUmmaAsync` 2x1SM consumer_release semantics.** `consumer_release_2x1SM` uses
   `umma_arrive_2x1SM_sm0` (only sm0's empty barrier). With an epilogue warpgroup on *both* CTAs, the
   arrive count / which CTA releases may mismatch `consumer_arv_count`. If it hangs, gate the release to
   one CTA or adjust `consumer_arv_count`.
3. **`make_fragment_A/B` as a static vs instance call.** Used as `TiledMma::make_fragment_A(sX)`
   (static, matches `sm100_mma_warpspecialized.hpp:558`). If the overload requires an instance, switch
   to `tiled_mma.make_fragment_A(sX)`.
4. **`partition_fragment_C` data() assignment type.** `acc.data() = uint32_t` mirrors FMHA `:527`; if
   the tmem engine rejects the `uint32_t+uint32_t`, use `acc.data().get() + offset` like FMHA `:297`.
5. **`Config::CollectiveMma::Arguments` aggregate init.** We default-construct then assign
   `ptr_A/dA/ptr_B/dB`; if the struct has no default ctor for the runtime-datatype members, brace-init
   the trailing fields.
6. **`transaction_bytes` for 3 TMA boxes.** If TMA reports a transaction-byte mismatch, recompute as
   `bits_to_bytes(2-SM × (cosize(take<0,3>(SmemLayoutX)) + 2×cosize(take<0,3>(SmemLayoutW1))) × 16)`.
7. **`cluster_sync()` placement.** Done after `init_masks` + `fence_barrier_init`; if barriers aren't
   visible cluster-wide before first use, replace with `pipeline_init_arrive_relaxed`/`_wait`
   (`sm90_pipeline.hpp:1364,1377`).

### Evidence index (Step-2 additions)
| Claim | File:line |
|---|---|
| Single-tensor SM100 collective: Arguments/Params/`to_underlying_arguments` | `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:291-415` |
| Single-tensor collective load_init / load / mma_init / mma | same `:494-544,585-626,548-576,648-708` |
| Acc fragment shape `take<0,2>(TileShape)` via `partition_shape_C` | same `:453` |
| `make_fragment_A/B` static on TiledMma | same `:558-559` |
| `PipelineUmmaAsync<Stages,AtomThrShapeMNK>` 2x1SM commit/release | `include/cutlass/pipeline/sm100_pipeline.hpp:118,260-277` |
| `PipelineTmaUmmaAsync` Params (transaction_bytes/role/is_leader) | `include/cutlass/pipeline/sm90_pipeline.hpp:292-299` |
| `make_cute_packed_stride` (RowMajor [M,K,L]) | `tools/util/include/cutlass/util/packed_stride.hpp:78` |
| `ClusterLauncher::make_cluster_launch_config` + `cudaLaunchKernelExC` | `include/cutlass/cluster_launch.hpp:150-208,248` |
| `device_kernel<Operator>` entry contract | `include/cutlass/device_kernel.h:112-126` |
| `fence_view_async_tmem_store/load` | `include/cutlass/arch/barrier.h:885,897` |
