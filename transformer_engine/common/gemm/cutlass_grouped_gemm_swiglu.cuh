/***************************************************************************************************
 * F2 route#2 — SwiGLU-fused MoE up-proj grouped GEMM (SM100 tcgen05, dual TMEM accumulator).
 *   A[T,I] = silu(X·W1[0:I]ᵀ) · (X·W1[I:2I]ᵀ),  W1 kept standard [2I,d] (no permute → zero-Muon).
 *
 * STATUS: P1 Step-3 — single-launch GROUPED (multi-expert) device kernel (this file).
 *   §2.1 verified type config + §2.2 TMEM column map (unchanged from Step-1).
 *   §2.3 Sm100SwiGluKernel  : custom warp-specialized kernel (Load / MMA(dual cute::gemm) / Epilogue).
 *           GROUPED: the expert is DERIVED FROM THE M-TILE (tokens contiguous by expert), so ONE X
 *           TMA desc over [G*Me,d] + ONE W1 TMA desc over [G*2I,d] suffice (no ptr arrays / tensormap
 *           swaps); gate/up are the two per-expert N-tile slices of the SAME W1 descriptor.
 *   §2.4  LaunchSwiGluGrouped      : grouped host launcher (2 TMA descs, single cluster-launch).
 *   §2.4b LaunchSwiGluSingleExpert : thin G=1 wrapper over LaunchSwiGluGrouped.
 *   SCOPE: contiguous stacked MoE, UNIFORM Me; X:[G*Me,d], W1:[G*2I,d], A:[G*Me,I]; N=I=128,K=d.
 *          Assumes Me % kTileM == 0 and I % kTileN == 0 (non-uniform Me ⇒ per-expert prefix-sum arrays).
 *
 * Patterned on (all file:line are this repo's bundled CUTLASS 4.2.0):
 *   include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp   (single-tensor SM100 collective:
 *       to_underlying_arguments:352, load_init:494, load:585, mma_init:548, mma:648 — we mirror these)
 *   examples/77_blackwell_fmha/.../sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
 *       (dual TMEM acc as partition_fragment_C+offset:293-304; gemm_zero_acc:334; epilogue
 *        TMEM-load→convert→store:795-863)
 *   examples/77_blackwell_fmha/.../sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
 *       (warp roles:53-72, canonical_warp_idx_sync:260, reg_set, TMEM allocate/free:539,625)
 *   include/cute/arch/tmem_allocator_sm100.hpp  (Allocator2Sm:117, allocate:135, free:159)
 *   include/cutlass/cluster_launch.hpp (ClusterLauncher:81) ; device_kernel.h:118 (entry wrapper)
 **************************************************************************************************/
#pragma once

#include <type_traits>

// cute/tensor.hpp MUST come first: it pulls cute/atom/copy_atom.hpp (defines Copy_Atom) before
// cute/algorithm/copy.hpp is parsed. The cute/arch/* + cute/atom/copy_traits_* headers below
// transitively include copy.hpp; if they precede tensor.hpp, copy.hpp sees Copy_Atom undefined and
// re-declares copy_if/copy (a latent include-order bug — surfaces as CUTLASS copy.hpp compile errors).
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm90_tma.hpp"         // cute::SM90_TMA_STORE, tma_store_fence/arrive/wait
#include "cute/arch/tmem_allocator_sm100.hpp"  // cute::TMEM::Allocator2Sm, Sm100TmemCapacityColumns
#include "cute/atom/copy_traits_sm90_tma.hpp"  // make_tma_copy(SM90_TMA_STORE,...) traits + tma_partition
#include "cutlass/arch/barrier.h"       // fence_view_async_tmem_store, NamedBarrier
#include "cutlass/arch/reg_reconfig.h"  // warpgroup_reg_alloc/dealloc
#include "cutlass/bfloat16.h"
#include "cutlass/cluster_launch.hpp"            // cutlass::ClusterLauncher
#include "cutlass/cutlass.h"                     // canonical_warp_idx_sync, NumThreadsPerWarp
#include "cutlass/device_kernel.h"               // cutlass::device_kernel<Operator>
#include "cutlass/epilogue/thread/activation.h"  // cutlass::epilogue::thread::SiLu
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"  // TagToStride*_t
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/numeric_conversion.h"    // NumericArrayConverter
#include "cutlass/numeric_size.h"          // cutlass::bits_to_bytes
#include "cutlass/pipeline/pipeline.hpp"   // PipelineTmaUmmaAsync, PipelineUmmaAsync
#include "cutlass/util/packed_stride.hpp"  // cutlass::make_cute_packed_stride

namespace transformer_engine {
namespace grouped_gemm_swiglu {

using namespace cute;

using ProblemShapeType = cute::Shape<int, int, int>;  // (M=tokens, N=I, K=d)
using ProblemShape = cutlass::gemm::GroupProblemShape<ProblemShapeType>;

// ============================================================================================
// §2.1 · Type config — reuse the FINAL SM100 2-SM schedule of cutlass_grouped_gemm.cuh, but with
//        N-tile = 128 (NOT 256): two fp32 accumulators must share the 512-col TMEM. We instantiate
//        ONE CollectiveMma type-provider and use its TiledMma twice (gate, up) at two TMEM offsets.
//
// BUILDER CHOICE (P1 Step-2): for SINGLE-EXPERT we use the NON-ptr-array schedule
// KernelTmaWarpSpecialized2SmSm100 (not KernelPtrArrayTmaWarpSpecialized2SmSm100).  Rationale:
// the non-ptr-array single-tensor collective (sm100_mma_warpspecialized.hpp) builds its TMA atoms
// from plain (ptr,stride) in to_underlying_arguments (line 352), with NO per-group tensormap swap —
// exactly what single-expert needs.  Its Params::TMA_A / TMA_B and TransactionBytes are reused here.
// (The ptr-array variant would force device-side tensormap updates we explicitly skip for step-2.)
// ============================================================================================
// TUNABLES (template params, defaulted to the current tuned values so default behavior is identical):
//   TileM_/TileN_/TileK_ : TileShape extents (256/64/16 = the tuned 2-SM optimum, see notes below).
//   kStages_             : load-pipeline depth (16 = the clean 398-TFLOPS config; perf ≈ the old 10).
//   ClusterM_            : cluster X-extent (2 = 2-SM, 1 = 1-SM); selects the KernelSchedule below.
//   MinBlocks_           : launch-bounds MinBlocksPerMultiprocessor hint (1 = current default).
//   AccStages_           : TMEM accumulator pipeline depth (2 = DOUBLE-BUFFERED = current default → MMA of
//                          tile N+1 overlaps the epilogue of tile N; 1 = single-buffer serialized handoff
//                          = the old behavior). Tunable; the per-tile buffer alternation is implemented.
template <typename Element_ /*bf16/fp16*/, typename ElementOut_ /*bf16/fp16*/, int TileM_ = 256,
          int TileN_ = 64, int TileK_ = 16, int kStages_ = 16, int ClusterM_ = 2,
          int MinBlocks_ = 1, int AccStages_ = 2>
struct SwiGluConfig {
  static_assert(ClusterM_ == 1 || ClusterM_ == 2, "ClusterM_ must be 1 (1-SM) or 2 (2-SM)");

  using Element = Element_;  // X and W1 element
  using ElementAcc = float;  // fp32 accumulate (required)
  using ElementOut = ElementOut_;

  using ArchTag = cutlass::arch::Sm100;
  using OpClass = cutlass::arch::OpClassTensorOp;

  using LayoutX =
      cutlass::layout::RowMajor;  // A operand (M,K)=(tokens,d): RowMajor ⇒ K-contiguous ✓
  // B operand (N,K)=(I,d). W1 is physically RowMajor [I,d] (row n contiguous in K), i.e. K-CONTIGUOUS.
  // CUTLASS's B-operand convention is FLIPPED vs A: TagToStrideB<RowMajor> = Stride<_1,int,int> makes the
  // N-mode unit-stride (N-contiguous), which would read W1 transposed (acc[m][n]=X[m]·colₙ(W1) instead of
  // X[m]·rowₙ(W1)). The K-contiguous physical layout is TagToStrideB<ColumnMajor> = Stride<int,_1,int>.
  // (cutlass/detail/layout.hpp:79/86; make_cute_packed_stride then yields the correct (d,1,0) stride.)
  using LayoutW1 = cutlass::layout::ColumnMajor;  // B operand (N,K)=(I,d): K-contiguous weights
  using LayoutA = cutlass::layout::RowMajor;      // output A[T,I]

  static constexpr int AlignX = 128 / cutlass::sizeof_bits<Element>::value;     // 8 (bf16)
  static constexpr int AlignW1 = 128 / cutlass::sizeof_bits<Element>::value;    // 8
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

  // N=128 (vs base 256) so two acc fit 512-col TMEM (see P1 draft §3 #1). Single-stage acc.
  // P1 Step-2 CORRECTNESS SPIKE uses 1-SM (ClusterShape (1,1,1), TileShape M=128 = single CTA tile):
  // the 1-SM accumulator is the standard layout the FMHA TMEM-read pattern (this kernel's template) is
  // PROVEN correct on. The 2-SM accumulator's V-mode layout scrambled the read (spike rounds 1-9). 2-SM
  // perf is a follow-up: flip these three back to (256,128,64)/(2,1,1)/2Sm + re-apply the documented
  // 2-SM deadlock fixes (leader-gate, ×2 epi arrival, TMEM NamedBarrier) once 1-SM numerics pass.
  // 1-SM (WORKING, validated grouped @ 301 TFLOPS at user shape). 2-SM (256/(2,1,1)/2Sm) is numerically
  // CORRECT and works at <=2 clusters after the fixes below (load-pipeline is_leader gated to the leader
  // CTA; TMEM alloc+free in the same MMA warp via tmem_free_bar), BUT a deep TIMING RACE remains at
  // >2 clusters / repeated launches — "unspecified launch failure" with NO memcheck/synccheck error and
  // 2-SM (all cycling bugs fixed). Small TileN/TileK + large TileM (capped at 256 for 2-SM = 128 TMEM
  // datapaths × 2 CTAs). Small N/K tiles shrink per-stage smem → many more pipeline stages (the lever
  // that works: kStages 2→3 gave +11%; TileN=256 was WORSE for losing stages). TileN=64,TileK=16 →
  // per-stage = X(256×16=8KB)+2×W1(64×16=2KB)=12KB → kStages=8 → 96KB « 228KB (room for even more).
  // TUNED OPTIMUM: TileM=256 (2-SM max), TileN=64 (sweet spot — 128 loses pipeline stages, 32 collapses
  // MMA efficiency), TileK=16. With kStages=16 → 344 TFLOPS at the user shape (M=16384,I=512,K=2048),
  // +18% vs the untuned 292; +15-31% on M-large/N-K-small shapes.
  // Tunable via TileM_/TileN_/TileK_ (defaults reproduce the tuned 256/64/16 above). cute::Int<N> is
  // cute::C<N>; cute::_256 == cute::Int<256>, so the default instantiation is the same type as before.
  using TileShape = cute::Shape<cute::Int<TileM_>, cute::Int<TileN_>, cute::Int<TileK_>>;
  // ClusterM_==2 ⇒ cute::Int<2> == cute::_2 ⇒ identical to the old cute::Shape<_2,_1,_1>.
  using ClusterShape = cute::Shape<cute::Int<ClusterM_>, cute::_1, cute::_1>;
  // Schedule follows the cluster: 2-SM (ClusterM_==2) is the current default; 1-SM otherwise.
  using KernelSchedule =
      std::conditional_t<ClusterM_ == 2, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
  static_assert(2 * cute::size<1>(TileShape{}) <= 512,
                "gate+up accumulators must fit 512-col TMEM");

  // Pipeline depth (tunable via kStages_; default 16 = the clean 398-TFLOPS config). TileN=64,TileK=16
  // → ~8KB/stage; the smem static_assert below guards against exceeding the 228KB SM100 capacity.
  static constexpr int kStages = kStages_;

  // Launch-bounds MinBlocksPerMultiprocessor hint, read by the kernel (default 1). The warpgroup
  // reg-reconfig (wg0 dealloc<40> → wg1 alloc<160>) keeps the epilogue at 160 dynamically.
  static constexpr int kMinBlocks = MinBlocks_;
  // TMEM accumulator pipeline depth (default 2 = double-buffered). AccStages_==2 reserves & USES two acc
  // TMEM buffers and runs a 2-stage PipelineEpi so MMA(tile N+1) overlaps epilogue(tile N); AccStages_==1
  // collapses to the single-buffer serialized handoff. The per-tile buffer alternation is in operator().
  static constexpr int kAccStages = AccStages_;

  // CollectiveMma used purely as a type-provider (TiledMma / SmemLayout / fragments / TMA atoms /
  // TransactionBytes), exactly like FMHA reuses CollectiveBuilder for CollectiveMmaQK/PV.
  using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, Element, LayoutX, AlignX, Element, LayoutW1, AlignW1, ElementAcc, TileShape,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>, KernelSchedule>::CollectiveOp;

  using TiledMma = typename CollectiveMma::TiledMma;
  using AtomThrShapeMNK = typename CollectiveMma::AtomThrShapeMNK;  // (2,1,1) under 2-SM
  using SmemLayoutX = typename CollectiveMma::SmemLayoutA;          // (MMA,M,K,PIPE)
  using SmemLayoutW1 = typename CollectiveMma::SmemLayoutB;         // (MMA,N,K,PIPE)
  static constexpr int Stages = CollectiveMma::DispatchPolicy::Stages;

  // CTA-local tile (after dividing by the 2-SM atom): MMA M is split across 2 CTAs.
  using CtaShapeMNK = typename CollectiveMma::CtaShape_MNK;  // (128,128,64) for 256/2-SM

  // Stride aliases used by the host launcher (single expert, single pointer).
  // X/W1 strides come from the collective itself (guaranteed compatible with its Arguments/TMA);
  // A (output) stride is just RowMajor (M,N,L) and is only used for the direct global store.
  using StrideX = typename CollectiveMma::StrideA;
  using StrideW1 = typename CollectiveMma::StrideB;
  using StrideA = cutlass::detail::TagToStrideC_t<LayoutA>;

  // The single-tensor collective's TMA atom + Params types (built by to_underlying_arguments).
  using MainloopArguments = typename CollectiveMma::Arguments;  // {ptr_A,dA, ptr_B,dB, ...}
  using MainloopParams = typename CollectiveMma::Params;        // {tma_load_a, tma_load_b, ...}
  using TMA_X = typename MainloopParams::TMA_A;
  using TMA_W1 = typename MainloopParams::TMA_B;

  static constexpr uint32_t TmaTransactionBytes = CollectiveMma::TmaTransactionBytes;

  // ---- TMA-STORE epilogue (sA → global A) ----------------------------------------------------
  // Per-CTA output staging-tile extents (must match Sm100SwiGluKernel::kEpiTileM/kEpiTileN; defined
  // here too so the host launcher can build the store descriptor and Params can name its type):
  //   kEpiTileM = TileShape M / AtomThrShapeMNK (per-CTA M slice = 128 for 256/(2,1,1); 128 for 1-SM)
  //   kEpiTileN = TileShape N (output n-tile width = 64)
  static constexpr int kEpiTileM_cfg =
      cute::size<0>(cute::take<0, 2>(TileShape{})) / int(cute::size(AtomThrShapeMNK{}));
  static constexpr int kEpiTileN_cfg = cute::size<1>(cute::take<0, 2>(TileShape{}));
  // sA smem layout: row-major (kEpiTileM, kEpiTileN). PLAIN (no swizzle) — this is exactly the layout
  // the epilogue already uses for ss.tensors.smem_out, and make_tma_copy below builds the matching
  // (no-swizzle) TMA-store descriptor box = product_each(shape) = (kEpiTileM,kEpiTileN). The contiguous
  // (N) box width = kEpiTileN*sizeof(ElementOut) = 64*2 = 128B (16B-aligned ✓; TMA store needs ≥16B).
  using SmemLayoutA = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kEpiTileM_cfg>{}, cute::Int<kEpiTileN_cfg>{}),
      cute::make_stride(cute::Int<kEpiTileN_cfg>{}, cute::_1{})));
  // TMA-store descriptor type over the WHOLE output A[M,I] (row-major, rank-2). The descriptor is built
  // ONCE on the host over the true (M,I) extent; per-tile/per-CTA placement is just the box coordinate
  // (local_tile by (kEpiTileM,kEpiTileN)). OOB on a partial tile is clamped by the descriptor extent.
  using TMA_A = decltype(make_tma_copy(
      cute::SM90_TMA_STORE{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOut*>(nullptr)),
                        cute::make_layout(cute::make_shape(int(0), int(0)),
                                          cute::make_stride(int(0), cute::_1{}))),
      SmemLayoutA{}));
};

// ============================================================================================
// §2.2 · TMEM column map for the two accumulators (mirrors FMHA TmemAllocation:121-134).
//        512 cols total; with N=128 one fp32 acc = 128 cols. ACC_GATE | ACC_UP = 256 ≤ 512.
// ============================================================================================
enum class SwiGluTmem : uint32_t {
  kAccCols = 128,  // == TileShape N
  ACC_GATE = 0,
  ACC_UP = ACC_GATE + kAccCols,  // 128
  kEnd = ACC_UP + kAccCols,      // 256  (≤ 512)
};

// ============================================================================================
// §2.3 · Custom single-expert warp-specialized kernel.
//        Warp roles:  warp 0 = Load (TMA producer), warp 1 = MMA (UMMA + TMEM owner),
//                      warps 4..7 (warpgroup 1) = Epilogue (TMEM→reg→silu·mul→global store).
//        We mirror the single-tensor collective's load/mma partitioning verbatim, but issue
//        a SECOND B-TMA (up) and a SECOND cute::gemm into a second TMEM accumulator.
// ============================================================================================
template <typename Config>
struct Sm100SwiGluKernel {
  using Element = typename Config::Element;
  using ElementAcc = typename Config::ElementAcc;
  using ElementOut = typename Config::ElementOut;
  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  using TiledMma = typename Config::TiledMma;
  using AtomThrShapeMNK = typename Config::AtomThrShapeMNK;
  using SmemLayoutX = typename Config::SmemLayoutX;
  using SmemLayoutW1 = typename Config::SmemLayoutW1;
  using StrideX = typename Config::StrideX;
  using StrideW1 = typename Config::StrideW1;
  using StrideA = typename Config::StrideA;
  using TMA_X = typename Config::TMA_X;
  using TMA_W1 = typename Config::TMA_W1;
  using TMA_A = typename Config::TMA_A;  // TMA-STORE descriptor type over A[M,I] (row-major)
  using SmemLayoutA =
      typename Config::SmemLayoutA;  // sA staging-tile smem layout (kEpiTileM,kEpiTileN)
  static constexpr int Stages = Config::Stages;

  using ArchTag = cutlass::arch::Sm100;
  // TMEM allocator follows the MMA atom: 2-SM (cta_group::2) when AtomThrShapeMNK size==2, else 1-SM.
  static constexpr bool kIs2Sm = (cute::size(AtomThrShapeMNK{}) == 2);
  using TmemAllocator =
      std::conditional_t<kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  // Load(warp0) -> MMA(warp1) : protects X + gate-B + up-B in smem. AtomThrShapeMNK threads the
  // 2-SM peer mask through the pipeline (sm100_mma_warpspecialized.hpp:153-156).
  using PipelineLoad = cutlass::PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShapeMNK>;
  // MMA(warp1) -> Epilogue(wg1) : protects BOTH TMEM accumulators (single commit after K-loop).
  // Config::kAccStages stages → double-buffered acc pipeline (AccStages=2): the MMA may produce the
  // accumulator for tile (N+1) in buffer (N+1)%kAccStages while the epilogue still reads tile N's result
  // from buffer N%kAccStages → MMA(tile N+1) OVERLAPS epilogue(tile N). AccStages=1 collapses to the old
  // 1-stage serialized handoff (identical behavior). Each stage gets its OWN mbarrier pair; the per-tile
  // acc buffer offset (= stage index * kAccBufStride) is selected from epi_prod.index()/epi_cons.index().
  using PipelineEpi = cutlass::PipelineUmmaAsync<Config::kAccStages, AtomThrShapeMNK>;
  // Cols per acc buffer (one stage) = gate(kEpiTileN) + up(kEpiTileN) = 2*kEpiTileN. The acc buffer for a
  // given tile is its acc-pipeline stage index: buf_base = stage_index * kAccBufStride (defined below
  // after kEpiTileN). gate lives at buf_base, up at buf_base + kEpiTileN within the stage's buffer.

  // Warp layout: 2 warpgroups (256 threads). wg0 = {Load, MMA, -, -}; wg1 = Epilogue (4 warps).
  static constexpr int NumEpiWarps = 4;
  static constexpr int NumWarps = 8;  // 256 threads
  static constexpr int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  // Launch-bounds occupancy hint (tunable via Config::kMinBlocks; default 1). ncu showed occupancy
  // capped at 1 block/SM by registers (160/thr) AND smem. The warpgroup reg-reconfig (wg0 dealloc<40>
  // frees regs for wg1 alloc<160>) keeps the epilogue at 160 dynamically. Setting this >1 hints the
  // compiler to fit that many blocks/SM (smem permitting under the 228KB cap).
  static constexpr int MinBlocksPerMultiprocessor = Config::kMinBlocks;
  static constexpr int NumEpiThreads =
      NumEpiWarps * cutlass::NumThreadsPerWarp;  // 128 epilogue threads

  // Epilogue SMEM staging tile (sA): the PER-CTA output tile that the epilogue writes coalesced to
  // global A.  Rows = the per-CTA M-block (TileShape M / 2-SM atom = 128 for 256/(2,1,1); 128 for
  // 1-SM).  Cols = the OUTPUT n-tile width (TileShape N = 64).  These are exactly the LOCAL (row,col)
  // ranges spanned by the epilogue coordinate tile tTMc (V-0 slice → rows[0,128), col[0,64)), so each
  // result maps to sA at its LOCAL (row,col) and then to the SAME global (row,col) it wrote before.
  static constexpr int kEpiTileM =
      cute::size<0>(take<0, 2>(TileShape{})) / int(cute::size(AtomThrShapeMNK{}));  // 128
  static constexpr int kEpiTileN = cute::size<1>(take<0, 2>(TileShape{}));          // 64
  // The TMA-STORE staging layout (Config::SmemLayoutA) and descriptor (Config::TMA_A) are built from
  // Config::kEpiTileM_cfg/kEpiTileN_cfg; assert they equal the Kernel's kEpiTileM/kEpiTileN so the smem
  // sA tile, the descriptor box, and the per-tile local_tile box-coordinate math all agree.
  static_assert(kEpiTileM == Config::kEpiTileM_cfg && kEpiTileN == Config::kEpiTileN_cfg,
                "Config TMA-store tile extents must match Kernel kEpiTileM/kEpiTileN");
  // Cols per acc buffer (ONE acc-pipeline stage) = gate(kEpiTileN) + up(kEpiTileN) = 2*kEpiTileN. With
  // AccStages>1 each stage occupies its own [stage*kAccBufStride, (stage+1)*kAccBufStride) TMEM window:
  // gate at stage*kAccBufStride, up at stage*kAccBufStride + kEpiTileN. The MMA writes the producer
  // stage's buffer (epi_prod.index()*kAccBufStride); the epilogue reads the consumer stage's buffer
  // (epi_cons.index()*kAccBufStride). These offsets are set PER TILE (see operator()).
  static constexpr int kAccBufStride = 2 * kEpiTileN;  // 128 for TileN=64
  // TMEM columns actually used = kAccStages buffers × (gate(TileN) + up(TileN)) = kAccStages*2*TileN.
  // Allocate ONLY these (not the full 512) so MULTIPLE blocks share the SM's 512-col TMEM → unlocks
  // occupancy (ncu: full-512 alloc capped achieved occ at 1 block/SM despite 2-block theoretical).
  // Must be a power of two in [32,512] for tcgen05.alloc.
  // Config::kAccStages scales the reservation AND is now FUNCTIONALLY used (double-buffered acc pipe):
  //   AccStages=1 → 2*TileN cols (e.g. 128 for TileN=64): single buffer, old 1-stage serialized behavior.
  //   AccStages=2 → 4*TileN cols (e.g. 256 for TileN=64): two buffers, MMA(N+1) overlaps epilogue(N).
  // CONSTRAINT: kAccStages*2*TileN ≤ 512. For TileN=64 → AccStages=2 gives 256 ≤ 512 ✓ (AccStages up to
  // 4 would fit). For TileN=256 the single-buffer assert below (2*TileN=512) already forbids any double
  // buffering (AccStages=2 → 1024 > 512); the static_assert here catches that explicitly.
  static constexpr int kTmemCols =
      Config::kAccStages * 2 * kEpiTileN;  // 256 for TileN=64, AccStages=2
  static_assert(kTmemCols >= 32 && kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0,
                "kTmemCols (= AccStages*2*TileN) must be a pow2 in [32,512]; "
                "for AccStages=2 keep TileN<=128 (AccStages*2*TileN=256<=512)");
  enum WarpRole { kLoad = 0, kMMA = 1, kEpilogue = 2, kEmpty = 3 };
  static CUTLASS_DEVICE WarpRole warp_role(int warp_idx) {
    if (warp_idx == 0) return kLoad;
    if (warp_idx == 1) return kMMA;
    if (warp_idx >= 4 && warp_idx < 4 + NumEpiWarps) return kEpilogue;
    return kEmpty;
  }

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutX>> smem_x;
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutW1>> smem_w1_gate;
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutW1>> smem_w1_up;
      // Epilogue output-staging tile sA (kEpiTileM × kEpiTileN, row-major).  Separate (NOT union-ed)
      // field: the input buffers above are DEAD by epilogue time, but a separate field is the simplest
      // correct option and the extra smem (128×64×sizeof(ElementOut bf16=2) = 16KB) keeps
      // SharedStorageSize well under the 228KB SM100 limit (input tensors ≈128KB at kStages=16).
      cute::ArrayEngine<ElementOut, kEpiTileM * kEpiTileN> smem_out;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _0> {
      alignas(16) typename PipelineLoad::SharedStorage load;
      alignas(16) typename PipelineEpi::SharedStorage epi;
    } pipelines;
    uint32_t tmem_base_ptr;
  };
  static constexpr int SharedStorageSize = static_cast<int>(sizeof(SharedStorage));
  // SM100 has 228 KB smem/CTA; the dynamic-smem attribute below requests SharedStorageSize.
  static_assert(
      SharedStorageSize <= (228 * 1024),
      "SwiGLU smem (X + gate-B + up-B + pipelines) exceeds SM100 capacity; reduce kStages.");

  // -------------------------------------------------------------------------------------------
  // Device params.  GROUPED (contiguous, uniform Mₑ): plain single pointers and ONE X TMA
  // descriptor over [G*Me, d] + ONE W1 TMA descriptor over [G*2I, d].  The gate/up slices and the
  // per-expert selection are derived IN-KERNEL from the m-tile (tokens are contiguous by expert),
  // so NO pointer arrays / tensormap swaps are needed.  Single-expert is just the G=1 special case.
  //   M  = G*Me  (total tokens across all experts)   N  = I  (output width)   K = d
  //   Me = tokens per expert (uniform)                W1N = G*2I (rows of the W1 descriptor)
  // -------------------------------------------------------------------------------------------
  struct Params {
    TMA_X tma_load_x;
    TMA_W1 tma_load_w1;  // ONE descriptor over [G*2I, d]; gate/up are per-expert N-tile slices
    TMA_A tma_store_a;   // ONE TMA-STORE descriptor over the whole output A[M,I] (row-major); the
    // per-tile/per-CTA placement is just the box coordinate (local_tile offset)
    ElementOut* ptr_A;
    StrideA dA;
    int M, N, K;  // M == G*Me, N == I, K == d
    int Me;       // tokens per expert (uniform)
    int W1N;      // == G*2I, the N-extent of the single W1 TMA descriptor
    // PERSISTENT (software grid-stride) scheduling fields.  The grid launches a FIXED, smaller set of
    // persistent clusters; each cluster grid-strides over a subset of the logical (m_tile,n_tile) tiles,
    // processing each fully before the next.  total_tiles = num_m_tiles * n_local_tiles is the LOGICAL
    // tile count (NOT multiplied by cluster); n_local_tiles = I/kTileN lets the kernel decode
    // m_tile = tile / n_local_tiles, n_tile = tile % n_local_tiles; num_clusters is the grid-stride step.
    int total_tiles;    // num_m_tiles * num_n_local_tiles (logical tiles to process)
    int n_local_tiles;  // num_n_local_tiles == I/kTileN (decode stride for m_tile/n_tile)
    int num_clusters;   // number of persistent clusters launched (grid-stride step)
    // VARLEN-M (uneven, TileM-aligned experts): device array [num_m_tiles] mapping each GLOBAL m-tile to
    // its expert id.  nullptr => UNIFORM Me fast path (e = m_tile / mtiles_per_expert).  Because experts
    // are TileM-aligned and packed, the X/A row offset is still global_m_tile*kTileM (unchanged); ONLY the
    // expert→W1-slice mapping becomes non-uniform, so this is the single varlen hook.  SonicMoE handles
    // arbitrary token counts upstream via token-rounding (pad each expert to a kTileM multiple).
    const int* m_tile_expert;
  };

  // ---- device entry ----
  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
    if (cute::thread0()) {
      printf("ERROR: Sm100SwiGluKernel requires SM100a/f MMA. Compile with -arch=sm_100a.\n");
    }
    return;
#else
    using X = Underscore;
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    WarpRole role = warp_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Prefetch TMA descriptors from the Load warp (X over [G*Me,d] + ONE W1 over [G*2I,d]).
    if (role == kLoad && lane_predicate) {
      cute::prefetch_tma_descriptor(params.tma_load_x.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_w1.get_tma_descriptor());
    }
    // Prefetch the TMA-STORE descriptor (A[M,I]) from the elected epilogue lane: the store is issued by
    // the epilogue warps, so warm its descriptor in their warpgroup (a hint only; harmless to prefetch).
    if (role == kEpilogue && lane_predicate && warp_idx == 4) {
      cute::prefetch_tma_descriptor(params.tma_store_a.get_tma_descriptor());
    }

    // ---- pipeline construction (one stamp of role per warp; FMHA kernel hpp:284-391) ----
    typename PipelineLoad::Params lp;
    lp.role = (role == kLoad) ? PipelineLoad::ThreadCategory::Producer
                              : PipelineLoad::ThreadCategory::Consumer;
    // 2-SM: ONLY the leader CTA's elected lane issues arrive_and_expect_tx on the cluster transaction
    // barrier (stock sm100_gemm_tma_warpspecialized.hpp:464 is_leader = lane && is_mma_leader_cta).
    // Both CTAs being is_leader → the follower double-sets expect_tx → cluster-barrier corruption that
    // faults only once the pipeline CYCLES (k_tile_count>Stages) — the exact crash compute-sanitizer
    // pinned to the follower CTA's producer_acquire. transaction_bytes already counts both CTAs (×Atom).
    lp.is_leader = (role == kLoad) && lane_predicate &&
                   ((block_rank_in_cluster % size(typename TiledMma::AtomThrID{})) == 0);
    // Three TMA boxes (X + gate-B + up-B) land in one stage. X+one-B == TmaTransactionBytes; we add
    // a second B box (gate and up B share the same SmemLayoutW1, hence equal byte counts).
    lp.transaction_bytes =
        Config::TmaTransactionBytes +
        cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cute::cosize(take<0, 3>(SmemLayoutW1{})) *
                               cute::sizeof_bits_v<Element>);
    lp.num_consumers = cutlass::NumThreadsPerWarp;  // single MMA warp consumes
    lp.initializing_warp = 0;
    PipelineLoad pipeline_load(ss.pipelines.load, lp, ClusterShape{},
                               /*InitBarriers*/ cute::true_type{},
                               /*InitMasks*/ cute::false_type{});

    typename PipelineEpi::Params ep;
    ep.role = (role == kMMA) ? PipelineEpi::ThreadCategory::Producer
                             : PipelineEpi::ThreadCategory::Consumer;
    ep.producer_arv_count = 1;  // MMA warp (1 thread elects)
    // 2-SM: both CTAs' epilogue consumer_release redirect (Sm100MmaPeerBitMask) onto the LEADER's
    // empty barrier, so it must expect size(AtomThrShapeMNK) * NumEpilogueThreads arrivals
    // (stock sm100_gemm_tma_warpspecialized.hpp:529). Without the ×2 the follower's barrier hangs.
    ep.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpiWarps * cutlass::NumThreadsPerWarp;
    ep.initializing_warp = 1;
    PipelineEpi pipeline_epi(ss.pipelines.epi, ep, ClusterShape{},
                             /*InitBarriers*/ cute::true_type{},
                             /*InitMasks*/ cute::false_type{});

    TmemAllocator tmem_allocator{};

    // Make pipeline-init visible cluster-wide, then init masks (sm100 gemm kernel:682,694).
    pipeline_load.init_masks(ClusterShape{});
    pipeline_epi.init_masks(ClusterShape{});
    cutlass::arch::fence_barrier_init();
    cute::cluster_sync();

    typename PipelineLoad::PipelineState load_prod =
        cutlass::make_producer_start_state<PipelineLoad>();
    typename PipelineLoad::PipelineState load_cons;
    typename PipelineEpi::PipelineState epi_prod =
        cutlass::make_producer_start_state<PipelineEpi>();
    typename PipelineEpi::PipelineState epi_cons;

    const int M = params.M, N = params.N, K = params.K;
    const int k_tile_count = K / size<2>(TileShape{});  // d / 64
    auto problem_shape_MNKL = make_shape(M, N, K, 1);

    // 2-SM leader predicate: a tcgen05.mma.cta_group::2 is ONE cluster-wide op issued by the even
    // (leader) CTA only; the peer must not issue MMA, release the load pipe, or commit the acc pipe
    // (stock sm100_gemm_tma_warpspecialized.hpp:427,761-771).
    const bool is_mma_leader_cta =
        (block_rank_in_cluster % size(typename TiledMma::AtomThrID{})) == 0;

    // Tile scheduler (PERSISTENT, GROUPED): the grid launches a FIXED set of params.num_clusters
    // persistent clusters; each cluster grid-strides over a subset of the logical (m_tile,n_tile_local)
    // tiles, processing each fully before the next (software grid-stride, NO hardware CLC).  kTileM/kTileN
    // are the FULL cluster-tile extents (TileShape); the per-CTA M-block (2-SM within-cluster split) is
    // added separately via block_rank in the epilogue.
    constexpr int kTileM = size<0>(take<0, 2>(TileShape{}));  // 128 (1-SM) / 256 (2-SM)
    constexpr int kTileN = size<1>(take<0, 2>(TileShape{}));  // 128
    constexpr int kClusterX = size<0>(ClusterShape{});        // 1 (1-SM) / 2 (2-SM)
    // PERSISTENT: this CTA's cluster id.  BOTH CTAs of a cluster share the same cluster_id (blockIdx.x is
    // contiguous within a cluster: CTAs [cluster_id*kClusterX, cluster_id*kClusterX+kClusterX)), so both
    // process the SAME tile sequence — required for the 2-SM cta_group::2 MMA (the M-split is per-CTA via
    // block_rank, not per-tile).  The grid-stride loop in each role branch is
    //   for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters)
    // and decodes m_tile = tile / n_local_tiles, n_tile_local = tile % n_local_tiles.
    const int cluster_id = blockIdx.x / kClusterX;

    // GROUPED expert derivation constants (contiguous, uniform Me) — tile-INVARIANT, computed once.
    // Tokens are contiguous by expert, so the expert is DERIVED FROM THE M-TILE — no pointer arrays /
    // tensormap swaps.  Assumes Me % kTileM == 0 and I % kTileN == 0 (uniform Me, no boundary predication).
    // NOTE: non-uniform Me would need per-expert cumulative-offset arrays (prefix sums of Mₑ) instead.
    // The per-tile expert e and gate/up n-tile derivation move INSIDE each role's grid-stride loop.
    const int mtiles_per_expert = params.Me / kTileM;  // m-tiles per expert
    const int nI_per_expert = N / kTileN;              // output n-tiles per expert (I/kTileN)
    const int n2_per_expert = (2 * N) / kTileN;        // W1 n-tiles per expert ((2*I)/kTileN)

    // MMA→Epilogue handoff for the TMEM base pointer: the MMA warp allocates TMEM and publishes
    // ss.tmem_base_ptr; the epilogue must not read it until then (stock NamedBarrier,
    // sm100_gemm_tma_warpspecialized.hpp:557,727-729,870-871). Count = MMA warp (32) + epi warps.
    cutlass::arch::NamedBarrier tmem_alloc_bar(
        (1 + NumEpiWarps) * cutlass::NumThreadsPerWarp,
        cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // TMEM free is sequenced by a cluster_sync() AFTER all role branches (see end of operator()): it is
    // reached by ALL warps of BOTH CTAs, so it guarantees (a) the epilogue finished reading TMEM and
    // (b) the two CTAs are ALIGNED before the cta_group::2 free — fixing both the intra-CTA "don't free
    // while epilogue reads" and the cross-CTA "both warp-1s must reach the paired dealloc together" races.

    // =========================================================================================
    // LOAD warp — TMA X-tile + gate row-block + up row-block per K-tile (single output tile).
    // Mirrors sm100_mma_warpspecialized.hpp load_init:494-544 + load:585-626, with a 2nd B TMA.
    // =========================================================================================
    if (role == kLoad) {
      cutlass::arch::warpgroup_reg_dealloc<40>();

      // Defer-sliced TMA tensors. X is over (M=G*Me, K); W1 is over (W1N=G*2I, K) — ONE descriptor.
      Tensor mX = params.tma_load_x.get_tma_tensor(make_shape(M, K, 1));
      Tensor mW = params.tma_load_w1.get_tma_tensor(make_shape(params.W1N, K, 1));

      Tensor gX =
          local_tile(mX, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});  // (BM,BK,m,k,l)
      Tensor gW =
          local_tile(mW, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});  // (BN,BK,n,k,l)

      ThrMMA cta_mma =
          TiledMma{}.get_slice(block_rank_in_cluster % size(typename TiledMma::AtomThrID{}));
      Tensor tCgX = cta_mma.partition_A(gX);  // (MMA,MMA_M,MMA_K,m,k,l)
      Tensor tCgW = cta_mma.partition_B(gW);  // (MMA,MMA_N,MMA_K,n,k,l)

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
      Tensor sWg = make_tensor(make_smem_ptr(ss.tensors.smem_w1_gate.begin()), SmemLayoutW1{});
      Tensor sWu = make_tensor(make_smem_ptr(ss.tensors.smem_w1_up.begin()), SmemLayoutW1{});

      // CTA-in-cluster layout for tma_partition (sm100_mma_warpspecialized.hpp:521-538).
      Layout cta_layout_mnk = make_layout(ClusterShape{});
      Layout cta_layout_vmnk =
          tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
      auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

      auto [tXgX, tXsX] = tma_partition(params.tma_load_x, get<2>(cta_coord_vmnk),
                                        make_layout(size<2>(cta_layout_vmnk)),
                                        group_modes<0, 3>(sX), group_modes<0, 3>(tCgX));
      // ONE W1 descriptor (params.tma_load_w1), same gmem partition tCgW, but TWO different smem
      // buffers (sWg, sWu).  This yields a single gmem-partition tensor tWggW/tWugW (identical, both
      // == the W1 partition) which we then slice at the two DIFFERENT n-tiles (gate vs up) below.
      auto [tWggW, tWgsWg] = tma_partition(params.tma_load_w1, get<1>(cta_coord_vmnk),
                                           make_layout(size<1>(cta_layout_vmnk)),
                                           group_modes<0, 3>(sWg), group_modes<0, 3>(tCgW));
      auto [tWugW, tWusWu] = tma_partition(params.tma_load_w1, get<1>(cta_coord_vmnk),
                                           make_layout(size<1>(cta_layout_vmnk)),
                                           group_modes<0, 3>(sWu), group_modes<0, 3>(tCgW));

      uint16_t mcast_mask_x = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
      uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

      // PERSISTENT grid-stride over logical tiles.  load_prod is NOT re-init per tile: it advances
      // continuously across tiles (TMA pipelines are designed for continuous use; the producer state
      // wraps by Stages).  producer_tail is drained ONCE after the loop (a per-tile tail would stall).
      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;  // GLOBAL token m-tile, across ALL experts
        const int n_tile_local =
            tile % params.n_local_tiles;  // OUTPUT column tile, ∈ [0, nI_per_expert)
        // VARLEN-M: look up the expert from the per-m-tile table (uneven, TileM-aligned experts); else
        // UNIFORM fast path (m_tile / mtiles_per_expert).  Only the W1 slice depends on e; X/A row =
        // m_tile*kTileM regardless (experts are packed + TileM-aligned).  One L1-hot gmem read per tile.
        const int e = (params.m_tile_expert != nullptr) ? params.m_tile_expert[m_tile]
                                                        : (m_tile / mtiles_per_expert);
        // W1 row math (verified): gate_w1_ntile*kTileN = e*2I + n_tile_local*kTileN  → gate rows of e ✓
        //                         up_w1_ntile  *kTileN = e*2I + I + n_tile_local*kTileN → up rows of e ✓
        const int gate_w1_ntile = e * n2_per_expert + n_tile_local;
        const int up_w1_ntile = gate_w1_ntile + nI_per_expert;

        // Slice THIS CTA's output tile; iterate K only.  X picks the GLOBAL m_tile (over [G*Me,d]);
        // gate picks the expert's gate row-tile (gate_w1_ntile), up picks the expert's up row-tile
        // (up_w1_ntile = gate_w1_ntile + nI_per_expert) — BOTH from the SAME W1 descriptor [G*2I,d].
        Tensor tXgX_k = tXgX(_, m_tile, _, _0{});  // (TMA, k)
        Tensor tWggWg_k = tWggW(_, gate_w1_ntile, _, _0{});
        Tensor tWugWu_k = tWugW(_, up_w1_ntile, _, _0{});

        for (int k = 0; k < k_tile_count; ++k) {
          pipeline_load.producer_acquire(load_prod);
          auto* bar = pipeline_load.producer_get_barrier(load_prod);
          int wr = load_prod.index();
          if (cute::elect_one_sync()) {
            copy(params.tma_load_x.with(*bar, mcast_mask_x), tXgX_k(_, k), tXsX(_, wr));
            copy(params.tma_load_w1.with(*bar, mcast_mask_b), tWggWg_k(_, k), tWgsWg(_, wr));
            copy(params.tma_load_w1.with(*bar, mcast_mask_b), tWugWu_k(_, k), tWusWu(_, wr));
          }
          ++load_prod;
        }
      }
      // Drain ONCE after all tiles so peer CTAs in the cluster don't exit early.
      pipeline_load.producer_tail(load_prod);
    }

    // =========================================================================================
    // MMA warp — allocate full TMEM once; two acc views at fixed col offsets; K-loop = 2 cute::gemm.
    // Mirrors sm100_mma_warpspecialized.hpp mma_init:548-576 + mma:648-708, doubled for gate/up.
    // =========================================================================================
    else if (role == kMMA) {
      cutlass::arch::warpgroup_reg_dealloc<40>();

      tmem_allocator.allocate(kTmemCols, &ss.tmem_base_ptr);
      __syncwarp();
      tmem_alloc_bar.arrive();  // publish ss.tmem_base_ptr to the epilogue warps (#3)

      TiledMma tiled_mma;
      // Two accumulator TMEM views (FMHA mainloop:293-304). The FRAGMENT (layout) is buffer-INDEPENDENT
      // and hoisted; only its .data() base pointer changes per tile (set inside the loop AFTER
      // producer_acquire, using the ACQUIRED producer stage epi_prod.index()). With AccStages=1 this
      // always resolves to buf=0 → identical to the old fixed-offset behavior.
      Tensor accB = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));  // (MMA,MMA_M,MMA_N)
      Tensor acc_gate = accB;
      Tensor acc_up = accB;

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
      Tensor sWg = make_tensor(make_smem_ptr(ss.tensors.smem_w1_gate.begin()), SmemLayoutW1{});
      Tensor sWu = make_tensor(make_smem_ptr(ss.tensors.smem_w1_up.begin()), SmemLayoutW1{});
      Tensor tCrX = TiledMma::make_fragment_A(sX);    // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tCrWg = TiledMma::make_fragment_B(sWg);  // (MMA,MMA_N,MMA_K,PIPE)
      Tensor tCrWu = TiledMma::make_fragment_B(sWu);

      // PERSISTENT grid-stride over the SAME tile sequence the Load/Epilogue warps walk (identical
      // cluster_id + loop bounds → identical tile count → both CTAs reach the post-loop cluster_sync).
      // load_cons / epi_prod advance CONTINUOUSLY across tiles (NOT re-init per tile).  The accumulator
      // TMEM region (allocated ONCE above) is REUSED every tile: the K-loop's first k_block writes it
      // with ScaleOut::Zero (OVERWRITE, not accumulate), so there is no cross-tile contamination.
      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        // 2-SM: only the leader CTA touches the acc pipe (producer_acquire/commit), issues the tcgen05
        // MMA, and consumes/releases the load pipe (its umma_arrive multicasts to BOTH CTAs). The PEER
        // CTA must do NONE of these (stock sm100_gemm_tma_warpspecialized.hpp:761-771). CRITICAL for the
        // PERSISTENT loop: the epi empty-barrier arrivals from BOTH epilogues redirect to the LEADER's
        // barrier (Sm100MmaPeerBitMask), so the follower's own empty barrier never advances — a follower
        // producer_acquire would pass tile-0 (initial state) then DEADLOCK on tile-1. Gate it to leader.
        if (is_mma_leader_cta) {
          // Acquire the producer acc stage. With kAccStages=2 this waits for the EMPTY of stage
          // epi_prod.index() — released by the epilogue TWO tiles ago (tile N-2 for the current tile N),
          // NOT the immediately-preceding tile. So MMA(tile N) may run while the epilogue still reads
          // tile (N-1)'s result from the OTHER buffer → overlap. With kAccStages=1 the producer and
          // consumer share the single stage, so this acquire still serializes on tile (N-1)'s release
          // (the original 1-stage behavior — no acc clobber).
          pipeline_epi.producer_acquire(epi_prod);
          // Select THIS tile's acc buffer = the ACQUIRED producer stage's TMEM window. epi_prod.index()
          // is valid AFTER producer_acquire (it names the stage just claimed). gate at buf, up at
          // buf+kEpiTileN. Only .data() changes; the fragment layout is fixed. (AccStages=1 → buf=0.)
          const uint32_t mma_buf = epi_prod.index() * uint32_t(kAccBufStride);
          acc_gate.data() = ss.tmem_base_ptr + mma_buf;
          acc_up.data() = ss.tmem_base_ptr + mma_buf + uint32_t(kEpiTileN);
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;  // first k_block zeroes both accs THIS tile
          for (int k = 0; k < k_tile_count; ++k) {
            pipeline_load.consumer_wait(load_cons);
            int rs = load_cons.index();
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < size<2>(tCrX); ++kb) {
              // Both MMAs are INDEPENDENT and share tCrX; the accumulate_ flag is shared, so we zero on
              // the very first k_block (k==0 && kb==0) and set One thereafter for BOTH accs.
              cute::gemm(tiled_mma, tCrX(_, _, kb, rs), tCrWg(_, _, kb, rs), acc_gate);
#ifndef SWIGLU_SINGLE_ACC
              cute::gemm(tiled_mma, tCrX(_, _, kb, rs), tCrWu(_, _, kb, rs), acc_up);
#endif
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            pipeline_load.consumer_release(load_cons);
            ++load_cons;
          }
          // Signal epilogue: both accs complete. The PipelineUmmaAsync commit inserts the tcgen05
          // commit (multicast to both CTAs' full barriers) so the epilogue's TMEM loads are ordered.
          pipeline_epi.producer_commit(epi_prod);
          ++epi_prod;  // leader-only: advance the producer state only where we acquire/commit
        }
      }
      // (TMEM release_lock + free happen AFTER the cluster_sync() below, issued by this same MMA warp.)
    }

    // =========================================================================================
    // EPILOGUE warps (wg1) — wait, TMEM-load both accs → silu(gate)*up (fp32) → bf16 → global store.
    // TMEM→reg mirrors FMHA correction:795-863; we then write directly to global A via the identity
    // coordinate tensor (no TMA store — simplest correct path for a single tile).
    // =========================================================================================
    else if (role == kEpilogue) {
#ifndef EPI_REGS
#define EPI_REGS 160
#endif
      cutlass::arch::warpgroup_reg_alloc<
          EPI_REGS>();  // tunable (1 CTA/SM ⇒ huge reg headroom; bump only helps if epilogue spills)

      // TMEM is published ONCE (before the persistent loop); wait here once.  All tile-invariant epilogue
      // setup (MMA layout, TMEM accumulator views, coordinate tile, TMEM-load tiling, register tensors, sA
      // staging + TMA-store partition mA_tma/bSG_sA) is hoisted out of the loop below — only
      // cta_row_offset/cta_col_offset, the per-tile pipeline handshake (consumer_wait/release), and the
      // per-tile TMA-store box coordinate (local_tile → bSG_gA) are per-tile.
      tmem_alloc_bar.arrive_and_wait();  // wait for MMA to publish ss.tmem_base_ptr (#3)

      TiledMma tiled_mma;
      // partition_fragment_C(tiled_mma, (256,128)) returns the PER-CTA accumulator fragment, NOT a
      // full 256-row one: for the SM100 2-SM UMMA the M-split lives in the TMEM *value* layout (one
      // CTA owns 128 of the 256 rows via its own tmem_base_ptr), so the fragment's M-mode is per-SM
      // (cute::mma_atom.hpp partition_shape_C:563 slices thrfrg_C at thread 0 → per-CTA value extent;
      // mma_traits_sm100.hpp tmem_frg::make:478 "M_MMA_SM will be 64"). Hence the +128 follower row
      // offset must be injected EXPLICITLY through the global coordinate tile (the fragment carries no
      // CTA identity), exactly as the stock collective does via the per-CTA m_coord
      // (sm100_epilogue_array_tma_warpspecialized.hpp:740-742, fed CtaShape_MNK + per-CTA cta_coord
      // from sm100_gemm_tma_warpspecialized.hpp:841-843; the per-CTA M offset is added in
      // sm100_tile_scheduler.hpp:800 `new_cta_coord_m += cta_in_cluster_offset_m`).
      Tensor tAcc = partition_fragment_C(
          tiled_mma, take<0, 2>(TileShape{}));  // (MMA,MMA_M,MMA_N) tmem (per-CTA)

      // Acc TMEM views: FRAGMENT layout is buffer-INDEPENDENT and hoisted; the .data() base pointer is set
      // PER TILE to the CONSUMER stage's buffer (epi_cons.index()*kAccBufStride), AFTER consumer_wait and
      // BEFORE partition_S/the TMEM-load (see loop). The initial .data() here (buf 0) is ONLY a placeholder
      // so make_tmem_copy below has a well-formed tensor to read the (buffer-independent) LAYOUT from; the
      // actual source pointer used by every copy comes from the per-tile partition_S below. (AccStages=1 →
      // the per-tile buf is always 0, identical to the old fixed-offset path.)
      Tensor tAcc_gate = tAcc;
      tAcc_gate.data() = ss.tmem_base_ptr + 0u;
      Tensor tAcc_up = tAcc;
      tAcc_up.data() = ss.tmem_base_ptr + uint32_t(kEpiTileN);

      // GLOBAL (M,N) identity coordinate tensor partitioned through the IDENTICAL MMA C-path that
      // produced the data fragment tAcc, sliced at THIS CTA. Rationale (rank + offset, both load-bearing):
      //   * tAcc = partition_fragment_C(mma,(256,128)) is rank-3 (MMA,MMA_M,MMA_N) (mma_atom.hpp
      //     partition_shape_C:563 → shape(dummy_v)). make_tmem_copy therefore builds Tiler_MN of rank-3
      //     (copy_atom.hpp make_cotiled_copy:553 tiles per mode of product_each(shape(tAcc))), so
      //     tidfrg_D asserts rank(dtensor) >= 3 (copy_atom.hpp:244). The OLD rank-2 (CTA_M,CTA_N)
      //     cAcc_cta violated this → "Rank of tensor to be partitioned too small."
      //   * cta_mma.partition_C(identity) returns rank-3 (MMA,MMA_M,MMA_N) with the SAME extents as tAcc
      //     (make_fragment_C:142 == shape(partition_C)), so partition_D is rank-legal AND tTMc ends up
      //     element-for-element congruent with partition_S(tAcc_gate) (same Tiler_MN, same thread slice).
      //   * The +128 follower offset enters AUTOMATICALLY: the 2-SM C-layout
      //     CLayout = (_2, (M/2, N)) (mma_traits_sm100.hpp:1700, ThrID=_2) puts the M-half in the V-mode,
      //     and we slice partition_C at this CTA's V = block_rank_in_cluster % AtomThrID = the same slice
      //     the load/mma warps use (line 314). V=0(leader)→rows[0,128); V=1(follower)→rows[128,256).
      //     (The prior compiling-but-wrong version sliced the (256,128) identity at thread/CTA 0 for BOTH
      //     CTAs — no V offset → follower rows never written, leader rows scrambled.)
      // Coordinate slice MUST be get_slice(0) — the SAME slice partition_fragment_C uses internally
      // (make_fragment_C(partition_shape_C(mma,MN)) ≈ thread/V-0), so the coordinate is element-aligned
      // with the DATA fragment tAcc in BOTH CTAs. Using get_slice(block_rank) instead (a per-CTA V-slice)
      // re-orders the coordinate relative to the data → scramble (observed: 2.8% match). The per-CTA
      // fragment is V-independent in structure (both CTAs index element i to the same LOCAL (row,col)),
      // so get_slice(0) gives LOCAL rows [0,128); the +128 follower offset is added explicitly in the
      // store loop below (NOT via the coordinate slice).
      ThrMMA cta_mma_epi = tiled_mma.get_slice(0);
      Tensor cAcc = make_identity_tensor(
          take<0, 2>(TileShape{}));  // (256,128) coords; V-0 → local rows[0,128)
      Tensor cAcc_cta =
          cta_mma_epi.partition_C(cAcc);  // (MMA,MMA_M,MMA_N) local coords (aligned w/ data)

      // TMEM load atom: 4 warps × 32 lanes, 128 cols of 32b (FMHA mainloop:539).
      using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;
      int thread_idx = threadIdx.x % (NumEpiWarps * cutlass::NumThreadsPerWarp);

      // make_tmem_copy / thr_tmem_load are BUFFER-INDEPENDENT (they capture only the copy LAYOUT and the
      // per-thread slice, NOT the source pointer), so they are built ONCE here and reused for every tile,
      // even when AccStages=2 alternates the source buffer.
      auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tAcc_gate);
      auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

      // NOTE: partition_S(tAcc_gate)/partition_S(tAcc_up) — which build the TMEM-source tensors tTMgate/
      // tTMup — are deliberately NOT hoisted here. partition_S CAPTURES the tensor's ITERATOR (data ptr +
      // layout) at call time, so a tensor partitioned with the hoisted (buf-0) .data() would keep reading
      // buffer 0 even after the per-tile .data() update. With AccStages=2 that would make the epilogue read
      // the WRONG buffer. Therefore partition_S is REDONE INSIDE the loop, AFTER tAcc_gate.data()/
      // tAcc_up.data() are set to the consumer stage's buffer (epi_cons.index()*kAccBufStride). (The
      // tiled_tmem_load copy object above carries the buffer-independent layout, so re-partitioning the
      // source is sufficient — the copy itself is NOT rebuilt.)
      // DATA is partition_S (TMEM source ordering); the COORDINATE must be partition_D, NOT partition_S.
      // Reason (this is the STOCK pattern; deviating from it caused every prior scramble):
      //   * The T2R copy `copy(tiled_tmem_load, tTMgate, rGate)` delivers each TMEM value into the
      //     register tensor in the copy atom's DST value ordering. For SM100_TMEM_LOAD_32dp32b32x the
      //     Src and Dst value layouts DIFFER (copy_traits_sm100.hpp:1633-1636 SrcLayout (32,32768):(0,1)
      //     vs DstLayout (32,1024):(1024,1)); RefLayout==SrcLayout, so tidfrg_S yields the Src ordering
      //     and tidfrg_D yields the Dst ordering (copy_atom.hpp:226,247). rGate is built from
      //     shape(tTMc) and read by linear index i in the loop, so element i of rGate sits in the DST
      //     ordering — hence tTMc(i) must ALSO be in the DST ordering, i.e. partition_D(coord).
      //   * This is exactly the proven FMHA mainloop pattern this kernel mirrors:
      //     tTMEM_LOADtS = partition_S(tStS) (DATA), tTMEM_LOADcS = partition_D(tScS) (COORD),
      //     rS = make_tensor(shape(tTMEM_LOADcS)), copy(load, tS, rS), then apply_mask uses rS(i)/cS(i)
      //     element-aligned (sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:548-549,568-569,572;
      //     fmha_fusion.hpp:126-130). Note FMHA's tScS is itself get_slice(_).partition_C(cS):524 — i.e.
      //     partition_D(partition_C(identity)) IS the canonical, element-correct composition, not a
      //     "double permutation".
      //   * Rank-legal: cAcc_cta is rank-3 (MMA,MMA_M,MMA_N) >= rank-2 Tiler_MN, so tidfrg_D's
      //     `rank(dtensor) >= rank(Tiler_MN)` assert passes (copy_atom.hpp:244).
      //   * partition_S of the coord (the previous attempt) compiles for the coord itself but forces
      //     rGate into the Src value count/ordering, which then fails the register-vectorize assert
      //     `size(rD)==RegNumDst` in the DATA copy_unpack (copy_traits_sm100.hpp:396) — that is the
      //     reported compile error; partition_D sizes rGate to exactly NumValDst=1024 so it vectorizes.
      //   * The +128 follower offset still enters via the V-slice baked into cAcc_cta: the 2-SM
      //     CLayout (_2,(M/2,N)):(M/2,(1,M)) (mma_traits_sm100.hpp:1700) puts the M-half in V with
      //     stride M/2=128; partition_C sliced at this CTA's V (line 461) selects V=0→rows[0,128) /
      //     V=1→rows[128,256). partition_D only RE-ORDERS those coordinate VALUES into the Dst register
      //     ordering — it does not drop or alter the +128 offset.
      Tensor tTMc =
          thr_tmem_load.partition_D(cAcc_cta);  // (T2R,T2R_M,T2R_N) global (row,col), DST order

      // Per-tile register staging (tile-invariant SHAPE; values overwritten each tile by the TMEM copies).
      Tensor rGate = make_tensor<ElementAcc>(shape(tTMc));
      Tensor rUp = make_tensor<ElementAcc>(shape(tTMc));

      // ---- TMA-STORE EPILOGUE (stage in SMEM, then async TMA bulk-tensor store SMEM→global) ----------
      // The result tile is computed into registers rA, scattered into a per-CTA SMEM staging tile sA at
      // the LOCAL (row,col) tTMc(i) gives, then DRAINED to global A by a SINGLE async TMA bulk-tensor
      // store (cp.async.bulk.tensor) per CTA per tile.  TMA bypasses the L1/TEX path (the old manual
      // 128-thread coalesced sA→global loop went through the LSU and was the 87% L1/TEX bottleneck) and
      // frees the epilogue threads for the next tile's TMEM-load+silu while the store drains.  Numerics
      // are IDENTICAL: every element lands at the same global (row,col)=(local_row+cta_row_offset,
      // local_col+cta_col_offset) with the same silu(gate)*up value; sA is only an intermediate.
      Tensor sA = make_tensor(make_smem_ptr(ss.tensors.smem_out.begin()),
                              SmemLayoutA{});  // (kEpiTileM,kEpiTileN) row-major
      cutlass::epilogue::thread::SiLu<ElementAcc> silu{};

      // TMA-STORE partition (HOISTED — buffer/coord-independent).  mA_tma is the descriptor's identity
      // coord-tensor over the WHOLE A[M,N] (congruent with the host stride (N,1)); per-tile we local_tile
      // it at the box coordinate.  thrblk_s2g.partition_S(sA) is the smem source (sA pointer is fixed →
      // hoistable); thrblk_s2g.partition_D(gA) is the gmem dst (gA changes per tile → built in the loop).
      // The TMA bulk-store is issued by ONE elected thread (the partition from get_slice(Int<0>{}) is the
      // full box assigned to logical thread 0), exactly as the stock SM100/SM90 TMA epilogue does.
      Tensor mA_tma = params.tma_store_a.get_tma_tensor(make_shape(M, N));  // (M,N) identity coords
      ThrCopy thrblk_s2g = params.tma_store_a.get_slice(Int<0>{});
      Tensor bSG_sA = thrblk_s2g.partition_S(sA);  // (TMA,TMA_M,TMA_N)
      // ONE issuing thread per CTA = elected lane of epi-warp 4 (lane_predicate = cute::elect_one_sync()
      // computed once at entry, valid per-warp). All store-issuing ops (copy/arrive/wait) are gated on it.
      const bool is_tma_store_lane = (warp_idx == 4) && (lane_predicate != 0u);

      // PERSISTENT grid-stride over the SAME tile sequence the Load/MMA warps walk (identical cluster_id
      // + bounds → identical tile count → all 128 epilogue threads loop the same number of times and the
      // post-loop cluster_sync is reached in lockstep).  epi_cons advances CONTINUOUSLY across tiles (NOT
      // re-init per tile).  Each tile: consumer_wait (waits for THIS tile's MMA commit) → TMEM-load →
      // silu·mul → WAR (drain PREVIOUS tile's TMA store) → sA scatter → consumer_release (frees the acc
      // slot so the NEXT tile's MMA may overwrite TMEM) → fence+epi_smem_bar → async TMA bulk-store sA→A.
      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;  // GLOBAL token m-tile, across ALL experts
        const int n_tile_local =
            tile % params.n_local_tiles;  // OUTPUT column tile, ∈ [0, nI_per_expert)
        // Global row offset = this tile's M-base (m_tile * kTileM) + the per-CTA M-block within a 2-SM
        // cluster (block_rank * kTileM/AtomThrID; 0 for 1-SM). Column offset = this tile's N-base.
        const int cta_row_offset =
            m_tile * kTileM +
            int(block_rank_in_cluster) * (kTileM / int(size(typename TiledMma::AtomThrID{})));
        // Output A is [G*Me, I]: col ∈ [0, I).  The expert's row-block is selected by the GLOBAL m_tile
        // (cta_row_offset = m_tile*kTileM, already global), so the column only needs the LOCAL n-tile.
        const int cta_col_offset = n_tile_local * kTileN;

        // Wait for THIS tile's MMA commit, then order the UMMA TMEM writes before the TMEM loads.
        pipeline_epi.consumer_wait(epi_cons);
        cutlass::arch::
            fence_view_async_tmem_store();  // make UMMA TMEM writes visible to TMEM loads

        // Point the acc views at the CONSUMER stage's buffer (the stage epi_cons just observed full), then
        // RE-PARTITION the TMEM-source tensors. partition_S captures (.data()+layout) at call time, so it
        // MUST be redone here AFTER updating .data() — otherwise tTMgate/tTMup would read the stale (buf-0)
        // pointer and, with AccStages=2, the WRONG buffer. The copy layout (tiled_tmem_load) is unchanged.
        // (AccStages=1 → epi_buf is always 0, so this reproduces the original fixed-offset reads.)
        const uint32_t epi_buf = epi_cons.index() * uint32_t(kAccBufStride);
        tAcc_gate.data() = ss.tmem_base_ptr + epi_buf;
        tAcc_up.data() = ss.tmem_base_ptr + epi_buf + uint32_t(kEpiTileN);
        Tensor tTMgate =
            thr_tmem_load.partition_S(tAcc_gate);           // (T2R,T2R_M,T2R_N) tmem source (gate)
        Tensor tTMup = thr_tmem_load.partition_S(tAcc_up);  // (T2R,T2R_M,T2R_N) tmem source (up)

        copy(tiled_tmem_load, tTMgate, rGate);
        copy(tiled_tmem_load, tTMup, rUp);

#ifdef SWIGLU_DEBUG_PRINT
        // Guard to the FIRST tile only (tile == cluster_id) so the persistent loop doesn't spam.
        if (tile == cluster_id && thread_idx == 0 && block_rank_in_cluster == 0) {
          for (int i = 0; i < size(rGate) && i < 24; ++i) {
            printf("T0 i=%2d row=%3d col=%3d gate=% .4f\n", i, get<0>(tTMc(i)) + cta_row_offset,
                   get<1>(tTMc(i)), float(rGate(i)));
          }
        }
#endif

        // WAR (cross-tile sA reuse): sA (ss.tensors.smem_out) is a SINGLE staging buffer reused every
        // tile.  The PREVIOUS tile's TMA bulk-store reads sA asynchronously; before we OVERWRITE sA with
        // THIS tile's scatter, that store MUST have drained.  tma_store_wait<0>() drains it on the issuing
        // thread (it tracks the pending cp.async.bulk commit group); the NamedBarrier then makes ALL 128
        // epilogue threads observe the drain before any begins the scatter.  Placed HERE (after this tile's
        // TMEM-load+silu, before its scatter) so the PREVIOUS store overlapped this tile's TMEM-load+silu
        // (the perf win).  On the FIRST iteration nothing is pending → tma_store_wait<0>() is a no-op.
        // (Replaces the old epi_done_bar; same NumEpiThreads, distinct hw barrier id 1u.)
        if (is_tma_store_lane) {
          cute::tma_store_wait<0>();
        }
        cutlass::arch::NamedBarrier epi_war_bar(NumEpiThreads, /*id=*/1u);
        epi_war_bar.arrive_and_wait();

        // (1) compute rA, (2) scatter into SMEM at the LOCAL (row,col).  No row<M/col<N guard here: the
        // LOCAL (row,col) is always in [0,kEpiTileM)×[0,kEpiTileN) by construction; OOB global rows/cols
        // are clamped by the TMA-store descriptor's box (it never writes past the (M,N) extent).
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rGate); ++i) {
#if defined(SWIGLU_DEBUG_RAW_GATE)
          ElementOut rA =
              static_cast<ElementOut>(rGate(i));  // ISOLATION: raw gate acc (= X·W1[0:I]ᵀ)
#elif defined(SWIGLU_DEBUG_RAW_UP)
          ElementOut rA = static_cast<ElementOut>(rUp(i));  // ISOLATION: raw up acc (= X·W1[I:2I]ᵀ)
#else
          ElementOut rA = static_cast<ElementOut>(silu(rGate(i)) * rUp(i));
#endif
          int lrow = get<0>(tTMc(i));  // LOCAL row ∈ [0,kEpiTileM)
          int lcol = get<1>(tTMc(i));  // LOCAL col ∈ [0,kEpiTileN)
          sA(lrow, lcol) = rA;
        }

        // TMEM read is done (rGate/rUp already in registers; sA staging touches no TMEM).
        // fence_view_async_tmem_load orders these last TMEM loads; consumer_release returns THIS tile's
        // acc->epi stage (epi_cons.index()) so the MMA warp may reuse/free that buffer. With kAccStages=2
        // this release frees the stage that the MMA's producer_acquire for tile (current+2) will wait on,
        // so MMA(tile N+1) — which uses the OTHER stage — is NOT gated by this release and runs concurrently
        // with this epilogue. With kAccStages=1 producer and consumer share the one stage, so this release
        // is exactly what lets the NEXT tile's MMA producer_acquire proceed (the old serialized gating).
        cutlass::arch::fence_view_async_tmem_load();
        pipeline_epi.consumer_release(epi_cons);
        ++epi_cons;

        // Make all of sA visible across the 128 epilogue threads, AND make those generic (LSU) smem
        // writes visible to the async TMA proxy, BEFORE issuing the TMA bulk-store.
        //  - fence_view_async_shared() (== fence.proxy.async.shared::cta) is issued by EVERY epi thread so
        //    its own sA writes become visible to the TMA (async) proxy.  This is the store-side analogue
        //    of the load's cp.async fence; the stock SM90/SM100 TMA epilogue does exactly this before the
        //    TMA store (sm100_epilogue_tma_warpspecialized.hpp:768).
        //  - epi_smem_bar (NamedBarrier over EXACTLY the 128 epi threads — NOT __syncthreads(), which would
        //    deadlock on the Load/MMA/empty warps that never enter this branch) then cross-orders ALL 128
        //    threads' sA writes + fences before the single issuing thread reads sA via TMA.  All 128 reach
        //    it every iteration (no divergent exit precedes it); distinct hw barrier id 0u (vs WAR id 1u).
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier epi_smem_bar(NumEpiThreads, /*id=*/0u);
        epi_smem_bar.arrive_and_wait();

        // (3) ASYNC TMA BULK-TENSOR STORE: ONE elected thread per CTA drains sA → global A, bypassing the
        // L1/TEX/LSU path.  The box coordinate reuses the manual loop's EXACT per-tile/per-CTA offset math:
        //   cta_row_offset = m_tile*kTileM + block_rank*kEpiTileM  (kTileM = AtomThrID * kEpiTileM, so this
        //   is an exact multiple of kEpiTileM) → box-row index = cta_row_offset / kEpiTileM.  Each CTA of a
        //   2-SM cluster issues its OWN store for its kEpiTileM-row slice (NOT leader-gated; the epilogue
        //   runs on both CTAs and block_rank already encodes which CTA).  cta_col_offset = n_tile_local*kTileN
        //   (== n_tile_local*kEpiTileN since kTileN==kEpiTileN) → box-col index = cta_col_offset / kEpiTileN.
        // local_tile selects gA = A[ box_m*kEpiTileM : +kEpiTileM, box_n*kEpiTileN : +kEpiTileN ]; the TMA
        // descriptor (built over the TRUE (M,N) extent) clamps any partial tile (OOB rows/cols never written).
        // tma_store_arrive() commits the cp.async.bulk group; the WAR tma_store_wait<0>() at the TOP of the
        // NEXT iteration drains it before sA is overwritten — so THIS store overlaps the next tile's
        // TMEM-load+silu compute.  (For the tested shapes M=16384%256==0, I=512%64==0 every tile is FULL.)
        if (is_tma_store_lane) {
          const int box_m = cta_row_offset / kEpiTileM;  // exact (kTileM = AtomThrID*kEpiTileM)
          const int box_n = cta_col_offset / kEpiTileN;  // exact (kTileN == kEpiTileN)
          Tensor gA = local_tile(mA_tma, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}),
                                 make_coord(box_m, box_n));  // (kEpiTileM, kEpiTileN)
          Tensor bSG_gA = thrblk_s2g.partition_D(gA);        // (TMA,TMA_M,TMA_N)
          copy(params.tma_store_a, bSG_sA, bSG_gA);
          cute::tma_store_arrive();
        }
      }
      // Drain the LAST tile's TMA store before the cluster_sync()/TMEM-free below (no further WAR wait will
      // run for it). Only the issuing thread has a pending commit group; harmless no-op on the others.
      if (is_tma_store_lane) {
        cute::tma_store_wait<0>();
      }
    } else {
      cutlass::arch::warpgroup_reg_dealloc<40>();
    }

    // Cross-CTA + cross-warp TMEM-free barrier: every warp of BOTH CTAs reaches here AFTER finishing its
    // role (load drained, MMA committed, epilogue done reading TMEM, empty warps idle). cluster_sync()
    // therefore guarantees the epilogue is done with TMEM AND both CTAs' warp-1 are aligned, so the
    // cta_group::2 release_lock + dealloc that follow (issued by the SAME warp that allocated = the MMA
    // warp, in BOTH CTAs) are race-free. This replaces the intra-CTA tmem_free_bar + the missing stock
    // peer-CTA dealloc ClusterBarrier (sm100_gemm_tma_warpspecialized.hpp:783,790-803) in one barrier.
    cute::cluster_sync();
    if (role == kMMA) {
      tmem_allocator.release_allocation_lock();
      tmem_allocator.free(ss.tmem_base_ptr, kTmemCols);
    }
#endif
  }
};

// ============================================================================================
// §2.4 · GROUPED (contiguous, uniform Mₑ) host launcher — SINGLE LAUNCH for G experts.
//        Standard stacked MoE layout, all tensors contiguous:
//          X  : [G*Me, d]   (expert e owns rows [e*Me, (e+1)*Me))
//          W1 : [G*2I, d]   (expert e: gate rows [e*2I, e*2I+I), up rows [e*2I+I, e*2I+2I))
//          A  : [G*Me, I]   (expert e owns rows [e*Me, (e+1)*Me))
//        The GEMM is BLOCK-DIAGONAL; the expert is DERIVED FROM THE M-TILE in-kernel, so we build
//        just TWO TMA descriptors via ONE to_underlying_arguments call:
//          A-operand (X) over (G*Me, d)   and   B-operand (W1) over (G*2I, d).
//        There is NO separate up descriptor — up is the (gate_w1_ntile + nI_per_expert) N-tile slice
//        of the same W1 descriptor (the grouped generalization of the single-expert "up = W1+I*d").
//        CONSTRAINT: uniform Me only (assumes Me % kTileM == 0 and I % kTileN == 0; no boundary
//        predication).  Non-uniform Me would need per-expert cumulative-offset (prefix-sum) arrays.
// ============================================================================================
// The tuning params default to the SwiGluConfig defaults, so the existing call
// LaunchSwiGluGrouped<Element, ElementOut>(...) resolves to the current (default-tuned) behavior.
template <typename Element, typename ElementOut, int TileM_ = 256, int TileN_ = 64, int TileK_ = 16,
          int kStages_ = 16, int ClusterM_ = 2, int MinBlocks_ = 1, int AccStages_ = 2>
cudaError_t LaunchSwiGluGrouped(
    const Element* X,   // [G*Me, d] row-major
    const Element* W1,  // [G*2I, d] row-major (per-expert gate||up)
    ElementOut* A,      // [G*Me, I] row-major
    int G, int Me, int I, int d, cudaStream_t stream, int device = 0, int sm_count = 0,
    // VARLEN-M (uneven, TileM-aligned experts): device array [M_varlen/kTileM]
    // mapping each global m-tile → expert id, and the total packed token count
    // M_varlen.  nullptr => UNIFORM (M = G*Me).  Experts must be TileM-aligned.
    const int* d_m_tile_expert = nullptr, int M_varlen = 0) {
  using Config = SwiGluConfig<Element, ElementOut, TileM_, TileN_, TileK_, kStages_, ClusterM_,
                              MinBlocks_, AccStages_>;
  using Kernel = Sm100SwiGluKernel<Config>;
  using CollectiveMma = typename Config::CollectiveMma;
  using StrideX = typename Config::StrideX;
  using StrideW1 = typename Config::StrideW1;
  using StrideA = typename Config::StrideA;

  // VARLEN-M: M = total packed (TileM-aligned) tokens across uneven experts; else uniform G*Me.
  const int M = (d_m_tile_expert != nullptr) ? M_varlen : (G * Me);  // total tokens (problem M)
  const int W1N = G * 2 * I;  // total W1 rows (problem N, B-operand) — I uniform across experts

  // Strides. make_cute_packed_stride wants {extent..., L}. X is RowMajor (G*Me,d) → A K-major (d,1,0).
  // W1 is K-contiguous [G*2I,d] → B K-major: LayoutW1=ColumnMajor ⇒ StrideB=Stride<int,_1,int>,
  // make_cute_packed_stride sets N-stride=d ⇒ dW1=(d,1,0), i.e. W1[n,k] at n*d+k (the physical layout).
  StrideX dX = cutlass::make_cute_packed_stride(StrideX{}, cute::make_shape(M, d, 1));
  StrideW1 dW1 = cutlass::make_cute_packed_stride(StrideW1{}, cute::make_shape(W1N, d, 1));
  StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, I, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = sm_count;

  // Problem shape (M=G*Me, N=G*2I, K=d): one to_underlying_arguments builds BOTH the A-TMA over the
  // full token stack (M=G*Me) AND the B-TMA over the full W1 stack (N=G*2I) — no up descriptor.
  auto problem = cute::make_shape(M, W1N, d);

  typename CollectiveMma::Arguments mainloop_args{};
  mainloop_args.ptr_A = X;
  mainloop_args.dA = dX;
  mainloop_args.ptr_B = W1;
  mainloop_args.dB = dW1;
  typename CollectiveMma::Params mainloop_params = CollectiveMma::to_underlying_arguments(
      problem, mainloop_args, /*workspace=*/nullptr, hw_info);

  // TMA-STORE descriptor for the fused output A[M,I] (row-major, single contiguous tensor, expert
  // implicit in the row range — mirrors how ONE W1 load descriptor covers all experts). Built ONCE
  // over the TRUE (M,I) extent so the descriptor's box clamping handles any partial tile (OOB) for
  // free; per-tile/per-CTA placement is just the box coordinate (local_tile by (kEpiTileM,kEpiTileN)).
  // Stride (I,1) = row-major; sA smem layout = Config::SmemLayoutA (kEpiTileM,kEpiTileN) row-major, the
  // SAME layout ss.tensors.smem_out uses → make_tma_copy builds a no-swizzle box (kEpiTileM,kEpiTileN).
  cute::Tensor mA_store = cute::make_tensor(
      cute::make_gmem_ptr(A),
      cute::make_layout(cute::make_shape(M, I), cute::make_stride(I, cute::_1{})));
  auto tma_store_a =
      make_tma_copy(cute::SM90_TMA_STORE{}, mA_store, typename Config::SmemLayoutA{});

  typename Kernel::Params params;
  params.tma_load_x = mainloop_params.tma_load_a;
  params.tma_load_w1 = mainloop_params.tma_load_b;
  params.tma_store_a = tma_store_a;
  params.ptr_A = A;
  params.dA = dA;
  params.M = M;  // G*Me
  params.N = I;  // output width
  params.K = d;
  params.Me = Me;    // tokens per expert (uniform fallback; unused when m_tile_expert != nullptr)
  params.W1N = W1N;  // G*2I
  params.m_tile_expert = d_m_tile_expert;  // VARLEN-M expert table (nullptr => uniform fast path)

  // Logical tile grid: m_tile spans ALL experts' tokens (num_m_tiles = G*Me/kTileM); n_tile_local spans
  // ONE expert's output width (I/kTileN).  total_tiles = num_m_tiles * num_n_local_tiles is the LOGICAL
  // tile count (NOT multiplied by cluster.x).  The kernel decodes m_tile = tile / num_n_local_tiles,
  // n_tile_local = tile % num_n_local_tiles from a linear tile id (see Sm100SwiGluKernel::operator()).
  constexpr int kTileM = cute::size<0>(typename Config::TileShape{});
  constexpr int kTileN = cute::size<1>(typename Config::TileShape{});
  int num_m_tiles = (M + kTileM - 1) / kTileM;
  int num_n_local_tiles = (I + kTileN - 1) / kTileN;
  int total_tiles = num_m_tiles * num_n_local_tiles;
  dim3 cluster(cute::size<0>(typename Config::ClusterShape{}),
               cute::size<1>(typename Config::ClusterShape{}),
               cute::size<2>(typename Config::ClusterShape{}));

  // Kernel resource info — defined BEFORE the grid so the occupancy query below sees the real smem
  // opt-in and cluster attribute (cudaOccupancyMaxActiveClusters reads them from the launch config).
  dim3 block(Kernel::MaxThreadsPerBlock, 1, 1);
  int smem_size = Kernel::SharedStorageSize;
  void const* kernel_ptr = reinterpret_cast<void const*>(cutlass::device_kernel<Kernel>);

  if (smem_size >= (48 << 10)) {
    cudaError_t attr =
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr != cudaSuccess) return attr;
  }
  // Allow non-portable cluster size (matches ClusterLauncher::init).  Set BEFORE the occupancy query.
  cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // PERSISTENT launch (occupancy-driven, SINGLE wave — mirrors CUTLASS StaticPersistentTileScheduler /
  // KernelHardwareInfo::query_device_max_active_clusters and MegaMoE's grid = num_sms).  Each cluster
  // grid-strides over a subset of the total_tiles logical tiles, amortizing the per-cluster setup (TMEM
  // alloc/free, cluster_sync) over many tiles.  We launch EXACTLY the number of clusters that actually
  // co-reside on the device for THIS kernel — NOT a guessed multiple of the SM count: this is a heavy
  // (kStages=16, large-smem, TMEM) kernel that packs ~1 CTA/SM, so over-launching merely queues a 2nd
  // wave and re-pays the setup the persistent design exists to amortize (ncu confirmed achieved occupancy
  // = 12.5% = 1 cluster/SM-pair; the 2-SM cluster co-residency is a HW cap, not raised by more blocks).
  // cudaOccupancyMaxActiveClusters accounts for smem/reg + 2-SM cluster gang-scheduling (TMEM is runtime-
  // allocated and not counted, but smem is the binding limiter here, so the count is correct).
  // cluster_size = ClusterShape.M (= cluster.x = 2 for 2-SM, 1 for 1-SM) is the CTAs per cluster.
  int cluster_size = int(cluster.x);
  int persistent_clusters = 0;
  {
    // Build the occupancy config WITH the real dynamic smem (the bare CUTLASS one-liner helper omits it
    // and would over-count for a large-smem kernel).  grid = one cluster is the "minimum valid grid";
    // cudaOccupancyMaxActiveClusters still returns the DEVICE-WIDE max active cluster count.
    auto occ_cfg = cutlass::ClusterLauncher::make_cluster_launch_config(cluster, cluster, block,
                                                                        smem_size, stream);
    int max_active_clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&max_active_clusters, kernel_ptr, &occ_cfg.launch_config) ==
            cudaSuccess &&
        max_active_clusters > 0) {
      persistent_clusters = max_active_clusters;
    }
  }
  if (persistent_clusters < 1) {
    // Fallback: one cluster per cluster_size SMs (single wave, occupancy-1 — matches MegaMoE grid=num_sms).
    int sm_count_eff = (sm_count > 0)
                           ? sm_count
                           : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    persistent_clusters = sm_count_eff / cluster_size;
  }
  if (persistent_clusters < 1) persistent_clusters = 1;  // always launch at least one cluster
  // Capped at total_tiles so we never launch idle clusters (cluster_id >= total_tiles → zero iterations).
  int num_clusters = persistent_clusters < total_tiles ? persistent_clusters : total_tiles;

  params.total_tiles = total_tiles;
  params.n_local_tiles = num_n_local_tiles;
  params.num_clusters = num_clusters;

  // Grid is 1-D in clusters: cluster.x CTAs per cluster, num_clusters clusters → grid.x = num_clusters *
  // cluster.x.  In-kernel cluster_id = blockIdx.x / cluster.x (both CTAs of a cluster share it).
  dim3 grid(num_clusters * cluster.x, 1, 1);

  auto cfg =
      cutlass::ClusterLauncher::make_cluster_launch_config(grid, cluster, block, smem_size, stream);
  void* kernel_params[] = {&params};
  cudaError_t status = cudaLaunchKernelExC(&cfg.launch_config, kernel_ptr, kernel_params);
  return status;
}

// ============================================================================================
// §2.4b · Single-expert host launcher — now a thin G=1 wrapper over LaunchSwiGluGrouped.
//        X:[M,d], W1:[2I,d] (gate||up), A:[M,I].  With G=1, Me=M, W1N=2I and the grouped scheduler
//        collapses to: e==0, gate_w1_ntile==n_tile_local, up_w1_ntile==n_tile_local + I/kTileN —
//        exactly the original single-expert "up = W1 + I*d" pairing.
// ============================================================================================
// Same defaulted tuning params as LaunchSwiGluGrouped; forwarded through so a single-expert call can be
// tuned identically. The default call LaunchSwiGluSingleExpert<Element, ElementOut>(...) is unchanged.
template <typename Element, typename ElementOut, int TileM_ = 256, int TileN_ = 64, int TileK_ = 16,
          int kStages_ = 16, int ClusterM_ = 2, int MinBlocks_ = 1, int AccStages_ = 2>
cudaError_t LaunchSwiGluSingleExpert(const Element* X,   // [M, d] row-major
                                     const Element* W1,  // [2I, d] row-major (concat: gate||up)
                                     ElementOut* A,      // [M, I] row-major
                                     int M, int I, int d, cudaStream_t stream, int device = 0,
                                     int sm_count = 0) {
  return LaunchSwiGluGrouped<Element, ElementOut, TileM_, TileN_, TileK_, kStages_, ClusterM_,
                             MinBlocks_, AccStages_>(X, W1, A, /*G=*/1, /*Me=*/M, I, d, stream,
                                                     device, sm_count);
}

}  // namespace grouped_gemm_swiglu
}  // namespace transformer_engine
