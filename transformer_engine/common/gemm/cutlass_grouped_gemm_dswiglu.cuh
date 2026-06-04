/***************************************************************************************************
 * B2 — SwiGLU-fused MoE DOWN-proj BACKWARD grouped GEMM (SM100 tcgen05, SINGLE TMEM acc).
 *   Computes dA = dY · W2ᵀ in TMEM (the FC2 dgrad) and, in the epilogue, reads the SAVED SwiGLU input
 *   h = [gate || up] and applies the SwiGLU BACKWARD — h is READ from HBM (NOT recomputed):
 *     grad[m,i] = dA[m,i] · prob[m]                         (grad wrt the (pre-prob) SwiGLU output)
 *     dY1[m, i]      = grad · up · silu'(gate)              (dgate-half of dh, cols [0, I))
 *     dY1[m, I + i]  = grad · silu(gate)                    (dup-half  of dh, cols [I, 2I))
 *   APPROACH A: W2 is fed TRANSPOSED (W2ᵀ [G·I,d], expert-in-N) so the GEMM is STRUCTURALLY IDENTICAL
 *   to the forward up-proj (X·W1ᵀ): A=dY[M,d] RowMajor, B=W2ᵀ[G·I,d] K-contiguous (LayoutB=ColumnMajor).
 *   This REUSES the forward's proven >graph-safe structure and avoids the expert-in-K B-operand crux.
 *   The W2ᵀ transpose is cached on the FC2 weight (id,version) — amortized over grad-accum. The router-
 *   prob gradient dprob = <dA, silu(gate)·up> is computed OUTSIDE this kernel for now (M2b will fuse it
 *   as an in-epilogue colvec-reduce).
 *
 * STATUS: B2 — single-launch GROUPED (multi-expert) device kernel (this file).
 *   §2.3 Sm100DSwiGluKernel  : Load + MMA mirror the forward — ONE A=dY + ONE B=W2ᵀ into ONE dA TMEM
 *           accumulator (SINGLE GEMM, no gate/up split); ONLY the epilogue differs (read saved h →
 *           dswiglu-bwd → 2-wide dY1 store). GROUPED: the expert is DERIVED FROM THE M-TILE (tokens
 *           contiguous by expert), so ONE X TMA desc over [G*Me,d] + ONE W2ᵀ TMA desc over [G·I,d]
 *           suffice (no ptr arrays / tensormap swaps); expert e = the N-rows [e·I, (e+1)·I) of W2ᵀ.
 *   §2.4  LaunchDSwiGluGrouped      : grouped host launcher (2 TMA load descs + 1 TMA store desc over
 *           dY1[M,2I], single cluster-launch).  Inputs: dY[M,d], W2ᵀ[G·I,d], SAVED h[M,2I]; wider
 *           [M,2I] output dY1 = dgate||dup.
 *   SCOPE: contiguous stacked MoE; dY:[G*Me,d], W2ᵀ:[G·I,d], h:[G*Me,2I], dY1:[G*Me,2I]; N=I,K=d.
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
namespace grouped_gemm_dswiglu {

using namespace cute;

using ProblemShapeType = cute::Shape<int, int, int>;  // (M=tokens, N=I, K=d)
using ProblemShape = cutlass::gemm::GroupProblemShape<ProblemShapeType>;

// ============================================================================================
// §2.1 · Type config — reuse the FINAL SM100 2-SM schedule of cutlass_grouped_gemm.cuh. We instantiate
//        ONE CollectiveMma type-provider and use its TiledMma ONCE into a SINGLE dA TMEM accumulator
//        (the GEMM dA = dY·W2ᵀ); the gate/up split is an EPILOGUE concern (output dY1), not the GEMM.
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
struct DSwiGluConfig {
  static_assert(ClusterM_ == 1 || ClusterM_ == 2, "ClusterM_ must be 1 (1-SM) or 2 (2-SM)");

  using Element = Element_;  // X(=dY) and W2ᵀ element (bf16/fp16)
  using ElementAcc = float;  // fp32 accumulate (required)
  using ElementOut = ElementOut_;

  using ArchTag = cutlass::arch::Sm100;
  using OpClass = cutlass::arch::OpClassTensorOp;

  // SINGLE-GEMM B2 (DownProj-bwd dA = dY · W2). APPROACH A — feed W2 TRANSPOSED (W2ᵀ [G·I,d], expert in
  // the N-dim) so the GEMM is STRUCTURALLY IDENTICAL to the forward (X·W1ᵀ): A=dY [M,d] RowMajor
  // (K=d contig), B=W2ᵀ [G·I,d] K-contiguous (LayoutB=ColumnMajor, exactly like the fwd's W1). This
  // REUSES the forward's proven, >graph-safe structure and AVOIDS the expert-in-K B-operand crux. The
  // W2ᵀ transpose is done in Python and cached on the FC2 weight (id,version) — amortized over grad-accum.
  using LayoutX =
      cutlass::layout::RowMajor;  // A operand = dY (M,K)=(tokens,d): RowMajor ⇒ K-contiguous ✓
  using LayoutW2t = cutlass::layout::ColumnMajor;  // B operand = W2ᵀ (N,K)=(I,d): K-contiguous (like W1)
  using LayoutA = cutlass::layout::RowMajor;      // output dA[M,I] (M1a passthrough) / dY1[M,2I] (M2)

  static constexpr int AlignX = 128 / cutlass::sizeof_bits<Element>::value;     // 8 (bf16)
  static constexpr int AlignW2t = 128 / cutlass::sizeof_bits<Element>::value;    // 8
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

  // Single dA accumulator (TileN cols); the old dual gate+up rationale (2 acc in 512-col TMEM) is gone.
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
  // per-stage = X(256×16=8KB)+1×W2ᵀ(64×16=2KB)=10KB → kStages=8 → 80KB « 228KB (room for even more).
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
                "TileN<=256: the single dA accumulator (kAccStages*TileN cols) must leave room for "
                "2-stage double-buffering in 512-col TMEM");

  // Pipeline depth (tunable via kStages_; default 16 = the clean 398-TFLOPS config). TileN=64,TileK=16
  // → ~8KB/stage; the smem static_assert below guards against exceeding the 228KB SM100 capacity.
  static constexpr int kStages = kStages_;

  // Launch-bounds MinBlocksPerMultiprocessor hint, read by the kernel (default 1). The warpgroup
  // reg-reconfig (wg0 dealloc<40> → wg1 alloc<160>) keeps the epilogue at 160 dynamically.
  static constexpr int kMinBlocks = MinBlocks_;
  // TMEM accumulator pipeline depth (default 2 = double-buffered). AccStages_==2 reserves & USES two
  // TMEM buffers (two PIPELINE STAGES of the single dA acc, NOT gate+up) and runs a 2-stage PipelineEpi
  // so MMA(tile N+1) overlaps epilogue(tile N); AccStages_==1
  // collapses to the single-buffer serialized handoff. The per-tile buffer alternation is in operator().
  static constexpr int kAccStages = AccStages_;

  // CollectiveMma used purely as a type-provider (TiledMma / SmemLayout / fragments / TMA atoms /
  // TransactionBytes), exactly like FMHA reuses CollectiveBuilder for CollectiveMmaQK/PV.
  using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, Element, LayoutX, AlignX, Element, LayoutW2t, AlignW2t, ElementAcc, TileShape,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>, KernelSchedule>::CollectiveOp;

  using TiledMma = typename CollectiveMma::TiledMma;
  using AtomThrShapeMNK = typename CollectiveMma::AtomThrShapeMNK;  // (2,1,1) under 2-SM
  using SmemLayoutX = typename CollectiveMma::SmemLayoutA;          // (MMA,M,K,PIPE)
  using SmemLayoutW2t = typename CollectiveMma::SmemLayoutB;         // (MMA,N,K,PIPE)
  static constexpr int Stages = CollectiveMma::DispatchPolicy::Stages;

  // CTA-local tile (after dividing by the 2-SM atom): MMA M is split across 2 CTAs.
  using CtaShapeMNK = typename CollectiveMma::CtaShape_MNK;  // (128,128,64) for 256/2-SM

  // Stride aliases used by the host launcher (single expert, single pointer).
  // X/W2ᵀ strides come from the collective itself (guaranteed compatible with its Arguments/TMA);
  // A (output) stride is just RowMajor (M,N,L) and is only used for the direct global store.
  using StrideX = typename CollectiveMma::StrideA;
  using StrideW2t = typename CollectiveMma::StrideB;
  using StrideA = cutlass::detail::TagToStrideC_t<LayoutA>;

  // The single-tensor collective's TMA atom + Params types (built by to_underlying_arguments).
  using MainloopArguments = typename CollectiveMma::Arguments;  // {ptr_A,dA, ptr_B,dB, ...}
  using MainloopParams = typename CollectiveMma::Params;        // {tma_load_a, tma_load_b, ...}
  using TMA_X = typename MainloopParams::TMA_A;
  using TMA_W2t = typename MainloopParams::TMA_B;

  // ---- F1 GATHER FUSION descriptor type (optional path) --------------------------------------
  // Gather TMA descriptor over the FULL UNPERMUTED X as a 2D [T_src, d] tensor (RowMajor, K=d
  // contiguous).  The gather CopyOp (SM100_TMA_LOAD_MULTICAST_2D_GATHER4 under 2-SM cluster) triggers
  // make_tma_copy's gather-descriptor transform (copy_traits_sm90_tma.hpp:1153-1165): it reduces the
  // gmem column basis to (cols,1) and ×4 the box, so ONE gather4 op moves 4 ROWS × the K-tile width
  // into smem.  The smem layout passed is the SAME per-stage X smem tile the contiguous TMA_X uses —
  // SmemLayoutX{}(_,_,_,0) — so the swizzle baked into the descriptor matches the MMA-fragment layout
  // the mma warp reads.  We materialize the type with a 2D placeholder X[0,0] (RowMajor stride (d,1)=
  // (0,1) here; the runtime descriptor in LaunchDSwiGluGrouped uses the true (d,1) stride).
  //
  // GUARDED by SWIGLU_ENABLE_GATHER_LOAD: the gather-descriptor decltype exercises make_tma_copy's gather
  // path against the 3D swizzled SmemLayoutX, which is the UNVERIFIED-without-compile risk.  To keep the
  // DEFAULT build byte-identical to the validated kernel (and immune to any gather-path type-resolution
  // failure), TMA_X_GATHER aliases the contiguous TMA_X when the macro is OFF.  The Params member, host
  // descriptor, and m_gather_idx are still PLUMBED in both modes; only the gather TMA TYPE + the on-GPU
  // gather4 issue are macro-gated.
#if defined(SWIGLU_ENABLE_GATHER_LOAD)
  using TMA_X_GATHER = decltype(make_tma_copy(
      cute::SM100_TMA_LOAD_MULTICAST_2D_GATHER4{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<Element const*>(nullptr)),
                        cute::make_layout(cute::make_shape(int(0), int(0)),
                                          cute::make_stride(int(0), cute::_1{}))),
      SmemLayoutX{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
      cute::size<0>(ClusterShape{})));
#else
  using TMA_X_GATHER = TMA_X;  // placeholder (gather disabled): keeps Params well-formed, zero new state
#endif

  static constexpr uint32_t TmaTransactionBytes = CollectiveMma::TmaTransactionBytes;

  // ---- TMA-STORE epilogue (sA → global A) ----------------------------------------------------
  // Per-CTA output staging-tile extents (must match Sm100DSwiGluKernel::kEpiTileM/kEpiTileN; defined
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
// §2.3 · Custom grouped warp-specialized kernel (SINGLE GEMM dA = dY · W2ᵀ).
//        Warp roles:  warp 0 = Load (TMA producer), warp 1 = MMA (UMMA + TMEM owner),
//                      warps 4..7 (warpgroup 1) = Epilogue (TMEM→reg→dswiglu-bwd→global store).
//        We mirror the single-tensor collective's load/mma partitioning verbatim: ONE B-TMA (W2ᵀ)
//        and ONE cute::gemm into ONE TMEM accumulator (dA). The dswiglu split into dgate‖dup happens
//        only in the EPILOGUE (output side), not in the GEMM — there is no second B / second accumulator.
// ============================================================================================
template <typename Config>
struct Sm100DSwiGluKernel {
  using Element = typename Config::Element;
  using ElementAcc = typename Config::ElementAcc;
  using ElementOut = typename Config::ElementOut;
  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  using TiledMma = typename Config::TiledMma;
  using AtomThrShapeMNK = typename Config::AtomThrShapeMNK;
  using SmemLayoutX = typename Config::SmemLayoutX;
  using SmemLayoutW2t = typename Config::SmemLayoutW2t;
  using StrideX = typename Config::StrideX;
  using StrideW2t = typename Config::StrideW2t;
  using StrideA = typename Config::StrideA;
  using TMA_X = typename Config::TMA_X;
  using TMA_X_GATHER = typename Config::TMA_X_GATHER;  // F1 gather descriptor (unpermuted X [T_src,d])
  using TMA_W2t = typename Config::TMA_W2t;
  using TMA_A = typename Config::TMA_A;  // TMA-STORE descriptor type over A[M,I] (row-major)
  using SmemLayoutA =
      typename Config::SmemLayoutA;  // sA staging-tile smem layout (kEpiTileM,kEpiTileN)
  static constexpr int Stages = Config::Stages;

  using ArchTag = cutlass::arch::Sm100;
  // TMEM allocator follows the MMA atom: 2-SM (cta_group::2) when AtomThrShapeMNK size==2, else 1-SM.
  static constexpr bool kIs2Sm = (cute::size(AtomThrShapeMNK{}) == 2);
  using TmemAllocator =
      std::conditional_t<kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  // Load(warp0) -> MMA(warp1) : protects X + W2ᵀ-B in smem. AtomThrShapeMNK threads the
  // 2-SM peer mask through the pipeline (sm100_mma_warpspecialized.hpp:153-156).
  using PipelineLoad = cutlass::PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShapeMNK>;
  // MMA(warp1) -> Epilogue(wg1) : protects the single dA TMEM accumulator (single commit after K-loop).
  // Config::kAccStages stages → double-buffered acc pipeline (AccStages=2): the MMA may produce the
  // accumulator for tile (N+1) in buffer (N+1)%kAccStages while the epilogue still reads tile N's result
  // from buffer N%kAccStages → MMA(tile N+1) OVERLAPS epilogue(tile N). AccStages=1 collapses to the old
  // 1-stage serialized handoff (identical behavior). Each stage gets its OWN mbarrier pair; the per-tile
  // acc buffer offset (= stage index * kAccBufStride) is selected from epi_prod.index()/epi_cons.index().
  using PipelineEpi = cutlass::PipelineUmmaAsync<Config::kAccStages, AtomThrShapeMNK>;
  // Cols per acc buffer (one stage) = kEpiTileN (the SINGLE dA accumulator; no gate/up split — that was
  // the forward). The acc buffer for a tile is its acc-pipeline stage index: buf_base = stage_index *
  // kAccBufStride (= stage_index * kEpiTileN; kAccBufStride defined below after kEpiTileN).

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
  // Cols per acc buffer (ONE acc-pipeline stage) = kEpiTileN (the SINGLE dA accumulator; no gate/up).
  // With AccStages>1 each stage occupies its own [stage*kAccBufStride, (stage+1)*kAccBufStride) TMEM
  // window. The MMA writes the producer stage's buffer (epi_prod.index()*kAccBufStride); the epilogue
  // reads the consumer stage's buffer (epi_cons.index()*kAccBufStride). Offsets set PER TILE (operator()).
  // M1a SINGLE accumulator (dA): one stage's buffer = kEpiTileN cols (no up). (was 2*kEpiTileN dual.)
  static constexpr int kAccBufStride = kEpiTileN;  // 64 for TileN=64
  // TMEM columns actually used = kAccStages buffers × kEpiTileN (the single dA acc) = kAccStages*TileN.
  // Allocate ONLY these (not the full 512) so MULTIPLE blocks share the SM's 512-col TMEM → unlocks
  // occupancy (ncu: full-512 alloc capped achieved occ at 1 block/SM despite 2-block theoretical).
  // Must be a power of two in [32,512] for tcgen05.alloc.
  // Config::kAccStages scales the reservation AND is now FUNCTIONALLY used (double-buffered acc pipe).
  // SINGLE dA accumulator → one stage = kEpiTileN(=TileN) cols (the forward's dual gate+up was 2*TileN):
  //   AccStages=1 → TileN cols (e.g. 64 for TileN=64): single buffer, 1-stage serialized handoff.
  //   AccStages=2 → 2*TileN cols (e.g. 128 for TileN=64): two buffers, MMA(N+1) overlaps epilogue(N).
  // CONSTRAINT: kAccStages*TileN ≤ 512. For TileN=64 → AccStages=2 gives 128 ≤ 512 ✓ (AccStages up to 8
  // would fit). kTmemCols must also be a pow2 for tcgen05.alloc (TileN=64 pow2 → AccStages pow2 keeps it).
  static constexpr int kTmemCols =
      Config::kAccStages * kEpiTileN;  // single dA acc: 128 for TileN=64, AccStages=2 (pow2 ✓)
  static_assert(kTmemCols >= 32 && kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0,
                "kTmemCols (= AccStages*TileN) must be a pow2 in [32,512]; "
                "for AccStages=2 keep TileN<=256 (AccStages*TileN=512<=512)");
  enum WarpRole { kLoad = 0, kMMA = 1, kEpilogue = 2, kEmpty = 3 };
  static CUTLASS_DEVICE WarpRole warp_role(int warp_idx) {
    if (warp_idx == 0) return kLoad;
    if (warp_idx == 1) return kMMA;
    if (warp_idx >= 4 && warp_idx < 4 + NumEpiWarps) return kEpilogue;
    return kEmpty;
  }

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      // M1a single-GEMM dA = dY · W2ᵀ: ONE A buffer (dY tiles) + ONE B buffer (W2ᵀ tiles) — the forward's
      // second B (the forward's up-weight) is dropped (single accumulator, no gate/up split).
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutX>> smem_x;        // dY tiles  (A operand)
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutW2t>> smem_w2t;  // W2ᵀ tiles (B operand)
      // M2: TWO output-staging tiles for the dswiglu output dY1[M,2I] — dgate → dY1[:, :I], dup → dY1[:, I:].
      cute::ArrayEngine<ElementOut, kEpiTileM * kEpiTileN> smem_out_gate;  // dgate staging (dY1[:, :I])
      cute::ArrayEngine<ElementOut, kEpiTileM * kEpiTileN> smem_out_up;    // dup   staging (dY1[:, I:])
      // M2b: per-tile column-reduction accumulator for dprob — ONE float per LOCAL row. The compute loop
      // atomicAdds dA·silu(gate)·up into dprob_smem[lrow] (sum over this tile's I-slice cols); after the
      // epi_smem barrier each row's partial is atomicAdded to the GLOBAL dprob[grow] (cross-CTA, since an
      // m_tile's I/kTileN n-tiles land on different CTAs). 128 floats = 512B (negligible smem).
      float dprob_smem[kEpiTileM];
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
      "SwiGLU smem (X + W2ᵀ-B + pipelines) exceeds SM100 capacity; reduce kStages.");

  // -------------------------------------------------------------------------------------------
  // Device params.  GROUPED (contiguous, uniform Mₑ): plain single pointers and ONE X TMA
  // descriptor over [G*Me, d] + ONE W2ᵀ TMA descriptor over [G·I, d].  The per-expert N-slice
  // selection is derived IN-KERNEL from the m-tile (tokens are contiguous by expert),
  // so NO pointer arrays / tensormap swaps are needed.  Single-expert is just the G=1 special case.
  //   M  = G*Me  (total tokens across all experts)   N  = I  (output width)   K = d
  //   Me = tokens per expert (uniform)                W2tN = G·I (rows of the W2ᵀ descriptor)
  // -------------------------------------------------------------------------------------------
  struct Params {
    TMA_X tma_load_x;
    TMA_W2t tma_load_w2t;     // ONE descriptor over [G·I, d]; expert e = the N-rows [e·I, (e+1)·I)
    TMA_A tma_store_dy1;    // ONE TMA-STORE descriptor over the whole output dY1[M,2I] (row-major); each
    // tile issues TWO box-stores into it — dgate-half at box col n_tile_local, dup-half at nI_per_expert
    // + n_tile_local (the [M,2I] dgate||dup layout).  Per-tile placement is just the box coordinate.
    ElementOut* ptr_dY1;    // [M, 2I] output (dgate || dup), row-major
    StrideA dA;             // (kept for API symmetry; the dY1 store uses the TMA descriptor)
    // ---- B2 BACKWARD-SPECIFIC INPUT --------------------------------------------------------------
    // dGrad is REPURPOSED as the SAVED SwiGLU input h[M,2I] pointer (NOT a gradient).  dA = dY·W2ᵀ is
    // computed in TMEM by the GEMM; the epilogue then reads h per element — gate = dGrad[grow*2I + gcol],
    // up = dGrad[grow*2I + I + gcol] — scales dA by prob[grow], and applies the SwiGLU backward.
    const ElementOut* dGrad = nullptr;  // [M, 2I] SAVED SwiGLU input h = gate‖up (read in the epilogue)
    int M, N, K;  // M == G*Me, N == I, K == d
    int Me;       // tokens per expert (uniform)
    int W2tN;      // == G·I, the N-extent of the single W2ᵀ TMA descriptor
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
    // expert→W2ᵀ-slice mapping becomes non-uniform, so this is the single varlen hook.  SonicMoE handles
    // arbitrary token counts upstream via token-rounding (pad each expert to a kTileM multiple).
    const int* m_tile_expert;
    // ---- F1 GATHER FUSION (optional, strictly backward-compatible) ----------------------------
    // m_gather_idx != nullptr => the LOAD warp GATHERS X rows directly from the UNPERMUTED source X via
    // SM100 TMA gather4 instead of reading the contiguous (already-permuted) X.  Length = M (the total
    // grouped/permuted row count); m_gather_idx[m] = the SOURCE row in the unpermuted X[T_src,d] to load
    // for grouped position m.  nullptr => the CURRENT contiguous tma_load_x path runs UNCHANGED (byte-
    // identical to the validated kernel).  tma_load_x_gather is the gather descriptor over the FULL
    // unpermuted X; it is ONLY referenced on the gather branch (never on the null path).
    const int* m_gather_idx = nullptr;  // [M] grouped-row -> source-row; nullptr => contiguous path
    TMA_X_GATHER tma_load_x_gather;     // gather descriptor over the unpermuted X [T_src_gather, d]
    int T_src_gather = 0;               // row extent of the unpermuted source X (== M for pure permute)
    // ---- PROB (per-token router gate, optional) -----------------------------------------------
    // Real MoE SwiGLU scales the activation by the token's routing probability:
    //   A[m, :] = prob[m] * silu(gate[m]) * up[m]    (the prob_tensor of the cuTe-DSL FP8 ref).
    // prob != nullptr => fp32 array of length M (grouped-row order, same row index as A); the epilogue
    // multiplies each output row by prob[m]. nullptr => no gating (byte-identical to the validated path).
    const float* prob = nullptr;
    // M2b: router-prob gradient OUTPUT [M] fp32. dprob[m] += Σ_i dA[m,i]·silu(gate[m,i])·up[m,i] (col-reduce
    // over i, atomicAdd-accumulated since an m_tile's n-tiles are spread across CTAs). CALLER PRE-ZEROES it;
    // nullptr => dprob not computed (M2a behavior).
    float* dprob = nullptr;
  };

  // ---- device entry ----
  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
    if (cute::thread0()) {
      printf("ERROR: Sm100DSwiGluKernel requires SM100a/f MMA. Compile with -arch=sm_100a.\n");
    }
    return;
#else
    using X = Underscore;
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    WarpRole role = warp_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Prefetch TMA descriptors from the Load warp (X over [G*Me,d] + ONE W2ᵀ over [G·I,d]).  On the F1
    // gather path the X descriptor is the gather one over the UNPERMUTED X; otherwise the contiguous one.
    if (role == kLoad && lane_predicate) {
      if (params.m_gather_idx != nullptr) {
        cute::prefetch_tma_descriptor(params.tma_load_x_gather.get_tma_descriptor());
      } else {
        cute::prefetch_tma_descriptor(params.tma_load_x.get_tma_descriptor());
      }
      cute::prefetch_tma_descriptor(params.tma_load_w2t.get_tma_descriptor());
    }
    // Prefetch the TMA-STORE descriptor (A[M,I]) from the elected epilogue lane: the store is issued by
    // the epilogue warps, so warm its descriptor in their warpgroup (a hint only; harmless to prefetch).
    if (role == kEpilogue && lane_predicate && warp_idx == 4) {
      cute::prefetch_tma_descriptor(params.tma_store_dy1.get_tma_descriptor());
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
    // M1a: TWO TMA boxes per stage (dY + ONE W2ᵀ B) == TmaTransactionBytes (X + one B). No second B box.
    lp.transaction_bytes = Config::TmaTransactionBytes;
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
    const int nI_per_expert = N / kTileN;              // M1a: W2ᵀ n-tiles per expert (I/kTileN)

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

      // F1 GATHER FUSION: when params.m_gather_idx != nullptr the X tile is GATHERED from the unpermuted
      // source X via TMA gather4 (one op per 4 rows), instead of the contiguous TMA below.  Strictly
      // gated: on the null path NONE of the gather setup/tensors are touched and the code path below is
      // byte-identical to the validated kernel.
      const bool kGather = (params.m_gather_idx != nullptr);

      // Defer-sliced TMA tensors. X is over (M=G*Me, K); W2ᵀ is over (W2tN=G·I, K) — ONE descriptor.
      Tensor mX = params.tma_load_x.get_tma_tensor(make_shape(M, K, 1));
      Tensor mW = params.tma_load_w2t.get_tma_tensor(make_shape(params.W2tN, K, 1));

      Tensor gX =
          local_tile(mX, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});  // (BM,BK,m,k,l)
      Tensor gW =
          local_tile(mW, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});  // (BN,BK,n,k,l)

      ThrMMA cta_mma =
          TiledMma{}.get_slice(block_rank_in_cluster % size(typename TiledMma::AtomThrID{}));
      Tensor tCgX = cta_mma.partition_A(gX);  // (MMA,MMA_M,MMA_K,m,k,l)
      Tensor tCgW = cta_mma.partition_B(gW);  // (MMA,MMA_N,MMA_K,n,k,l)

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});        // dY
      Tensor sWg = make_tensor(make_smem_ptr(ss.tensors.smem_w2t.begin()), SmemLayoutW2t{});  // W2ᵀ

      // CTA-in-cluster layout for tma_partition (sm100_mma_warpspecialized.hpp:521-538).
      Layout cta_layout_mnk = make_layout(ClusterShape{});
      Layout cta_layout_vmnk =
          tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
      auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

      auto [tXgX, tXsX] = tma_partition(params.tma_load_x, get<2>(cta_coord_vmnk),
                                        make_layout(size<2>(cta_layout_vmnk)),
                                        group_modes<0, 3>(sX), group_modes<0, 3>(tCgX));
      // M1a: ONE W2ᵀ descriptor, ONE smem B buffer (sWg). The expert's N-tile is sliced below.
      auto [tWggW, tWgsWg] = tma_partition(params.tma_load_w2t, get<1>(cta_coord_vmnk),
                                           make_layout(size<1>(cta_layout_vmnk)),
                                           group_modes<0, 3>(sWg), group_modes<0, 3>(tCgW));

      uint16_t mcast_mask_x = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
      uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

      // ---- F1 GATHER setup (only compiled when SWIGLU_ENABLE_GATHER_LOAD) ---------------------
      // gather4 = 4 ROWS per op.  The X smem tile per stage is kEpiTileM=128 rows × TileK (this CTA's
      // half of the 2-SM 256-row tile) → kGather4Ops = kEpiTileM/4 = 32 gather4 ops per (tile,k).  Each
      // op supplies 4 source-row indices (from m_gather_idx) as PER-COPY register operands via a zip
      // tensor (copy_traits_sm100_tma.hpp:248-280 unzip_tensor → (col coord, 4 idx)).  The destination is
      // the matching 4-row sub-slice of the swizzled SmemLayoutX stage.  The gather descriptor's column
      // coord (crd0) is the K-tile base (k * TileK) along the K-contiguous dim of the unpermuted X.
      // Guarded so the default build (and the validated contiguous path) carry zero extra state.
#if defined(SWIGLU_ENABLE_GATHER_LOAD)
      constexpr int kGather4Ops = kEpiTileM / 4;  // 32 for kEpiTileM=128
      Tensor mXg = params.tma_load_x_gather.get_tma_tensor(make_shape(params.T_src_gather, K));
      // Per-CTA global row base within the 256-row 2-SM tile: leader CTA owns local rows [0,128),
      // follower [128,256).  Added to m_tile*kTileM in the loop to index m_gather_idx.
      const int cta_local_row_base =
          int(block_rank_in_cluster) * (kTileM / int(size(typename TiledMma::AtomThrID{})));
#endif

      // PERSISTENT grid-stride over logical tiles.  load_prod is NOT re-init per tile: it advances
      // continuously across tiles (TMA pipelines are designed for continuous use; the producer state
      // wraps by Stages).  producer_tail is drained ONCE after the loop (a per-tile tail would stall).
      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;  // GLOBAL token m-tile, across ALL experts
        const int n_tile_local =
            tile % params.n_local_tiles;  // OUTPUT column tile, ∈ [0, nI_per_expert)
        // VARLEN-M: look up the expert from the per-m-tile table (uneven, TileM-aligned experts); else
        // UNIFORM fast path (m_tile / mtiles_per_expert).  Only the W2ᵀ slice depends on e; X/A row =
        // m_tile*kTileM regardless (experts are packed + TileM-aligned).  One L1-hot gmem read per tile.
        const int e = (params.m_tile_expert != nullptr) ? params.m_tile_expert[m_tile]
                                                        : (m_tile / mtiles_per_expert);
        // M1a W2ᵀ [G·I, d] row math: expert e owns N-rows [e·I, (e+1)·I) of the W2ᵀ stack, so its
        // n-tile = e * nI_per_expert + n_tile_local (nI_per_expert = I/kTileN). Single B, no gate/up.
        const int w2_ntile = e * nI_per_expert + n_tile_local;

        Tensor tXgX_k = tXgX(_, m_tile, _, _0{});       // dY (TMA, k)
        Tensor tWggWg_k = tWggW(_, w2_ntile, _, _0{});  // W2ᵀ tile for (expert e, n_tile)

        // F1 GATHER: this CTA's GLOBAL grouped-row base = m_tile*kTileM + the per-CTA half offset.
        // m_gather_idx[ row_base + 4*j + r ] (j∈[0,32), r∈[0,4)) gives the 4 source rows for gather op j.
#if defined(SWIGLU_ENABLE_GATHER_LOAD)
        const int row_base = m_tile * kTileM + cta_local_row_base;
#endif

        for (int k = 0; k < k_tile_count; ++k) {
          pipeline_load.producer_acquire(load_prod);
          auto* bar = pipeline_load.producer_get_barrier(load_prod);
          int wr = load_prod.index();
          if (cute::elect_one_sync()) {
            if (kGather) {
#if defined(SWIGLU_ENABLE_GATHER_LOAD)
              // ---- F1 GATHER X path (EXPERIMENTAL — gather4 issue; NOT yet on-GPU-verified) --------
              // gather4 = 4 ROWS per op.  The K-tile column coordinate (crd0) is k*TileK along the
              // unpermuted X's contiguous K dim.  Stage smem destination = the per-stage X tile
              // sX(_,_,_,wr).  For each of the 32 gather4 groups j, supply the 4 source-row indices
              // (from m_gather_idx) and the column coord via a zip tensor; the gather Copy_Atom's
              // copy_unpack (copy_traits_sm100_tma.hpp:259-279) unzips it into (col, idx0..3) and issues
              // cp.async.bulk.tensor.2d...tile::gather4 into the matching 4-row sub-slice.  mcast_mask_x
              // preserves the 2-SM multicast semantics of the contiguous path; the transaction barrier's
              // expect_tx is UNCHANGED (lp.transaction_bytes already counts the full X tile; the 32
              // gather4 ops + 1 W2ᵀ copy deliver exactly those bytes to the same *bar).
              //
              // WARNING: the zip-tensor coordinate construction and the swizzled-smem 4-row destination
              // partition below are the UNRESOLVED risk points (no gather4 mainloop reference exists in
              // CUTLASS 4.5.1).  This block is intentionally guarded by SWIGLU_ENABLE_GATHER_LOAD (OFF by
              // default) so the default build + the validated contiguous path are byte-identical and the
              // file always compiles.  See the deliverable report for the phased plan to finish this.
              Tensor sXg = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
              Tensor sXg_stage = sXg(_, _, _, wr);  // (MMA,MMA_M,MMA_K) this stage's X smem tile
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < kGather4Ops; ++j) {
                Tensor idx4 = make_tensor<int>(make_shape(_4{}));
                idx4(0) = params.m_gather_idx[row_base + 4 * j + 0];
                idx4(1) = params.m_gather_idx[row_base + 4 * j + 1];
                idx4(2) = params.m_gather_idx[row_base + 4 * j + 2];
                idx4(3) = params.m_gather_idx[row_base + 4 * j + 3];
                Tensor cXcol = mXg(make_coord(_, k * int(size<2>(TileShape{}))));  // K-tile col coord
                Tensor zsrc = make_zip_tensor(cXcol, idx4);
                Tensor dst4 = sXg_stage(make_coord(make_coord(4 * j, _), _));  // 4-row smem sub-slice
                copy(params.tma_load_x_gather.with(*bar, mcast_mask_x), zsrc, dst4);
              }
#else
              // SWIGLU_ENABLE_GATHER_LOAD not defined: the gather descriptor + index are PLUMBED end-to-end
              // (Params/host/test) but the on-GPU gather4 issue is not enabled.  Fall back to the contiguous
              // X load so the kernel still runs (it will read the PERMUTED contiguous X, not the gathered
              // rows — so SWIGLU_GATHER correctness requires building with -DSWIGLU_ENABLE_GATHER_LOAD).
              copy(params.tma_load_x.with(*bar, mcast_mask_x), tXgX_k(_, k), tXsX(_, wr));
#endif
            } else {
              copy(params.tma_load_x.with(*bar, mcast_mask_x), tXgX_k(_, k), tXsX(_, wr));
            }
            copy(params.tma_load_w2t.with(*bar, mcast_mask_b), tWggWg_k(_, k), tWgsWg(_, wr));
          }
          ++load_prod;
        }
      }
      // Drain ONCE after all tiles so peer CTAs in the cluster don't exit early.
      pipeline_load.producer_tail(load_prod);
    }

    // =========================================================================================
    // MMA warp — allocate TMEM once; ONE dA acc view (per acc-stage buffer); K-loop = 1 cute::gemm.
    // Mirrors sm100_mma_warpspecialized.hpp mma_init:548-576 + mma:648-708 (single GEMM, not doubled).
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
      Tensor acc_dA = accB;  // M1a: SINGLE accumulator (dA = dY · W2ᵀ). (the up accumulator is dropped)

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});       // dY
      Tensor sWg = make_tensor(make_smem_ptr(ss.tensors.smem_w2t.begin()), SmemLayoutW2t{});  // W2ᵀ
      Tensor tCrX = TiledMma::make_fragment_A(sX);    // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tCrWg = TiledMma::make_fragment_B(sWg);  // (MMA,MMA_N,MMA_K,PIPE)

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
          acc_dA.data() = ss.tmem_base_ptr + mma_buf;  // single acc at the stage's buffer base
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;   // first k_block zeroes the acc THIS tile
          for (int k = 0; k < k_tile_count; ++k) {
            pipeline_load.consumer_wait(load_cons);
            int rs = load_cons.index();
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < size<2>(tCrX); ++kb) {
              // M1a: ONE MMA into the single dA accumulator. Zero on the first k_block, One thereafter.
              cute::gemm(tiled_mma, tCrX(_, _, kb, rs), tCrWg(_, _, kb, rs), acc_dA);
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
    // EPILOGUE warps (wg1) — wait, TMEM-load both accs (gate,up) → SwiGLU BACKWARD: form grad =
    // dGrad·prob, then dgate = grad·up·silu'(gate) and dup = grad·silu(gate) (fp32) → bf16 → two TMA
    // bulk-stores into dY1[M,2I] (gate-half cols [0,I), up-half cols [I,2I)).  TMEM→reg mirrors FMHA
    // correction:795-863; h (gate||up) is NEVER written out.
    // =========================================================================================
    else if (role == kEpilogue) {
#ifndef EPI_REGS
#define EPI_REGS 160
#endif
      cutlass::arch::warpgroup_reg_alloc<
          EPI_REGS>();  // tunable (1 CTA/SM ⇒ huge reg headroom; bump only helps if epilogue spills)

      // TMEM is published ONCE (before the persistent loop); wait here once.  All tile-invariant epilogue
      // setup (MMA layout, TMEM accumulator views, coordinate tile, TMEM-load tiling, register tensors,
      // sGate/sUp staging + TMA-store partition mDY1_tma/bSG_sGate/bSG_sUp) is hoisted out of the loop —
      // only cta_row_offset/cta_col_offset, the per-tile pipeline handshake (consumer_wait/release), the
      // per-element dGrad read, and the per-tile TMA-store box coordinates are per-tile.
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
      Tensor tAcc_dA = tAcc;
      tAcc_dA.data() = ss.tmem_base_ptr + 0u;  // M1a: single dA accumulator (no up)

      // GLOBAL (M,N) identity coordinate tensor partitioned through the IDENTICAL MMA C-path that
      // produced the data fragment tAcc, sliced at THIS CTA. Rationale (rank + offset, both load-bearing):
      //   * tAcc = partition_fragment_C(mma,(256,128)) is rank-3 (MMA,MMA_M,MMA_N) (mma_atom.hpp
      //     partition_shape_C:563 → shape(dummy_v)). make_tmem_copy therefore builds Tiler_MN of rank-3
      //     (copy_atom.hpp make_cotiled_copy:553 tiles per mode of product_each(shape(tAcc))), so
      //     tidfrg_D asserts rank(dtensor) >= 3 (copy_atom.hpp:244). The OLD rank-2 (CTA_M,CTA_N)
      //     cAcc_cta violated this → "Rank of tensor to be partitioned too small."
      //   * cta_mma.partition_C(identity) returns rank-3 (MMA,MMA_M,MMA_N) with the SAME extents as tAcc
      //     (make_fragment_C:142 == shape(partition_C)), so partition_D is rank-legal AND tTMc ends up
      //     element-for-element congruent with partition_S(tAcc_dA) (same Tiler_MN, same thread slice).
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
      auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tAcc_dA);
      auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

      // NOTE: partition_S(tAcc_dA) — which builds the TMEM-source tensor tTM_dA — is deliberately NOT
      // hoisted here. partition_S CAPTURES the tensor's ITERATOR (data ptr +
      // layout) at call time, so a tensor partitioned with the hoisted (buf-0) .data() would keep reading
      // buffer 0 even after the per-tile .data() update. With AccStages=2 that would make the epilogue read
      // the WRONG buffer. Therefore partition_S is REDONE INSIDE the loop, AFTER tAcc_dA.data()
      // is set to the consumer stage's buffer (epi_cons.index()*kAccBufStride). (The
      // tiled_tmem_load copy object above carries the buffer-independent layout, so re-partitioning the
      // source is sufficient — the copy itself is NOT rebuilt.)
      // DATA is partition_S (TMEM source ordering); the COORDINATE must be partition_D, NOT partition_S.
      // Reason (this is the STOCK pattern; deviating from it caused every prior scramble):
      //   * The T2R copy `copy(tiled_tmem_load, tTM_dA, r_dA)` delivers each TMEM value into the
      //     register tensor in the copy atom's DST value ordering. For SM100_TMEM_LOAD_32dp32b32x the
      //     Src and Dst value layouts DIFFER (copy_traits_sm100.hpp:1633-1636 SrcLayout (32,32768):(0,1)
      //     vs DstLayout (32,1024):(1024,1)); RefLayout==SrcLayout, so tidfrg_S yields the Src ordering
      //     and tidfrg_D yields the Dst ordering (copy_atom.hpp:226,247). r_dA is built from
      //     shape(tTMc) and read by linear index i in the loop, so element i of r_dA sits in the DST
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
      //     r_dA into the Src value count/ordering, which then fails the register-vectorize assert
      //     `size(rD)==RegNumDst` in the DATA copy_unpack (copy_traits_sm100.hpp:396) — that is the
      //     reported compile error; partition_D sizes r_dA to exactly NumValDst=1024 so it vectorizes.
      //   * The +128 follower offset still enters via the V-slice baked into cAcc_cta: the 2-SM
      //     CLayout (_2,(M/2,N)):(M/2,(1,M)) (mma_traits_sm100.hpp:1700) puts the M-half in V with
      //     stride M/2=128; partition_C sliced at this CTA's V (line 461) selects V=0→rows[0,128) /
      //     V=1→rows[128,256). partition_D only RE-ORDERS those coordinate VALUES into the Dst register
      //     ordering — it does not drop or alter the +128 offset.
      Tensor tTMc =
          thr_tmem_load.partition_D(cAcc_cta);  // (T2R,T2R_M,T2R_N) global (row,col), DST order

      // Per-tile register staging (tile-invariant SHAPE; values overwritten each tile by the TMEM copy).
      Tensor r_dA = make_tensor<ElementAcc>(shape(tTMc));  // M1a: the dA values (single acc)

      // ---- M2 TMA-STORE EPILOGUE (dswiglu): dA is in TMEM (the GEMM result); gate/up are read from the
      // SAVED h[M,2I] (global). Per element: grad = dA·prob; dgate = grad·up·silu'(gate) → sGate; dup =
      // grad·silu(gate) → sUp; then TWO TMA bulk-stores → dY1[:, :I] (gate) and dY1[:, I:2I] (up).
      Tensor sGate = make_tensor(make_smem_ptr(ss.tensors.smem_out_gate.begin()), SmemLayoutA{});  // dgate
      Tensor sUp = make_tensor(make_smem_ptr(ss.tensors.smem_out_up.begin()), SmemLayoutA{});      // dup
      Tensor mDY1_tma = params.tma_store_dy1.get_tma_tensor(make_shape(M, 2 * N));  // (M,2I) coords
      ThrCopy thrblk_s2g = params.tma_store_dy1.get_slice(Int<0>{});
      Tensor bSG_sGate = thrblk_s2g.partition_S(sGate);  // (TMA,TMA_M,TMA_N)
      Tensor bSG_sUp = thrblk_s2g.partition_S(sUp);      // (TMA,TMA_M,TMA_N)
      const int nI_local_tiles = params.n_local_tiles;   // == I/kTileN: up-half box-col offset
      const int64_t h_row = static_cast<int64_t>(2) * params.N;  // saved h[M,2I] row stride (= 2I)
      // ONE issuing thread per CTA = elected lane of epi-warp 4 (lane_predicate = cute::elect_one_sync()).
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
        // MUST be redone here AFTER updating .data() — otherwise tTM_dA would read the stale (buf-0)
        // pointer and, with AccStages=2, the WRONG buffer. The copy layout (tiled_tmem_load) is unchanged.
        // (AccStages=1 → epi_buf is always 0, so this reproduces the original fixed-offset reads.)
        const uint32_t epi_buf = epi_cons.index() * uint32_t(kAccBufStride);
        tAcc_dA.data() = ss.tmem_base_ptr + epi_buf;
        Tensor tTM_dA = thr_tmem_load.partition_S(tAcc_dA);  // (T2R,T2R_M,T2R_N) tmem source (dA)
        copy(tiled_tmem_load, tTM_dA, r_dA);

#ifdef SWIGLU_DEBUG_PRINT
        // Guard to the FIRST tile only (tile == cluster_id) so the persistent loop doesn't spam.
        if (tile == cluster_id && thread_idx == 0 && block_rank_in_cluster == 0) {
          for (int i = 0; i < size(r_dA) && i < 24; ++i) {
            printf("T0 i=%2d row=%3d col=%3d gate=% .4f\n", i, get<0>(tTMc(i)) + cta_row_offset,
                   get<1>(tTMc(i)), float(r_dA(i)));
          }
        }
#endif

        // WAR (cross-tile sGate/sUp reuse): both staging buffers are reused every tile.  The PREVIOUS
        // tile's two TMA bulk-stores read them asynchronously; before we OVERWRITE them with THIS tile's
        // scatter, those stores MUST have drained.  tma_store_wait<0>() drains the pending cp.async.bulk
        // commit group (both stores share ONE group via the single tma_store_arrive below); the
        // NamedBarrier then makes ALL 128 epilogue threads observe the drain before any begins the
        // scatter.  Placed HERE (after this tile's TMEM-load, before its scatter) so the PREVIOUS stores
        // overlapped this tile's TMEM-load+dswiglu.  On the FIRST iteration nothing is pending → no-op.
        if (is_tma_store_lane) {
          cute::tma_store_wait<0>();
        }
        // M2b: zero this tile's dprob column-accumulator (one float per epi thread == per LOCAL row) BEFORE
        // the WAR barrier, so the barrier doubles as the "all rows zeroed" fence before the compute loop's
        // atomicAdds. thread_idx ∈ [0,128) == kEpiTileM, so all rows are covered.
        if (params.dprob != nullptr) ss.tensors.dprob_smem[thread_idx] = 0.0f;
        cutlass::arch::NamedBarrier epi_war_bar(NumEpiThreads, /*id=*/1u);
        epi_war_bar.arrive_and_wait();

        // M2 SwiGLU BACKWARD per element: dA = r_dA(i) (from TMEM, the GEMM result); gate/up read from the
        // SAVED h[M,2I] (params.dGrad repurposed as the h pointer): gate = h[grow,gcol], up = h[grow,I+gcol].
        // These SCATTERED LSU reads (the M3 TMA-prefetch experiment was a net loss — TMA contends with the
        // store-TMA; see the .cu note) overlap the unrolled compute and run parallel to the store-TMA engine.
        //   grad = dA·prob ;  dgate = grad·up·silu'(gate) → sGate ;  dup = grad·silu(gate) → sUp
        // silu(x)=x·σ(x), silu'(x)=σ(x)+silu(x)·(1-σ(x)). grow<M guards the h/prob reads on OOB rows.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(r_dA); ++i) {
          int lrow = get<0>(tTMc(i));  // LOCAL row ∈ [0,kEpiTileM)
          int lcol = get<1>(tTMc(i));  // LOCAL col ∈ [0,kEpiTileN)
          const int grow = lrow + cta_row_offset;  // GLOBAL token row
          const int gcol = lcol + cta_col_offset;  // GLOBAL intermediate col ∈ [0,I)
          const ElementAcc dA = r_dA(i);          // dA = dY·W2ᵀ (the FC2 dgrad, from TMEM)
          ElementAcc gate = ElementAcc(0), up = ElementAcc(0), p = ElementAcc(1);
          if (grow < params.M) {
            const int64_t hbase = static_cast<int64_t>(grow) * h_row + gcol;  // saved h[grow, gcol]
            gate = static_cast<ElementAcc>(params.dGrad[hbase]);              // h gate-half
            up = static_cast<ElementAcc>(params.dGrad[hbase + params.N]);     // h up-half (col I+gcol)
            if (params.prob != nullptr) p = params.prob[grow];
          }
          const ElementAcc grad = dA * p;
          const ElementAcc sig = ElementAcc(1) / (ElementAcc(1) + expf(-gate));  // σ(gate)
          const ElementAcc s = gate * sig;                                       // silu(gate)
          const ElementAcc siluprime = sig + s * (ElementAcc(1) - sig);          // d silu / d gate
          sGate(lrow, lcol) = static_cast<ElementOut>(grad * up * siluprime);    // dgate → dY1[:, :I]
          sUp(lrow, lcol) = static_cast<ElementOut>(grad * s);                   // dup   → dY1[:, I:2I]
          // M2b: dprob col-reduce partial — dprob[grow] += dA · A', A' = silu(gate)·up = s·up (the forward
          // SwiGLU output, NOT prob-scaled; QuACK's postact). A' is a FREE byproduct (s, up already in regs).
          // Accumulated into smem per LOCAL row; flushed to global dprob below. OOB rows have up=0 → adds 0.
          if (params.dprob != nullptr) {
            atomicAdd(&ss.tensors.dprob_smem[lrow], static_cast<float>(dA * s * up));
          }
        }

        // TMEM read is done (r_dA already in registers — single accumulator, no rUp; sA touches no TMEM).
        // fence_view_async_tmem_load orders these last TMEM loads; consumer_release returns THIS tile's
        // acc->epi stage (epi_cons.index()) so the MMA warp may reuse/free that buffer. With kAccStages=2
        // this release frees the stage that the MMA's producer_acquire for tile (current+2) will wait on,
        // so MMA(tile N+1) — which uses the OTHER stage — is NOT gated by this release and runs concurrently
        // with this epilogue. With kAccStages=1 producer and consumer share the one stage, so this release
        // is exactly what lets the NEXT tile's MMA producer_acquire proceed (the old serialized gating).
        cutlass::arch::fence_view_async_tmem_load();
        pipeline_epi.consumer_release(epi_cons);
        ++epi_cons;

        // Make all of sGate+sUp visible across the 128 epilogue threads, AND make those generic (LSU) smem
        // writes visible to the async TMA proxy, BEFORE issuing the two TMA bulk-stores.
        //  - fence_view_async_shared() (== fence.proxy.async.shared::cta) is issued by EVERY epi thread so
        //    its own sGate/sUp writes become visible to the TMA (async) proxy.  Store-side analogue of the
        //    load's cp.async fence (sm100_epilogue_tma_warpspecialized.hpp:768).
        //  - epi_smem_bar (NamedBarrier over EXACTLY the 128 epi threads — NOT __syncthreads(), which would
        //    deadlock on the Load/MMA/empty warps that never enter this branch) then cross-orders ALL 128
        //    threads' writes + fences before the single issuing thread reads sGate/sUp via TMA.  All 128
        //    reach it every iteration (no divergent exit precedes it); distinct hw barrier id 0u (vs WAR 1u).
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier epi_smem_bar(NumEpiThreads, /*id=*/0u);
        epi_smem_bar.arrive_and_wait();

        // M2b: flush THIS tile's dprob column-partials to GLOBAL dprob — one row per epi thread (thread_idx
        // ∈ [0,128) == LOCAL row; global row = thread_idx + cta_row_offset). atomicAdd because an m_tile's
        // I/kTileN n-tiles land on DIFFERENT CTAs that all contribute to the same dprob[m]. The epi_smem_bar
        // above made every thread's compute-loop atomicAdds into dprob_smem visible. Each thread reads only
        // its OWN dprob_smem[thread_idx], so the next tile's zero/accumulate of that slot is hazard-free.
        if (params.dprob != nullptr) {
          const int grow_p = thread_idx + cta_row_offset;
          if (grow_p < params.M) atomicAdd(&params.dprob[grow_p], ss.tensors.dprob_smem[thread_idx]);
        }

        // (3) TWO ASYNC TMA BULK-TENSOR STORES: ONE elected thread per CTA drains sGate → dY1[:, :I] and
        // sUp → dY1[:, I:2I], bypassing the L1/TEX/LSU path.  box_m = cta_row_offset/kEpiTileM (exact).
        // The gate-half box column = cta_col_offset/kEpiTileN = n_tile_local.  The up-half lives I columns
        // to the right in the [M,2I] output, so its box column = nI_local_tiles + n_tile_local (I/kEpiTileN
        // == nI_local_tiles).  Both stores commit into ONE cp.async.bulk group (a single tma_store_arrive
        // after both copies); the WAR tma_store_wait<0>() at the TOP of the next iteration drains it before
        // sGate/sUp are overwritten — so these stores overlap the next tile's TMEM-load+dswiglu.  The TMA
        // descriptor (built over the TRUE (M,2I) extent) clamps any partial tile (OOB never written).
        if (is_tma_store_lane) {
          const int box_m = cta_row_offset / kEpiTileM;  // exact (kTileM = AtomThrID*kEpiTileM)
          const int box_n = cta_col_offset / kEpiTileN;  // gate-half box col == n_tile_local
          Tensor gGate = local_tile(mDY1_tma, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}),
                                    make_coord(box_m, box_n));                 // dY1[:, :I] tile
          Tensor gUp = local_tile(mDY1_tma, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}),
                                  make_coord(box_m, nI_local_tiles + box_n));  // dY1[:, I:2I] tile
          copy(params.tma_store_dy1, bSG_sGate, thrblk_s2g.partition_D(gGate));
          copy(params.tma_store_dy1, bSG_sUp, thrblk_s2g.partition_D(gUp));
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
//        M2 SwiGLU-BACKWARD. The operand slots are REPURPOSED from the forward launcher; the generic
//        names X / W1 / dGrad are kept (the whole body uses them), but their M2 MEANING is:
//          X     ← dY    : [G*Me, d]   incoming grad wrt the FC2 output Y          (A operand)
//          W1    ← W2ᵀ   : [G*I,  d]   the TRANSPOSED FC2 weight, expert-in-N      (B operand)
//          dGrad ← h     : [G*Me, 2I]  the SAVED SwiGLU input h = gate‖up (read scattered in epilogue)
//          dY1           : [G*Me, 2I]  OUTPUT dgate‖dup = grad wrt h
//        This is ONE GEMM dA = dY · W2ᵀ (single accumulator) followed by the dswiglu-backward epilogue.
//        There is NO gate/up WEIGHT split here — that was the forward (W1=[gate‖up]). W2ᵀ carries the
//        expert in the N-dim exactly like the forward's W1ᵀ, so the GEMM is BLOCK-DIAGONAL with the
//        expert DERIVED FROM THE M-TILE in-kernel, and we build just TWO TMA descriptors via ONE
//        to_underlying_arguments call: A-operand (dY) over (G*Me, d), B-operand (W2ᵀ) over (G*I, d).
//        The gate‖up that DOES appear below is the epilogue OUTPUT dY1[:, :I]‖dY1[:, I:] (dgate‖dup),
//        not a weight. CONSTRAINT: uniform Me only (Me % kTileM == 0 and I % kTileN == 0; no boundary
//        predication).  Non-uniform Me would need per-expert cumulative-offset (prefix-sum) arrays.
// ============================================================================================
// The tuning params default to the DSwiGluConfig defaults, so the existing call
// LaunchDSwiGluGrouped<Element, ElementOut>(...) resolves to the current (default-tuned) behavior.
template <typename Element, typename ElementOut, int TileM_ = 256, int TileN_ = 64, int TileK_ = 16,
          int kStages_ = 16, int ClusterM_ = 2, int MinBlocks_ = 1, int AccStages_ = 2>
cudaError_t LaunchDSwiGluGrouped(
    const Element* X,      // [G*Me, d]  row-major  M2: dY  — incoming grad wrt FC2 output Y (A operand)
    const Element* W1,     // [G*I,  d]  row-major  M2: W2ᵀ — transposed FC2 weight, expert-in-N (B operand)
    const ElementOut* dGrad,  // [G*Me, 2I] row-major  M2: SAVED h = gate‖up (read scattered in epilogue)
    ElementOut* dY1,       // [G*Me, 2I] row-major  (OUTPUT: dgate‖dup = grad wrt h)
    int G, int Me, int I, int d, cudaStream_t stream, int device = 0, int sm_count = 0,
    // VARLEN-M (uneven, TileM-aligned experts): device array [M_varlen/kTileM]
    // mapping each global m-tile → expert id, and the total packed token count
    // M_varlen.  nullptr => UNIFORM (M = G*Me).  Experts must be TileM-aligned.
    const int* d_m_tile_expert = nullptr, int M_varlen = 0,
    // F1 GATHER FUSION (optional, plumbed for parity with the forward; the backward always passes
    // nullptr — X is the already-grouped activation).  nullptr => contiguous X path.
    const int* d_m_gather_idx = nullptr, int T_src = 0,
    // PROB (optional): per-token router gate, fp32 [M] in grouped-row order; the epilogue forms
    // grad = dA · prob[m] before the SwiGLU backward. nullptr => no prob-scaling (grad = dA).
    const float* d_prob = nullptr,
    // M2b: router-prob gradient OUTPUT [M] fp32 (CALLER PRE-ZEROES; atomicAdd-accumulated). nullptr => skip.
    float* d_dprob = nullptr) {
  using Config = DSwiGluConfig<Element, ElementOut, TileM_, TileN_, TileK_, kStages_, ClusterM_,
                              MinBlocks_, AccStages_>;
  using Kernel = Sm100DSwiGluKernel<Config>;
  using CollectiveMma = typename Config::CollectiveMma;
  using StrideX = typename Config::StrideX;
  using StrideW2t = typename Config::StrideW2t;
  using StrideA = typename Config::StrideA;

  // VARLEN-M: M = total packed (TileM-aligned) tokens across uneven experts; else uniform G*Me.
  const int M = (d_m_tile_expert != nullptr) ? M_varlen : (G * Me);  // total tokens (problem M)
  const int W2tN = G * I;  // M1a: total W2ᵀ rows (problem N, B-operand) = G·I (expert e → N-rows [e·I,+I))

  // Strides. make_cute_packed_stride wants {extent..., L}. X is RowMajor (G*Me,d) → A K-major (d,1,0).
  // W2ᵀ is K-contiguous [G·I,d] → B K-major: LayoutW2t=ColumnMajor ⇒ StrideB=Stride<int,_1,int>,
  // make_cute_packed_stride sets N-stride=d ⇒ dW2t=(d,1,0), i.e. W2ᵀ[n,k] at n*d+k (the physical layout).
  StrideX dX = cutlass::make_cute_packed_stride(StrideX{}, cute::make_shape(M, d, 1));
  StrideW2t dW2t = cutlass::make_cute_packed_stride(StrideW2t{}, cute::make_shape(W2tN, d, 1));
  StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, I, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = sm_count;

  // Problem shape (M=G*Me, N=G·I, K=d): one to_underlying_arguments builds BOTH the A-TMA over the
  // full token stack (M=G*Me) AND the B-TMA over the full W2ᵀ stack (N=G·I).
  auto problem = cute::make_shape(M, W2tN, d);

  typename CollectiveMma::Arguments mainloop_args{};
  mainloop_args.ptr_A = X;
  mainloop_args.dA = dX;
  mainloop_args.ptr_B = W1;
  mainloop_args.dB = dW2t;
  typename CollectiveMma::Params mainloop_params = CollectiveMma::to_underlying_arguments(
      problem, mainloop_args, /*workspace=*/nullptr, hw_info);

  // TMA-STORE descriptor for the output dY1[M, 2I] (row-major, single contiguous tensor; the gate-half
  // occupies cols [0,I), the up-half cols [I,2I), per the [M,2I] gate||up layout).  Built ONCE over the
  // TRUE (M, 2I) extent so the descriptor's box clamping handles any partial tile (OOB) for free; each
  // tile issues TWO box-stores into it (gate column n_tile_local, up column nI_per_expert + n_tile_local).
  // Stride (2I,1) = row-major; the box layout = Config::SmemLayoutA (kEpiTileM,kEpiTileN) row-major, the
  // SAME layout sGate/sUp use → make_tma_copy builds a no-swizzle box (kEpiTileM,kEpiTileN).
  // M2: TMA-STORE descriptor for the output dY1[M, 2I] (row-major; gate-half cols [0,I), up-half [I,2I)).
  // Built ONCE over the TRUE (M, 2I) extent; each tile issues TWO box-stores (gate col n_tile_local, up
  // col nI_per_expert + n_tile_local). Stride (2I,1); box = Config::SmemLayoutA (kEpiTileM,kEpiTileN).
  cute::Tensor mDY1_store = cute::make_tensor(
      cute::make_gmem_ptr(dY1),
      cute::make_layout(cute::make_shape(M, 2 * I), cute::make_stride(2 * I, cute::_1{})));
  auto tma_store_dy1 =
      make_tma_copy(cute::SM90_TMA_STORE{}, mDY1_store, typename Config::SmemLayoutA{});

  // ---- F1 GATHER FUSION: build the gather TMA descriptor over the FULL UNPERMUTED X ------------
  // The gmem tensor is the unpermuted X viewed as a 2D [T_src, d] RowMajor tensor (K=d contiguous,
  // stride (d,1)).  make_tma_copy with SM100_TMA_LOAD_MULTICAST_2D_GATHER4 auto-detects gather: it
  // reduces the column basis to (d,1) and ×4 the box so one gather4 op moves 4 ROWS × the K-tile into
  // smem (copy_traits_sm90_tma.hpp:1153-1165).  cluster_size = ClusterM selects the multicast variant's
  // box truncation (same num_multicast the contiguous X descriptor uses).  The smem layout is the SAME
  // per-stage X tile the contiguous TMA_X uses (SmemLayoutX(_,_,_,0)) so the baked-in swizzle matches the
  // MMA-fragment layout the mma warp reads.  GUARDED by SWIGLU_ENABLE_GATHER_LOAD (see Config::TMA_X_
  // GATHER): when OFF, the gather descriptor TYPE == TMA_X, so we initialize the Params member from the
  // contiguous tma_load_a placeholder (it is never read on the null path nor on the disabled gather path).
  const int T_src_eff = (d_m_gather_idx != nullptr) ? (T_src > 0 ? T_src : M) : M;
#if defined(SWIGLU_ENABLE_GATHER_LOAD)
  cute::Tensor mX_gather = cute::make_tensor(
      cute::make_gmem_ptr(X),
      cute::make_layout(cute::make_shape(T_src_eff, d), cute::make_stride(d, cute::_1{})));
  typename Config::TMA_X_GATHER tma_load_x_gather = make_tma_copy(
      cute::SM100_TMA_LOAD_MULTICAST_2D_GATHER4{}, mX_gather,
      typename Config::SmemLayoutX{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
      cute::size<0>(typename Config::ClusterShape{}));
#else
  typename Config::TMA_X_GATHER tma_load_x_gather = mainloop_params.tma_load_a;  // placeholder (== TMA_X)
#endif

  typename Kernel::Params params;
  params.tma_load_x = mainloop_params.tma_load_a;
  params.tma_load_w2t = mainloop_params.tma_load_b;
  params.tma_store_dy1 = tma_store_dy1;
  params.ptr_dY1 = dY1;
  params.dGrad = dGrad;  // [M, I] incoming grad wrt the prob-scaled SwiGLU output
  params.dA = dA;
  params.M = M;  // G*Me
  params.N = I;  // intermediate width I (dGrad stride; dY1 is [M,2I])
  params.K = d;
  params.Me = Me;    // tokens per expert (uniform fallback; unused when m_tile_expert != nullptr)
  params.W2tN = W2tN;  // G·I
  params.m_tile_expert = d_m_tile_expert;  // VARLEN-M expert table (nullptr => uniform fast path)
  params.m_gather_idx = d_m_gather_idx;    // F1 gather index (nullptr => contiguous X path, unchanged)
  params.tma_load_x_gather = tma_load_x_gather;  // gather descriptor over the unpermuted X [T_src,d]
  params.T_src_gather = T_src_eff;               // unpermuted source row extent
  params.prob = d_prob;                          // per-token router gate (nullptr => no gating)
  params.dprob = d_dprob;                        // M2b router-prob grad OUTPUT (nullptr => skip)

  // Logical tile grid: m_tile spans ALL experts' tokens (num_m_tiles = G*Me/kTileM); n_tile_local spans
  // ONE expert's output width (I/kTileN).  total_tiles = num_m_tiles * num_n_local_tiles is the LOGICAL
  // tile count (NOT multiplied by cluster.x).  The kernel decodes m_tile = tile / num_n_local_tiles,
  // n_tile_local = tile % num_n_local_tiles from a linear tile id (see Sm100DSwiGluKernel::operator()).
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

// (No single-expert wrapper for the backward — the grouped launcher above is the only entry; the
// per-op backward always calls it grouped, mirroring how the forward op calls LaunchSwiGluGrouped.)

}  // namespace grouped_gemm_dswiglu
}  // namespace transformer_engine
