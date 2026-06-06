/***************************************************************************************************
 * F2 route#2 SwiGLU-fused MoE up-proj grouped GEMM — V2 (MegaMoE 5D-TMA gran-8 gate/up interleave).
 *
 * GOAL: replace V1's "2 W1-TMAs (gate,up) -> 2 GEMMs(N=TileN) -> 2 TMEM accs -> epilogue reads both"
 *       with MegaMoE's design: ONE 5D-TMA delivers gate/up INTERLEAVED at granularity 8 straight from
 *       the CONTIGUOUS (Muon-safe, un-permuted) W1, so a SINGLE GEMM(N=2*TileN) fills ONE interleaved
 *       acc and the epilogue pairs silu(gate)*up WITHIN ONE TMEM load.
 *
 * WHY (vs cutlass_grouped_gemm_swiglu.cuh V1):
 *   V1 already co-locates gate/up per epilogue-thread via 2 accs, so the win is NOT "fusion" (V1 has
 *   it) — it is: 1 TMA issue vs 2, 1 wider (more MMA-efficient) GEMM vs 2, and 1 TMEM-load + paired
 *   SiLU vs 2 loads — i.e. fewer instructions in the ~33%-elementwise epilogue (the measured hot spot)
 *   and fewer load/MMA issues, all WITHOUT touching the weight layout (W1 stays [G*2I,d] contiguous).
 *
 * REFERENCE (this repo's sibling project /Users/min.yang/workcode/transformerengine):
 *   common/gemm/megamoe_vendor/deep_gemm/impls/sm100_bf16_mega_moe.cuh:34-37  (WEIGHT LAYOUT CONTRACT)
 *   common/gemm/megamoe_vendor/deep_gemm/impls/sm100_bf16_mega_moe.cuh:751-756 (tma::copy_5d call)
 *   common/gemm/megamoe_vendor/deep_gemm/common/tma_copy.cuh:92-119            (copy_5d: c0=K,c1=w8,
 *                                                                               c2=gu,c3=group,c4=expert)
 *   qa/test_tma_interleave.cu:33-51  (GROUND TRUTH: 4D-on-contiguous == 2D-on-host-gran8, BITWISE,
 *                                     for swizzle 0 AND 128B — the layout this file's host desc mirrors)
 *
 * STATUS: FIRST-CUT (compile-iterate on a SM100 box). IMPLEMENTED in this file:
 *   - DEVICE body operator() is concrete (NOT a stub): single interleaved GEMM at N=kMmaN into ONE acc,
 *     ONE TMEM-load, gran-G de-interleave epilogue (silu(gate)*up) + prob + V1 TMA-store. Mirrors the
 *     down_v2 single-B-tile structure (load 1 W partition, 1 cute::gemm, 1 acc buffer, 1 partition_S).
 *   - DEFAULT path = V2a: W1 is HOST-PERMUTED to gran-G interleaved per output-tile and fed through the
 *     SAME plain 2D TMA machinery V1/down_v2 use (Config::TMA_W1 at box-N=kMmaN). This is the compilable
 *     path: it touches only proven V1 partition/gemm/store ops, so it should "compile or close".
 *   - OPTIONAL path = V2b (5D-TMA-from-CONTIGUOUS, Muon-safe), macro-gated by SWIGLU_V2_5D_TMA: the
 *     make_w1_gran8_5d_desc() raw CUtensorMap is built in the launcher and the LOAD warp issues a raw
 *     cute::SM100_TMA_2SM_LOAD_5D / SM90_TMA_LOAD_5D into the SAME smem_w1_il buffer (no host permute).
 *     This path is NOT yet expected to pass without box iteration (R1 swizzle match) and is OFF by
 *     default so the build stays green; it is wired end-to-end (descriptor + Params + device issue) so
 *     it can be flipped on and debugged in place. See the deliverable report for the phased plan.
 *   The whole file must still be compile-iterated on a SM100 box (TMEM/MMA/partition correctness only
 *   converges by building, per the V1 kernel's history) and then AUTOTUNED for 性能最优.
 *
 * OPEN RISKS that REQUIRE the box (cannot be resolved by reading source):
 *   R1 SWIZZLE MATCH: the 5D desc swizzle MUST equal the COMPILED Config::SmemLayoutW1 swizzle. V1 uses
 *      TileK=16 (32B K-tile) — CUTLASS likely picks 32B/none, NOT MegaMoE's 128B. Either (a) set the
 *      desc swizzle from the actual SmemLayoutW1, or (b) retile to TileK=64 (128B atom, MegaMoE-aligned)
 *      and re-autotune. Verify by dumping one SMEM tile and diffing vs the host-gran8 permute (mirror
 *      test_tma_interleave.cu) BEFORE wiring the MMA.
 *   R2 MMA CONSUMES INTERLEAVED B: the single GEMM's B fragment is the standard SmemLayoutW1 at N=2*TileN;
 *      the interleave is in the DATA (which N-row is gate vs up), not the layout — so the MMA "just works"
 *      IFF R1 holds. Confirm acc[n] == X·W1row(group,gu,w8) with a raw-acc dump (SWIGLU_DEBUG_RAW_GATE-style).
 *   R3 EPILOGUE DATAPATH PAIRING: acc col n is gate iff (n/8)%2==0 else up; silu pairs col(16k+j) with
 *      col(16k+8+j). gran-8 is chosen so both land in the SAME epilogue thread under SM100_TMEM_LOAD_
 *      32dp32b32x — verify the per-thread tTMc col mapping actually co-locates the octet pair (else a
 *      warp shuffle is needed, which would erase the epilogue win).
 **************************************************************************************************/
#pragma once

#include <cuda.h>  // CUtensorMap, cuTensorMapEncodeTiled

#include "cute/tensor.hpp"
#include "cute/arch/copy_sm90_tma.hpp"   // cute::SM90_TMA_LOAD_5D (V2b raw 5D issue, 1-SM)
#include "cute/arch/copy_sm100_tma.hpp"  // cute::SM100_TMA_2SM_LOAD_5D (V2b raw 5D issue, 2-SM)
#include "cutlass_grouped_gemm_swiglu.cuh"  // V1: SwiGluConfig, Sm100SwiGluKernel, LaunchSwiGluGrouped

namespace transformer_engine {
namespace grouped_gemm_swiglu {

// ============================================================================================
// HOST · 5D TMA descriptor for W1[G*2I, d] delivering the gran-8 gate/up interleave from CONTIGUOUS
// HBM (no permute → Muon-safe).  Row axis 2I decomposed (group=I/8, gu=2, w8=8); the 5th mode is the
// expert.  Coords consumed by the LOAD warp via cute::SM90_TMA_LOAD_5D / SM100_TMA_2SM_LOAD_5D:
//     c0 = k_idx (K, inner)      c1 = w8 (0)         c2 = gu (0)
//     c3 = n_block * (LOAD_BN/16) [group base]       c4 = local_expert_idx
// (the 0s for c1/c2 are the box origins; the box EXTENT covers w8∈[0,8), gu∈[0,2)).
//
// globalDim / globalStride / boxDim are the EXACT layout proven bitwise-equal to the host-gran8 permute
// in qa/test_tma_interleave.cu:38-43 (extended here with the expert mode):
//     gd[5] = { d, 8(w8), 2(gu), I/8(group), G(expert) }            // inner→outer
//     gs[4] = { d, I*d, 8*d, 2*I*d } * sizeof(Element)              // strides for modes 1..4
//             ( w8:+1 row | gu:+I rows gate→up | group:+8 rows | expert:+2I rows )
//     sd[5] = { K_box, 8, 2, LOAD_BN/16, 1 }                        // box (per stage tile)
//
// kSwizzleBytes MUST match the compiled Config::SmemLayoutW1 swizzle (see RISK R1). Pass it in from the
// caller after inspecting SmemLayoutW1 (or after retiling to a 128B K-atom). K_box = swizzle? swizzle/
// sizeof(Element) : TileK.
// --------------------------------------------------------------------------------------------
template <typename Element>
inline CUresult make_w1_gran8_5d_desc(CUtensorMap* desc, const Element* W1, int G, int I, int d,
                                      int TileK, int LOAD_BN, int kSwizzleBytes) {
  static_assert(sizeof(Element) == 2, "bf16/fp16 only (gran-8 octet = 16B at 2B/elt)");
  // CU_TENSOR_MAP_DATA_TYPE: bf16 vs fp16. Caller guarantees Element ∈ {nv_bfloat16, half}.
  const CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;  // TODO: half → FLOAT16 variant
  const cuuint64_t e = sizeof(Element);
  const cuuint64_t gd[5] = {(cuuint64_t)d, 8u, 2u, (cuuint64_t)(I / 8), (cuuint64_t)G};
  const cuuint64_t gs[4] = {(cuuint64_t)d * e, (cuuint64_t)I * d * e, (cuuint64_t)8 * d * e,
                            (cuuint64_t)2 * I * d * e};
  // box-K = TileK (the per-op K-tile extent), DECOUPLED from the swizzle pattern. The old
  // kSwizzleBytes/sizeof coincidentally == TileK only at TileK16/sw32; at other TileK it under/over-fills
  // the K-tile → barrier byte mismatch → deadlock. The swizzle below is the smem PATTERN (separate).
  const int kbox = TileK;
  const cuuint32_t sd[5] = {(cuuint32_t)kbox, 8u, 2u, (cuuint32_t)(LOAD_BN / 16), 1u};
  const cuuint32_t es[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMapSwizzle sw = kSwizzleBytes == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                          : kSwizzleBytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                          : kSwizzleBytes == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                                : CU_TENSOR_MAP_SWIZZLE_NONE;
  return cuTensorMapEncodeTiled(desc, dtype, /*rank=*/5, const_cast<Element*>(W1), gd, gs, sd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ============================================================================================
// SwiGluConfigV2 — MODEL A (single interleaved GEMM). DECOUPLE MMA-N from OUTPUT-N:
//   kMmaN  = 2*kOutN  -> CollectiveMma/TiledMma/SmemLayoutB/acc built at N=kMmaN (gate||up interleaved)
//   kOutN  = TileN_   -> scheduler / TMA-store / epilogue-out tiling IDENTICAL to V1
// This REPLACES the inherit-only scaffold (v2.cuh:102-122). It does NOT inherit SwiGluConfig because
// the base builds the collective at N=TileN; V2 must build it at N=kMmaN. All output-side constants are
// re-derived from kOutN so they are byte-identical to V1 (verified vs swiglu.cuh:215-233, 326, 337-339).
// --------------------------------------------------------------------------------------------
#ifndef GLU_G
#define GLU_G 1   // sweepable {1,8,16,32}. gran-1 = QuACK-style adjacent cols = lowest R3 risk (default).
#endif

template <typename Element_, typename ElementOut_, int TileM_ = 256, int TileN_ = 64, int TileK_ = 32,
          int kStages_ = 8, int ClusterM_ = 2, int MinBlocks_ = 1, int AccStages_ = 2>
struct SwiGluConfigV2 {
  static_assert(ClusterM_ == 1 || ClusterM_ == 2, "ClusterM_ must be 1 or 2");

  using Element = Element_;
  using ElementAcc = float;
  using ElementOut = ElementOut_;

  using ArchTag = cutlass::arch::Sm100;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using LayoutX = cutlass::layout::RowMajor;      // X (M,d) K-contiguous  (== V1 swiglu.cuh:97)
  using LayoutW1 = cutlass::layout::ColumnMajor;  // W1 (N,d) K-contiguous (== V1 swiglu.cuh:104)
  using LayoutA = cutlass::layout::RowMajor;      // output A[M,I]

  static constexpr int AlignX = 128 / cutlass::sizeof_bits<Element>::value;     // 8
  static constexpr int AlignW1 = 128 / cutlass::sizeof_bits<Element>::value;    // 8
  static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

  // ---- DECOUPLED N -------------------------------------------------------------------------
  static constexpr int kOutN = TileN_;        // OUTPUT n-tile (scheduler/store/epi-out) == V1 TileN
  static constexpr int kMmaN = 2 * TileN_;    // interleaved gate||up MMA width (single GEMM consumes)
  static constexpr int kGluGran = GLU_G;
  static constexpr int kInterleaveN = kMmaN;  // scaffold alias kept for the 5D-desc (V2b) path
  static_assert(kOutN % kGluGran == 0, "kOutN must be a multiple of the interleave gran");
  // ATOM limit (sm100_common.inl:379): N%8==0 && N<=256 for the 2-SM bf16 atom (no f4/f6 N%256 gate).
  static_assert(kMmaN % 8 == 0 && kMmaN <= 256, "kMmaN=2*TileN must be %8 and <=256 (2-SM bf16 atom)");

  // *** THE structural delta vs V1: TileShape N = kMmaN feeds CollectiveBuilder/TiledMma/SmemLayoutB ***
  using TileShape = cute::Shape<cute::Int<TileM_>, cute::Int<kMmaN>, cute::Int<TileK_>>;
  using ClusterShape = cute::Shape<cute::Int<ClusterM_>, cute::_1, cute::_1>;
  using KernelSchedule =
      std::conditional_t<ClusterM_ == 2, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

  static constexpr int kStages = kStages_;
  static constexpr int kMinBlocks = MinBlocks_;
  static constexpr int kAccStages = AccStages_;

  using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, Element, LayoutX, AlignX, Element, LayoutW1, AlignW1, ElementAcc, TileShape,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>, KernelSchedule>::CollectiveOp;

  using TiledMma = typename CollectiveMma::TiledMma;
  using AtomThrShapeMNK = typename CollectiveMma::AtomThrShapeMNK;  // (2,1,1) under 2-SM
  using SmemLayoutX = typename CollectiveMma::SmemLayoutA;          // (MMA,M,K,PIPE) — N-independent
  using SmemLayoutW1 = typename CollectiveMma::SmemLayoutB;         // (MMA,N=kMmaN,K,PIPE) — ONE B tile
  static constexpr int Stages = CollectiveMma::DispatchPolicy::Stages;
  using CtaShapeMNK = typename CollectiveMma::CtaShape_MNK;

  using StrideX = typename CollectiveMma::StrideA;
  using StrideW1 = typename CollectiveMma::StrideB;
  using StrideA = cutlass::detail::TagToStrideC_t<LayoutA>;
  using MainloopArguments = typename CollectiveMma::Arguments;
  using MainloopParams = typename CollectiveMma::Params;
  using TMA_X = typename MainloopParams::TMA_A;
  using TMA_W1 = typename MainloopParams::TMA_B;   // V2a: plain 2D TMA at N=kMmaN over host-permuted W1
  using TMA_X_GATHER = TMA_X;                       // gather disabled in V2 (placeholder, == V1 OFF path)
  static constexpr uint32_t TmaTransactionBytes = CollectiveMma::TmaTransactionBytes;

  // R1 (B200-confirmed in scaffold): SmemLayoutW1 = Sw<1,4,3> = 32B swizzle for TileK=16 bf16.
  static constexpr int kSwizzleBytesW1 = 32;  // smem PATTERN (R1: match SmemLayoutW1); box-K is TileK (decoupled)

  // ---- OUTPUT side derived from kOutN (NOT kMmaN) — byte-identical to V1 swiglu.cuh:215-233 -------
  static constexpr int kEpiTileM_cfg =
      cute::size<0>(cute::take<0, 2>(TileShape{})) / int(cute::size(AtomThrShapeMNK{}));  // 128
  static constexpr int kEpiTileN_cfg = kOutN;                                             // 64
  using SmemLayoutA = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kEpiTileM_cfg>{}, cute::Int<kEpiTileN_cfg>{}),
      cute::make_stride(cute::Int<kEpiTileN_cfg>{}, cute::_1{})));
  using TMA_A = decltype(make_tma_copy(
      cute::SM90_TMA_STORE{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOut*>(nullptr)),
                        cute::make_layout(cute::make_shape(int(0), int(0)),
                                          cute::make_stride(int(0), cute::_1{}))),
      SmemLayoutA{}));

  // ---- TMEM accounting (kMmaN == V1's 2*kEpiTileN -> SAME numbers, asserts verbatim) -------------
  static constexpr int kAccBufStride = kMmaN;                 // == V1 2*kEpiTileN
  static constexpr int kTmemCols = kAccStages * kMmaN;        // == V1 AccStages*2*kEpiTileN
  static_assert(kTmemCols >= 32 && kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0,
                "kTmemCols (= AccStages*kMmaN) must be pow2 in [32,512]; AccStages=2 -> kOutN<=128");
};

// gran-G de-interleave helpers (output col c -> gate/up acc col). swiglu_pair = silu(gate)*up.
// (kGluGran lives in the Config; the kernel epilogue inlines these formulas, the test reuses them.)
template <int G>
CUTLASS_DEVICE float swiglu_pair(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}
CUTLASS_HOST_DEVICE constexpr int v2_gate_acc_col(int c, int G) { return 2 * (c / G) * G + (c % G); }
CUTLASS_HOST_DEVICE constexpr int v2_up_acc_col(int c, int G) { return 2 * (c / G) * G + G + (c % G); }

// ============================================================================================
// §V2.3 · Sm100SwiGluKernelV2 — single interleaved GEMM. Near-verbatim copy of Sm100SwiGluKernel
// (swiglu.cuh:254-1097) with EXACTLY these deltas (each marked //<<V2):
//   SharedStorage:   ONE smem_w1_il  (drop smem_w1_gate/up)
//   Load warp:       ONE W1 TMA partition + ONE copy (drop the up partition/copy)
//   MMA warp:        ONE acc of kMmaN cols + ONE cute::gemm (drop acc_up + 2nd gemm)
//   Epilogue:        ONE TMEM-load into rAll, then gran-G de-interleave (silu(gate)*up by column)
//   transaction_bytes: ONE wide B box (== V1 gate+up bytes; numerically unchanged)
// Everything else (pipelines, persistent scheduler, TMEM alloc/free, TMA-store tail) is V1-identical.
// ============================================================================================
template <typename Config>
struct Sm100SwiGluKernelV2 {
  using Element = typename Config::Element;
  using ElementAcc = typename Config::ElementAcc;
  using ElementOut = typename Config::ElementOut;
  using TileShape = typename Config::TileShape;       // N == kMmaN
  using ClusterShape = typename Config::ClusterShape;
  using TiledMma = typename Config::TiledMma;
  using AtomThrShapeMNK = typename Config::AtomThrShapeMNK;
  using SmemLayoutX = typename Config::SmemLayoutX;
  using SmemLayoutW1 = typename Config::SmemLayoutW1; // N == kMmaN  //<<V2
  using StrideX = typename Config::StrideX;
  using StrideW1 = typename Config::StrideW1;
  using StrideA = typename Config::StrideA;
  using TMA_X = typename Config::TMA_X;
  using TMA_X_GATHER = typename Config::TMA_X_GATHER;
  using TMA_W1 = typename Config::TMA_W1;
  using TMA_A = typename Config::TMA_A;
  using SmemLayoutA = typename Config::SmemLayoutA;
  static constexpr int Stages = Config::Stages;

  static constexpr int kGluGran = Config::kGluGran;   //<<V2  de-interleave gran
  static constexpr int kMmaN = Config::kMmaN;         //<<V2  acc width
  static constexpr int kOutN = Config::kOutN;         //<<V2  output width

  using ArchTag = cutlass::arch::Sm100;
  static constexpr bool kIs2Sm = (cute::size(AtomThrShapeMNK{}) == 2);
  using TmemAllocator =
      std::conditional_t<kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  using PipelineLoad = cutlass::PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShapeMNK>;
  using PipelineEpi = cutlass::PipelineUmmaAsync<Config::kAccStages, AtomThrShapeMNK>;

  static constexpr int NumEpiWarps = 4;
  static constexpr int NumWarps = 8;
  static constexpr int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  static constexpr int MinBlocksPerMultiprocessor = Config::kMinBlocks;
  static constexpr int NumEpiThreads = NumEpiWarps * cutlass::NumThreadsPerWarp;

  // OUTPUT staging tile = kEpiTileM x kOutN (== V1; uses kOutN, NOT kMmaN).
  static constexpr int kEpiTileM =
      cute::size<0>(take<0, 2>(TileShape{})) / int(cute::size(AtomThrShapeMNK{}));  // 128
  static constexpr int kEpiTileN = kOutN;                                           // 64  //<<V2
  static_assert(kEpiTileM == Config::kEpiTileM_cfg && kEpiTileN == Config::kEpiTileN_cfg,
                "Config TMA-store tile extents must match Kernel kEpiTileM/kEpiTileN");
  // ONE interleaved acc / stage of kMmaN cols (== V1's 2*kEpiTileN window, reinterpreted as interleaved).
  static constexpr int kAccBufStride = kMmaN;                       //<<V2 (== V1 2*kEpiTileN)
  static constexpr int kTmemCols = Config::kAccStages * kMmaN;      //<<V2 (== V1 AccStages*2*kEpiTileN)
  static_assert(kTmemCols >= 32 && kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0,
                "kTmemCols must be pow2 in [32,512]");

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
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutW1>> smem_w1_il;  //<<V2 ONE wide B (kMmaN)
      cute::ArrayEngine<ElementOut, kEpiTileM * kOutN> smem_out;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _0> {
      alignas(16) typename PipelineLoad::SharedStorage load;
      alignas(16) typename PipelineEpi::SharedStorage epi;
    } pipelines;
    uint32_t tmem_base_ptr;
  };
  static constexpr int SharedStorageSize = static_cast<int>(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= (228 * 1024), "V2 smem exceeds SM100 capacity; reduce kStages.");

  struct Params {
    TMA_X tma_load_x;
    TMA_W1 tma_load_w1;   // V2a: ONE plain-2D descriptor over host-permuted [G*2I, d] at box-N = kMmaN
    // V2b (SWIGLU_V2_5D_TMA): raw 5D CUtensorMap over the CONTIGUOUS (un-permuted, Muon-safe) W1, built
    // by make_w1_gran8_5d_desc in the launcher. Carried unconditionally (a CUtensorMap is 128B POD); the
    // LOAD warp only dereferences it on the 5D path. n5d_per_expert = I/(LOAD_BN/16) groups per expert.
    CUtensorMap w1_5d_desc{};
    int load_bn_5d = 0;     // == kMmaN box-N for the 5D path (groups of 16 N-rows per box col)
    TMA_A tma_store_a;
    ElementOut* ptr_A;
    StrideA dA;
    int M, N, K;          // M==G*Me, N==I (output width), K==d
    int Me;
    int W1N;              // == G*2I
    int total_tiles;
    int n_local_tiles;    // == I/kOutN  (decode stride)
    int num_clusters;
    const int* m_tile_expert;
    const int* m_gather_idx = nullptr;   // V2: kept for ABI parity; gather not wired (treated as null)
    TMA_X_GATHER tma_load_x_gather;
    int T_src_gather = 0;
    const float* prob = nullptr;
  };

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && \
    !defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM103F_ENABLED)
    if (cute::thread0())
      printf("ERROR: Sm100SwiGluKernelV2 requires SM100a/f or SM103a/f. Compile -arch=sm_100a.\n");
    return;
#else
    using X = Underscore;
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    WarpRole role = warp_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    if (role == kLoad && lane_predicate) {
      cute::prefetch_tma_descriptor(params.tma_load_x.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_w1.get_tma_descriptor());
    }
    if (role == kEpilogue && lane_predicate && warp_idx == 4) {
      cute::prefetch_tma_descriptor(params.tma_store_a.get_tma_descriptor());
    }

    typename PipelineLoad::Params lp;
    lp.role = (role == kLoad) ? PipelineLoad::ThreadCategory::Producer
                              : PipelineLoad::ThreadCategory::Consumer;
    lp.is_leader = (role == kLoad) && lane_predicate &&
                   ((block_rank_in_cluster % size(typename TiledMma::AtomThrID{})) == 0);
    // ONE wide B box (N=kMmaN). CollectiveMma::TmaTransactionBytes already counts X + ONE kMmaN-B box,
    // whose bytes == V1's X + gate-B + up-B (kMmaN == TileN+TileN). NO manual second-box add.  //<<V2
    lp.transaction_bytes = Config::TmaTransactionBytes;
    lp.num_consumers = cutlass::NumThreadsPerWarp;
    lp.initializing_warp = 0;
    PipelineLoad pipeline_load(ss.pipelines.load, lp, ClusterShape{}, cute::true_type{},
                               cute::false_type{});

    typename PipelineEpi::Params ep;
    ep.role = (role == kMMA) ? PipelineEpi::ThreadCategory::Producer
                             : PipelineEpi::ThreadCategory::Consumer;
    ep.producer_arv_count = 1;
    ep.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpiWarps * cutlass::NumThreadsPerWarp;
    ep.initializing_warp = 1;
    PipelineEpi pipeline_epi(ss.pipelines.epi, ep, ClusterShape{}, cute::true_type{},
                             cute::false_type{});

    TmemAllocator tmem_allocator{};
    pipeline_load.init_masks(ClusterShape{});
    pipeline_epi.init_masks(ClusterShape{});
    cutlass::arch::fence_barrier_init();
    cute::cluster_sync();

    typename PipelineLoad::PipelineState load_prod =
        cutlass::make_producer_start_state<PipelineLoad>();
    typename PipelineLoad::PipelineState load_cons;
    typename PipelineEpi::PipelineState epi_prod = cutlass::make_producer_start_state<PipelineEpi>();
    typename PipelineEpi::PipelineState epi_cons;

    const int M = params.M, N = params.N, K = params.K;
    const int k_tile_count = K / size<2>(TileShape{});
    const bool is_mma_leader_cta =
        (block_rank_in_cluster % size(typename TiledMma::AtomThrID{})) == 0;

    constexpr int kTileM = size<0>(take<0, 2>(TileShape{}));
    constexpr int kClusterX = size<0>(ClusterShape{});
    const int cluster_id = blockIdx.x / kClusterX;

    const int mtiles_per_expert = params.Me / kTileM;
    const int nI_per_expert = N / kOutN;          //<<V2 output n-tiles per expert (== n_local_tiles)
    // ONE wide W1 n-tile per OUTPUT tile: (2*I)/kMmaN == I/kOutN == nI_per_expert.  //<<V2
    const int n2w_per_expert = (2 * N) / kMmaN;   //<<V2 == nI_per_expert

    cutlass::arch::NamedBarrier tmem_alloc_bar(
        (1 + NumEpiWarps) * cutlass::NumThreadsPerWarp,
        cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

    // =========================================================================================
    // LOAD warp — ONE X TMA + ONE wide W1 TMA per K-tile.  //<<V2 (drop the 2nd W1 partition/copy)
    // =========================================================================================
    if (role == kLoad) {
      cutlass::arch::warpgroup_reg_dealloc<40>();

      Tensor mX = params.tma_load_x.get_tma_tensor(make_shape(M, K, 1));
      Tensor mW = params.tma_load_w1.get_tma_tensor(make_shape(params.W1N, K, 1));
      Tensor gX = local_tile(mX, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
      Tensor gW = local_tile(mW, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});  // BN==kMmaN

      ThrMMA cta_mma =
          TiledMma{}.get_slice(block_rank_in_cluster % size(typename TiledMma::AtomThrID{}));
      Tensor tCgX = cta_mma.partition_A(gX);
      Tensor tCgW = cta_mma.partition_B(gW);  // (MMA,MMA_N=kMmaN,MMA_K,n,k,l)

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
      Tensor sWil = make_tensor(make_smem_ptr(ss.tensors.smem_w1_il.begin()), SmemLayoutW1{}); //<<V2

      Layout cta_layout_mnk = make_layout(ClusterShape{});
      Layout cta_layout_vmnk =
          tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
      auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

      auto [tXgX, tXsX] = tma_partition(params.tma_load_x, get<2>(cta_coord_vmnk),
                                        make_layout(size<2>(cta_layout_vmnk)),
                                        group_modes<0, 3>(sX), group_modes<0, 3>(tCgX));
      auto [tWgW, tWsW] = tma_partition(params.tma_load_w1, get<1>(cta_coord_vmnk),  //<<V2 ONE part.
                                        make_layout(size<1>(cta_layout_vmnk)),
                                        group_modes<0, 3>(sWil), group_modes<0, 3>(tCgW));

      uint16_t mcast_mask_x = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
      uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;
        const int n_tile_local = tile % params.n_local_tiles;
        const int e = (params.m_tile_expert != nullptr) ? params.m_tile_expert[m_tile]
                                                        : (m_tile / mtiles_per_expert);
        // ONE wide W1 n-tile = expert e's interleaved gate||up block for THIS output tile.  //<<V2
        // (host-permuted: rows [e*2I + n_tile_local*kMmaN, +kMmaN) hold gate||up gran-G interleaved)
        const int w1_il_ntile = e * n2w_per_expert + n_tile_local;

        Tensor tXgX_k = tXgX(_, m_tile, _, _0{});
        Tensor tWgW_k = tWgW(_, w1_il_ntile, _, _0{});  //<<V2 ONE slice (V2a host-permuted path)

#ifdef SWIGLU_V2_5D_TMA
        // V2b: raw 5D coords into the CONTIGUOUS W1 (no host permute). The 5D desc box covers
        // (k:K_box, w8:8, gu:2, group:LOAD_BN/16, expert:1). For THIS output tile:
        //   c3 (group base) = n_tile_local * (kMmaN/16)   c4 (expert) = e
        //   c1=w8 origin=0   c2=gu origin=0               c0=k*K_box (K-tile base, set in the k-loop)
        const int grp16 = params.load_bn_5d / 16;            // per-CTA box: (kMmaN/2)/16 under 2-SM
        // Each CTA loads its OWN N-half: full-tile group start + this CTA's half-offset. Without the
        // per-CTA offset both CTAs fetch the same full kMmaN tile → barrier byte mismatch → deadlock.
        const int c3_group_base =
            n_tile_local * (Config::kMmaN / 16) +
            (kIs2Sm ? (block_rank_in_cluster % int(cute::size(typename TiledMma::AtomThrID{}))) * grp16 : 0);
        const int c4_expert = e;
#endif

        for (int k = 0; k < k_tile_count; ++k) {
          pipeline_load.producer_acquire(load_prod);
          auto* bar = pipeline_load.producer_get_barrier(load_prod);
          int wr = load_prod.index();
          if (cute::elect_one_sync()) {
            copy(params.tma_load_x.with(*bar, mcast_mask_x), tXgX_k(_, k), tXsX(_, wr));
#ifdef SWIGLU_V2_5D_TMA
            // RAW 5D issue into the SAME smem_w1_il stage buffer. The cp.async.bulk.tensor.5d delivers
            // gate/up interleaved at gran-8 straight from contiguous HBM (Muon-safe). R1: the desc swizzle
            // (kSwizzleBytesW1) MUST equal the COMPILED SmemLayoutW1 swizzle or the smem tile mismatches.
            // K_box columns are moved per op; the K-tile origin is k*K_box along the inner (d) axis.
            const int K_box = (Config::kSwizzleBytesW1 != 0)
                                  ? (Config::kSwizzleBytesW1 / int(sizeof(Element)))
                                  : int(size<2>(TileShape{}));
            void* dst5d = static_cast<void*>(&(*tWsW(_, wr).data()));  // stage-buffer smem base
            uint64_t* mbar5d = reinterpret_cast<uint64_t*>(bar);  // producer barrier as raw mbar ptr
            uint64_t cache_hint = 0;  // CU_TENSOR_MAP_L2_PROMOTION baked in the desc; no per-op hint
            // (void)mcast_mask_b: the raw 5D ops take NO mask arg — the 2-SM variant sets the peer bit on
            // the mbar internally (Sm100MmaPeerBitMask). coords = (c0=k, c1=w8, c2=gu, c3=group, c4=e).
            (void)mcast_mask_b;
            if (kIs2Sm) {
              cute::SM100_TMA_2SM_LOAD_5D::copy(&params.w1_5d_desc, mbar5d, cache_hint, dst5d,
                                                k * K_box, 0, 0, c3_group_base, c4_expert);
            } else {
              cute::SM90_TMA_LOAD_5D::copy(&params.w1_5d_desc, mbar5d, cache_hint, dst5d, k * K_box, 0,
                                           0, c3_group_base, c4_expert);
            }
#else
            copy(params.tma_load_w1.with(*bar, mcast_mask_b), tWgW_k(_, k), tWsW(_, wr)); //<<V2 1 copy
#endif
          }
          ++load_prod;
        }
      }
      pipeline_load.producer_tail(load_prod);
    }

    // =========================================================================================
    // MMA warp — ONE acc of kMmaN cols, ONE cute::gemm at N=kMmaN.  //<<V2 (drop acc_up + 2nd gemm)
    // =========================================================================================
    else if (role == kMMA) {
      cutlass::arch::warpgroup_reg_dealloc<40>();

      tmem_allocator.allocate(kTmemCols, &ss.tmem_base_ptr);
      __syncwarp();
      tmem_alloc_bar.arrive();

      TiledMma tiled_mma;
      Tensor acc_il = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));  //<<V2 (MMA,M,N=kMmaN)

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
      Tensor sWil = make_tensor(make_smem_ptr(ss.tensors.smem_w1_il.begin()), SmemLayoutW1{}); //<<V2
      Tensor tCrX = TiledMma::make_fragment_A(sX);
      Tensor tCrW = TiledMma::make_fragment_B(sWil);  //<<V2 ONE B fragment at N=kMmaN

      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        if (is_mma_leader_cta) {
          pipeline_epi.producer_acquire(epi_prod);
          acc_il.data() = ss.tmem_base_ptr + epi_prod.index() * uint32_t(kAccBufStride);  //<<V2
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
          for (int k = 0; k < k_tile_count; ++k) {
            pipeline_load.consumer_wait(load_cons);
            int rs = load_cons.index();
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < size<2>(tCrX); ++kb) {
              cute::gemm(tiled_mma, tCrX(_, _, kb, rs), tCrW(_, _, kb, rs), acc_il);  //<<V2 SINGLE gemm
              tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            pipeline_load.consumer_release(load_cons);
            ++load_cons;
          }
          pipeline_epi.producer_commit(epi_prod);
          ++epi_prod;
        }
      }
    }

    // =========================================================================================
    // EPILOGUE — ONE TMEM-load of the kMmaN-col acc into rAll, then gran-G de-interleave.  //<<V2
    // =========================================================================================
    else if (role == kEpilogue) {
#ifndef EPI_REGS
#define EPI_REGS 160
#endif
      cutlass::arch::warpgroup_reg_alloc<EPI_REGS>();
      tmem_alloc_bar.arrive_and_wait();

      TiledMma tiled_mma;
      // ONE per-CTA acc fragment at N=kMmaN. (same V-slice semantics as V1; M-half via the coord tile)
      Tensor tAcc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));  //<<V2 (MMA,M,N=kMmaN)
      Tensor tAcc_il = tAcc;
      tAcc_il.data() = ss.tmem_base_ptr + 0u;  // placeholder for layout; per-tile .data() set in loop

      ThrMMA cta_mma_epi = tiled_mma.get_slice(0);
      Tensor cAcc = make_identity_tensor(take<0, 2>(TileShape{}));   // (kTileM, kMmaN) coords
      Tensor cAcc_cta = cta_mma_epi.partition_C(cAcc);

      using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;  // SAME atom as V1 (swiglu.cuh:863)
      int thread_idx = threadIdx.x % (NumEpiWarps * cutlass::NumThreadsPerWarp);
      auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tAcc_il);
      auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
      Tensor tTMc = thr_tmem_load.partition_D(cAcc_cta);   // (T2R,T2R_M,T2R_N) global (row,col) DST order

      Tensor rAll = make_tensor<ElementAcc>(shape(tTMc));  //<<V2 ONE register tensor (kMmaN cols worth)

      Tensor sA = make_tensor(make_smem_ptr(ss.tensors.smem_out.begin()), SmemLayoutA{});
      cutlass::epilogue::thread::SiLu<ElementAcc> silu{};
      Tensor mA_tma = params.tma_store_a.get_tma_tensor(make_shape(M, N));
      ThrCopy thrblk_s2g = params.tma_store_a.get_slice(Int<0>{});
      Tensor bSG_sA = thrblk_s2g.partition_S(sA);
      const bool is_tma_store_lane = (warp_idx == 4) && (lane_predicate != 0u);

      constexpr int G = kGluGran;
      const int nElem = size(rAll);

      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;
        const int n_tile_local = tile % params.n_local_tiles;
        const int cta_row_offset =
            m_tile * kTileM +
            int(block_rank_in_cluster) * (kTileM / int(size(typename TiledMma::AtomThrID{})));
        const int cta_col_offset = n_tile_local * kOutN;   //<<V2 OUTPUT col base (== V1, uses kOutN)

        pipeline_epi.consumer_wait(epi_cons);
        cutlass::arch::fence_view_async_tmem_store();

        // ONE acc buffer of kMmaN cols; ONE partition_S; ONE TMEM-load.  //<<V2
        const uint32_t epi_buf = epi_cons.index() * uint32_t(kAccBufStride);
        tAcc_il.data() = ss.tmem_base_ptr + epi_buf;
        Tensor tTM_il = thr_tmem_load.partition_S(tAcc_il);
        copy(tiled_tmem_load, tTM_il, rAll);

#ifdef SWIGLU_DEBUG_PRINT
        if (tile == cluster_id && thread_idx == 0 && block_rank_in_cluster == 0) {
          for (int i = 0; i < nElem && i < 32; ++i)
            printf("V2 T0 i=%2d row=%3d col=%3d acc=% .4f\n", i, get<0>(tTMc(i)),
                   get<1>(tTMc(i)), float(rAll(i)));
        }
#endif

        // WAR drain of previous tile's TMA store (IDENTICAL to V1 swiglu.cuh:996-1000).
        if (is_tma_store_lane) cute::tma_store_wait<0>();
        cutlass::arch::NamedBarrier epi_war_bar(NumEpiThreads, /*id=*/1u);
        epi_war_bar.arrive_and_wait();

        // ---- gran-G DE-INTERLEAVE (R3 core) ------------------------------------------------
        // For each register element that is a GATE column (in [0,kMmaN), block-parity 0), find the UP
        // partner in THIS thread's register file: SAME local row, col == gate_col + G. Both are present
        // in rAll because a 32dp32b32x lane owns a contiguous 2G-aligned column run (R3). The search is
        // over this thread's own elements (no shuffle). out_col collapses the 2G-block to a G-block.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < nElem; ++i) {
          const int lrow = get<0>(tTMc(i));   // LOCAL row in [0,kEpiTileM)
          const int lc = get<1>(tTMc(i));     // LOCAL acc col in [0,kMmaN)
          if (((lc / G) & 1) != 0) continue;  // skip UP cols; handled via their gate partner
          const int up_lc = (lc / (2 * G)) * (2 * G) + G + (lc % G);  // == v2_up_acc_col(out,G)
          // find UP partner reg: same row, col==up_lc. (contiguous datapath -> typically i + dN.)
          ElementAcc up = ElementAcc(0);
          bool found = false;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < nElem; ++j) {
            if (get<0>(tTMc(j)) == lrow && get<1>(tTMc(j)) == up_lc) {
              up = rAll(j);
              found = true;
              break;
            }
          }
          // R3 FALLBACK: if the partner is NOT in this thread (found==false), the pair is cross-lane
          // for this G -> the build is INCORRECT for this G (the gran sweep will report n_fail>0).
          // Use SWIGLU_V2_TWO_ACC (compile gate) to revert to V1's epilogue if every G fails.
          ElementAcc act = found ? (silu(rAll(i)) * up) : ElementAcc(1.0e30f);  // huge sentinel: co-location FAIL screams (NaN slips past abs/rel tol)
          if (params.prob != nullptr) {
            const int grow = lrow + cta_row_offset;
            if (grow < params.M) act *= params.prob[grow];
          }
          const int out_c = (lc / (2 * G)) * G + (lc % G);  // de-interleaved OUTPUT col in [0,kOutN)
          sA(lrow, out_c) = static_cast<ElementOut>(act);
        }

        cutlass::arch::fence_view_async_tmem_load();
        pipeline_epi.consumer_release(epi_cons);
        ++epi_cons;

        // sA visibility + ASYNC TMA bulk-store — BYTE-IDENTICAL to V1 swiglu.cuh:1049-1073.
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier epi_smem_bar(NumEpiThreads, /*id=*/0u);
        epi_smem_bar.arrive_and_wait();
        if (is_tma_store_lane) {
          const int box_m = cta_row_offset / kEpiTileM;
          const int box_n = cta_col_offset / kOutN;  // == n_tile_local
          Tensor gA = local_tile(mA_tma, make_shape(Int<kEpiTileM>{}, Int<kOutN>{}),
                                 make_coord(box_m, box_n));
          Tensor bSG_gA = thrblk_s2g.partition_D(gA);
          copy(params.tma_store_a, bSG_sA, bSG_gA);
          cute::tma_store_arrive();
        }
      }
      if (is_tma_store_lane) cute::tma_store_wait<0>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<40>();
    }

    cute::cluster_sync();
    if (role == kMMA) {
      tmem_allocator.release_allocation_lock();
      tmem_allocator.free(ss.tmem_base_ptr, kTmemCols);
    }
#endif
  }
};

// ============================================================================================
// §V2.4 · LaunchSwiGluGroupedV2 — mirror of LaunchSwiGluGrouped (swiglu.cuh:1115-1298). Only deltas:
//   - W1 is expected HOST-PERMUTED to gran-G interleaved per output-tile (V2a). W1N == G*2I unchanged.
//   - the W1 TMA box-N is kMmaN (induced by Config::TileShape), so ONE descriptor covers it.
//   - num_n_local_tiles uses kOutN (NOT kMmaN). Persistent occupancy launch identical to V1.
// ============================================================================================
template <typename Element, typename ElementOut, int TileM_ = 256, int TileN_ = 64, int TileK_ = 32,
          int kStages_ = 8, int ClusterM_ = 2, int MinBlocks_ = 1, int AccStages_ = 2>
cudaError_t LaunchSwiGluGroupedV2(
    const Element* X, const Element* W1_il /*[G*2I,d] gran-G interleaved per output-tile*/,
    ElementOut* A, int G, int Me, int I, int d, cudaStream_t stream, int device = 0, int sm_count = 0,
    const int* d_m_tile_expert = nullptr, int M_varlen = 0,
    const int* d_m_gather_idx = nullptr, int T_src = 0, const float* d_prob = nullptr) {
  using Config = SwiGluConfigV2<Element, ElementOut, TileM_, TileN_, TileK_, kStages_, ClusterM_,
                                MinBlocks_, AccStages_>;
  using Kernel = Sm100SwiGluKernelV2<Config>;
  using CollectiveMma = typename Config::CollectiveMma;
  using StrideX = typename Config::StrideX;
  using StrideW1 = typename Config::StrideW1;
  using StrideA = typename Config::StrideA;

  const int M = (d_m_tile_expert != nullptr) ? M_varlen : (G * Me);
  const int W1N = G * 2 * I;
  StrideX dX = cutlass::make_cute_packed_stride(StrideX{}, cute::make_shape(M, d, 1));
  StrideW1 dW1 = cutlass::make_cute_packed_stride(StrideW1{}, cute::make_shape(W1N, d, 1));
  StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, I, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = sm_count;

  auto problem = cute::make_shape(M, W1N, d);   // TMA_B box-N = kMmaN comes from Config::TileShape
  typename CollectiveMma::Arguments mainloop_args{};
  mainloop_args.ptr_A = X;
  mainloop_args.dA = dX;
  mainloop_args.ptr_B = W1_il;
  mainloop_args.dB = dW1;
  typename CollectiveMma::Params mainloop_params =
      CollectiveMma::to_underlying_arguments(problem, mainloop_args, nullptr, hw_info);

  cute::Tensor mA_store = cute::make_tensor(
      cute::make_gmem_ptr(A),
      cute::make_layout(cute::make_shape(M, I), cute::make_stride(I, cute::_1{})));
  auto tma_store_a = make_tma_copy(cute::SM90_TMA_STORE{}, mA_store, typename Config::SmemLayoutA{});

  typename Kernel::Params params;
  params.tma_load_x = mainloop_params.tma_load_a;
  params.tma_load_w1 = mainloop_params.tma_load_b;   // V2a: N=kMmaN interleaved B descriptor
  // V2b 5D box-N = PER-CTA N width (kMmaN/2 under 2-SM). Each CTA must load ONLY its N-half so the
  // delivered bytes match the per-CTA SmemLayoutW1 + per-CTA transaction_bytes. Using the full kMmaN made
  // both CTAs duplicate the whole tile → CTA0 barrier got 2x bytes → mbarrier never completes → deadlock.
  const int per_cta_n = Config::kMmaN / int(cute::size(typename Config::AtomThrShapeMNK{}));
  params.load_bn_5d = per_cta_n;
#ifdef SWIGLU_V2_5D_TMA
  // V2b: build the raw 5D CUtensorMap over the CONTIGUOUS, un-permuted W1 (Muon-safe). W1_il here is the
  // CONTIGUOUS [G*2I, d] (NOT host-permuted) when this path is on; gate/up are interleaved at gran-8 by
  // the 5D box geometry, not by a host permute. The swizzle MUST match Config::kSwizzleBytesW1 (R1).
  {
    constexpr int TileK = TileK_;
    CUresult r = make_w1_gran8_5d_desc<Element>(&params.w1_5d_desc, W1_il, G, I, d, TileK,
                                                /*LOAD_BN=*/per_cta_n, Config::kSwizzleBytesW1);
    if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
  }
#endif
  params.tma_store_a = tma_store_a;
  params.ptr_A = A;
  params.dA = dA;
  params.M = M;
  params.N = I;
  params.K = d;
  params.Me = Me;
  params.W1N = W1N;
  params.m_tile_expert = d_m_tile_expert;
  params.m_gather_idx = nullptr;                     // V2: gather not wired
  params.tma_load_x_gather = mainloop_params.tma_load_a;
  params.T_src_gather = M;
  params.prob = d_prob;

  constexpr int kTileM = cute::size<0>(typename Config::TileShape{});
  constexpr int kOutN = Config::kOutN;
  int num_m_tiles = (M + kTileM - 1) / kTileM;
  int num_n_local_tiles = (I + kOutN - 1) / kOutN;   //<<V2 kOutN, NOT kMmaN
  int total_tiles = num_m_tiles * num_n_local_tiles;

  dim3 cluster(cute::size<0>(typename Config::ClusterShape{}),
               cute::size<1>(typename Config::ClusterShape{}),
               cute::size<2>(typename Config::ClusterShape{}));
  dim3 block(Kernel::MaxThreadsPerBlock, 1, 1);
  int smem_size = Kernel::SharedStorageSize;
  void const* kernel_ptr = reinterpret_cast<void const*>(cutlass::device_kernel<Kernel>);

  if (smem_size >= (48 << 10)) {
    cudaError_t attr =
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr != cudaSuccess) return attr;
  }
  cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  int cluster_size = int(cluster.x);
  int persistent_clusters = 0;
  {
    auto occ_cfg = cutlass::ClusterLauncher::make_cluster_launch_config(cluster, cluster, block,
                                                                        smem_size, stream);
    int max_active_clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&max_active_clusters, kernel_ptr, &occ_cfg.launch_config) ==
            cudaSuccess &&
        max_active_clusters > 0)
      persistent_clusters = max_active_clusters;
  }
  if (persistent_clusters < 1) {
    int sm_count_eff = (sm_count > 0)
                           ? sm_count
                           : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    persistent_clusters = sm_count_eff / cluster_size;
  }
  if (persistent_clusters < 1) persistent_clusters = 1;
  int num_clusters = persistent_clusters < total_tiles ? persistent_clusters : total_tiles;

  params.total_tiles = total_tiles;
  params.n_local_tiles = num_n_local_tiles;
  params.num_clusters = num_clusters;

  dim3 grid(num_clusters * cluster.x, 1, 1);
  auto cfg =
      cutlass::ClusterLauncher::make_cluster_launch_config(grid, cluster, block, smem_size, stream);
  void* kernel_params[] = {&params};
  return cudaLaunchKernelExC(&cfg.launch_config, kernel_ptr, kernel_params);
}

}  // namespace grouped_gemm_swiglu
}  // namespace transformer_engine
