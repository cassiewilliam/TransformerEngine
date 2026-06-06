/***************************************************************************************************
 * MoE DOWN-projection grouped GEMM — V2 (SM100 tcgen05, deep-pipeline warp-specialized).
 *
 *   per expert e:  Y_e[Me, N] = X_e[Me, K] @ W_e[N, K]^T        (row-major X and Y; W [N,K] used Tᵀ)
 *   Real shape:    K = I = 512 (contraction, SHORT), N = H = 2048 (output width), M grouped
 *                  (sum of per-expert Me; e.g. Me=768, G=32 -> M=24576). bf16 in/out, fp32 accum.
 *
 * WHAT THIS IS: the V2/V1 SwiGLU up-proj grouped GEMM machinery (cutlass_grouped_gemm_swiglu.cuh /
 *   _v2.cuh) MINUS the SwiGLU specifics. The down-proj is a PLAIN grouped GEMM — NO gate/up, NO
 *   interleave, NO silu, NO per-token prob. We reuse V1's PROVEN deep-pipeline warp-specialized
 *   design (TMA-load X + W, tcgen05 2-SM MMA, multi-stage PipelineTmaUmmaAsync load +
 *   PipelineUmmaAsync epilogue, persistent grid-stride scheduler over m-tiles with a per-m-tile
 *   expert-id table, deep kStages, TMA-store epilogue), which beat the generic CUTLASS
 *   CollectiveBuilder grouped GEMM ~2.1x on the up-proj (1141 vs ~550 TFLOP/s). The generic builder
 *   is tile-insensitive at K=512 (~699 TFLOP/s); this custom deep-pipeline kernel should beat it.
 *
 * HOW IT DIFFERS FROM V1 (cutlass_grouped_gemm_swiglu.cuh) — the exact strip-down:
 *   1. NO interleave / NO 5D descriptor. W_e is a plain [N,K] block; ONE plain 2D TMA (the SAME
 *      machinery V1 uses for ONE of its two W tiles) into ONE smem buffer. Dropped:
 *      make_w1_gran8_5d_desc, GLU_G, v2_gate_acc_col / v2_up_acc_col, swiglu_pair.
 *   2. kMmaN = TileN (NOT 2*TileN). The up-proj doubled MMA-N to pack gate||up; the down has a
 *      single output, so MMA-N == output-N == TileN. acc is acc[TileM, TileN]; TMEM cols =
 *      AccStages * TileN (NOT AccStages*2*TileN). kAccBufStride = TileN.
 *   3. Plain epilogue: NO silu, NO multiply, NO de-interleave, NO prob. ONE TMEM-load of the
 *      TileN-col acc, fp32 -> ElementOut (bf16) convert, then the SAME V1 TMA-store epilogue to
 *      Y[M,N] (SM100_TMEM_LOAD_32dp32b32x + cp.async.bulk.tensor store + WAR drain).
 *   4. EVERYTHING else identical to V1: 2-SM cluster (ClusterM=2), warp specialization
 *      (Load/MMA/Epilogue), PipelineTmaUmmaAsync/PipelineUmmaAsync, persistent grid-stride
 *      scheduler over m-tiles (varlen-M via m_tile_expert), TileK/kStages/AccStages knobs,
 *      SmemLayoutX/SmemLayoutW swizzle, static_assert(SharedStorageSize <= 228*1024).
 *   5. Grouped/varlen-M plumbing IDENTICAL to V1, but the per-expert W base is e*N*K (a [N,K]
 *      block, NOT e*2I*K) and Y is written at [m, :N].
 *
 * DIM / TRANS MAPPING (CUTLASS gemm m/n/k vs our shapes):
 *   CUTLASS-A = X[M,K] RowMajor (LayoutX = RowMajor)   -> m = M (tokens), k = K = I = 512
 *   CUTLASS-B = W[N,K] used Tᵀ  (LayoutW = ColumnMajor) -> n = N = H = 2048, k = K (K-contiguous W)
 *   CUTLASS-C/D = Y[M,N] RowMajor (LayoutY = RowMajor)  -> m = M, n = N
 *   This is the SAME m/n/k/trans mapping V1 uses (X is A, W is B-with-ColumnMajor-tag for
 *   K-contiguous physical weights). ONLY N changes meaning: V1's N = I (per gate/up tile, doubled
 *   to 2I in W); here N = H (full output width), and the per-expert W block is exactly N rows.
 *
 * STATUS: FIRST DRAFT, NOT COMPILED (no GPU available). Must be compile-iterated on a SM100 box.
 *   See the deliverable report for the RANKED list of parts most likely to need iteration.
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass_grouped_gemm_swiglu.cuh"  // V1 machinery: includes, pipeline types, TMEM atoms,
                                            // ClusterLauncher, SiLu(unused), make_cute_packed_stride

namespace transformer_engine {
namespace grouped_gemm_down {

using namespace cute;

// ============================================================================================
// DownGemmConfigV2 — derived from the V1 SwiGluConfig (cutlass_grouped_gemm_swiglu.cuh) with:
//   * kMmaN = TileN  (NOT 2*TileN): single output, single GEMM, single acc.   <<DOWN
//   * NO GLU_G / kGluGran / kInterleaveN.                                       <<DOWN
//   * LayoutW = ColumnMajor (K-contiguous weights, == V1 LayoutW1).
//   * Output Y[M,N] RowMajor; N == H (full output width).
//   * TMEM cols = AccStages * TileN (NOT *2*TileN); kAccBufStride = TileN.      <<DOWN
// Default tile knobs match the up-proj tuned optimum (TileM=256, TileK=16, kStages=16, AccStages=2)
// EXCEPT TileN: the prompt's signature default is TileN=64. Note: at AccStages=2, TileN<=256 keeps
// kTmemCols = AccStages*TileN <= 512 (and pow2 for TileN in {64,128,256}); see the static_assert.
// ============================================================================================
template <typename Element_ /*bf16/fp16*/, typename ElementOut_ /*bf16/fp16*/, int TileM_ = 256,
          int TileN_ = 64, int TileK_ = 64, int kStages_ = 8, int ClusterM_ = 2, int MinBlocks_ = 1,
          int AccStages_ = 2>
struct DownGemmConfigV2 {
  static_assert(ClusterM_ == 1 || ClusterM_ == 2, "ClusterM_ must be 1 (1-SM) or 2 (2-SM)");

  using Element = Element_;   // X and W element
  using ElementAcc = float;   // fp32 accumulate (required)
  using ElementOut = ElementOut_;

  using ArchTag = cutlass::arch::Sm100;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using LayoutX = cutlass::layout::RowMajor;     // A operand (M,K)=(tokens,K): RowMajor => K-contiguous
  using LayoutW = cutlass::layout::ColumnMajor;  // B operand (N,K)=(H,K): ColumnMajor tag => K-contiguous
                                                 // physical weights W[n,k] at n*K+k (== V1 LayoutW1).
  using LayoutY = cutlass::layout::RowMajor;     // output Y[M,N]

  static constexpr int AlignX = 128 / cutlass::sizeof_bits<Element>::value;     // 8 (bf16)
  static constexpr int AlignW = 128 / cutlass::sizeof_bits<Element>::value;     // 8
  static constexpr int AlignY = 128 / cutlass::sizeof_bits<ElementOut>::value;  // 8

  // ---- N: single output, MMA-N == output-N == TileN (NO doubling). ----------------------------
  static constexpr int kOutN = TileN_;  // OUTPUT n-tile (scheduler/store/epi-out) == MMA n-tile
  static constexpr int kMmaN = TileN_;  // <<DOWN single-output MMA width (V2 had 2*TileN)
  // ATOM limit (sm100_common.inl:379): N%8==0 && N<=256 for the 2-SM bf16 atom.
  static_assert(kMmaN % 8 == 0 && kMmaN <= 256, "kMmaN=TileN must be %8 and <=256 (2-SM bf16 atom)");

  using TileShape = cute::Shape<cute::Int<TileM_>, cute::Int<kMmaN>, cute::Int<TileK_>>;
  using ClusterShape = cute::Shape<cute::Int<ClusterM_>, cute::_1, cute::_1>;
  using KernelSchedule =
      std::conditional_t<ClusterM_ == 2, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

  static constexpr int kStages = kStages_;
  static constexpr int kMinBlocks = MinBlocks_;
  static constexpr int kAccStages = AccStages_;

  // CollectiveMma type-provider (TiledMma / SmemLayout / fragments / TMA atoms / TransactionBytes).
  // Built at TileShape N = kMmaN = TileN (a SINGLE B tile, NOT the V2 doubled gate||up tile). <<DOWN
  using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, Element, LayoutX, AlignX, Element, LayoutW, AlignW, ElementAcc, TileShape,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>, KernelSchedule>::CollectiveOp;

  using TiledMma = typename CollectiveMma::TiledMma;
  using AtomThrShapeMNK = typename CollectiveMma::AtomThrShapeMNK;  // (2,1,1) under 2-SM
  using SmemLayoutX = typename CollectiveMma::SmemLayoutA;          // (MMA,M,K,PIPE)
  using SmemLayoutW = typename CollectiveMma::SmemLayoutB;          // (MMA,N=TileN,K,PIPE) — ONE B tile
  static constexpr int Stages = CollectiveMma::DispatchPolicy::Stages;
  using CtaShapeMNK = typename CollectiveMma::CtaShape_MNK;

  using StrideX = typename CollectiveMma::StrideA;
  using StrideW = typename CollectiveMma::StrideB;
  using StrideY = cutlass::detail::TagToStrideC_t<LayoutY>;
  using MainloopArguments = typename CollectiveMma::Arguments;
  using MainloopParams = typename CollectiveMma::Params;
  using TMA_X = typename MainloopParams::TMA_A;
  using TMA_W = typename MainloopParams::TMA_B;  // plain 2D TMA at N=TileN over W[G*N, K]
  static constexpr uint32_t TmaTransactionBytes = CollectiveMma::TmaTransactionBytes;

  // ---- OUTPUT-side derived from kOutN (== kMmaN here). Byte-identical to V1 swiglu.cuh:215-233. ----
  static constexpr int kEpiTileM_cfg =
      cute::size<0>(cute::take<0, 2>(TileShape{})) / int(cute::size(AtomThrShapeMNK{}));  // 128
  static constexpr int kEpiTileN_cfg = kOutN;                                             // TileN
  using SmemLayoutY = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kEpiTileM_cfg>{}, cute::Int<kEpiTileN_cfg>{}),
      cute::make_stride(cute::Int<kEpiTileN_cfg>{}, cute::_1{})));
  using TMA_Y = decltype(make_tma_copy(
      cute::SM90_TMA_STORE{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOut*>(nullptr)),
                        cute::make_layout(cute::make_shape(int(0), int(0)),
                                          cute::make_stride(int(0), cute::_1{}))),
      SmemLayoutY{}));

  // ---- TMEM accounting: ONE acc of kMmaN(=TileN) cols per stage (V2 had 2*TileN). -------- <<DOWN
  static constexpr int kAccBufStride = kMmaN;           // == TileN  (V2: 2*TileN)
  static constexpr int kTmemCols = kAccStages * kMmaN;  // == AccStages*TileN  (V2: AccStages*2*TileN)
  static_assert(kTmemCols >= 32 && kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0,
                "kTmemCols (= AccStages*TileN) must be a pow2 in [32,512]; "
                "for AccStages=2 keep TileN in {16,32,64,128,256} so AccStages*TileN is pow2 <=512.");
};

// ============================================================================================
// Sm100DownGemmKernelV2 — plain grouped GEMM kernel. Near-verbatim copy of Sm100SwiGluKernel
// (cutlass_grouped_gemm_swiglu.cuh:254-1097) with EXACTLY these deltas (each marked //<<DOWN):
//   SharedStorage:  ONE smem_w (drop smem_w1_gate / smem_w1_up).
//   Load warp:      ONE W TMA partition + ONE copy (drop the up partition/copy + gather path).
//   MMA warp:       ONE acc of kMmaN(=TileN) cols + ONE cute::gemm (drop acc_up + 2nd gemm).
//   Epilogue:       ONE TMEM-load -> fp32->bf16 convert -> store (drop silu/mul/de-interleave/prob).
//   Per-expert W base: e*nNtiles_per_expert (a [N,K] block), W picks rows [e*N + n_tile*TileN, ...).
// Everything else (pipelines, persistent scheduler, TMEM alloc/free, TMA-store tail) is V1-identical.
// ============================================================================================
template <typename Config>
struct Sm100DownGemmKernelV2 {
  using Element = typename Config::Element;
  using ElementAcc = typename Config::ElementAcc;
  using ElementOut = typename Config::ElementOut;
  using TileShape = typename Config::TileShape;  // N == kMmaN == TileN
  using ClusterShape = typename Config::ClusterShape;
  using TiledMma = typename Config::TiledMma;
  using AtomThrShapeMNK = typename Config::AtomThrShapeMNK;
  using SmemLayoutX = typename Config::SmemLayoutX;
  using SmemLayoutW = typename Config::SmemLayoutW;  // N == TileN  //<<DOWN single B tile
  using StrideX = typename Config::StrideX;
  using StrideW = typename Config::StrideW;
  using StrideY = typename Config::StrideY;
  using TMA_X = typename Config::TMA_X;
  using TMA_W = typename Config::TMA_W;
  using TMA_Y = typename Config::TMA_Y;
  using SmemLayoutY = typename Config::SmemLayoutY;
  static constexpr int Stages = Config::Stages;

  static constexpr int kMmaN = Config::kMmaN;  // acc width == TileN  //<<DOWN
  static constexpr int kOutN = Config::kOutN;  // output width == TileN

  using ArchTag = cutlass::arch::Sm100;
  static constexpr bool kIs2Sm = (cute::size(AtomThrShapeMNK{}) == 2);
  using TmemAllocator =
      std::conditional_t<kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  using PipelineLoad = cutlass::PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShapeMNK>;
  using PipelineEpi = cutlass::PipelineUmmaAsync<Config::kAccStages, AtomThrShapeMNK>;

  static constexpr int NumEpiWarps = 4;
  static constexpr int NumWarps = 8;  // 256 threads
  static constexpr int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  static constexpr int MinBlocksPerMultiprocessor = Config::kMinBlocks;
  static constexpr int NumEpiThreads = NumEpiWarps * cutlass::NumThreadsPerWarp;  // 128

  // OUTPUT staging tile = kEpiTileM x kOutN (== V1; kOutN == TileN == kMmaN here).
  static constexpr int kEpiTileM =
      cute::size<0>(take<0, 2>(TileShape{})) / int(cute::size(AtomThrShapeMNK{}));  // 128
  static constexpr int kEpiTileN = kOutN;                                           // TileN  //<<DOWN
  static_assert(kEpiTileM == Config::kEpiTileM_cfg && kEpiTileN == Config::kEpiTileN_cfg,
                "Config TMA-store tile extents must match Kernel kEpiTileM/kEpiTileN");
  // ONE acc / stage of kMmaN(=TileN) cols (V1 had 2*kEpiTileN for gate+up).  //<<DOWN
  static constexpr int kAccBufStride = kMmaN;                   // == TileN  (V1: 2*kEpiTileN)
  static constexpr int kTmemCols = Config::kAccStages * kMmaN;  // == AccStages*TileN  //<<DOWN
  static_assert(kTmemCols >= 32 && kTmemCols <= 512 && (kTmemCols & (kTmemCols - 1)) == 0,
                "kTmemCols must be a pow2 in [32,512]");

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
      cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutW>> smem_w;  //<<DOWN ONE B (drop gate/up)
      cute::ArrayEngine<ElementOut, kEpiTileM * kOutN> smem_out;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _0> {
      alignas(16) typename PipelineLoad::SharedStorage load;
      alignas(16) typename PipelineEpi::SharedStorage epi;
    } pipelines;
    uint32_t tmem_base_ptr;
  };
  static constexpr int SharedStorageSize = static_cast<int>(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= (228 * 1024),
                "Down-proj smem (X + W + pipelines) exceeds SM100 capacity; reduce kStages.");

  // -------------------------------------------------------------------------------------------
  // Device params.  GROUPED (contiguous, uniform/TileM-aligned Me): plain single pointers + ONE X
  // TMA over [M, K] + ONE W TMA over [G*N, K].  The per-expert W-block is selected in-kernel from
  // the m-tile (tokens contiguous by expert).  NO pointer arrays / tensormap swaps.
  //   M = sum(Me)  (total tokens)      N = H (output width)       K = I (contraction)
  //   WN = G*N     (rows of the W TMA descriptor)
  // -------------------------------------------------------------------------------------------
  struct Params {
    TMA_X tma_load_x;
    TMA_W tma_load_w;  // ONE descriptor over W[G*N, K]; the expert's [N,K] block is the per-tile slice
    TMA_Y tma_store_y;
    ElementOut* ptr_Y;
    StrideY dY;
    int M, N, K;  // M == total tokens, N == H (output width), K == I (contraction)
    int Me;       // tokens per expert (uniform fallback; unused when m_tile_expert != nullptr)
    int WN;       // == G*N, the N-extent of the single W TMA descriptor
    int total_tiles;    // num_m_tiles * num_n_local_tiles
    int n_local_tiles;  // == N/kOutN (decode stride for m_tile/n_tile)
    int num_clusters;   // persistent clusters launched (grid-stride step)
    const int* m_tile_expert;  // [num_m_tiles] global m-tile -> expert id; nullptr => uniform Me
  };

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && \
    !defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM103F_ENABLED)
    if (cute::thread0())
      printf("ERROR: Sm100DownGemmKernelV2 requires SM100a/f or SM103a/f. Compile -arch=sm_100a.\n");
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
      cute::prefetch_tma_descriptor(params.tma_load_w.get_tma_descriptor());
    }
    if (role == kEpilogue && lane_predicate && warp_idx == 4) {
      cute::prefetch_tma_descriptor(params.tma_store_y.get_tma_descriptor());
    }

    // ---- pipeline construction (identical to V1; ONE W box per stage) ----  //<<DOWN
    typename PipelineLoad::Params lp;
    lp.role = (role == kLoad) ? PipelineLoad::ThreadCategory::Producer
                              : PipelineLoad::ThreadCategory::Consumer;
    lp.is_leader = (role == kLoad) && lane_predicate &&
                   ((block_rank_in_cluster % size(typename TiledMma::AtomThrID{})) == 0);
    // ONE wide B box (N=TileN). CollectiveMma::TmaTransactionBytes already counts X + ONE TileN-B box.
    // NO manual second-box add (V1 added a 2nd B box for the up weight; the down has only one). //<<DOWN
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
    const int k_tile_count = K / size<2>(TileShape{});  // K / TileK
    const bool is_mma_leader_cta =
        (block_rank_in_cluster % size(typename TiledMma::AtomThrID{})) == 0;

    constexpr int kTileM = size<0>(take<0, 2>(TileShape{}));
    constexpr int kClusterX = size<0>(ClusterShape{});
    const int cluster_id = blockIdx.x / kClusterX;

    const int mtiles_per_expert = params.Me / kTileM;  // m-tiles per expert (uniform fallback)
    const int nN_per_expert = N / kOutN;               // output n-tiles per expert == n_local_tiles
    // ONE W n-tile per OUTPUT tile: the expert's [N,K] block is N/kOutN tiles of TileN.  //<<DOWN
    // (V1 had n2_per_expert = (2*I)/kTileN for the doubled gate||up W; here it is just N/kOutN.)
    const int nw_per_expert = N / kOutN;  //<<DOWN == nN_per_expert (single W block, NOT doubled)

    cutlass::arch::NamedBarrier tmem_alloc_bar(
        (1 + NumEpiWarps) * cutlass::NumThreadsPerWarp,
        cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

    // =========================================================================================
    // LOAD warp — ONE X TMA + ONE W TMA per K-tile.  //<<DOWN (drop the 2nd W partition/copy + gather)
    // =========================================================================================
    if (role == kLoad) {
      cutlass::arch::warpgroup_reg_dealloc<40>();

      Tensor mX = params.tma_load_x.get_tma_tensor(make_shape(M, K, 1));
      Tensor mW = params.tma_load_w.get_tma_tensor(make_shape(params.WN, K, 1));
      Tensor gX = local_tile(mX, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});  // (BM,BK,m,k,l)
      Tensor gW = local_tile(mW, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});  // (BN,BK,n,k,l)

      ThrMMA cta_mma =
          TiledMma{}.get_slice(block_rank_in_cluster % size(typename TiledMma::AtomThrID{}));
      Tensor tCgX = cta_mma.partition_A(gX);
      Tensor tCgW = cta_mma.partition_B(gW);  // (MMA,MMA_N=TileN,MMA_K,n,k,l)

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
      Tensor sW = make_tensor(make_smem_ptr(ss.tensors.smem_w.begin()), SmemLayoutW{});  //<<DOWN ONE B

      Layout cta_layout_mnk = make_layout(ClusterShape{});
      Layout cta_layout_vmnk =
          tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
      auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

      auto [tXgX, tXsX] = tma_partition(params.tma_load_x, get<2>(cta_coord_vmnk),
                                        make_layout(size<2>(cta_layout_vmnk)),
                                        group_modes<0, 3>(sX), group_modes<0, 3>(tCgX));
      auto [tWgW, tWsW] = tma_partition(params.tma_load_w, get<1>(cta_coord_vmnk),  //<<DOWN ONE part.
                                        make_layout(size<1>(cta_layout_vmnk)),
                                        group_modes<0, 3>(sW), group_modes<0, 3>(tCgW));

      uint16_t mcast_mask_x = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
      uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;
        const int n_tile_local = tile % params.n_local_tiles;
        const int e = (params.m_tile_expert != nullptr) ? params.m_tile_expert[m_tile]
                                                        : (m_tile / mtiles_per_expert);
        // W row math: w_ntile*TileN = e*N + n_tile_local*TileN -> the expert's output rows. //<<DOWN
        // (V1 had gate at e*2I + n_tile_local*kTileN and up at +I; the down has ONE block at e*N.)
        const int w_ntile = e * nw_per_expert + n_tile_local;

        Tensor tXgX_k = tXgX(_, m_tile, _, _0{});
        Tensor tWgW_k = tWgW(_, w_ntile, _, _0{});  //<<DOWN ONE slice

        for (int k = 0; k < k_tile_count; ++k) {
          pipeline_load.producer_acquire(load_prod);
          auto* bar = pipeline_load.producer_get_barrier(load_prod);
          int wr = load_prod.index();
          if (cute::elect_one_sync()) {
            copy(params.tma_load_x.with(*bar, mcast_mask_x), tXgX_k(_, k), tXsX(_, wr));
            copy(params.tma_load_w.with(*bar, mcast_mask_b), tWgW_k(_, k), tWsW(_, wr));  //<<DOWN 1 copy
          }
          ++load_prod;
        }
      }
      pipeline_load.producer_tail(load_prod);
    }

    // =========================================================================================
    // MMA warp — ONE acc of kMmaN(=TileN) cols, ONE cute::gemm.  //<<DOWN (drop acc_up + 2nd gemm)
    // =========================================================================================
    else if (role == kMMA) {
      cutlass::arch::warpgroup_reg_dealloc<40>();

      tmem_allocator.allocate(kTmemCols, &ss.tmem_base_ptr);
      __syncwarp();
      tmem_alloc_bar.arrive();

      TiledMma tiled_mma;
      Tensor acc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));  // (MMA,M,N=TileN) //<<DOWN

      Tensor sX = make_tensor(make_smem_ptr(ss.tensors.smem_x.begin()), SmemLayoutX{});
      Tensor sW = make_tensor(make_smem_ptr(ss.tensors.smem_w.begin()), SmemLayoutW{});  //<<DOWN
      Tensor tCrX = TiledMma::make_fragment_A(sX);
      Tensor tCrW = TiledMma::make_fragment_B(sW);  //<<DOWN ONE B fragment at N=TileN

      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        if (is_mma_leader_cta) {
          pipeline_epi.producer_acquire(epi_prod);
          acc.data() = ss.tmem_base_ptr + epi_prod.index() * uint32_t(kAccBufStride);  //<<DOWN ONE buf
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
          for (int k = 0; k < k_tile_count; ++k) {
            pipeline_load.consumer_wait(load_cons);
            int rs = load_cons.index();
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < size<2>(tCrX); ++kb) {
              cute::gemm(tiled_mma, tCrX(_, _, kb, rs), tCrW(_, _, kb, rs), acc);  //<<DOWN SINGLE gemm
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
    // EPILOGUE — ONE TMEM-load of the TileN-col acc, fp32 -> bf16 convert, TMA-store to Y[M,N].
    // NO silu, NO multiply, NO de-interleave, NO prob.  //<<DOWN
    // =========================================================================================
    else if (role == kEpilogue) {
#ifndef EPI_REGS
#define EPI_REGS 160
#endif
      cutlass::arch::warpgroup_reg_alloc<EPI_REGS>();
      tmem_alloc_bar.arrive_and_wait();

      TiledMma tiled_mma;
      Tensor tAcc = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));  // (MMA,M,N=TileN)
      tAcc.data() = ss.tmem_base_ptr + 0u;  // placeholder layout; per-tile .data() set in loop

      ThrMMA cta_mma_epi = tiled_mma.get_slice(0);
      Tensor cAcc = make_identity_tensor(take<0, 2>(TileShape{}));  // (kTileM, TileN) coords
      Tensor cAcc_cta = cta_mma_epi.partition_C(cAcc);

      using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;  // SAME atom as V1 (swiglu.cuh:863)
      int thread_idx = threadIdx.x % (NumEpiWarps * cutlass::NumThreadsPerWarp);
      auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tAcc);
      auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
      Tensor tTMc = thr_tmem_load.partition_D(cAcc_cta);  // (T2R,T2R_M,T2R_N) global (row,col) DST order

      Tensor rAcc = make_tensor<ElementAcc>(shape(tTMc));  //<<DOWN ONE register tensor (TileN cols)

      Tensor sY = make_tensor(make_smem_ptr(ss.tensors.smem_out.begin()), SmemLayoutY{});
      Tensor mY_tma = params.tma_store_y.get_tma_tensor(make_shape(M, N));
      ThrCopy thrblk_s2g = params.tma_store_y.get_slice(Int<0>{});
      Tensor bSG_sY = thrblk_s2g.partition_S(sY);
      const bool is_tma_store_lane = (warp_idx == 4) && (lane_predicate != 0u);

      for (int tile = cluster_id; tile < params.total_tiles; tile += params.num_clusters) {
        const int m_tile = tile / params.n_local_tiles;
        const int n_tile_local = tile % params.n_local_tiles;
        const int cta_row_offset =
            m_tile * kTileM +
            int(block_rank_in_cluster) * (kTileM / int(size(typename TiledMma::AtomThrID{})));
        const int cta_col_offset = n_tile_local * kOutN;  // OUTPUT col base (== V1, uses kOutN)

        pipeline_epi.consumer_wait(epi_cons);
        cutlass::arch::fence_view_async_tmem_store();

        // ONE acc buffer of TileN cols; ONE partition_S; ONE TMEM-load.  //<<DOWN
        const uint32_t epi_buf = epi_cons.index() * uint32_t(kAccBufStride);
        tAcc.data() = ss.tmem_base_ptr + epi_buf;
        Tensor tTM = thr_tmem_load.partition_S(tAcc);
        copy(tiled_tmem_load, tTM, rAcc);

#ifdef DOWN_DEBUG_PRINT
        if (tile == cluster_id && thread_idx == 0 && block_rank_in_cluster == 0) {
          for (int i = 0; i < size(rAcc) && i < 32; ++i)
            printf("DOWN T0 i=%2d row=%3d col=%3d acc=% .4f\n", i, get<0>(tTMc(i)),
                   get<1>(tTMc(i)), float(rAcc(i)));
        }
#endif

        // WAR drain of previous tile's TMA store (IDENTICAL to V1 swiglu.cuh:996-1000).
        if (is_tma_store_lane) cute::tma_store_wait<0>();
        cutlass::arch::NamedBarrier epi_war_bar(NumEpiThreads, /*id=*/1u);
        epi_war_bar.arrive_and_wait();

        // ---- PLAIN convert: fp32 acc -> ElementOut (bf16), scatter into sY at LOCAL (row,col). ----
        // NO silu, NO multiply, NO prob, NO de-interleave (the up-proj's hot elementwise epilogue is
        // GONE — the down-proj is a pure GEMM result store).  //<<DOWN
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rAcc); ++i) {
          int lrow = get<0>(tTMc(i));  // LOCAL row in [0,kEpiTileM)
          int lcol = get<1>(tTMc(i));  // LOCAL col in [0,kOutN)
          sY(lrow, lcol) = static_cast<ElementOut>(rAcc(i));
        }

        cutlass::arch::fence_view_async_tmem_load();
        pipeline_epi.consumer_release(epi_cons);
        ++epi_cons;

        // sY visibility + ASYNC TMA bulk-store — BYTE-IDENTICAL to V1 swiglu.cuh:1049-1073.
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier epi_smem_bar(NumEpiThreads, /*id=*/0u);
        epi_smem_bar.arrive_and_wait();
        if (is_tma_store_lane) {
          const int box_m = cta_row_offset / kEpiTileM;
          const int box_n = cta_col_offset / kOutN;  // == n_tile_local
          Tensor gY = local_tile(mY_tma, make_shape(Int<kEpiTileM>{}, Int<kOutN>{}),
                                 make_coord(box_m, box_n));
          Tensor bSG_gY = thrblk_s2g.partition_D(gY);
          copy(params.tma_store_y, bSG_sY, bSG_gY);
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
// LaunchDownGemmGroupedV2 — derived from V1 LaunchSwiGluGrouped (cutlass_grouped_gemm_swiglu.cuh) MINUS the
// SwiGLU-only args (W is plain [G*N,K], NO interleave; NO gather, NO prob).  Same occupancy launch.
//   X   : [M, K]   row-major (expert e owns a contiguous, TileM-aligned row range)
//   W   : [G*N, K] row-major (expert e: rows [e*N, (e+1)*N); used Tᵀ as the B operand)
//   Y   : [M, N]   row-major (expert e owns the same row range as X)
//   m_tile_expert : device [num_m_tiles] global-m-tile -> expert id; nullptr => uniform Me.
//   M_total       : total tokens (== sum Me; required when m_tile_expert != nullptr).
// ============================================================================================
template <typename Element, typename ElementOut, int TileM_ = 256, int TileN_ = 64, int TileK_ = 64,
          int kStages_ = 8, int ClusterM_ = 2, int MinBlocks_ = 1, int AccStages_ = 2>
cudaError_t LaunchDownGemmGroupedV2(
    const Element* X,   // [M, K] row-major
    const Element* W,   // [G*N, K] row-major (per-expert [N,K] weight block)
    ElementOut* Y,      // [M, N] row-major
    int G, int N, int K, int M_total, cudaStream_t stream, int device = 0, int sm_count = 0,
    const int* d_m_tile_expert = nullptr, int Me = 0) {
  using Config = DownGemmConfigV2<Element, ElementOut, TileM_, TileN_, TileK_, kStages_, ClusterM_,
                                  MinBlocks_, AccStages_>;
  using Kernel = Sm100DownGemmKernelV2<Config>;
  using CollectiveMma = typename Config::CollectiveMma;
  using StrideX = typename Config::StrideX;
  using StrideW = typename Config::StrideW;
  using StrideY = typename Config::StrideY;

  const int M = M_total;
  const int WN = G * N;  // total W rows (problem N, B operand)
  StrideX dX = cutlass::make_cute_packed_stride(StrideX{}, cute::make_shape(M, K, 1));
  StrideW dW = cutlass::make_cute_packed_stride(StrideW{}, cute::make_shape(WN, K, 1));
  StrideY dY = cutlass::make_cute_packed_stride(StrideY{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = sm_count;

  // Problem (M, WN=G*N, K): one to_underlying_arguments builds the X-TMA over [M,K] AND the W-TMA
  // over [G*N,K].  The W TMA box-N = kMmaN = TileN comes from Config::TileShape (a single B block).
  auto problem = cute::make_shape(M, WN, K);
  typename CollectiveMma::Arguments mainloop_args{};
  mainloop_args.ptr_A = X;
  mainloop_args.dA = dX;
  mainloop_args.ptr_B = W;
  mainloop_args.dB = dW;
  typename CollectiveMma::Params mainloop_params =
      CollectiveMma::to_underlying_arguments(problem, mainloop_args, nullptr, hw_info);

  cute::Tensor mY_store = cute::make_tensor(
      cute::make_gmem_ptr(Y),
      cute::make_layout(cute::make_shape(M, N), cute::make_stride(N, cute::_1{})));
  auto tma_store_y = make_tma_copy(cute::SM90_TMA_STORE{}, mY_store, typename Config::SmemLayoutY{});

  typename Kernel::Params params;
  params.tma_load_x = mainloop_params.tma_load_a;
  params.tma_load_w = mainloop_params.tma_load_b;  // N=TileN single-block B descriptor
  params.tma_store_y = tma_store_y;
  params.ptr_Y = Y;
  params.dY = dY;
  params.M = M;
  params.N = N;  // output width H
  params.K = K;  // contraction I
  params.Me = Me;
  params.WN = WN;
  params.m_tile_expert = d_m_tile_expert;

  constexpr int kTileM = cute::size<0>(typename Config::TileShape{});
  constexpr int kOutN = Config::kOutN;
  int num_m_tiles = (M + kTileM - 1) / kTileM;
  int num_n_local_tiles = (N + kOutN - 1) / kOutN;  // N/kOutN (output width tiles)
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

}  // namespace grouped_gemm_down
}  // namespace transformer_engine
