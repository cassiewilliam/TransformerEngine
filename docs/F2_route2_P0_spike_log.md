# F2 route#2 · P0 风险门 spike —— 详细步骤日志

> 目标：在投入自研 dual-accumulator gated mainloop 之前，先验两个 go/no-go 风险门：
> - **G1**：SM100 TMEM 能否容下**两个累加器**（gate_acc + up_acc）。
> - **G2**：ptr-array(grouped) + 2-SM 调度下能否构造**两条 TMA B-load**（gate 行块 / up 行块=+I 偏移）。
> 任一不过 → 回退 A1。本文件逐步记录每一步的命令、输出、结论。
>
> 环境：本地 CUTLASS `3rdparty/cutlass`（4.2.0）只读调研；构建期 spike 在 B200 容器 `/data1/min.yang/te_build`。

---

## Step 1 · 调研 CUTLASS 4.2 是否已有 gated / dual-accumulator / SM100 现成构件
**动机：** 若 CUTLASS 已有 SM100 的 gated / dual-B / 多累加器构件，可省掉大量自研。
**做法：** grep examples + epilogue fusion ops + SM100 collective。

**结果：**
- `epilogue/fusion/operations.hpp` 的 fusion op 菜单只有 `LinCombEltAct` / `LinCombEltActBlockScaleFactor`（单源 elementwise act）—— **无 gated/GLU 节点**（证实 sub-agent 结论：gated 不能靠现成 epilogue，需自研 mainloop）。
- collective builders 里**无** SM100 gated / dual-acc / multi-accumulator 构件。
- `examples/45_dual_gemm` 是 **Sm80**（5 处 `arch::Sm80`）→ 不可用于 SM100。
- **关键正面发现 ①：`examples/75_blackwell_grouped_gemm/`** —— CUTLASS 4.2 **有 SM100 grouped GEMM 范例**（含 block-scaled 变体）。这是 ptr-array 2-SM grouped 的权威参考（G2 取证点）。
- **关键正面发现 ②：`examples/77_blackwell_fmha/`** —— Blackwell FMHA 在**一个 persistent kernel 里做两次 matmul（S=QKᵀ、O=PV），用多个 TMEM 累加器**。这直接证明 **SM100 单 kernel 可持有/使用多个累加器 → G1 大概率 PASS**，且是 dual-accumulator mainloop 的现成模板。

**Step 1 结论：** 无捷径（须自研 dual-accumulator gated mainloop），但 (a) ex.75 给 SM100 grouped 模板、(b) ex.77 证明 SM100 多累加器可行（G1 有戏）。下一步精读这两个范例取证 G1/G2。

---

## Step 2 · 精读 ex.77(FMHA, 多累加器→G1) 与 ex.75(grouped, ptr-array→G2)
**动机：** ex.77 证 G1（两累加器 + TMEM 分配方式）；ex.75 证 G2（ptr-array 下 TMA B-load 的构造、能否加第二条）。

**G2 结果（ex.75 blackwell grouped，证据充分）：**
- `ProblemShape = GroupProblemShape<Shape<int,int,int>>`（每 group M,N,K）—— 与我们 `cutlass_grouped_gemm.cuh` 一致。
- 用 `KernelPtrArrayTmaWarpSpecialized2SmSm100` + `PtrArrayTmaWarpSpecialized2Sm`（正是我们的 2-SM 配置）。
- B 操作数 = **per-group 指针数组** `ptr_B`（`DeviceAllocation<const ElementB*>`），`ptr_B_host[i] = block_B.get() + offset_B[i]`；`stride_B = make_cute_packed_stride(StrideB{}, {N,K,1})`。
- **⇒ G1 取证 PASS：** 要加 “up” B 操作数,只需构造**第二个指针数组** `W1_base[i] + I·row_stride`（up 行块）,StrideB 相同 —— 与现有 B-load 同款构造,只是 base 指针 +I 行偏移。第二条 TMA + 2-SM multicast 复制 B-load 路径即可。**G2 可行。**

**G1 结果（ex.77 FMHA，证据充分）：**
- 一个 kernel 里 **两个 collective MMA**：`CollectiveMmaQK`、`CollectiveMmaPV`。
- **两个 TMEM 累加器并存**：`tStS = partition_fragment_C(mma_qk, …)`（S=QKᵀ）、`tOtO = partition_fragment_C(mma_pv, …)`（O=PV）。
- TMEM 容量：`cute/arch/tmem_allocator_sm100.hpp` → `Sm100TmemCapacityColumns = 512`（128 DP × 512 COL）。
- **⇒ G1 PASS：** 两个 `[BM, BN=128]` fp32 累加器 = 256 col ≪ 512,余量充足（BN=256 时 2×=512 刚好,但取 128 稳妥）。FMHA 已证「SM100 单 kernel 双累加器 + TMEM 分配」可行,且是 dual-accumulator mainloop 的现成模板。

---

## ✅ P0 结论：GO（两风险门均通过，有 CUTLASS 4.2 实证）
| 门 | 结论 | 证据 |
|---|---|---|
| **G1**（TMEM 容两累加器） | ✅ PASS | ex.77 FMHA 双累加器(tStS+tOtO)；TMEM 512 col，2×128=256 col 富余 |
| **G2**（ptr-array 双 TMA B-load + 2-SM） | ✅ PASS | ex.75 grouped 的 per-group `ptr_B`；up 用 `base+I·stride` 第二指针数组同款构造 |

**⇒ route#2 在 CUTLASS 4.2 / SM100 上可行,无需回退 A1。** 模板已锁定：
- **ex.75 `75_blackwell_grouped_gemm`** = grouped(ptr-array, 2-SM) 基座（≈ 我们 cutlass_grouped_gemm.cuh）。
- **ex.77 `77_blackwell_fmha` mainloop** = 单 kernel 双 collective-MMA + 双 TMEM 累加器的写法模板（`CollectiveMmaQK/PV` → `CollectiveMmaGate/Up`）。

## P1 实施步骤（grounded 在上述模板）
1. 复制 cutlass_grouped_gemm.cuh 的前向 SM100 collective,扩成 **dual-collective**：`CollectiveMmaGate`、`CollectiveMmaUp`（B 操作数分别取 W1 行块 0/+I 的指针数组,A 操作数=X 共享）。
2. 自研 mainloop（仿 ex.77）：K-loop 内 load X-tile(共享)+ gate-B-tile + up-B-tile,发两条 tcgen05 MMA → `acc_gate`、`acc_up`（两 TMEM 累加器）。
3. 自研 epilogue：`A = silu(acc_gate)·acc_up`（fp32 → cast → 写 A[T,I]）。
4. host 侧：构造两个 ptr_B（base、base+I·row_stride），problem_shape 的 N 用 I（A 的宽度）。
5. 编译 spike（单 expert/uniform）验证 G1/G2 在**真实编译**下成立(P0 的最终落锤)→ 通过即 P1 数值验证。

---

## P1 · dual-accumulator gated mainloop（自研）—— 起步
**做法：** 这是最深的 CUTLASS 工程（自研 SM100 warp-specialized 双-MMA mainloop）。派聚焦 sub-agent 精读 ex.77 mainloop 的「双 collective-MMA + 双 TMEM 累加器」写法 + ex.75 grouped host setup + 我们的 cutlass_grouped_gemm.cuh，产出**可编译骨架草案**（写入 `docs/F2_route2_P1_kernel_draft.md` + 草案 header），我再集成 + 编译迭代。

### P1.0 · sub-agent 草案返回 ✅（docs/F2_route2_P1_kernel_draft.md，519 行，27 个 `// VERIFY`）
**构造决策：** 选 **(a) 两个 CollectiveBuilder mainloop(gate/up)在自研 SM100 warp-specialized kernel 里组合（仿 FMHA）**；否决 (b) gather-B（CUTLASS 4.2 无跨列 GLU epilogue 节点，gather 两行块本质也是两条 TMA = (a) 的机制）。我们的情形比 FMHA 简单：两条 MMA 独立、共享 X(A) 操作数，可把 FMHA 的 softmax/correction warp + ~6 pipeline 砍成 1 load pipeline + 1 mma→epilogue pipeline；up 操作数 = 同款 ptr-array 偏移 `+I*row_stride`、StrideB 相同 → 零权重置换。

**草案揭示的 3 大风险（修正 P0）：**
1. **TMEM 比 P0 估计更紧**：stock SM100 kernel 把累加器**双缓冲**(ACC_PIPE=2，`sm100_mma_array_warpspecialized.hpp:477-478`)，N=256 时**单个**累加器就占满 512 col。两累加器只能 **N=128 + 单级累加器**(2×128=256≤512，仿 FMHA 非流水累加片段)。→ **route#2 kernel 必须用 N-tile=128**（比纯 GEMM 的 256 小，是性能折中）。编译期首验。
2. **ptr-array+2-SM 下的第二条 TMA B-descriptor**（唯一全新机制）：`to_underlying_arguments` 要建两个 `TMA_B`；per-group device 侧 tensormap 更新(`tensormaps_replace_global_address:778-779`)要**同时更新 gate 和 up**。
3. **`+I*row_stride` 必须 int64_t**（大 expert 下 int32 静默溢出是首个可能 bug）+ 保持 16B TMA 对齐。

草案是「手工集成 + 迭代」骨架（type config + host setup 是验证过的承重部分；device mainloop/epilogue 每个不确定 CUTLASS 调用都标了 `// VERIFY`），附 4 步单-expert 最小编译/数值测试计划。

### P1.1 · Step-1 编译验证 = ✅ PASS（G1 编译期落锤）
**做法：** 把草案的 §2.1 type config + §2.2 TMEM 列映射落成真头文件 `transformer_engine/common/gemm/cutlass_grouped_gemm_swiglu.cuh`（device kernel/host 暂留 TODO）。写最小测试 `qa/swiglu_step1_compile.cu`：实例化 `SwiGluConfig<bf16,bf16>` 的 `CollectiveMma`/`TiledMma` + `partition_fragment_C(TiledMma{}, take<0,2>(TileShape{}))` + `static_assert(kEnd ≤ 512)`。
**命令：** `nvcc -std=c++17 -arch=sm_100a --expt-relaxed-constexpr -I gemm -I cutlass/include -I cutlass/tools/util/include -c qa/swiglu_step1_compile.cu`
**结果：** **RC=0，无错误，产出 68KB .o。**
- ⇒ `SwiGluConfig` 的 CollectiveBuilder 在 SM100 2-SM、**N=128** 下成功实例化；`partition_fragment_C` 累加器片段合法；`static_assert` 通过 → **两个 N=128 fp32 累加器(256 col)确实 ≤ 512-col TMEM。G1 编译期已坐实。**
- 这坐实了 type config + TMEM 布局；route#2 底座可编译。

### P1 剩余核心（多周工程，草案是地图）
P1 Step 2-4 = 自研 device kernel：warp-specialized mainloop（2 条 `cute::gemm` → acc_gate/acc_up）+ load/epi pipeline + epilogue（silu·mul → A）+ host（两个 ptr_B：gate / up=+I*d）。draft 把这部分写成带 `// VERIFY` 的伪码骨架 + 5 个最难点。这是写**真 CUTLASS device 代码** + 编译/数值多轮调试，属多周专家级 kernel 工程。
- 状态：**设计 + 底座编译 = 已完成；device kernel 实现 = 待续（按 draft Step 2 单-expert 起）。**

### P1.2 · Step-2 单-expert device kernel（sub-agent 写真代码，进行中）
**做法：** 派聚焦 sub-agent 深读 ex.77 FMHA kernel/mainloop，把 §2.3/§2.4 伪码写成**真可编译 device 代码**（单-expert，num_groups=1, M=256/N=128/K=128，去掉 grouped/ptr-array/tensormap 复杂度）：Load/MMA/Epilogue warp 角色 + 两条 `cute::gemm`→acc_gate/up + silu·mul epilogue + 单-expert launcher + 数值测试 `qa/swiglu_single_expert_test.cu`（对 host fp32 参考）。返回后我 nvcc 编译 → 逐轮修错（每轮记录）。

#### Round 1（编译）：3× "Mismatched Ranks"（cute/stride.hpp:109）
- 根因：epilogue line 431 `make_layout(make_shape(M,N), params.dA)` —— `params.dA` 是 rank-3 `TagToStrideC_t<RowMajor>` `(int64,_1,int64)`，与 rank-2 shape `(M,N)` 秩不匹配。
- 修复：A 连续 → 直接给 rank-2 row-major stride：`make_tensor(make_gmem_ptr(params.ptr_A), make_layout(make_shape(M,N), make_stride(N,_1{})))`。`params.dA` 仅剩存储、device 端不再用。

#### Round 2（编译）：✅ RC=0，零 error，产出 2.0 MB binary `/tmp/swiglu_test`
- rank-2 stride 修复解决全部 3 个错误。**device kernel 编译通过**（type config + load/mma/epi warp + 双 `cute::gemm` + silu·mul epilogue + 单-expert launcher 全部实例化成功）。

#### Round 2（运行）：❌ **kernel 死锁（deadlock）**
- **测试纪律（用户要求"用空卡"）：** 容器 `sonic-moe-test-min` 仅挂 `/dev/nvidia0`（被他人 100%/71GB 占用），无法见到空闲卡。→ 改用**同镜像 sibling 容器 `--gpus device=7`（GPU7 实测 0 MiB/0% 空卡）**+ `-v /data1/min.yang` 挂载，binary 先 `cp` 到挂载目录。配 `timeout -s KILL 90` 看门狗 + `--name`/`rm -f` 自动清理，**不留残留进程**。
- **现象：** 在确认空闲的 GPU7 上，kernel 启动后**挂死 90s → 被 SIGKILL（EXIT=137）**。早前在 GPU0 上的多个 `swiglu_test` 进程因 GPU hang 进入 D 态（kill -9 后转 Z 僵尸，GPU 显存已释放，已清理）。
- **定性：空卡复现 → 真死锁，非显存争用。**

#### Round 2 死锁定位（静态分析，对照 stock `sm100_mma_warpspecialized.hpp`）
- ✅ **load pipeline 排除**：multicast mask 模式（X=`<2>`、W1=`<1>`）+ `tma_partition` 模式（A=`get<2>/size<2>`、B=`get<1>/size<1>`）与 stock collective **逐字一致**；transaction bytes 用 stock 同款 `size(AtomThrShapeMNK)`×box 公式，up-B 第三个 box 加在 expected 上与实际 3 条 TMA 投递一致 → **字节数匹配，非 load 死锁**。
- ✅ **UMMA→epi arrive 机制存在**：stock mainloop 靠 `accumulator_pipeline.producer_acquire`(loop 前) + 后续 `producer_commit` 完成 UMMA async arrive；本 kernel 两者都调了（line 369/388）。
- ⚠️ **最可疑：2-SM(2-CTA cluster) 的 MMA-issue/epi handshake**：tcgen05 2-SM MMA 应只由 **leader CTA(rank0)** 发射、peer CTA 不重复发；本 kernel 两 CTA 的 MMA warp **无 rank guard 均发 `cute::gemm`**。且 `PipelineEpi` 的 producer/consumer arrival 在 2-CTA 下是否 cluster-scoped 正确存疑。→ 派 sub-agent 聚焦诊断 2-SM 语义，返回补丁。

#### Round 2 死锁诊断（sub-agent 静态分析，对照 stock kernel-layer + sm100_pipeline.hpp）—— 4 处 2-SM bug
内核仿 1-SM FMHA 写，却配成 2-SM（ClusterShape (2,1,1)、Allocator2Sm），stock 2-SM driver 几乎每个 UMMA/acc handshake 都是 **leader-only** 且 arrival count ×`size(AtomThrShapeMNK)=2`：
1. **(A) 根因：MMA 发射 + acc commit 未 leader-gate** —— 两 CTA 都发 `cute::gemm` + `pipeline_load.consumer_release`（其 `umma_arrive_multicast_2x1SM` 同时 arrive 两 CTA 的 empty barrier，期望仅 1/stage）→ 每 stage 多一次 arrive → load empty barrier 相位错乱 → Load warp 后续 `producer_acquire`/`producer_tail` 永挂。stock 证据 `sm100_gemm_tma_warpspecialized.hpp:427`(`is_mma_leader_cta`)、`:761-771`(leader-only MMA+commit)。
2. **(#2) epi `consumer_arv_count` 少一半** —— 两 CTA epilogue 的 `consumer_release` 经 `Sm100MmaPeerBitMask`(`barrier.h:870`) 都重定向到 **leader** 的 empty barrier，故须 `size(Atom)*NumEpiThreads=2*128=256`（stock `:529`），原写 128 → follower barrier 挂。
3. **(#3) 缺 MMA→Epilogue 的 TMEM `tmem_base_ptr` 发布同步** —— stock 用专用 NamedBarrier（`:557,727-729,870-871`）；原内核仅 MMA warp 内 `__syncwarp`，epilogue 读 `ss.tmem_base_ptr` 是 UB。
4. **(#4 MEDIUM，单 tile 暂不致挂)** TMEM free 需 peer-CTA ClusterBarrier 握手（stock `:559-575,791-803`）。
- sub-agent 另强烈建议：**先塌成 1-SM** 验证 route#2 数学（dual-acc + silu·mul + up=base+I*d 在 1-SM/2-SM 完全相同），可绕开整个 2-SM 死锁面 —— 作为 fallback 记录。

#### Round 3（修 A+#2+#3，保留 #4 / 1-SM 作 fallback）
- 应用：epi `consumer_arv_count *= size(AtomThrShapeMNK)`；声明 `tmem_alloc_bar`(NamedBarrier, 160 线程, id=TmemAllocBarrier=6)，MMA `allocate` 后 `arrive()`、Epilogue 读 `tmem_base_ptr` 前 `arrive_and_wait()`；MMA 的 K-loop(双 `cute::gemm` + load consume/release) + `producer_commit` 全部 `if (is_mma_leader_cta)` 包裹，`producer_acquire`/`++epi_prod`/`allocate` 仍两 CTA 都做。
- **编译 RC=0**；**空卡 GPU7 运行 → 不再挂死（EXIT=1，非 137）→ 死锁已解！** ✅
- **但数值 FAIL：`max_abs=115.8 max_rel=2.39e5`** —— 误差量级远超"半数行错位"，指向 **epilogue 的 2-SM accumulator TMEM 读取 / 坐标映射**（risk #5：`tiled_mma.get_slice(0)` 在 follower CTA 仍取 leader 坐标；`make_identity_tensor((256,128))` 全 cluster tile vs per-CTA 128 行 fragment 不一致）。→ 下一步修 epilogue 坐标/读取。

#### Round 4（test-only 分半诊断，零 kernel 改动）—— 精确定位 epilogue
增强 test：分 rows[0,128)/[128,256) 各自 max_abs + 抽样 got/ref。结果（空卡 GPU0 跑，不再挂）：
```
max_abs=125.0  max_rel=2.39e5  within-5e-2: 2346/32768 (7.2%)
rows0..127 max_abs=125.0   rows128..255 max_abs=111.6
(m=0,n=0) got=-8.94 ref=-1.41 | (m=1,n=0) got=0.30 ref=0.34 | (m=64,n=0) got=7.5 ref=-0.30
(m=128..255, n=*) got=0.0000  ← 整段为 0（memset 未被覆写）
```
- **follower rows[128,256) 全 0** → epilogue **缺 per-CTA 行偏移 `+128*block_rank`**（两 CTA 算出的 row 都落在 [0,128)，follower 数据写进 leader 区或被丢，128-255 保持 memset 0）。
- **leader rows[0,128) 非 0 但 scrambled**（量级对、值错，7.2% 偶然命中）→ `get_slice(0).partition_C(identity(256,128))` 的坐标与 TMEM-load 的 thread→element 布局不一致，real accumulator 值被散到错误 (row,col)。
- **结论：** 自研 identity-coord 直接散射 TMEM→global 的写法对 2-SM TMEM accumulator 布局错误。FMHA fwd epilogue 实为 **TMEM→smem→TMA-store gO**（`partition_S(sO)/partition_D(gO)`，cute 处理布局），非手工散射。→ 派 sub-agent 用 stock SM100 TMEM-epilogue 范式重写 epilogue（坐标 tmem-load 一致 + block_rank 偏移，或 smem+copy/TMA-store）。

#### Round 5（epilogue 坐标修复，sub-agent 对照 stock collective epilogue）
- **诊断（带 stock 证据）：** `partition_fragment_C(tiled_mma,(256,128))` 返回 **per-CTA 128 行** fragment（非 256）——SM100 2-SM 的 M-split 在 TMEM *value* 布局里（`cute/atom/mma_atom.hpp partition_shape_C:563` 取 thread0 → per-SM value extent；`mma_traits_sm100.hpp tmem_frg::make:478` 注 "M_MMA_SM will be 64"）。fragment **不带 CTA 身份**，故 follower 的 +128 行偏移**必须经全局坐标 tile 显式注入**（stock 经 per-CTA `m_coord`：`sm100_tile_scheduler.hpp:800 += cta_in_cluster_offset_m` → `sm100_epilogue_array_tma_warpspecialized.hpp:740-742` 用 `CtaShape_MNK`+per-CTA coord）。
- **修复（已应用）：** 删除错误的 `tiled_mma.get_slice(0).partition_C(identity(256,128))`；改为
  ```cpp
  Tensor cAcc = make_identity_tensor(make_shape(M, N));               // 全局 (M,N) 坐标
  Tensor cAcc_cta = local_tile(cAcc, take<0,2>(CtaShapeMNK{}),        // (128,128) per-CTA tile
                               make_coord(int(block_rank_in_cluster), _0{}));  // leader→[0,128) follower→[128,256)
  Tensor tTMc = thr_tmem_load.partition_D(cAcc_cta);                  // 用同一 T2R thread-slice 分区
  ```
  `CtaShapeMNK = CollectiveMma::CtaShape_MNK = (128,128,64)`（config line 108）。坐标与数据用**同一 tmem-copy `thr_tmem_load`** 分区 → `tTMc(i)` 给出每个寄存器元素精确的全局 (row,col)（含 +128）。`size(rGate)==size(tTMc)` 保持一致。
- **编译 FAIL（20 err）**：`partition_D(cAcc_cta)` 对 rank-2 (128,128) 坐标 → `copy_atom.hpp:244 "Rank of tensor to be partitioned too small"` + `logical_divide: Too many modes`。数据侧 `partition_S(tAcc_gate)` 是 rank-3 `(T2R,T2R_M,T2R_N)`，坐标也须 rank-3；sub-agent 漏了 stock 的 `flat_divide(cD_mn, EpilogueTile)`（epilogue:742）来补模式。→ 派 fresh sub-agent 给 rank-correct 坐标修复（带编译错误 + rank 约束）。若 2-SM epilogue 坐标本质难表达，fallback 塌 1-SM。

#### Round 6（V-slice 坐标修复，fresh sub-agent 找到优雅根因）✅ 设计自洽
- **根因（精确）：** 2-SM MMA atom `SM100_MMA_F16BF16_2x1SM_SS` 的 `ThrID=Layout<_2>`、`CLayout=(_2,(M/2,N))`（`mma_traits_sm100.hpp:1700`）—— **M-half 在 V/thread 模式里**。`partition_fragment_C` 是 rank-3 `(MMA,MMA_M,MMA_N)`，`make_tmem_copy` 据此建 rank-3 `Tiler_MN`，故 `partition_D` 要求 rank≥3（`copy_atom.hpp:244`）→ 之前 rank-2 (128,128) 非法。
- **修复（rank + 偏移一箭双雕）：**
  ```cpp
  ThrMMA cta_mma_epi = tiled_mma.get_slice(block_rank_in_cluster % size(typename TiledMma::AtomThrID{}));
  Tensor cAcc = make_identity_tensor(take<0,2>(TileShape{}));   // (256,128) 全局坐标
  Tensor cAcc_cta = cta_mma_epi.partition_C(cAcc);              // rank-3 (MMA,MMA_M,MMA_N) 本-CTA 坐标
  Tensor tTMc = thr_tmem_load.partition_D(cAcc_cta);            // rank-legal
  ```
  `partition_C(identity)` 返回 rank-3 与 tAcc 同 extent → partition_D 合法且与 `partition_S(tAcc_gate)` 元素逐一对齐。**+128 follower 偏移自动产生**：在本 CTA 的 V（=`block_rank % AtomThrID`，与 load/mma warp line 314 同一 slice）切 partition_C → V=0(leader) rows[0,128)、V=1(follower) rows[128,256)。
- **关键对比：** 之前用 `get_slice(0)` 把两 CTA 都钉在 V=0 → follower 永不写、leader 错序 = 正是 Round 4 诊断的现象。换成 per-CTA V-slice 同时解决偏移与错序。
- 编译 RC=0；**空卡运行：两 half 现在都被写（follower 不再全 0 → V-slice 偏移生效）但仍 scrambled**：`max_abs=123.7 within-5e-2=909/32768(2.8%)`，real 量级散到错位。
- **诊断深化：** `partition_D(partition_C(identity))` = 双重置换（partition_C 先按 MMA C-序，partition_D 再按 copy dst-序）→ 错序。
#### Round 7（试 partition_S）❌ 编译 FAIL
- 把坐标改 `partition_S(cAcc_cta)`（想与数据 partition_S 对齐）→ `copy_traits_sm100.hpp:396 "dst layout doesn't vectorize into registers"`：copy 的**目标寄存器必须 partition_D 形状**，partition_S 不行。
- **锁定约束：** 寄存器 rGate 必须 partition_D 形；坐标须与之对齐；但 `partition_D(partition_C)` 错序。stock 正解是 `partition_D(flat_divide(plain (M,N) coord, EpilogueTile))`（epilogue:742）+ 数据走 reg→smem→global（非手工散射）。→ 派 sub-agent 给可编译且正确的 epilogue（优先正确坐标 flat_divide；否则 reg→smem→global 重写）。

#### Round 8（关键洞察：get_slice(0) 数据对齐 + 手工 +128 偏移）
- **第三个 sub-agent 揭示关键事实：** FMHA mainloop **数据 = `partition_fragment_C` ↔ 坐标 = `get_slice(0).partition_C` + `partition_D`** 是 element-aligned 的（`sm100_fmha_fwd_mainloop:524,548-549,568-569`）。即 `partition_D(partition_C(identity))` **是**正解，但坐标须用 **`get_slice(0)`**（与 `partition_fragment_C` 数据同 slice），**不是** `get_slice(block_rank)`。
- **据此重解此前现象：** Round5 用 get_slice(0) 但**漏 +128 偏移** → follower 数据写进 [0,128) 与 leader 竞争（故 follower [128,256)=0、leader "scramble" 实为两 CTA 抢写）。Round6 用 get_slice(block_rank) 加偏移 → 偏移到位但 V-slice **破坏了 CTA 内 element 对齐** → 真 scramble。
- **Round 8 修复：** `cta_mma_epi = tiled_mma.get_slice(0)`（数据对齐）+ 散射循环里 `row = get<0>(tTMc(i)) + block_rank*128`（手工偏移，per-CTA M=256/2=128）。这与 Round5/6 都不同。
- **空卡运行：与 Round 6 bit-identical**（`max_abs=123.688644` 完全相同、抽样值逐位相同）→ get_slice(0)+手工偏移 与 get_slice(block_rank) **输出完全一致** = V-slice 确实恰好 +128（sub-agent 对偏移判断正确），**坐标不是 bug**。错乱在**数据读取本身**（CTA 内 element 错位）。

#### Round 9（隔离：只输出 raw gate 累加器 vs ref_gate = X·W1[0:I]ᵀ）—— 决定性
- 加 `-DSWIGLU_DEBUG_RAW_GATE`：epilogue 直接写 `rGate(i)`，test ref 改为纯 gate。
- **结果：raw gate 也 scrambled**（`max_abs=23.4 within-5e-2=248/32768(0.8%)`，gate 量级的值散在错位 = permutation）。→ **bug 在 gate GEMM→accumulator→read 链**，非 epilogue silu·mul/坐标。
- 读取范式与 FMHA 逐字一致（`SM100_TMEM_LOAD_32dp32b32x` + `get_slice(0).partition_C` + `partition_S`数据/`partition_D`坐标，FMHA mla mainloop:558,543,567-568），FMHA 是 **1-SM**；本 kernel 2-SM 累加器布局是最大嫌疑。

#### ⇒ 决策：塌成 1-SM 验证 route#2 正确性（第一个 sub-agent 早期强烈建议）
2-SM 已耗 ~6 轮 epilogue（死锁→累加器读取），根因高度疑为 2-SM 累加器布局。1-SM = FMHA 实证配置，去掉整个 2-SM 复杂面（V-mode），验证 route#2 核心思想（dual-acc + silu·mul + up=base+I*d）。**2-SM 性能留作后续**。改：TileShape (128,128,64)、ClusterShape (1,1,1)、KernelTmaWarpSpecialized1SmSm100、Allocator1Sm（`std::conditional` on `size(AtomThrShapeMNK)`）、test M=128。先带 `-DSWIGLU_DEBUG_RAW_GATE` 验 1-SM 下 gate 读取是否正确。

#### Round 10（1-SM raw gate + checksum）—— **scramble 非 2-SM 特有**
- 1-SM 编译 RC=0；空卡运行 **raw gate 仍 scrambled**（0.8%，rows0-127；m≥128 的 1e29 是 test 样本越界，已属正常）。→ **bug 不是 2-SM 累加器布局**，1-SM/2-SM 同样错。
- **坐标无关 checksum（判 permutation vs 值错）：**
  ```
  sumsq: got=235520.9  ref=236859.8   （仅差 0.6%）→ 值的“量级分布”是对的（确在算 gate 点积）
  sum:   got=-540.8    ref=-232.6      （差很大）→ 不是干净 permutation；具体 [m,n] 配对错
  within-5e-2 = 0.8% = 恰好 1/128       → 系统性置换（沿 128 宽维）
  ```
- **定性：gate 累加器是“正确量级的值 + 错误 [m,n] 位置”** → 自研 mainloop 的 GEMM 操作数布局 或 TMEM 读取/散射的 layout bug（与 2-SM 无关）。FMHA 读取范式逐字照搬却仍错，说明问题在更底层的算子/布局对应，属深层 CUTLASS layout 调试（已迭代 ~15 个编译周期）。
- **里程碑回顾：** 死锁已彻底解决（2-SM leader-gate + ×2 arrival + TMEM NamedBarrier）；kernel 能跑；数值 bug 已隔离到 gate GEMM/read 的 layout。剩余为多周期深层 layout 调试。

#### Round 11（单累加器隔离 + device printf）—— **定位到 mainloop 操作数喂入**
- **单累加器（`-DSWIGLU_SINGLE_ACC` 去掉 up GEMM）：与 dual bit-identical**（checksum/样本逐位相同）→ **dual-acc 不是 bug**，就是一个**普通单 GEMM 也错**。
- **device printf（epilogue thread0 的 i→row,col,gate）：**
  ```
  T0 i=0  row=0 col=0  gate=-5.131    T0 i=1 row=0 col=1 gate=-1.325 ...（thread0 拥有 row0, col0..23）
  ROW0 ref: -1.111  7.153  5.125 -2.902 -3.230  2.185 ...（正确 gate[0][:]）
  ROW0 got: -5.125 -1.328 -3.000 -1.227 -1.141 -5.031 ...（kernel 写入 A[0][:]，= thread0 的值）
  ```
- **结论：** thread0 坐标说 (row0,col i)，但读到的值**不是 gate[0][i]**（量级对、配对错）。坐标与数据同出一个 `thr_tmem_load` slice → 即**MMA 写入的累加器 ≠ `gate[m][n]=X[m]·Wg[n]`** → **自研 mainloop 的操作数喂入 layout 错**（算成了错误 row/col 配对的点积）。非 epilogue/dual-acc/2-SM。
- → 派 sub-agent 逐行对比自研 load+MMA vs 已验证的 stock collective `sm100_mma_warpspecialized.hpp` load/mma，找操作数喂入分歧。

#### Round 12（sub-agent 找到根因：B 操作数 layout tag 错）✅ **route#2 数值验证通过**
- **根因：CUTLASS A/B 操作数 layout 约定是镜像翻转的。** W1 物理是 RowMajor `[I,d]`（row n 在 K 上连续 = **K-contiguous**），但 `TagToStrideB<RowMajor> = Stride<_1,int,int>` 让 **N 维**单位步长（N-contiguous）→ 把 W1 **读成转置**：`acc[m,n]=X[m]·colₙ(W1)` 而非 `·rowₙ(W1)`。K-contiguous 的 B 正确 tag 是 **ColumnMajor**（`Stride<int,_1,int>`）。证据：`cutlass/detail/layout.hpp:79/86`、`tag_to_umma_major_B`（`sm1xx_common.inl:117-124`）。
- **完美解释全部现象：** full-K（量级对）、`Σx²` 匹配但 `Σx` 差（转置≠置换）、1/128 正确（仅 W1 近似对称处巧合）。X 本来就对（RowMajor A = K-major ✓）。
- **修复：** `LayoutW1 = cutlass::layout::RowMajor → ColumnMajor`（config line 80）。经 `CollectiveMma::StrideB/SmemLayoutB` 自动流通，host `make_cute_packed_stride` 得 `dW1=(d,1,0)` = 真实 `W1[n,k]=ptr[n*d+k]`。up 指针 `W1+I*d` 偏移与 layout 无关，不变。
- **结果（full SwiGLU，空卡 GPU，1-SM M=128）：**
  ```
  max_abs=0.278  max_rel=0.0224(2.2%)  within-5e-2=15924/16384(97.2%)
  CHECKSUM sum: got=2729.391 ref=2729.084 | sumsq: got=1704514 ref=1704529  ← 5+ 位有效数字吻合
  样本逐一吻合：(0,0)0.1348/0.1350 (0,1)16.75/16.74 (64,1)-17.125/-17.107 (96,1)-24.125/-24.128 ...
  ```
- **`silu(gate)·up` 数值正确**。原 test 的 `max_abs<5e-2` 绝对阈值对大幅值 bf16 输出（silu·up 达 ~24，2% bf16→abs~0.28）过严 → 改 elementwise **abs-OR-rel 5e-2** 判据（`n_fail` 计两者都超的元素）。
- ✅ **route#2 核心思想（dual-acc + silu·mul + up=base+I*d + zero-Muon 标准 W1 布局）在 1-SM 上数值验证通过。** 关键修复链：死锁(2-SM)→塌 1-SM→**B 操作数 ColumnMajor**。
- **改 test 为 bf16 abs-OR-rel 判据后复跑：`n_fail=0/16384  PASS  EXIT=0`。** ✅✅

---

## ✅✅ P1 Step-2 里程碑达成：route#2 单-expert 融合内核 **数值验证通过**（1-SM, M=128/N=128/K=128, bf16）
**核心结论：** SonicMoE 的 SwiGLU 融合思想（kernel 内 dual TMEM accumulator 同时算 gate=X·W1[0:I]ᵀ、up=X·W1[I:2I]ᵀ，epilogue 内 silu(gate)·up，up 操作数 = W1_base + I*d 零权重置换 → 零 Muon 影响、1·IT HBM）在 SM100 上**可行且正确**。

**本轮关键修复（12 轮迭代浓缩）：**
1. **2-SM 死锁**（leader-only MMA/commit + load-release；epi consumer_arv_count ×size(AtomThrShapeMNK)；MMA→Epi TMEM-publish NamedBarrier）。
2. **塌 1-SM** 作正确性 spike（去 2-SM V-mode 复杂面；`std::conditional` Allocator + 1Sm schedule + TileShape M=128 单 tile）。
3. **B 操作数 layout tag**：`LayoutW1 = ColumnMajor`（K-contiguous 权重；RowMajor 会把 W1 读成转置）—— **这是数值正确的最后一把钥匙**。

**测试纪律：** 全程**空闲 GPU**（sibling 容器 `--gpus device=N` + SIGKILL 看门狗 + 自动 rm），不留残留进程，性能/正确性结论稳定。

## 待续（按用户 goal：每个优化点需精度 + 性能验证）
- **P1 Step-3：** 多 tile / grouped（ptr-array）—— 当前是单 expert 单 tile spike。
- **性能验证：** 融合 1·IT vs A1 2-GEMM 3·IT 在真实 MoE shape 上的 TFLOPS/HBM 对比。
- **2-SM 性能恢复：** 把 config 翻回 2-SM（256/2,1,1/2Sm）+ 复用已记录的 2-SM 死锁修复；2-SM 累加器读取需重验（Round 1-9 的 epilogue 坐标在 2-SM 下还需结合本轮 ColumnMajor 修复复测）。
- **P3 TE 集成 / P4 backward / P5 memory-reduce（最后）。**

---

## P1 Step-3a：多-tile（单 expert）✅ 数值通过
**做法：** 加 tile scheduler（1-SM：每 block=一 cluster=一 (m_tile,n_tile)，grid=(num_n_tiles,num_m_tiles,1)，`m_tile=blockIdx.y/n_tile=blockIdx.x`）；load 按 tile 切片 `tXgX(_,m_tile,_,_0)`/`tWggWg(_,n_tile,_,_0)`（gate/up 同 n_tile → up 自动取 W1[I+n_tile*128..]）；epilogue 全局偏移 `row += m_tile*kTileM (+2-SM split)`、`col += n_tile*kTileN`；launcher grid 覆盖全 tile，problem N=I 全宽。
**结果（空卡，M=256 I=256 d=128 = 2×2 tiles）：`n_fail=0/65536  PASS  max_rel=0.39%  checksum 5+位吻合`。** ✅
**待续：** 真实 expert shape 验正确性 + 计时（fused TFLOPS）；2-GEMM baseline 对比；真 grouped（多 expert ptr-array）。

## P1 Step-3b：真实 shape 正确性 + 首批 fused 性能（1-SM, N-tile=128）
**正确性（空卡，abs-OR-rel 5e-2，n_fail=0 PASS）：**
- M=512  I=1024 d=1024 ✅；M=2048 I=1024 d=1024 ✅（2.1M 元素全过）。

**fused kernel 计时（CUDA event, 100 iter, 空卡 B200）：**
| shape (M,I,d) | tiles | ms/iter | TFLOPS | eff GB/s |
|---|---|---|---|---|
| 512,1024,1024 | 4×8=32 | 0.0473 | 45.4 | 133 |
| 2048,1024,1024 | 16×8=128 | 0.0475 | **180.7** | 265 |

**分析：**
- 45→181 TFLOPS（32→128 tiles）= 小 shape 是**占用率/延迟限制**（32 tile « 148 SM）；128 tile≈1 wave 才到 181。
- 181 TFLOPS 仍仅 ~16% 已验证 2-SM grouped（~1100）。**限制：1-SM + N-tile=128（≈4× 低于 2-SM N=256）、kStages=2、非 persistent。**
- 265 GB/s « B200 ~8 TB/s → **非内存瓶颈**：此规模下 fusion 的 HBM 节省**不体现为加速**。route#2 的收益是 **HBM 流量/不物化 I 宽中间张量**，在内存瓶颈 / grouped 多专家场景才显著。

**下一步性能杠杆（按收益）：**
1. **恢复 2-SM**（TileShape 256/ClusterShape 2,1,1/2Sm + grid 的 cluster→tile 重映射 + 复用死锁修复）—— GEMM 效率最大杠杆。**关键再认识：** Round 6 的 2-SM "2.8% scramble" 当时 LayoutW1 还是 RowMajor（转置 bug），很可能根因就是转置而非 2-SM 累加器读取；ColumnMajor 修复后 2-SM 累加器读取大概率本就正确（epilogue 坐标 get_slice(0)+手工偏移 已验证 1-SM 正确，2-SM 同构）。需复测确认。
2. **2-GEMM baseline 对比**（量化 fusion 的 HBM/时间收益，尤其内存瓶颈 shape）。
3. **真 grouped（多 expert ptr-array）** —— 多专家 → 多 tile → 填满 GPU + 真实 MoE 口径。
4. kStages↑ / persistent scheduler。

## P1 Step-3c：2-SM 恢复 —— 正确性✅但大 K 崩溃（pipeline cycling bug）
- **2-SM 数值正确性✅：** 翻回 2-SM（TileShape 256/ClusterShape (2,1,1)/2Sm + `n_tile=blockIdx.x/kClusterX` cluster-aware tile 调度）后，M=256/I=256/**d=128** PASS（与 1-SM bit-identical）。**证实"Round 6 的 2.8% scramble 是 LayoutW1 转置 bug 而非 2-SM 读取"** —— 死锁修复 + ColumnMajor + epilogue 偏移在 2-SM 下全部正确组合。
- **但 2-SM 在 k_tile_count>Stages(=2) 时崩溃（`unspecified launch failure` = 非法访存）：** 扫 (256,256,2048)/(512,512,2048)/(2048,1024,1024) 全崩（d≥1024 → k≥16）；d=128（k=2）才过。1-SM 在所有 K 都正常（d=1024 验证过）。→ **2-SM load-pipeline 在真正"循环"复用 stage 时有相位/越界 bug**（疑 follower producer vs leader multicast-release 的 stage 同步）。属下一步 2-SM 调试（修好 = ~2× perf 杠杆）。
- **决策：先回 1-SM**（全 K 正常）拿用户 shape 的可用性能数，2-SM cycling bug 单列调试。
- **用户 shape (G,M,N,K)=(32,16384,512,2048)：** 用单大 GEMM 代理（M=16384 总 token, I=N=512, d=K=2048；总计算量=grouped）。test 改 argv 取 (M I d)，host ref 仅前 512 行（大 M 全 ref 太慢）。

**1-SM fused 性能（空卡 B200，正确性全 PASS）：**
| shape (M,I=N,d=K) | tiles | ms/iter | TFLOPS | eff GB/s |
|---|---|---|---|---|
| **16384,512,2048（用户）** | 512 | 0.218 | **315.3** | 404 |
| 8192,512,2048 | 256 | 0.107 | 321.0 | 431 |
| 16384,1024,1024 | 512 | 0.308 | 223.2 | 232 |
- 用户 shape **315 TFLOPS（1-SM, N-tile=128）**，~28% 已验证 2-SM grouped（~1100）。大 K=2048 算术强度高 → 比之前 d=1024 的 181/223 TFLOPS 更好。
- 404 GB/s « 8 TB/s → 仍**计算/占用瓶颈，非内存瓶颈** → fusion 的 HBM 节省此规模不体现为加速（收益是流量/不物化中间张量）。
- **性能升级路径：** (1) 修 2-SM K-cycling bug（~2×）；(2) 2-GEMM baseline 对比（量化 fusion 收益）；(3) 真 grouped ptr-array。

## P1 Step-4：单次启动 GROUPED（多 expert）✅ 正确 + 用户 shape 性能
**设计（contiguous stacked MoE, uniform Me，单次 launch，无 ptr-array tensormap swap）：**
- expert **由 m-tile 推导**（token 按 expert 连续）：`e = m_tile / (Me/kTileM)`。
- **一个** W1 描述符覆盖 `[G*2I, d]`；gate/up = 同描述符的两个 per-expert N-tile 切片：`gate_ntile = e*(2I/kTileN) + n_local`、`up_ntile = gate_ntile + I/kTileN`（= 单-expert "up=W1+I*d" 的 grouped 泛化，无跨 expert OOB）。
- X 描述符 `[G*Me,d]`、A `[G*Me,I]`；grid=(I/kTileN·clusterX, G*Me/kTileM, 1)。单-expert = G=1 wrapper（回归测试 PASS，无退化）。
**结果（空卡 B200，per-expert 不同权重，正确性逐 expert 验证）：**
| (G,Me,I,d) | M | n_fail | TFLOPS | ms/iter | eff GB/s |
|---|---|---|---|---|---|
| (2,256,256,128) | 512 | 0 ✅ | 1.8 | 0.038 | 24 |
| (4,256,256,128) | 1024 | 0 ✅ | 3.5 | 0.039 | 47 |
| **(32,512,512,2048) 用户** | 16384 | **0 ✅** | **301.6** | 0.228 | 1546 |
- **用户 grouped shape 301.6 TFLOPS**（≈ 单 GEMM 代理 315 → grouped 调度开销可忽略）；HBM 升到 1546 GB/s（真实 per-expert 权重流量 G×）。仍计算瓶颈（«8 TB/s）。
- ⇒ **route#2 SwiGLU 融合在真实 grouped MoE 口径上：正确 + 单次启动 + 301.6 TFLOPS（1-SM）。** 2-SM（~2×）待修 K-cycling bug。非 uniform Me 需 per-expert prefix-sum 偏移数组（已注释）。

## P1 Step-5：2-SM 修复尝试（compute-sanitizer 制导）—— 修了 2 个 bug，第 3 个深层 race 未解
**方法：** 翻 2-SM，`-lineinfo` 编译，compute-sanitizer 逐个定位。
- **Bug #1（已修）：load-pipeline `is_leader` 未 gate 到 leader CTA。** sanitizer 定位到 follower CTA(block(1,0,0)) 的 `producer_acquire→arrive_and_expect_tx`（barrier.h:588）。stock `is_leader = lane && is_mma_leader_cta`（`sm100_gemm_tma_warpspecialized.hpp:464`），我原来两 CTA 都 is_leader → follower 重复 expect_tx → cluster 事务 barrier 损坏，循环复用 stage 时崩。**修复：`is_leader &&= (block_rank % AtomThrID)==0`。** → **k=16 单 expert(2 cluster) PASS**（之前崩）。
- **Bug #2（已修）：TMEM alloc 在 MMA warp(1)、free 在 epilogue warp(4) —— 不同 warp。** `Allocator2Sm` alloc/free 是 `tcgen05.{alloc,dealloc}.cta_group::2.sync` 配对 op，前置条件要求"同一逻辑 warp + 两 CTA 同 warpID"（`tmem_allocator_sm100.hpp:130-180`）。**修复：free 移到 MMA warp（同 alloc warp）+ `release_allocation_lock`，新增 `tmem_free_bar`(EpilogueBarrier id) 做 MMA↔epilogue 的"epilogue 读完 TMEM"同步。**
- **Bug #3（未解）：>2 cluster / 重复启动 的深层 timing RACE。** 现象：`unspecified launch failure`，但 **memcheck 0 errors、synccheck 0 errors，且在 sanitizer 减速下不复现**（典型 cross-CTA 数据/资源竞争）。(256,256,1024,G=1)[2 cluster] PASS；(512,256,1024,1)[4 cluster]、重复启动 timing loop 崩。3 次定向修复（is_leader/release_lock/alloc-free-same-warp）均未解。**根因在 2-SM cluster/tcgen05 更底层，需更深专精，本轮未解。**
- **决策：回退 config 到 1-SM**（已验证 grouped 301 TFLOPS 工作态），2-SM 的 #1/#2 修复保留在 kernel（1-SM 下惰性）。**2-SM ~2× 性能杠杆留作后续深挖。**

### Bug #3 续修 ✅ **解决！**（cluster_sync 跨-CTA 对齐 free）
- **根因（精确）：** `cta_group::2` 的 `release_lock`+`dealloc` 是配对 op，需两 CTA 的 warp-1 **同时**发；但两 CTA 经过变长的 MMA+epilogue 后到达 free 点的**时刻不同 → 配对 dealloc 竞争**（timing race，故工具抓不到、减速即消失）。stock 用专用 peer-CTA dealloc `ClusterBarrier`(`sm100_gemm_tma_warpspecialized.hpp:790-803`) 协调；我之前只加了 intra-CTA 的 `tmem_free_bar`，缺跨-CTA 对齐。
- **修复：** 删除 `tmem_free_bar`，在**所有 role 分支之后**加 **`cute::cluster_sync()`**（两 CTA 全 warp 都到达 → 既保证 epilogue 读完 TMEM，又对齐两 CTA），其后 MMA warp 发 `release_lock`+`free`。一个 barrier 同时解决 intra-CTA 与 cross-CTA 两个竞争。
- **验证全 PASS：** (256,256,1024,1)[2cl]、**(512,256,1024,1)[4cl]**、**(256,256,1024,4)[8cl]**、**(256,256,2048,1)[k=32 timing loop]** 全部 `n_fail=0 PASS` 且 timing 完成（无重复启动崩）。**2-SM cycling bug 三个全部修复！**

## ✅✅✅ 2-SM 修复完成：3 个 bug（is_leader / TMEM-same-warp / cluster_sync-free）全解
**2-SM 性能（用户 grouped shape，正确性全 PASS）：**
| (G,Me,I,d) | 1-SM TFLOPS | 2-SM TFLOPS |
|---|---|---|
| (32,512,512,2048) 用户 | 300.4 | **292.5** |
| (1,2048,512,2048) | — | 160.7 |
| (16,512,1024,1024) | — | 180.8 |

**关键诚实发现：2-SM ≈ 1-SM（略慢），并非 ~2×。** 原因：用户 shape 下两者都启 **同样 512 个 CTA、每 CTA 同样 128×128 工作量**（2-SM 只是把 256-M tile 拆给 2 个 CTA）。**"2× 杠杆"假设是错的** —— 真正的性能天花板是 **N-tile=128 dual-acc + kStages=2 + 手写 kernel 结构**（vs stock collective 的 ~1100 TFLOPS 用 N=256/多 stage/persistent），不是 1-SM-vs-2-SM。新增的 cluster_sync 还带来几 % 开销。

**⇒ 价值：2-SM cycling bug 三个全部修复（is_leader / TMEM-same-warp / cluster_sync-free，compute-sanitizer 系统制导），kernel 在 1-SM 与 2-SM 下均正确。但 perf 与 1-SM 相当。真正的提速杠杆是：N-tile（256 需 TMEM 调优，dual-acc 满 512 列）、kStages↑、persistent scheduler —— 非 SM 模式。**

## P1 Step-6：2-SM 性能调优（TileShape × kStages 扫描，空卡，全 PASS）
**关键发现：pipeline 深度（kStages）> tile 大小。** 用户提示"M 大 N/K 小 + 小 TileN/TileK 大 TileM"验证有效。
| 配置 | (32,512,512,2048) M16384 | (32,1024,256,1024) M32768 | (32,2048,128,512) M65536 | (32,2048,256,512) M65536 |
|---|---|---|---|---|
| TileN128/BK64/kS2（基线） | 292 | — | — | — |
| TileN128/BK64/**kS3** | **324**(+11%) | 190 | 105 | 123 |
| TileN**256**/BK64/kS2 | 270(**差**) | 160 | — | 89 |
| **TileN64/TileK16/kS8** | 325 | **222**(+17%) | **137**(+31%) | **142**(+16%) |
- **TileN=256 更差**（满 TMEM 但 kStages 被迫降到 2，pipeline 深度损失大于 tile 增益）。
- **小 TileN(64)+小 TileK(16)+多 stages(8)** 在 M 大/N&K 小的目标 regime 上显著更优（+16~31%）。
- **TileM 硬上限 256**（2-SM = 128 TMEM datapath × 2 CTA，再大需 M-subtile 循环，本 kernel 无）。
- **kStages=16**（TileN64/TK16）：主 shape 325→**344**（+6%，K=2048 仍吃 stage 红利）；K=512 shape 已饱和。
- **TileN=32 太小**：MMA 效率崩（344→217），即使 stages 更多也补不回。**TileN=64 是甜点。**

### ⇒ 调优最优配置：**TileShape (256, 64, 16) + kStages=16**（2-SM）
| shape (G,Me,I,d) | 原始 292 起 | **调优后** | 提升 |
|---|---|---|---|
| (32,512,512,2048) M16384 **用户** | 292 | **344 TFLOPS** | **+18%** |
| (32,1024,256,1024) M32768 | 190 | 229 | +21% |
| (32,2048,128,512) M65536 | 105 | 138 | +31% |
| (32,2048,256,512) M65536 | 123 | 142 | +15% |
**调优要点（用户制导有效）：** (1) **pipeline 深度是主杠杆**（kStages 2→16，受 smem 限）；(2) **小 TileN(64)/TileK(16)** 释放 smem 让 stages 加深，且贴合 M 大/N&K 小的 memory-ish regime；(3) TileN 太大(256)失 stages、太小(32)失 MMA 效率 → 64 甜点；(4) TileM 硬顶 256（2-SM）。

## P1 Step-7：结构性优化（ncu profile 制导）
**ncu profile（用户 shape, 调优 344 配置）：Compute(SM) 31.9% / L1-TEX 84.4%(瓶颈) / DRAM 9.6% / 占用率 12.4%。** ncu 提示 "memory replay (coalescing)" → **根因 = epilogue 的逐元素全局散射 `mA(row,col)=...` 不合并 → L1 抖动 84%**。
### #4 Epilogue 合并（reg→smem sA→合并 smem→global）✅ **+16%**
- 把逐元素全局散射换成：先把 `silu(gate)*up` 结果按 local (row,col) 写入 per-CTA smem tile `sA`（smem 无合并惩罚），epilogue-only NamedBarrier 同步，再让 128 epilogue 线程 **flat strided 合并写** sA→global A（相邻线程→相邻列→合并）。smem +16KB（sA），总 ~144KB<228KB。
- **结果（正确性全 PASS）：用户 shape 344→398.4 TFLOPS(+16%)；(2048,128,512)138→174(+26%)。** 证实 epilogue 散射就是 L1 瓶颈。
- **累计：未调 2-SM 292 → tile 调优 344 → epilogue 合并 398 TFLOPS（+36%）。**
**用户给的后续结构优化 roadmap：#1 dual-gemm(1X+2W 作最小计算单元)、#2 Cluster Sync + DSMEM、#3 persistent kernel + CLC 调度、#4 epilogue(已完成)。**

### 占用率分析（re-profile after epilogue fix）
- L1 84%→59%（epilogue 修复见效），Compute 32%→37%，但 **占用率成为瓶颈：理论 12.5%，1 block/SM**，受 **registers(160/thr) + smem 双限**。ncu "Est. Local Speedup 87.5%"。
- 尝试解占用率：**MinBlocks=2**（编译器按 2 block/SM 配寄存器，warpgroup reconfig 仍给 epilogue 160 动态）+ **kStages 16→10**（smem ~96KB）+ **TMEM 只 alloc 128 列（非满 512）**（让多 block 共享 SM 的 512 列 TMEM）。→ **理论占用率 12.5%→25%**（Block Limit reg/smem/TMEM 都变 2），但 **achieved 仍 12.5%**！
- **根因：2-SM cluster 共驻限制 —— cta_group::2 cluster 占住一对 SM，硬件不在同一 SM-pair 上叠 2 个 cluster**（资源够但调度不叠）。**⇒ 2-SM 占用率硬顶 ~12.5%，"87.5% headroom" 在 2-SM 下不可达。** 要更高占用率需 1-SM（用户要 2-SM）或 persistent/CLC（#3，待验）。
- perf：kStages=10/MinBlocks=2/TMEM128 = **396.8 TFLOPS**（≈ kStages=16 的 398，说明 stage 10 已够、占用率才是天花板）。TMEM-cols 修复正确且保留（好实践 + 为未来占用率铺路）。

**⇒ 当前最优 ~398 TFLOPS（+36% from 未调 292），占用率 12.5% 2-SM 硬顶。**

### Config 抽象（用户要求，为 tuning/JIT）✅
`SwiGluConfig<Element, ElementOut, TileM=256, TileN=64, TileK=16, kStages=16, ClusterM=2, MinBlocks=1, AccStages=1>` —— 所有 perf 旋钮模板化，默认值 = 已调最优。`ClusterM` 选 1-SM/2-SM schedule；`MinBlocks` 进 launch bounds；`AccStages` 乘 `kTmemCols`（=AccStages×2×TileN，为未来 TMEM 双缓冲预留，alternation 逻辑 TODO）。`LaunchSwiGluGrouped<Element,ElementOut,…旋钮默认…>` 向后兼容（现有调用不变）+ 可显式实例化别的 config 做 autotuning。**验证：默认实例化编译通过、正确、398.8 TFLOPS（== 调优值，行为不变）。**

## 性能优化阶段小结（结构性，ncu 制导）
| 阶段 | 用户 shape TFLOPS |
|---|---|
| 未调 2-SM | 292 |
| tile 调优（TileN64/TK16/kS16） | 344 (+18%) |
| + epilogue 合并（smem staging） | **398 (+36% 累计)** |
- **#4 epilogue（合并写）= 最大单项收益（+16%），ncu 制导命中 L1 瓶颈。**
- 占用率被 2-SM cluster 共驻硬顶 12.5%，smem/reg/TMEM 修复（MinBlocks=2/kStages=10/TMEM 128 列）都无法在 2-SM 下叠 2 block/SM。
- Config 已模板化，可后续 autotune/JIT。
**剩余 roadmap（更大结构性工程）：#1 dual-gemm 单计算单元、#2 Cluster Sync + DSMEM、#3 persistent kernel + CLC（可能松动 2-SM 占用率上限）。**

## P1 Step-8：#3 Persistent kernel（软件 grid-stride）✅
**重构：** 每 cluster grid-stride 遍历 tile（`tile=cluster_id; tile<total_tiles; tile+=num_clusters`），TMEM alloc/free + cluster_sync **整个循环只一次**，load/MMA/epi 流水状态跨 tile 连续。launch grid 改为 `num_clusters=min(total_tiles, 2×SM_pairs)` 个 cluster（超订 2/SM-pair 试探占用率）。
**两个多-tile 死锁（compute-sanitizer 不抓 deadlock，靠分析）：**
1. **epi_smem_bar 与 epi_done_bar 同 NamedBarrier id 0**（→ 同一 hw barrier 8）→ 跨 tile 两次同 barrier sync 互锁。修：epi_done_bar 用 id 1（hw 9）。
2. **follower CTA 的 epi `producer_acquire` 跨 tile 死锁**：2-SM 下两 epilogue 的 empty-arrive 经 `Sm100MmaPeerBitMask` 都重定向到 **leader** 的 empty barrier，follower 自己的 empty barrier 永不前进 → tile-0 用初始态过、tile-1 起死锁。修：**把 epi producer_acquire + commit + ++epi_prod 全 gate 到 leader CTA**（follower MMA 不碰 acc pipe）。
**结果（正确性全 PASS）：**
| shape | 非-persistent | **persistent** | 提升 |
|---|---|---|---|
| (32,512,512,2048) M16384 用户 | 398 | 394 | ≈ 持平 |
| (32,1024,256,1024) M32768 | 222 | **259** | **+17%** |
| (32,2048,128,512) M65536 | 174 | **223** | **+28%** |
- **persistent 在 M 大/N&K 小 regime 大胜（+17~28%）**（per-tile TMEM alloc/free + cluster_sync + setup 被摊薄，小-K tile 计算少、overhead 占比大）；compute-heavy 用户 shape 持平（few tiles/cluster，摊薄少）。
- 占用率：persistent 仍 12.5%（kStages=16→1 block smem-limited + 2-SM co-residency 硬上限）；增益来自 **overhead 摊薄**，非占用率。commit `abebe85c`。

## P1 Step-9：TMEM 跨-phase 复用（double-buffered accumulator, AccStages=2）✅ **最大单点增益**
**原理：** 原 1-stage acc pipeline 把 epilogue(tile N) 与 MMA(tile N+1) **串行化**（MMA 必须等 epilogue 读完 acc 才能复写）。改成 **2-stage acc pipeline + 双 TMEM accumulator buffer**（buffer = acc-pipeline stage index）：tile N 用 buffer N%2、tile N+1 用 buffer (N+1)%2，物理不同 TMEM 窗口 → leader MMA 可在 buffer1 跑 K-loop，同时 epilogue 读 buffer0 → **MMA(N+1) 与 epi(N) 重叠（ping-pong）**。
**实现（Config::kAccStages 参数化，AccStages=1 退化为原行为）：**
- `PipelineEpi = PipelineUmmaAsync<kAccStages, AtomThrShapeMNK>`（1→2 stage）。
- `kAccBufStride = 2*kEpiTileN`；MMA 每 tile（leader）`producer_acquire` 后按 `epi_prod.index()*kAccBufStride` 设 acc_gate/up `.data()`；epilogue 每 tile `consumer_wait` 后按 `epi_cons.index()*kAccBufStride` 设 tAcc_gate/up `.data()` **并把 partition_S 移进循环**（partition_S 固化 ptr，必须按 buffer 重切）。
- `kTmemCols = kAccStages*2*kEpiTileN`（AccStages=2→256≤512 OK；约束 AccStages*2*TileN≤512）。
- 无死锁：2-stage 下 MMA(N) acquire stage N%2 的 empty 由 epi(N-2) 释放，依赖链 MMA(N)→epi(N)→释放给 MMA(N+2)，严格前向无环；leader-gate 不变。
**结果（正确性全 PASS n_fail=0）：**
| shape | AccStages=1 (persistent) | **AccStages=2** | 提升 |
|---|---|---|---|
| (32,512,512,2048) M16384 **用户** | 394 | **546** | **+39%** |
| (32,1024,256,1024) M32768 | 259 | **405** | **+56%** |
| (32,2048,128,512) M65536 | 223 | 226 | 持平（mem-bound） |
| (4,256,256,512) M1024 | 23 | 24 | 持平（太小） |
- **compute-heavy regime 单点最大增益**：用户 shape **394→546 TFLOPS（+39%）**，**累计 292→546（+87%）**。
- mem-bound 小-NK 持平（epilogue 本就只占小比例，bottleneck 在 load）。
- **AccStages sweep：2=545.6/404.3，4=546.0/404.9（持平），3 编译失败**（kTmemCols=384 非 2 的幂，static_assert）。→ **AccStages=2 为 sweet spot**：MMA 是 bottleneck，epilogue 落后从不超过 1 tile，2 buffer 已完全 hide，更深无益。default 锁定 AccStages=2。
- **ncu（用户 shape, AccStages=2）：Compute(SM) 37%→49.76%**（MMA 不再被 epilogue 阻塞 → overlap 生效）；**新 bottleneck = L1/TEX 79.91% / Memory 75.14%**（smem 流量：load TMA + epilogue staging），DRAM 仅 15.24%（已 compute/smem-bound，非 DRAM-bound）；占用率仍 12.48%（2-SM 硬上限不变，增益纯 latency-hiding）。
- commit `18ffde6c`。tile/stage re-tune sweep（double-buffer 后重扫）：256_64_16_16 仍最优，128_128_16_16 噪声内（+0.4%），TileN=128 伤小-N shape（404→328），kStages=10 中性 → **tile/stage 已收敛**。

## P1 Step-10：persistent grid 单波化（occupancy-driven launch）✅ **意外 +8~9%**
**问题（persistent_grid_fix.md 提出，已核实）：** 旧 launch `persistent_clusters = 2*(sm_count/cluster_size)` 用硬编码 ×2 过订阅，假设每 SM-pair 驻留 2 cluster。但 ncu 已证 achieved occupancy=12.5%=**1 cluster/SM-pair（2-SM co-residency 硬件上限）**→ ×2 发 296 CTA 但仅 148 co-reside → **强制第二波**：74 个 2nd-wave cluster 在 wave 边界空等 + 重付 per-cluster setup（TMEM alloc/free + cluster_sync）= **wave quantization 尾巴**。
**修复：** 用 `cudaOccupancyMaxActiveClusters`（自建带真实 `smem_size` 的 config，比裸 CUTLASS helper 不漏 smem）查设备实际可驻留 cluster 数（≈74，单波），失败退回 `sm_count/cluster_size`。kernel_ptr/block/smem + func-attr opt-in 上移到查询前。镜像 CUTLASS `query_device_max_active_clusters` + MegaMoE `grid=num_sms`。
**结果（正确性全 PASS，n_fail=0）：**
| shape | ×2 过订阅 | **occupancy 单波** | 提升 |
|---|---|---|---|
| (32,512,512,2048) M16384 **用户** | 546 | **596.6** | **+9.3%** |
| (32,1024,256,1024) M32768 | 404 | **440.4** | **+9%** |
| (32,2048,128,512) M65536 | 226 | **243.9** | **+8%** |
| (4,256,256,512) M1024 | 24 | 23 | 持平（太小） |
- 比预期（中性）好：×2 是真 perf bug（wave 量化尾巴），非仅设计瑕疵。**累计 292→596.6 = +104%（2.04×）。**
- 注：doc 称 occupancy query「含 TMEM」不准（TMEM 运行时 tcgen05.alloc，API 不可见）——但 smem 是 binding limiter，count 仍正确；单波仅对 uniform grouped 最优，uneven expert 需 dynamic scheduler（CLC）。commit `0edfa121`。

## P1 Step-11：TMA-store epilogue（= SonicMoE「async TMA store」）✅ **小-NK regime +23~48%**
**问题（597 重 profile）：** tcgen05 MMA 从 smem 读操作数走专用通路（不过 L1），所以 **L1/TEX 87% 几乎全是 epilogue 的 `reg→sA→global` LSU staging** + barrier stall 33.7%。
**修复：** 保留 reg→sA（reorder staging 必需），把 128-thread 手动 coalesced `sA→global` 循环换成 **async TMA store**（`SM90_TMA_STORE` / `cp.async.bulk.tensor`）：单一 A[M,I] descriptor（expert 隐含在行范围，无需 per-expert ptr）；`fence_view_async_shared`→elect-lane `copy(tma_store)`→`tma_store_arrive`；WAR 用 `tma_store_wait<0>`（移到下个 tile TMEM-load+silu 之后）替代 epi_done_bar → **tile N 的 store 与 tile N+1 的 epilogue 计算重叠**；单 sA buffer（smem ~0 增量）。OOB 由 descriptor box clamping 处理。镜像 stock `sm100_epilogue_tma_warpspecialized`。
**结果（正确性全 PASS n_fail=0）：**
| shape | manual epi | **TMA-store** | 提升 |
|---|---|---|---|
| (32,512,512,2048) M16384 用户 | 597 | 609 | +2%（compute-heavy，epilogue 本就 overlap MMA） |
| (32,1024,256,1024) M32768 | 440 | **543** | **+23%** |
| (32,2048,128,512) M65536 | 244 | **360** | **+48%** |
| (4,256,256,512) M1024 | 23 | 20.5 | −11%（极小问题，TMA-store setup overhead 占比大） |
- **小-K/mem-bound（epilogue 在关键路径）大胜 +23~48%；compute-heavy 用户 shape +2%**（epilogue 已被 double-buffer overlap，非关键路径）。极小问题轻微回退（可加 size 启发式 fallback，暂不做）。
- 用户 shape 累计 **292→609 = +109%（2.09×）**。新 stall（用户 shape）= reg→sA smem-scoreboard 40.4%（reorder staging 固有）。commit `adcd6ae5`。

## P1 Step-12：varlen-M（非均匀 expert，= SonicMoE varlen-M Grouped GEMM）✅
**动机：** 真实 MoE 的 per-expert token 数不均匀；原 kernel 假设 uniform Me（`e = m_tile / mtiles_per_expert`），在非均匀 expert 上结果错误。
**关键洞察 → 极小改动：** experts 若 **TileM-aligned**（token-rounding，SonicMoE 做法），packed 后 X/A 行偏移仍 = `global_m_tile*kTileM`（与 uniform 相同）——**只有 expert→W1-slice 映射变非均匀**。故唯一改动是 load warp 的 expert 推导（kernel 仅 1 处）：`e = m_tile_expert ? m_tile_expert[m_tile] : m_tile/mtiles_per_expert`，加一个 device 数组 `m_tile_expert[num_m_tiles]`（每 m-tile→expert id，host 前缀和构建）。X/A 偏移、W1 偏移、TMA-store epilogue 全不变。向后兼容（nullptr→uniform 快路径）。
**验证（B200，全 PASS n_fail=0）：**
| 测试 | 结果 |
|---|---|
| 小 varlen 全参考（8 expert 全覆盖, M=3840） | n_fail=0 ✓ 映射对每个 expert 正确 |
| uniform 回归（M=16384） | 608 TFLOPS（无回退） |
| **(32, m[1024,2048,3072], 512, 2048)** M=64512, 252 m-tiles | n_fail=0, **647.5 TFLOPS** |
| **(32, m[512,1024,1536], 512, 2048)** M=32256, 126 m-tiles | n_fail=0, **628.5 TFLOPS** |
- 非均匀 shape FLOPS（628~647）**高于** uniform M16384（608）：总 M 更大 → 更多 tile → 固定开销摊薄更好。
- test：`SWIGLU_VARLEN=<逗号 per-expert 模式>`（cycle across G，round 到 TILEM 倍数）；`m[min,avg,max]` 即 cycle {min,avg,max}。
- **下一步 CLC**：现 round-robin 已对 uniform-cost tile 均衡；CLC 预期中性（待测 dynamic vs static）。commit `07167133`。

## P1 Step-13：深度优化高方差 shape (32, m[512,4096,8192], 512, 2048)
**baseline（round-robin static, 4.2）= 673 TFLOPS**（M=132608, 518 m-tiles, n_fail=0）。ncu bound：**L1/TEX 95.18% / Memory 92.71% / Compute(SM) 62.19% / DRAM 8.16%**；stall = smem-scoreboard(reg→sA) 40.7% + CTA barrier 32.3%。→ **epilogue-smem-bound，非负载不均**（DRAM 8%，round-robin 已均衡）→ **CLC 对此 shape 确定中性**。
**bank-conflict 诊断（决定性）：** reg→sA scalar store **~90% wavefront 冲突**（store conflicts 10.15M / 11.2M wavefronts，~42 wf/inst = 32-way）；load 冲突 3（可忽略）。根因：sA row-major stride 64 bf16 = 128B = 正好 32 banks → 每行同 bank；tcgen05.ld 的 32 lane 写同列不同行 → 32-way 冲突。
**修复尝试（epilogue bound，3 次）：**
1. **TMA-store（Step-11）**：修的是 sA→global（async，不过 L1），**没碰 reg→sA 冲突**。
2. **128B swizzle (Layout_K_SW128) 单独用**：❌ scalar store + swizzle 反而 anti-align → 冲突 10.15M→**28.2M（2.8× 更糟）**，target 673→601（−11%），uniform 608→545。**revert**。（swizzle 是为 vectorized stmatrix 设计，scalar scatter 不适用。）
3. **stmatrix R2S 重写**（SM90_U32x2_STSM_N + swizzle + TMEM_LOAD→16dp256b1x，镜像 stock sm100 epilogue）：❌ **编译失败**——make_tiled_copy_D(stmatrix, 我的双累加器 tcgen05.ld 分区) rank 不匹配（copy_atom.hpp:244 "Rank too small"、logical_divide "Too many modes"）。hand-written 双累加器 epilogue 与 stmatrix 的 layout 组合冲突。**revert 回 673**。
→ **bank-conflict bound 在 hand-written kernel 中难修**：真正修复需 stmatrix（layout 组合受阻）；padding 能修但破坏 TMA-store layout。

## P1 Step-14：CUTLASS 4.2→4.5.1 升级 + 寄存器实验（用户要求）
- **submodule 升 v4.5.1**（local + remote build dir git checkout）。working kernel（无 stmatrix）在 4.5.1 **编译通过 + n_fail=0 + 同性能**（uniform 608.0、target 674.6，与 4.2 一致）→ **4.5.1 无 API break、不回退**。⚠️ 仅本 SwiGLU kernel 验证；全 TE 库在 4.5.1 上的构建未验证（后续）。
- **4.5.1 新 feature 调研**：`epilogue/fusion/` **无 gated/SwiGLU/GLU EVT 节点**（grep 空）→ route#2 仍需手写。example 92 = blackwell_moe_gemm（grouped/fp4，参考 kernel 非 drop-in）；4.3 simplified MoE API + MoEProblemShape(counts=varlen)；4.5 仅加 Snake activation。**无直接可用于 epilogue bound 的 feature。**
- **Q2 寄存器实验（EPI_REGS sweep, target shape）**：160→674.6 / 168→675.0 / 200→672.7 / 240→673.9 TFLOPS（全 ±0.3% 噪声内，n_fail=0）。→ **bump 无效**：1 CTA/SM 有 ~40K idle 寄存器，但 epilogue 在 160 不 spill，bound 是 bank conflict 非寄存器压力。EPI_REGS 抽象为可调宏（default 160）。commit `54efb668`。

## P1 Step-15：collective-epilogue 可行性调研（决定性结论）
深度 trace stock CUTLASS 4.5.1 sm100 epilogue + builder selector，对本 kernel 精确实例化（DisableSource, 2-SM, MmaTile(256,64,16)→CtaTile(128,64), ElementD=bf16, Acc=f32, N-major）：
- **builder 对本 config 选的不是 stmatrix，而是 `AutoVectorizingCopy`**：EpilogueTile=(128,32)→WarpTile=(32,32)→`num_dp=32`→`SM100_TMEM_LOAD_32dp32b32x`（已用）+ `AutoVectorizingCopyWithAssumedAlignment<128>`。stmatrix（`SM90_U32x{2,4}_STSM_N`）只能从 `16dp*` load 到达，需 `WarpM==16`，本 N=64 tile 永远 num_dp=32 → **stmatrix 不可达**。
- stock 的 conflict-free 靠 **vectorized copy + rank-2(EPI-tile)、带 PIPE 维的 swizzled `sD_epi`**（同时是 TMA-store 源）。
- **本 kernel 受阻根因 = CuTe rank 不匹配**：我的 acc = `partition_fragment_C` → **rank-3** `(MMA,MMA_M,MMA_N)` → tmem-copy Tiler rank-3；而 sA rank-2 → `partition_D(sA)` 报 "Rank too small"/"Too many modes"。stock Tiler 是 rank-2，与带 PIPE 的 rank-2 sD_epi + TMA 源一致。
- **要修 = store-path 重构**：rank-2-congruent tmem-copy + AutoVectorizingCopy R2S + SW128 swizzle + PIPE-bearing sD_epi + 配套 swizzled TMA-store descriptor。**且 dual-accumulator(gate||up) N-halving SwiGLU 不匹配 collective epilogue 的 single-accumulator-elementwise 模型**（其 fusion 是 acc→输出同形 element-wise，无 N 折半 + 双 TMEM-acc 组合）→ 全 collective epilogue 不能直接套。
**结论：** bank-conflict bound 的真正修复 = store-path 重构（≈ collective epilogue 的 store 部分，手工适配 dual-acc），中高工作量、风险高、4 次尝试已证 CuTe layout 难点。**当前收敛 673/608 TFLOPS（+109%），已在 4.5.1 验证。**

## P1 Step-16：store-path 重构第5/6次尝试（compose + AutoVectorizingCopy）— ❌ CUTLASS 编译墙
**思路（绕开 rank 墙）：** `tCsA = sA.compose(tTMc.layout())`（用 compose 而非 partition_D(sA) → 不触发 rank assert）+ `copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, rOut, tCsA)` 进 SW128-swizzled sA；128-bit 向量写匹配 swizzle 16B 周期 → 理论 conflict-free。逻辑上 element-i congruent（compose 把 swizzle 推过 basis-strided coord layout）。
**结果：** **编译失败**，错误全在 CUTLASS `cute/algorithm/copy.hpp` 内部（`copy_if/copy/prefetch already declared` + `Copy_Atom/AutoFilter undefined`）= copy.hpp 被**双重包含/解析顺序破坏**。对比：swizzled SmemLayoutA + **标量**存（Step-13 swizzle 尝试）能编译；只有加上 `compose`+`AutoVectorizingCopy` 才触发 copy.hpp 墙。加 mma_traits/copy_traits include 无效（非缺头，是 copy.hpp 自身解析顺序）。本地无法编译迭代 → 远程往返难诊断。
**裁决：** revert 回 clean 673（patch 存 `docs/F2_storepath_refactor_WIP.patch`）。**bank-conflict bound 经 6 次尝试（TMA-store/swizzle/stmatrix×2/investigation/compose）确认：在手写 dual-accumulator kernel 内不可解，真正修复需全 collective-epilogue 重写（而 dual-acc N-halving SwiGLU 不匹配其 single-acc 模型）。**
- 注：double-buffer(AccStages=2) 已部分 hide epilogue，673 是在 bank-conflict **存在**下达到的 → 修复的边际收益受 overlap 限制（非满 +40%）。
- **F2 route#2 收敛：673（高方差）/608（uniform）/628-647（不均匀）TFLOPS，+109%，varlen-M 验证，CUTLASS 4.5.1。下一步建议：接入 TE GroupedLinear 端到端验证（融合省 [M,2I] 中间 HBM 往返）。**

## P1 Step-17：⚠️ 前述"编译墙"= include-order BUG，非 CuTe wall —— 推翻"unsolvable"结论
**根因（决定性诊断）：** Step-16 及 swizzle/compose 的"编译失败"**不是 CuTe layout 墙，而是一个潜伏的 include-order bug**：kernel 把 `cute/arch/copy_sm90_tma.hpp` 等放在 `cute/tensor.hpp` **之前** → 它们 transitively 拉 `cute/algorithm/copy.hpp`，而 `Copy_Atom`（在 copy_atom.hpp，由 tensor.hpp 拉）尚未定义 → copy.hpp 解析失败 + 重复声明 copy_if/copy。**之前靠运气编过；某次容器/头状态变化让它暴露**，且**连干净 kernel（曾编过 608/674）都失败**。隔离证明：`cute/tensor.hpp` 单独编 OK，4 个 cute include 把 tensor.hpp 放最后则 fail，放最前则 OK。**修复：tensor.hpp 第一**（commit `18cd4de8`），608/674 恢复，n_fail=0。
**→ 在修好的 build 上重测 swizzle，得到真正答案：**
**padding（确认不可用）：** stride 66（无冲突）+ stride 72（16B-aligned, 4-way）→ TMA store 运行时 **"misaligned address"**。TMA 要求 dense 或 canonical-swizzle 的 smem，任意 padded pitch 不行 → **padding 与 TMA 根本不兼容**。
**SW128 swizzle + compose + AutoVectorizingCopy<128>（修好 build 后编译通过）：**
| 指标 | baseline | SW128 swizzle |
|---|---|---|
| store bank conflicts | 10.15M | **370K（27× 更少！）** |
| store wavefronts | 11.2M | 1.43M（8× 更少） |
| **target 性能** | 672 | **674（持平）** |
| uniform 性能 | 608 | 607（持平） |
| **L1/TEX** | 95.18% | **95.11%（持平）** |
| Compute(SM) | 62.2% | 62.9% |
- **结论1：bank-conflict 可解**（swizzle 27× 更少冲突）——"6 次尝试 unsolvable" 是 **build bug**，非真墙。
- **结论2：解了也不快**（性能持平 674≈672，L1/TEX 仍 95%）——**bank-conflict 是高-SOL 假象，非 wall-clock bound**：`AccStages=2` double-buffer 让 epilogue 与下一 tile MMA **重叠**，冲突的 store 并发执行、不延长关键路径。
- （swizzle 的 `sA.compose(tTMc.layout())` 映射有正确性 bug，n_fail 高；但 store 工作量相同→timing 结论不变。已 revert 回 clean 672。）
- **真正 bound = 2-SM occupancy 12.5% + MMA/memory throughput，非 epilogue。kernel 真正收敛在 673。** swizzle patch 价值仅"消冲突"（不提速），如需 clean 无冲突版可修 compose 正确性（低 ROI）。

## P1 Step-18：occupancy 12.5% 为何这么低 + 能否提高（实测验证）
**为何 12.5%：** achieved = 1 block/SM × 8 warp/CTA = 8/64 = 12.5%。
**实测对比（CLEAN kernel, 高方差 target, n_fail=0）：**
| config | Block Limit Reg/Smem | Theoretical | **Achieved** | 性能 |
|---|---|---|---|---|
| baseline (kStages=16, EPI_REGS=160) | 1 / 1 | 12.5% | 12.5% | 673.6 |
| 2-block 尝试 (kStages=8, EPI_REGS=128) | **2 / 2** | **25%** | **12.50%（仍是！）** | 617.4（更慢） |
- **theoretical 可升到 25%**：降寄存器（EPI_REGS 160→128）+ 降 smem（kStages 16→8）→ Block Limit Reg/Smem 都变 2 → theoretical 25%。所以**不是纯寄存器/smem 上限**。
- **但 achieved 仍 12.50%**（即使 theoretical 25%）：调度器每 SM-pair 只放 **1 个 cluster**（=1 block/SM）。**这才是真上限 = 2-SM `cta_group::2` cluster gang-scheduling 的 co-residency**（每 SM-pair 1 cluster）。**实测证实**（theoretical 25% vs achieved 12.5%）——之前这个 claim 是对的。
- **且强行追求会更慢**：kStages=8/EPI128（启用 theoretical 25%）反而 617<673——牺牲了 pipeline 深度 + epilogue 寄存器，而 achieved 没变。
- **结论：** 在 **2-SM 设计内（用户要求，为 B-operand 共享 + M-tile 翻倍），achieved occupancy 硬上限 12.5%（cluster co-residency），实际不可升**。且 occupancy 对 **warp-specialized kernel 是错指标**（8 warp 中 load1+MMA1+epi4+idle2，多数故意 idle）——真正看 **Compute(SM)=62%（tensor-core 利用率）**。1-SM kernel 可更高 occupancy 但失去 2-SM B-共享，净更差。**occupancy 非 bound（如 bank-conflict 一样是假象）；kernel 收敛 673。**

## P1 Step-19：vs TE GroupedLinear（同 shape、参数/FLOP 对齐）—— 决定性 baseline 对比
**对齐核实（B200 bf16，target shape M=132608, in d=2048, out 2I=1024）：** params 67.11M=67.11M ✓ · FLOP 556.2G=556.2G（=4·M·I·d）✓ · output [M,512]=[M,512] ✓。fused 的 2 GEMM（gate+up）== TE 的单个 [M,d]·[d,2I]。
| 路径 | ms（3 run） | TFLOPS |
|---|---|---|
| TE GroupedLinear（仅 GEMM） | 0.519/0.485/0.484 | **~1090-1150** |
| TE GroupedLinear + (torch) SwiGLU（e2e） | 0.794/0.813/0.848 | ~656-701 |
| **fused kernel（GEMM+SwiGLU 一趟）** | 0.824 | 673 |
- **TE 的 grouped GEMM 比 fused 的有效 GEMM 吞吐快 ~1.65×（1120 vs 673）**。TE/cuBLAS grouped GEMM 远比手写 2-SM warp-spec + 重 SwiGLU-epilogue 高效。
- **e2e：fused ≈ 打平 TE+SwiGLU**（~0.82ms 双方，均值 0.818 vs 0.824，噪声内）。fusion 省掉 [M,2I] 往返 + SwiGLU pass，把慢-GEMM 的 fused 拉回打平——**但不 win**。（之前"TE 快 4%"是噪声。）
- **裁决：route#2 fusion 不 beat baseline，只打平**。standalone kernel 无法补上 1.65× GEMM gap（2-SM occupancy 封顶；bank-conflict/occupancy 均已证非 bound）。
- **真正的赢家方向 = TE 的 1120-TFLOPS GEMM + fused SwiGLU epilogue**（TE GEMM 效率 × fusion IO 省），同时 beat standalone kernel 和 TE+separate-SwiGLU。**继续磨手写 kernel 无意义；杠杆在把 SwiGLU 融进 TE/cuBLAS 的 GEMM。**
- fused kernel 仍有价值：不物化 [M,2I]（activation-memory 省，对内存受限 MoE 训练）；正确/已验证参考（varlen-M, 4.5.1, n_fail=0）。
