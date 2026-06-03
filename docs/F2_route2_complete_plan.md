# F2 · route#2 — In-Kernel Gate/Up Pairing 完整优化方案
**SwiGLU 在 up-proj grouped GEMM 内寄存器级融合 · W1 保持标准布局 · 对 Muon/优化器/checkpoint 零影响**

> 决策依据：性能 route#2 ≈ 交错-B（都 1·IT、单 GEMM、寄存器内 silu·mul），但 route#2 **不改 HBM 权重布局** → 优化器(含 Muon)/checkpoint/TP 全程零感知。代价是最大的 kernel 工作量（自研 dual-accumulator gated mainloop，CUTLASS 无现成 collective，≈ SonicMoE 的 `gemm_gated`）。
> 现状底座：`transformer_engine/common/gemm/cutlass_grouped_gemm.cuh` 已有可用的 SM100 grouped GEMM（`KernelPtrArrayTmaWarpSpecialized2SmSm100`，2-CTA tcgen05，前向 256×256×64）。本方案在其上扩出 gated 变体。

---

## 0. 目标与不变量
- up-proj 前向：`A[T,I] = SwiGLU(X @ W1ᵀ)`，`W1 ∈ [2I,d]` 标准拼接（行 0..I-1=gate，I..2I-1=up）。`A[:,j] = silu(gate[:,j]) · up[:,j]`。
- **gate/up 永不落 HBM，只写 A[T,I]**：activation 相关 HBM `5·IT → 1·IT`，up-proj 输出 store 减半。
- **W1 在 HBM 始终标准 `[2I,d]`**（不置换）→ 优化器/Muon/checkpoint/TP 不动。
- bf16/fp16 in、**fp32 累加**、无 fp8/fp4。
- 数值对齐参考实现（单 GEMM + torch SwiGLU）。

## 1. Kernel 结构 — dual-accumulator gated grouped GEMM
对每个 expert g 的每个 A 输出 tile `[BM tokens, BN channels]`（起点 m0,j0）：
```
gate_acc[BM,BN] = Σ_k X[m0:+BM, k] · W1[g][   j0:+BN, k]   // gate 行块
up_acc  [BM,BN] = Σ_k X[m0:+BM, k] · W1[g][ I+j0:+BN, k]   // up 行块（行偏移 +I）
A[m0:+BM, j0:+BN] = silu(gate_acc) · up_acc                // fp32 → cast → 写
```
- **X-tile（token 操作数）两路 MMA 共享** → 不增加 token 端 HBM。
- W1 总读字节 = 一次正常 `[2I,d]` 读（gate 块 + up 块即这些 channel 的全部 W1）。
- 只写 A（H 的一半）。

## 2. CUTLASS 实现（SM100 tcgen05，grouped/ptr-array）
CollectiveBuilder 只产单累加器 mainloop，故需**自研 mainloop**。在现有 SM100 grouped collective 基础上：
- 持有**两个 B 操作数 TMA descriptor**（gate、up），由**同一个 W1 ptr-array**加 0 / +I 行偏移构造；
- 分配**两个 TMEM 累加器**；
- K-loop 内：load X-tile（共享）+ gate-W1-tile + up-W1-tile，发**两条 tcgen05 MMA** 累加到 `acc_gate` / `acc_up`；
- 自研 epilogue：从 TMEM 取两累加器，`silu(acc_gate)·acc_up` → 写 A。

> 这就是 `gemm_gated` 的结构；CUTLASS 不发货，需在 SM100 collective 原语（tcgen05 MMA atom、TMA、warp-specialized pipeline）上自己拼。**这是本方案的主要工作量。**

### 🚦 风险门（必须最先验，决定 go/no-go）
- **G1 · TMEM 容纳两累加器**：tcgen05 累加器在 TMEM（128 lane × 512 col）。两个 `[BM,BN]` fp32 累加器（如 2-SM 下 BM=256、BN=128）≈ 2×128 = 256 col ≤ 512，加 pipeline 的 SMEM/TMEM carveout 后是否仍 fit。先用最小编译验证；不够则缩 BN。
- **G2 · ptr-array 下的双 TMA B-load + 2-SM cluster**：两条 TMA descriptor（gate/up）能否在 grouped 的 W1 ptr-array 上以固定 +I 行偏移构造，且 2-CTA multicast 对两个 B 操作数仍成立。（route#2 版的 OQ。）
- 任一门不过 → **回退 A1**（3·IT、零 Muon、stock CUTLASS 可建），route#2 待 CUTLASS 约束解除后再上。

## 3. SwiGLU epilogue（寄存器级）
- `silu(g)=g·sigmoid(g)` 在 fp32；× up 在 fp32；cast bf16/fp16；写 A。
- 复用 `cutlass::epilogue::thread` 的 SiLU + multiply，或对两 TMEM 累加器写一个小 EVT。难点在喂它的 dual-accumulator mainloop，epilogue 本身简单。

## 4. TE 前向集成
- 新增 launch 入口 `cutlass_grouped_gemm_swiglu(X_list, W1_list[2I,d], A_out[T,I], m_splits, …)`，内部按 +I 行偏移取 gate/up。
- `grouped_linear.py` / MLP 前向：当 activation==SwiGLU 且 SM100 且开关开启，把 up-proj（现在「general_grouped_gemm 产 H + 单独 SwiGLU」）路由到融合路径直接产 A。开关 `NVTE_FUSED_SWIGLU_GROUPED_GEMM=1`。
- A[T,I] 喂现有 down-proj，不变。

## 5. 反向（须协同设计，接 B1）
H 不再 materialize，反向需要 gate/up 来算 `dgate = dA·up·silu'(gate)`、`dup = dA·silu(gate)`：
- **F2 先行版（简单正确，零 Muon）**：前向**仍 cache H[T,2I]**（标准布局）；反向从 cached H 算 dgate/dup → 组成**标准布局 dH[T,2I]** → `dW1 = Xᵀ@dH`（标准）、`dX = dH@W1`（标准）。**优化器/Muon 完全不动。**（暂不省 H 显存——显存优化按既定排在最后。）
- **峰值版（= B1/MEM）**：不 cache H，反向用两路 GEMM 重算 gate/up（dH overlap kernel），届时仍是标准布局 dH/dW1。

## 6. 验证
- **数值**：`A_fused` vs `general_grouped_gemm(H) + torch SwiGLU` 逐元素（bf16/fp16）；反向 dX/dW1 vs 参考 GroupedLinear+SwiGLU。扩 `tests/pytorch/test_grouped_linear.py` 加 SwiGLU-fused 用例。
- **性能**：up-proj 前向 融合 vs（GEMM+单独 act）；端到端 MoE 前向。用 `general_grouped_gemm` 直测口径（避免 autograd/cast 稀释，见 `qa/wgrad_direct_bench.py` 的教训）。
- 确认只写 A、无 H 写、无单独 activation kernel（profiler 核对）。

## 7. 分阶段里程碑（风险门优先）
| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | G1(TMEM 两累加器) + G2(双 TMA ptr-array) 构建期 spike | 两门通过；否则回退 A1 |
| **P1** | dual-accumulator gated mainloop（单 expert / uniform） | 数值对齐参考 |
| **P2** | grouped(ptr-array) + 2-SM + varlen tokens | `test_grouped_linear` 数值过 |
| **P3** | TE 前向集成（开关）+ perf | up-proj 前向提速、只写 A |
| **P4** | 反向（cache-H 版）fwd+bwd 数值 + 优化器/Muon 不受影响 | dW1/dX 标准布局、数值过 |
| **P5**（后续=MEM/B1） | 去掉 H cache，改重算 | 显存降，数值过 |

## 8. 回退
P0 门若不过（TMEM/ptr-array 约束）→ 用 **A1**（2 GEMM + `Sm90AuxLoad` EVT）：3·IT、零 Muon、stock CUTLASS 可建，拿约一半收益；route#2 作为约束解除后的峰值目标。

## 9. 零-Muon 论证（核心）
W1 以标准 `[2I,d]` 读入，up 块靠 **kernel 内 +I 行偏移**取，HBM/模块/优化器 state/checkpoint **无任何置换**。反向产**标准布局 dW1**（dH 标准）。Muon 正交化的还是今天这同一个 `[2I,d]` 矩阵；TP 切分同今天。→ 优化器/Muon/checkpoint/TP **零改动**。

## 10. 性能预期
按 B200 trace，SwiGLU 激活占 MoE 前向 ~7.5%(E=128)→12.6%(E=32)。route#2 基本全收（gate/up 不落盘、只写 A）+ up-proj store 减半。相对基线（GEMM+单独 act），up-proj 前向预计提升 ~该 activation 比例 + store 减半的部分；细粒度/memory-bound 区间最明显。
