# MoE Grouped-GEMM 优化测试配置

用于驱动 `qa/` 下的 grouped-GEMM / wgrad benchmark,覆盖真实 MoE 训练里出现的 **Var-M(token 不均)** 与 **Var-K(token 进 reduction 维)** 两类 GEMM。

---

## 场景 1(canonical):4K MoE

| 参数 | 值 | 说明 |
|---|---|---|
| BS | 4 | batch |
| seq | 4096 | 序列长度(4K) |
| H | 512 | hidden |
| FFN (I) | 2048 | FFN 中间维(SwiGLU 单分支) |
| E | 256 | 全局 expert 数 |
| EP | 8 | expert parallel |
| TopK | 12 | 每 token 路由数 |

### 核心统计(派生)

| 量 | 计算 | 值 |
|---|---|---|
| Tokens | BS×seq = 4×4096 | **16384** |
| Routed | Tokens×TopK = 16384×12 | **196608** |
| 本地 expert G | E/EP = 256/8 | **32** |
| avg tokens/expert | Routed/E = 196608/256 | **768** |
| per-rank routed | Routed/EP = 32×768 | 24576 |
| 不均 (min/avg/max) | 长尾 | **650 / 768 / 900** |

> 一句话:**FW/DGrad 看 M 不均,WGrad 看 K 不均(Mi = token load)。**

---

## 每个 kernel 的 GEMM 形状(per-expert,M × K × N)

`M×K×N` = `[M,K] @ [K,N] → [M,N]`,`Mi` = 该 expert 的 token 数(650/768/900)。

### Forward

| Kernel | M × K × N | 收缩维 K | 变长维 | 类型 |
|---|---|---|---|---|
| Gate(router,dense,1 group) | 16384 × 512 × 256 | 512 | — | 普通 GEMM(cuBLAS) |
| Up(grouped) | Mi × 512 × **2048** | 512(H,uniform) | **M** | Var-M |
| Down(grouped) | Mi × 2048 × 512 | 2048(I,uniform) | **M** | Var-M |

> **SwiGLU 融合 kernel** 的 FC1 输出是 gate‖up = **2I = 4096**:即 `Up` 在融合路径下是 `Mi × 512 × 4096`,激活后降回 `Mi × 2048` 再喂 Down。非融合(分两次 GEMM)时按上表 N=2048。

### Backward — DGrad(Var-M,镜像 FW)

| Kernel | M × K × N | 收缩维 K | 变长维 |
|---|---|---|---|
| Up dgrad | Mi × 2048 × 512 | 2048 | M |
| Down dgrad | Mi × 512 × 2048 | 512 | M |

`dX = dY @ Wᵀ`,收缩维是对应 FW 的输出维,uniform;M=Mi 变长。

### Backward — WGrad(Var-K,关键路径)

| Kernel | M × K × N | 收缩维 K | 变长维 |
|---|---|---|---|
| Up WGrad | 512 × **Mi** × 2048 | **Mi(token,ragged)** | **K** |
| Down WGrad | 2048 × **Mi** × 512 | **Mi(token,ragged)** | **K** |

`dW = Xᵀ @ dY`,M/N 固定为权重维(H/I),token 数 Mi 落进收缩维 K → ragged-K。

---

## 映射到本仓 kernel + 数值类型

| 路径 | 调度 kernel | dtype | 备注 |
|---|---|---|---|
| FW Up/Down、DGrad Up/Down | [`cutlass_grouped_gemm`](../transformer_engine/common/gemm/cutlass_grouped_gemm.cuh)(uniform-K 快路) | BF16 | K∈{512,2048} 均 %128==0,走快路;var-M 由 m_splits 天然支持 |
| FW 融合(Up+SwiGLU[+quant]) | [`cutlass_grouped_gemm_swiglu.cuh`](../transformer_engine/common/gemm/cutlass_grouped_gemm_swiglu.cuh) | BF16 / MXFP8(SM100) | **当前 kernel 要求 uniform-Me** → 用 Me=768 代表点;I=2048→W1 N=4096 |
| WGrad Up/Down | [`cutlass_grouped_gemm_varlen_k`](../transformer_engine/common/gemm/cutlass_grouped_gemm.cuh)(varlen-K) | BF16 | ragged-K = token 数;`!transa && transb && grad` NT wgrad |

---

## FLOP(avg 点:G=32, Me=768, sumK=sumM=24576)

每条主 GEMM ≈ `2·sumTok·K·N`:

| Kernel | FLOP |
|---|---|
| FW Up(N=2048) | 2·24576·512·2048 ≈ **51.5 G** |
| FW Up 融合(N=4096,gate‖up) | ≈ **103 G** |
| FW Down | 2·24576·2048·512 ≈ **51.5 G** |
| DGrad Up / Down | ≈ **51.5 G** each |
| WGrad Up / Down | 2·512·2048·24576 ≈ **51.5 G** each |

> 单个 expert 的 GEMM 很小(Mi≈768),整体 **latency / 小-M-tile 效率受限**,瓶颈在 group 调度、tail 不均与 epilogue,而非纯算力。这正是优化目标。

---

## 可直接运行的 bench 命令

> 设 `NVTE_USE_CUTLASS_GROUPED_GEMM=1` 走 CUTLASS;不设则对比 cuBLAS。

### FW / DGrad(Var-M,uniform-K)

```bash
# FW Up  (H=512 → FFN=2048)
NVTE_USE_CUTLASS_GROUPED_GEMM=1 python qa/grouped_gemm_bench.py \
    --experts 32 --K 512  --N 2048 --mper 768 --dtype bf16
# FW Up 融合宽度 (gate‖up = 2I)：--N 4096
# FW Down (FFN=2048 → H=512)
NVTE_USE_CUTLASS_GROUPED_GEMM=1 python qa/grouped_gemm_bench.py \
    --experts 32 --K 2048 --N 512  --mper 768 --dtype bf16
```

### WGrad(Var-K,ragged)

token 数对齐到 128 倍数(650→640, 768→768, 900→896):

```bash
# Up WGrad  (M=H=512, N=FFN=2048, K=Mi ragged) —— 端到端(含 autograd)
python qa/wgrad_ragged_bench.py --g 32 --m 512  --n 2048 --mink 640 --maxk 896
# Down WGrad (M=FFN=2048, N=H=512)
python qa/wgrad_ragged_bench.py --g 32 --m 2048 --n 512  --mink 640 --maxk 896

# kernel-only(无 autograd/cast,纯 general_grouped_gemm NT)—— 看 kernel 本身
python qa/wgrad_direct_bench.py --g 32 --m 512  --n 2048 --mink 640 --maxk 896
python qa/wgrad_direct_bench.py --g 32 --m 2048 --n 512  --mink 640 --maxk 896
```

---

## 扩展 sweep 轴(覆盖更多真实场景)

固定 4K 为基准点,按需在以下轴上扫:

| 轴 | 取值 | 触发的场景 |
|---|---|---|
| **token 规模** | seq=1K / 4K / 8K / 16K(或 BS×seq) | decode 小 batch → 长上下文 prefill;avg/expert 从 ~190 → ~3000 |
| **不均强度** | uniform(640) / 中尾(640–896) / 重尾(256–1280) | 压 load-balance 与 tail 效率(min 决定空泡,max 决定 wave) |
| **dtype** | BF16(全路径) / MXFP8(仅融合 FW,SM100) | 量化融合 epilogue |
| **K 对齐** | 128 倍数 vs 非对齐余数 | varlen-K 的 partial-K tile 处理 |

> 重尾点尤其重要:`min` 决定小 group 的空泡/启动占比,`max` 决定最长 group 的 wave 数,二者拉开时最能暴露 grouped 调度与本仓 varlen-K / persistent grid 的优劣。

---

## 关键注意事项

1. **融合 SwiGLU kernel 是 uniform-Me 的**:真实场景 var-M,测试时用 `Me=avg=768` 作代表点,或 pad-to-max=896/capacity;不要直接喂 ragged token 给融合 kernel(当前不支持)。
2. **SwiGLU 的 2× 宽度**:融合 FC1 输出 = 2I = 4096(gate‖up),算 FLOP / 配 W1 形状时别漏。
3. **WGrad 的 K 即 token 数**:Mi 进收缩维,所以 wgrad 是 ragged-K,走 `cutlass_grouped_gemm_varlen_k`;FW/DGrad 是 ragged-M,走 uniform-K 快路。
4. **token 数取 128 倍数**:CUTLASS 路径按 128 对齐最稳;非对齐留作 partial-tile 专项测试。
5. **Gate 是 dense GEMM**:1 个 group、无 SwiGLU,通常不是优化重点,但属于完整 step 的一部分(cuBLAS)。
