# SonicMoE 标准测试配置（后续测试都按这个来）

所有 SonicMoE / MoE grouped-GEMM 的性能与正确性测试，统一按下面的环境 + 形状 + 方法跑。

## 1. 环境（生产对齐）

| 项 | 值 |
|---|---|
| 容器 | **`sonic-moe-2605`** = `nvcr.io/nvidia/pytorch:26.05-py3` |
| CUDA / cuBLAS | **CUDA 13.2 / cuBLAS 13.4.1**（graph-safe grouped GEMM 需要 cuBLAS ≥ 13.3；25.10=13.1、26.02=13.2.1 都太旧） |
| torch / python | torch 2.12 / py3.12 |
| TE | editable 安装于 `/data1/min.yang/te_build` |
| GPU | **必须用空卡**（0 MiB / 0% util）。手写 2-SM tcgen05 kernel 强 compute-bound、对频率极敏感，共享/限频卡会把结论测反（412 限频 → 652 空卡）。自动挑空卡：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \| sort -t, -k3 -n -k2 -n \| head -1 \| cut -d, -f1` |
| import 顺序 | **`import transformer_engine` 必须在 `import transformer_engine_torch` 之前**（否则 undefined-symbol：common lib 要先 RTLD_GLOBAL 加载） |

## 2. 形状（真实 per-card 4K-MoE，来自配置表）

| 维度 | 值 | 说明 |
|---|---|---|
| Batch / 序列 | BS 4 / 4K | tokens = 4×4096 = 16384 |
| EP / 专家数 / TopK | EP8 / 256E / Top-12 | routed = 16384×12 = 196608 |
| 每卡本地专家 **G** | **32** | 256 ÷ EP8 |
| Hidden **H (= d = K)** | **2048** | GEMM 输入/收缩维 |
| Intermediate **I** | **512** | SwiGLU 单分支；FC1 gate‖up = **2I = 1024** |
| 每卡 tokens **M** | **24576** | 196608 ÷ EP8 = 32×768 |
| 平均 token/expert **Me** | **768** | 3×256，非零且 256 对齐 |
| dtype | **bf16** | |

GEMM 形状：up-proj `[M,2048]×[1024,2048]ᵀ→[M,1024]`（N=2I=1024, K=H=2048, 103.1 GFLOP）；
down-proj `[M,512]×[2048,512]ᵀ→[M,2048]`（N=H=2048, K=I=512, 51.5 GFLOP）。

env 覆盖（脚本里读）：`MOE_G=32 MOE_D=2048 MOE_I=512 MOE_ME=768`。

## 3. 四个 grouped-GEMM 后端（统一对比口径）

| 后端 | 开关 |
|---|---|
| 1. legacy cuBLAS（multi-stream） | ops: monkeypatch `_is_graph_safe_path_supported→False`, `NVTE_USE_CUTLASS_GROUPED_GEMM=0` |
| 2. graph-safe cuBLAS 13.4（ops 默认，最快非融合基线） | ops 默认（SM100+bf16）/ module `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1` |
| 3. CUTLASS F0 | `NVTE_USE_CUTLASS_GROUPED_GEMM=1`（+ ops 需 monkeypatch graph-safe→False） |
| 4. **FUSED_MOE**（SonicMoE 融合 up+SwiGLU） | `NVTE_USE_FUSED_MOE=1`（op 层）；kernel 直调用 `tex.te_cutlass_grouped_swiglu` |

`NVTE_USE_FUSED_MOE` = op-fusion 层开关（两者分开）。

## 4. 标准脚本

| 脚本 | 用途 |
|---|---|
| `qa/te_fused_moe_e2e_test.py` | drop-in 正确性（fwd + 全梯度 vs torch，n_fail=0）+ 单后端 perf |
| `qa/grouped_gemm_backends_bench.py` | 整段 MoE 前向：legacy / CUTLASS F0 / graph-safe / 2-separate / fused 同跑对比 |
| `qa/run_glin_bench.sh` | 标准 `benchmarks/linear/benchmark_grouped_linear.py`：单 GroupedLinear 三后端 |
| `qa/moe_4backends_fwdbwd.py` | **ops MoE 四后端 FWD + FWD+BWD**（MoE-layer 级，主对比脚本） |
| `qa/op_overhead_bench.py` | forward_fused_moe op 封装开销拆解 |

## 5. 当前关键结论（空卡，正确形状）
- 融合 **kernel 直调 652 TFLOP/s = 1.55× vs graph-safe cuBLAS 13.4**（FWD）。
- 融合 **op**（forward_fused_moe.py）因 ~39% Python 封装开销退回 401 ≈ 持平；反向是逐 op 回退（重算 up-proj）→ FWD+BWD 略输。
- 要让端到端吃到 1.55×：① 砍 op 封装开销；② 做 B1 融合反向。
