# CopassV4 B300 优化测试教程

> 说明：远端脚本和数据路径里使用的是 `compass` / `CompassV4` 命名；本文按需求标题写作 `CopassV4`。测试记录来自 `10.251.210.1:/data1/baibaifan`，机器为 8x `NVIDIA B300 SXM6 AC`。

## 1. 测试目标

这组 B300 测试用于验证 CopassV4/CompassV4 MoE 训练优化是否可用、是否稳定，以及在不同优化组合下的 step time / tokens/s 表现。重点不是跑完整训练，而是用生产形状短跑，确认每个优化开关的收益和风险。

主要覆盖：

- TE/CUTLASS grouped GEMM：`NVTE_USE_CUTLASS_GROUPED_GEMM=1`
- MoE permute/router 融合：`--moe-permute-fusion`、`--moe-router-fusion`
- HybridEP/flex dispatcher：`--moe-token-dispatcher-type flex`、`--moe-flex-dispatcher-backend hybridep`
- HybridEP 内部 permute 融合：`--moe-permute-fusion-into-hybridep`
- CUDA Graph：`--cuda-graph-impl transformer_engine`、`--cuda-graph-scope attn moe_router moe_preprocess`
- loss 融合：`--cross-entropy-loss-fusion`、`--cross-entropy-fusion-impl te`
- optimizer/通信 overlap：`--optimizer muon`、`--overlap-grad-reduce`、`--overlap-param-gather`

## 2. 远端记录位置

```bash
smc toc 10.251.210.1 'ls -lah /data1/baibaifan'
smc toc 10.251.210.1 'sed -n "1,360p" /data1/baibaifan/compass-mcore/run.sh'
smc toc 10.251.210.1 'find /data1/baibaifan/tensorboard -maxdepth 1 -type f -printf "%f %s\n" | sort'
```

关键文件：

- 启动脚本：`/data1/baibaifan/compass-mcore/run.sh`
- TensorBoard：`/data1/baibaifan/tensorboard/events.out.tfevents.*`
- 数据：`/data1/baibaifan/data/code-part124`
- tokenizer：`/data1/baibaifan/data/compass_v4_tokenizer_200k`
- cache：`/data1/baibaifan/data-caches`
- checkpoint：`/data1/baibaifan/models`

## 3. B300 环境确认

先确认机器确实是 B300，并记录驱动版本：

```bash
smc toc 10.251.210.1 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 8'
```

当前记录：

```text
NVIDIA B300 SXM6 AC, 275040 MiB, 580.159.04
```

再确认 NCCL 网卡、数据路径和 tokenizer 是否存在：

```bash
smc toc 10.251.210.1 'ls -lah /data1/baibaifan/data /data1/baibaifan/data/compass_v4_tokenizer_200k /data1/baibaifan/data-caches'
smc toc 10.251.210.1 'cd /data1/baibaifan/compass-mcore && git rev-parse HEAD && git status --short'
```

## 4. 生产形状

`run.sh` 使用的是单机 8 卡 B300、EP8 的 CopassV4 MoE 训练形状：

| 配置 | 取值 |
|---|---:|
| GPU | 8x B300 |
| seq length | 4096 |
| hidden size | 2048 |
| layers | 45 |
| attention heads | 32 |
| GQA groups | 4 |
| MoE experts | 256 |
| EP size | 8 |
| ETP size | 1 |
| topK | 12 |
| MoE FFN hidden | 512 |
| shared expert intermediate | 1024 |
| micro batch | 4 |
| global batch | 3200 |
| dtype | bf16 |
| optimizer | muon |

MoE 层形状要点：

- 每 step token 数：`global_batch_size * seq_length`
- 当前满配：`3200 * 4096 = 13,107,200 tokens/step`
- routed tokens：`tokens * topK`
- 单卡本地专家：`256 / EP8 = 32`

## 5. 推荐测试矩阵

每个 case 建议短跑 20 到 50 step，前 3 step 作为 warmup，不纳入性能统计。遇到 OOM、NCCL hang、CUDA graph capture error，保留报错日志并停止该 case。

| Case | 目的 | 关键开关 |
|---|---|---|
| A. baseline | 建立可跑基线 | 关闭 CUDA Graph、关闭 HybridEP 内部融合，可保留 `--moe-grouped-gemm` |
| B. CUTLASS grouped GEMM | 验证 TE grouped GEMM 后端 | `NVTE_USE_CUTLASS_GROUPED_GEMM=1` |
| C. MoE fusion | 验证 router/permute 融合 | `--moe-permute-fusion`、`--moe-router-fusion` |
| D. HybridEP | 验证 flex + HybridEP | `--moe-token-dispatcher-type flex`、`--moe-flex-dispatcher-backend hybridep` |
| E. HybridEP permute fusion | 验证 dispatcher 内融合 | `--moe-permute-fusion-into-hybridep`、`--moe-hybridep-num-sms 32` |
| F. CUDA Graph | 验证 graph capture 收益 | `--cuda-graph-impl transformer_engine`、`--cuda-graph-scope attn moe_router moe_preprocess` |
| G. full optimization | 当前脚本配置 | B+C+D+E+F，加 CE fusion、overlap、Muon |

不要一次只看 full optimization。B300 上很多优化会互相影响，必须用矩阵拆开看，否则无法判断是 grouped GEMM、HybridEP 还是 CUDA Graph 在贡献收益。

## 6. 运行方法

进入远端工程目录：

```bash
smc toc 10.251.210.1
cd /data1/baibaifan/compass-mcore
bash run.sh
```

如果只跑测试，不想保存 checkpoint，可以临时把 `run.sh` 里的这些参数调小：

```bash
--train-iters 30
--save-interval 100000000
--eval-interval 1000000000000000
```

如果要测试不同 global batch，在 `TRAINING_ARGS` 中改：

```bash
--global-batch-size 1600
--global-batch-size 2400
--global-batch-size 3200
```

每个 case 建议单独设置 TensorBoard 目录，避免 event 混在一起：

```bash
TENSORBOARD_DIR=/data1/baibaifan/tensorboard/case_full_gbs3200
```

当前 `run.sh` 默认写到：

```bash
/data1/baibaifan/tensorboard
```

## 7. 结果读取

TensorBoard event 中当前没有显式写入 throughput tag；可用 `lm loss` 的 wall time 估算 step time，因为 `--log-interval 1` 每 step 都会写一次。

复制 event 到本地：

```bash
mkdir -p /tmp/baibaifan-tensorboard
smc scp '10.251.210.1:/data1/baibaifan/tensorboard/events.out.tfevents.*' /tmp/baibaifan-tensorboard/
python3 -m pip install --user tensorboard
```

解析脚本：

```python
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import statistics

base = Path("/tmp/baibaifan-tensorboard")
seq_len = 4096

for f in sorted(base.glob("events.out.tfevents.*")):
    if f.stat().st_size < 1000:
        continue
    ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "lm loss" not in tags or "batch-size" not in tags:
        continue

    losses = ea.Scalars("lm loss")
    batch = ea.Scalars("batch-size")[-1].value
    if len(losses) < 5:
        continue

    times = [v.wall_time for v in losses]
    dts = [times[i] - times[i - 1] for i in range(1, len(times))]
    warm = dts[3:]
    tokens_s = [batch * seq_len / dt for dt in warm]

    print(
        f.name,
        "steps=", len(losses),
        "GBS=", int(batch),
        "median_step_s=", round(statistics.median(warm), 2),
        "avg_step_s=", round(statistics.mean(warm), 2),
        "tokens/s=", round(statistics.mean(tokens_s)),
        "loss=", round(losses[0].value, 2), "->", round(losses[-1].value, 2),
    )
```

## 8. 当前测试记录摘要

以下为 `/data1/baibaifan/tensorboard` 中已有 event 按 wall time 估算的结果。因为没有 stdout 日志，表中性能来自 event 时间差；正式报告建议补充 Megatron stdout 中的 iter time / throughput。

| event suffix | start | steps | GBS | median step(s) | avg step(s) | tokens/s | lm loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `40889` | 14:28 | 32 | 256 | 16.74 | 16.74 | 62,633 | 12.61 -> 9.16 |
| `48438` | 14:48 | 13 | 1600 | 77.29 | 77.30 | 84,782 | 12.61 -> 11.32 |
| `57567` | 15:23 | 9 | 1600 | 77.25 | 77.24 | 84,852 | 12.61 -> 12.06 |
| `61493` | 15:41 | 8 | 1600 | 74.71 | 74.67 | 87,773 | 12.61 -> 12.21 |
| `65847` | 15:57 | 7 | 2400 | 94.27 | 94.26 | 104,285 | 12.61 -> 12.33 |
| `70924` | 16:12 | 8 | 3200 | 114.98 | 115.00 | 113,975 | 12.61 -> 12.20 |
| `79967` | 16:39 | 8 | 2400 | 130.03 | 130.05 | 75,590 | 12.61 -> 12.21 |
| `83949` | 17:02 | 15 | 1600 | 222.78 | 222.95 | 29,395 | 12.61 -> 10.99 |
| `87936` | 18:46 | 7 | 3200 | 110.75 | 110.74 | 118,361 | 12.61 -> 12.33 |
| `91921` | 19:08 | 8 | 3200 | 118.81 | 118.82 | 110,313 | 12.61 -> 12.20 |

读表方法：

- 同一 GBS 下比较 median step：越低越好。
- GBS=3200 的较好记录是 `87936`，约 `110.75s/step`、`118k tokens/s`。
- `79967` 和 `83949` 明显退化，优先回看对应启动参数、系统负载、NCCL 报警和 CUDA Graph capture 是否失败。
- loss、grad-norm、loss-scale 没有明显异常；这些记录更像性能/配置探索，而不是数值稳定性失败。

## 9. 正式测试报告模板

每个 case 记录下面字段：

| 字段 | 示例 |
|---|---|
| 机器 | 8x NVIDIA B300 SXM6 AC, driver 580.159.04 |
| 代码版本 | `git rev-parse HEAD` |
| 数据/tokenizer | `/data1/baibaifan/data/code-part124`, `/data1/baibaifan/data/compass_v4_tokenizer_200k` |
| GBS/MBS/seq | GBS 3200, MBS 4, seq 4096 |
| 优化开关 | CUTLASS grouped GEMM, HybridEP, CUDA Graph scope |
| warmup | 前 3 step |
| 统计区间 | step 4 到最后 |
| median step | 例如 110.75s |
| tokens/s | `GBS * 4096 / step_time` |
| 稳定性 | loss-scale、grad-norm、lm loss、是否 OOM/hang |
| 备注 | 是否共享机器、是否有 NCCL warning |

## 10. 注意事项

- `smc toc 10-251-210-1` 这类 hostname 要按 PAM 习惯转成 `10.251.210.1` 使用。
- B300 测试要优先保留 stdout 日志；当前脚本没有把 stdout 重定向到文件，后续建议恢复 `log_path` 逻辑或外层使用 `tee`。
- TensorBoard 只有 loss/lr/batch-size 时，step time 只能用 wall time 估算；如果要正式对外汇报，应同时采集 Megatron 打印的 iter time 和 throughput。
- GBS 不同不能只比较 step time，要比较 tokens/s；同 GBS 才直接比较 step time。
- CUDA Graph case 第一次 capture 往往较慢，至少丢弃前 3 step。
- 如果测试 DeepEP，把 `--moe-flex-dispatcher-backend hybridep` 改为 `deepep`，并确认 NVSHMEM 环境和 `libnvshmem` 路径已配置好。

