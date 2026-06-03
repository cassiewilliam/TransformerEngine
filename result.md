# K-grouped BF16 Wgrad (varlen-k) — 正确性 & 性能测试结果

**日期**: 2026-06-01
**特性**: `cutlass_grouped_gemm_varlen_k` —— K 维分组(ragged-K)BF16-in / (FP32|BF16)-out NT wgrad 专用路径
**结论**: ✅ 正确性全部通过;CUTLASS 路径相对 cuBLAS 基线 **1.0–1.34×** 加速(wgrad 1.27×)。

---

## 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SM90) |
| 镜像 base | `nvcr.io/nvidia/pytorch:26.02-py3`(官方一致) |
| torch | 2.11.0a0+nv26.02 / CUDA 13.1 |
| TransformerEngine | 2.17.0.dev0(本分支源码全量重编) |
| 容器 | `miny_mhc_bench_v2` |
| 开关 | `NVTE_USE_CUTLASS_GROUPED_GEMM=1` |

**MoE 形状**: 160 experts, EP=8(每 rank 20 个本地 group), seq_len=8192, topk=8, hidden(K)=2048, expert/ffn(N)=512, ≈66560 routed tokens/rank。

---

## 1. 正确性

| 测试 | 范围 | 结果 |
|---|---|---|
| `test_grouped_linear.py::test_grouped_linear_accuracy` (`-k "bfloat16 and None"`) | 纯 BF16 GroupedLinear,fwd + dgrad + **wgrad** 对比逐 Linear 参考实现 | **64 / 64 passed** |
| grouped GEMM ragged-K `--verify` | 分组 GEMM 输出对比逐 GEMM `general_gemm` 参考(ragged splits) | **verification: passed** |
| varlen-k 路径命中确认 | 插桩 marker 验证 GroupedLinear 的 wgrad 实际路由到 `cutlass_grouped_gemm_varlen_k` | ✅ 命中且数值正确 |

> 说明:GroupedLinear 反向的 wgrad GEMM(NT 布局、ragged K)在 `NVTE_USE_CUTLASS_GROUPED_GEMM=1` 下走 `cutlass_grouped_gemm_varlen_k`,结果与参考实现一致。

---

## 2. 性能(H100, BF16, 100 iters / 20 warmup)

### 2.1 Uniform K（每 group 3328 = 26×128）—— CUTLASS vs cuBLAS

| case | cuBLAS avg_ms | cuBLAS TFLOPS | CUTLASS avg_ms | CUTLASS TFLOPS | 加速 |
|---|---|---|---|---|---|
| fwd | 0.3175 | 439.59 | 0.2372 | **588.37** | **1.34×** |
| dgrad | 0.2679 | 521.06 | 0.2603 | 536.21 | 1.03× |
| **wgrad** | 0.3133 | 445.50 | 0.2459 | **567.65** | **1.27×** |
| bwd (dgrad+wgrad) | 0.5957 | 468.63 | 0.5147 | 542.44 | 1.16× |
| fwd+bwd | 0.9215 | 454.42 | 0.7909 | 529.49 | 1.17× |

### 2.2 Ragged / Jagged K（varlen-k 本职场景，已 `--verify` 通过）

`m_splits = [512, 1024, 1536, 2048, 2560, 3072, 3328, 3584, 4096, 4096, 4608, 5120, 1280, 2304, 3456, 4480, 1152, 2176, 5632, 6016]`

| case | CUTLASS avg_ms | CUTLASS TFLOPS |
|---|---|---|
| fwd | 0.2226 | 584.77 |
| dgrad | 0.2439 | 533.88 |
| **wgrad** | 0.2499 | **520.92** |
| bwd (dgrad+wgrad) | 0.4976 | 523.29 |
| fwd+bwd | 0.7641 | 511.15 |

---

## 备注

- 运行间抖动约 ±5%,对 <10% 的差异不必过度解读。
- 全量 `test_grouped_linear.py`(768+ 用例)另有 20 个失败,均为 `test_grouped_gemm` 中 `use_cutlass=False`(cuBLAS 路径)+ `accumulate=True` 的 **`rtol=0,atol=0` 逐位相等** 断言,**与本特性(cutlass 路径)无关**;所有 `use_cutlass=True` 用例全部通过。
- 镜像基于官方 `nvcr.io/nvidia/pytorch:26.02-py3`,仅叠加本分支 TE 源码全量重编,torch/CUDA/cuDNN 层与官方一致。
