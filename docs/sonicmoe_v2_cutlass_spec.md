# SonicMoE gemm_gated / gemm_dgated → CUTLASS V2 实现 spec

目标：把 QuACK 的 cuTe-DSL `gemm_gated`(fwd) + `gemm_dgated`(bwd) **复现为 CUTLASS SM100 kernel**，
对齐当前 TE 集成所需 I/O，**interleave 操作用 MegaMoE 的 TMA 5D layout 解决**，达性能最优
（目标：fused MoE backward 路 < 1.133ms 的 separate 路）。

参考来源：
- QuACK: https://github.com/Dao-AILab/quack — `quack/gemm_act.py`(gemm_act), `quack/gemm_dact.py`(gemm_dact)
- 原版调用: `sonicmoe/functional/forward.py:82`(gemm_gated), `backward.py:233`(gemm_dgated)
- MegaMoE 5D-TMA: `/Users/min.yang/workcode/transformerengine/.../mega_moe/mega_moe_launch.cuh:67` make_tma_5d_glu
- 算法 docs: `docs/sonicmoe.html` §反向数据流 (3780-3920)
- 现有 CUTLASS base: `transformer_engine/common/gemm/cutlass_grouped_gemm_swiglu.cuh`(fwd Sm100SwiGlu),
  `cutlass_grouped_gemm_dswiglu.cuh`(bwd Sm100DSwiGlu, = SonicMoE 同源 kernel)

---

## 1. 当前状态 (baseline, G=32 H=2048 I=512 ragged, B200 graph)

| 路径 | kernel | 时间 | 瓶颈 |
|---|---|---|---|
| Default(cuBLASLt) | nvjet | 1.260ms | FC1/SwiGLU 不融合，elementwise 581us |
| **Cutlass Group GEMM**(separate bwd) | Sm100SwiGlu + GemmUniversal | **1.133ms 最快** | — |
| Fuse MoE(B2 bwd) | + Sm100DSwiGlu 349us | 1.355ms 最慢 | **dswiglu epilogue scattered-LSU-read h** |

forward Sm100SwiGlu = 233us（OK）；backward Sm100DSwiGlu = **349us（瓶颈）**。
dswiglu epilogue per-element 读 gate=`h[grow*2I+gcol]` + up=`+I`（concatenated, **scattered LSU**），
memory 记 store/LSU-op-bound；OPT1(hoist prob, commit 9dcc7653) 只到 1.343ms（微弱）。

---

## 2. QuACK 参考接口（要对齐的 I/O）

### gemm_gated = gemm_act（forward）
```
in:  A=x[T,H] (A_idx gather), B=w1[H,2I] (per-expert), bias, cu_seqlens_m(varlen-M)
out: preact_out = h[TK,2I]  (gate‖up，store_preact=True 时存，喂 bwd)
     postact_out = a[TK,I]  (= SwiGLU(h) = silu(gate)*up)
layout: concat_layout 控制 [gate;up] 是 concat 还是 interleaved；
        kernel 内 gate=preact[...,::2], up=preact[...,1::2]  → ELEMENT-interleaved
```

### gemm_dgated = gemm_dact（backward，一 kernel 三输出）
```
in:  A=dO[T,H] (A_idx gather), B=W2[H,I], PreAct=h[TK,2I], colvec_scale=s(prob)[TK]
mainloop: dA' = dO·W2ᵀ  → TMEM，不落 HBM
epilogue (colvec_scale 只作用 dx/postact；colvec_reduce 在 scale 前抓未 scale 的 A):
  dx_out      = dH = s·(dA' ⊙ J_SwiGLU(h))   [TK,2I] interleaved 写  (输出#1)
  postact_out = A' = s·SwiGLU(h)             [TK,I]  写 HBM 喂 dW2    (输出#2)
  colvec_reduce → dS = <dA', A>  (A 未 scale, 行归约)                 (输出#3=dprob)
J_SwiGLU: J_gate=σ(Hg)(1+Hg(1-σ(Hg)))·Hu, J_up=silu(Hg)
```
对齐我 TE 侧 backward_fused_moe.py：dx_out→dY1(FC1 bwd 用)，postact_out→A'(替 saved-A 喂 dW2 wgrad)，
colvec_reduce→grad_scales(dprob)。**注意 V2 多产 A'，可省 forward 的 A cache**。

---

## 3. interleave → TMA 5D layout（核心 perf 方案）

MegaMoE `make_tma_5d_glu(ptr, H, I, LE, BK, n_groups_box, sw)`（gran-8 block-interleave）：
```
gd[5] = {H, 8, 2, I/8, LE}            // 内→外: [H, w8=8, gu=2, group=I/8, expert]
gs[4] = {H·2B (w8:+1row), I·H·2B (gu:+I row 跳gate→up), 8·H·2B (group:+8row), 2I·H·2B (expert)}
sd[5] = {sin, 8, 2, n_groups_box, 1}  // box 一次 co-load gate+up (gran-8) 进 smem
swizzle 128B; CU_TENSOR_MAP_INTERLEAVE_NONE（interleave 靠 stride 编码，非硬件 interleave）
```
精髓：**gu=2 维用 stride=I·H 一步从 gate 跳 up，一次 TMA 同时 co-load gate+up**，HBM 权重保持 contiguous
（Muon-safe，无 host permute），kernel 在 smem 拿到 interleaved gate/up → 向量化读。
验证：`qa/test_tma_interleave.cu` bit-identical to host gran8 + 2D TMA。

**用到 dswiglu 的 h-read**：h[TK,2I] concatenated，给它建一个 3D-TMA desc（gd=[I, gu=2, TK]，
gu 维 stride=I 跳 gate→up），box 一次 co-load gate+up tile 进新 smem buffer，epilogue float2 向量化读，
**消灭 per-element scattered LSU**。

---

## 4. V2 kernel 改动点

### 4a. forward Sm100SwiGlu V2（产 interleaved h）
- W1 load 换 make_tma_5d_glu 风格（gran-8 gate/up co-load），epilogue 产 **interleaved h**（gate/up gran-8）
- 同时产 a=SwiGLU(h)（已有）
- gate: `cutlass_grouped_gemm_swiglu.cuh` 现用两个 N-tile slice（gate/up），改为 gran-8 interleave 的单 desc

### 4b. backward Sm100DSwiGlu V2（读 interleaved h，核心 perf）
文件 `cutlass_grouped_gemm_dswiglu.cuh`，epilogue（~970-1009 那段 per-element loop）：
1. 新增 smem buffer `smem_h`（co-load 的 gate+up tile）+ TMA desc over h（3D, gu=2 跳）
2. load warp 里发 TMA load h tile（overlap mainloop dA'=dO·W2ᵀ）
3. epilogue 从 smem 向量化读 gate/up（替 `params.dGrad[hbase]`/`+N` 的 scattered LSU）
4. 加 postact_out A'=s·SwiGLU(h) 输出（省 saved-A）
5. dprob 对齐 MegaMoE 的 smem 归约版（`dprob_smem[kEpiTileM]`）
- smem 预算：SM100 228KB，现有 smem_x+smem_w2t+smem_out_gate/up+pipelines；加 smem_h(~32KB) 需验证 ≤228KB
- gate（已 OPT1 hoist 的 per-row 常量保留）

---

## 5. 风险 & 流程
- **M3「TMA h-prefetch」memory 记过 NEGATIVE（LSU-op-bound）**——但那是朴素 prefetch（仍 LSU read smem）；
  V2 的 co-load + swizzle + **float2 向量化读**（gate/up 相邻）才是关键差异，有翻盘机会，需实测
- rebuild：`/data1/min.yang/te_build/build/cmake` → `rm CMakeFiles/.../cutlass_grouped_gemm_dswiglu.cu.o && ninja libtransformer_engine.so && cp 到 te_build/`（~10min/轮）
- 验证：每轮 e2e DROP-IN（`te_fused_moe_e2e_test.py --correctness`，5 梯度 n_fail=0）+ moe_cg_real perf（GPU4 warmup30 avg100）
- 目标判据：Fuse MoE(V2) < 1.133ms（打过 separate 路）才算赢
- **注**：此 spec 描述的 V2 retune 路线后被实测证伪（5D-TMA / V2a 两条路皆慢于 V1），
  本仓已删 V2 scaffold 与 `NVTE_DSWIGLU_V2` / `NVTE_FUSE_MOE_DSWIGLU` 等历史 A/B gate。
  F 走 QuACK `gemm_gated` (fwd) + CUTLASS B2 `te_cutlass_grouped_dswiglu` (bwd) 单一路径。

---

## 6. 已 bank
- OPT1（commit 9dcc7653）：dswiglu epilogue hoist per-row 常量，DROP-IN PASS，1.343ms < 原 Fuse MoE 1.355（hook 已达成，但非最优）
- ~~env gate `NVTE_FUSE_MOE_DSWIGLU`（commit 4b7065a8）：B2 on/off A/B~~  *已废弃：调试 knob，commit 705f28aa 后 F 永远走 B2，flag 移除。*
