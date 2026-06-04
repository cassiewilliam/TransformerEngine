"""Sweep the B1 dswiglu kernel's tile config (env NVTE_DSWIGLU_TILE_CFG) on an IDLE GPU.

The .cu reads NVTE_DSWIGLU_TILE_CFG via getenv PER CALL, so one process can sweep all configs by
setting os.environ before each timing loop. For each cfg: (1) correctness vs cfg0 (all cfgs compute the
SAME math -> dY1 must match cfg0 to ~bf16 eps; a large diff flags a buggy tile config), (2) TF/s on the
recompute up-proj (M*2I*d = 103.1 GFLOP). Real 4K-MoE shape G=32/H=2048/I=512/Me=768/M=24576, bf16.

  CUDA_VISIBLE_DEVICES=<idle> python qa/b1_cfg_sweep.py
"""

import os
import torch
import transformer_engine  # noqa: F401  (RTLD_GLOBAL first)
import transformer_engine_torch as tex

G, D, I = 32, 2048, 512
TWO_I, ME = 2 * I, 768
M = G * ME
dev, dt = "cuda", torch.bfloat16
up_flop = 2.0 * M * TWO_I * D  # 103.1 GFLOP

torch.manual_seed(0)
x = torch.randn(M, D, dtype=dt, device=dev)
w1 = (torch.randn(TWO_I * G, D, dtype=dt, device=dev) * 0.02)
dA = torch.randn(M, I, dtype=dt, device=dev)
prob = (torch.rand(M, device=dev, dtype=torch.float32) + 0.25)
mte = torch.repeat_interleave(
    torch.arange(G, device=dev, dtype=torch.int32),
    torch.div(torch.tensor([ME] * G, device=dev), 256, rounding_mode="floor"),
    output_size=(M + 255) // 256,
)
print(f"CONFIG G={G} H={D} I={I} 2I={TWO_I} Me={ME} M={M} up_flop={up_flop/1e9:.1f}GF  {torch.cuda.get_device_name(0)}")

CFGS = {
    0: "TileN64 kStages16 Acc2 (fwd-tuned default)",
    1: "TileN64 kStages16 Acc3",
    2: "TileN64 kStages16 Acc4",
    3: "TileN128 kStages8 Acc2",
    4: "TileN128 kStages10 Acc2",
}


def call():
    return tex.te_cutlass_grouped_dswiglu(x, w1, dA, mte, prob, G, 0, I, D, M, 0)


def bench(n=60, w=15):
    for _ in range(w):
        call()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        call()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n  # ms


ref = None
print(f"{'cfg':>3}  {'desc':40s}  {'us':>8}  {'TF/s':>7}  {'max|Δ vs cfg0|':>14}")
for c, desc in CFGS.items():
    os.environ["NVTE_DSWIGLU_TILE_CFG"] = str(c)
    try:
        out = call().float()
        torch.cuda.synchronize()
    except Exception as e:  # a bad cfg (e.g. smem/TMEM over budget) may error at launch
        print(f"{c:>3}  {desc:40s}  {'LAUNCH_ERROR: '+str(e)[:50]}")
        continue
    if ref is None:
        ref = out
        d = 0.0
    else:
        d = (out - ref).abs().max().item()
    ms = bench()
    flag = "" if d < 0.05 else "  <-- DIVERGES (buggy cfg)"
    print(f"{c:>3}  {desc:40s}  {ms*1000:8.1f}  {up_flop/(ms/1e3)/1e12:7.1f}  {d:14.5f}{flag}")
print("SWEEP_DONE")
