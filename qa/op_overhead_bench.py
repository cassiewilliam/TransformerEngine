"""Quantify the forward_fused_moe.py op-wrapper overhead sources (the per-call host marshalling).

Standard test config (see qa/STANDARD_TEST_CONFIG.md): real 4K-MoE per-card shape
  G=32, H(d)=2048, I=512, 2I=1024, Me=768, M=24576, bf16.
Run in container sonic-moe-2605 (CUDA 13.2 / cuBLAS 13.4) on an IDLE GPU:
  CUDA_VISIBLE_DEVICES=<idle> python qa/op_overhead_bench.py
"""

import torch
import transformer_engine  # noqa: F401 - load libtransformer_engine RTLD_GLOBAL BEFORE the extension
import transformer_engine_torch as tex

G, D, I = 32, 2048, 512
TWO_I, ME = 2 * I, 768
M = G * ME
dev, dt = "cuda", torch.bfloat16
print(f"CONFIG G={G} H(d)={D} I={I} 2I={TWO_I} Me={ME} M={M} dtype={dt}  GPU={torch.cuda.get_device_name(0)}")

w1 = [torch.randn(TWO_I, D, dtype=dt, device=dev) for _ in range(G)]  # per-expert FC1 (as GroupedLinear holds)
w2 = [torch.randn(D, I, dtype=dt, device=dev) for _ in range(G)]      # per-expert FC2
splits = torch.tensor([ME] * G, dtype=torch.int64, device=dev)        # device splits


def stack_w1():
    return torch.stack(w1, 0).view(G * TWO_I, D).contiguous()          # forward_fused_moe.py:393-395


def stack_w2():
    return torch.stack(w2, 0).contiguous().reshape(-1)                 # forward_fused_moe.py:424


def sync_splits():
    return [int(s) for s in splits.tolist()]                          # forward_fused_moe.py:226 (CPU sync)


def offsets():
    return tex.splits_to_offsets(splits, 1)                            # forward_fused_moe.py:207


def bench(fn, n=100, w=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n  # ms


for name, fn in [
    ("FC1 weight stack+contig (134MB copy)", stack_w1),
    ("FC2 weight stack+contig", stack_w2),
    ("splits.tolist() [CPU<->GPU sync]", sync_splits),
    ("splits_to_offsets", offsets),
]:
    print(f"  {name:38s}: {bench(fn)*1000:7.2f} us/call")
