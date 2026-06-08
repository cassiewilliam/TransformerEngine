"""Perf of the QuACK matched pair at Case-7 shape (G=32, M=24576, I=512, D=2048).
gemm_gated (up+swiglu, store h) vs the recompute it replaces; gemm_dgated (down-dgrad+dswiglu) vs B2 143.6us.
"""
import os
from itertools import accumulate
os.environ["NVTE_USE_FUSED_MOE"] = "1"
import torch, time
import transformer_engine  # noqa
from quack.gemm_interface import gemm_gated, gemm_dgated

torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
G, I, D, ME = 32, 512, 2048, 768
TWO_I = 2 * I
split = [ME] * G
M = sum(split)
cu = torch.tensor([0] + list(accumulate(split)), dtype=torch.int32, device=DEV)
W1 = torch.randn(G, TWO_I, D, dtype=DT, device=DEV) / (D ** 0.5)
W2 = torch.randn(G, D, I, dtype=DT, device=DEV) / (I ** 0.5)
x = torch.randn(M, D, dtype=DT, device=DEV)
dY = torch.randn(M, D, dtype=DT, device=DEV)
s = (torch.rand(M, device=DEV, dtype=torch.float32) + 0.25)
B1 = W1.permute(0, 2, 1).contiguous()
A = torch.empty(M, I, dtype=DT, device=DEV); h = torch.empty(M, TWO_I, dtype=DT, device=DEV)
dh = torch.empty(M, TWO_I, dtype=DT, device=DEV); a_prime = torch.empty(M, I, dtype=DT, device=DEV)


def gated():
    gemm_gated(x, B1, activation="swiglu", cu_seqlens_m=cu, preact_out=h, postact_out=A,
               store_preact=True, concat_layout=("B",))

def dgated():
    gemm_dgated(dY, W2, PreAct=h, activation="swiglu", dx_out=dh, postact_out=a_prime,
                colvec_scale=s, colvec_reduce=True, cu_seqlens_m=cu, dynamic_scheduler=False)


def bench(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6  # us


print(f"gemm_gated  (up+swiglu, store h) : {bench(gated):.1f} us")
print(f"gemm_dgated (down-dgrad+dswiglu) : {bench(dgated):.1f} us")
print("PAIR_PERF_DONE")
