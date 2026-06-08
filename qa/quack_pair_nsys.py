"""nsys-confirm the QuACK matched-pair optimization: gemm_gated / gemm_dgated under aligned NVTX,
to compare against the current fused (Sm100SwiGlu 133.5us + h-recompute 101.9us / B2 143.6us)."""
import os
from itertools import accumulate
os.environ["NVTE_USE_FUSED_MOE"] = "1"
import torch
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
rp, pp = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop


def gated():
    gemm_gated(x, B1, activation="swiglu", cu_seqlens_m=cu, preact_out=h, postact_out=A,
               store_preact=True, concat_layout=("B",))

def dgated():
    gemm_dgated(dY, W2, PreAct=h, activation="swiglu", dx_out=dh, postact_out=a_prime,
                colvec_scale=s, colvec_reduce=True, cu_seqlens_m=cu, dynamic_scheduler=False)


for _ in range(10):
    gated(); dgated()
torch.cuda.synchronize()
for _ in range(int(os.environ.get("NREP", "30"))):
    rp("NVTX_FWD_UP_SWIGLU_GEMM"); gated(); pp()
    rp("NVTX_BWD_DOWN_DSWILU_GEMM"); dgated(); pp()
torch.cuda.synchronize()
print("QUACK_PAIR_NSYS_DONE")
