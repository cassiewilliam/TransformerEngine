"""6-phase MoE expert fwd+bwd with ALIGNED NVTX ranges (one stable iteration), per backend.

Explicit grouped-tensor GEMM orchestration so each logical operator gets ONE aligned NVTX range:
  1 NVTX_FWD_UP_SWIGLU_GEMM   : h = X@W1^T ; A = swiglu(h)
  2 NVTX_FWD_DOWN_GEMM        : Y = A@W2^T
  3 NVTX_BWD_DOWN_DSWILU_GEMM : dA = dY@W2 ; dh = dswiglu(dA,h)
  4 NVTX_BWD_UP_GEMM          : dX = dh@W1
  5 NVTX_DWGRAD_DOWN_GEMM     : dW2 = A^T@dY
  6 NVTX_DWGRAD_UP_GEMM       : dW1 = X^T@dh
Backend via env MODE in {gt_cublas, cutlass_gt} (grouped-tensor path; CUTLASS via env).
Validated vs torch autograd. Run: MODE=cutlass_gt NVTX=1 CUDA_VISIBLE_DEVICES=<idle> python moe_nvtx_ops.py
"""
import os, sys, random
MODE = os.environ.get("MODE", "cutlass_gt")
os.environ["NVTE_USE_FUSED_MOE"] = "1" if MODE == "fused" else "0"
if MODE in ("cutlass_list", "cutlass_gt"):
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"

import torch
import torch.nn.functional as F
import transformer_engine  # noqa: F401  (RTLD_GLOBAL libtransformer_engine.so BEFORE _torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor as GGT
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

NVTX = os.environ.get("NVTX", "") == "1"
def rp(n):
    if NVTX: torch.cuda.nvtx.range_push(n)
def pp():
    if NVTX: torch.cuda.nvtx.range_pop()

torch.manual_seed(0); random.seed(0)
DEV, DT = "cuda", torch.bfloat16
G = int(os.environ.get("MOE_G", 32)); D = int(os.environ.get("MOE_D", 2048))
I = int(os.environ.get("MOE_I", 512)); ME = int(os.environ.get("MOE_ME", 768))
TWO_I = 2 * I; TILE = 256
nt = (G * ME) // TILE
tiles = [min(8, max(1, round(random.gauss(3, 2.2)))) for _ in range(G)]
d_ = nt - sum(tiles); j = 0
while d_ != 0:
    k = j % G
    if d_ > 0 and tiles[k] < 8: tiles[k] += 1; d_ -= 1
    elif d_ < 0 and tiles[k] > 1: tiles[k] -= 1; d_ += 1
    j += 1
    if j > 100 * G: break
split = [t * TILE for t in tiles]
M = sum(split)
split_t = torch.tensor(split, dtype=torch.int64, device=DEV)
base = tex.splits_to_offsets(split_t, 1)
TILE_M = 256
_num_tiles = (M + TILE_M - 1) // TILE_M
m_tile = torch.repeat_interleave(
    torch.arange(G, device=DEV, dtype=torch.int32),
    torch.div(split_t, TILE_M, rounding_mode="floor"),
    output_size=_num_tiles,
)
FUSED = MODE == "fused"

# weights packed [G*out, in]; activations [M, dim]
W1 = (torch.randn(G * TWO_I, D, dtype=DT, device=DEV) / (D ** 0.5))
W2 = (torch.randn(G * D, I, dtype=DT, device=DEV) / (I ** 0.5))
x = torch.randn(M, D, dtype=DT, device=DEV)
dY = torch.randn(M, D, dtype=DT, device=DEV)


def wGT(data, out_f, in_f):
    return GroupedTensor(shape=(G * out_f, in_f), dtype=DT, num_tensors=G,
                         shapes=[(out_f, in_f)] * G, quantizer=None, data=data.reshape(-1))


def aGT(data, dim):
    return GroupedTensor(shape=(M, dim), dtype=DT, num_tensors=G, quantizer=None,
                         data=data.reshape(-1), first_dims=split_t, tensor_offsets=base * dim)


def swiglu(h):
    return (F.silu(h[:, :I].float()).to(DT) * h[:, I:])


def dswiglu(dA, h):
    g_ = h[:, :I].float(); u_ = h[:, I:].float()
    sg = torch.sigmoid(g_); sig = g_ * sg
    return torch.cat([dA.float() * u_ * (sg + sig * (1 - sg)), dA.float() * sig], 1).to(DT)


def run_once():
    h = torch.empty(M, TWO_I, dtype=DT, device=DEV)
    dX = torch.empty(M, D, dtype=DT, device=DEV)
    dW2 = torch.empty(G * D, I, dtype=DT, device=DEV)
    dW1 = torch.empty(G * TWO_I, D, dtype=DT, device=DEV)

    rp("NVTX_FWD_UP_SWIGLU_GEMM")
    if FUSED:
        A = tex.te_cutlass_grouped_swiglu(x, W1, m_tile, None, G, 0, I, D, M, 0).view(M, I)
        GGT(wGT(W1, TWO_I, D), aGT(x, D), aGT(h, TWO_I), layout="TN")   # recompute h (plain) for B2
    else:
        GGT(wGT(W1, TWO_I, D), aGT(x, D), aGT(h, TWO_I), layout="TN")   # h = x@W1^T
        A = swiglu(h)
    pp()
    rp("NVTX_FWD_DOWN_GEMM")
    if FUSED:
        Y = tex.te_cutlass_grouped_down(A, W2, m_tile, G, D, I, M).view(M, D)
    else:
        Y = torch.empty(M, D, dtype=DT, device=DEV)
        GGT(wGT(W2, D, I), aGT(A, I), aGT(Y, D), layout="TN")           # Y = A@W2^T
    pp()
    rp("NVTX_BWD_DOWN_DSWILU_GEMM")
    if FUSED:
        dh = tex.te_cutlass_grouped_dswiglu(dY.contiguous(), W2, h, m_tile, None,
                                            G, 0, I, D, M, 0, None).view(M, TWO_I)  # dA=dY@W2 + dswiglu
    else:
        dA = torch.empty(M, I, dtype=DT, device=DEV)
        GGT(wGT(W2, D, I), aGT(dY, D), aGT(dA, I), layout="NN")         # dA = dY@W2
        dh = dswiglu(dA, h)
    pp()
    rp("NVTX_BWD_UP_GEMM")
    GGT(wGT(W1, TWO_I, D), aGT(dh, TWO_I), aGT(dX, D), layout="NN")     # dX = dh@W1
    pp()
    rp("NVTX_DWGRAD_DOWN_GEMM")
    GGT(aGT(A, I), aGT(dY, D), wGT(dW2, D, I), layout="NT")             # dW2 = A^T@dY
    pp()
    rp("NVTX_DWGRAD_UP_GEMM")
    GGT(aGT(x, D), aGT(dh, TWO_I), wGT(dW1, TWO_I, D), layout="NT")     # dW1 = x^T@dh
    pp()
    return h, A, Y, None, dX, dW2, dW1


if os.environ.get("CHECK", "0") == "1":
    h, A, Y, dA, dX, dW2, dW1 = run_once()
    offs = base.tolist()
    def rel(a, b): return ((a.float() - b).norm() / (b.norm() + 1e-9)).item()
    rY = rX = r1 = r2 = 0.0
    for g in range(G):
        s, e = offs[g], offs[g + 1]
        xg = x[s:e].float().requires_grad_(True)
        w1 = W1[g * TWO_I:(g + 1) * TWO_I].float().requires_grad_(True)
        w2 = W2[g * D:(g + 1) * D].float().requires_grad_(True)
        yg = (F.silu((xg @ w1.T)[:, :I]) * (xg @ w1.T)[:, I:]) @ w2.T
        yg.backward(dY[s:e].float())
        rY = max(rY, rel(Y[s:e], yg.detach()))
        rX = max(rX, rel(dX[s:e], xg.grad))
        r1 = max(r1, rel(dW1[g * TWO_I:(g + 1) * TWO_I], w1.grad))
        r2 = max(r2, rel(dW2[g * D:(g + 1) * D], w2.grad))
    print(f"CHECK MODE={MODE} relY={rY:.3e} reldX={rX:.3e} reldW1={r1:.3e} reldW2={r2:.3e}")
    print("CHECK_DONE")
    sys.exit(0)

for _ in range(8):
    run_once()
torch.cuda.synchronize()
import time
N = int(os.environ.get("ITERS", "30"))
t0 = time.perf_counter()
for _ in range(N):
    run_once()
torch.cuda.synchronize()
print(f"OPS6,{MODE},{(time.perf_counter() - t0) / N * 1000:.4f},M={M}")
print("MOE_NVTX_OPS_DONE")
