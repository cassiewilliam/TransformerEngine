"""cudagraph fwd+bwd total-time: current fused (Sm100SwiGlu + recompute-h + B2) vs
matched-pair (QuACK gemm_gated store-h + gemm_dgated). Explicit 6-phase, same shapes, 10-round avg."""
import os, random
from itertools import accumulate
os.environ["NVTE_USE_FUSED_MOE"] = "1"
import torch
import transformer_engine  # noqa
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor as GGT
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from quack.gemm_interface import gemm_gated, gemm_dgated

torch.manual_seed(0); random.seed(0)
DEV, DT = "cuda", torch.bfloat16
G, D, I, ME = 32, 2048, 512, 768
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
cu = torch.tensor([0] + list(accumulate(split)), dtype=torch.int32, device=DEV)
num_tiles = (M + TILE - 1) // TILE
m_tile = torch.repeat_interleave(torch.arange(G, device=DEV, dtype=torch.int32),
                                 torch.div(split_t, TILE, rounding_mode="floor"), output_size=num_tiles)
W1 = torch.randn(G * TWO_I, D, dtype=DT, device=DEV) / (D ** 0.5)
W2 = torch.randn(G * D, I, dtype=DT, device=DEV) / (I ** 0.5)
W1_B = W1.view(G, TWO_I, D).permute(0, 2, 1).contiguous()
W2_3 = W2.view(G, D, I).contiguous()
x = torch.randn(M, D, dtype=DT, device=DEV)
dY = torch.randn(M, D, dtype=DT, device=DEV)
s = (torch.rand(M, device=DEV, dtype=torch.float32) + 0.25)
# persistent buffers
h = torch.empty(M, TWO_I, dtype=DT, device=DEV); A = torch.empty(M, I, dtype=DT, device=DEV)
dX = torch.empty(M, D, dtype=DT, device=DEV); dW2 = torch.empty(G * D, I, dtype=DT, device=DEV)
dW1 = torch.empty(G * TWO_I, D, dtype=DT, device=DEV); ap = torch.empty(M, I, dtype=DT, device=DEV)
dh = torch.empty(M, TWO_I, dtype=DT, device=DEV)

def wGT(data, o, i): return GroupedTensor(shape=(G*o, i), dtype=DT, num_tensors=G, shapes=[(o, i)]*G, quantizer=None, data=data.reshape(-1))
def aGT(data, dim): return GroupedTensor(shape=(M, dim), dtype=DT, num_tensors=G, quantizer=None, data=data.reshape(-1), first_dims=split_t, tensor_offsets=base*dim)
import torch.nn.functional as F
def swiglu(t): return (F.silu(t[:, :I].float()).to(DT) * t[:, I:])
def dswiglu(dA, t):
    g_=t[:,:I].float(); u_=t[:,I:].float(); sg=torch.sigmoid(g_); sig=g_*sg
    return torch.cat([dA.float()*u_*(sg+sig*(1-sg)), dA.float()*sig],1).to(DT)


def fused_step():
    Aout = tex.te_cutlass_grouped_swiglu(x, W1, m_tile, None, G, 0, I, D, M, 0).view(M, I)   # ① up+swiglu
    GGT(wGT(W1, TWO_I, D), aGT(x, D), aGT(h, TWO_I), layout="TN")                              #   h recompute
    Y = tex.te_cutlass_grouped_down(Aout, W2, m_tile, G, D, I, M).view(M, D)                   # ② down
    dh2 = tex.te_cutlass_grouped_dswiglu(dY.contiguous(), W2, h, m_tile, None, G, 0, I, D, M, 0, None).view(M, TWO_I)  # ③ B2
    GGT(wGT(W1, TWO_I, D), aGT(dh2, TWO_I), aGT(dX, D), layout="NN")                           # ④ up-dgrad
    GGT(aGT(Aout, I), aGT(dY, D), wGT(dW2, D, I), layout="NT")                                 # ⑤ down-wgrad
    GGT(aGT(x, D), aGT(dh2, TWO_I), wGT(dW1, TWO_I, D), layout="NT")                           # ⑥ up-wgrad


def quack_step():
    gemm_gated(x, W1_B, activation="swiglu", cu_seqlens_m=cu, postact_out=A, preact_out=h,
               store_preact=True, concat_layout=("B",))                                        # ① up+swiglu+store h
    Y = tex.te_cutlass_grouped_down(A, W2, m_tile, G, D, I, M).view(M, D)                       # ② down
    gemm_dgated(dY, W2_3, PreAct=h, activation="swiglu", dx_out=dh, postact_out=ap,
                colvec_scale=s, colvec_reduce=True, cu_seqlens_m=cu, dynamic_scheduler=False)   # ③ dgrad+dswiglu
    GGT(wGT(W1, TWO_I, D), aGT(dh, TWO_I), aGT(dX, D), layout="NN")                            # ④ up-dgrad
    GGT(aGT(A, I), aGT(dY, D), wGT(dW2, D, I), layout="NT")                                    # ⑤ down-wgrad
    GGT(aGT(x, D), aGT(dh, TWO_I), wGT(dW1, TWO_I, D), layout="NT")                            # ⑥ up-wgrad


def cg_time(step, label):
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(8): step()
    torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(10): g.replay()
    b.record(); torch.cuda.synchronize()
    ms = a.elapsed_time(b) / 10
    print(f"CGTIME {label}: {ms:.4f} ms")
    return ms


m_f = cg_time(fused_step, "fused (Sm100SwiGlu+recompute+B2)")
m_q = cg_time(quack_step, "quack-pair (gemm_gated+gemm_dgated)")
print(f"SPEEDUP fused/quack = {m_f/m_q:.3f}x   saved={(m_f-m_q)*1000:.1f}us")
print("CG_QUACK_VS_FUSED_DONE")
