"""WGrad GEMM: QuACK gemm (ragged-K, SonicMoE form) vs CUTLASS GGT-NT. Real Case-7 ragged. dW1 (up)."""
import os, random
from itertools import accumulate
os.environ["NVTE_USE_FUSED_MOE"] = "1"
os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
import torch, time
import transformer_engine  # noqa
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor as GGT
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from quack.gemm_interface import gemm as qgemm

torch.manual_seed(0); random.seed(0)
DEV, DT = "cuda", torch.bfloat16
G, D, I, ME = 32, 2048, 512, 768
TWO_I = 2 * I; TILE = 256
nt = (G * ME) // TILE
tiles = [min(8, max(1, round(random.gauss(3, 2.2)))) for _ in range(G)]
df = nt - sum(tiles); j = 0
while df != 0:
    k = j % G
    if df > 0 and tiles[k] < 8: tiles[k] += 1; df -= 1
    elif df < 0 and tiles[k] > 1: tiles[k] -= 1; df += 1
    j += 1
    if j > 100 * G: break
split = [t * TILE for t in tiles]; M = sum(split)
split_t = torch.tensor(split, dtype=torch.int64, device=DEV)
base = tex.splits_to_offsets(split_t, 1)
cu = torch.tensor([0] + list(accumulate(split)), dtype=torch.int32, device=DEV)
off = [0] + list(accumulate(split))
x = torch.randn(M, D, dtype=DT, device=DEV) / 8
dh = torch.randn(M, TWO_I, dtype=DT, device=DEV) / 8
dW1_ref = torch.stack([dh[off[g]:off[g+1]].float().T @ x[off[g]:off[g+1]].float() for g in range(G)])  # [G,2I,D]

def aGT(t, dim): return GroupedTensor(shape=(M, dim), dtype=DT, num_tensors=G, quantizer=None, data=t.reshape(-1), first_dims=split_t, tensor_offsets=base*dim)
def wGT(t, o, i): return GroupedTensor(shape=(G*o, i), dtype=DT, num_tensors=G, shapes=[(o,i)]*G, quantizer=None, data=t.reshape(-1))
def rel(a, b): return ((a.float()-b.float()).norm()/(b.float().norm()+1e-9)).item()

# CUTLASS (current path): GGT NT, A=x B=dh -> dW1[G*2I, D]
dW1_cut = torch.empty(G*TWO_I, D, dtype=DT, device=DEV)
GGT(aGT(x, D), aGT(dh, TWO_I), wGT(dW1_cut, TWO_I, D), layout="NT")
print(f"CUTLASS dW1 rel: {rel(dW1_cut.view(G,TWO_I,D), dW1_ref):.3e}")

# QuACK (SonicMoE form): gemm(x.T, dh, cu_seqlens_k) -> out[G, D, 2I] = dW1^T per expert
xT = x.transpose(0, 1)  # [D, M] view (token dim = K, segmented by cu_seqlens_k)
oq = torch.empty(G, D, TWO_I, dtype=DT, device=DEV)
try:
    qgemm(xT, dh, out=oq, cu_seqlens_k=cu, dynamic_scheduler=False)
    print(f"QuACK dW1 rel: {rel(oq.transpose(1,2), dW1_ref):.3e}")
    quack_ok = True
except Exception as e:
    print(f"QuACK FAIL: {repr(e)[:200]}"); quack_ok = False

def bench(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6

t_cut = bench(lambda: GGT(aGT(x, D), aGT(dh, TWO_I), wGT(dW1_cut, TWO_I, D), layout="NT"))
print(f"TIME dW1  CUTLASS GGT-NT : {t_cut:.1f} us")
if quack_ok:
    t_q = bench(lambda: qgemm(xT, dh, out=oq, cu_seqlens_k=cu, dynamic_scheduler=False))
    print(f"TIME dW1  QuACK gemm     : {t_q:.1f} us   (speedup {t_cut/t_q:.2f}x)")

# ---- dW2 (down-wgrad): dW2[D,I] = dY^T @ A_act ; QuACK gemm(A.T, dY, cu_k) -> [G,I,D]=dW2^T ----
A_act = torch.randn(M, I, dtype=DT, device=DEV) / 8
dY = torch.randn(M, D, dtype=DT, device=DEV) / 8
dW2_ref = torch.stack([dY[off[g]:off[g+1]].float().T @ A_act[off[g]:off[g+1]].float() for g in range(G)])  # [G,D,I]
dW2_cut = torch.empty(G*D, I, dtype=DT, device=DEV)
GGT(aGT(A_act, I), aGT(dY, D), wGT(dW2_cut, D, I), layout="NT")
print(f"CUTLASS dW2 rel: {rel(dW2_cut.view(G,D,I), dW2_ref):.3e}")
AT = A_act.transpose(0, 1)  # [I, M]
oq2 = torch.empty(G, I, D, dtype=DT, device=DEV)
try:
    qgemm(AT, dY, out=oq2, cu_seqlens_k=cu, dynamic_scheduler=False)
    print(f"QuACK dW2 rel: {rel(oq2.transpose(1,2), dW2_ref):.3e}")
    q2_ok = True
except Exception as e:
    print(f"QuACK dW2 FAIL: {repr(e)[:160]}"); q2_ok = False
t_cut2 = bench(lambda: GGT(aGT(A_act, I), aGT(dY, D), wGT(dW2_cut, D, I), layout="NT"))
print(f"TIME dW2  CUTLASS GGT-NT : {t_cut2:.1f} us")
if q2_ok:
    t_q2 = bench(lambda: qgemm(AT, dY, out=oq2, cu_seqlens_k=cu, dynamic_scheduler=False))
    print(f"TIME dW2  QuACK gemm     : {t_q2:.1f} us   (speedup {t_cut2/t_q2:.2f}x)")

# ============ Forward/dgrad type (cu_seqlens_m): FWD_DOWN + BWD_UP ============
W1w = torch.randn(G, TWO_I, D, dtype=DT, device=DEV) / (D ** 0.5)   # up weight [G,2I,D]
W2w = torch.randn(G, D, I, dtype=DT, device=DEV) / (I ** 0.5)       # down weight [G,D,I]
W2_ID = W2w.transpose(1, 2).contiguous()                            # [G,I,D] = W2^T per expert

# FWD_DOWN: Y = A @ W2^T  -> [M,D]
Y_ref = torch.cat([A_act[off[g]:off[g+1]].float() @ W2w[g].float().T for g in range(G)]).to(DT)
Y_cut = torch.empty(M, D, dtype=DT, device=DEV)
GGT(wGT(W2w, D, I), aGT(A_act, I), aGT(Y_cut, D), layout="TN")
Yq = torch.empty(M, D, dtype=DT, device=DEV)
qgemm(A_act, W2_ID, out=Yq, cu_seqlens_m=cu, dynamic_scheduler=False)
print(f"FWD_DOWN  rel cut={rel(Y_cut, Y_ref):.3e} quack={rel(Yq, Y_ref):.3e}")
tc = bench(lambda: GGT(wGT(W2w, D, I), aGT(A_act, I), aGT(Y_cut, D), layout="TN"))
tq = bench(lambda: qgemm(A_act, W2_ID, out=Yq, cu_seqlens_m=cu, dynamic_scheduler=False))
print(f"TIME FWD_DOWN CUTLASS {tc:.1f} / QuACK {tq:.1f} us  (speedup {tc/tq:.2f}x)")

# BWD_UP (dgrad): dX = dh @ W1  -> [M,D]
dX_ref = torch.cat([dh[off[g]:off[g+1]].float() @ W1w[g].float() for g in range(G)]).to(DT)
dX_cut = torch.empty(M, D, dtype=DT, device=DEV)
GGT(wGT(W1w, TWO_I, D), aGT(dh, TWO_I), aGT(dX_cut, D), layout="NN")
dXq = torch.empty(M, D, dtype=DT, device=DEV)
qgemm(dh, W1w, out=dXq, cu_seqlens_m=cu, dynamic_scheduler=False)
print(f"BWD_UP    rel cut={rel(dX_cut, dX_ref):.3e} quack={rel(dXq, dX_ref):.3e}")
tc = bench(lambda: GGT(wGT(W1w, TWO_I, D), aGT(dh, TWO_I), aGT(dX_cut, D), layout="NN"))
tq = bench(lambda: qgemm(dh, W1w, out=dXq, cu_seqlens_m=cu, dynamic_scheduler=False))
print(f"TIME BWD_UP   CUTLASS {tc:.1f} / QuACK {tq:.1f} us  (speedup {tc/tq:.2f}x)")
print("WGRAD_BENCH_DONE")
