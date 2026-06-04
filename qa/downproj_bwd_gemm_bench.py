"""Isolated perf of the BW DownProj GEMM dA = dY @ W2 (FC2 dgrad), graph-safe cuBLAS vs CUTLASS F0.

This is the GEMM the B2 fused kernel is built on. The user's gate: the CUTLASS grouped GEMM must beat
graph-safe BEFORE fusing the dAct on it. Shape = the dA GEMM: dY[M,d] @ W2[d,I] -> dA[M,I], grouped over
G experts (per-expert dY[Me,d] @ W2[e][d,I]). Equivalent to a GroupedLinear(in=d, out=I) forward GEMM:
out = A @ W^T with W[I,d], i.e. [M,d] @ [I,d]^T -> [M,I], K=d, N=I. Real 4K-MoE: G=32, d=2048, I=512,
Me=768, M=24576, bf16. FLOP = 2*M*I*d = 51.5 GFLOP.

  CUDA_VISIBLE_DEVICES=<idle> MODE=graphsafe python qa/downproj_bwd_gemm_bench.py
  CUDA_VISIBLE_DEVICES=<idle> MODE=cutlass   python qa/downproj_bwd_gemm_bench.py
"""

import os

MODE = os.environ.get("MODE", "cutlass")  # graphsafe | cutlass
if MODE == "cutlass":
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"

import torch
import transformer_engine  # noqa: F401  (RTLD_GLOBAL first)
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops.basic.grouped_linear import GroupedLinear as _OpGL

# Force CUTLASS/graphsafe: the ops GroupedLinear picks graph-safe on SM100+bf16 unless monkeypatched.
if MODE == "cutlass":
    _OpGL._is_graph_safe_path_supported = staticmethod(lambda **kw: False)

torch.manual_seed(0)
G, D, I = 32, 2048, 512
DEV, DT = "cuda", torch.bfloat16

# REAL load-imbalanced MoE per-expert token counts: M(min,avg,max) = (128, 768, 2048), ΣM = G*768 = 24576.
# Representative distribution: 6 hot experts @2048, 6 cold @128, 20 mid @576 (Σ = 24576, avg 768).
SHAPE = os.environ.get("SHAPE", "imbal")  # imbal | uniform
if SHAPE == "uniform":
    splits = [768] * G
else:
    splits = [2048] * 6 + [576] * 20 + [128] * 6
assert len(splits) == G and sum(splits) == G * 768, (len(splits), sum(splits))
M = sum(splits)
gemm_flop = 2.0 * M * I * D  # 51.5 GFLOP (the dA = dY @ W2 GEMM)

# GroupedLinear(in=d, out=I): forward out = dY @ W2^T-equivalent (K=d, N=I) -> the dA GEMM shape.
gl = te_ops.GroupedLinear(G, D, I, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
model = te_ops.Sequential(gl)
dY = torch.randn(M, D, dtype=DT, device=DEV)
split_t = torch.tensor(splits, dtype=torch.int64, device=DEV)
print(f"SHAPE={SHAPE} M={M} G={G} per-expert(min/avg/max)={min(splits)}/{M//G}/{max(splits)}")


def fwd():
    with torch.no_grad():
        return model(dY, split_t)


def bench(fn, n=80, w=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


ms = bench(fwd)
print(f"DOWNPROJ_BWD_GEMM,{MODE},{ms*1000:.2f}us,{gemm_flop/(ms/1e3)/1e12:.1f}TF/s")
print(f"[{MODE:9s}] dA=dY@W2  M={M} K(d)={D} N(I)={I} G={G}  {ms*1000:7.2f} us   {gemm_flop/(ms/1e3)/1e12:6.1f} TF/s")
