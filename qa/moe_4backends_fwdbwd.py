"""MoE FFN (GroupedLinear up-proj -> ScaledSwiGLU -> GroupedLinear down-proj) FWD + FWD+BWD timing,
across 4 grouped-GEMM backends, on the te.pytorch.ops framework (where the fused op lives).

MODE (env) selects the backend (run one per process; flags/fusion are import-time):
  legacy    : legacy multi-stream cuBLAS  (monkeypatch graph-safe->False, no CUTLASS flag)
  graphsafe : graph-safe cuBLAS 13.4       (ops default on SM100+bf16)
  cutlass   : CUTLASS F0 grouped GEMM      (monkeypatch graph-safe->False, NVTE_USE_CUTLASS_GROUPED_GEMM=1)
  fused     : SonicMoE fused up+SwiGLU      (NVTE_USE_FUSED_MOE=1; down-proj keeps graph-safe cuBLAS)

Real 4K-MoE per-card shape: G=32, H=2048, I=512 (2I=1024), Me=768, M=24576, bf16.
Run one: CUDA_VISIBLE_DEVICES=<idle> MODE=fused python qa/moe_4backends_fwdbwd.py
"""

import os

MODE = os.environ.get("MODE", "fused")
os.environ["NVTE_USE_FUSED_MOE"] = "1" if MODE == "fused" else "0"
if MODE == "cutlass":
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"

import torch
import transformer_engine  # noqa: F401  (load libtransformer_engine RTLD_GLOBAL first)
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops.basic.grouped_linear import GroupedLinear as _OpGL

# Force the legacy/CUTLASS path: the ops GroupedLinear otherwise always picks graph-safe on SM100+bf16.
if MODE in ("legacy", "cutlass"):
    _OpGL._is_graph_safe_path_supported = staticmethod(lambda **kw: False)

torch.manual_seed(0)
G, D, I = 32, 2048, 512
TWO_I, ME = 2 * I, 768
M = G * ME
DEV, DT = "cuda", torch.bfloat16
fwd_flop = 2 * M * TWO_I * D + 2 * M * D * I  # up + down = 154.6 GFLOP

fc1 = te_ops.GroupedLinear(G, D, TWO_I, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
act = te_ops.ScaledSwiGLU(glu_interleave_size=32)  # mandated by validate_grouped_mlp_dims
fc2 = te_ops.GroupedLinear(G, I, D, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
model = te_ops.Sequential(fc1, act, fc2)

x = torch.randn(M, D, dtype=DT, device=DEV, requires_grad=True)
prob = (torch.rand(M, device=DEV, dtype=torch.float32) + 0.25).requires_grad_(True)
split_t = torch.tensor([ME] * G, dtype=torch.int64, device=DEV)
dy = torch.randn(M, D, dtype=DT, device=DEV)


def fwd():
    with torch.no_grad():
        return model(x, split_t, prob, split_t)  # 3 extra inputs: split, prob, split (no bias)


def fwd_bwd():
    for p in model.parameters():
        p.grad = None
    x.grad = None
    prob.grad = None
    y = model(x, split_t, prob, split_t)
    y.backward(dy)


def bench(fn, n=50, w=15):
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


fwd_ms = bench(fwd)
fb_ms = bench(fwd_bwd)
bwd_ms = fb_ms - fwd_ms
print(f"PERF4,{MODE},{fwd_ms:.4f},{fb_ms:.4f},{bwd_ms:.4f}")
print(
    f"[{MODE:9s}] FWD {fwd_ms:.4f}ms ({fwd_flop/(fwd_ms/1e3)/1e12:6.1f} TF/s) | "
    f"FWD+BWD {fb_ms:.4f}ms | BWD {bwd_ms:.4f}ms"
)
