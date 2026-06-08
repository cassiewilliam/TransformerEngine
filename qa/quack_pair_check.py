"""Validate the QuACK gemm_gated(store_preact) -> gemm_dgated matched pair (items 1+2).
gemm_gated stores h; gemm_dgated reads h and recomputes A and a_prime=s*swiglu(h).
If A and a_prime match torch, the h emit+read round-trip is correct (no recompute, no layout mismatch).
"""
import os, sys
from itertools import accumulate
os.environ["NVTE_USE_FUSED_MOE"] = "1"
import torch
import torch.nn.functional as F
import transformer_engine  # noqa: F401
from quack.gemm_interface import gemm_gated, gemm_dgated

torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
G, I, D = 4, 512, 2048
ME = 768; TWO_I = 2 * I
split = [ME] * G
M = sum(split)
cu = torch.tensor([0] + list(accumulate(split)), dtype=torch.int32, device=DEV)

W1 = torch.randn(G, TWO_I, D, dtype=DT, device=DEV) / (D ** 0.5)   # gate||up
W2 = torch.randn(G, D, I, dtype=DT, device=DEV) / (I ** 0.5)
x = torch.randn(M, D, dtype=DT, device=DEV)
dY = torch.randn(M, D, dtype=DT, device=DEV)
s = (torch.rand(M, device=DEV, dtype=torch.float32) + 0.25)

# ---- forward: gemm_gated stores preact h (concat_layout=("B",) as in forward_fused_moe.py) ----
B1 = W1.permute(0, 2, 1).contiguous()      # [G, D, 2I]
A = torch.empty(M, I, dtype=DT, device=DEV)
h = torch.empty(M, TWO_I, dtype=DT, device=DEV)
gemm_gated(x, B1, activation="swiglu", cu_seqlens_m=cu,
           preact_out=h, postact_out=A, store_preact=True, concat_layout=("B",))

# ---- backward: gemm_dgated reads h, recomputes a_prime = s*swiglu(h), dh = dswiglu ----
dh = torch.empty(M, TWO_I, dtype=DT, device=DEV)
a_prime = torch.empty(M, I, dtype=DT, device=DEV)
for b2desc, B2 in (("W2", W2), ("W2.perm021", W2.permute(0, 2, 1).contiguous())):
    try:
        out = gemm_dgated(dY, B2, PreAct=h, activation="swiglu", dx_out=dh,
                          postact_out=a_prime, colvec_scale=s, colvec_reduce=True,
                          cu_seqlens_m=cu, dynamic_scheduler=False)
        print(f"gemm_dgated OK with B2={b2desc}; returns {type(out)} len={len(out) if isinstance(out,(tuple,list)) else '-'}")
        break
    except Exception as e:
        print(f"gemm_dgated B2={b2desc} FAIL: {repr(e)[:160]}")

# ---- torch reference (per-expert, plain) ----
off = [0] + list(accumulate(split))
A_ref = torch.empty(M, I, dtype=DT, device=DEV)
for g in range(G):
    sg, eg = off[g], off[g + 1]
    hp = x[sg:eg].float() @ W1[g].float().T            # [m,2I] gate||up
    A_ref[sg:eg] = (F.silu(hp[:, :I]) * hp[:, I:]).to(DT)
aprime_ref = (A_ref.float() * s.view(-1, 1)).to(DT)

def rel(a, b): return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-9)).item()
print(f"A (gemm_gated postact)      rel vs torch swiglu: {rel(A, A_ref):.3e}")
print(f"a_prime (gemm_dgated recomp) rel vs torch s*swiglu: {rel(a_prime, aprime_ref):.3e}")
# QuACK SELF-CONSISTENCY: a_prime should == s * A  (gemm_dgated re-derived postact from the h that
# gemm_gated stored). If ~0, the h emit->read round-trip through the matched pair is CORRECT,
# regardless of whether my standalone torch ref models QuACK's gather/layout convention.
print(f"a_prime vs s*A (QuACK round-trip self-consistency): {rel(a_prime, (s.view(-1,1)*A.float()).to(DT)):.3e}")
print("PAIR_CHECK_DONE")
