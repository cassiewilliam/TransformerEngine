"""SonicMoE F2 rung-2a: direct test of the tex.te_cutlass_grouped_swiglu binding vs a torch reference.

Verifies the whole Python -> pybind -> C-API -> CUTLASS SM100 kernel path produces silu(gate)*up:
  A[M, I] = silu(X @ W1[0:I]^T) * (X @ W1[I:2I]^T),  per-expert grouped (gate||up = first/second half).
Run inside the built TE container on a clean GPU:
  CUDA_VISIBLE_DEVICES=1 python qa/te_cutlass_swiglu_test.py
"""

import torch
import torch.nn.functional as F
import transformer_engine_torch as tex

torch.manual_seed(0)
G, Me, I, d = 32, 768, 2048, 512  # canonical 4K-MoE shape (uniform, Me=768)
M = G * Me
dtype = torch.bfloat16
dev = "cuda"

x = torch.randn(M, d, dtype=dtype, device=dev) * 0.1
w1 = torch.randn(G * 2 * I, d, dtype=dtype, device=dev) * 0.02  # [G*2I, d] gate||up stacked per expert

# Fused kernel: uniform packing (m_tile_expert=None, M_varlen=0). sm_count=0 => kernel auto-detects.
A = tex.te_cutlass_grouped_swiglu(x, w1, None, G, Me, I, d, 0, 0)  # [M, I]
assert tuple(A.shape) == (M, I), A.shape
assert A.dtype == dtype

# fp32 reference, per expert: gate = X @ W1[:I]^T, up = X @ W1[I:2I]^T, A = silu(gate) * up
w1v = w1.view(G, 2 * I, d).float()
ref = torch.empty(M, I, dtype=torch.float32, device=dev)
for e in range(G):
    xe = x[e * Me : (e + 1) * Me].float()
    h = xe @ w1v[e].t()  # [Me, 2I]
    gate, up = h[:, :I], h[:, I:]
    ref[e * Me : (e + 1) * Me] = F.silu(gate) * up

diff = (A.float() - ref).abs()
rel = diff / (ref.abs() + 1e-3)
n_fail = int(((diff > 5e-2) & (rel > 5e-2)).sum().item())
max_abs = diff.max().item()
print(f"shape M={M} I={I} d={d} G={G} dtype={dtype}")
print(f"n_fail (abs>5e-2 AND rel>5e-2): {n_fail} / {M * I}  max_abs={max_abs:.4f}")
print("PASS" if n_fail == 0 else "FAIL")
