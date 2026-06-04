"""M2a: validate the fused DownProj-GEMM + dswiglu kernel (dA=dY·W2ᵀ in TMEM, gate/up from SAVED h).

  dA[m,:] = dY[m,:] @ W2[e]                 (FC2 dgrad, in TMEM)
  grad    = dA · prob[m]
  dY1[:, :I]  = grad · up · silu'(gate)     ;  dY1[:, I:] = grad · silu(gate)
  where gate = h[:, :I], up = h[:, I:2I]    (the SAVED SwiGLU input, read in the epilogue)
Validate dY1 vs a torch reference; then time the fused kernel. Uniform Me=768 (256-aligned).

  CUDA_VISIBLE_DEVICES=<idle> python qa/m2a_dswiglu.py
"""

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex

G, D, I = 32, 2048, 512
ME = 768
M = G * ME
dev, dt = "cuda", torch.bfloat16
gemm_flop = 2.0 * M * I * D  # the dA = dY·W2 GEMM (the fused kernel = this GEMM + dswiglu epilogue)

torch.manual_seed(0)
dY = torch.randn(M, D, dtype=dt, device=dev)
W2 = (torch.randn(G, D, I, dtype=dt, device=dev) * 0.02)       # FC2 weight [G, d, I]
W2t = W2.transpose(-1, -2).contiguous().reshape(G * I, D)       # W2ᵀ [G·I, d]
h = (torch.randn(M, 2 * I, dtype=dt, device=dev) * 0.5)        # SAVED SwiGLU input h = [gate || up]
prob = (torch.rand(M, device=dev, dtype=torch.float32) + 0.25)
print(f"M={M} G={G} d={D} I={I}  {torch.cuda.get_device_name(0)}")


def call():
    # te_cutlass_grouped_dswiglu(x=dY, w1=W2ᵀ, dgrad=h, m_tile_expert=None, prob, G, Me, I, d,
    #                            M_varlen=0, math_sm_count=0) -> dY1[M, 2I]
    return tex.te_cutlass_grouped_dswiglu(dY, W2t, h, None, prob, G, ME, I, D, 0, 0)


dY1 = call().view(M, 2 * I)

# ---- torch reference ----
dA = torch.empty(M, I, dtype=torch.float32, device=dev)
off = 0
for e in range(G):
    dA[off:off + ME] = dY[off:off + ME].float() @ W2[e].float()
    off += ME
gate = h[:, :I].float()
up = h[:, I:].float()
grad = dA * prob[:, None]
sig = torch.sigmoid(gate)
s = gate * sig                       # silu(gate)
siluprime = sig + s * (1 - sig)      # d silu / d gate
dY1_ref = torch.cat([grad * up * siluprime, grad * s], dim=1)  # [M, 2I] = dgate || dup

diff = (dY1.float() - dY1_ref).abs()
tol = 0.25 + 0.125 * dY1_ref.abs()
n_fail = int((diff > tol).sum().item())
print(f"M2A_VALIDATE n_fail={n_fail}/{dY1.numel()} max_abs={diff.max().item():.5f} "
      f"{'PASS' if n_fail == 0 else 'FAIL'}")


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


ms = bench(call)
print(f"M2A_PERF {ms*1000:.2f}us {gemm_flop/(ms/1e3)/1e12:.1f}TF/s  (M1a GEMM-only was 157us/328TF/s)")
