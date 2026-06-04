"""M1a: validate + bench the custom DownProj-bwd GEMM dA = dY · W2ᵀ (Approach A, single-acc passthrough).

The kernel reuses the forward (X·W1ᵀ) structure by feeding W2 TRANSPOSED (W2ᵀ [G·I,d], expert in N-dim).
  dA[m,i] = Σ_j dY[m,j] · W2[e][j,i]   (FC2 dgrad; per-expert dY[Me,d] @ W2[e][d,I] = dA[Me,I])
Validate vs a per-expert torch reference, then measure TF/s vs graph-safe (ref ≈ 295 TF/s uniform).
Uniform Me=768 (=3·256, 256-ALIGNED → no padding) first.

  CUDA_VISIBLE_DEVICES=<idle> python qa/m1a_dproj_gemm.py
"""

import torch
import transformer_engine  # noqa: F401  (RTLD_GLOBAL first)
import transformer_engine_torch as tex

G, D, I = 32, 2048, 512
ME = 768
M = G * ME
dev, dt = "cuda", torch.bfloat16
gemm_flop = 2.0 * M * I * D  # 51.5 GFLOP

torch.manual_seed(0)
dY = torch.randn(M, D, dtype=dt, device=dev)
W2 = (torch.randn(G, D, I, dtype=dt, device=dev) * 0.02)        # FC2 weight [G, d=out, I=in]
W2t = W2.transpose(-1, -2).contiguous().reshape(G * I, D)        # W2ᵀ stacked [G·I, d] (expert in N)
dgrad_dummy = torch.zeros(M, I, dtype=dt, device=dev)           # unused by M1a passthrough (binding needs it)
print(f"M={M} G={G} d={D} I={I}  dY{tuple(dY.shape)} W2t{tuple(W2t.shape)}  {torch.cuda.get_device_name(0)}")


def call():
    # te_cutlass_grouped_dswiglu(x=dY, w1=W2ᵀ, dgrad, m_tile_expert=None(uniform), prob=None, G, Me, I, d,
    #                            M_varlen=0, math_sm_count=0) -> dA[M, I]
    return tex.te_cutlass_grouped_dswiglu(dY, W2t, dgrad_dummy, None, None, G, ME, I, D, 0, 0)


dA = call().view(M, I)

# ---- correctness vs per-expert torch reference ----
ref = torch.empty(M, I, dtype=torch.float32, device=dev)
off = 0
for e in range(G):
    ref[off:off + ME] = dY[off:off + ME].float() @ W2[e].float()  # [Me,d] @ [d,I] = [Me,I]
    off += ME
diff = (dA.float() - ref).abs()
tol = 0.25 + 0.125 * ref.abs()
n_fail = int((diff > tol).sum().item())
print(f"M1A_VALIDATE n_fail={n_fail}/{dA.numel()} max_abs={diff.max().item():.5f} "
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
print(f"M1A_PERF {ms*1000:.2f}us {gemm_flop/(ms/1e3)/1e12:.1f}TF/s  (graph-safe ref ~295 TF/s uniform)")
