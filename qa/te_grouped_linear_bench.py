import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
torch.manual_seed(0)
G, d, I = 32, 2048, 512          # I = ffn half-width (gate=I, up=I); d = model dim
twoI = 2*I                        # = 1024 (gate||up)
m_splits = [[512,4096,8192][e%3] for e in range(G)]; M = sum(m_splits)

# --- alignment check ---
my_params = G*twoI*d              # fused kernel W1 = [G*2I, d]  (gate I rows + up I rows per expert)
my_flop   = 4*M*I*d               # 2 GEMMs (gate + up), each 2*M*I*d
gl = te.GroupedLinear(G, d, twoI, bias=False, params_dtype=torch.bfloat16).cuda()
te_params = sum(p.numel() for p in gl.parameters())
te_flop   = 2*M*twoI*d            # one [M,d]x[d,2I] GEMM == 4*M*I*d
print(f"ALIGN  G={G} d={d} I={I} (2I={twoI})  M={M}")
print(f"  params: fused(G*2I*d)={my_params/1e6:.2f}M   TE GroupedLinear={te_params/1e6:.2f}M   {'MATCH' if my_params==te_params else 'MISMATCH'}")
print(f"  FLOP:   fused(4*M*I*d)={my_flop/1e9:.1f}G    TE(2*M*2I*d)={te_flop/1e9:.1f}G    {'MATCH' if my_flop==te_flop else 'MISMATCH'}")
print(f"  output: fused [M,I]=[{M},{I}]   TE h[M,2I]->SwiGLU[M,I]=[{M},{I}]")

x = torch.randn(M, d, dtype=torch.bfloat16, device="cuda")
def gemm():
    with torch.no_grad(): return gl(x, m_splits)
def e2e():
    with torch.no_grad():
        h = gl(x, m_splits); return F.silu(h[:, :I]) * h[:, I:]
def bench(fn, N=50):
    for _ in range(15): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(N): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/N
for r in range(3):
    mg, me = bench(gemm), bench(e2e)
    print(f"  run{r}: TE GEMM {mg:.4f}ms {my_flop/(mg/1e3)/1e12:.0f}T | TE+SwiGLU {me:.4f}ms {my_flop/(me/1e3)/1e12:.0f}T")
print(f"  fused kernel (GEMM+SwiGLU, 1 pass): 0.824 ms, 673 TFLOPS  [same FLOP basis 4*M*I*d]")
