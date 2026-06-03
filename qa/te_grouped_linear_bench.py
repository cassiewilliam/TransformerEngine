import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
torch.manual_seed(0)
G, d, I = 32, 2048, 512
twoI = 2*I
m_splits = [[512,4096,8192][e%3] for e in range(G)]; M = sum(m_splits)
my_flop = 4*M*I*d                 # gate+up = 2 GEMMs (2*M*I*d each)
print(f"shape G={G} d={d} I={I} M={M}  FLOP(gate+up)={my_flop/1e9:.1f}G  [fused: 0.824ms, 673 TFLOPS]")
x = torch.randn(M, d, dtype=torch.bfloat16, device="cuda")

# (A) TWO SEPARATE grouped GEMMs (gate, up) + SwiGLU  <-- the realistic baseline
gl_gate = te.GroupedLinear(G, d, I, bias=False, params_dtype=torch.bfloat16).cuda()
gl_up   = te.GroupedLinear(G, d, I, bias=False, params_dtype=torch.bfloat16).cuda()
def e2e_2gemm():
    with torch.no_grad():
        gate = gl_gate(x, m_splits); up = gl_up(x, m_splits)
        return F.silu(gate) * up
# (B) one merged grouped GEMM (out=2I) + SwiGLU  <-- reference (merged is usually faster)
gl_m = te.GroupedLinear(G, d, twoI, bias=False, params_dtype=torch.bfloat16).cuda()
def e2e_merged():
    with torch.no_grad():
        h = gl_m(x, m_splits); return F.silu(h[:, :I]) * h[:, I:]

def bench(fn, N=50):
    for _ in range(15): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(N): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/N

p2 = sum(p.numel() for p in gl_gate.parameters())+sum(p.numel() for p in gl_up.parameters())
pm = sum(p.numel() for p in gl_m.parameters())
print(f"params: 2-sep(gate+up)={p2/1e6:.2f}M  merged={pm/1e6:.2f}M  (fused W1={G*twoI*d/1e6:.2f}M)")
for r in range(3):
    t2, tm = bench(e2e_2gemm), bench(e2e_merged)
    print(f"run{r}: TE 2-GEMM+SwiGLU {t2:.4f}ms {my_flop/(t2/1e3)/1e12:.0f}T | TE merged+SwiGLU {tm:.4f}ms {my_flop/(tm/1e3)/1e12:.0f}T")
print(f"=> compare end-to-end vs FUSED 0.824ms / 673 TFLOPS")
