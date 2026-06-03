import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
torch.manual_seed(0)
# canonical 4K MoE: H=d=512, FFN I=2048, 2I=4096, G=32, Me=768 (avg/uniform rep)
G, d, I = 32, 512, 2048
twoI = 2*I
m_splits = [768]*G; M = sum(m_splits)
flop = 4*M*I*d   # gate+up
print(f"4K-MoE FW-Up shape: G={G} d(H)={d} I(FFN)={I} 2I={twoI} M={M}  FLOP={flop/1e9:.1f}G")
x = torch.randn(M, d, dtype=torch.bfloat16, device="cuda")
gl_gate = te.GroupedLinear(G, d, I, bias=False, params_dtype=torch.bfloat16).cuda()
gl_up   = te.GroupedLinear(G, d, I, bias=False, params_dtype=torch.bfloat16).cuda()
gl_m    = te.GroupedLinear(G, d, twoI, bias=False, params_dtype=torch.bfloat16).cuda()
def e2e_2gemm():
    with torch.no_grad():
        return F.silu(gl_gate(x, m_splits)) * gl_up(x, m_splits)
def e2e_merged():
    with torch.no_grad():
        h = gl_m(x, m_splits); return F.silu(h[:, :I]) * h[:, I:]
def bench(fn, N=50):
    for _ in range(15): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(N): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/N
for r in range(3):
    t2, tm = bench(e2e_2gemm), bench(e2e_merged)
    print(f"  run{r}: TE 2-GEMM+SwiGLU {t2:.4f}ms {flop/(t2/1e3)/1e12:.0f}T | TE merged+SwiGLU {tm:.4f}ms {flop/(tm/1e3)/1e12:.0f}T")
