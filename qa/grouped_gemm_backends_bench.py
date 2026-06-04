"""Non-fused MoE forward: cuBLAS grouped GEMM vs CUTLASS grouped GEMM (F0) vs the SonicMoE fused kernel.

Real 4K-MoE per-card shape (H=2048 hidden, I=512 intermediate -- NOT the swapped d=512/I=2048).
The non-fused path uses the MODULE te.pytorch.GroupedLinear, whose forward goes
  general_grouped_gemm -> tex.te_general_grouped_gemm -> nvte_multi_tensor_gemm
which honors NVTE_USE_CUTLASS_GROUPED_GEMM (0 => cuBLAS, 1 => CUTLASS F0 grouped GEMM). The fused path
uses tex.te_cutlass_grouped_swiglu for up-proj+SwiGLU, then the same down-proj GroupedLinear.

Run:  CUDA_VISIBLE_DEVICES=<idle> python qa/grouped_gemm_backends_bench.py
"""

import os
import torch
import torch.nn.functional as F
import transformer_engine  # noqa: F401  (load libtransformer_engine RTLD_GLOBAL first)
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

torch.manual_seed(0)

# ---- detailed test configuration (real per-card 4K-MoE) ----------------------------------------
G = 32        # local experts per card (256E / EP8)
H = 2048      # hidden size = GEMM input/contraction dim (d)
I = 512       # expert intermediate (SwiGLU single branch)
TWO_I = 2 * I  # 1024 = FC1 gate||up output width
ME = 768      # avg tokens per expert (196608 routed / 256 experts)
M = G * ME    # 24576 routed tokens per card
DTYPE = torch.bfloat16
DEV = "cuda"
m_splits = [ME] * G

fc1_flop = 2 * M * TWO_I * H   # up-proj   [M,H] x [2I,H]^T -> [M,2I]
fc2_flop = 2 * M * H * I       # down-proj [M,I] x [H,I]^T -> [M,H]
fwd_flop = fc1_flop + fc2_flop

print("=" * 78)
print("CONFIG (real 4K-MoE per-card)")
print(f"  G={G}  H(d)={H}  I={I}  2I={TWO_I}  Me={ME}  M={M}  dtype={DTYPE}")
print(f"  up-proj   GEMM: [M,{H}] x [{TWO_I},{H}]^T -> [M,{TWO_I}]   (N=2I={TWO_I}, K=H={H})")
print(f"  down-proj GEMM: [M,{I}] x [{H},{I}]^T -> [M,{H}]   (N=H={H}, K=I={I})")
print(f"  fwd GEMM FLOP = {fwd_flop/1e9:.1f} G  (up {fc1_flop/1e9:.1f} + down {fc2_flop/1e9:.1f})")
print(f"  device={torch.cuda.get_device_name(0)}")
print("=" * 78)

x = torch.randn(M, H, dtype=DTYPE, device=DEV) * 0.1
gl_up = te.GroupedLinear(G, H, TWO_I, bias=False, params_dtype=DTYPE).cuda()  # up-proj MERGED (gate||up, out=2I)
gl_gate = te.GroupedLinear(G, H, I, bias=False, params_dtype=DTYPE).cuda()    # up-proj SPLIT: gate (out=I)
gl_upx = te.GroupedLinear(G, H, I, bias=False, params_dtype=DTYPE).cuda()     # up-proj SPLIT: up   (out=I)
gl_dn = te.GroupedLinear(G, I, H, bias=False, params_dtype=DTYPE).cuda()      # down-proj
# stacked [G*2I, H] gate||up weight for the fused kernel (same shape as gl_up's weights)
w1 = torch.randn(G * TWO_I, H, dtype=DTYPE, device=DEV) * 0.02


def unfused_merged_fwd():    # up-proj = 1 MERGED GEMM (out=2I) + silu  (the strong baseline)
    up = gl_up(x, m_splits)                 # [M, 2I]
    a = F.silu(up[:, :I]) * up[:, I:]       # [M, I]
    return gl_dn(a, m_splits)               # [M, H]


m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device=DEV)  # device splits for graph-safe path


def unfused_2gemm_fwd():     # up-proj = 2 SEPARATE GEMMs (gate + up, each out=I) + silu
    gate = gl_gate(x, m_splits)             # [M, I]
    up = gl_upx(x, m_splits)                # [M, I]
    a = F.silu(gate) * up                   # [M, I]
    return gl_dn(a, m_splits)               # [M, H]


def unfused_graphsafe_fwd():  # MERGED up-proj via the graph-safe cuBLAS 13.4 path (device m_splits)
    up = gl_up(x, m_splits_t)               # [M, 2I]
    a = F.silu(up[:, :I]) * up[:, I:]       # [M, I]
    return gl_dn(a, m_splits_t)             # [M, H]


def fused_fwd():
    a = tex.te_cutlass_grouped_swiglu(x, w1, None, None, G, ME, I, H, 0, 0)  # [M, I]  up+SwiGLU fused
    return gl_dn(a, m_splits)               # [M, H]  down-proj (same GroupedLinear)


def bench(fn, n=50, w=10):
    with torch.no_grad():
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        a = torch.cuda.Event(True)
        b = torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def report(name, ms):
    print(f"  {name:28s}: {ms:.4f} ms/iter   {fwd_flop/(ms/1e3)/1e12:7.1f} TFLOP/s")


print("FORWARD (up-proj + SwiGLU + down-proj), 10 warmup + 100 iters, ALL same-run:")
# graph-safe cuBLAS 13.4 (the e2e/ops default path; device m_splits)
os.environ["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "1"
report("unfused MERGED+silu [graph-safe cuBLAS13.4]", bench(unfused_graphsafe_fwd, n=100))
os.environ["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "0"
for label, flag in [("legacy cuBLAS", "0"), ("CUTLASS_F0", "1")]:
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = flag
    report(f"unfused MERGED(2I)+silu     [{label}]", bench(unfused_merged_fwd, n=100))
    report(f"unfused 2-SEPARATE(g+u)+silu [{label}]", bench(unfused_2gemm_fwd, n=100))
os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"   # down uses CUTLASS F0; up = the fused kernel
report("fused DIRECT(swiglu kernel)+CUTLASS dn", bench(fused_fwd, n=100))
print("=" * 78)
print("NOTE: up-proj FLOP identical for merged vs 2-separate (2*M*2I*H=103.1G); fused fuses silu.")
print("NOTE: this measures the KERNEL (direct call), NOT the forward_fused_moe op (which adds Python overhead).")
