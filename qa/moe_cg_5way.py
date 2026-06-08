"""5-backend CUDA-graph fwd+bwd bench of the mcore MoE expert-GEMM path.

The mcore TEGroupedMLP / op-fuser expert compute IS this te_ops stack:
  GroupedLinear(up, 2I) -> ScaledSwiGLU -> GroupedLinear(down).
Router/all-to-all dispatch is backend-invariant, so the 5 grouped-GEMM backends
differ exactly here. Each is captured in a CUDA graph (fwd+bwd) and timed; configs
that are not graph-safe surface as GRAPH_FAIL (a real result).

MODE (env), mapped onto docs/grouped_gemm_dispatch_map.html:
  multistream : list path, no CUTLASS/FUSED  -> multi_stream_cublas_gemm        (legacy)
  gt_cublas   : grouped-tensor, no CUTLASS    -> execute_grouped_gemm/cublasLtMatmul
  cutlass_list: list path + CUTLASS_GROUPED   -> nvte_multi_tensor_gemm/cutlass_grouped_gemm
  cutlass_gt  : grouped-tensor + CUTLASS      -> execute_grouped_gemm/cutlass_grouped_gemm_device_ptrs
  fused       : NVTE_USE_FUSED_MOE            -> SonicMoE fused swiglu/down/dswiglu
Run: CUDA_VISIBLE_DEVICES=<idle> MODE=cutlass_gt python moe_cg_5way.py
"""
import os, random, sys

MODE = os.environ.get("MODE", "fused")
os.environ["NVTE_USE_FUSED_MOE"] = "1" if MODE == "fused" else "0"
if MODE in ("cutlass_list", "cutlass_gt"):
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"

import torch
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops.basic.grouped_linear import GroupedLinear as _OpGL

# Force the list/legacy (non graph-safe) path for the two list-based configs.
if MODE in ("multistream", "cutlass_list"):
    _OpGL._is_graph_safe_path_supported = staticmethod(lambda **kw: False)

torch.manual_seed(0)
G = int(os.environ.get("MOE_G", 32)); D = int(os.environ.get("MOE_D", 2048))
I = int(os.environ.get("MOE_I", 512)); ME = int(os.environ.get("MOE_ME", 768))
TWO_I = 2 * I; M = G * ME; TILE = 256
DEV, DT = "cuda", torch.bfloat16
random.seed(0); nt = M // TILE
if os.environ.get("MOE_RAGGED", "1") == "1":
    tiles = [min(8, max(1, round(random.gauss(3, 2.2)))) for _ in range(G)]
    diff = nt - sum(tiles); idx = 0
    while diff != 0:
        j = idx % G
        if diff > 0 and tiles[j] < 8: tiles[j] += 1; diff -= 1
        elif diff < 0 and tiles[j] > 1: tiles[j] -= 1; diff += 1
        idx += 1
        if idx > 100 * G: break
    split = [t * TILE for t in tiles]
else:
    split = [ME] * G
mn, mx, av = min(split), max(split), sum(split) // G
fwd = 2 * M * D * TWO_I + 2 * M * I * D     # up(gate||up) + down
flop = fwd * 3.0                            # fwd + ~2x bwd
print(f"MODE={MODE} G={G} D={D} I={I} K[min/avg/max]={mn}/{av}/{mx} "
      f"imbal={mx/(M/G):.1f}x M={M}", file=sys.stderr)

fc1 = te_ops.GroupedLinear(G, D, TWO_I, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
act = te_ops.ScaledSwiGLU(glu_interleave_size=32)
fc2 = te_ops.GroupedLinear(G, I, D, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
model = te_ops.Sequential(fc1, act, fc2)
x = torch.randn(M, D, dtype=DT, device=DEV, requires_grad=True)
dy = torch.randn(M, D, dtype=DT, device=DEV)
prob = (torch.rand(M, device=DEV, dtype=torch.float32) + 0.25)
split_t = torch.tensor(split, dtype=torch.int64, device=DEV)

NVTX = os.environ.get("NVTX", "") == "1"
if NVTX:
    # te_ops.Sequential runs through OperationFuser which calls op.fuser_forward/backward
    # (bypassing nn.Module hooks). Wrap those methods on the op CLASSES so EVERY backend
    # emits the SAME op-level NVTX ranges -> aligned per-operator speedup comparison.
    def _wrap_nvtx(cls, default_label):
        for mname in ("fuser_forward", "fuser_backward"):
            orig = cls.__dict__.get(mname) or getattr(cls, mname, None)
            if orig is None or getattr(orig, "_nvtx_wrapped", False):
                continue
            phase = "fwd" if "forward" in mname else "bwd"

            def make(orig, phase):
                def wrapped(self, *a, **k):
                    lbl = getattr(self, "_nvtx_label", None) or default_label
                    torch.cuda.nvtx.range_push(f"{lbl}.{phase}")
                    try:
                        return orig(self, *a, **k)
                    finally:
                        torch.cuda.nvtx.range_pop()
                wrapped._nvtx_wrapped = True
                return wrapped
            setattr(cls, mname, make(orig, phase))

    from transformer_engine.pytorch.ops.basic.grouped_linear import GroupedLinear as _GL
    from transformer_engine.pytorch.ops.basic.swiglu import ScaledSwiGLU as _SG
    _wrap_nvtx(_GL, "glin")
    _wrap_nvtx(_SG, "swiglu")
    try:
        from transformer_engine.pytorch.ops.fused.forward_fused_moe import (
            ForwardFusedMoE_CutlassSwiGLU_BF16 as _FF,
        )
        from transformer_engine.pytorch.ops.fused.backward_fused_moe import (
            BackwardFusedMoE_CutlassSwiGLU_BF16 as _BF,
        )
        _wrap_nvtx(_FF, "fusedmoe")
        _wrap_nvtx(_BF, "fusedmoe")
    except Exception:
        pass
    fc1._nvtx_label = "up"
    fc2._nvtx_label = "down"


def fb():
    if NVTX:
        torch.cuda.nvtx.range_push("FWD")
    y = model(x, split_t, prob, split_t)
    if NVTX:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("BWD")
    y.backward(dy)
    if NVTX:
        torch.cuda.nvtx.range_pop()
    return y

NSYS = os.environ.get("NSYS_REPLAYS", "")   # if set, just run N iters under profiler then exit
EAGER_NSYS = os.environ.get("EAGER", "") == "1"  # eager (no graph) -> NVTX tags every kernel
if NSYS and EAGER_NSYS:
    # eager profiling pass: full NVTX attribution (python+autograd re-run each iter)
    for _ in range(5):
        fb()
    torch.cuda.synchronize()
    for _ in range(int(NSYS)):
        fb()
    torch.cuda.synchronize()
    print(f"NSYS_EAGER_DONE,{MODE},{NSYS}")
    raise SystemExit(0)
try:
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(5):
            fb()
    torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fb()
    if NSYS:
        for _ in range(int(NSYS)):
            g.replay()
        torch.cuda.synchronize()
        print(f"NSYS_REPLAY_DONE,{MODE},{NSYS}")
    else:
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(100):
            g.replay()
        b.record(); torch.cuda.synchronize()
        ms = a.elapsed_time(b) / 100; tf = flop / (ms / 1e3) / 1e12
        print(f"GRAPH_OK,{MODE},{ms:.4f},{tf:.0f}")
except Exception as e:
    print(f"GRAPH_FAIL,{MODE},{repr(e)[:400]}")
