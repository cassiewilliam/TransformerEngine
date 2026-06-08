"""cudagraph FWD+BWD A/B: NVTE_QUACK_EMIT_H 0 (baseline: recompute h + B2) vs 1 (interleaved emit-h + B2).
Reports element-wise grad diff (emit-h vs baseline) + cudagraph perf (10 rounds avg)."""
import os
os.environ["NVTE_USE_FUSED_MOE"] = "1"
import random, torch
import transformer_engine  # noqa
import transformer_engine.pytorch.ops as te_ops

torch.manual_seed(0); random.seed(0)
DEV, DT = "cuda", torch.bfloat16
G, D, I, ME = 32, 2048, 512, 768
TWO_I = 2 * I; TILE = 256
nt = (G * ME) // TILE
tiles = [min(8, max(1, round(random.gauss(3, 2.2)))) for _ in range(G)]
d_ = nt - sum(tiles); j = 0
while d_ != 0:
    k = j % G
    if d_ > 0 and tiles[k] < 8: tiles[k] += 1; d_ -= 1
    elif d_ < 0 and tiles[k] > 1: tiles[k] -= 1; d_ += 1
    j += 1
    if j > 100 * G: break
split = [t * TILE for t in tiles]
M = sum(split)
fc1 = te_ops.GroupedLinear(G, D, TWO_I, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
act = te_ops.ScaledSwiGLU(glu_interleave_size=32)
fc2 = te_ops.GroupedLinear(G, I, D, bias=False, device=DEV, dtype=DT, single_grouped_weight=False)
model = te_ops.Sequential(fc1, act, fc2)
x = torch.randn(M, D, dtype=DT, device=DEV, requires_grad=True)
dy = torch.randn(M, D, dtype=DT, device=DEV)
prob = (torch.rand(M, device=DEV, dtype=torch.float32) + 0.25)
split_t = torch.tensor(split, dtype=torch.int64, device=DEV)
params = list(model.parameters())


def zero():
    x.grad = None
    for p in params:
        p.grad = None


def capture():
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(3):
            zero(); y = model(x, split_t, prob, split_t); y.backward(dy)
    torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
    zero()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = model(x, split_t, prob, split_t); y.backward(dy)
    return g, y


def grab(emit):
    os.environ["NVTE_QUACK_EMIT_H"] = "1" if emit else "0"
    g, y = capture()
    # one clean iter: zero the (static) grads in-place, replay once
    x.grad.zero_(); [p.grad.zero_() for p in params]
    g.replay(); torch.cuda.synchronize()
    grads = {"y": y.detach().float().clone(), "dx": x.grad.detach().float().clone()}
    for i, p in enumerate(params):
        grads[f"dW{i}"] = p.grad.detach().float().clone()
    # perf: 10 rounds avg
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(10): g.replay()
    b.record(); torch.cuda.synchronize()
    return grads, a.elapsed_time(b) / 10, g


g0, ms0, _graph0 = grab(False)   # baseline (recompute + B2)
g1, ms1, _graph1 = grab(True)    # emit-h interleaved + B2


def rel(a, b): return ((a - b).norm() / (b.norm() + 1e-9)).item()
def mx(a, b): return (a - b).abs().max().item()

print(f"PERF cudagraph fwd+bwd (10-round avg): baseline(emit0)={ms0:.4f}ms  emit-h(emit1)={ms1:.4f}ms  "
      f"speedup={ms0/ms1:.3f}x  saved={ (ms0-ms1)*1000:.1f}us")
np = len(params)
up_keys = [f"dW{i}" for i in (0, np // 4, np // 2 - 1)]      # fc1/up weights
dn_keys = [f"dW{i}" for i in (np // 2, 3 * np // 4, np - 1)]  # fc2/down weights
print(f"ELEMENT diff (emit-h vs baseline)  [params={np}: 0..{np//2-1}=UP, {np//2}..{np-1}=DOWN]:")
for k in ["y", "dx"] + up_keys + dn_keys:
    if k in g0:
        tag = "UP" if (k[0] == "d" and k[1] == "W" and int(k[2:]) < np // 2) else ("DOWN" if k.startswith("dW") else "")
        print(f"  {k:6s}{tag:5s}: max_abs={mx(g1[k], g0[k]):.4e}  rel={rel(g1[k], g0[k]):.4e}")
print("EMITH_AB_DONE")
