"""SonicMoE F2: end-to-end DROP-IN + speedup test for the BF16 fused MoE op.

This proves that the committed op-fusion

    [GroupedLinear(up, d -> 2I) -> ScaledSwiGLU -> GroupedLinear(down, I -> d)]

fused into the single CUTLASS BF16 kernel ``tex.te_cutlass_grouped_swiglu``
(gated by ``NVTE_USE_FUSED_MOE=1``, SM100 / B200 only) is a faithful, semantically
identical drop-in replacement for the un-fused per-op path:

  1. forward output ``y`` matches a high-precision torch reference,
  2. ALL gradients match: d(input), d(fc1 weight per expert), d(fc2 weight per
     expert), and d(prob) (the per-token router gate),
  3. it then times fused vs un-fused and reports the speedup.

It mirrors ``tests/pytorch/test_fusible_ops.py::test_grouped_mlp`` (the canonical
template) in construction / reference / asserts, stripped to BF16, no bias, no
quantization, and adapted to the CUTLASS kernel's gate||up weight layout.

----------------------------------------------------------------------------
HOW TO RUN ON B200 (the fusion registers at IMPORT time, gated by the env var
+ an lru_cache, so the only clean fused-vs-unfused toggle is two processes):

  # correctness (runs the fused path and asserts drop-in equality)
  CUDA_VISIBLE_DEVICES=0 NVTE_USE_FUSED_MOE=1 python qa/te_fused_moe_e2e_test.py

  # perf, fused
  CUDA_VISIBLE_DEVICES=0 NVTE_USE_FUSED_MOE=1 MODE=fused   python qa/te_fused_moe_e2e_test.py
  # perf, un-fused baseline (fusion never registers -> Sequential runs per-op)
  CUDA_VISIBLE_DEVICES=0 NVTE_USE_FUSED_MOE=0 MODE=unfused python qa/te_fused_moe_e2e_test.py

  # one-shot: correctness + both perf modes spawned as subprocesses + speedup
  CUDA_VISIBLE_DEVICES=0 python qa/te_fused_moe_e2e_test.py --all

Default (no args): correctness asserts (needs NVTE_USE_FUSED_MOE=1) + a perf run
in whatever mode the current process registered.
----------------------------------------------------------------------------

============================================================================
INTERLEAVE DECISION  (the #1 correctness risk -- read this).
============================================================================
``validate_grouped_mlp_dims`` (transformer_engine/pytorch/ops/_common.py:249)
HARD-REQUIRES the activation's ``glu_interleave_size == 32`` for ANY fused grouped
MLP, and the fused op's ``__init__`` calls it
(forward_fused_moe.py:123). So the ``ScaledSwiGLU`` MUST be built with
``glu_interleave_size=32`` or the fused op refuses to construct.

BUT the CUTLASS kernel does NOT interleave. Per qa/te_cutlass_swiglu_test.py:21-30
and forward_fused_moe.py:_get_fc1_weight_2d (forward_fused_moe.py:369-395), the
kernel reads the FC1 weight rows VERBATIM and splits each expert's [2I, d] block as
    gate = rows [e*2I : e*2I + I]   (plain first half)
    up   = rows [e*2I + I : e*2I + 2I] (plain second half)
i.e. PLAIN, un-interleaved gate||up. The kernel ignores ``glu_interleave_size``.

The un-fused ScaledSwiGLU path, in contrast, takes FC1's output h=[M,2I] and
DE-INTERLEAVES its 2I columns in 32-wide blocks before chunking into gate/up
(swiglu.py:447-456: reshape to [..., 2I/(2*32), 2, 32], transpose(1,2)). Equivalent
to permuting the 2I FC1 weight rows by the same de-interleave.

To make FUSED == UNFUSED == REFERENCE we therefore keep ONE canonical
"plain gate||up" weight ``W_plain[2I, d]`` (gate=[:I], up=[I:2I]) and:
  * REFERENCE (fp32): use W_plain directly: gate=h[:, :I], up=h[:, I:].
  * FUSED path: GroupedLinear.fc1 weight = W_plain  (kernel reads it plain -> correct).
  * UN-FUSED path: GroupedLinear.fc1 weight = INTERLEAVE_ROWS(W_plain), so that the
    activation's 32-wide DE-interleave of FC1's output undoes it and restores plain
    gate||up before the silu*up. ``_interleave_rows`` / ``_deinterleave_rows`` below
    are exact inverses (verified in-process before any GPU work).

Net: a single ``W_plain`` drives all three; the fused weight is fed plain, the
unfused weight is fed pre-interleaved. d(fc1 weight) is compared in the SAME plain
frame on both sides (we de-interleave the unfused grad back before comparing).
VERIFY on B200: if asserts (1)/(2) fail ONLY for fc1-weight-grad or only on the
unfused side, the interleave handedness here is the prime suspect -- flip
``_interleave_rows`` <-> ``_deinterleave_rows`` (they are labeled).
============================================================================
"""

from __future__ import annotations

import os
import sys

# Import order mandate: transformer_engine (loads libtransformer_engine.so with
# RTLD_GLOBAL so the tex symbols resolve) BEFORE transformer_engine_torch.
import torch
import transformer_engine  # noqa: F401
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # noqa: F401
from transformer_engine.pytorch import ops as te_ops

# ---------------------------------------------------------------------------
# Config -- BF16, no bias, no quantization (minimal + correct, per the ask).
# ---------------------------------------------------------------------------
GLU_INTERLEAVE = 32           # MANDATED by validate_grouped_mlp_dims (==32).
TILE_M = 256                  # kernel requires each expert's token count % 256 == 0.

# Default shape: realistic-ish 4K-MoE-flavored but small enough for a quick run.
#   G experts, hidden d, ffn I (fc1 out = 2I).  Each expert: Me tokens (mult of 256).
G = int(os.environ.get("MOE_G", "8"))
D = int(os.environ.get("MOE_D", "512"))
I = int(os.environ.get("MOE_I", "2048"))   # noqa: E741 -- matches kernel arg name
ME = int(os.environ.get("MOE_ME", "768"))  # tokens per expert: 768 = 3*256 (NON-zero, mult of 256)
MODE = os.environ.get("MODE", "auto")      # "fused" | "unfused" | "auto"
DTYPE = torch.bfloat16
DEV = "cuda"

# bf16-appropriate tolerances, matching test_grouped_mlp's loose sanity tols
# (test_fusible_ops.py:4010) plus a relative gate like the kernel qa test
# (te_cutlass_swiglu_test.py:40).
ATOL = float(os.environ.get("MOE_ATOL", "0.25"))
RTOL = float(os.environ.get("MOE_RTOL", "0.125"))


# ---------------------------------------------------------------------------
# Interleave helpers (exact inverses; see the INTERLEAVE DECISION block).
# These permute the 2I FC1 *rows* (dim 0 of a [2I, d] weight). They mirror the
# column reshape/transpose the activation applies to the [M, 2I] FC1 output
# (swiglu.py:447-456 forward, :519-528 backward inverse).
# ---------------------------------------------------------------------------
def _interleave_rows(w_plain: torch.Tensor, two_i: int, blk: int = GLU_INTERLEAVE) -> torch.Tensor:
    """plain gate||up rows -> block-interleaved rows.

    Inverse of the activation's de-interleave: if the GroupedLinear holds
    interleaved rows, FC1 output columns are interleaved, and the activation's
    de-interleave restores plain gate||up. Shapes: w_plain [2I, d] -> [2I, d].
    """
    d = w_plain.shape[1]
    # Forward de-interleave is: reshape(-1, 2I/(2*blk), 2, blk).transpose(1,2).
    # Its inverse (to PRE-bake into the weight) is: reshape(-1, 2, 2I/(2*blk), blk)
    # .transpose(1,2). We apply it to the row dim by treating rows as the "2I" axis.
    w = w_plain.reshape(2, two_i // (2 * blk), blk, d)
    w = w.transpose(0, 1).contiguous()  # [2I/(2blk), 2, blk, d]
    return w.reshape(two_i, d).contiguous()


def _deinterleave_rows(w_inter: torch.Tensor, two_i: int, blk: int = GLU_INTERLEAVE) -> torch.Tensor:
    """block-interleaved rows -> plain gate||up rows (exact inverse of above)."""
    d = w_inter.shape[1]
    w = w_inter.reshape(two_i // (2 * blk), 2, blk, d)
    w = w.transpose(0, 1).contiguous()  # [2, 2I/(2blk), blk, d]
    return w.reshape(two_i, d).contiguous()


def _self_check_interleave() -> None:
    """Prove _interleave_rows / _deinterleave_rows are inverses (CPU, pre-GPU)."""
    two_i = 2 * I
    w = torch.randn(two_i, D)
    back = _deinterleave_rows(_interleave_rows(w, two_i), two_i)
    assert torch.equal(w, back), "interleave/deinterleave are NOT inverses -- fix the layout helpers"
    # Also prove that interleaving the rows then de-interleaving the resulting
    # FC1-output columns (as the activation does) restores plain order. We model
    # the activation's column de-interleave on an identity-ish [tokens, 2I].
    h = torch.arange(two_i, dtype=torch.float32).reshape(1, two_i)  # columns 0..2I-1
    # If weight rows were interleaved, output columns are interleaved the same way.
    h_inter = h[:, _interleave_rows(torch.arange(two_i).reshape(two_i, 1).float(), two_i)
                .reshape(two_i).long()]
    # activation de-interleave (swiglu.py:447-456):
    blk = GLU_INTERLEAVE
    h_de = h_inter.reshape(-1, two_i // (2 * blk), 2, blk).transpose(1, 2).reshape(1, two_i)
    assert torch.equal(h_de, h), "activation de-interleave does not restore plain gate||up"
    print("[selfcheck] interleave/de-interleave inverses OK")


# ---------------------------------------------------------------------------
# Reference (high-precision torch) of the UN-fused MoE semantics.
# Mirrors test_grouped_mlp's x_ref path (test_fusible_ops.py:3848-3883) but in
# the PLAIN gate||up frame (gate=h[:, :I], up=h[:, I:]) the kernel uses.
# ---------------------------------------------------------------------------
def torch_reference(x, w1_plain, w2, prob, split_sizes):
    """fp32 reference. Returns (y, dx, dw1_plain, dw2, dprob) for loss=(y*dy).sum().

    x          : [M, d]        requires_grad
    w1_plain   : [G, 2I, d]    plain gate||up per expert, requires_grad
    w2         : [G, d, I]     down-proj per expert, requires_grad
    prob       : [M]           per-token router gate, requires_grad
    split_sizes: list[int] (each a multiple of 256), sum == M
    """
    xf = x.detach().double().requires_grad_(True)
    w1f = w1_plain.detach().double().requires_grad_(True)
    w2f = w2.detach().double().requires_grad_(True)
    pf = prob.detach().double().requires_grad_(True)

    xs = torch.split(xf, split_sizes)
    ps = torch.split(pf, split_sizes)
    ys = []
    for e in range(G):
        xe = xs[e]
        h = torch.nn.functional.linear(xe, w1f[e])  # [Me, 2I] = xe @ w1[e]^T
        gate, up = h[:, :I], h[:, I:]                # PLAIN gate||up (kernel layout)
        a = torch.nn.functional.silu(gate) * up      # SwiGLU
        a = a * ps[e].unsqueeze(-1)                   # router-prob multiply
        ye = torch.nn.functional.linear(a, w2f[e])   # [Me, d] = a @ w2[e]^T
        ys.append(ye)
    y = torch.cat(ys, dim=0)

    # Deterministic upstream grad dy (same seed-derived tensor used by the TE run).
    dy = _make_dy(y.shape, y.device)
    (y.double() * dy.double()).sum().backward()

    return (
        y.detach(),
        xf.grad.detach(),
        w1f.grad.detach(),    # [G, 2I, d] PLAIN frame
        w2f.grad.detach(),    # [G, d, I]
        pf.grad.detach(),     # [M]
    )


# ---------------------------------------------------------------------------
# Deterministic data so the two perf subprocesses + the reference all agree.
# ---------------------------------------------------------------------------
def _make_dy(shape, device):
    g = torch.Generator(device="cpu").manual_seed(1234)
    return (torch.rand(shape, generator=g, dtype=torch.float32) * 0.5 - 0.25).to(
        device=device, dtype=DTYPE
    )


def make_inputs():
    """Build deterministic x, per-expert W_plain[2I,d], W2[d,I], prob, split_sizes."""
    torch.manual_seed(0)
    split_sizes = [ME] * G                      # uniform, each = ME (mult of 256), non-zero
    for s in split_sizes:
        assert s % TILE_M == 0 and s > 0, f"split {s} must be a non-zero multiple of {TILE_M}"
    M = sum(split_sizes)

    g = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.rand(M, D, generator=g) * 0.5 - 0.25).to(DEV, DTYPE)
    w1_plain = (torch.rand(G, 2 * I, D, generator=g) * 0.04 - 0.02).to(DEV, DTYPE)
    w2 = (torch.rand(G, D, I, generator=g) * 0.04 - 0.02).to(DEV, DTYPE)
    prob = (torch.rand(M, generator=g) + 0.25).to(DEV, torch.float32)  # avoid ~0 for rel checks
    return x, w1_plain, w2, prob, split_sizes, M


def build_model(w1_plain, w2, split_sizes, *, fused_mode: bool):
    """Build te_ops.Sequential(GroupedLinear, ScaledSwiGLU, GroupedLinear).

    fc1 weight is fed PLAIN (fused) or INTERLEAVED (unfused) per the INTERLEAVE
    DECISION block, so fused==unfused==reference. fc2 weight is identical in both.
    """
    two_i = 2 * I
    fc1 = te_ops.GroupedLinear(
        G, D, two_i, bias=False, device=DEV, dtype=DTYPE, single_grouped_weight=False
    )
    # MANDATED glu_interleave_size=32 (validate_grouped_mlp_dims); the kernel
    # ignores it but the constructor + the unfused activation honor it.
    act = te_ops.ScaledSwiGLU(glu_interleave_size=GLU_INTERLEAVE)
    fc2 = te_ops.GroupedLinear(
        G, I, D, bias=False, device=DEV, dtype=DTYPE, single_grouped_weight=False
    )

    with torch.no_grad():
        for e in range(G):
            if fused_mode:
                # Kernel reads weight rows plain -> feed W_plain verbatim.
                getattr(fc1, f"weight{e}").copy_(w1_plain[e])
            else:
                # Activation de-interleaves FC1 output -> pre-interleave the rows
                # so plain gate||up is restored before silu*up.
                getattr(fc1, f"weight{e}").copy_(_interleave_rows(w1_plain[e], 2 * I))
            getattr(fc2, f"weight{e}").copy_(w2[e])

    model = te_ops.Sequential(fc1, act, fc2)
    return model, fc1, fc2


def run_te(model, fc1, fc2, x, prob, split_sizes, *, backward: bool):
    """Run the Sequential. Returns (y, dx, dw1_plain, dw2, dprob).

    Call convention mirrors test_grouped_mlp (test_fusible_ops.py:3971-3972) with
    NO bias: 3 extra inputs -> FC1 gets split_sizes, ScaledSwiGLU gets prob, FC2
    gets split_sizes.
    """
    split_t = torch.tensor(split_sizes, dtype=torch.int64, device=DEV)
    xin = x.detach().clone().requires_grad_(True)
    pin = prob.detach().clone().requires_grad_(True)

    # module(x, split_sizes, probs, split_sizes)  -- VERIFY: 3 extra inputs, no bias.
    y = model(xin, split_t, pin, split_t)

    dx = dw1 = dw2 = dp = None
    if backward:
        dy = _make_dy(y.shape, y.device)
        y.backward(dy)
        dx = xin.grad.detach()
        dp = pin.grad.detach()
        # Per-expert weight grads. fc1 grad is in the layout the GroupedLinear
        # holds (plain for fused, interleaved for unfused) -> normalize to PLAIN.
        dw1_list, dw2_list = [], []
        for e in range(G):
            g1 = getattr(fc1, f"weight{e}").grad.detach()
            dw1_list.append(g1 if MODE_IS_FUSED else _deinterleave_rows(g1, 2 * I))
            dw2_list.append(getattr(fc2, f"weight{e}").grad.detach())
        dw1 = torch.stack(dw1_list, dim=0)   # [G, 2I, d] PLAIN
        dw2 = torch.stack(dw2_list, dim=0)   # [G, d, I]
    return y.detach(), dx, dw1, dw2, dp


# Whether THIS process built the model in fused mode (controls dw1 frame above).
MODE_IS_FUSED = True


# ---------------------------------------------------------------------------
# Assert helpers (report n_fail + max_abs + PASS/FAIL, like the qa kernel test).
# ---------------------------------------------------------------------------
def _check(name, out, ref, atol=ATOL, rtol=RTOL):
    out = out.detach().float()
    ref = ref.detach().float()
    if out.shape != ref.shape:
        print(f"  {name:14s}: SHAPE MISMATCH out={tuple(out.shape)} ref={tuple(ref.shape)} FAIL")
        return False
    diff = (out - ref).abs()
    tol = atol + rtol * ref.abs()
    n_fail = int((diff > tol).sum().item())
    max_abs = diff.max().item() if diff.numel() else 0.0
    ok = n_fail == 0
    print(f"  {name:14s}: n_fail {n_fail:>10d} / {out.numel():<10d}  "
          f"max_abs={max_abs:.5f}  {'PASS' if ok else 'FAIL'}")
    return ok


def is_fused_op_present(model) -> bool:
    """Report whether the forward fusion actually replaced the triple."""
    try:
        from transformer_engine.pytorch.ops.fused.forward_fused_moe import (
            ForwardFusedMoE_CutlassSwiGLU_BF16,
        )
    except Exception:  # pylint: disable=broad-except
        return False
    fuser = model._module_groups[0]
    fwd = getattr(fuser, "_forward_ops", None)
    if fwd is None:
        # Force a build by faking the fuser bookkeeping is not trivial; just report.
        return False
    return any(isinstance(op, ForwardFusedMoE_CutlassSwiGLU_BF16) for op, _ in fwd)


# ---------------------------------------------------------------------------
# Correctness: run fused (NVTE_USE_FUSED_MOE=1) and assert drop-in vs reference.
# ---------------------------------------------------------------------------
def run_correctness():
    global MODE_IS_FUSED
    MODE_IS_FUSED = True
    fused_supported = (
        int(os.environ.get("NVTE_USE_FUSED_MOE", "0")) > 0
        and torch.cuda.get_device_capability(0)[0] == 10
    )
    print("=" * 76)
    print("CORRECTNESS: fused (NVTE_USE_FUSED_MOE=1) vs high-precision torch reference")
    print(f"  G={G} d={D} I={I} Me={ME} M={G*ME} dtype={DTYPE} "
          f"interleave={GLU_INTERLEAVE} tile_m={TILE_M}")
    print(f"  NVTE_USE_FUSED_MOE={os.environ.get('NVTE_USE_FUSED_MOE','0')} "
          f"fused_supported={fused_supported}")
    if not fused_supported:
        print("  NOTE: NVTE_USE_FUSED_MOE!=1 or not SM100 -> the fusion did NOT register; "
              "this run validates the UN-FUSED path semantics only. Re-run with "
              "NVTE_USE_FUSED_MOE=1 on B200 to validate the actual fused kernel.")
    print("=" * 76)

    x, w1_plain, w2, prob, split_sizes, _M = make_inputs()

    # Reference (plain frame).
    y_ref, dx_ref, dw1_ref, dw2_ref, dp_ref = torch_reference(
        x, w1_plain, w2, prob, split_sizes
    )

    # Fused TE run (built plain because the kernel reads weight rows plain).
    model, fc1, fc2 = build_model(w1_plain, w2, split_sizes, fused_mode=True)
    y, dx, dw1, dw2, dp = run_te(model, fc1, fc2, x, prob, split_sizes, backward=True)

    print(f"  fused op actually applied: {is_fused_op_present(model)} "
          "(False here just means the fuser graph wasn't introspectable; the "
          "asserts below still validate the executed path)")
    print("-" * 76)

    results = [
        _check("y (fwd)", y, y_ref),
        _check("d_input", dx, dx_ref),
        _check("d_fc1_weight", dw1, dw1_ref),   # per-expert, PLAIN gate||up frame
        _check("d_fc2_weight", dw2, dw2_ref),   # per-expert
        _check("d_prob", dp, dp_ref),
    ]
    ok = all(results)
    print("-" * 76)
    print("DROP-IN RESULT:", "PASS -- fused op is a faithful drop-in" if ok
          else "FAIL -- fused op DIVERGES from reference (see per-tensor lines)")
    print("=" * 76)
    return ok


# ---------------------------------------------------------------------------
# Perf: time fused vs un-fused (forward, and forward+backward) with CUDA events.
# ---------------------------------------------------------------------------
def _flops_fwd(M):
    """FLOPs for one MoE forward: FC1 (M*2I*d) + FC2 (M*I*d), x2 for MAC."""
    return 2.0 * M * (2 * I) * D + 2.0 * M * I * D


def run_perf(mode: str, iters: int = 50, warmup: int = 10):
    global MODE_IS_FUSED
    MODE_IS_FUSED = (mode == "fused")
    x, w1_plain, w2, prob, split_sizes, M = make_inputs()
    model, fc1, fc2 = build_model(w1_plain, w2, split_sizes, fused_mode=(mode == "fused"))
    split_t = torch.tensor(split_sizes, dtype=torch.int64, device=DEV)

    def fwd_only():
        with torch.no_grad():
            return model(x, split_t, prob, split_t)

    def fwd_bwd():
        xin = x.detach().clone().requires_grad_(True)
        pin = prob.detach().clone().requires_grad_(True)
        for e in range(G):
            getattr(fc1, f"weight{e}").grad = None
            getattr(fc2, f"weight{e}").grad = None
        y = model(xin, split_t, pin, split_t)
        y.backward(_make_dy(y.shape, y.device))

    def time_it(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return ms[len(ms) // 2]  # median ms/iter

    fwd_ms = time_it(fwd_only)
    fb_ms = time_it(fwd_bwd)
    tflops_fwd = _flops_fwd(M) / (fwd_ms * 1e-3) / 1e12

    print("=" * 76)
    print(f"PERF [mode={mode}]  M={M} G={G} d={D} I={I} Me={ME} dtype={DTYPE}")
    print(f"  NVTE_USE_FUSED_MOE={os.environ.get('NVTE_USE_FUSED_MOE','0')}  "
          f"iters={iters} warmup={warmup}")
    print(f"  fwd      : {fwd_ms:8.3f} ms/iter   {tflops_fwd:8.2f} TFLOP/s (fwd GEMMs)")
    print(f"  fwd+bwd  : {fb_ms:8.3f} ms/iter")
    print("=" * 76)
    # Machine-readable line for the --all driver to parse.
    print(f"PERFCSV,{mode},{fwd_ms:.5f},{fb_ms:.5f},{tflops_fwd:.4f}")
    return fwd_ms, fb_ms


# ---------------------------------------------------------------------------
# --all driver: correctness here, then spawn two subprocesses for fused/unfused
# perf (env-toggled), parse PERFCSV, print the speedup.
# ---------------------------------------------------------------------------
def run_all():
    import subprocess
    import re

    # Correctness needs the fusion registered in THIS process.
    if int(os.environ.get("NVTE_USE_FUSED_MOE", "0")) <= 0:
        print("[--all] NOTE: re-exec correctness child with NVTE_USE_FUSED_MOE=1")
    ok = True
    env_fused = dict(os.environ, NVTE_USE_FUSED_MOE="1", MODE="correctness")
    cp = subprocess.run([sys.executable, __file__, "--correctness"], env=env_fused)
    ok = ok and cp.returncode == 0

    perf = {}
    for mode, flag in (("fused", "1"), ("unfused", "0")):
        env = dict(os.environ, NVTE_USE_FUSED_MOE=flag, MODE=mode)
        out = subprocess.run(
            [sys.executable, __file__, "--perf"], env=env, capture_output=True, text=True
        )
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        for line in out.stdout.splitlines():
            m = re.match(r"PERFCSV,(\w+),([\d.]+),([\d.]+),([\d.]+)", line)
            if m:
                perf[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

    print("=" * 76)
    print("SPEEDUP SUMMARY")
    if "fused" in perf and "unfused" in perf:
        f_fwd, f_fb, _ = perf["fused"]
        u_fwd, u_fb, _ = perf["unfused"]
        print(f"  forward     : unfused {u_fwd:8.3f} ms -> fused {f_fwd:8.3f} ms  "
              f"=> {u_fwd / f_fwd:5.2f}x")
        print(f"  fwd+bwd     : unfused {u_fb:8.3f} ms -> fused {f_fb:8.3f} ms  "
              f"=> {u_fb / f_fb:5.2f}x")
    else:
        print("  (missing one of fused/unfused perf runs -- check subprocess output above)")
    print("=" * 76)
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- this test requires a B200 (SM100). Aborting.")
        sys.exit(2)
    _self_check_interleave()

    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--all":
        ok = run_all()
        sys.exit(0 if ok else 1)
    if arg == "--correctness":
        sys.exit(0 if run_correctness() else 1)
    if arg == "--perf":
        run_perf(MODE if MODE in ("fused", "unfused") else "fused")
        sys.exit(0)

    # Default: correctness (if fused registered) + a single perf run in current mode.
    ok = True
    if int(os.environ.get("NVTE_USE_FUSED_MOE", "0")) > 0:
        ok = run_correctness()
    else:
        print("NVTE_USE_FUSED_MOE!=1 -> skipping fused correctness; running perf only. "
              "Use --all for the full fused-vs-unfused comparison.")
    run_perf(MODE if MODE in ("fused", "unfused") else ("fused" if ok else "unfused"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
