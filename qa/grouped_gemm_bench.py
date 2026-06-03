#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped GEMM (TE GroupedLinear) fwd / fwd+bwd benchmark — ms + TFLOPS.
Uniform-K so the CUTLASS fast path is eligible (K%128==0). Compares the active
backend (set NVTE_USE_CUTLASS_GROUPED_GEMM) on the current GPU.
  python grouped_gemm_bench.py --experts 128 --K 2048 --N 512 --mper 512 --dtype bf16
"""
import argparse, torch
import transformer_engine.pytorch as te

def bench(fn, iters=50, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--K", type=int, default=2048)   # in_features (contraction)
    ap.add_argument("--N", type=int, default=512)    # out_features
    ap.add_argument("--mper", type=int, default=512) # tokens per expert (uniform)
    ap.add_argument("--dtype", choices=["bf16","fp16"], default="bf16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ragged", action="store_true",
                    help="uneven tokens/expert -> exercises the varlen-K wgrad (ragged) path")
    a = ap.parse_args()
    dt = torch.bfloat16 if a.dtype=="bf16" else torch.float16
    E, K, N, mper = a.experts, a.K, a.N, a.mper
    if a.ragged:
        # vary tokens/expert around mper (multiples of 128, never 0) -> ragged K for wgrad
        splits = [max(128, mper + 128 * ((i % 7) - 3)) for i in range(E)]
    else:
        splits = [mper] * E
    Mtot = sum(splits)
    import os
    print(f"GPU {torch.cuda.get_device_name(0)} | dtype {dt} | E={E} K={K} N={N} mper={mper} Mtot={Mtot} "
          f"| CUTLASS={os.environ.get('NVTE_USE_CUTLASS_GROUPED_GEMM','0')}")
    m = te.GroupedLinear(E, K, N, bias=False, params_dtype=dt, device="cuda")
    x = torch.randn(Mtot, K, device="cuda", dtype=dt, requires_grad=True)
    flop_fwd = 2*Mtot*K*N            # one grouped GEMM
    def fwd():
        with torch.no_grad(): return m(x, splits)
    def fb():
        x.grad=None
        y = m(x, splits); y.sum().backward()
    fwd_ms = bench(fwd, a.iters)
    fb_ms  = bench(fb,  a.iters)
    print(f"  fwd      : {fwd_ms:.4f} ms | {flop_fwd/(fwd_ms*1e-3)/1e12:7.2f} TFLOPS")
    print(f"  fwd+bwd  : {fb_ms:.4f} ms | {3*flop_fwd/(fb_ms*1e-3)/1e12:7.2f} TFLOPS (≈3× GEMM)")

if __name__=="__main__":
    main()
