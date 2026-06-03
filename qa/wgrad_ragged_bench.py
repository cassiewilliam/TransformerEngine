#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ragged-K grouped GEMM (the wgrad / varlen-K path) benchmark.
Shape(g, m, n, k[mink,avgk,maxk]): g groups, each D[m,n] = A[k_i,m]^T @ B[k_i,n], k_i ragged.
Maps to the wgrad of GroupedLinear(g, in=m, out=n) with tokens-per-expert = k_i.
Times ONLY the wgrad GEMM (input.requires_grad=False -> dgrad skipped; autograd.grad reuses
the forward graph). Reports ms / TFLOPS for cuBLAS baseline vs CUTLASS SM100, and speed-up.
  python wgrad_ragged_bench.py --g 32 --m 512 --n 2048 --mink 1024 --maxk 3072
"""
import argparse, torch
import transformer_engine.pytorch as te

def ramp_k(g, mink, maxk, mult=128):
    if g == 1:
        ks = [round((mink + maxk) / 2 / mult) * mult]
    else:
        ks = [round((mink + (maxk - mink) * i / (g - 1)) / mult) * mult for i in range(g)]
    return [max(mult, k) for k in ks]

def time_backward(m, x, ks, params, iters=50, warmup=20):
    # Re-run forward each iter (TE frees saved tensors after backward, so we can't reuse the
    # graph). Bracket ONLY the backward with CUDA events. x.requires_grad=False -> backward is
    # the wgrad GEMM only (no dgrad).
    for _ in range(warmup):
        y = m(x, ks); y.sum().backward()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    total = 0.0
    for _ in range(iters):
        for p in params: p.grad = None
        y = m(x, ks)
        torch.cuda.synchronize()
        s.record(); y.sum().backward(); e.record(); torch.cuda.synchronize()
        total += s.elapsed_time(e)
    return total / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--m", type=int, default=512)   # in_features
    ap.add_argument("--n", type=int, default=2048)  # out_features
    ap.add_argument("--mink", type=int, default=1024)
    ap.add_argument("--maxk", type=int, default=3072)
    ap.add_argument("--dtype", choices=["bf16"], default="bf16")  # varlen-K wgrad CUTLASS path is bf16
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    import os
    dt = torch.bfloat16
    g, M, N = a.g, a.m, a.n
    ks = ramp_k(g, a.mink, a.maxk)
    totK = sum(ks)
    flop = 2 * M * N * totK   # one ragged-K grouped GEMM (wgrad)
    print(f"GPU {torch.cuda.get_device_name(0)} | wgrad ragged-K | g={g} M={M} N={N} "
          f"k[min={min(ks)},avg={totK//g},max={max(ks)}] sumK={totK} "
          f"| CUTLASS={os.environ.get('NVTE_USE_CUTLASS_GROUPED_GEMM','0')}")
    m = te.GroupedLinear(g, M, N, bias=False, params_dtype=dt, device="cuda")
    x = torch.randn(totK, M, device="cuda", dtype=dt, requires_grad=False)  # no dgrad
    params = [p for p in m.parameters() if p.requires_grad]
    ms = time_backward(m, x, ks, params, a.iters)
    print(f"  wgrad: {ms:.4f} ms | {flop/(ms*1e-3)/1e12:8.2f} TFLOPS")

if __name__ == "__main__":
    main()
