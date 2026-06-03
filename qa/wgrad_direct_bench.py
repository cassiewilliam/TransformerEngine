#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct ragged-K wgrad grouped-GEMM benchmark — times ONLY the kernel (no autograd / cast /
copy overhead), matching the H100 table methodology. Replicates GroupedLinear's wgrad call:
  general_grouped_gemm(inputmats[k_i,M], grad_output[k_i,N], wgrad[N,M], layout="NT", grad=True,
                       m_splits=k_i)
Shape(g,m,n,k[mink,avgk,maxk]); FLOP = 2*M*N*sum(k_i); TFLOPS = FLOP/time.
"""
import argparse, torch
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

def ramp_k(g, mink, maxk, mult=128):
    if g == 1:
        return [round((mink + maxk) / 2 / mult) * mult]
    return [max(mult, round((mink + (maxk - mink) * i / (g - 1)) / mult) * mult) for i in range(g)]

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
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--m", type=int, default=512)   # in_features
    ap.add_argument("--n", type=int, default=2048)  # out_features
    ap.add_argument("--mink", type=int, default=1024)
    ap.add_argument("--maxk", type=int, default=3072)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    import os
    dt = torch.bfloat16
    g, M, N = a.g, a.m, a.n
    ks = ramp_k(g, a.mink, a.maxk)
    totK = sum(ks)
    flop = 2 * M * N * totK
    dev = "cuda"
    A   = [torch.randn(k, M, device=dev, dtype=dt) for k in ks]     # inputmats [k, M]
    B   = [torch.randn(k, N, device=dev, dtype=dt) for k in ks]     # grad_output [k, N]
    out = [torch.empty(N, M, device=dev, dtype=dt) for _ in ks]     # dW [N, M]
    qp  = [None] * g
    def run():
        general_grouped_gemm(A, B, out, qp, dt, layout="NT", grad=True, m_splits=ks)
    print(f"GPU {torch.cuda.get_device_name(0)} | direct wgrad NT | g={g} M={M} N={N} "
          f"k[min={min(ks)},avg={totK//g},max={max(ks)}] sumK={totK} "
          f"| CUTLASS={os.environ.get('NVTE_USE_CUTLASS_GROUPED_GEMM','0')}")
    ms = bench(run, a.iters)
    print(f"  wgrad-kernel: {ms:.4f} ms | {flop/(ms*1e-3)/1e12:8.2f} TFLOPS")

if __name__ == "__main__":
    main()
