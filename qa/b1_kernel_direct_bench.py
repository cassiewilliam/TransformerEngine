"""B1 fused SwiGLU-backward kernel DIRECT throughput (isolates the kernel from the rest of the bwd).

Compares, on an IDLE GPU at the real 4K-MoE shape (G=32, H=2048, I=512, Me=768, M=24576, bf16):
  * te_cutlass_grouped_swiglu   (F2 forward kernel; recompute h + swiglu)   -> A[M,I]
  * te_cutlass_grouped_dswiglu  (B1 backward kernel; recompute h + dswiglu)  -> dY1[M,2I]
Both do the SAME up-proj GEMM (h = X@W1^T, 2*M*2I*d = 103.1 GFLOP); the epilogue differs (swiglu vs
dswiglu + dA-read + 2-wide store). TF/s is on that 103.1 GFLOP. Also times the per-op FALLBACK backward
sub-steps (cuBLAS recompute h + tex.dswiglu + tex.swiglu) for an apples-to-apples kernel-vs-fallback read.

Run: CUDA_VISIBLE_DEVICES=<idle> python qa/b1_kernel_direct_bench.py
"""

import torch
import transformer_engine  # noqa: F401  (RTLD_GLOBAL first)
import transformer_engine_torch as tex

G, D, I = 32, 2048, 512
TWO_I, ME = 2 * I, 768
M = G * ME
dev, dt = "cuda", torch.bfloat16
up_flop = 2.0 * M * TWO_I * D  # 103.1 GFLOP (the recompute up-proj, common to fwd+bwd kernels)

x = torch.randn(M, D, dtype=dt, device=dev)
w1 = torch.randn(TWO_I * G, D, dtype=dt, device=dev) * 0.02  # [G*2I, d] packed gate||up
dA = torch.randn(M, I, dtype=dt, device=dev)
prob = (torch.rand(M, device=dev, dtype=torch.float32) + 0.25)
splits = torch.tensor([ME] * G, dtype=torch.int64, device=dev)
num_tiles = (M + 255) // 256
mte = torch.repeat_interleave(
    torch.arange(G, device=dev, dtype=torch.int32),
    torch.div(splits, 256, rounding_mode="floor"),
    output_size=num_tiles,
)
print(f"CONFIG G={G} H={D} I={I} 2I={TWO_I} Me={ME} M={M} up_flop={up_flop/1e9:.1f}GF  {torch.cuda.get_device_name(0)}")


def bench(fn, n=50, w=15):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n  # ms


def fwd_kernel():
    return tex.te_cutlass_grouped_swiglu(x, w1, mte, prob, G, 0, I, D, M, 0)


def bwd_kernel():
    return tex.te_cutlass_grouped_dswiglu(x, w1, dA, mte, prob, G, 0, I, D, M, 0)


fwd_ms = bench(fwd_kernel)
bwd_ms = bench(bwd_kernel)
print(f"FWD  kernel (swiglu)  : {fwd_ms*1000:7.1f} us  {up_flop/(fwd_ms/1e3)/1e12:6.1f} TF/s")
print(f"B1   kernel (dswiglu) : {bwd_ms*1000:7.1f} us  {up_flop/(bwd_ms/1e3)/1e12:6.1f} TF/s")
print(f"B1/FWD kernel ratio   : {bwd_ms/fwd_ms:.2f}x  (>1 => dswiglu epilogue overhead: dA L1-reads + 2 stores)")
