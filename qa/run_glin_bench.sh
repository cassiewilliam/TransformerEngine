#!/bin/bash
# Drive the STANDARD benchmarks/linear/benchmark_grouped_linear.py for the real 4K-MoE grouped GEMM,
# across the 3 grouped-GEMM backends, bf16, fwd-only. Shape per the user's table: H=2048, I=512, G=32.
#   up-proj:   M=24576 K(H)=2048  N(2I)=1024   FLOP=2*M*N*K=103.1G
#   down-proj: M=24576 K(I)=512   N(H)=2048    FLOP=2*M*N*K=51.5G
# Backends: legacy cuBLAS (multi-stream), CUTLASS F0 (NVTE_USE_CUTLASS_GROUPED_GEMM=1),
#           graph-safe cuBLAS 13.4 (NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1).
cd /data1/min.yang/te_build || exit 1
GPU=${1:-6}
J=$(python3 -c "print(','.join(['768']*32))")   # 32 experts x 768 tokens = M 24576

run() {  # $1=label $2=hidden(K) $3=out(N) ; rest = env assignments
  local label=$1 K=$2 N=$3; shift 3
  echo "----- $label  (K=$K N=$N) : $* -----"
  env "$@" CUDA_VISIBLE_DEVICES="$GPU" python benchmarks/linear/benchmark_grouped_linear.py \
    --recipe bf16 --hidden-dim "$K" --output-dim "$N" --jagged-input "$J" --fwd-only 2>&1 \
    | grep -aE "grouped_fwd_time_ms|^0 +[0-9]" | tail -2
}

echo "############ UP-PROJ  M=24576 K(H)=2048 N(2I)=1024 G=32 bf16 fwd-only (FLOP 103.1G) ############"
run "legacy_cuBLAS"    2048 1024 NVTE_USE_CUTLASS_GROUPED_GEMM=0 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0
run "CUTLASS_F0"       2048 1024 NVTE_USE_CUTLASS_GROUPED_GEMM=1 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0
run "graphsafe_cuBLAS" 2048 1024 NVTE_USE_CUTLASS_GROUPED_GEMM=0 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1
echo "############ DOWN-PROJ M=24576 K(I)=512 N(H)=2048 G=32 bf16 fwd-only (FLOP 51.5G) ############"
run "legacy_cuBLAS"    512 2048 NVTE_USE_CUTLASS_GROUPED_GEMM=0 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0
run "CUTLASS_F0"       512 2048 NVTE_USE_CUTLASS_GROUPED_GEMM=1 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0
run "graphsafe_cuBLAS" 512 2048 NVTE_USE_CUTLASS_GROUPED_GEMM=0 NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1
