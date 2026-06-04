#!/bin/bash
# B1 v2 verify + perf, run INSIDE sonic-moe-2605 on the GPU set by CUDA_VISIBLE_DEVICES.
cd /data1/min.yang/te_build
LOG=/tmp/b1v2.log
rm -f "$LOG"
echo "=== CORRECTNESS (G=8, real shape) ===" >> "$LOG"
NVTE_USE_FUSED_MOE=1 MOE_G=8 MOE_D=2048 MOE_I=512 MOE_ME=768 \
  python qa/te_fused_moe_e2e_test.py --correctness 2>&1 | grep -aE "n_fail|DROP-IN" >> "$LOG"
echo "=== KERNEL-DIRECT (B1 v2 vs forward) ===" >> "$LOG"
python qa/b1_kernel_direct_bench.py 2>&1 | grep -aE "CONFIG|FWD  kernel|B1   kernel|B1/FWD" >> "$LOG"
echo "=== 4-BACKEND FWD+BWD (graphsafe baseline + fused B1 v2) ===" >> "$LOG"
for MM in graphsafe fused; do
  MODE=$MM python qa/moe_4backends_fwdbwd.py 2>&1 | grep -aE "PERF4|^\[" >> "$LOG"
done
echo B1V2_ALL_DONE >> "$LOG"
