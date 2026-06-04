#!/bin/bash
# Phase 0 (Design B validation): forward recomputes+SAVES h, backward READS it (no recompute).
# Pure-Python change (no rebuild). Runs INSIDE sonic-moe-2605 on the GPU set by CUDA_VISIBLE_DEVICES.
cd /data1/min.yang/te_build
LOG=/tmp/phase0.log
rm -f "$LOG"
echo "=== CORRECTNESS (Design B: backward reads saved h) ===" >> "$LOG"
NVTE_USE_FUSED_MOE=1 MOE_G=8 MOE_D=2048 MOE_I=512 MOE_ME=768 \
  python qa/te_fused_moe_e2e_test.py --correctness 2>&1 | grep -aE "n_fail|DROP-IN" >> "$LOG"
echo "=== 4-BACKEND FWD+BWD (graphsafe baseline vs fused=Design B) ===" >> "$LOG"
for MM in graphsafe fused; do
  MODE=$MM python qa/moe_4backends_fwdbwd.py 2>&1 | grep -aE "PERF4|^\[" >> "$LOG"
done
echo PHASE0_DONE >> "$LOG"
