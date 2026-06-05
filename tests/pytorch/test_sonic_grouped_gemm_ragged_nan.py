# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression test for the SonicMoE on-device CUTLASS grouped-GEMM NaN.

Root cause (fixed in cutlass_grouped_gemm.cuh / cublaslt_grouped_gemm.cu):
``CutlassGroupedGemmDevice`` filled the persistent scheduler's per-expert
``problem_sizes_host`` with the AVERAGE (avg_m, avg_n, avg_k) for every expert.
For ragged / non-256-aligned per-expert M (real MoE routing), the average
under-counts ``sum(ceil(M_e/tileM) * ceil(N_e/tileN))`` so some partial output
tiles are never scheduled -> the output is left uninitialized -> NaN.

Uniform per-expert M (e.g. Me=768 for every expert) made avg == exact and hid
the bug. This test deliberately drives RAGGED, non-256-aligned per-expert token
counts through the SONIC on-device CUTLASS path (NVTE_USE_FUSED_MOE=1)
via ``general_grouped_gemm_for_grouped_tensor`` and asserts:

  1. the CUTLASS output is finite (no NaN / Inf), and
  2. it matches the cuBLAS reference (NVTE_USE_FUSED_MOE=0).

WITHOUT the fix this fails (NaN / mismatch on the un-scheduled tiles); WITH the
fix it passes.

Run:
  cd /Users/min.yang/workcode/sonic-moe/transformerengine/tests/pytorch
  NVTE_USE_FUSED_MOE=0 pytest -sv test_sonic_grouped_gemm_ragged_nan.py
"""

import os
from typing import List

import pytest
import torch

from transformer_engine.pytorch import is_bf16_available
from transformer_engine.pytorch.cpp_extensions import (
    general_grouped_gemm_for_grouped_tensor,
)
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
import transformer_engine_torch as tex


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/pytorch/test_grouped_linear.py)
# ---------------------------------------------------------------------------
def _pack_grouped_tensor(grouped_tensor: GroupedTensor, tensors: List[torch.Tensor]) -> None:
    data = grouped_tensor.rowwise_data
    if data is None:
        data = grouped_tensor.columnwise_data
    if data is None:
        raise ValueError("GroupedTensor has no data buffers to pack.")
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        data[offset : offset + numel].copy_(tensor.reshape(-1))
        offset += numel


def _make_grouped_tensor_from_splits(
    m_sizes: List[int],
    last_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GroupedTensor:
    first_dims = torch.tensor(m_sizes, device=device, dtype=torch.int64)
    return GroupedTensor.make_grouped_tensor(
        num_tensors=len(m_sizes),
        first_dims=first_dims,
        last_dims=None,
        logical_first_dim=sum(m_sizes),
        logical_last_dim=last_dim,
        quantizer=None,
        device=device,
        dtype=dtype,
    )


def _make_grouped_tensor_uniform(
    num_tensors: int,
    first_dim: int,
    last_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GroupedTensor:
    return GroupedTensor.make_grouped_tensor(
        num_tensors=num_tensors,
        first_dims=None,
        last_dims=None,
        logical_first_dim=num_tensors * first_dim,
        logical_last_dim=last_dim,
        quantizer=None,
        device=device,
        dtype=dtype,
    )


def _run_sonic_cutlass(enable: bool):
    """Set/restore NVTE_USE_FUSED_MOE around a code block."""
    prev = os.environ.get("NVTE_USE_FUSED_MOE")
    os.environ["NVTE_USE_FUSED_MOE"] = "1" if enable else "0"

    def _restore():
        if prev is None:
            os.environ.pop("NVTE_USE_FUSED_MOE", None)
        else:
            os.environ["NVTE_USE_FUSED_MOE"] = prev

    return _restore


# RAGGED, NON-256-ALIGNED per-expert token counts (M). This is the exact shape
# class that real MoE routing produces and that the buggy AVERAGE-based tile
# estimate under-counts. Uniform Me would hide the bug (avg == exact).
RAGGED_M_SIZES = [
    [800, 736, 900, 512, 640, 333, 1024, 1100],  # z=8, sum=6045, none % 256 == 0
    [129, 257, 513, 65],  # z=4, tiny + non-aligned
    [1700, 200, 950, 77, 640, 480],  # z=6, very skewed
]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for grouped GEMM.",
)
@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and (
            torch.cuda.get_device_capability() == (9, 0)
            or torch.cuda.get_device_capability()[0] == 10
        )
    ),
    reason="SONIC CUTLASS grouped GEMM is for Hopper (SM90) / Blackwell (SM100/SM103).",
)
@pytest.mark.parametrize("m_sizes", RAGGED_M_SIZES, ids=lambda v: "z%d" % len(v))
# TN = forward FC1 (out = x @ Wt), NN = dgrad (dx = dY @ W) -- the SONIC-gated,
# uniform-K cases where M is the ragged token dim. (NT/wgrad has ragged K and is
# not routed through the device-ptr SONIC path that had the bug.)
@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("n, k", [(2048, 768), (768, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sonic_cutlass_ragged_no_nan(m_sizes, layout, n, k, dtype):
    """SONIC on-device CUTLASS grouped GEMM with ragged M must be finite and
    match the cuBLAS reference."""
    if dtype == torch.bfloat16 and not is_bf16_available():
        pytest.skip("bfloat16 is required.")
    if tex.get_cublasLt_version() < 130300:
        pytest.skip("Grouped GEMM requires cuBLAS 13.3+.")
    if (
        torch.cuda.get_device_capability() < (10, 0)
        and tex.get_cublasLt_version() < 130400
    ):
        pytest.skip("Grouped GEMM on Hopper requires cuBLAS 13.4+.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    z = len(m_sizes)
    m = sum(m_sizes)

    # Sanity: this MUST be ragged + non-256-aligned, otherwise the test does not
    # exercise the bug (uniform/aligned M makes avg == exact tile count).
    assert len(set(m_sizes)) > 1, "m_sizes must be ragged to reproduce the bug."
    assert any(ms % 256 != 0 for ms in m_sizes), "m_sizes must be non-256-aligned."

    # Weights are uniform per-expert (n, k); input/grad_output is the ragged
    # grouped operand. Layout dims follow test_grouped_gemm_grouped_tensor:
    #   TN: A=weight[n,k], B=input[m,k]      -> out[m,n]
    #   NN: A=weight[n,k], B=grad_output[m,n] -> out[m,k]
    A = [torch.randn(n, k, dtype=dtype, device=device) for _ in range(z)]
    b_last = k if layout == "TN" else n
    out_last = n if layout == "TN" else k
    B = [torch.randn(ms, b_last, dtype=dtype, device=device) for ms in m_sizes]

    if layout == "TN":  # out = B @ A^T
        out_ref_fp32 = [torch.matmul(B[i].float(), A[i].t().float()) for i in range(z)]
    else:  # NN: out = B @ A
        out_ref_fp32 = [torch.matmul(B[i].float(), A[i].float()) for i in range(z)]

    def _build_inputs():
        gA = _make_grouped_tensor_uniform(z, n, k, device, dtype)
        _pack_grouped_tensor(gA, A)
        gB = _make_grouped_tensor_from_splits(m_sizes, b_last, device, dtype)
        _pack_grouped_tensor(gB, B)
        # Pre-fill output with NaN so an un-scheduled (un-written) tile stays NaN
        # -> the bug surfaces deterministically instead of relying on stale mem.
        gOut = _make_grouped_tensor_from_splits(m_sizes, out_last, device, dtype)
        nan_fill = gOut.rowwise_data
        if nan_fill is None:
            nan_fill = gOut.columnwise_data
        nan_fill.fill_(float("nan"))
        return gA, gB, gOut

    # --- cuBLAS reference (NVTE_USE_FUSED_MOE=0) -----------------------
    restore = _run_sonic_cutlass(False)
    try:
        gA, gB, gOut = _build_inputs()
        general_grouped_gemm_for_grouped_tensor(
            gA, gB, gOut, layout=layout, accumulate=False, bias=None
        )
        torch.cuda.synchronize()
        cublas_out = [t.float() for t in gOut.split_into_quantized_tensors()]
    finally:
        restore()

    # --- SONIC on-device CUTLASS (NVTE_USE_FUSED_MOE=1) ----------------
    restore = _run_sonic_cutlass(True)
    try:
        gA, gB, gOut = _build_inputs()
        general_grouped_gemm_for_grouped_tensor(
            gA, gB, gOut, layout=layout, accumulate=False, bias=None
        )
        torch.cuda.synchronize()
        cutlass_out = [t.float() for t in gOut.split_into_quantized_tensors()]
    finally:
        restore()

    # 1) No NaN / Inf in the CUTLASS output -- this is the direct repro: with the
    #    avg-based estimate, partial tiles of ragged experts are never written and
    #    stay at the NaN pre-fill.
    for i, o in enumerate(cutlass_out):
        assert torch.isfinite(o).all(), (
            f"SONIC CUTLASS produced non-finite output for expert {i} "
            f"(layout={layout}, m_sizes={m_sizes}, n={n}, k={k}, dtype={dtype}): "
            f"{(~torch.isfinite(o)).sum().item()} bad elements -- "
            f"avg-based tile estimate left tiles unscheduled."
        )

    # 2) Matches cuBLAS reference (and the fp32 matmul) within bf16/fp16 tol.
    tol = dict(rtol=1.6e-2, atol=1e-2) if dtype == torch.bfloat16 else dict(rtol=1e-3, atol=1e-3)
    for i in range(z):
        torch.testing.assert_close(cutlass_out[i], cublas_out[i], **tol)
        torch.testing.assert_close(cutlass_out[i], out_ref_fp32[i], **tol)
