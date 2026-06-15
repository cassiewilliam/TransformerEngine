# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for BF16 fused MoE single-grouped-weight helpers."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from transformer_engine.pytorch.ops import GroupedLinear
from transformer_engine.pytorch.ops.fused.backward_fused_moe import (
    BackwardFusedMoE_CutlassSwiGLU_BF16,
)
from transformer_engine.pytorch.ops.fused.forward_fused_moe import (
    ForwardFusedMoE_CutlassSwiGLU_BF16,
)
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


def _run_grouped_mlp_reference_with_helper_weights(
    fc1: GroupedLinear,
    fc2: GroupedLinear,
    *,
    num_groups: int,
    hidden_size: int,
    ffn_hidden_size: int,
    dtype: torch.dtype,
    x: torch.Tensor,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Run a grouped SwiGLU MLP using the fused-MoE helper-produced weight views."""
    w1 = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack(
        None,
        fc1,
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
    )
    w2 = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
    )
    assert isinstance(w1, torch.Tensor)
    assert isinstance(w2, torch.Tensor)

    h = torch.einsum("gmd,god->gmo", x, w1.view(num_groups, 2 * ffn_hidden_size, hidden_size))
    gate, up = torch.chunk(h, 2, dim=-1)
    a = F.silu(gate) * up
    y = torch.einsum("gmi,ghi->gmh", a, w2.view(num_groups, hidden_size, ffn_hidden_size))
    loss = (y * dy).sum()

    params = [getattr(fc1, f"weight{idx}") for idx in range(num_groups)] + [
        getattr(fc2, f"weight{idx}") for idx in range(num_groups)
    ]
    grads = torch.autograd.grad(loss, params)
    fc1_grads = list(grads[:num_groups])
    fc2_grads = list(grads[num_groups:])
    return y.detach(), fc1_grads, fc2_grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_fused_moe_sgw_weight_helpers_return_plain_views(monkeypatch) -> None:
    """SGW helpers unwrap GroupedTensor before reshaping weights."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    monkeypatch.setenv("NVTE_USE_BF16_FUSED_MOE_SGW", "1")

    num_groups = 4
    hidden_size = 2048
    ffn_hidden_size = 512
    dtype = torch.bfloat16

    fc1 = GroupedLinear(
        num_groups,
        hidden_size,
        2 * ffn_hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=True,
    )
    fc2 = GroupedLinear(
        num_groups,
        ffn_hidden_size,
        hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=True,
    )

    assert fc1.single_grouped_weight
    assert fc2.single_grouped_weight
    assert isinstance(fc1.weight, GroupedTensor)
    assert isinstance(fc2.weight, GroupedTensor)

    # This is the historical failure mode fixed by _plain_grouped_tensor_data:
    # GroupedTensor only permits view(-1), not structured reshapes.
    with pytest.raises(RuntimeError, match="GroupedTensor only supports view"):
        fc1.weight.view(num_groups, 2 * ffn_hidden_size, hidden_size)

    fc1_forward = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack(
        None,
        fc1,
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
    )
    fc2_forward = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
    )
    fc1_backward = BackwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack_from_saved(
        fc1,
        fc1.weight,
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
        transpose_for_b=False,
    )
    fc2_backward = BackwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
        transpose_for_b=False,
    )

    assert fc1_forward.shape == (num_groups * 2 * ffn_hidden_size, hidden_size)
    assert fc2_forward.shape == (num_groups * hidden_size, ffn_hidden_size)
    assert fc1_backward.shape == fc1_forward.shape
    assert fc2_backward.shape == fc2_forward.shape

    assert not isinstance(fc1_forward, GroupedTensor)
    assert not isinstance(fc2_forward, GroupedTensor)
    assert fc1_forward.data_ptr() == fc1.weight.rowwise_data.data_ptr()
    assert fc2_forward.data_ptr() == fc2.weight.rowwise_data.data_ptr()
    assert fc1_backward.data_ptr() == fc1.weight.rowwise_data.data_ptr()
    assert fc2_backward.data_ptr() == fc2.weight.rowwise_data.data_ptr()


def test_bf16_fused_moe_quack_calls_use_dynamic_scheduler() -> None:
    """Keep QuACK fused MoE calls on the dynamic scheduler path."""
    fused_dir = Path(__file__).parents[2] / "transformer_engine" / "pytorch" / "ops" / "fused"
    fused_sources = [
        fused_dir / "forward_fused_moe.py",
        fused_dir / "backward_fused_moe.py",
    ]

    for source in fused_sources:
        text = source.read_text(encoding="utf-8")
        assert "dynamic_scheduler=False" not in text, f"{source} still disables dynamic scheduler"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_fused_moe_sgw_rebuilds_shared_storage_without_stack(monkeypatch) -> None:
    """Discrete expert params can still take the dense grouped path when storage is shared."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    monkeypatch.setenv("NVTE_USE_BF16_FUSED_MOE_SGW", "1")

    num_groups = 4
    hidden_size = 128
    ffn_hidden_size = 64
    dtype = torch.bfloat16

    fc1 = GroupedLinear(
        num_groups,
        hidden_size,
        2 * ffn_hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )
    fc2 = GroupedLinear(
        num_groups,
        ffn_hidden_size,
        hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )

    fc1_base = torch.randn(
        num_groups, 2 * ffn_hidden_size, hidden_size, device="cuda", dtype=dtype
    )
    fc2_base = torch.randn(
        num_groups, hidden_size, ffn_hidden_size, device="cuda", dtype=dtype
    )
    for idx in range(num_groups):
        setattr(fc1, f"weight{idx}", nn.Parameter(fc1_base[idx]))
        setattr(fc2, f"weight{idx}", nn.Parameter(fc2_base[idx]))

    fc1_forward = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack(
        None,
        fc1,
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
    )
    fc2_forward = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
    )
    fc1_backward = BackwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack_from_saved(
        fc1,
        [getattr(fc1, f"weight{idx}") for idx in range(num_groups)],
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
        transpose_for_b=True,
    )
    fc2_backward = BackwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
        transpose_for_b=False,
    )

    assert isinstance(fc1_forward, torch.Tensor)
    assert isinstance(fc2_forward, torch.Tensor)
    assert isinstance(fc1_backward, torch.Tensor)
    assert isinstance(fc2_backward, torch.Tensor)

    assert fc1_forward.data_ptr() == fc1_base.data_ptr()
    assert fc2_forward.data_ptr() == fc2_base.data_ptr()
    assert fc1_backward.data_ptr() == fc1_base.data_ptr()
    assert fc2_backward.data_ptr() == fc2_base.data_ptr()

    assert torch.equal(
        fc1_forward, fc1_base.view(num_groups * 2 * ffn_hidden_size, hidden_size)
    )
    assert torch.equal(
        fc2_forward, fc2_base.view(num_groups * hidden_size, ffn_hidden_size)
    )
    assert torch.equal(fc1_backward, fc1_forward)
    assert torch.equal(fc2_backward, fc2_forward)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_fused_moe_sgw_grouped_linear_registers_shared_storage(monkeypatch) -> None:
    """GroupedLinear discrete BF16 params are registered as packed slices under the macro."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    monkeypatch.setenv("NVTE_USE_BF16_FUSED_MOE_SGW", "1")

    num_groups = 4
    hidden_size = 128
    ffn_hidden_size = 64
    dtype = torch.bfloat16

    fc1 = GroupedLinear(
        num_groups,
        hidden_size,
        2 * ffn_hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )

    weights = [getattr(fc1, f"weight{idx}") for idx in range(num_groups)]
    first = weights[0]
    expert_elems = (2 * ffn_hidden_size) * hidden_size
    storage_ptr = first.untyped_storage().data_ptr()
    base_offset = first.storage_offset()

    for idx, weight in enumerate(weights):
        assert weight.untyped_storage().data_ptr() == storage_ptr
        assert weight.storage_offset() == base_offset + idx * expert_elems
        assert weight.shape == (2 * ffn_hidden_size, hidden_size)
        assert weight.stride() == (hidden_size, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_fused_moe_sgw_stacks_when_storage_is_not_packed(monkeypatch) -> None:
    """Macro path stacks expert weights when it cannot rebuild one packed dense view."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    monkeypatch.setenv("NVTE_USE_BF16_FUSED_MOE_SGW", "1")

    num_groups = 4
    hidden_size = 128
    ffn_hidden_size = 64
    dtype = torch.bfloat16

    fc1 = GroupedLinear(
        num_groups,
        hidden_size,
        2 * ffn_hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )
    fc2 = GroupedLinear(
        num_groups,
        ffn_hidden_size,
        hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )

    for idx in range(num_groups):
        fc1_weight = torch.empty_like(getattr(fc1, f"weight{idx}"))
        fc1_weight.copy_(getattr(fc1, f"weight{idx}"))
        setattr(
            fc1,
            f"weight{idx}",
            nn.Parameter(fc1_weight),
        )
        fc2_weight = torch.empty_like(getattr(fc2, f"weight{idx}"))
        fc2_weight.copy_(getattr(fc2, f"weight{idx}"))
        setattr(
            fc2,
            f"weight{idx}",
            nn.Parameter(fc2_weight),
        )

    fc1_ptrs = [getattr(fc1, f"weight{idx}").untyped_storage().data_ptr() for idx in range(num_groups)]
    fc2_ptrs = [getattr(fc2, f"weight{idx}").untyped_storage().data_ptr() for idx in range(num_groups)]
    assert len(set(fc1_ptrs)) == num_groups
    assert len(set(fc2_ptrs)) == num_groups

    fc1_forward = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack(
        None,
        fc1,
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
    )
    fc2_forward = ForwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
    )
    fc1_backward = BackwardFusedMoE_CutlassSwiGLU_BF16._get_fc1_weight_for_quack_from_saved(
        fc1,
        [getattr(fc1, f"weight{idx}") for idx in range(num_groups)],
        num_groups,
        (2 * ffn_hidden_size, hidden_size),
        dtype,
        transpose_for_b=True,
    )
    fc2_backward = BackwardFusedMoE_CutlassSwiGLU_BF16._get_fc2_weight_for_quack(
        None,
        fc2,
        num_groups,
        (hidden_size, ffn_hidden_size),
        dtype,
        transpose_for_b=False,
    )

    fc1_expected = torch.stack(
        [getattr(fc1, f"weight{idx}") for idx in range(num_groups)], dim=0
    ).view(num_groups * 2 * ffn_hidden_size, hidden_size).contiguous()
    fc2_expected = torch.stack(
        [getattr(fc2, f"weight{idx}") for idx in range(num_groups)], dim=0
    ).view(num_groups * hidden_size, ffn_hidden_size).contiguous()

    assert isinstance(fc1_forward, torch.Tensor)
    assert isinstance(fc2_forward, torch.Tensor)
    assert isinstance(fc1_backward, torch.Tensor)
    assert isinstance(fc2_backward, torch.Tensor)

    assert fc1_forward.is_contiguous()
    assert fc2_forward.is_contiguous()
    assert fc1_backward.is_contiguous()
    assert fc2_backward.is_contiguous()

    assert torch.equal(fc1_forward, fc1_expected)
    assert torch.equal(fc2_forward, fc2_expected)
    assert torch.equal(fc1_backward, fc1_expected)
    assert torch.equal(fc2_backward, fc2_expected)

    assert fc1_forward.untyped_storage().data_ptr() not in set(fc1_ptrs)
    assert fc2_forward.untyped_storage().data_ptr() not in set(fc2_ptrs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_fused_moe_sgw_matches_macro_off_numerics(monkeypatch) -> None:
    """Packed no-stack reconstruction matches the legacy stack path numerically."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")

    num_groups = 4
    hidden_size = 128
    ffn_hidden_size = 64
    tokens_per_group = 3
    dtype = torch.bfloat16

    fc1_off = GroupedLinear(
        num_groups,
        hidden_size,
        2 * ffn_hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )
    fc2_off = GroupedLinear(
        num_groups,
        ffn_hidden_size,
        hidden_size,
        bias=False,
        device="cuda",
        dtype=dtype,
        single_grouped_weight=False,
    )
    fc1_on = copy.deepcopy(fc1_off)
    fc2_on = copy.deepcopy(fc2_off)
    x = torch.randn(num_groups, tokens_per_group, hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn(num_groups, tokens_per_group, hidden_size, device="cuda", dtype=dtype)

    monkeypatch.setenv("NVTE_USE_BF16_FUSED_MOE_SGW", "0")
    out_off, fc1_grads_off, fc2_grads_off = _run_grouped_mlp_reference_with_helper_weights(
        fc1_off,
        fc2_off,
        num_groups=num_groups,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        dtype=dtype,
        x=x,
        dy=dy,
    )

    monkeypatch.setenv("NVTE_USE_BF16_FUSED_MOE_SGW", "1")
    out_on, fc1_grads_on, fc2_grads_on = _run_grouped_mlp_reference_with_helper_weights(
        fc1_on,
        fc2_on,
        num_groups=num_groups,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        dtype=dtype,
        x=x,
        dy=dy,
    )

    torch.testing.assert_close(out_on, out_off, atol=0, rtol=0)
    for grad_on, grad_off in zip(fc1_grads_on, fc1_grads_off):
        torch.testing.assert_close(grad_on, grad_off, atol=0, rtol=0)
    for grad_on, grad_off in zip(fc2_grads_on, fc2_grads_off):
        torch.testing.assert_close(grad_on, grad_off, atol=0, rtol=0)
