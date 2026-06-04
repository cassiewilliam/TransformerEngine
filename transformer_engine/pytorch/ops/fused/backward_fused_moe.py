# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Backward fused operation for BF16 MoE grouped MLP (CUTLASS SwiGLU forward).

This is the BF16 SonicMoE counterpart of ``backward_grouped_mlp.py`` (the MXFP8
cuTe-DSL reference). It is the backward of the
``ForwardFusedMoE_CutlassSwiGLU_BF16`` forward op.

IMPORTANT -- there is currently NO fused CUTLASS backward kernel. The SonicMoE
"dH-overlap B1" kernel that would fuse the FC1+SwiGLU backward is future work.
So this op implements a CORRECT, registrable backward by FALLING BACK to the
standard per-op grouped-GEMM backward:

    grad wrt FC2 in   = standard GroupedLinear bf16 backward of FC2
    grad wrt SwiGLU in = recompute the FC1 up-proj [M, 2I], then torch SwiGLU grad
    grad wrt FC1 in   = standard GroupedLinear bf16 backward of FC1

The forward never materialized the [M, 2I] gate||up tensor (the CUTLASS kernel
fuses it away), so the SwiGLU backward here RECOMPUTES that up-proj on demand
(a plain bf16 grouped GEMM) before applying ``dswiglu``. This trades a small
recompute for correctness; the fused dH kernel will replace it later.

Compared to the MXFP8 reference this file DROPS all FP8/MXFP8:
quantizers, swizzle, sfa/sfb/scale tensors, cuDNN-FE dactivation/quant/wgrad
kernels, alpha/norm-const. Everything is pure bf16 grouped GEMM.
"""

from __future__ import annotations
from collections.abc import Iterable
import functools
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm_for_grouped_tensor
from ...module.base import _2X_ACC_DGRAD, _2X_ACC_FPROP, _2X_ACC_WGRAD
from ...quantization import Recipe
from ...tensor import Quantizer
from ...tensor.grouped_tensor import GroupedTensor
from ...utils import clear_tensor_data, get_device_compute_capability
from ..basic import GroupedLinear
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    is_glu_activation,
    maybe_dequantize,
    validate_grouped_mlp_dims,
)

# Mirror the forward's tile constant (kept local to avoid a forward import).
_CUTLASS_TILE_M = 256


class BackwardFusedMoE_CutlassSwiGLU_BF16(FusedOperation):
    """Backward fused op for BF16 GroupedLinear + SwiGLU + GroupedLinear.

    Mirrors ``_BackwardGroupedMLP_CuTeGEMMDBase_MXFP8`` in
    ``backward_grouped_mlp.py`` but, because there is no fused CUTLASS backward
    kernel yet, FALLS BACK to per-op bf16 grouped-GEMM backward (see module
    docstring). Like the forward, no GLU-vs-SReLU split is needed (SwiGLU only).
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused backward op is supported on the current system.

        Mirrors the reference ``is_supported`` (backward_grouped_mlp.py:289-304)
        but swaps the env flag to ``NVTE_USE_FUSED_MOE`` and does NOT require any
        cuDNN-FE backward kernel (there is no fused backward kernel; we fall back
        to per-op bf16 grouped GEMM, which is always available on SM100).
        """
        # bf16: SonicMoE gate flag, not NVTE_CUTEDSL_FUSED_GROUPED_MLP.
        if int(os.environ.get("NVTE_USE_FUSED_MOE", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        # No kernel probe: the fallback uses general_grouped_gemm_for_grouped_tensor
        # + tex.dswiglu, both core bf16 ops. Gate symmetrically with the forward
        # so the pair registers together.
        try:
            # Probe the forward binding so the backward only registers when the
            # forward op could have actually run (and thus saved ctx for us).
            _ = tex.te_cutlass_grouped_swiglu
        except AttributeError:
            return False
        return True

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        activation: Optional[FusibleOperation],
        fc2: GroupedLinear,
    ) -> None:
        if activation is None:
            raise TypeError("Expected a grouped MLP activation op.")
        super().__init__((fc1, activation, fc2))
        if not self.is_supported():
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        validate_grouped_mlp_dims(fc1, activation, fc2)
        # bf16 CUTLASS path is SwiGLU-only (matches the forward op).
        if not is_glu_activation(activation):
            raise TypeError(
                f"{self.__class__.__name__} requires a SwiGLU activation, "
                f"got {activation.__class__.__name__}."
            )
        from ..basic import ScaledClampedQGeGLU  # pylint: disable=import-outside-toplevel

        if isinstance(activation, ScaledClampedQGeGLU):
            raise TypeError(
                f"{self.__class__.__name__} does not support clamped GeGLU; "
                "only silu-gated SwiGLU."
            )

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: Optional[list[tuple[torch.Tensor, ...]]] = None,
        **unused,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        list[Iterable[Optional[torch.Tensor]]],
        list[Iterable[Optional[torch.Tensor]]],
    ]:
        # Get basic operations (ref backward_grouped_mlp.py:354-355)
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties (ref backward_grouped_mlp.py:358-364)
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)  # (2I, d)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)  # (d_out, I)
        d = fc1_weight_shape[1]
        two_i = fc1_weight_shape[0]
        I = fc2_weight_shape[1]  # noqa: E741
        grad_output = grad_output.reshape(-1, fc2_weight_shape[0])
        out_shape = list(grad_output.size())
        M = out_shape[0]
        num_groups = fc1_op.num_groups
        device = fc1_op._get_weight_tensors()[0].device
        dtype = fc1_ctx.dtype

        # ---- Saved tensors from forward (layout set by forward_fused_moe.py) ----
        # FC1 ctx layout: [split_sizes, base_split_offsets, split_points,
        #                  grouped_fc1_x, *fc1_weights]
        fc1_saved = fc1_ctx.saved_tensors
        split_sizes, base_split_offsets, split_points = fc1_saved[:3]
        grouped_fc1_x, fc1_saved = fc1_saved[3], fc1_saved[4:]
        if fc1_op.single_grouped_weight:
            fc1_weight = fc1_saved[0]
        else:
            fc1_weight = fc1_saved[:num_groups]

        # Activation ctx layout: (None, scales) -- the [M, 2I] SwiGLU input was
        # never materialized in forward (CUTLASS fused it away), so we recompute
        # it below. ``scales`` is the per-token router prob.
        _swiglu_in_unused, scales = activation_ctx.saved_tensors

        # FC2 ctx layout: [split_sizes, base_split_offsets, split_points,
        #                  grouped_fc2_x, *fc2_weights]; grouped_fc2_x == A.
        fc2_saved = fc2_ctx.saved_tensors
        grouped_fc2_x, fc2_saved = fc2_saved[3], fc2_saved[4:]
        if fc2_op.single_grouped_weight:
            fc2_weight = fc2_saved[0]
        else:
            fc2_weight = fc2_saved[:num_groups]

        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")

        # Common grouped-GEMM scaffolding (all bf16).
        dtype = dtype if dtype is not None else grad_output.dtype
        grad_output = maybe_dequantize(grad_output, dtype)

        # ======================================================================
        # Step 1: FC2 backward (down-proj).  out = A @ W2^T (TN in forward).
        #   dA      = grad_output @ W2          (layout "NN")
        #   dW2     = grad_output^T @ A         (layout "NT", per expert)
        # This is exactly GroupedLinear's bf16 grouped-tensor backward; we run
        # the GEMMs directly to keep the fallback self-contained and to obtain
        # dA (needed by the SwiGLU+FC1 backward).
        # ======================================================================
        grouped_fc2_weight = self._wrap_weight_grouped(
            fc2_op, fc2_weight, num_groups, fc2_weight_shape, dtype, device
        )
        grouped_dy2 = GroupedTensor(
            shape=(M, fc2_weight_shape[0]),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=grad_output.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * fc2_weight_shape[0],
        )

        # dA = dY2 @ W2  -> [M, I]
        dA = torch.empty(M, I, dtype=dtype, device=device)
        grouped_dA = GroupedTensor(
            shape=(M, I),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=dA.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * I,
        )
        if fc2_ctx.input_requires_grad or fc1_ctx.input_requires_grad or fc1_ctx.weight_requires_grad:
            general_grouped_gemm_for_grouped_tensor(
                grouped_fc2_weight,
                grouped_dy2,
                grouped_dA,
                layout="NN",
                use_split_accumulator=_2X_ACC_DGRAD,
            )

        # dW2 = dY2^T @ A   (NT, per expert), using saved activation output A.
        fc2_grad_params = self._compute_grouped_wgrad(
            fc_op=fc2_op,
            ctx=fc2_ctx,
            num_groups=num_groups,
            weight_shape=fc2_weight_shape,
            grouped_x=grouped_fc2_x,
            grouped_dy=grouped_dy2,
            dtype=dtype,
            device=device,
        )

        # ======================================================================
        # Step 2: SwiGLU backward.  Recompute the up-proj [M, 2I] then dswiglu.
        #   h        = x @ W1^T            (the gate||up the forward fused away)
        #   dY1      = dswiglu(dA * prob, h)   -> grad wrt h, shape [M, 2I]
        #   dprob    = <swiglu(h), dA>     (router-prob grad, if required)
        # bf16: this recompute is the fallback for the missing fused dH kernel.
        # TODO(sonic-moe): replace recompute+dswiglu with the fused B1 kernel.
        # ======================================================================
        # Recompute h = x @ W1^T using the saved FC1 input + weight.
        grouped_fc1_weight = self._wrap_weight_grouped(
            fc1_op, fc1_weight, num_groups, fc1_weight_shape, dtype, device
        )
        x = self._grouped_x_data(grouped_fc1_x, M, d, dtype, device)
        h = torch.empty(M, two_i, dtype=dtype, device=device)
        grouped_x = GroupedTensor(
            shape=(M, d),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=x.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * d,
        )
        grouped_h = GroupedTensor(
            shape=(M, two_i),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=h.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * two_i,
        )
        general_grouped_gemm_for_grouped_tensor(
            grouped_fc1_weight,
            grouped_x,
            grouped_h,
            layout="TN",
            use_split_accumulator=_2X_ACC_FPROP,
        )

        # Apply router-prob (the forward applied A *= prob inside the kernel; the
        # grad wrt the SwiGLU output is therefore dA * prob).
        prob = None
        grad_swiglu_out = dA
        if scales is not None:
            prob = maybe_dequantize(scales, dtype).reshape(-1)
            grad_swiglu_out = dA * prob.unsqueeze(-1)

        # dY1 = dswiglu(grad_swiglu_out, h) -> [M, 2I].  tex.dswiglu matches the
        # silu(gate)*up convention used by the CUTLASS kernel and ScaledSwiGLU
        # (swiglu.py:568). No GLU interleaving: the forward kernel consumes the
        # plain gate||up stacked weight, so the activation runs un-interleaved.
        # VERIFY: confirm dswiglu gate/up halves match the CUTLASS kernel
        # (gate = first half, up = second half) on B200.
        dY1 = tex.dswiglu(grad_swiglu_out, h, None)
        dY1 = maybe_dequantize(dY1, dtype).reshape(M, two_i)

        # Router-prob gradient: dprob[m] = <swiglu(h)[m,:], dA[m,:]>.
        grad_scales = None
        if scales is not None and activation_ctx.extra_input_requires_grad:
            swiglu_out = tex.swiglu(h, None)
            swiglu_out = maybe_dequantize(swiglu_out, dtype).reshape(M, I)
            grad_scales = torch.linalg.vecdot(swiglu_out, dA).to(dtype=dtype)

        # ======================================================================
        # Step 3: FC1 backward (up-proj).  out h = x @ W1^T (TN in forward).
        #   dX      = dY1 @ W1            (layout "NN")  -> grad wrt MoE input
        #   dW1     = dY1^T @ x          (layout "NT", per expert)
        # ======================================================================
        grouped_dy1 = GroupedTensor(
            shape=(M, two_i),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=dY1.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * two_i,
        )

        grad_input = None
        if fc1_ctx.input_requires_grad:
            grad_input = torch.empty(M, d, dtype=dtype, device=device)
            grouped_grad_input = GroupedTensor(
                shape=(M, d),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=grad_input.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets * d,
            )
            general_grouped_gemm_for_grouped_tensor(
                grouped_fc1_weight,
                grouped_dy1,
                grouped_grad_input,
                layout="NN",
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            grad_input = grad_input.view(out_shape[:-1] + [d])

        fc1_grad_params = self._compute_grouped_wgrad(
            fc_op=fc1_op,
            ctx=fc1_ctx,
            num_groups=num_groups,
            weight_shape=fc1_weight_shape,
            grouped_x=grouped_x,
            grouped_dy=grouped_dy1,
            dtype=dtype,
            device=device,
        )

        # Clear saved activation buffers if possible (ref backward_grouped_mlp.py:662-673).
        if grouped_fc2_x is not None and not (
            fc2_ctx.weight_requires_grad
            and fc2_op.wgrad_store is not None
            and fc2_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(grouped_fc2_x.rowwise_data)
        if grouped_fc1_x is not None and not (
            fc1_ctx.weight_requires_grad
            and fc1_op.wgrad_store is not None
            and fc1_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(grouped_fc1_x.rowwise_data)

        # ---- Assemble return (ref backward_grouped_mlp.py:772-778) ----
        # Per-op grad_params: [fc1_params, (), fc2_params].
        # Per-op grad_extra_inputs: FC1 GroupedLinear has 1 extra input
        # (split_sizes) -> (None,); ScaledSwiGLU has 1 extra input (scales) ->
        # (grad_scales,); FC2 GroupedLinear -> (None,).
        activation_grad_extra: tuple = (grad_scales,) if scales is not None else ()
        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [(None,), activation_grad_extra, (None,)],
        )

    # ----------------------------------------------------------------------
    # Helpers (bf16-only).
    # ----------------------------------------------------------------------
    @staticmethod
    def _wrap_weight_grouped(
        fc_op: GroupedLinear,
        weight,
        num_groups: int,
        weight_shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> GroupedTensor:
        """Return a uniform bf16 GroupedTensor [G*out, in] for grouped GEMM.

        Accepts the saved weight as either a single GroupedTensor (when
        single_grouped_weight) or a list/tuple of per-expert tensors, mirroring
        ``GroupedLinear._get_grouped_weight_for_gemm`` for the bf16 case
        (grouped_linear.py:791-804).
        """
        out_features, in_features = weight_shape
        if fc_op.single_grouped_weight:
            if isinstance(weight, GroupedTensor):
                w = maybe_dequantize(weight.rowwise_data, dtype)
            else:
                w = maybe_dequantize(weight, dtype)
            weight_data = w.reshape(-1)
        else:
            weights = [maybe_dequantize(w, dtype) for w in weight]
            weight_data = torch.stack(weights, dim=0).contiguous().reshape(-1)
        return GroupedTensor(
            shape=(num_groups * out_features, in_features),
            dtype=dtype,
            num_tensors=num_groups,
            shapes=[(out_features, in_features)] * num_groups,
            quantizer=None,
            data=weight_data,
        )

    @staticmethod
    def _grouped_x_data(
        grouped_x: Optional[GroupedTensor],
        M: int,
        in_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the [M, in_features] bf16 activation buffer from a GroupedTensor.

        Raises if the saved input was cleared (weight_requires_grad=False), since
        the SwiGLU recompute needs the FC1 input.
        """
        if grouped_x is None or grouped_x.rowwise_data is None:
            raise RuntimeError(
                "BackwardFusedMoE: FC1 input was not saved (weight_requires_grad=False), "
                "but the SwiGLU-input recompute requires it. This first version requires "
                "weight grads; the future fused dH kernel will avoid the recompute."
            )
        return maybe_dequantize(grouped_x.rowwise_data, dtype).reshape(M, in_features)

    @staticmethod
    def _compute_grouped_wgrad(
        *,
        fc_op: GroupedLinear,
        ctx: OperationContext,
        num_groups: int,
        weight_shape: tuple[int, int],
        grouped_x: Optional[GroupedTensor],
        grouped_dy: GroupedTensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Optional[torch.Tensor]]:
        """Compute bf16 wgrad and return grad_params in registration order.

        Simplified bf16 analogue of ``backward_grouped_mlp._compute_grad_params``
        (backward_grouped_mlp.py:138). Drops the cuDNN wgrad kernel, MXFP8
        packing, and bias paths (the SonicMoE grouped MLP has no bias). Supports
        accumulate_into_main_grad (Megatron-LM wgrad fusion) and delayed wgrad
        via ``wgrad_store``.
        """
        from .._common import (  # pylint: disable=import-outside-toplevel
            get_accumulate_flag_in_param,
            get_dummy_wgrads_for_params,
            get_main_grad_from_param,
            view_main_grad_as_grouped_buffer,
        )

        out_features, in_features = weight_shape
        weights = fc_op._get_weight_tensors()
        single = fc_op.single_grouped_weight

        if not ctx.weight_requires_grad:
            # No wgrad requested -> return Nones (no bias in SonicMoE MLP).
            return [None] if single else [None] * num_groups

        accumulate = False
        final_weight_grads: list[Optional[torch.Tensor]] = [None] if single else [None] * num_groups
        grouped_wgrad: Optional[GroupedTensor] = None
        wgrad_output: Any = None
        grouped_shape = (num_groups, out_features, in_features)

        if single:
            if fc_op._accumulate_into_main_grad:
                main_grad = get_main_grad_from_param(weights[0], op_label="Fused MoE backward")
                main_grad = view_main_grad_as_grouped_buffer(
                    main_grad, num_groups, weight_shape, label="Fused MoE weight"
                )
                grouped_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                    num_tensors=num_groups,
                    tensor_shape=weight_shape,
                    rowwise_data=main_grad.view(-1),
                    dtype=main_grad.dtype,
                )
                accumulate = get_accumulate_flag_in_param(weights[0])
            else:
                grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                    num_tensors=num_groups,
                    shapes=[weight_shape] * num_groups,
                    quantizer=None,
                    device=device,
                    dtype=dtype,
                )
            final_weight_grads[0] = grouped_wgrad.rowwise_data.view(num_groups, *weight_shape)
            wgrad_output = grouped_wgrad
        else:
            if fc_op._accumulate_into_main_grad:
                final_weight_grads = [
                    get_main_grad_from_param(w, op_label="Fused MoE backward") for w in weights
                ]
                accumulate = get_accumulate_flag_in_param(weights[0])
            else:
                final_weight_grads = [
                    torch.empty(weight_shape, dtype=dtype, device=device)
                    for _ in range(num_groups)
                ]
            wgrad_output = final_weight_grads

        # wgrad GEMM: dW = dY^T @ X  (layout "NT", per expert).
        delay_wgrad = fc_op.wgrad_store is not None and fc_op.wgrad_store.delay_wgrad_compute()
        wgrad_gemm = functools.partial(
            general_grouped_gemm_for_grouped_tensor,
            layout="NT",
            accumulate=accumulate,
            use_split_accumulator=_2X_ACC_WGRAD,
        )
        if delay_wgrad:
            fc_op.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], wgrad_gemm)
        else:
            wgrad_gemm(grouped_x, grouped_dy, wgrad_output)

        # Megatron-LM wgrad fusion bookkeeping (ref backward_grouped_mlp.py:237-240).
        if fc_op._accumulate_into_main_grad:
            final_weight_grads = get_dummy_wgrads_for_params(weights)
        elif delay_wgrad:
            final_weight_grads = [None] if single else [None] * num_groups

        return list(final_weight_grads)


def fuse_backward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply BF16 GroupedLinear + SwiGLU + GroupedLinear fusion for backward pass.

    Mirrors ``backward_grouped_mlp.fuse_backward_ops`` (backward_grouped_mlp.py:811)
    but substitutes the CUTLASS BF16 backward op and uses the bf16 matcher (the
    shared ``fuse_grouped_mlp_ops`` requires an mxfp8 recipe; this path is bf16).
    """
    return _fuse_backward_ops_bf16(ops, fused_op_cls=BackwardFusedMoE_CutlassSwiGLU_BF16)


def _fuse_backward_ops_bf16(
    ops: list[FusibleOperation],
    *,
    fused_op_cls,
) -> list[FusibleOperation]:
    """Sliding-window [GroupedLinear, ScaledSwiGLU, GroupedLinear] matcher (bf16).

    Structurally identical to the forward bf16 matcher
    (forward_fused_moe._fuse_forward_ops_bf16); kept in sync so the forward and
    backward fusions substitute the same triples.
    """
    from ..basic import ScaledSwiGLU  # pylint: disable=import-outside-toplevel

    if not fused_op_cls.is_supported():
        return ops

    out: list[FusibleOperation] = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:
        matches_pattern = (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], ScaledSwiGLU)
            and isinstance(window[2], GroupedLinear)
        )
        if matches_pattern:
            try:
                validate_grouped_mlp_dims(window[0], window[1], window[2])
            except (TypeError, ValueError):
                matches_pattern = False

        if matches_pattern:
            op = fused_op_cls(fc1=window[0], activation=window[1], fc2=window[2])
            window = [op]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out


# Register fusion if available (ref backward_grouped_mlp.py:856-858)
if BackwardFusedMoE_CutlassSwiGLU_BF16.is_supported():
    register_backward_fusion(fuse_backward_ops, prepend=True)
