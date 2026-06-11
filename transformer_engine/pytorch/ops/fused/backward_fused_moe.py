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
        but swaps the env flag to ``NVTE_USE_BF16_FUSED_MOE`` and does NOT require any
        cuDNN-FE backward kernel (there is no fused backward kernel; we fall back
        to per-op bf16 grouped GEMM, which is always available on SM100).
        """
        # bf16: SonicMoE gate flag, not NVTE_CUTEDSL_FUSED_GROUPED_MLP.
        # F group is gated by NVTE_USE_BF16_FUSED_MOE (single canonical flag).
        if int(os.environ.get("NVTE_USE_BF16_FUSED_MOE", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        # Probe the QuACK backend (gemm_dgated) so the backward only registers when the
        # QuACK pair the forward uses is available -- gates symmetrically with the forward.
        try:
            from quack.gemm_interface import gemm_dgated  # noqa: F401  pylint: disable=import-outside-toplevel
        except ImportError:
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

        # Activation ctx layout: (h_saved, scales). Design B: the forward SAVES the SwiGLU input
        # h=[gate||up] [M,2I], so the backward fuses dA=dY@W2 + dswiglu reading saved h -- NO recompute.
        # h_saved is None only on legacy/inference ctxs; then we fall back to recompute. ``scales`` is
        # the per-token router prob.
        swiglu_in_saved, scales = activation_ctx.saved_tensors

        # FC2 ctx layout: [split_sizes, base_split_offsets, split_points, None, *fc2_weights].
        # The fused backward does NOT read the ctx-saved FC2 weights: dW2's wgrad fetches them from
        # the live fc2_op (_compute_grouped_wgrad -> _get_weight_tensors / _get_fc2_weight_2d), and
        # the A' slot (index 3) is None (SonicMoE recomputes a_prime). So we skip reading them here.

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
        grouped_dy2 = GroupedTensor(
            shape=(M, fc2_weight_shape[0]),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=grad_output.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * fc2_weight_shape[0],
        )

        # dW2 = sum_t dO_t^T @ A'_{e,t} is computed in Step 2 below, AFTER the gemm_dgated recomputes
        # A' as `a_prime` from cached H (SonicMoE design -- no cached A'). grouped_dy2 (= dO) is built
        # above; the wgrad runs once a_prime exists.

        # ======================================================================
        # Step 2: FC2 dgrad + SwiGLU backward -> dY1 [M, 2I] = dgate||dup, and dprob (router-prob grad).
        #   dA[m,:]  = dY2[m,:] @ W2[e]                          (FC2 dgrad, K=d)
        #   grad     = dA * prob                                 (forward applied A *= prob)
        #   dY1      = dswiglu(grad, h)                          (dgate || dup)
        #   dprob[m] = <silu(gate)*up, dA>[m] = <swiglu(h), dA>  (router-prob grad, if required)
        #
        # B2 FUSED (h saved + binding present): te_cutlass_grouped_dswiglu does the FC2-dgrad GEMM in
        #   TMEM (dA never reaches HBM) and the SwiGLU-backward epilogue reads the SAVED h, writing dY1
        #   and column-reducing dprob -- ONE SM100 CUTLASS kernel replacing (separate dA grouped GEMM +
        #   dA*prob + dswiglu kernel + swiglu-for-dprob).
        # FALLBACK (h not saved, or binding absent): materialize dA via a grouped GEMM, use the saved h
        #   or recompute h = x@W1^T, then the per-op dswiglu + (optional) swiglu-for-dprob.
        # ======================================================================
        grouped_fc1_weight = self._wrap_weight_grouped(
            fc1_op, fc1_weight, num_groups, fc1_weight_shape, dtype, device
        )

        # x / grouped_x are needed for: (a) the FC1 wgrad dW1 = dY1^T @ x (only if weight_requires_grad),
        # and (b) the recompute fallback. Build only when needed -- Design B with saved h and frozen
        # weights needs NEITHER (h saved + no wgrad).
        grouped_x: Optional[GroupedTensor] = None
        need_recompute = swiglu_in_saved is None
        if fc1_ctx.weight_requires_grad or need_recompute:
            x = self._grouped_x_data(grouped_fc1_x, M, d, dtype, device)
            grouped_x = GroupedTensor(
                shape=(M, d), dtype=dtype, num_tensors=num_groups, quantizer=None,
                data=x.reshape(-1), first_dims=split_sizes, tensor_offsets=base_split_offsets * d,
            )

        need_dprob = scales is not None and activation_ctx.extra_input_requires_grad
        grad_scales = None

        if swiglu_in_saved is not None:
            h = maybe_dequantize(swiglu_in_saved, dtype).reshape(M, two_i).contiguous()
        else:
            # NVTE_FUSED_MOE_RECOMPUTE_H: forward skipped h_saved (~20GiB/rank). Recompute h by
            # replicating the forward gemm_gated preact from saved x@W1 -- bit-identical, +1 up gemm.
            from quack.gemm_interface import gemm as _quack_gemm_rc
            _w1_rc = fc1_weight if fc1_op.single_grouped_weight else torch.cat(list(fc1_weight), dim=0)
            _B_rc = _w1_rc.view(num_groups, two_i, d).permute(0, 2, 1)
            h = torch.empty(M, two_i, dtype=dtype, device=device)
            _quack_gemm_rc(x, _B_rc, out=h,
                           cu_seqlens_m=base_split_offsets.to(torch.int32),
                           dynamic_scheduler=True, tuned=True)

        # The QuACK gemm_dgated / wgrad path below is driven by ``cu`` (cumsum of split_sizes,
        # built just below) and handles ragged, non-256-aligned per-expert counts. The old CUTLASS
        # ``m_tile_expert`` tag-array (split//256 + repeat_interleave(output_size=ceil(M/256))) that
        # ASSERTED on non-mod-256 splits is dead under QuACK and was removed.
        w2_2d = self._get_fc2_weight_2d(fc2_op, num_groups, fc2_weight_shape, dtype)
        prob_f32 = (
            maybe_dequantize(scales, torch.float32).reshape(-1).contiguous()
            if scales is not None
            else None
        )
        # FC2-dgrad + dswiglu + dprob col-reduce in ONE QuACK gemm_dgated, reading the interleaved h
        # that the forward gemm_gated stored. dY1 stays INTERLEAVED ([gate0,up0,...]) and is consumed
        # directly by the QuACK up-dgrad/wgrad below (concat_layout de-interleaves -> plain dX/dW1).
        from quack.gemm_interface import gemm as quack_gemm, gemm_dgated
        # Reuse the prefix-sum computed once in the forward (compute_moe_offsets) and restored from
        # ctx as base_split_offsets [G+1] int64 -- cast to int32 cu_seqlens, NO cumsum recompute.
        cu = base_split_offsets.to(torch.int32)
        dY1 = torch.empty(M, two_i, dtype=dtype, device=device)   # INTERLEAVED [gate0,up0,...]
        a_prime = torch.empty(M, I, dtype=dtype, device=device)
        _res = gemm_dgated(
            grad_output.contiguous(),                # dY2 [M, d]
            w2_2d.view(num_groups, d, I),            # W2  [G, d, I]
            PreAct=h,                                # interleaved saved h [M, 2I]
            activation="swiglu",
            dx_out=dY1,
            postact_out=a_prime,
            colvec_scale=prob_f32,
            colvec_reduce=bool(need_dprob),
            cu_seqlens_m=cu,
            dynamic_scheduler=False,
            tuned=True,  # default tuning
        )
        # colvec_reduce=True returns (preact, postact, ds_dprob); False returns 2 values.
        ds = _res[2] if (isinstance(_res, (tuple, list)) and len(_res) >= 3) else None
        if need_dprob and ds is not None:
            grad_scales = ds.to(dtype=dtype)

        # ---- dW2 = sum_t dO_t^T @ A'_{e,t}  (NT, varlen-K) ----------------------------------------
        # SonicMoE: A' = `a_prime` (= s*SwiGLU(h)) was just recomputed by the gemm_dgated from the
        # cached pre-activation H, so no forward A' cache is needed. a_prime equals the forward's
        # prob-scaled activation, so this dW2 matches the previous `dO^T @ (cached A')`.
        grouped_a_prime = GroupedTensor(
            shape=(M, I),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=a_prime.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * I,
        )
        fc2_grad_params = self._compute_grouped_wgrad(
            fc_op=fc2_op,
            ctx=fc2_ctx,
            num_groups=num_groups,
            weight_shape=fc2_weight_shape,
            grouped_x=grouped_a_prime,
            grouped_dy=grouped_dy2,
            dtype=dtype,
            device=device,
        )
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
            # up-dgrad via QuACK gemm on the INTERLEAVED dY1.  concat_layout=("B",) => W1 plain (gate||up),
            # dY1 (A) interleaved.  dX = dY1 @ W1  -> [M, d].
            grad_input = torch.empty(M, d, dtype=dtype, device=device)
            _w1_3d = grouped_fc1_weight.rowwise_data.view(num_groups, two_i, d)  # [G, 2I, D]
            quack_gemm(dY1, _w1_3d, out=grad_input, cu_seqlens_m=cu,
                       concat_layout=("B",), dynamic_scheduler=False, tuned=True)
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
            dy_interleaved=True,  # dY1 is interleaved -> concat_layout=("out",) de-interleaves to plain dW1
        )

        # Clear saved activation buffers if possible (ref backward_grouped_mlp.py:662-673).
        # No FC2 activation cache to clear: SonicMoE recomputes A' (a_prime) in the gemm_dgated
        # rather than caching it. a_prime is backward-local -- freed naturally, or held by the
        # wgrad_store closure until the deferred dW2 runs.
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

        def _make(weight_data: torch.Tensor) -> GroupedTensor:
            return GroupedTensor(
                shape=(num_groups * out_features, in_features),
                dtype=dtype,
                num_tensors=num_groups,
                shapes=[(out_features, in_features)] * num_groups,
                quantizer=None,
                data=weight_data,
            )

        if fc_op.single_grouped_weight:
            # ALIGNED with the reference: reuse the packed buffer as a view (no copy).
            w = (
                maybe_dequantize(weight.rowwise_data, dtype)
                if isinstance(weight, GroupedTensor)
                else maybe_dequantize(weight, dtype)
            )
            return _make(w.reshape(-1))
        # Per-expert: stack + wrap, mirroring the forward op-wrapper fix. CACHE on
        # fc_op (distinct for FC1 vs FC2) keyed on the source weights' (id, _version)
        # so the stack only re-runs when the optimizer updates them.
        key = tuple((id(w), w._version) for w in weight)
        cache = getattr(fc_op, "_fused_moe_bwd_wcache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        weights = [maybe_dequantize(w, dtype) for w in weight]
        gt = _make(torch.stack(weights, dim=0).contiguous().reshape(-1))
        fc_op._fused_moe_bwd_wcache = (key, gt)
        return gt

    def _get_fc2_weight_2d(
        self,
        fc2_op: GroupedLinear,
        num_groups: int,
        fc2_weight_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the FC2 weight as a contiguous bf16 [G*d, I] tensor (per-expert [d, I] stacked).

        This is the W2 B-operand layout ``te_cutlass_grouped_dswiglu`` expects (the SM100 dswiglu
        kernel computes dA = dY2 @ W2 with K=d, N=I). Mirrors the forward's ``_get_fc1_weight_2d``
        ([G*2I, d]).
        """
        out_features, in_features = fc2_weight_shape  # (d, I)
        if fc2_op.single_grouped_weight:
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            w = maybe_dequantize(fc2_op.weight.rowwise_data, dtype)
            return w.view(num_groups * out_features, in_features)
        # Per-expert params: stack into one contiguous [G*d, I]; CACHE keyed on (id, _version) so the
        # copy only re-runs when the optimizer updates the weights in-place.
        weight_params = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
        key = tuple((id(w), w._version) for w in weight_params)
        cache = getattr(self, "_fc2_w2d_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        weights = [maybe_dequantize(w, dtype) for w in weight_params]
        stacked = (
            torch.stack(weights, dim=0).view(num_groups * out_features, in_features).contiguous()
        )
        self._fc2_w2d_cache = (key, stacked)
        return stacked

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
        dy_interleaved: bool = False,
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

        # wgrad via QuACK gemm: dW^T = X^T @ dY (ragged-K via cu_seqlens_k), written into the [G,out,in]
        # weight-grad buffer through a transposed [G,in,out] view (no extra transpose; ~1.2-1.3x vs the
        # CUTLASS grouped GEMM). concat_layout=("out",) de-interleaves an interleaved dY (fc1) -> plain
        # dW (Muon-safe). main_grad accumulate uses the in-place C=out/beta=1 path (QuACK varlen-k
        # accumulate fix); for the per-expert (multi-weight) layout we accumulate via tensor add.
        delay_wgrad = fc_op.wgrad_store is not None and fc_op.wgrad_store.delay_wgrad_compute()
        if grouped_x is None:
            raise RuntimeError("Fused MoE backward: grouped_x is required for wgrad")
        from quack.gemm_interface import gemm_tuned  # noqa
        from quack.moe_offsets import compute_moe_offsets  # pylint: disable=import-outside-toplevel

        # Fused QuACK Triton prefix-sum (one kernel) instead of pad(cumsum()) (DeviceScan + pad).
        cu_k = compute_moe_offsets(grouped_x.first_dims)[0]
        xT = grouped_x.rowwise_data.view(-1, in_features).transpose(0, 1)  # [in, M] (M=K, varlen)
        dyrow = grouped_dy.rowwise_data.view(-1, out_features)             # [M, out]
        concat = ("out",) if dy_interleaved else None

        # NVTE_USE_BF16_FUSED_MOE_TUNNING (default OFF): OFF -> gemm_tuned.fn(config=None) bypasses the
        # autotuner (default_config, no precompile workers); ON -> gemm_tuned() autotunes for best config.
        _wgrad_fn = gemm_tuned  # default tuning (autotuned wgrad)

        def _wgrad_into(dst_g3, accum):
            # dst_g3: [G, out, in]. QuACK writes dW^T into the [G, in, out] transposed view.
            dstT = dst_g3.transpose(1, 2)
            if accum:
                _wgrad_fn(xT, dyrow, dstT, C=dstT, beta=1.0, cu_seqlens_k=cu_k,
                          concat_layout=concat, dynamic_scheduler=False)
            else:
                _wgrad_fn(xT, dyrow, dstT, cu_seqlens_k=cu_k,
                          concat_layout=concat, dynamic_scheduler=False)

        def _run_wgrad():
            if single:
                # contiguous [G*out,in] buffer (main_grad when accumulate) -> in-place += via C=out.
                _wgrad_into(grouped_wgrad.rowwise_data.view(num_groups, out_features, in_features),
                            accum=accumulate)
            else:
                _tmp = torch.empty(num_groups, out_features, in_features, dtype=dtype, device=device)
                _wgrad_into(_tmp, accum=False)
                for i in range(num_groups):
                    if accumulate:
                        final_weight_grads[i].add_(_tmp[i])   # main_grad[i] += dW[i]
                    else:
                        final_weight_grads[i] = _tmp[i]

        if delay_wgrad:
            fc_op.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], lambda *a: _run_wgrad())
        else:
            _run_wgrad()

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
