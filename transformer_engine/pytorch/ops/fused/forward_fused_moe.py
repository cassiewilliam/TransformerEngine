# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for BF16 MoE grouped MLP using a CUTLASS fused SwiGLU kernel.

This is the BF16 SonicMoE counterpart of ``forward_grouped_mlp.py`` (the MXFP8
cuTe-DSL reference). It wires the validated CUTLASS kernel

    ``tex.te_cutlass_grouped_swiglu(x, w1, m_tile_expert, prob, G, Me, I, d,
                                    M_varlen, math_sm_count) -> A``

which fuses the FC1 up-projection (``d -> 2I``) with the SwiGLU activation (and
optional per-token router-prob multiply) into a single grouped GEMM, returning
``A: bf16 [M, I]`` -- the ``[M, 2I]`` gate||up tensor is never materialized.

Compared to the MXFP8 reference this file DROPS, by design:
  * all MXFP8/FP8 quantization (sfa/sfb scale tensors, swizzle permutes,
    alpha_tensor, norm_const, c_dtype/d_dtype fp8 logic, discrete-vs-grouped
    fp8 weight packing). Everything here is pure bf16.
  * the cuDNN-FE grouped-GEMM quant kernel for FC2; instead the FC2 down-proj
    reuses the standard bf16 grouped GEMM primitive
    (``general_grouped_gemm_for_grouped_tensor``), exactly the kernel that
    ``GroupedLinear``'s bf16 graph-safe path uses for its forward GEMM.
"""

from __future__ import annotations
from collections.abc import Callable, Iterable
import functools
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm_for_grouped_tensor
from ...module.base import _2X_ACC_FPROP
from ...quantization import Recipe
from ...tensor import Quantizer
from ...tensor.grouped_tensor import GroupedTensor
from ...utils import get_device_compute_capability
from ..basic import GroupedLinear
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    is_glu_activation,
    maybe_dequantize,
    validate_grouped_mlp_dims,
)

# Each expert's token count MUST be a multiple of this for the CUTLASS kernel's
# uniform-M (m_tile_expert=None) and varlen-M m-tiling. See the 256-alignment
# assert in ``fuser_forward``.
_CUTLASS_TILE_M = 256


class ForwardFusedMoE_CutlassSwiGLU_BF16(FusedOperation):
    """Fused op for BF16 GroupedLinear + SwiGLU + GroupedLinear (CUTLASS).

    Mirrors ``_ForwardGroupedMLP_CuTeGEMMBase_MXFP8`` in
    ``forward_grouped_mlp.py`` but uses the validated CUTLASS fused
    up-proj+SwiGLU kernel and a plain bf16 grouped GEMM for the down-proj.

    Unlike the MXFP8 reference there is no ``_*Base`` / GLU-vs-SReLU split:
    the CUTLASS kernel is SwiGLU-only, so a single class suffices.
    """

    # bf16: there is no separate activation/quant kernel selector like the
    # reference's ``grouped_gemm_activation_kernel`` -- the one CUTLASS kernel
    # does up-proj + SwiGLU. We still expose it as an lru_cache classmethod so
    # ``is_supported`` can probe for the binding exactly like the reference
    # probes its cuDNN-FE imports (ref forward_grouped_mlp.py:100-104).
    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_swiglu_kernel(cls) -> Callable:
        """CUTLASS fused grouped up-proj + SwiGLU kernel (raises if missing)."""
        # Probe the binding; AttributeError is mapped to ImportError so that
        # ``is_supported`` can treat a missing binding like the reference treats
        # a failed ``from cudnn import ...`` (ImportError).
        try:
            return tex.te_cutlass_grouped_swiglu
        except AttributeError as e:
            raise ImportError("transformer_engine_torch.te_cutlass_grouped_swiglu") from e

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system.

        Mirrors the reference ``is_supported`` (forward_grouped_mlp.py:90-105)
        but swaps the env flag to ``NVTE_USE_FUSED_MOE`` and probes the CUTLASS
        binding instead of the cuDNN-FE wrappers. SwiGLU-ness of the activation
        is validated per-instance in ``__init__`` (the reference likewise relies
        on ``fuse_grouped_mlp_ops`` to only feed it GLU triples).
        """
        # bf16: SonicMoE gate flag, not NVTE_CUTEDSL_FUSED_GROUPED_MLP.
        if int(os.environ.get("NVTE_USE_FUSED_MOE", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        try:
            cls.grouped_swiglu_kernel()
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
            self.grouped_swiglu_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        # validate_grouped_mlp_dims enforces fc1.out_features == 2*fc2.in_features
        # (GLU), matching num_groups, and 32-wide GLU interleave (ref __init__
        # forward_grouped_mlp.py:120).
        validate_grouped_mlp_dims(fc1, activation, fc2)
        # bf16 CUTLASS kernel is SwiGLU-only: reject GeGLU/SReLU here. The
        # reference instead selected a cuDNN act_func string; we only support
        # silu(gate)*up.
        if not is_glu_activation(activation):
            raise TypeError(
                f"{self.__class__.__name__} requires a SwiGLU activation, "
                f"got {activation.__class__.__name__}."
            )
        # VERIFY: ScaledClampedQGeGLU also passes is_glu_activation; the CUTLASS
        # kernel only implements silu-gated SwiGLU, so reject the clamped-GeGLU
        # variant explicitly. Mirror import is local to avoid a hard dependency.
        from ..basic import ScaledClampedQGeGLU  # pylint: disable=import-outside-toplevel

        if isinstance(activation, ScaledClampedQGeGLU):
            raise TypeError(
                f"{self.__class__.__name__} does not support clamped GeGLU; "
                "only silu-gated SwiGLU."
            )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        # Get basic operations (ref forward_grouped_mlp.py:155-156)
        fc1_op, activation_op, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties (ref forward_grouped_mlp.py:159-163)
        # FC1 weight: (out=2I, in=d).  FC2 weight: (out=d_out, in=I).
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        d = fc1_weight_shape[1]  # hidden in-dim
        two_i = fc1_weight_shape[0]  # 2 * I
        intermediate = fc2_weight_shape[1]  # I (FC2 in_features)
        assert two_i == 2 * intermediate, (
            f"FC1 out_features ({two_i}) must equal 2*FC2 in_features "
            f"(2*{intermediate})."
        )
        input_ = input_.reshape(-1, d)
        in_shape = list(input_.size())
        M = in_shape[0]

        num_groups = fc1_op.num_groups
        G = num_groups
        I = intermediate  # noqa: E741 - matches kernel arg name
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        device = fc1_weight_param.device
        # bf16: autocast dtype or the weight dtype (ref forward_grouped_mlp.py:169-172).
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_weight_param.dtype

        # Check which grads are required (ref forward_grouped_mlp.py:175-179)
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = requires_grad and (
            fc1_weight_param.requires_grad or fc2_weight_param.requires_grad
        )

        # Extract split sizes from extra inputs (ref forward_grouped_mlp.py:190-202).
        # Both GroupedLinears carry the per-expert token counts as extra input 0.
        fc1_split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            fc1_split_sizes.size() != fc2_split_sizes.size()
            or fc1_split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} got different split points for FC1 and FC2."
            )
        split_sizes = fc1_split_sizes
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        split_sizes = split_sizes.to(dtype=torch.int64, device=device)
        # base_split_offsets[i] = start row of expert i, base_split_offsets[-1]=M.
        base_split_offsets = tex.splits_to_offsets(split_sizes, 1)

        # Extract per-row activation probabilities from the SwiGLU op's extra
        # input (ref forward_grouped_mlp.py:208 -> prob_tensor:337). This is the
        # per-token router gate. The CUTLASS kernel applies A[m,:] *= prob[m].
        scales = basic_op_extra_inputs[1][0]

        # --- 256-alignment contract -------------------------------------------
        # The CUTLASS kernel m-tiles in blocks of 256 rows and requires each
        # expert's token count to be a multiple of 256 (uniform-M when
        # m_tile_expert=None, or per-tile expert ids for varlen-M). The SonicMoE
        # dispatcher upstream usually pads tokens to 256 (token-rounding). For
        # this first version we FAIL LOUDLY on ragged counts rather than
        # silently mis-handling them.
        #
        # TODO(sonic-moe): general path -- pad each expert's tokens up to a
        # multiple of 256 before this kernel + FC2, then unpad (slice) the FC2
        # output back to the original token layout. Until then the dispatcher
        # must hand us already-padded (mod-256) per-expert counts.
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        for e, n_tok in enumerate(split_sizes_int):
            if n_tok % _CUTLASS_TILE_M != 0:
                raise ValueError(
                    f"{self.__class__.__name__}: expert {e} has {n_tok} tokens, which is "
                    f"not a multiple of TileM={_CUTLASS_TILE_M}. The CUTLASS fused-SwiGLU "
                    "kernel requires each expert's token count to be 256-aligned. Pad "
                    "tokens upstream (SonicMoE token-rounding) before the fused MoE op."
                )

        # --- Build m_tile_expert / M_varlen for the CUTLASS kernel ------------
        # Uniform-M fast path: when every expert has exactly Me = M/G tokens we
        # pass m_tile_expert=None and M_varlen=0 (kernel derives the layout from
        # G, Me). This matches the validated binding's ungated/gated calls
        # (qa/te_cutlass_swiglu_test.py:47,53).
        uniform = M % G == 0 and all(n == split_sizes_int[0] for n in split_sizes_int)
        if uniform and split_sizes_int[0] > 0:
            Me = split_sizes_int[0]
            m_tile_expert = None
            M_varlen = 0
        else:
            # Varlen-M path: map each 256-row m-tile to its expert id (int32
            # CUDA [ceil(M/256)]). Because every expert count is a multiple of
            # 256 (asserted above), expert boundaries fall on tile boundaries.
            Me = M // G if (M % G == 0) else 0  # unused by kernel when m_tile_expert!=None
            num_tiles = (M + _CUTLASS_TILE_M - 1) // _CUTLASS_TILE_M
            tile_expert = torch.empty(num_tiles, dtype=torch.int32, device=device)
            tile = 0
            for e, n_tok in enumerate(split_sizes_int):
                e_tiles = n_tok // _CUTLASS_TILE_M
                if e_tiles > 0:
                    tile_expert[tile : tile + e_tiles] = e
                    tile += e_tiles
            # Any trailing tiles (should not happen given the mod-256 assert)
            # default to the last expert.
            if tile < num_tiles:
                tile_expert[tile:num_tiles] = max(num_groups - 1, 0)
            m_tile_expert = tile_expert
            M_varlen = M

        # --- FC1 (up-proj) + SwiGLU via the CUTLASS kernel --------------------
        # w1: bf16 [G*2I, d] per-expert gate||up stacked. The GroupedLinear
        # stores its weight either as a single GroupedTensor (single_grouped_weight)
        # or as per-expert weight{idx} params -- both are logically [G, 2I, d].
        # We materialize a contiguous bf16 [G*2I, d] view for the kernel.
        x = maybe_dequantize(input_, dtype).contiguous()
        w1 = self._get_fc1_weight_2d(fc1_op, num_groups, fc1_weight_shape, dtype)

        # prob: fp32 CUDA [M] per-token router gate, or None when absent.
        prob = None
        if scales is not None:
            prob = maybe_dequantize(scales, torch.float32).reshape(-1).contiguous()

        # The single kernel call: replaces up-GroupedLinear(d->2I) + SwiGLU
        # (+ optional router-prob mul). Returns A: bf16 [M, I].
        # math_sm_count=0 => kernel auto-detects (qa test passes 0).
        A = self.grouped_swiglu_kernel()(
            x,
            w1,
            m_tile_expert,
            prob,
            G,
            Me,
            I,
            d,
            M_varlen,
            0,
        )
        # VERIFY: kernel returns A in bf16 [M, I]; confirm on B200.
        A = A.view(M, I)

        # --- FC2 (down-proj): plain bf16 grouped GEMM -------------------------
        # NOT the CUTLASS kernel. We reuse general_grouped_gemm_for_grouped_tensor
        # (the same primitive GroupedLinear's bf16 graph-safe forward uses, see
        # grouped_linear.py:1244) with layout="TN": out = A @ W2^T per expert.
        w2 = self._get_fc2_weight(fc2_op, num_groups, fc2_weight_shape, dtype, device)

        grouped_A = GroupedTensor(
            shape=(M, I),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=A.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * I,
        )
        fc2_out = torch.empty(M, fc2_weight_shape[0], dtype=dtype, device=device)
        grouped_fc2_out = GroupedTensor(
            shape=(M, fc2_weight_shape[0]),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=fc2_out.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * fc2_weight_shape[0],
        )
        general_grouped_gemm_for_grouped_tensor(
            w2,
            grouped_A,
            grouped_fc2_out,
            layout="TN",
            use_split_accumulator=_2X_ACC_FPROP,
        )

        # Reshape output to the original leading dims (ref forward_grouped_mlp.py:439).
        out = fc2_out.view(M, fc2_weight_shape[0])

        # --- Save state for backward ------------------------------------------
        # bf16: there is no fused CUTLASS backward yet (see backward_fused_moe.py),
        # so we save each basic op's ctx in EXACTLY the layout that op's own
        # ``fuser_backward`` expects, so the per-op backward (run via the fused
        # backward's delegation, or via the basic ops if no backward fusion is
        # registered) is correct.
        if requires_grad:
            self._save_backward_ctx(
                fc1_op=fc1_op,
                activation_op=activation_op,
                fc2_op=fc2_op,
                fc1_ctx=fc1_ctx,
                activation_ctx=activation_ctx,
                fc2_ctx=fc2_ctx,
                num_groups=num_groups,
                split_sizes=split_sizes,
                base_split_offsets=base_split_offsets,
                x=x,
                w1_input=input_,
                A=A,
                scales=scales,
                dtype=dtype,
                device=device,
                fc1_weight_shape=fc1_weight_shape,
                fc2_weight_shape=fc2_weight_shape,
                input_requires_grad=input_requires_grad,
                weight_requires_grad=weight_requires_grad,
            )

        # Extra outputs per basic op: FC1 / activation / FC2 each emit none
        # (matches ref forward_grouped_mlp.py:586 -> [(), (), ()]).
        return out, [(), (), ()]

    # ----------------------------------------------------------------------
    # Helpers (bf16-only; the reference inlines these with fp8 packing).
    # ----------------------------------------------------------------------
    @staticmethod
    def _get_fc1_weight_2d(
        fc1_op: GroupedLinear,
        num_groups: int,
        fc1_weight_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return FC1 weight as a contiguous bf16 [G*2I, d] tensor.

        gate = rows [e*2I : e*2I + I], up = rows [+I : +2I] per expert, exactly
        the layout the CUTLASS kernel expects (qa/te_cutlass_swiglu_test.py:21-24).
        """
        out_features, in_features = fc1_weight_shape
        if fc1_op.single_grouped_weight:
            # Single GroupedTensor param: rowwise_data is the packed
            # [G, 2I, d] (== [G*2I, d]) bf16 buffer.
            if not isinstance(fc1_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC1 expected GroupedTensor weight with single_grouped_weight=True."
                )
            w = maybe_dequantize(fc1_op.weight.rowwise_data, dtype)
            return w.view(num_groups * out_features, in_features).contiguous()
        # Per-expert params: stack [G, 2I, d] -> [G*2I, d].
        weights = [
            maybe_dequantize(getattr(fc1_op, f"weight{idx}"), dtype) for idx in range(num_groups)
        ]
        return torch.stack(weights, dim=0).view(num_groups * out_features, in_features).contiguous()

    @staticmethod
    def _get_fc2_weight(
        fc2_op: GroupedLinear,
        num_groups: int,
        fc2_weight_shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> GroupedTensor:
        """Return FC2 weight as a uniform bf16 GroupedTensor for grouped GEMM.

        Mirrors ``GroupedLinear._get_grouped_weight_for_gemm`` /
        ``_get_discrete_weights_for_gemm`` for the unquantized (bf16) case
        (grouped_linear.py:791-804), producing a [G*out, in] packed buffer.
        """
        out_features, in_features = fc2_weight_shape
        if fc2_op.single_grouped_weight:
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            w = maybe_dequantize(fc2_op.weight.rowwise_data, dtype)
            weight_data = w.reshape(-1)
        else:
            weights = [
                maybe_dequantize(getattr(fc2_op, f"weight{idx}"), dtype)
                for idx in range(num_groups)
            ]
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
    def _save_backward_ctx(
        *,
        fc1_op: GroupedLinear,
        activation_op: FusibleOperation,
        fc2_op: GroupedLinear,
        fc1_ctx: OperationContext,
        activation_ctx: OperationContext,
        fc2_ctx: OperationContext,
        num_groups: int,
        split_sizes: torch.Tensor,
        base_split_offsets: torch.Tensor,
        x: torch.Tensor,
        w1_input: torch.Tensor,
        A: torch.Tensor,
        scales: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        fc1_weight_shape: tuple[int, int],
        fc2_weight_shape: tuple[int, int],
        input_requires_grad: bool,
        weight_requires_grad: bool,
    ) -> None:
        """Save per-op ctx so the standard per-op bf16 backward runs correctly.

        bf16: there is no fused CUTLASS backward, so each basic op's ctx is
        populated in the exact layout its own ``fuser_backward`` consumes:
          * FC1 GroupedLinear  -> ``_fuser_backward_grouped_tensor`` layout
            [split_sizes, base_split_offsets, split_points, grouped_x, *weights]
            (grouped_linear.py:1504-1521).
          * SwiGLU (_ScaledGLU) -> saves (swiglu_in, scales)
            (swiglu.py:469-472); we save the FC1 up-proj output as swiglu_in.
          * FC2 GroupedLinear  -> same grouped-tensor layout
            [split_sizes, base_split_offsets, split_points, grouped_x, *weights]
            with grouped_x = the activation output A.
        See backward_fused_moe.py for how these ctxs are consumed.
        """
        out_features_1, in_features_1 = fc1_weight_shape
        out_features_2, in_features_2 = fc2_weight_shape
        M = x.size(0)
        base_split_offsets_i64 = base_split_offsets
        split_points = base_split_offsets[1:].to(dtype=torch.int)

        # ---- FC1 GroupedLinear ctx (bf16 grouped-tensor backward) ----
        # grouped_x for FC1 wgrad is the FC1 input (x).
        grouped_fc1_x = None
        if weight_requires_grad:
            grouped_fc1_x = GroupedTensor(
                shape=(M, in_features_1),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=x.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets_i64 * in_features_1,
            )
        fc1_weight_tensors = fc1_op._get_weight_tensors()  # len 1 or num_groups
        fc1_saved: list[Optional[torch.Tensor]] = [
            split_sizes,
            base_split_offsets_i64,
            split_points,
            grouped_fc1_x,
        ]
        fc1_saved.extend(fc1_weight_tensors)
        fc1_ctx.save_for_backward(*fc1_saved)
        # bf16: not the grouped-tensor MXFP8 path -- but GroupedLinear's bf16
        # backward also dispatches on use_grouped_tensor_path=True (it just
        # skips quantization). with_quantized_compute=False selects bf16.
        fc1_ctx.use_grouped_tensor_path = True
        fc1_ctx.with_quantized_compute = False
        fc1_ctx.input_quantizers = [None] * num_groups
        fc1_ctx.weight_quantizers = [None] * num_groups
        fc1_ctx.grad_output_quantizers = [None] * num_groups
        fc1_ctx.grad_input_quantizers = None
        fc1_ctx.dtype = dtype
        fc1_ctx.input_requires_grad = input_requires_grad
        fc1_ctx.weight_requires_grad = weight_requires_grad

        # ---- SwiGLU activation ctx ----
        # _ScaledGLU.fuser_backward expects (input_, scales) where input_ is the
        # SwiGLU input (the FC1 up-proj [M, 2I]) (swiglu.py:494). bf16: we DID
        # NOT materialize the [M, 2I] gate||up tensor (the CUTLASS kernel fuses
        # it away), so the standard SwiGLU backward CANNOT recompute dx from it.
        # The backward instead recomputes the up-proj on demand; see
        # backward_fused_moe.py. We save (None, scales) here and let the fused
        # backward recompute the SwiGLU input.
        # VERIFY: confirm backward recompute path against the per-op reference.
        activation_ctx.save_for_backward(None, scales if input_requires_grad else None)
        activation_ctx.extra_input_requires_grad = (
            scales is not None and scales.requires_grad
        )
        activation_ctx.input_requires_grad = True
        activation_ctx.dtype = dtype

        # ---- FC2 GroupedLinear ctx (bf16 grouped-tensor backward) ----
        # grouped_x for FC2 wgrad is the activation output A [M, I].
        grouped_fc2_x = None
        if weight_requires_grad:
            grouped_fc2_x = GroupedTensor(
                shape=(M, in_features_2),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=A.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets_i64 * in_features_2,
            )
        fc2_weight_tensors = fc2_op._get_weight_tensors()
        fc2_saved: list[Optional[torch.Tensor]] = [
            split_sizes,
            base_split_offsets_i64,
            split_points,
            grouped_fc2_x,
        ]
        fc2_saved.extend(fc2_weight_tensors)
        fc2_ctx.save_for_backward(*fc2_saved)
        fc2_ctx.use_grouped_tensor_path = True
        fc2_ctx.with_quantized_compute = False
        fc2_ctx.input_quantizers = [None] * num_groups
        fc2_ctx.weight_quantizers = [None] * num_groups
        fc2_ctx.grad_output_quantizers = [None] * num_groups
        fc2_ctx.grad_input_quantizers = None
        fc2_ctx.dtype = dtype
        fc2_ctx.input_requires_grad = input_requires_grad
        fc2_ctx.weight_requires_grad = weight_requires_grad


def fuse_forward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply BF16 GroupedLinear + SwiGLU + GroupedLinear fusion for forward pass.

    Mirrors ``forward_grouped_mlp.fuse_forward_ops`` (forward_grouped_mlp.py:619)
    but substitutes the CUTLASS BF16 fused op.
    """
    # bf16: fuse_grouped_mlp_ops gates on ``recipe.mxfp8()`` (see _common.py:288).
    # The CUTLASS path is bf16 and runs WITHOUT an FP8 autocast recipe (recipe is
    # None), so the shared matcher would early-return. We replicate its
    # sliding-window matching here with a bf16-appropriate gate.
    return _fuse_forward_ops_bf16(ops, fused_op_cls=ForwardFusedMoE_CutlassSwiGLU_BF16)


def _fuse_forward_ops_bf16(
    ops: list[FusibleOperation],
    *,
    fused_op_cls,
) -> list[FusibleOperation]:
    """Sliding-window [GroupedLinear, ScaledSwiGLU, GroupedLinear] matcher (bf16).

    Structurally identical to ``fuse_grouped_mlp_ops`` (_common.py:256) but
    drops the ``recipe.mxfp8()`` requirement (this kernel is bf16) and only
    matches the silu-gated ``ScaledSwiGLU`` activation.
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


# Register fusion if available (ref forward_grouped_mlp.py:664-666)
if ForwardFusedMoE_CutlassSwiGLU_BF16.is_supported():
    register_forward_fusion(fuse_forward_ops, prepend=True)
