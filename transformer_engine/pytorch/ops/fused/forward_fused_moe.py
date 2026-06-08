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
        # F group is gated by NVTE_USE_FUSED_MOE (single canonical flag).
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
        # validate_grouped_mlp_dims enforces fc1.out_features == 2*fc2.in_features (GLU),
        # matching num_groups, and (for GLU) 32-wide interleaving. The bf16 CUTLASS SwiGLU
        # kernel reads plain gate||up and ignores glu_interleave_size, so accept plain (None)
        # interleave too: present 32 to the shared validator for the dim/num_groups checks,
        # then restore. No compute effect -- the kernel never reads this field. (Adapted
        # locally so validate_grouped_mlp_dims keeps its 32-wide contract for other callers.)
        _plain_glu = getattr(activation, "glu_interleave_size", 32) is None
        if _plain_glu:
            activation.glu_interleave_size = 32
        try:
            validate_grouped_mlp_dims(fc1, activation, fc2)
        finally:
            if _plain_glu:
                activation.glu_interleave_size = None
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
        # --- Build m_tile_expert ON-DEVICE (no host sync) --------------------
        # ALIGNED with forward_grouped_mlp.py, which keeps the per-expert offsets
        # on-device (split_points, ~:203) and never reads them to host. Each
        # expert e contributes split_e//256 m-tiles, all tagged expert id e
        # (varlen-M). repeat_interleave with output_size=ceil(M/256) does NOT
        # sync, and it enforces 256-alignment loudly: if any split is not a
        # multiple of 256, sum(split//256) != ceil(M/256) and it raises (no
        # silent mis-handling). The previous version read split_sizes.tolist()
        # (a ~21us CPU<->GPU sync) + looped on the host -- removed.
        # TODO(sonic-moe): a Ptr-Array W1 kernel could take the per-expert
        # offsets directly (like the reference's padded_offsets) and skip this.
        num_tiles = (M + _CUTLASS_TILE_M - 1) // _CUTLASS_TILE_M
        tiles_per_expert = torch.div(split_sizes, _CUTLASS_TILE_M, rounding_mode="floor")
        m_tile_expert = torch.repeat_interleave(
            torch.arange(num_groups, device=device, dtype=torch.int32),
            tiles_per_expert,
            output_size=num_tiles,
        )
        M_varlen = M
        Me = 0  # unused by the kernel in varlen-M mode (m_tile_expert != None)

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
        if int(os.environ.get("NVTE_USE_FUSED_MOE", "0")) > 0:
            # QuACK gemm_gated branch (fastest fused up-proj+SwiGLU; ~1.6x the CUTLASS kernel E2E).
            # concat_layout=("B",) consumes the SAME plain [G*2I,d] weights as the CUTLASS path -- no
            # re-interleave, so NO Muon/checkpoint impact and NO perf loss (verified: 1340 vs 1348 TF,
            # mean_rel 8e-4). Handles ragged tokens via cu_seqlens (no 256-pad requirement). See
            # docs/swiglu_v2_quack_analysis_v2.html. gemm_gated has no per-token prob param, so the
            # router-gate multiply (applied INSIDE the CUTLASS kernel) is done here post-hoc.
            from quack.gemm_interface import gemm_gated  # local import: optional QuACK dependency

            cu_seqlens_m = torch.nn.functional.pad(
                split_sizes.to(torch.int32).cumsum(0, dtype=torch.int32), (1, 0)
            )  # [G+1] per-expert m-offsets, on-device (no host sync)
            # w1 [G*2I,d] row-major == [E,2I,d]; -> B=[E,d,2I] (K=d contiguous, expert blocks
            # contiguous) which is exactly what gemm_gated + concat_layout=("B",) wants.
            B_gated = w1.view(num_groups, two_i, d).permute(0, 2, 1)
            A = torch.empty(M, I, dtype=dtype, device=device)
            # NVTE_QUACK_EMIT_H=1: let QuACK store the pre-activation h directly (preact_out) instead
            # of recomputing X@W1^T in the backward-h block below (saves ~102us, see
            # docs/b300_fused_moe_validation.md "Items 1+2"). NOTE: QuACK stores h *interleaved*
            # ([gate0,up0,...]); it is ONLY correct paired with the QuACK gemm_dgated backward
            # (which consumes interleaved h) -- the B2 te_cutlass_grouped_dswiglu reads PLAIN h, so
            # do NOT enable this without the matching backward. Default OFF.
            _emit_h = requires_grad and int(os.environ.get("NVTE_QUACK_EMIT_H", "1")) > 0
            _h_quack = torch.empty(M, two_i, dtype=dtype, device=device) if _emit_h else None
            gemm_gated(
                x,
                B_gated,
                activation="swiglu",
                cu_seqlens_m=cu_seqlens_m,
                postact_out=A,
                preact_out=_h_quack,
                store_preact=_emit_h,
                concat_layout=("B",),
            )
            if prob is not None:
                A = A * prob.view(-1, 1).to(A.dtype)
        else:
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
        # VERIFY: A is bf16 [M, I]; confirm on B200.
        A = A.view(M, I)

        # --- FC2 (down-proj) --------------------------------------------------
        # H = fc2 out_features (= model dim d); I = fc2 in_features (the ffn half, == K for the kernel).
        H = fc2_weight_shape[0]
        if hasattr(tex, "te_cutlass_grouped_down"):  # down V2 = DEFAULT in the fused op (FUSED_MOE & QUACK_SONIC)
            # SonicMoE CUTLASS down-proj kernel (down V2): a dedicated SM100 2-SM
            # tcgen05 grouped GEMM Y[M,H] = A[M,I] @ W2[G*H,I]^T per expert (~726 TFLOP/s). REUSES the
            # SAME TileM=256 m_tile_expert table built above for the up-proj (both kernels m-tile by
            # 256) -- no separate table. w2_2d is the FC2 weight as a contiguous [G*H, I] bf16 tensor,
            # mirroring _get_fc1_weight_2d. Replaces the general_grouped_gemm_for_grouped_tensor down
            # path below. Gated OFF by default (env flag); see docs / cutlass_grouped_gemm_down_v2.cuh.
            w2_2d = self._get_fc2_weight_2d(fc2_op, num_groups, fc2_weight_shape, dtype)
            # te_cutlass_grouped_down(a=A[M,I], w2=[G*H,I], m_tile_expert, G, N=H, K=I, M) -> Y[M,H].
            fc2_out = tex.te_cutlass_grouped_down(A, w2_2d, m_tile_expert, num_groups, H, I, M)
            out = fc2_out.view(M, H)
        else:
            # Plain bf16 grouped GEMM (default). NOT the CUTLASS kernel. We reuse
            # general_grouped_gemm_for_grouped_tensor (the same primitive GroupedLinear's bf16
            # graph-safe forward uses, see grouped_linear.py:1244) with layout="TN": out = A @ W2^T
            # per expert.
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
            fc2_out = torch.empty(M, H, dtype=dtype, device=device)
            grouped_fc2_out = GroupedTensor(
                shape=(M, H),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=fc2_out.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets * H,
            )
            general_grouped_gemm_for_grouped_tensor(
                w2,
                grouped_A,
                grouped_fc2_out,
                layout="TN",
                use_split_accumulator=_2X_ACC_FPROP,
            )

            # Reshape output to the original leading dims (ref forward_grouped_mlp.py:439).
            out = fc2_out.view(M, H)

        # --- SAVE h (Design B: backward fuses dA=dY@W2 + dswiglu reading saved h) -----------
        # The reference (backward_grouped_mlp.py) reads the SAVED SwiGLU input h=[gate||up] in the
        # fused backward instead of recomputing it. So the backward needs h[M,2I]. PHASE 0 recomputes
        # it here via a bf16 grouped GEMM (X@W1^T) -- the SAME cost the backward used to pay, just moved
        # to the forward -- to VALIDATE Design B's backward (correctness + no-recompute speed) with NO
        # C++ build. PHASE 1 replaces this recompute with the forward kernel EMITTING h directly (the
        # gate/up accumulators are already in TMEM), making it cheap. Gated on requires_grad.
        # TODO(sonic-moe B1): replace this recompute with the forward kernel's h-emit (Phase 1).
        h_saved = None
        if requires_grad and locals().get("_emit_h", False):
            # QuACK gemm_gated already emitted the (interleaved) preact h above -> no recompute GEMM.
            h_saved = _h_quack
        elif requires_grad:
            h_saved = torch.empty(M, two_i, dtype=dtype, device=device)
            grouped_w1_for_h = GroupedTensor(
                shape=(num_groups * two_i, d),
                dtype=dtype,
                num_tensors=num_groups,
                shapes=[(two_i, d)] * num_groups,
                quantizer=None,
                data=w1.reshape(-1),
            )
            grouped_x_for_h = GroupedTensor(
                shape=(M, d), dtype=dtype, num_tensors=num_groups, quantizer=None,
                data=x.reshape(-1), first_dims=split_sizes, tensor_offsets=base_split_offsets * d,
            )
            grouped_h_out = GroupedTensor(
                shape=(M, two_i), dtype=dtype, num_tensors=num_groups, quantizer=None,
                data=h_saved.reshape(-1), first_dims=split_sizes,
                tensor_offsets=base_split_offsets * two_i,
            )
            general_grouped_gemm_for_grouped_tensor(
                grouped_w1_for_h, grouped_x_for_h, grouped_h_out,
                layout="TN", use_split_accumulator=_2X_ACC_FPROP,
            )

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
                h_saved=h_saved,
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
    def _get_fc1_weight_2d(
        self,
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
            # ALIGNED with the reference (forward_grouped_mlp.py:365-368): the packed
            # [G, 2I, d] buffer is reused as a VIEW -> [G*2I, d]. No copy (the buffer is
            # already contiguous so .view() suffices; the old .contiguous() was a no-op).
            if not isinstance(fc1_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC1 expected GroupedTensor weight with single_grouped_weight=True."
                )
            w = maybe_dequantize(fc1_op.weight.rowwise_data, dtype)
            return w.view(num_groups * out_features, in_features)
        # Per-expert params: the kernel needs ONE contiguous [G*2I, d] buffer, so we
        # stack. The reference avoids this with a device pointer array, but our kernel
        # uses a single W1 TMA descriptor. CACHE the stack keyed on the source weights'
        # (id, _version): weights are constant across micro-batches, so the 134MB copy
        # only re-runs when the optimizer updates them in-place (bumps _version).
        weight_params = [getattr(fc1_op, f"weight{idx}") for idx in range(num_groups)]
        key = tuple((id(w), w._version) for w in weight_params)
        cache = getattr(self, "_fc1_w_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        weights = [maybe_dequantize(w, dtype) for w in weight_params]
        stacked = torch.stack(weights, dim=0).view(num_groups * out_features, in_features).contiguous()
        self._fc1_w_cache = (key, stacked)
        return stacked

    def _get_fc2_weight(
        self,
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

        def _make_grouped(weight_data: torch.Tensor) -> GroupedTensor:
            return GroupedTensor(
                shape=(num_groups * out_features, in_features),
                dtype=dtype,
                num_tensors=num_groups,
                shapes=[(out_features, in_features)] * num_groups,
                quantizer=None,
                data=weight_data,
            )

        if fc2_op.single_grouped_weight:
            # ALIGNED with the reference: reuse the packed buffer as a view (no copy).
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            w = maybe_dequantize(fc2_op.weight.rowwise_data, dtype)
            return _make_grouped(w.reshape(-1))
        # Per-expert: stack + wrap. CACHE the GroupedTensor keyed on the source weights'
        # (id, _version) -- the FC2 weight GroupedTensor depends only on the weights (not
        # the splits), so it rebuilds only when the optimizer updates them.
        weight_params = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
        key = tuple((id(w), w._version) for w in weight_params)
        cache = getattr(self, "_fc2_w_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        weights = [maybe_dequantize(w, dtype) for w in weight_params]
        weight_data = torch.stack(weights, dim=0).contiguous().reshape(-1)
        gt = _make_grouped(weight_data)
        self._fc2_w_cache = (key, gt)
        return gt

    def _get_fc2_weight_2d(
        self,
        fc2_op: GroupedLinear,
        num_groups: int,
        fc2_weight_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return FC2 weight as a contiguous bf16 [G*H, I] tensor for the CUTLASS down kernel.

        Mirrors ``_get_fc1_weight_2d`` (which returns [G*2I, d]). The CUTLASS down kernel
        ``cutlass_grouped_down`` wants ONE contiguous [G*N, K] = [G*H, I] buffer (per-expert
        [H, I] = [out_features, in_features] blocks stacked over experts), used as the B operand
        (Y = A @ W2^T). ``fc2_weight_shape`` is (out_features=H, in_features=I).
        """
        out_features, in_features = fc2_weight_shape  # (H, I)
        if fc2_op.single_grouped_weight:
            # ALIGNED with _get_fc1_weight_2d / the reference: the packed [G, H, I] buffer is reused
            # as a VIEW -> [G*H, I]. No copy (the buffer is already contiguous).
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_weight=True."
                )
            w = maybe_dequantize(fc2_op.weight.rowwise_data, dtype)
            return w.view(num_groups * out_features, in_features)
        # Per-expert params: the kernel needs ONE contiguous [G*H, I] buffer, so we stack. CACHE the
        # stack keyed on the source weights' (id, _version): weights are constant across micro-batches,
        # so the copy only re-runs when the optimizer updates them in-place (bumps _version). Separate
        # cache from _get_fc2_weight (that one returns a GroupedTensor; this returns a raw 2D tensor).
        weight_params = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
        key = tuple((id(w), w._version) for w in weight_params)
        cache = getattr(self, "_fc2_w2d_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        weights = [maybe_dequantize(w, dtype) for w in weight_params]
        stacked = torch.stack(weights, dim=0).view(num_groups * out_features, in_features).contiguous()
        self._fc2_w2d_cache = (key, stacked)
        return stacked

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
        h_saved: Optional[torch.Tensor],
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
        # SwiGLU input h = FC1 up-proj [M, 2I] (swiglu.py:494). Design B: we SAVE h
        # (h_saved, computed above) so the fused backward fuses dA=dY@W2 + dswiglu
        # reading saved h -- NO recompute (the reference design, the fastest backward).
        # h_saved is None when no grad is required. See backward_fused_moe.py.
        activation_ctx.save_for_backward(h_saved, scales if input_requires_grad else None)
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
