# SPDX-License-Identifier: Apache-2.0
"""Standalone Gemma residual/norm/INT8 boundary candidate, not a serving patch.

Only the residual-present HCU path is screened. The deployed residual-free path
uses a precomputed BF16 gamma and has a different rounding contract.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gemma_norm_quant(
    X, R, W, Q, S, Y,
    N: tl.constexpr, EPS: tl.constexpr,
    ROUND_RESIDUAL_FOR_NORM: tl.constexpr,
    EMIT_NORM: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + row*N + col, valid, other=0).to(tl.float32)
    residual = tl.load(R + row*N + col, valid, other=0).to(tl.float32)
    combined = x + residual
    stored_residual = combined.to(tl.bfloat16)
    tl.store(R + row*N + col, stored_residual, valid)
    if ROUND_RESIDUAL_FOR_NORM:
        combined = stored_residual.to(tl.float32)
    combined = tl.where(valid, combined, 0.0)
    variance = tl.sum(combined * combined, axis=0) / N
    inv_rms = tl.rsqrt(variance + EPS)
    # Do not precompute (weight + 1) in BF16: load the original gamma offset,
    # promote to FP32, then add 1. The normalized result is rounded to BF16
    # before the quantizer's amax, scale, and nearest-even integer rounding.
    gamma = tl.load(W + col, valid, other=0).to(tl.float32) + 1.0
    normalized = ((combined * inv_rms) * gamma).to(tl.bfloat16)
    if EMIT_NORM:
        tl.store(Y + row*N + col, normalized, valid)
    value = tl.where(valid, normalized.to(tl.float32), 0.0)
    absmax = tl.maximum(tl.max(tl.abs(value), axis=0), 1e-10)
    scale = absmax / 127
    quant = libdevice.nearbyint(value * (127 / absmax)).to(tl.int8)
    tl.store(Q + row*N + col, quant, valid)
    tl.store(S + row, scale)


def fused_norm_quant(x, residual, weight, *, eps=1e-6, emit_norm=False,
                     round_residual_for_norm=False, num_warps=4):
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Only BF16 inputs, residual, and original gamma offset are supported")
    if not x.is_cuda or residual.device != x.device or weight.device != x.device:
        raise ValueError("All inputs must be on the same GPU")
    if x.ndim != 2 or x.shape[1] != 5120 or residual.shape != x.shape or weight.shape != (5120,):
        raise ValueError("Restricted to contiguous [tokens,5120] residual boundary")
    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Noncontiguous input is outside this candidate's scope")
    if x.data_ptr() == residual.data_ptr():
        raise ValueError("Aliased input/residual is outside this candidate's scope")
    if num_warps not in (4, 8):
        raise ValueError("Bounded candidate uses 4 or 8 warps")
    rows, n = x.shape
    quant = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(rows, 1, device=x.device, dtype=torch.float32)
    norm = torch.empty_like(x) if emit_norm else None
    _gemma_norm_quant[(rows,)](
        x, residual, weight, quant, scale, norm if norm is not None else x,
        n, eps, round_residual_for_norm, emit_norm, triton.next_power_of_2(n),
        num_warps=num_warps, num_stages=1, enable_fp_fusion=False,
    )
    return quant, scale, residual, norm
