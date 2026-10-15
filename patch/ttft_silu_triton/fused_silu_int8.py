# SPDX-License-Identifier: Apache-2.0
"""BF16-rounded SiLU, BF16-rounded multiply, then deployed INT8 quantization.

One program per row keeps the activation in registers. The power-of-two tail is
masked; no BF16 activation tensor is stored to global memory. Independent screen
only: this helper is not installed into the model or automatically selected.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _silu_mul_int8(
    X, Q, S,
    STRIDE_X: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * STRIDE_X + cols, mask, other=0).to(tl.float32)
    up = tl.load(X + row * STRIDE_X + N + cols, mask, other=0).to(tl.float32)
    # The vendor BF16 activation returns SiLU in BF16 before multiplying up.
    # probe_activation.py verified this ordering exactly on 4096-token inputs.
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    activation = silu * up
    # The multiply is separately materialized in BF16 before INT8 amax/round.
    activation = activation.to(tl.bfloat16).to(tl.float32)
    activation = tl.where(mask, activation, 0.0)
    absmax = tl.maximum(tl.max(tl.abs(activation), axis=0), 1e-10)
    scale = absmax / 127
    # Match lightop/_lmslim_native/layers/gemm/int8_utils.py exactly:
    # nearest-even nearbyint, no explicit saturation/clamp, [-127,127] for
    # finite input because the row's absmax defines the dynamic range.
    quant = libdevice.nearbyint(activation * (127 / absmax)).to(tl.int8)
    tl.store(Q + row * N + cols, quant, mask)
    tl.store(S + row, scale)


def fused_silu_int8(x, *, num_warps=8, return_kernel=False):
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("The candidate accepts only device BF16 input")
    if x.ndim != 2 or x.shape[1] != 17408 or x.stride(1) != 1:
        raise ValueError("Expected [tokens,17408] with contiguous last dimension")
    if num_warps not in (4, 8, 16):
        raise ValueError("Bounded sweep supports num_warps=4,8,16 only")
    rows, width = x.shape
    n = width // 2
    q = torch.empty(rows, n, device=x.device, dtype=torch.int8)
    scales = torch.empty(rows, 1, device=x.device, dtype=torch.float32)
    compiled = _silu_mul_int8[(rows,)](
        x, q, scales, x.stride(0), n, triton.next_power_of_2(n),
        num_warps=num_warps, num_stages=1, enable_fp_fusion=False,
    )
    if return_kernel:
        return (q, scales), compiled
    return q, scales
