# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang PR #38806, head 62e3bdfb899cf158d1aecd2c070d3f1b3a4264da.
# Source: https://github.com/sgl-project/sglang/pull/38806
# Modifications: BF16/TP2 geometry guard, configurable launch for measured tuning.
import torch
import triton
import triton.language as tl



@triton.jit
def fused_gdn_prefill_prepare_kernel(
    q,
    k,
    v,
    g,
    beta_output,
    mixed_qkv,
    A_log,
    a,
    b,
    dt_bias,
    eps,
    MIXED_QKV_STRIDE_T: tl.constexpr,
    MIXED_QKV_STRIDE_D: tl.constexpr,
    A_STRIDE_T: tl.constexpr,
    B_STRIDE_T: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    NUM_QK_HEADS_POW2: tl.constexpr,
    HEAD_QK_POW2: tl.constexpr,
    NUM_V_HEADS_POW2: tl.constexpr,
    V_BLOCK: tl.constexpr,
):
    i_t = tl.program_id(0)
    row = mixed_qkv + i_t * MIXED_QKV_STRIDE_T

    qk_dim: tl.constexpr = NUM_QK_HEADS * HEAD_QK
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V

    # [NUM_QK_HEADS_POW2, HEAD_QK_POW2] so the reduction runs along the head dim
    # and every head of this token is normalized in one pass. Both axes are
    # padded because tl.arange needs a power of two, and num_qk_heads is
    # cdiv(num_k_heads, attn_tp_size), which need not be one.
    head = tl.arange(0, NUM_QK_HEADS_POW2)[:, None]
    dim = tl.arange(0, HEAD_QK_POW2)[None, :]
    inner = (head < NUM_QK_HEADS) & (dim < HEAD_QK)
    flat = head * HEAD_QK + dim

    b_q = tl.load(row + flat * MIXED_QKV_STRIDE_D, mask=inner, other=0.0).to(tl.float32)
    # Divide by sqrt rather than multiplying by the reciprocal, matching the
    # form l2norm_fwd_kernel uses. The reduction block shape still differs, so
    # this lands within an ulp of it rather than bit-exactly on it.
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q, axis=1) + eps)[:, None]
    tl.store(q + i_t * qk_dim + flat, b_q.to(q.dtype.element_ty), mask=inner)

    b_k = tl.load(row + (qk_dim + flat) * MIXED_QKV_STRIDE_D, mask=inner, other=0.0).to(
        tl.float32
    )
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, axis=1) + eps)[:, None]
    tl.store(k + i_t * qk_dim + flat, b_k.to(k.dtype.element_ty), mask=inner)

    v_off = tl.arange(0, V_BLOCK)
    v_mask = v_off < v_dim
    b_v = tl.load(row + (2 * qk_dim + v_off) * MIXED_QKV_STRIDE_D, mask=v_mask)
    tl.store(v + i_t * v_dim + v_off, b_v, mask=v_mask)

    # Match fused_gdn_gating_kernel, including its bf16/fp16 beta rounding.
    gate_head = tl.arange(0, NUM_V_HEADS_POW2)
    gate_mask = gate_head < NUM_V_HEADS
    gate_off = i_t * NUM_V_HEADS + gate_head
    b_a_log = tl.load(A_log + gate_head, mask=gate_mask)
    b_a = tl.load(a + i_t * A_STRIDE_T + gate_head, mask=gate_mask)
    b_b = tl.load(b + i_t * B_STRIDE_T + gate_head, mask=gate_mask)
    b_dt_bias = tl.load(dt_bias + gate_head, mask=gate_mask)
    x = b_a.to(tl.float32) + b_dt_bias.to(tl.float32)
    softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    b_g = -tl.exp(b_a_log.to(tl.float32)) * softplus_x
    tl.store(g + gate_off, b_g.to(g.dtype.element_ty), mask=gate_mask)
    b_beta = tl.sigmoid(b_b.to(tl.float32))
    tl.store(beta_output + gate_off, b_beta.to(b.dtype.element_ty), mask=gate_mask)


def fused_gdn_prefill_prepare(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    num_qk_heads: int,
    num_v_heads: int,
    head_qk: int,
    head_v: int,
    eps: float = 1e-6,
    num_warps: int = 4,
):
    """Split/normalize QKV and prepare GDN gates in a single launch.

    Inputs may be the strided post-conv QKV and BA projection views. The caller
    must disable the chunk kernel's Q/K norm because it is already applied.
    """
    assert mixed_qkv.dtype == a.dtype == b.dtype == torch.bfloat16
    assert num_qk_heads == 8 and num_v_heads == 24 and head_qk == head_v == 128
    assert mixed_qkv.ndim == 2 and mixed_qkv.shape[1] == 5120
    assert A_log.numel() == dt_bias.numel() == 24
    assert num_warps in (4, 8, 16)
    seq_len = mixed_qkv.shape[0]
    assert a.shape == b.shape == (seq_len, num_v_heads)
    assert a.stride(1) == b.stride(1) == 1
    q = torch.empty(
        (1, seq_len, num_qk_heads, head_qk),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty_like(q)
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    g = torch.empty((1, seq_len, num_v_heads), dtype=torch.float32, device=a.device)
    beta_output = torch.empty_like(g)
    if seq_len == 0:
        return q, k, v, g, beta_output

    fused_gdn_prefill_prepare_kernel[(seq_len,)](
        q,
        k,
        v,
        g,
        beta_output,
        mixed_qkv,
        A_log,
        a,
        b,
        dt_bias,
        eps,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        a.stride(0),
        b.stride(0),
        num_qk_heads,
        num_v_heads,
        head_qk,
        head_v,
        NUM_QK_HEADS_POW2=triton.next_power_of_2(num_qk_heads),
        HEAD_QK_POW2=triton.next_power_of_2(head_qk),
        NUM_V_HEADS_POW2=triton.next_power_of_2(num_v_heads),
        V_BLOCK=triton.next_power_of_2(num_v_heads * head_v),
        num_warps=num_warps,
        num_stages=3,
    )
    return q, k, v, g, beta_output
