"""BW100 TTFT token-tiled replacement for ``_triton_mrope_forward_fused``.

Installed next to ``rotary_embedding/triton_kernels.py``. The vendor kernel runs
one program per token over a padded [pad_n_qh, pad_hd/2] tile; for Qwen3.8-27B
TP2 (12 Q heads padded to 16, head 256 -> 128 lanes, rotary 64 -> 32 lanes) only
384 of 2048 Q lanes do work. This kernel tiles BLOCK_T tokens x the real
rotary half (PAD_HALF lanes) and loops over the real heads.

Per element arithmetic is unchanged: the same interleaved t/h/w masks, the
same three masked cos (sin) loads with ``other=0`` summed as ``t + h + w`` in
the cache dtype, the same ``.to(sin_row.dtype)`` of q/k, and the same
``x1*cos - x2*sin`` / ``x2*cos + x1*sin`` expressions stored without a cast.
Only the neox, interleaved (non-GLM) layout is taken; anything else falls
back to the vendor launch. Set ``SGLANG_QWEN27B_MROPE_TILED=0`` before import
to always use the vendor kernel. Bitwise equality is gated by bench_mrope.py.
"""
import os

import torch
import triton
import triton.language as tl

ENV_FLAG = "SGLANG_QWEN27B_MROPE_TILED"
ENABLED = os.environ.get(ENV_FLAG, "1").strip().lower() not in ("0", "false", "off", "no")

DEFAULT_CONFIG = {"BLOCK_T": 16, "num_warps": 4, "PER_HEAD": False}
# The next line is rewritten by install.py from config.json (exactly one match).
CONFIG = {'BLOCK_T': 8, 'num_warps': 4, 'PER_HEAD': False}  # TTFT_MROPE_CONFIG (installed)


@triton.jit
def _rotate_head(
    base_ptr,
    row_off,  # [BLOCK_T, 1] int64 token offsets
    head,
    lanes,  # [1, PAD_HALF]
    m2,  # [BLOCK_T, PAD_HALF]
    cos_row,
    sin_row,
    hd: tl.constexpr,
    half_rd: tl.constexpr,
):
    p1 = base_ptr + row_off + head * hd + lanes
    x1 = tl.load(p1, mask=m2, other=0).to(sin_row.dtype)
    p2 = p1 + half_rd
    x2 = tl.load(p2, mask=m2, other=0).to(sin_row.dtype)
    tl.store(p1, x1 * cos_row - x2 * sin_row, mask=m2)
    tl.store(p2, x2 * cos_row + x1 * sin_row, mask=m2)


@triton.jit
def _triton_mrope_forward_tiled(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    num_tokens,
    q_stride,
    k_stride,
    positions_stride,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    PAD_HALF: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PER_HEAD: tl.constexpr,
):
    half_rd = rd // 2
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < num_tokens
    t = tl.load(positions_ptr + 0 * positions_stride + offs_t, mask=tmask, other=0)
    h = tl.load(positions_ptr + 1 * positions_stride + offs_t, mask=tmask, other=0)
    w = tl.load(positions_ptr + 2 * positions_stride + offs_t, mask=tmask, other=0)
    t_cos = cos_sin_cache_ptr + t * rd
    h_cos = cos_sin_cache_ptr + h * rd
    w_cos = cos_sin_cache_ptr + w * rd
    t_sin = t_cos + half_rd
    h_sin = h_cos + half_rd
    w_sin = w_cos + half_rd
    cos_offsets = tl.arange(0, PAD_HALF)
    # Vendor interleaved (non-GLM) masks, evaluated on the same lane indices.
    h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
    w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
    t_mask = ~(h_mask | w_mask)
    lane_ok = cos_offsets < half_rd
    m2 = tmask[:, None] & lane_ok[None, :]
    lanes = cos_offsets[None, :]
    t_cos_row = tl.load(t_cos[:, None] + lanes, mask=m2 & t_mask[None, :], other=0)
    t_sin_row = tl.load(t_sin[:, None] + lanes, mask=m2 & t_mask[None, :], other=0)
    h_cos_row = tl.load(h_cos[:, None] + lanes, mask=m2 & h_mask[None, :], other=0)
    h_sin_row = tl.load(h_sin[:, None] + lanes, mask=m2 & h_mask[None, :], other=0)
    w_cos_row = tl.load(w_cos[:, None] + lanes, mask=m2 & w_mask[None, :], other=0)
    w_sin_row = tl.load(w_sin[:, None] + lanes, mask=m2 & w_mask[None, :], other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    offs64 = offs_t.to(tl.int64)[:, None]
    q_row = offs64 * q_stride
    k_row = offs64 * k_stride
    if PER_HEAD:
        head = tl.program_id(1)
        if head < n_qh:
            _rotate_head(q_ptr, q_row, head, lanes, m2, cos_row, sin_row, hd, half_rd)
        else:
            _rotate_head(k_ptr, k_row, head - n_qh, lanes, m2, cos_row, sin_row, hd, half_rd)
    else:
        for hq in tl.static_range(n_qh):
            _rotate_head(q_ptr, q_row, hq, lanes, m2, cos_row, sin_row, hd, half_rd)
        for hk in tl.static_range(n_kh):
            _rotate_head(k_ptr, k_row, hk, lanes, m2, cos_row, sin_row, hd, half_rd)


def try_mrope_tiled(
    q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim,
    mrope_interleaved, mrope_interleaved_glm, is_neox_style, config=None,
):
    """Run the tiled kernel in place and return True, or return False (vendor path)."""
    if not ENABLED:
        return False
    if not (is_neox_style and mrope_interleaved and not mrope_interleaved_glm):
        return False
    if (
        q.dim() != 2 or k.dim() != 2 or positions.dim() != 2 or positions.shape[0] != 3
        or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16
        or cos_sin_cache.dtype != torch.bfloat16 or cos_sin_cache.dim() != 2
        or cos_sin_cache.stride(1) != 1 or cos_sin_cache.stride(0) != rotary_dim
        or q.stride(1) != 1 or k.stride(1) != 1 or positions.stride(1) != 1
        or rotary_dim % 2 or rotary_dim > head_size
        or q.shape[1] % head_size or k.shape[1] % head_size
        or q.shape[0] != k.shape[0] or positions.shape[1] != q.shape[0]
        or len(mrope_section) != 3
    ):
        return False
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return False
    cfg = CONFIG if config is None else config
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    block_t = cfg["BLOCK_T"]
    per_head = bool(cfg["PER_HEAD"])
    grid = (triton.cdiv(num_tokens, block_t), (n_qh + n_kh) if per_head else 1)
    _triton_mrope_forward_tiled[grid](
        q, k, cos_sin_cache, positions,
        num_tokens, q.stride(0), k.stride(0), positions.stride(0),
        n_qh, n_kh, head_size, rotary_dim, triton.next_power_of_2(rotary_dim // 2),
        mrope_section[1], mrope_section[2], block_t, per_head,
        num_warps=cfg["num_warps"],
    )
    return True
