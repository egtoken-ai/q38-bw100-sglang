"""Direct HCU paged BF16 K/V -> contiguous varlen buffers, without page copies."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(K, V, TABLE, CU, OUT_K, OUT_V,
          KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, KS3: tl.constexpr,
          VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr, VS3: tl.constexpr,
          TS0: tl.constexpr, TS1: tl.constexpr,
          H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    seq = tl.program_id(1)
    start = tl.load(CU + seq)
    length = tl.load(CU + seq + 1) - start
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offset // (H * D)
    head = offset // D % H
    dim = offset % D
    valid = token < length
    page = tl.load(TABLE + seq * TS0 + (token // PAGE) * TS1, valid, 0).to(tl.int64)
    in_page = token % PAGE
    kval = tl.load(K + page * KS0 + head * KS1 + in_page * KS2 + dim * KS3, valid, 0)
    vval = tl.load(V + page * VS0 + head * VS1 + dim * VS2 + in_page * VS3, valid, 0)
    dest = start.to(tl.int64) * H * D + offset
    tl.store(OUT_K + dest, kval, valid)
    tl.store(OUT_V + dest, vval, valid)


def fused_pack(key, value, page_table, cu_seqlens, total_tokens, max_seq_len,
               *, block=1024):
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError('Only validated BF16 cache is supported')
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError('Expected HCU K[P,H,T,D] and V[P,H,D,T]')
    pages, heads, page_size, dim = key.shape
    if (heads, page_size, dim) != (2, 64, 256) or tuple(value.shape) != (pages, heads, dim, page_size):
        raise ValueError('Expected Qwen27B TP2 H2/D256/page64 layout')
    if not cu_seqlens.is_contiguous() or cu_seqlens.ndim != 1:
        raise ValueError('cu_seqlens must be contiguous')
    if cu_seqlens.dtype not in (torch.int32, torch.int64) or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected integer offsets and page IDs')
    if not key.is_cuda or any(t.device != key.device for t in (value, page_table, cu_seqlens)):
        raise ValueError('All buffers must be on the same GPU')
    batch = cu_seqlens.numel() - 1
    if batch < 1 or page_table.ndim != 2 or page_table.shape[0] < batch:
        raise ValueError('Invalid batch metadata shape')
    if total_tokens < 1 or max_seq_len < 1 or page_table.shape[1] < triton.cdiv(max_seq_len, page_size):
        raise ValueError('Invalid token count or insufficient page table capacity')
    out_k = torch.empty((total_tokens, heads, dim), device=key.device, dtype=key.dtype)
    out_v = torch.empty_like(out_k)
    _pack[(triton.cdiv(max_seq_len * heads * dim, block), batch)](
        key, value, page_table, cu_seqlens, out_k, out_v,
        *key.stride(), *value.stride(), *page_table.stride(),
        heads, dim, page_size, block, num_warps=4)
    return out_k, out_v
