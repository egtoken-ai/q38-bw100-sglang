"""Validate page permutation, partial tails, causal alignment and opt-in guard."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from flash_attn import flash_attn_varlen_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ['SGLANG_KV_LAYOUT_HCU_FA'] = 'true'
    os.environ['SGLANG_QWEN27B_PACKED_PREFILL'] = 'true'
    path = Path(__file__).with_name('pack_paged_kv_to_varlen.py')
    spec = importlib.util.spec_from_file_location('packed_candidate', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.get_global_server_args = lambda: NS(pack_paged_kv_to_varlen='on', minimax_opt=False)
    torch.manual_seed(20261015)
    results = []
    for dtype in [torch.bfloat16]:
        for kv_lens, q_lens in [([129, 511], [63, 128]), ([64, 65, 257], [64, 1, 127])]:
            counts = [(n+63)//64 for n in kv_lens]
            count = sum(counts)
            kc = torch.randn(count+3, 2, 64, 256, device='cuda', dtype=dtype)
            vc = torch.randn(count+3, 2, 256, 64, device='cuda', dtype=dtype)
            page_ids = torch.randperm(count+3, device='cuda')[:count]
            table = torch.zeros(len(kv_lens), max(counts), device='cuda', dtype=torch.int32)
            keys, values = [], []
            offset = 0
            for b, (n, pages) in enumerate(zip(kv_lens, counts)):
                ids = page_ids[offset:offset+pages]
                table[b, :pages] = ids.int()
                keys.append(kc[ids].permute(0, 2, 1, 3).reshape(-1, 2, 256)[:n])
                values.append(vc[ids].permute(0, 3, 1, 2).reshape(-1, 2, 256)[:n])
                offset += pages
            lens = torch.tensor(kv_lens, dtype=torch.int32)
            k, v = mod.pack_paged_kv_to_varlen(kc, vc, table, lens, 64)
            torch.testing.assert_close(k, torch.cat(keys), atol=0, rtol=0)
            torch.testing.assert_close(v, torch.cat(values), atol=0, rtol=0)
            q = torch.randn(sum(q_lens), 12, 256, device='cuda', dtype=dtype)
            cq = torch.tensor([0]+q_lens, device='cuda', dtype=torch.int32).cumsum(0).int()
            ck = torch.tensor([0]+kv_lens, device='cuda', dtype=torch.int32).cumsum(0).int()
            actual = flash_attn_varlen_func(q, k, v, cq, ck, max(q_lens), max(kv_lens), causal=True)
            references = []
            offset = 0
            for n, m, key, val in zip(kv_lens, q_lens, keys, values):
                query = q[offset:offset+m].transpose(0, 1).float()
                key = key.transpose(0, 1).repeat_interleave(6, dim=0).float()
                val = val.transpose(0, 1).repeat_interleave(6, dim=0).float()
                mask = torch.arange(n, device='cuda')[None, :] <= (n-m+torch.arange(m, device='cuda')[:, None])
                ref = torch.nn.functional.scaled_dot_product_attention(query, key, val, attn_mask=mask)
                references.append(ref.transpose(0, 1));offset += m
            ref = torch.cat(references)
            torch.testing.assert_close(actual.float(), ref, atol=.02, rtol=.02)
            layer = NS(tp_q_head_num=12, tp_k_head_num=2, tp_v_head_num=2,
                       head_dim=256, v_head_dim=256, logit_cap=0.)
            batch = NS(seq_lens_cpu=lens, batch_size=len(kv_lens), mha_return_lse=False,
                       forward_mode=NS(is_extend=lambda: True), spec_info=None)
            metadata = NS(page_table=table, cu_seqlens_k=ck, max_seq_len_k=max(kv_lens))
            kwargs = dict(forward_batch=batch, metadata=metadata, layer=layer,
                          window_size=(-1,-1), sinks=None, key_cache=kc, value_cache=vc, page_size=64)
            assert mod.can_pack_paged_kv_to_varlen(**kwargs)
            os.environ['SGLANG_QWEN27B_PACKED_PREFILL'] = 'false'
            assert not mod.can_pack_paged_kv_to_varlen(**kwargs)
            os.environ['SGLANG_QWEN27B_PACKED_PREFILL'] = 'true'
            layer.head_dim = 128
            assert not mod.can_pack_paged_kv_to_varlen(**kwargs)
            layer.head_dim = 256;batch.spec_info = object()
            assert not mod.can_pack_paged_kv_to_varlen(**kwargs)
            results.append(dict(dtype=str(dtype), kv_lens=kv_lens, query_lens=q_lens,
                                max_abs_error=(actual.float()-ref).abs().max().item(), status='PASS'))
    Path(args.output).write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
