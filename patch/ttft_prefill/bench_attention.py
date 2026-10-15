"""Compare the deployed Triton extend attention against tiled/FA alternatives."""
import argparse
import gc
import json
from pathlib import Path
import statistics

import torch
import flash_attn
import sglang.srt.layers.attention.triton_ops.extend_attention as ext


def timed(fn, repetitions=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    original_config = ext._get_block_sizes_for_extend_attention
    results = []
    # Head geometry is per TP rank: Q=12, KV=2, D=256.
    for batch, qlen, prefix in [(1, 128, 384), (2, 4096, 0),
                                (1, 8192, 0), (1, 8192, 24576), (1, 8192, 57344)]:
        total = qlen + prefix
        q = torch.randn(batch * qlen, 12, 256, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch, total, 2, 256, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ke = k[:, prefix:].reshape(-1, 2, 256).contiguous()
        ve = v[:, prefix:].reshape(-1, 2, 256).contiguous()
        kb, vb = k.reshape(-1, 2, 256), v.reshape(-1, 2, 256)
        qo = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * qlen
        kp = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * prefix
        kc = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * total
        indices = torch.cat([torch.arange(i*total, i*total+prefix, device='cuda', dtype=torch.int64)
                             for i in range(batch)])
        output = torch.empty_like(q)

        def triton_fn():
            ext.extend_attention_fwd(q, ke, ve, output, kb, vb, qo, kp, indices,
                                     None, True, None, qlen, 1.0, 1.0)
            return output

        ext._get_block_sizes_for_extend_attention = original_config
        reference = triton_fn().clone()
        if qlen == 128:
            qf = q.view(batch, qlen, 12, 256).transpose(1, 2).float()
            kf = k.transpose(1, 2).repeat_interleave(6, dim=1).float()
            vf = v.transpose(1, 2).repeat_interleave(6, dim=1).float()
            mask = torch.arange(total, device='cuda')[None, :] <= (
                prefix + torch.arange(qlen, device='cuda')[:, None])
            independent = torch.nn.functional.scaled_dot_product_attention(
                qf, kf, vf, attn_mask=mask).transpose(1, 2).reshape_as(q)
            torch.testing.assert_close(reference.float(), independent, atol=.02, rtol=.02)

        choices = [('baseline', original_config)]
        for bm, bn, warps in [(32, 64, 4), (64, 32, 4), (64, 64, 8),
                              (64, 128, 4), (128, 64, 4), (128, 64, 8), (128, 128, 8)]:
            choices.append((f'tile-{bm}-{bn}-w{warps}',
                            lambda lq, lv, bm=bm, bn=bn, w=warps: (256, 0, 256, bm, bn, w)))
        choices.append(('flash_attn_varlen', None))
        for name, config in choices:
            row = dict(batch=batch, query_len=qlen, prefix_len=prefix, candidate=name)
            try:
                if config is not None:
                    ext._get_block_sizes_for_extend_attention = config
                    fn = triton_fn
                else:
                    fn = lambda: flash_attn.flash_attn_varlen_func(
                        q, kb, vb, qo, kc, qlen, total, causal=True)
                actual = fn()
                delta = (actual.float() - reference.float()).abs()
                row['max_abs_error'] = delta.max().item()
                row['relative_l2_error'] = (delta.norm() / reference.float().norm()).item()
                torch.testing.assert_close(actual, reference, atol=.02, rtol=.02)
                row['median_ms'] = timed(fn)
                row['status'] = 'PASS'
            except Exception as exc:
                row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:600])
            results.append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
        del q, k, v, ke, ve, kb, vb, output, reference
        gc.collect()
        torch.cuda.empty_cache()
    ext._get_block_sizes_for_extend_attention = original_config


if __name__ == '__main__':
    main()
