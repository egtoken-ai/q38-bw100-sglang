"""Bit-exact cache packing check and paired timing; not a model benchmark."""
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
from fused_kv_pack import fused_pack


def original(key, value, table, lengths):
    ks, vs = [], []
    for i, length in enumerate(lengths):
        pages = table[i, :(length + 63)//64].to(torch.long)
        k = key.index_select(0, pages).permute(0, 2, 1, 3).reshape(-1, 2, 256)
        v = value.index_select(0, pages).permute(0, 3, 1, 2).reshape(-1, 2, 256)
        ks.append(k[:length])
        vs.append(v[:length])
    return torch.cat(ks), torch.cat(vs)


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)/20)
    return {'median_ms': statistics.median(samples), 'samples_ms': samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    rows = []
    for lengths in ([1, 63, 64, 65, 129], [4096], [4096]*8, [4096]*64, [65536], [8192, 16385]):
        page_counts = [(n+63)//64 for n in lengths]
        pages = sum(page_counts) + 5
        key = torch.randn(pages, 2, 64, 256, device='cuda', dtype=torch.bfloat16)
        value = torch.randn(pages, 2, 256, 64, device='cuda', dtype=torch.bfloat16)
        permutation = torch.randperm(pages, device='cuda', dtype=torch.int32)
        # Noncontiguous table catches accidentally ignored row/column strides.
        table = torch.empty(len(lengths), max(page_counts)*2, device='cuda', dtype=torch.int32)[:, ::2]
        table.fill_(-999)
        cursor = 0
        for i, count in enumerate(page_counts):
            table[i, :count] = permutation[cursor:cursor+count]
            cursor += count
        cu = torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
        ref = original(key, value, table, lengths)
        for block in (1024, 4096):
            fn = lambda: fused_pack(key, value, table, cu, sum(lengths), max(lengths), block=block)
            actual = fn()
            exact = [torch.equal(a, b) for a, b in zip(actual, ref)]
            assert all(exact), f'Packing mismatch: {lengths}, block{block}'
            rows.append({'lengths': lengths, 'block': block, 'k_v_exact': exact,
                         'baseline': measure(lambda: original(key, value, table, lengths)),
                         'fused': measure(fn)})
        print(json.dumps(rows[-2:]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
