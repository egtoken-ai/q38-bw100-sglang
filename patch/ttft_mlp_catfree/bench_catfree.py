"""Cat-free vs original 8-part MLP pipeline: actual-layer exactness, guard and timing check.

Constructs the actual Qwen3_5 GDN and Full-Attention decoder classes with their
actual compressed-tensors W8A8 INT8 modules (random weights, native weight
processing and dispatch). Runs only prepare_mlp AR + GemmaRMSNorm + MLP + out
reduction. Does not load the checkpoint or run attention/GDN.

Two helpers are loaded side by side from explicit files with verified SHA256:
  old = audited 4/8-parts helper (a946a802..., backup or still-installed copy)
  new = cat-free helper (1796bea1...)
Modes compared: baseline (single batch, native path), old8, new8_cat (new file,
flag-off path), new8_catfree. Root runs it with torchrun on two idle GPUs.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

OLD_SHA256 = 'a946a802abb6a6faeb84108e4ccb10db957bc553bfc573333d026cc681febbad'
NEW_SHA256 = '1796bea1dbde29d135a6f013f63a39331ca8f38583ae09c4b983144687e217a7'
PARTS = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path, expected):
    if sha256(path) != expected:
        raise RuntimeError(f'{path}: SHA256 differs from {expected}')
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def check(a, b):
    diff = a.float() - b.float()
    result = dict(max_abs=float(diff.abs().max()), mismatches=int(torch.count_nonzero(a != b)))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='/model')
    p.add_argument('--tokens', type=int, default=65536, choices=[65536])
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--old-helper', default=None,
                   help='default: <sglang layers>/qwen27b_mlp_pipeline.py.ttft_catfree_original, '
                        'else the installed helper if it is still the old one')
    p.add_argument('--new-helper', default=str(Path(__file__).resolve().parent / 'mlp_pipeline.py'))
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        raise RuntimeError('TP2 only')
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT', '0') != '0':
        raise RuntimeError('TORCH_NCCL_BLOCKING_WAIT must be 0 (helper guard)')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    os.environ['SGLANG_QWEN27B_MLP_PIPELINE'] = 'true'
    os.environ['SGLANG_MLP_DIAG_ENABLED'] = '0'
    os.environ['SGLANG_QWEN27B_MLP_PIPELINE_PARTS'] = str(PARTS)  # both helpers read it at import
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    server = ServerArgs(model_path=args.model, tp_size=2, dtype='bfloat16', device='cuda',
                        disable_cuda_graph=True, disable_piecewise_cuda_graph=True,
                        disable_custom_all_reduce=True)
    set_global_server_args_for_scheduler(server)
    from sglang.srt.distributed import (
        init_distributed_environment, initialize_model_parallel, get_tp_group, set_custom_all_reduce)
    set_custom_all_reduce(False)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend='nccl')
    initialize_model_parallel(tensor_model_parallel_size=2, backend='nccl')
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    initialize_dp_attention(server, SimpleNamespace(hidden_size=5120, dtype=torch.bfloat16))
    from sglang.srt.layers.moe.utils import initialize_moe_config
    initialize_moe_config(server)
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer, Qwen3_5AttentionDecoderLayer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    import sglang.srt.layers as layers_pkg

    layers_dir = Path(next(iter(layers_pkg.__path__))).resolve()
    installed_path = layers_dir / 'qwen27b_mlp_pipeline.py'
    backup_path = layers_dir / 'qwen27b_mlp_pipeline.py.ttft_catfree_original'
    old_path = Path(args.old_helper) if args.old_helper else (backup_path if backup_path.exists() else installed_path)
    old = load_module('qwen27b_mlp_pipeline_old_bench', old_path, OLD_SHA256)
    new = load_module('qwen27b_mlp_pipeline_catfree_bench', args.new_helper, NEW_SHA256)
    if old._PIPELINE_PARTS != PARTS or new._PIPELINE_PARTS != PARTS:
        raise RuntimeError('Both helpers must run with parts=8')

    config_json = json.loads((Path(args.model) / 'config.json').read_text())
    config = Qwen3_5TextConfig(**config_json['text_config'])
    config.torch_dtype = torch.bfloat16
    cpu_group = get_tp_group().cpu_group
    if rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def make_batch(tokens, batch_size, mode=ForwardMode.EXTEND):
        return SimpleNamespace(batch_size=batch_size, forward_mode=mode, spec_info=None,
            input_ids=torch.zeros(tokens, device='cuda', dtype=torch.int64),
            rms_quant_flag=False, attn_cp_metadata=None, tbo_children=None,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False))

    batch = make_batch(args.tokens, args.tokens // 4096)
    environment = {key: os.environ.get(key) for key in (
        'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS', 'NCCL_ALGO', 'NCCL_PROTO',
        'TORCH_NCCL_AVOID_RECORD_STREAMS', 'Q38_HIPBLASLT_SHAPES', 'SGLANG_QWEN27B_MLP_CATFREE',
        'HIP_VISIBLE_DEVICES')}
    report = dict(old_helper=str(old_path), old_sha256=OLD_SHA256, new_helper=str(args.new_helper),
                  new_sha256=NEW_SHA256, installed_helper_sha256=sha256(installed_path),
                  tokens=args.tokens, parts=PARTS, environment=environment, guard={}, layers=[])

    def write_report():
        if rank == 0:
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n')

    def gather_status(local, label):
        values = [None, None]
        dist.all_gather_object(values, local, group=cpu_group)
        if any(v['status'] != 'PASS' for v in values):
            report.setdefault('failure', label)
            write_report()
            raise RuntimeError(f'{label} failed: {values}')
        return values

    for cls, layer_id in ((Qwen3_5LinearDecoderLayer, 0), (Qwen3_5AttentionDecoderLayer, 3)):
        quant = CompressedTensorsConfig.from_config(config_json['compression_config'])
        with torch.device(f'cuda:{rank}'):
            layer = cls(config, layer_id=layer_id, quant_config=quant,
                        prefix=f'model.language_model.layers.{layer_id}')
        torch.manual_seed(20261015 + rank)
        for proj in (layer.mlp.gate_up_proj, layer.mlp.down_proj):
            if type(proj.scheme).__name__ != 'CompressedTensorsW8A8Int8':
                raise RuntimeError(f'Unexpected quantization scheme: {type(proj.scheme)}')
            proj.weight.data.copy_(torch.randint(-16, 16, proj.weight.shape, device='cuda', dtype=torch.int8))
            proj.weight_scale.data.fill_(.001)
            proj.quant_method.process_weights_after_loading(proj)
        torch.manual_seed(50001)
        norm_weight = (torch.randn(5120, device='cuda') * .1).bfloat16()
        layer.post_attention_layernorm.weight.data.copy_(norm_weight)
        layer.post_attention_layernorm.gemma_weight = norm_weight + 1
        torch.manual_seed(61000 + rank)
        source = torch.randn(args.tokens, 5120, device='cuda', dtype=torch.bfloat16)
        torch.manual_seed(62000)
        residual_source = torch.randn_like(source)
        x = torch.empty_like(source)
        r = torch.empty_like(source)

        def reset():
            x.copy_(source)
            r.copy_(residual_source)

        # ---- Guard / pass-through checks (no collective may run) ----
        local = dict(rank=rank)
        try:
            reset()
            reasons = dict(old=old.rejection_reason(layer, x, r, batch), new=new.rejection_reason(layer, x, r, batch))
            if reasons['old'] is not None or reasons['new'] is not None:
                raise AssertionError(f'M{args.tokens} bs16 must be accepted: {reasons}')
            cases = (('M4096_bs1', 4096, 1, ForwardMode.EXTEND),
                     ('M65536_bs1', args.tokens, 1, ForwardMode.EXTEND),
                     ('M65536_decode', args.tokens, 16, ForwardMode.DECODE))
            for name, tokens, bs, mode in cases:
                gx, gr = x[:tokens], r[:tokens]
                gb = make_batch(tokens, bs, mode)
                ro, rn = old.rejection_reason(layer, gx, gr, gb), new.rejection_reason(layer, gx, gr, gb)
                if ro is None or ro != rn:
                    raise AssertionError(f'{name}: rejection differs or accepted: old={ro!r} new={rn!r}')
                for label, fn in (('old', old.try_mlp_pipeline),
                                  ('new_catfree', lambda *a: new.run_mlp_pipeline(*a, catfree=True)),
                                  ('new_cat', lambda *a: new.run_mlp_pipeline(*a, catfree=False))):
                    if fn(layer, gx, gr, gb) is not None:
                        raise AssertionError(f'{name}: {label} did not pass through')
                torch.cuda.synchronize()
                if not (torch.equal(x, source) and torch.equal(r, residual_source)):
                    raise AssertionError(f'{name}: inputs mutated by a rejected call')
                local[name] = ro
            local['status'] = 'PASS'
        except Exception as exc:
            local['status'] = 'FAIL'
            local['error'] = f'{type(exc).__name__}: {exc}'
        report['guard'][cls.__name__] = gather_status(local, f'{cls.__name__} guard')

        def baseline():
            reset()
            h, res = layer.layer_communicator.prepare_mlp(x, r, batch)
            out = layer.mlp(h, should_allreduce_fusion=False, use_reduce_scatter=False)
            return layer.layer_communicator.postprocess_layer(out, res, batch)

        def runner(fn):
            def run():
                reset()
                out = fn(layer, x, r, batch)
                if out is None:
                    raise RuntimeError('Pipeline did not run: ' + str(new.rejection_reason(layer, x, r, batch)))
                return out
            return run

        modes = [('baseline', baseline),
                 ('old8', runner(old.try_mlp_pipeline)),
                 ('new8_cat', runner(lambda *a: new.run_mlp_pipeline(*a, catfree=False))),
                 ('new8_catfree', runner(lambda *a: new.run_mlp_pipeline(*a, catfree=True)))]
        row = dict(layer=cls.__name__, checks={}, timings={})
        report['layers'].append(row)
        # Keep only the two references resident (output, residual, in-place
        # normalized buffer); each candidate is compared right after it runs.
        references = {}
        for name, fn in modes:
            out = fn()
            torch.cuda.synchronize()
            actual = (out[0], out[1], x)
            for ref, expected in references.items():
                local = dict(rank=rank)
                try:
                    local['output'] = check(actual[0], expected[0])
                    local['residual'] = check(actual[1], expected[1])
                    local['normalized_inplace_buffer'] = check(actual[2], expected[2])
                    local['residual_changed_elements'] = int(torch.count_nonzero(expected[1] != residual_source))
                    if local['residual_changed_elements'] == 0:
                        raise AssertionError('Vacuous residual test')
                    local['status'] = 'PASS'
                except Exception as exc:
                    local['status'] = 'FAIL'
                    local['error'] = f'{type(exc).__name__}: {exc}'
                row['checks'][f'{name}_vs_{ref}'] = gather_status(local, f'{cls.__name__} {name} vs {ref}')
            if name in ('baseline', 'old8'):
                references[name] = tuple(v.clone() for v in actual)
            del out, actual
        del references

        timed = [m for m in modes if m[0] != 'new8_cat']
        samples = {name: [] for name, _ in timed}
        for _, fn in timed:
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            offset = repeat % len(timed)
            for name, fn in timed[offset:] + timed[:offset]:
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    out = fn()
                end.record()
                end.synchronize()
                del out
                local = dict(rank=rank, ms=start.elapsed_time(end) / args.iterations,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_increment_bytes=max(0, torch.cuda.max_memory_allocated() - before))
                measurements = [None, None]
                dist.all_gather_object(measurements, local, group=cpu_group)
                samples[name].append(dict(max_rank_ms=max(v['ms'] for v in measurements), ranks=measurements))
        for name in samples:
            row['timings'][name] = dict(
                samples=samples[name],
                median_ms=statistics.median(v['max_rank_ms'] for v in samples[name]),
                max_peak_increment_bytes=max(m['peak_increment_bytes'] for v in samples[name] for m in v['ranks']))
        t = row['timings']
        row['catfree_saving_vs_old8_ms'] = t['old8']['median_ms'] - t['new8_catfree']['median_ms']
        row['catfree_saving_vs_old8_pct'] = row['catfree_saving_vs_old8_ms'] / t['old8']['median_ms'] * 100
        row['catfree_peak_delta_vs_old8_bytes'] = (t['new8_catfree']['max_peak_increment_bytes']
                                                   - t['old8']['max_peak_increment_bytes'])
        write_report()
        if rank == 0:
            print(json.dumps({k: v for k, v in row.items() if k != 'checks'} | {
                'timings': {k: dict(median_ms=v['median_ms'], max_peak_increment_bytes=v['max_peak_increment_bytes'])
                            for k, v in t.items()}}), flush=True)
        del layer
        torch.cuda.empty_cache()
    report['status'] = 'PASS'
    write_report()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
