#!/usr/bin/env python3
"""Bounded exact-output screen of actual W8A8 module entry, without installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parent


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def worker(args):
    import torch
    from compressed_tensors.quantization import QuantizationStrategy
    from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as original
    import gemm_dispatch
    from gemm_dispatch import patch_source, supported_device, try_legacy_mm
    # Candidate uses its eventual package-qualified import without installing it.
    sys.modules[gemm_dispatch.HELPER_MODULE] = gemm_dispatch
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    assert supported_device(torch.cuda.current_device()), 'requires actual gfx936 / 64 CU'
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    source = Path(original.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest['sha256'], 'source SHA mismatch'
    candidate = types.ModuleType('q38_w8a8_candidate')
    exec(compile(patch_source(source.decode()), '<q38_w8a8_candidate>', 'exec'), candidate.__dict__)
    # Suppress only process_weights' JSON I/O and optional tuning warmups.
    # The real class create/process/apply code and tensor transformations run.
    cache = types.SimpleNamespace(weight_shapes=[], triton_json_dict={},
                                  get_w8a8json_name=lambda n, k: '',
                                  get_triton_cache=lambda *a: {}, gen_model_json=lambda: None)
    original.W8A8_TRITONJSON = candidate.W8A8_TRITONJSON = cache
    scheme = original.CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
    trial = candidate.CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
    scheme.w8a8_strategy = trial.w8a8_strategy = 1
    m, n, k = args.shape
    os.environ['Q38_HIPBLASLT_SHAPES'] = f'{m}x{n}x{k}'
    layer = torch.nn.Module()
    with torch.device('cuda'):
        scheme.create_weights(layer, [n], k, torch.bfloat16, lambda *a, **kw: None)
    with torch.no_grad():
        layer.weight.copy_(torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8))
        layer.weight_scale.copy_(torch.rand(n, 1, device='cuda') * .01 + .0001)
    source_ptr = layer.weight.data_ptr()
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (k, n) and layer.weight.stride() == (1, k)
    assert layer.weight.data_ptr() == source_ptr and layer.weight.t().is_contiguous()
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    aq, sa = original.per_token_quant_int8(x)
    assert try_legacy_mm(trial, layer, x[:1], aq[:1], sa[:1], None) is None, 'decode fallback'
    assert try_legacy_mm(trial, layer, x, aq, sa, torch.empty(n, device='cuda')) is None, 'bias fallback'
    assert (m, 5120, 8704) not in __import__('gemm_dispatch').SHAPES, 'down excluded'
    fns = {'asm': lambda: scheme.apply_weights(layer, x, None),
           'candidate': lambda: trial.apply_weights(layer, x, None)}
    reference = fns['asm']()
    from lightop import gemm_ops
    native_fn = gemm_ops.hipblaslt_w8a8_gemm
    hits = [0]
    def counted_native(*a, **kw):
        hits[0] += 1
        return native_fn(*a, **kw)
    gemm_ops.hipblaslt_w8a8_gemm = counted_native
    try:
        actual = fns['candidate']()
    finally:
        gemm_ops.hipblaslt_w8a8_gemm = native_fn
    assert hits[0] == 1, 'candidate must reach the native hipBLASLt entry exactly once'
    torch.cuda.synchronize()
    assert actual.dtype == reference.dtype == torch.bfloat16
    exact = torch.equal(actual, reference)
    row = dict(shape=args.shape, status='FAIL', full_output_exact=exact, native_dispatch_hits=hits[0],
               source_sha256=manifest['sha256'], output_elements=actual.numel(),
               actual_cus=torch.cuda.get_device_properties(0).multi_processor_count)
    if not exact:
        max_error = max((actual[i:i+128].float()-reference[i:i+128].float()).abs().max().item()
                        for i in range(0, m, 128))
        row['max_abs_error'] = max_error
        write(args.output, row)
        raise AssertionError('Full module output is not bitwise identical; no tolerance relaxation')
    # Independent CPU exact integer accumulation; float scale rounding can differ
    # from the native epilogue, so this is reported, not used to waive ASM exact.
    cpu = ((aq[:8].cpu().double() @ layer.weight[:, :16].cpu().double())
           * sa[:8].cpu().double() * layer.weight_scale[:16].cpu().double().t()).to(torch.bfloat16)
    row['cpu_slice_max_abs'] = (reference[:8,:16].cpu().float()-cpu.float()).abs().max().item()
    torch.testing.assert_close(reference[:8,:16].cpu(), cpu, rtol=.02, atol=.01)
    del actual, reference
    for _ in range(4):
        fns['asm'](); fns['candidate']()
    torch.cuda.synchronize()
    samples = {name: [] for name in fns}
    # Alternating AB/BA rounds, multiple real module calls per timed interval.
    for round_id in range(8):
        for name in (('asm', 'candidate') if round_id % 2 == 0 else ('candidate', 'asm')):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3): fns[name]()
            end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end) / 3)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    improvements = [(a-b)/a*100 for a,b in zip(samples['asm'], samples['candidate'])]
    row.update(status='PASS', samples_ms=samples, median_ms=medians,
               improvement_pct=(medians['asm']-medians['candidate'])/medians['asm']*100,
               paired_improvement_pct=improvements,
               stable_faster=all(v > 0 for v in improvements))
    write(args.output, row)
    print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--shape', nargs=3, type=int)
    p.add_argument('--worker-timeout', type=float, default=35)
    p.add_argument('--total-timeout', type=float, default=450)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--same-process', action='store_true',
                   help='Reuse costly CPU imports for audited native kernels; abort on first failure')
    args = p.parse_args()
    shapes = [(m,n,k) for m in (4096,8192,16384,32768)
              for n,k in ((17408,5120),(8192,5120),(5120,3072))]
    if args.dry_run:
        print(json.dumps(dict(shapes=shapes, total_timeout=args.total_timeout,
                             gpu_executed=False, install_enabled=False)))
        return
    if args.worker:
        assert tuple(args.shape) in shapes
        worker(args)
        return
    assert 0 < args.worker_timeout <= 35 and 0 < args.total_timeout <= 450
    result = dict(scope='Module microbenchmark only; no TTFT/quality acceptance', rows=[], completed=False)
    result['execution_mode'] = 'same_process_native_only' if args.same_process else 'isolated_workers'
    write(args.output, result)
    start = time.monotonic()
    for shape in shapes:
        remaining = args.total_timeout - (time.monotonic()-start)
        if remaining <= 1:
            result['stop_reason'] = 'total deadline'
            break
        name = 'x'.join(map(str,shape))
        output = str(Path(args.output).with_suffix('')) + '-' + name + '.json'
        Path(output).unlink(missing_ok=True)  # Never reuse an earlier worker result.
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--shape', *map(str,shape), '--output', output]
        try:
            if args.same_process:
                # No kernel compilation/config search here: both native APIs
                # already ran on this device. An external timeout bounds the
                # whole process; a failed device check stops subsequent work.
                worker(argparse.Namespace(shape=shape, output=output))
                row = json.loads(Path(output).read_text())
                row['returncode'] = 0
                import gc
                gc.collect()
            else:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      timeout=min(args.worker_timeout,remaining), text=True)
                Path(output+'.log').write_text(proc.stdout)
                row = json.loads(Path(output).read_text()) if Path(output).exists() else dict(shape=shape,status='FAIL')
                row['returncode'] = proc.returncode
                if proc.returncode: row['status']='FAIL'
        except subprocess.TimeoutExpired as exc:
            Path(output+'.log').write_bytes(exc.stdout or b'')
            row = dict(shape=shape,status='TIMEOUT')
        except Exception as exc:
            result['stop_reason'] = f'{type(exc).__name__}: {exc}'
            result['rows'].append(dict(shape=shape,status='FAIL',error=str(exc)))
            write(args.output,result)
            raise
        result['rows'].append(row)
        write(args.output,result)
        print(json.dumps(row),flush=True)
    result['completed'] = len(result['rows']) == len(shapes)
    result['all_exact_passed'] = result['completed'] and all(r['status']=='PASS' and r.get('full_output_exact') for r in result['rows'])
    result['elapsed_s'] = time.monotonic()-start
    write(args.output,result)


if __name__ == '__main__':
    main()
