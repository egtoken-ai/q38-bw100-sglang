"""Snapshot of validated module worker; only baseline loading adapted for standalone 64K screen."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import types
ROOT = Path(__file__).resolve().parent

def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')

def worker(args):
    import torch
    from compressed_tensors.quantization import QuantizationStrategy
    from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as installed
    from gemm_dispatch import patch_source
    vendor_path = ROOT / 'vendor/compressed_tensors_w8a8_int8.py'
    vendor = vendor_path.read_bytes()
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    assert hashlib.sha256(vendor).hexdigest() == manifest['sha256'], 'vendor SHA mismatch'
    actual_sha = hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest()
    accepted_sha = {manifest['sha256'], hashlib.sha256(patch_source(vendor.decode()).encode()).hexdigest()}
    assert actual_sha in accepted_sha, 'installed source differs from audited original or exact dispatch patch'
    # Standalone micro process only: original baseline comes from SHA-checked vendor.
    # No installed source or service process is changed.
    original = types.ModuleType('q38_w8a8_original_micro')
    original.__file__ = str(vendor_path)
    exec(compile(vendor.decode(), str(vendor_path), 'exec'), original.__dict__)
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
