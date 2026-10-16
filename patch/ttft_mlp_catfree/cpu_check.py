"""CPU-only check of the cat-free helper (no GPU, no container, no sglang install needed).

python3 cpu_check.py --old-helper <audited helper text, SHA a946a802...> [--workdir DIR]

1. AST: rejection_reason unchanged; no Stream/Event/record_stream/synchronize calls.
2. Simulated install.py run against a copy of the real helper text: hashes, non-.py
   backup, refusal on second install and on parent drift.
3. Functional simulation (needs CPU torch): sglang modules stubbed, torch.distributed
   replaced by a TP2 emulator (peer partial = 2*y + 1 so the reduced value is 3*y + 1).
   old helper vs new helper (catfree=0 and =1) for emulated ranks 0 and 1 must give
   identical output/residual/normalized buffers; every Work must be waited.
"""
import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
OLD_SHA256 = 'a946a802abb6a6faeb84108e4ccb10db957bc553bfc573333d026cc681febbad'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    tree = ast.parse(Path(path).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def ast_check(old_path, new_path):
    old, new = functions(old_path), functions(new_path)
    assert ast.dump(old['rejection_reason']) == ast.dump(new['rejection_reason']), 'rejection_reason changed'
    assert ast.dump(old['_parts_from_env']) == ast.dump(new['_parts_from_env'])
    banned = {'Stream', 'Event', 'record_stream', 'synchronize', 'wait_stream'}
    attrs = {n.func.attr for n in ast.walk(ast.parse(Path(new_path).read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not attrs & banned, attrs & banned
    print('AST: rejection_reason/_parts_from_env identical; no stream/event/sync calls')


def install_check(old_path, workdir):
    layers = workdir / 'layers'
    layers.mkdir()
    shutil.copy(old_path, layers / 'qwen27b_mlp_pipeline.py')
    (layers / 'qwen27b_mlp_probe.py').write_text('')
    run = lambda: subprocess.run([sys.executable, str(HERE / 'install.py'), '--layers-dir', str(layers)],
                                 capture_output=True, text=True)
    first = run()
    assert first.returncode == 0, first.stderr
    assert sha(layers / 'qwen27b_mlp_pipeline.py') == sha(HERE / 'mlp_pipeline.py')
    backup = layers / 'qwen27b_mlp_pipeline.py.ttft_catfree_original'
    assert sha(backup) == OLD_SHA256
    assert not [p for p in layers.glob('*.py') if p.name not in ('qwen27b_mlp_pipeline.py', 'qwen27b_mlp_probe.py')]
    second = run()
    assert second.returncode != 0 and 'drift' in second.stderr, second.stderr
    backup.unlink()
    shutil.copy(old_path, layers / 'qwen27b_mlp_pipeline.py')
    with open(layers / 'qwen27b_mlp_pipeline.py', 'a') as f:
        f.write('\n')
    drift = run()
    assert drift.returncode != 0 and 'drift' in drift.stderr, drift.stderr
    print('install: hashes ok, backup', backup.name, '(no extra .py); refuses reinstall and parent drift')


def stub_sglang():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m
    for name in ('sglang', 'sglang.srt', 'sglang.srt.layers', 'sglang.srt.model_executor'):
        mod(name)
    group = types.SimpleNamespace(device_group=object(), world_size=2, ranks=[0, 1])
    mod('sglang.srt.distributed', get_tp_group=lambda: group)
    mod('sglang.srt.layers.communicator', CommunicateSummableTensorPairFn=object,
        CommunicateWithAllReduceAndLayerNormFn=object, ScatterMode=object, get_attn_tp_context=None)
    mod('sglang.srt.layers.dp_attention', get_attention_tp_group=lambda: group)
    mod('sglang.srt.model_executor.cuda_graph_runner', get_is_capture_mode=lambda: False)
    mod('sglang.srt.server_args', get_global_server_args=None)
    mod('sglang.srt.utils', get_bool_env_var=None, is_hcu=None)
    mod('sglang.srt.layers.qwen27b_mlp_probe', begin=lambda *a: None)


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


class FakeDist:
    """Single-process TP2 emulator; the peer's partial is 2*y + 1."""

    def __init__(self, torch, rank):
        self.torch, self.rank, self.works, self.pending = torch, rank, [], {}

    def _work(self):
        work = types.SimpleNamespace(waited=False)
        work.wait = lambda: setattr(work, 'waited', True)
        self.works.append(work)
        return work

    def get_rank(self, group=None):
        return self.rank

    def all_reduce(self, t, group=None, async_op=False):
        t.copy_(t + (2 * t + 1))
        return self._work()

    def reduce_scatter_tensor(self, out, inp, group=None, async_op=False):
        assert inp.is_contiguous() and out.is_contiguous() and inp.numel() == 2 * out.numel()
        full = inp + (2 * inp + 1)
        out.copy_(full[self.rank * out.numel():(self.rank + 1) * out.numel()])
        self.pending[out.data_ptr()] = full
        return self._work()

    def all_gather_into_tensor(self, out, inp, group=None, async_op=False):
        assert out.data_ptr() + self.rank * inp.numel() * inp.element_size() == inp.data_ptr(), 'not in-place'
        full = self.pending.pop(inp.data_ptr())
        assert self.torch.equal(out[self.rank * inp.numel():(self.rank + 1) * inp.numel()],
                                full[self.rank * inp.numel():(self.rank + 1) * inp.numel()])
        out.copy_(full)
        return self._work()


def functional_check(old_path, new_path):
    import torch
    stub_sglang()
    old, new = load('old_helper', old_path), load('new_helper', new_path)

    class Norm:
        def __call__(self, x, r):  # fused add + norm, both in place like GemmaRMSNorm
            r.add_(x)
            x.copy_(r * torch.rsqrt(r.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(r.dtype))
            return x, r

    class Layer:
        layer_id = 0
        post_attention_layernorm = Norm()
        layer_communicator = types.SimpleNamespace(postprocess_layer=lambda o, r, fb: (o, r))
        weight = torch.randn(16, 16, generator=torch.Generator().manual_seed(1)).bfloat16()

        def mlp(self, x, should_allreduce_fusion, use_reduce_scatter):
            assert not should_allreduce_fusion and use_reduce_scatter
            return torch.nn.functional.silu(x @ self.weight)

    layer = Layer()
    g = torch.Generator().manual_seed(7)
    source, residual_source = torch.randn(64, 16, generator=g).bfloat16(), torch.randn(64, 16, generator=g).bfloat16()
    for rank in (0, 1):
        results = {}
        for label, module, catfree in (('old', old, None), ('new_cat', new, False), ('new_catfree', new, True)):
            fake = FakeDist(torch, rank)
            module.dist = fake
            module.rejection_reason = lambda *a: None
            module._HITS = 1  # skip the CUDA-stream log line on CPU
            assert module._PIPELINE_PARTS == 8
            x, r = source.clone(), residual_source.clone()
            out = (module.try_mlp_pipeline(layer, x, r, None) if catfree is None
                   else module.run_mlp_pipeline(layer, x, r, None, catfree=catfree))
            assert all(w.waited for w in fake.works) and not fake.pending
            expected_works = 8 + 8 * (2 if catfree else 1)
            assert len(fake.works) == expected_works, (label, len(fake.works))
            results[label] = (out[0], out[1], x)
        for label in ('new_cat', 'new_catfree'):
            for a, b in zip(results[label], results['old']):
                assert torch.equal(a, b), (rank, label)
        assert results['new_catfree'][1].data_ptr() != 0
    print('functional: old == new(catfree=0) == new(catfree=1) for emulated ranks 0/1; all works waited')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-helper', required=True)
    p.add_argument('--skip-functional', action='store_true')
    args = p.parse_args()
    assert sha(args.old_helper) == OLD_SHA256, 'old helper text is not the audited a946a802... file'
    new_path = HERE / 'mlp_pipeline.py'
    ast_check(args.old_helper, new_path)
    with tempfile.TemporaryDirectory() as tmp:
        install_check(Path(args.old_helper), Path(tmp))
    if not args.skip_functional:
        import os
        os.environ['SGLANG_QWEN27B_MLP_PIPELINE_PARTS'] = '8'
        functional_check(args.old_helper, new_path)


if __name__ == '__main__':
    main()
