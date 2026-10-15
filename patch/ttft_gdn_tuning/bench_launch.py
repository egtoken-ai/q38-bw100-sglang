"""Bounded BW100 launch sweep. No installed files are changed.

Only an isolated Python process's kernel-launch objects are temporarily wrapped.
Every repetition resets the exact initial state; built-in multi-config autotune
is never used for the state-mutating chunk_h kernel.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path

import torch
import triton

HMOD = importlib.import_module('sglang.srt.layers.attention.fla.chunk_delta_h')
OMOD = importlib.import_module('sglang.srt.layers.attention.fla.chunk_o')
CMOD = importlib.import_module('sglang.srt.layers.attention.fla.chunk')
HNAME = 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64'
ONAME = 'chunk_fwd_kernel_o'
HBASE = dict(BV=32, num_warps=4, num_stages=2)
OBASE = dict(BK=128, BV=64, num_warps=4, num_stages=2)
HCANDIDATES = [
    dict(BV=16, num_warps=4, num_stages=2),
    dict(BV=32, num_warps=4, num_stages=3),
    dict(BV=64, num_warps=4, num_stages=2),
    dict(BV=32, num_warps=8, num_stages=2),
]
OCANDIDATES = [
    dict(BK=64, BV=128, num_warps=2, num_stages=3),
    dict(BK=64, BV=64, num_warps=4, num_stages=2),
    dict(BK=128, BV=128, num_warps=4, num_stages=2),
]


class FixedLaunch:
    def __init__(self, raw, config):
        self.raw = raw
        self.config = config

    def __getitem__(self, grid):
        # Both audited wrappers pass grid(meta), not an already evaluated tuple.
        # A tuple could have been calculated with the old BV and miss outputs.
        if not callable(grid):
            raise RuntimeError('Launch grid must be callable and recompute from candidate BV')

        def launch(*args, **kwargs):
            kwargs.update(self.config)
            return self.raw[grid](*args, **kwargs)
        return launch


@contextmanager
def config_override(hconfig=None, oconfig=None):
    old_h = getattr(HMOD, HNAME)
    old_o = getattr(OMOD, ONAME)
    try:
        if hconfig is not None:
            # Current vendor h is Autotuner(JITFunction). Bypass the autotuner,
            # retaining the original kernel and running one explicit config.
            raw_h = old_h.fn
            if not hasattr(raw_h, 'cache_key'):
                raise RuntimeError('Unexpected chunk_h wrapper; inspect before tuning')
            setattr(HMOD, HNAME, FixedLaunch(raw_h, hconfig))
        if oconfig is not None:
            setattr(OMOD, ONAME, FixedLaunch(old_o, oconfig))
        yield
    finally:
        setattr(HMOD, HNAME, old_h)
        setattr(OMOD, ONAME, old_o)


def compare(reference, actual):
    torch.testing.assert_close(reference, actual, rtol=0.02, atol=0.02)
    diff = reference.float() - actual.float()
    relative_l2 = float(diff.norm() / reference.float().norm().clamp_min(1e-12))
    if relative_l2 > 0.01:
        raise AssertionError(f'Relative L2 {relative_l2} exceeds 1%')
    return dict(max_abs=float(diff.abs().max()), relative_l2=relative_l2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--warmup-ms', type=int, default=30)
    p.add_argument('--rep-ms', type=int, default=100)
    p.add_argument('--include-long', action='store_true', help='Add one 64K sequence; diagnostic only')
    args = p.parse_args()
    if HMOD._use_prefill_aiter_linear_attn or OMOD._use_prefill_aiter_linear_attn:
        raise RuntimeError('AITER dispatch is active; this script only tunes the audited Triton path')
    if HMOD.CHUNK_SIZE != 64:
        raise RuntimeError('Unexpected CHUNK_SIZE')
    configs = getattr(getattr(HMOD, HNAME), 'configs', [])
    if len(configs) != 1 or configs[0].kwargs != {'BV': 32}:
        raise RuntimeError('chunk_h baseline configuration drift')
    if configs[0].num_warps != 4 or configs[0].num_stages != 2:
        raise RuntimeError('chunk_h baseline launch drift')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for m in (HMOD, OMOD, CMOD)}
    payload = dict(device=torch.cuda.get_device_name(), sources=sources,
                   h_baseline=HBASE, o_baseline=OBASE,
                   timing='complete chunk_gated_delta_rule plus identical initial-state reset',
                   input_dtype='bfloat16', state_dtype='float32', cases=[])
    torch.manual_seed(38806)
    lengths_cases = [[192], [4096], [4096, 4096], [8192]]
    if args.include_long:
        lengths_cases.append([65536])
    for lengths in lengths_cases:
        tokens = sum(lengths)
        q = torch.nn.functional.normalize(torch.randn(1, tokens, 8, 128, device='cuda'), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
        v = torch.randn(1, tokens, 24, 128, device='cuda', dtype=torch.bfloat16)
        g = -torch.rand(1, tokens, 24, device='cuda') * 0.1 - 0.01
        beta = torch.rand(1, tokens, 24, device='cuda').bfloat16().float()
        initial = torch.randn(len(lengths), 24, 128, 128, device='cuda') * 0.01
        state = initial.clone()
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        cu = torch.tensor(offsets, device='cuda', dtype=torch.int32)
        indices = torch.arange(len(lengths), device='cuda', dtype=torch.int32)

        def run():
            state.copy_(initial)
            return CMOD.chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                initial_state_indices=indices, cu_seqlens=cu,
                head_first=False, use_qk_l2norm_in_kernel=False)

        reference_return = run()
        reference = reference_return[0].clone()
        reference_state = state.clone()
        baseline_state_delta = reference_state - initial
        baseline_state_changed = int(torch.count_nonzero(baseline_state_delta))
        if baseline_state_changed == 0:
            raise AssertionError('State oracle is vacuous: baseline did not update initial_state')
        baseline_state_update = dict(
            changed_elements=baseline_state_changed,
            max_abs=float(baseline_state_delta.abs().max()),
            l2=float(baseline_state_delta.norm()),
        )
        baseline_ms = triton.testing.do_bench(run, warmup=args.warmup_ms, rep=args.rep_ms)
        case = dict(lengths=lengths, baseline_ms=baseline_ms,
                    baseline_state_update_vs_initial=baseline_state_update, trials=[])
        payload['cases'].append(case)

        def trial(label, hconfig=None, oconfig=None):
            entry = dict(label=label, hconfig=hconfig, oconfig=oconfig)
            case['trials'].append(entry)
            try:
                with config_override(hconfig, oconfig):
                    result = run()
                    torch.cuda.synchronize()
                    entry['output_check'] = compare(reference, result[0])
                    entry['state_check'] = compare(reference_state, state)
                    state_delta = state - initial
                    changed_elements = int(torch.count_nonzero(state_delta))
                    if changed_elements == 0:
                        raise AssertionError('Candidate did not update initial_state')
                    entry['state_update_vs_initial'] = dict(
                        changed_elements=changed_elements,
                        max_abs=float(state_delta.abs().max()),
                        l2=float(state_delta.norm()),
                    )
                    entry['ms'] = triton.testing.do_bench(run, warmup=args.warmup_ms, rep=args.rep_ms)
                    entry['latency_improvement_pct'] = (1-entry['ms']/baseline_ms)*100
                    entry['status'] = 'pass'
            except Exception as exc:
                entry['status'] = 'rejected'
                entry['error'] = f'{type(exc).__name__}: {str(exc)[:1200]}'
                # Compilation/resource failures reject a candidate. A device
                # execution fault invalidates the process: stop immediately.
                try:
                    torch.cuda.synchronize()
                except Exception:
                    output.write_text(json.dumps(payload, indent=2) + '\n')
                    raise
                if not (isinstance(exc, (AssertionError, RuntimeError))
                        or type(exc).__module__.startswith('triton.')):
                    raise
            output.write_text(json.dumps(payload, indent=2) + '\n')
            print(json.dumps(dict(lengths=lengths, **entry)), flush=True)
            return entry

        # Verify bypassing Autotuner at its own config matches the real baseline.
        control = trial('explicit_baseline_control', HBASE, OBASE)
        if control['status'] != 'pass':
            raise RuntimeError('Explicit baseline did not reproduce deployed baseline')
        htrials = [trial('h_only', hconfig=c) for c in HCANDIDATES]
        otrials = [trial('o_only', oconfig=c) for c in OCANDIDATES]
        hvalid = [x for x in htrials if x['status'] == 'pass' and x['ms'] < baseline_ms]
        ovalid = [x for x in otrials if x['status'] == 'pass' and x['ms'] < baseline_ms]
        if hvalid and ovalid:
            hbest = min(hvalid, key=lambda x:x['ms'])['hconfig']
            obest = min(ovalid, key=lambda x:x['ms'])['oconfig']
            trial('combined_best', hbest, obest)
    output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
