"""Exact screen for token-grouped GDN norm-only and norm+INT8 paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
from lightop.quant import per_token_quant_int8
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm

from gdn_norm_quant import norm_gate

EXPECTED={
    '/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/fla/layernorm_gated.py':'edd10f7088d5246b49145f8f3a546f30251e1ae9012eb0559bdaa3c0092d2b9d',
    '/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/layers/gemm/int8_utils.py':'e1e1ba8ffe63a070cff6920a025d55b8d1fa6d185a7da4bb0fe5f6bf09016114',
}


def stats(a,b):
    delta=a.float()-b.float()
    return dict(exact=bool(torch.equal(a,b)),mismatches=int((a!=b).sum().item()),
                max_abs=delta.abs().max().item() if delta.numel() else 0.0,
                all_finite=bool(torch.isfinite(a).all().item()))


def timing(fn,graph):
    for _ in range(10):fn()
    torch.cuda.synchronize()
    if graph:
        captured=torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            for _ in range(5):hold=fn()
        run,calls=captured.replay,5
        captured.replay();torch.cuda.synchronize()
    else:run,calls=fn,1
    samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):run()
        b.record();b.synchronize()
        samples.append(a.elapsed_time(b)/(20*calls))
    return dict(median_ms=statistics.median(samples),samples_ms=samples)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',default='4096,8192,65536');args=p.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES')!='4':raise RuntimeError('Reserved GPU4 only')
    torch.set_num_threads(2)
    for path,expected in EXPECTED.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Audited baseline source changed: {path}')
    report=dict(status='RUNNING',scope='independent GDN post-attention boundary',rows=[],source_sha256=EXPECTED,
                layout='core [T,3072], strided projection Z [T,24,128], standard weight [128]',
                baseline_ctas_per_token=6,candidate_ctas_per_token=1)
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for tokens in [int(v) for v in args.tokens.split(',')]:
        if tokens not in (4096,8192,32768,65536):raise ValueError('Unsupported bounded shape')
        for seed in (20261015,20261016):
            torch.manual_seed(seed)
            x=torch.randn(tokens,3072,device='cuda',dtype=torch.bfloat16)
            # Reproduce projection-view holes: row stride 8192, head stride128.
            projection=torch.randn(tokens,8192,device='cuda',dtype=torch.bfloat16)
            z=projection[:,5120:].view(tokens,24,128)
            x[0].zero_();z[1].zero_()
            weight=torch.randn(128,device='cuda',dtype=torch.bfloat16)
            layer=RMSNorm(128,eps=1e-6,norm_before_gate=True,device=x.device,dtype=torch.bfloat16)
            layer.weight.copy_(weight)
            baseline_norm=lambda:layer(x.view(-1,128),z).view(tokens,3072)
            baseline_boundary=lambda:per_token_quant_int8(baseline_norm())
            ref_norm=baseline_norm();ref_q,ref_s=per_token_quant_int8(ref_norm)
            if seed==20261015:
                base=dict(tokens=tokens,seed=seed,candidate='vendor_norm_then_quant',status='BASELINE',timings={})
                for mode,fn in (('norm_only',baseline_norm),('norm_then_quant',baseline_boundary)):
                    base['timings'][mode]=dict(eager=timing(fn,False),cuda_graph=timing(fn,True))
                report['rows'].append(base);print(json.dumps(base),flush=True);save()
            for warps in (4,8):
                for fused in (False,True):
                    row=dict(tokens=tokens,seed=seed,num_warps=warps,
                             candidate='fused_norm_quant' if fused else 'token_grouped_norm',z_stride=list(z.stride()))
                    try:
                        q,s,norm=norm_gate(x,z,weight,quantize=fused,emit_norm=True,num_warps=warps)
                        if not fused:q,s=per_token_quant_int8(norm)
                        row['correctness']=dict(norm=stats(norm,ref_norm),quant=stats(q,ref_q),scale=stats(s,ref_s))
                        exact=all(v['exact'] and v['all_finite'] for v in row['correctness'].values())
                        row['status']='EXACT' if exact else 'MISMATCH_NOT_TIMED'
                        if exact and seed==20261015:
                            if fused:
                                # Verify the no-norm-output specialization independently.
                                q_only,s_only,_=norm_gate(x,z,weight,num_warps=warps)
                                if not (torch.equal(q_only,ref_q) and torch.equal(s_only,ref_s)):
                                    raise RuntimeError('Quant-only specialization changed q/scale')
                                fn=lambda:norm_gate(x,z,weight,num_warps=warps)
                                row['timings']=dict(eager=timing(fn,False),cuda_graph=timing(fn,True))
                            else:
                                fn_norm=lambda:norm_gate(x,z,weight,quantize=False,num_warps=warps)[2]
                                fn_boundary=lambda:per_token_quant_int8(fn_norm())
                                row['timings']={}
                                for mode,fn in (('norm_only',fn_norm),('norm_then_quant',fn_boundary)):
                                    row['timings'][mode]=dict(eager=timing(fn,False),cuda_graph=timing(fn,True))
                    except Exception as exc:row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}'[:1500])
                    report['rows'].append(row);print(json.dumps(row),flush=True);save()
    report['status']='COMPLETE_WITH_DIAGNOSTICS';save()


if __name__=='__main__':main()
