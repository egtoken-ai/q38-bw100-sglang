"""Full exactness gate against actual torch sigmoid(BF16)*BF16, before timing."""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
from fa_gate import fused_gate


def compare(a,b):
    diff=a.float()-b.float()
    return dict(exact=bool(torch.equal(a,b)),mismatches=int((a!=b).sum().item()),
                max_abs=diff.abs().max().item(),all_finite=bool(torch.isfinite(a).all().item()))


def timing(fn, graph):
    for _ in range(10):fn()
    torch.cuda.synchronize()
    if graph:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(5): held=fn()
        run,calls=g.replay,5
        g.replay();torch.cuda.synchronize()
    else:run,calls=fn,1
    values=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):run()
        b.record();b.synchronize()
        values.append(a.elapsed_time(b)/(20*calls))
    return dict(median_ms=statistics.median(values),samples_ms=values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES')!='4':raise RuntimeError('Reserved GPU4 only')
    torch.set_num_threads(2)
    report=dict(status='RUNNING',scope='FA gate only; no model integration',rows=[],exhaustive=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    # Enumerate every finite BF16 gate bit pattern on CPU. Attention=1 makes the
    # result equal to the BF16 sigmoid, exposing intermediate rounding directly.
    gates=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gates=gates[torch.isfinite(gates)]
    count=gates.numel();padded=((count+3071)//3072)*3072
    exhaustive=torch.zeros(padded,dtype=torch.bfloat16)
    exhaustive[:count]=gates
    exhaustive=exhaustive.view(-1,3072).cuda()
    ones=torch.ones_like(exhaustive)
    expected=torch.sigmoid(exhaustive)*ones
    admission={}
    for lib_exp in (False,True):
        name='libdevice_exp' if lib_exp else 'tl_sigmoid'
        try:
            error=compare(fused_gate(ones,exhaustive,libdevice_exp=lib_exp),expected)
            admission[lib_exp]=error['exact']
            report['exhaustive'].append(dict(candidate=name,finite_gate_encodings=count,**error))
        except Exception as exc:
            admission[lib_exp]=False
            report['exhaustive'].append(dict(candidate=name,status='FAILED',error=str(exc)))
    save()
    for tokens in (4096,8192,65536):
        for seed in (20261015,20261016):
            torch.manual_seed(seed)
            attn=torch.randn(tokens,3072,device='cuda',dtype=torch.bfloat16)
            gate=torch.randn_like(attn)*3
            gate[0]=torch.linspace(-80,80,3072,device='cuda').to(torch.bfloat16)
            ref=torch.sigmoid(gate)*attn
            if seed==20261015:
                base=lambda:torch.sigmoid(gate)*attn
                row=dict(tokens=tokens,seed=seed,candidate='vendor_torch_two_step',status='BASELINE',
                         eager=timing(base,False),cuda_graph=timing(base,True))
                report['rows'].append(row);print(json.dumps(row),flush=True);save()
            for lib_exp in (False,True):
                name='libdevice_exp' if lib_exp else 'tl_sigmoid'
                row=dict(tokens=tokens,seed=seed,candidate=name)
                try:
                    fn=lambda:fused_gate(attn,gate,libdevice_exp=lib_exp)
                    row['correctness']=compare(fn(),ref)
                    exact=admission[lib_exp] and row['correctness']['exact']
                    row['status']='EXACT' if exact else 'MISMATCH_NOT_TIMED'
                    if exact and seed==20261015:
                        row.update(eager=timing(fn,False),cuda_graph=timing(fn,True))
                except Exception as exc:row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                report['rows'].append(row);print(json.dumps(row),flush=True);save()
    report['status']='COMPLETE_WITH_DIAGNOSTICS';save()


if __name__=='__main__':main()
