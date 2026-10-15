"""Exact real-stride QKV preparation micro; no model/service mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
from sglang.srt.models.utils import fused_qk_gemma_rmsnorm

from qk_norm_gate import fused_qk_gemma_rmsnorm_with_gate


def baseline(qkv, qw, kw):
    q_gate, k, v = qkv.split([6144,512,512], dim=-1)
    q_gate = q_gate.view(qkv.shape[0],12,512)
    q, gate = torch.chunk(q_gate,2,dim=-1)
    q, gate = q.reshape(qkv.shape[0],3072), gate.reshape(qkv.shape[0],3072)
    qn, kn = fused_qk_gemma_rmsnorm(q,k,qw,kw,1e-6,256)
    return qn.view(qkv.shape[0],3072),kn.view(qkv.shape[0],512),v,gate


def candidate(qkv, qw, kw):
    q_gate,k,v = qkv.split([6144,512,512],dim=-1)
    qn,kn,gate = fused_qk_gemma_rmsnorm_with_gate(q_gate,k,qw,kw,1e-6,256,12)
    return qn.view(qkv.shape[0],3072),kn.view(qkv.shape[0],512),v,gate.view(qkv.shape[0],3072)


def compare(a,b):
    diff = a.float()-b.float()
    return dict(exact=bool(torch.equal(a,b)),mismatches=int((a != b).sum().item()),
                max_absolute_error=diff.abs().max().item() if diff.numel() else 0.0)


def time_one(fn, graph):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            for _ in range(5):
                hold = fn()
        run = captured.replay
        calls = 5
        captured.replay(); torch.cuda.synchronize()
    else:
        run, calls = fn, 1
    samples=[]
    for _ in range(5):
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(20): run()
        end.record();end.synchronize()
        samples.append(begin.elapsed_time(end)/(20*calls))
    return dict(median_ms=statistics.median(samples),samples_ms=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tokens",type=int,nargs="+",choices=[4096,8192,32768,65536],default=[4096,8192,32768])
    args=p.parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Restricted to reserved experiment GPU4")
    torch.set_num_threads(2)
    # Fail if the actual two-step reference has drifted from the audited source.
    vendor_file=Path("/usr/local/lib/python3.10/dist-packages/sglang/srt/models/utils.py")
    if hashlib.sha256(vendor_file.read_bytes()).hexdigest() != "774a83d6bd492ef93922f8a3c24724132d3f9b345879b897c90fdb4037801efb":
        raise RuntimeError("Vendor QK normalization source changed")
    report=dict(status="RUNNING",scope="QKV split/deinterleave/QK norm/gate extraction only",rows=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    save()
    # Include padded row stride and zero-grid regression, beyond production sizes.
    for tokens,padding in [(0,0),(129,64)]+[(n,0) for n in args.tokens]:
        for seed in ((20261015,20261016) if tokens else (20261015,)):
            torch.manual_seed(seed)
            storage=torch.randn(tokens,7168+padding,device="cuda",dtype=torch.bfloat16)
            qkv=storage[:,:7168]
            if tokens: qkv[0].zero_()
            qw=torch.randn(256,device="cuda",dtype=torch.bfloat16)*.1
            kw=torch.randn_like(qw)*.1
            qw[::8]=.001953125;kw[::8]=.001953125
            row=dict(tokens=tokens,padding=padding,seed=seed,qkv_stride=list(qkv.stride()))
            try:
                if tokens:
                    reference=baseline(qkv,qw,kw)
                else:
                    reference=(qkv.new_empty(0,3072),qkv.new_empty(0,512),qkv[:,6656:],qkv.new_empty(0,3072))
                actual=candidate(qkv,qw,kw)
                row["correctness"]={name:compare(a,b) for name,a,b in zip(("q_norm","k_norm","v","gate"),actual,reference)}
                row["v_alias_preserved"] = actual[2].data_ptr()==reference[2].data_ptr()
                exact=all(x["exact"] for x in row["correctness"].values()) and row["v_alias_preserved"]
                row["status"]="EXACT" if exact else "NUMERICAL_MISMATCH_NOT_TIMED"
                if exact and tokens in args.tokens and seed==20261015:
                    row["timings"]={}
                    for graph in (False,True):
                        row["timings"]["cuda_graph" if graph else "eager"]={
                            "baseline":time_one(lambda:baseline(qkv,qw,kw),graph),
                            "candidate":time_one(lambda:candidate(qkv,qw,kw),graph)}
            except Exception as exc:
                row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}"[:1500])
            report["rows"].append(row);print(json.dumps(row),flush=True);save()
            if row["status"] != "EXACT":
                raise RuntimeError("QK gate validation failed; see retained diagnostic")
    report["status"]="COMPLETE_WITH_DIAGNOSTICS"
    save()


if __name__=="__main__":main()
