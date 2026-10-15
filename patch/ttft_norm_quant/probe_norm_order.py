"""Bounded diagnostic for native GemmaNorm operation/reduction ordering.

Eight named hypotheses, each tied to FP32 multiplication association, an epsilon
FMA, or the native binary's width-8 kernel symbol. No timing, tolerance change,
production mutation, or automatic acceptance into the existing fusion helper.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from lightop.norm import gemma_fused_add_rmsnorm
from lightop.quant import per_token_quant_int8


@triton.jit
def _probe(X, R, W, Y, V, N: tl.constexpr, BLOCK: tl.constexpr,
           REDUCE: tl.constexpr, RIGHT_ASSOC: tl.constexpr,
           EPS_FMA: tl.constexpr, THREADS: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < N
    x = tl.load(X + row*N + columns, valid, other=0).to(tl.float32)
    residual = tl.load(R + row*N + columns, valid, other=0).to(tl.float32)
    value = (x+residual).to(tl.bfloat16).to(tl.float32)
    value = tl.where(valid, value, 0.0)
    if REDUCE == 0:
        total = tl.sum(value*value, axis=0)
    else:
        # Emulate a width-8, 256-thread C++ layout: each thread accumulates
        # successive eight-value vectors, then block-reduces its FP32 total.
        # This is a hypothesis, not a claim to reproduce CUB/rocPRIM's exact
        # shuffle sequence; final native comparison remains authoritative.
        lanes = tl.arange(0, THREADS)
        thread_total = tl.full((THREADS,), 0, tl.float32)
        for iteration in tl.static_range(triton.cdiv(N, THREADS*8)):
            vector_total = tl.full((THREADS,), 0, tl.float32)
            for element in tl.static_range(8):
                pos = (iteration*THREADS + lanes)*8 + element
                xv = tl.load(X + row*N + pos, pos < N, other=0).to(tl.float32)
                rv = tl.load(R + row*N + pos, pos < N, other=0).to(tl.float32)
                z = (xv+rv).to(tl.bfloat16).to(tl.float32)
                if REDUCE == 2:
                    vector_total = tl.fma(z, z, vector_total)
                else:
                    vector_total = vector_total + z*z
            thread_total = thread_total + vector_total
        total = tl.sum(thread_total, axis=0)
    if EPS_FMA:
        variance_eps = tl.fma(total, 1.0/N, 1e-6)
    else:
        variance_eps = total/N + 1e-6
    inv = tl.rsqrt(variance_eps)
    weight = tl.load(W + columns, valid, other=0).to(tl.float32)+1.0
    if RIGHT_ASSOC:
        output = value*(inv*weight)
    else:
        output = (value*inv)*weight
    tl.store(Y + row*N + columns, output.to(tl.bfloat16), valid)
    tl.store(V + row*2, total)
    tl.store(V + row*2+1, inv)


VARIANTS = [
    ("flat_left_separate_epsilon", 0, False, False),
    ("flat_right_separate_epsilon", 0, True, False),
    ("flat_left_fma_epsilon", 0, False, True),
    ("flat_right_fma_epsilon", 0, True, True),
    ("vector8_separate_left_fma_epsilon", 1, False, True),
    ("vector8_separate_right_fma_epsilon", 1, True, True),
    ("vector8_fma_left_fma_epsilon", 2, False, True),
    ("vector8_fma_right_fma_epsilon", 2, True, True),
]


def stats(a, b):
    difference = a.float()-b.float()
    return dict(exact=bool(torch.equal(a,b)), mismatches=int((a != b).sum().item()),
                mismatch_fraction=(a != b).float().mean().item(),
                max_absolute_error=difference.abs().max().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", default="4096")
    args = p.parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Restricted to experiment GPU4")
    torch.set_num_threads(2)
    report = dict(status="RUNNING", rows=[], variant_count=len(VARIANTS),
                  purpose="operation-order diagnosis; no performance or quality acceptance")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")

    save()
    for tokens in [int(v) for v in args.tokens.split(",")]:
        if tokens not in (4096,8192,16384):
            raise ValueError("Probe is restricted to prefill shapes")
        torch.manual_seed(20261015)
        x = torch.randn(tokens,5120,device="cuda",dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = (torch.randn(5120,device="cuda",dtype=torch.bfloat16)*.1).contiguous()
        x[0].zero_(); residual[0].zero_()
        residual[1].copy_(-x[1]); weight[::8] = .001953125
        native, native_residual = gemma_fused_add_rmsnorm(x.clone(),residual.clone(),weight,1e-6)
        native_q, native_s = per_token_quant_int8(native)
        report.setdefault("residual", []).append(dict(tokens=tokens,
            comparison=stats((x.float()+residual.float()).to(torch.bfloat16),native_residual)))
        exact_names = []
        variance_reference = None
        for name, reduction, right, eps_fma in VARIANTS:
            row = dict(tokens=tokens,variant=name)
            try:
                output = torch.empty_like(x)
                moments = torch.empty(tokens,2,device=x.device,dtype=torch.float32)
                _probe[(tokens,)](x,residual,weight,output,moments,5120,8192,
                    reduction,right,eps_fma,256,num_warps=4,num_stages=1,enable_fp_fusion=False)
                q,s = per_token_quant_int8(output)
                row.update(norm=stats(output,native),quant=stats(q,native_q),scale=stats(s,native_s))
                if variance_reference is None:
                    variance_reference = moments.clone()
                row["moments_vs_flat"] = stats(moments,variance_reference)
                row["exact_all_outputs"] = all(row[k]["exact"] for k in ("norm","quant","scale"))
                if row["exact_all_outputs"]:
                    exact_names.append(name)
                mismatch = (output != native).nonzero()[:6].cpu().tolist()
                row["examples"] = [dict(row=r,col=c,input=x[r,c].item(),
                    residual=residual[r,c].item(),weight=weight[c].item(),
                    candidate=output[r,c].item(),native=native[r,c].item(),
                    sum_squares=moments[r,0].item(),inv_rms=moments[r,1].item()) for r,c in mismatch]
            except Exception as exc:
                row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}"[:1500])
            report["rows"].append(row); print(json.dumps(row),flush=True);save()
        report.setdefault("exact_variants",[]).append(dict(tokens=tokens,names=exact_names))
    report["status"] = "DIAGNOSED_NO_AUTOMATIC_ACCEPTANCE"
    save()


if __name__ == "__main__":
    main()
