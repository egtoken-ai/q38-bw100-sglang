"""Identify vendor BF16 SiLU/multiply rounding without changing the candidate.

Descriptive probe only. Compares actual BF16 activation (before INT8) with
multiple explicit formulas, then reports quantizer effects. No timing, service
integration, or relaxed correctness threshold.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from lightop.activation import fuse_silu_and_mul
from lightop.quant import per_token_quant_int8


@triton.jit
def _activation_probe(X, Y, N: tl.constexpr, TOTAL: tl.constexpr,
                      MODE: tl.constexpr, LIBDEVICE_EXP: tl.constexpr,
                      BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // N, idx % N
    g = tl.load(X + row * (2*N) + col, idx < TOTAL, other=0).to(tl.float32)
    u = tl.load(X + row * (2*N) + N + col, idx < TOTAL, other=0).to(tl.float32)
    if LIBDEVICE_EXP:
        exponential = libdevice.exp(-g)
    else:
        exponential = tl.exp(-g)
    if MODE <= 1:
        silu = g / (1 + exponential)
        if MODE == 1:
            silu = silu.to(tl.bfloat16).to(tl.float32)
        out = silu * u
    else:
        sigmoid = (1 / (1 + exponential)).to(tl.bfloat16).to(tl.float32)
        silu = g * sigmoid
        if MODE == 3:
            silu = silu.to(tl.bfloat16).to(tl.float32)
        out = silu * u
    tl.store(Y + idx, out.to(tl.bfloat16), idx < TOTAL)


def triton_formula(x, mode, use_libdevice):
    out = torch.empty(x.shape[0], x.shape[1]//2, dtype=torch.bfloat16, device=x.device)
    _activation_probe[(triton.cdiv(out.numel(), 256),)](
        x, out, out.shape[-1], out.numel(), mode, use_libdevice, 256,
        num_warps=4, enable_fp_fusion=False,
    )
    return out


def formulas(x):
    """Independent eager Torch formulas; all intermediate casts are explicit."""
    g, u = x.float().chunk(2, dim=-1)
    silu = g / (1 + torch.exp(-g))
    yield "torch_final_bf16_only", (silu*u).to(torch.bfloat16)
    yield "torch_silu_bf16_then_mul", (silu.to(torch.bfloat16).float()*u).to(torch.bfloat16)
    yield "torch_native_silu_bf16_then_mul", F.silu(x[:, :x.shape[-1]//2])*x[:, x.shape[-1]//2:]
    sig = torch.sigmoid(g).to(torch.bfloat16).float()
    yield "torch_sigmoid_bf16_then_two_mul", (g*sig*u).to(torch.bfloat16)
    yield "torch_sigmoid_bf16_silu_bf16_then_mul", ((g*sig).to(torch.bfloat16).float()*u).to(torch.bfloat16)


def stats(a, b):
    difference = a.float()-b.float()
    return {
        "exact": bool(torch.equal(a, b)),
        "mismatch_count": int((a != b).sum().item()),
        "mismatch_fraction": (a != b).float().mean().item(),
        "max_absolute_error": difference.abs().max().item(),
        "relative_l2_error": (difference.norm()/b.float().norm().clamp_min(1e-30)).item(),
    }


def describe(name, x, actual, vendor, vendor_quant):
    q, scale = per_token_quant_int8(actual)
    out = {
        "formula": name,
        "activation": stats(actual, vendor),
        "quant": stats(q, vendor_quant[0]),
        "scale": stats(scale, vendor_quant[1]),
    }
    mismatch = (actual != vendor).nonzero()
    examples = []
    for row, col in mismatch[:6].cpu().tolist():
        examples.append({
            "row": row, "col": col,
            "gate": x[row, col].item(), "up": x[row, col+8704].item(),
            "actual": actual[row, col].item(), "vendor": vendor[row, col].item(),
        })
    out["mismatch_examples"] = examples
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", default="128,4096")
    args = parser.parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Restricted to the experiment's GPU4")
    torch.set_num_threads(2)
    result = {"status": "RUNNING", "width_in": 17408, "rows": [],
              "note": "No existing candidate code changed; results identify activation rounding, not performance"}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for tokens in [int(v) for v in args.tokens.split(",")]:
        if tokens not in (128, 4096, 8192, 16384):
            raise ValueError("Use bounded shapes 128,4096,8192,16384")
        torch.manual_seed(20261015)
        x = torch.randn(tokens, 17408, device="cuda", dtype=torch.bfloat16)
        # Same useful edge rows as the initial microbenchmark.
        x[0].zero_()
        x[1, :8704].zero_()
        x[2, :8704] = torch.linspace(-80, 80, 8704, device="cuda").to(torch.bfloat16)
        x[3, :8704].fill_(16)
        x[3, 8704:] = ((torch.arange(8704, device="cuda") % 509 - 254).float()/32).to(torch.bfloat16)
        vendor = torch.empty(tokens, 8704, device="cuda", dtype=torch.bfloat16)
        fuse_silu_and_mul(x, vendor)
        vendor_quant = per_token_quant_int8(vendor)
        cpu_x = x[:8].cpu()
        cpu_vendor = vendor[:8].cpu()
        result.setdefault("cpu_formula_reference", []).append({
            "tokens": tokens, "sample_rows": 8,
            "formulas": [{"formula": name, "activation": stats(value, cpu_vendor)} for name, value in formulas(cpu_x)],
        })
        for name, value in formulas(x):
            row = {"tokens": tokens, **describe(name, x, value, vendor, vendor_quant)}
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
            save()
        labels = ("final_bf16_only", "silu_bf16_then_mul", "sigmoid_bf16_then_two_mul", "sigmoid_bf16_silu_bf16_then_mul")
        for use_libdevice in (False, True):
            for mode, label in enumerate(labels):
                name = f"triton_{'libdevice_exp' if use_libdevice else 'exp'}_{label}"
                try:
                    value = triton_formula(x, mode, use_libdevice)
                    row = {"tokens": tokens, **describe(name, x, value, vendor, vendor_quant)}
                except Exception as exc:
                    row = {"tokens": tokens, "formula": name, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:1500]}
                result["rows"].append(row)
                print(json.dumps(row), flush=True)
                save()
    result["status"] = "COMPLETE"
    save()


if __name__ == "__main__":
    main()
