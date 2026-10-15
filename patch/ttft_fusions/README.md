# Qwen27B SiLU + INT8 quantization candidate

Issue: https://github.com/egtoken-ai/q38-bw100-sglang/issues/3

The deployed Hygon SGLang image has `lm_fuse_silu_mul_quant` and a
`RowParallelLinear.forward(use_fused_silu_mul_quant=True)` route, but the dense
Qwen3.5 implementation calls SiLU and INT8 quantization separately. This patch
connects that existing route for the Qwen27B TP2 dimensions. It removes the BF16
activation output buffer and the separate quantization launch. Its benefit is
unmeasured until a real device test; it is not a claim of 20% model TTFT improvement.

The patch changes only the Qwen3.5 model's MLP import. It does not change shared
Qwen2MoE callers. The helper requires HCU, TP2, hidden size 5120, intermediate
size 17408, dynamic symmetric channel-wise INT8 with vendor strategy 1, BF16
input, and at least 256 flattened tokens. Smaller batches retain the original
path, including the currently tested decode batches up to 128 requests. It also
requires both enable flags and excludes on-policy RL and DP-attention reduction.
Gemma RMSNorm and residual arithmetic remain in their existing operations.

Build on bw100, after synchronizing only the project code:

```sh
docker build -t sglang-bw100:ttft-silu-issue3 patch/ttft_fusions
```

The Dockerfile derives from `sglang-bw100:ttft-packed-issue3`, retaining the packed
prefill optimization. The new fusion is **off by default**: the first single-call
measurement was slightly slower at 4096 and 8192 tokens, so it must not be enabled
without stronger evidence. Multi-call eager and CUDA-graph measurements have
been added to distinguish launch noise from kernel cost. Source SHA256 guards fail closed if the audited model,
linear, or quantization implementations differ. To disable only this new route,
launch with `SGLANG_QWEN27B_FUSED_SILU_QUANT=false` before model construction;
use `true` only for a deliberately selected candidate experiment.

Run the independently usable microbenchmark before launching a service:

```sh
HIP_VISIBLE_DEVICES=4 python3 /bench/patch/ttft_prefill/bench_fusion.py \
  --output /bench/result/ttft/issue3-attention/fusion.json
```

Its SiLU baseline calls exactly the HCU operation used by `SiluAndMul`, without
constructing a module that requires SGLang's global server arguments. The
benchmark reports quantized reconstruction error and individual operation time;
its norm experiment is diagnostic only and is not installed by this patch.

Fused vendor quantization can change rounding relative to materializing a BF16
SiLU output first. Quantization error, model logits on identical continuations,
and actual inference correctness must be checked; kernel tolerance alone does
not establish model equivalence. Only the unchanged two quick benchmark cases
can screen end-to-end gain, followed by the goal's matched repeated acceptance.
