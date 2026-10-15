# Residual Gemma RMSNorm + INT8 boundary candidate

Standalone screening code for Issue 3; nothing imports it into a model server.
The hypothesis is to remove an activation write/read and a launch by retaining
the normalized BF16 values in registers until per-token INT8 quantization.

## Source contract and limitations

The deployed `GemmaRMSNorm.forward_hip` uses the following HCU path when a residual
is present and the existing vendor norm backend is enabled:
`lightop.norm.gemma_fused_add_rmsnorm(x,residual,original_weight,epsilon)`.
The Python wrapper calls a native operator that overwrites `x` and `residual`,
then returns those tensors. A later dynamic INT8 quantizer consumes the BF16 `x`.
The original `forward_native` specification promotes the original weight to
FP32 before adding one. The candidate preserves `weight.float()+1` and rounds
the normalized result to BF16 **before** computing the INT8 amax and scale.
It never forms the Gemma gamma by BF16 addition.

The deployed native C++ body is packaged in a shared library, so the Python
wrapper alone does not prove whether its RMS calculation uses the unrounded
FP32 residual sum or the BF16-stored residual sum. Both hypotheses are explicit
microbenchmark variants. The stored residual is BF16 in both. The program must
match the actual native operator on full tensors before being timed; a CPU
formula match alone is insufficient. Reduction ordering can also cause exact
comparison failure and is not excused by a wider tolerance.

The residual-free HCU branch differs: it passes a precomputed
`gemma_weight = weight + 1` to `rms_norm_opt`. That precomputation can round in
BF16. **Residual-free calls are excluded from this candidate.** Post-residual
additions, noncontiguous tensors, and input/residual aliasing are also outside
the initial helper's scope.

The helper leaves `x` unchanged and updates only its explicit residual argument.
It returns INT8 quantization, FP32 scales, updated BF16 residual, and optionally a
BF16 norm output. This is a distinct experimental API, not a drop-in replacement
for an in-place norm call. Future serving integration would need to preserve
all observable aliases and consumers.

## Numerical and performance gate

At M=128,4096,8192,16384 and N=5120, compare every quantized value, FP32 scale,
updated residual, and normalized BF16 value to the deployed two-operation path.
Inputs include zero and exactly cancelling rows, and gamma offsets that would
be lost by BF16 `weight+1`. Failed variants are recorded without timing.
Accepted variants use four or eight warps. Timing restores both input and
residual before each call; reset work is outside the event interval. The graph
test captures reset separately from the operation to avoid accumulating state.

GDN's unquantized `in_proj_ba` still needs a BF16 norm output. The `emit_norm`
variant writes that output once while avoiding its reread by quantization.
The `quant_only` variant avoids that output only where consumer analysis proves
it unnecessary. Model dispatch, residual ownership, and all-consumer correctness
remain unresolved; even exact boundary output is not a model TTFT result.

Root runs this only when GPU4 is reserved:

```sh
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 timeout 180s \
python3 /bench/patch/ttft_norm_quant/bench_norm_quant.py \
  --output /bench/result/ttft/issue3-attention/norm-quant.json
```

## Root设备筛选结论

`norm-quant-recovery.json`显示残差先BF16舍入的候选与原生很接近，但没有满足预先声明的逐值门槛；16384 tokens量化约3.6e-7比例相差1，norm也有少量BF16末位差。保持门槛，不计时、不接入服务。

随后基于native width8符号的8组有限运算顺序探针也全部不exact；4096 tokens量化分别有9/20/15/27/15/26/15/26个不同元素。结果`result/ttft/issue3-attention/norm-order-probe.json`。停止该候选的盲目搜索；需要更完整native算术证据才重开，不放宽阈值伪造通过。
