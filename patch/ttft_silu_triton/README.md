# Register-resident BF16 SiLU / dynamic INT8 screen

Standalone candidate for project Issue 3. It is informed by the fusion problem
reported in [official vLLM Issue #60499](https://github.com/vllm-project/vllm/issues/60499),
not by transferring the author's NVIDIA performance numbers to BW100.

The deployed baseline executes `lightop.activation.fuse_silu_and_mul`, writing a
BF16 `[M,8704]` tensor, then `lightop.quant.per_token_quant_int8`, reading it again.
A single Triton program per row keeps this intermediate in registers and could
remove `4*M*8704` bytes of global-memory traffic plus one launch. At M=8192 this
is 272 MiB of nominal intermediate write/read traffic; cache effects and register
spills can change actual HBM traffic. This is a source-level traffic opportunity,
not a measured bandwidth or model improvement.

The vendor `lm_fuse_silu_mul_quant` also dispatches one native fused operator.
There is no evidence that this candidate removes more traffic than that operator.
The new experiment tests a distinct, inspectable Triton implementation with
4/8/16-warps layouts. The large masked 16384-column tile can increase register
pressure; compiled register and spill counts are recorded when exposed by Triton.
No production model imports this helper.

Exact quantizer specification, read from the installed lightop source:

* SiLU is calculated in FP32 and cast to BF16. That BF16 SiLU result is promoted
  to FP32 for multiplication by `up`, and the product is cast to BF16 again.
  Both intermediate roundings are required to reproduce the vendor activation.
* `absmax = max(max(abs(bf16_activation_as_fp32)), 1e-10)`.
* FP32 `scale = absmax / 127`.
* `q = nearbyint(activation * (127 / absmax)).to(int8)`; nearest-even rounding.
* The deployed source has no explicit saturation clamp. Finite inputs produce
  the signed range [-127,127]. A zero row produces zeros with scale 1e-10/127.

The initial candidate rounded only after multiplication and did not match the
vendor activation. Its historical results remain in `silu-triton.json`; they are
not exact candidates. The separate `probe_activation.py` experiment at 4096
tokens found that Torch and Triton formulas with **SiLU rounded to BF16 before
multiply, followed by a second BF16 rounding**, matched native activation,
quantized output, and scales element-for-element. Both `tl.exp` and
`libdevice.exp` variants matched in that probe. This established the correction;
no numerical tolerance was relaxed.

The benchmark compares every INT8 output and FP32 scale against the deployed
baseline and **admits timing only when both are exactly equal**. A mismatch is
recorded as `NUMERICAL_MISMATCH_NOT_TIMED`. It separately checks eight rows using
an independent CPU PyTorch reference with both BF16 roundings. Samples include
zero inputs, large finite gates, and half-step quantization values. It records
exactness, mismatch fraction, quantization steps, scale error, and reconstructed
output error. A successful numerical screen remains narrower than model quality
or TTFT validation; it does not justify service integration by itself.

Both eager and CUDA-graph timings use the same immutable input and allocation
semantics. Graph captures ten calls per replay; five timing samples each replay
ten times. Eager samples execute twenty calls. The fixed input shapes are
M=128,4096,8192,16384 and width 17408 (output width 8704). There is no FP16, FP8,
FP4, inline PTX, service change, or benchmark-case substitution.

Only the experiment owner runs this with GPU4 reserved:

```sh
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 timeout 180s \
python3 /bench/patch/ttft_silu_triton/bench_silu_int8.py \
  --output /bench/result/ttft/issue3-attention/silu-triton-rounded.json
```

If the candidate is not faster than both the unfused and native-fused alternatives
at prefill sizes, reject it. Positive microbenchmark results still require actual
model correctness, dispatch evidence, and the unchanged matched TTFT cases.
