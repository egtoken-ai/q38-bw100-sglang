# Token-grouped GDN norm/gate and optional INT8 quantization

Standalone experiments only; no production installer or model edits. This is
distinct from the rejected residual GemmaNorm fusion: GDN's deployed Triton
arithmetic is visible, has no residual addition, and uses a standard weight
without adding one.

Reference paths in the audited image:

* `sglang/srt/models/qwen3_5.py`: GDN calls `self.norm(core_attn_out,z)`, reshapes
  its BF16 result to `[T,3072]`, and invokes `self.out_proj`.
* `sglang/srt/layers/attention/fla/layernorm_gated.py`: `RMSNorm` calls the
  actual `rms_norm_gated` / `_layer_norm_fwd_1pass_kernel` path. With the installed
  projection-view patch, Z stays `[T,24,128]` with strides `(8192,128,1)`.
* `lightop/_lmslim_native/layers/gemm/int8_utils.py`: `per_token_quant_int8`
  consumes the BF16 `[T,3072]` output before W8A8 projection. Exact source hashes
  are checked by the diagnostic.

The formulas derive from the deployed SGLang/FLA implementation, whose original
[upstream source is FLA layernorm_gated.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py).
Current GDN norm already groups up to four head rows per CTA. At large T it
launches approximately `T*24/4 = 6T` CTAs, not `24T`. The candidate maps one token
to a padded `32×128` tile and uses only the valid 24 heads: `T` CTAs. The existing
headwise FP32 sum-of-squares, rsqrt, weight multiply, and `z*sigmoid(z)` order are
retained. Only the final norm/gate output is rounded to BF16; no intermediate
BF16 SiLU rounding is inserted.

Two independent candidate modes are screened:

1. **Norm-only:** write the same BF16 output, then retain the original separate
   INT8 quantizer. This can be integrated by replacing only the inference norm
   call if full output equality and real speedup are established.
2. **Norm+quant:** keep the rounded BF16 result in registers, then perform the
   exact original FP32 amax/scale and nearest-even INT8 quantization across all
   3072 values of that token. No FP4/FP8/FP16 conversion is used. A debug output
   also materializes norm to check every value before any timing.

The norm-only inference path omits the original unused rstd output allocation
and stores. It is not a replacement for training/backward code. Different
Triton layouts can still change FP32 reduction order, so matching formulas is
not accepted as proof: both modes must exactly match native BF16 norm, all INT8
values, and every FP32 scale at both seeds. Mismatching variants are not timed.
The quant-only specialization is checked separately. Each path is compared
against the actual vendor `RMSNorm` module plus `per_token_quant_int8`.

```sh
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 timeout 240s \
python3 /bench/patch/ttft_gdn_norm_quant/bench_gdn_norm_quant.py \
  --output /bench/result/ttft/issue3-attention/gdn-norm-quant.json
```

The default shapes are T=4096/8192/65536, with two seeds and real projection
strides. `--tokens 4096,8192` can perform an initial bounded screen; it does not
replace the final full-shape checks or fixed serving cases. Timings distinguish
norm-only work from the complete norm-plus-quant boundary. The original norm
module's unused rstd allocation is included in the baseline, as in serving.

## Minimal possible integration after measurement

Prefer the norm-only path if sufficient: guard ordinary BF16 TP2 prefill with
24 heads, dimension128, norm-before-gate, no bias, and standard weight; replace
only `self.norm(core_attn_out,z)`, reshape to its original return shape, and
retain `out_proj`, GEMM dispatch, and reduction unchanged.

The fused-quant path needs an explicit prequantized `(q,scale)` argument at the
`RowParallelLinear` / compressed-tensors boundary. The current RowParallelLinear
does not offer a generic prequantized-input argument: its fused-SiLU switch
computes SiLU itself and must not be reused on already-normalized GDN data.
The smallest clear change would add a dedicated optional keyword and forward it
to the W8A8 scheme's existing quantized-GEMM branch, retaining dtype, shape,
GEMM whitelist, bias/reduction, and symmetric-memory semantics. Merely setting
global RMS/SILU flags would not wire this safely. No such integration is included
until a fully exact, faster boundary is measured.

## BW100 decision: reject both routes

The device screen is recorded in
`result/ttft/issue3-attention/gdn-norm-quant.json`. The norm-only variants at four
and eight warps matched full norm/quant/scale exactly but were slower. At 65,536
tokens, eager medians were:

| Path | Norm-only ms | Complete norm+quant ms |
| --- | --- | --- |
| Original vendor | 1.530603 | 2.280796 |
| Token-grouped norm, 4 warps | 1.735123 | 2.485100 |
| Token-grouped norm, 8 warps | 1.805707 | 2.555852 |
| Fused norm+quant, 8 warps | — | 2.588476 |

The eight-warp fused variant was exact but slower. The four-warp fused variant
failed exactness and was not timed: the 65,536-token first seed had 543 differing
norm values and nine differing INT8 values, despite exact scales. This failure
is preserved; no tolerance was widened. Graph measurements gave the same
performance decision. Neither route is selected for production, no installer
is provided, and this bounded sweep is closed without expanding the search.
