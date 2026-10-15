# TP2 residual/norm/quant boundary diagnostic

Issue: https://github.com/egtoken-ai/q38-bw100-sglang/issues/3

This is a standalone microbenchmark, not a serving patch. It compares:

1. BF16 AllReduce of partial outputs, full residual addition + Gemma RMSNorm,
   and per-token INT8 quantization.
2. BF16 ReduceScatter, local-half residual addition + the same Gemma RMSNorm,
   the same INT8 quantization, then separate AllGather operations for INT8
   activations and FP32 token scales.

Both use the deployed `lightop.norm.gemma_fused_add_rmsnorm` and
`lightop.quant.per_token_quant_int8`, with unchanged BF16 materialization and
FP32 scale dtype. No FP16 conversion or new numerical approximation is used.
The candidate performs normalizer/quantizer work on half the rows and exchanges
INT8 rows instead of BF16 normalized rows. Extra collective launches and memory
allocation can offset that benefit, so only actual boundary timings decide
whether model integration deserves further work.

The fixed shapes are M=4096,8192,16384 and N=5120, TP2. Input partials differ by
rank; full residuals and weights are identical. Rank 0 owns the first half of
tokens, rank 1 the second. The candidate must match the baseline's complete
quantized tensor and scale tensor exactly, and its updated local residual and
normalized values must match the corresponding baseline slices. Both ranks
agree on correctness before timing; a mismatch stops the experiment.

Every timing invocation starts from restored residual/state. Reset copies are
outside the measured interval. Each path runs ten warmups and five batches of
twenty CUDA-event pairs, alternating path order between batches. The reported
median is across the five batch means. The baseline and candidate use ordinary
PyTorch distributed collectives with the same RCCL process group; no topology
checks or communication dispatch guards are modified.

Only the experiment owner runs this while GPU 4 and 5 are reserved:

```sh
HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5 timeout 180s \
torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_sequence_parallel/bench_boundary.py \
  --output /bench/result/ttft/issue3-attention/sequence-parallel-boundary.json
```

Important integration boundary: the candidate leaves the residual token-sharded
and provides a full INT8 activation plus FP32 scales. It does **not** provide a
full BF16 normalized activation. Qwen GDN has an unquantized `in_proj_ba` consumer
in addition to its INT8 `in_proj_qkvz` consumer. Serving integration would have to
preserve that BF16 consumer and residual ownership, potentially adding work or
communication absent from this microbenchmark. Its timing is therefore an
optimistic boundary-only screen, never a directly deployable model TTFT result.
Small or negative measured gain should reject a larger model restructuring.
