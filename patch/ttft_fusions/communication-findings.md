# Communication and duplicate quantization audit

Audited the deployed image and the recorded 4K C1 TP0 trace on 2026-10-15.

* GDN `in_proj_ba` is excluded from INT8 by checkpoint `compression_config.ignore`
  entries `re:.*linear_attn.in_proj_a.*` and `re:.*linear_attn.in_proj_b.*`.
  Its input projection consumes BF16 directly. `in_proj_qkvz` quantizes its input,
  but there is no second identical INT8 input quantization in `in_proj_ba` to reuse.
* Kernel names alone misidentify communication. The large
  `ncclDevKernel_AllGather_Ring_Simple_Sum_i8` events have profiler metadata
  `Collective name=allreduce`, `dtype=BFloat16`, and `In msg nelems=20971520`
  (= 4096 × 5120). These are hidden-state all-reduces. They do not prove an
  application-level INT8 communication path or repeated logits all-gathers.
* `LogitsProcessor` prunes states before LM-head GEMM in ordinary inference, then
  gathers the vocabulary logits. The actual `_all_gather_base` event is small;
  removing logits collection cannot explain the ~70 ms kernel category.
* The deployed `/opt/dtk/lib/librccl.so` contains `NCCL_ALGO`, `NCCL_PROTO`,
  `NCCL_MAX_NCHANNELS`, `NCCL_MIN_NCHANNELS`. Its diagnostic strings explicitly
  say minimum-channel tuning is ignored with fewer than eight GPUs, so that
  knob cannot be assumed to work on our two-GPU deployment.

The official [RCCL environment variable reference](https://rocm.docs.amd.com/projects/rccl/en/docs-7.1.0/api-reference/env-variables.html)
documents algorithm/protocol overrides. That is candidate selection evidence,
not BW100 performance evidence. The local library is a vendor fork and must be
measured directly. Safe screening candidates retain BF16 communication:

1. Default environment, for a same-run reference.
2. `NCCL_ALGO=Ring NCCL_PROTO=Simple`.
3. `NCCL_ALGO=Tree NCCL_PROTO=Simple`.
4. `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MAX_NCHANNELS=8`.

Use a fresh torchrun process per environment because the library caches settings.
Only the root experiment owner may run this after reserving GPU 4 and 5:

```sh
HIP_VISIBLE_DEVICES=4,5 NCCL_ALGO=Tree NCCL_PROTO=Simple \
torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_fusions/bench_rccl_algorithms.py \
  --output /bench/result/ttft/issue3-attention/rccl-tree-simple.json
```

The script checks a nonconstant BF16 sum exactly against an FP32-reference sum
rounded to BF16. Timed repeated collectives use zeros to avoid in-place numeric
overflow, with 20 operations per event pair and five samples. It never changes
service settings or claims a model gain from a collective microbenchmark.
