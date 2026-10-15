# Exact BF16 full-attention output gate screen

Current `qwen3_5.py::self_attention` performs `torch.sigmoid(gate)` on BF16,
materializes that result in BF16, then multiplies the BF16 attention output.
This candidate retains that intermediate cast while eliminating its global
write/read and one launch. It writes a new output; no in-place alias assumption
is introduced. Shape is `[tokens,3072]` for the deployed TP2 full-attention layer.

The [official SGLang fused sigmoid-multiply implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/elementwise/elementwise.py)
motivates the fusion, but currently multiplies the FP32 sigmoid directly.
That formula is not copied because it omits the deployed BF16 intermediate
rounding. This independent candidate tests exactly two sigmoid expressions:
Triton's sigmoid and reciprocal of one plus libdevice exponential. No reduced
precision dtype other than BF16/INT8 is involved; this particular helper only
uses BF16 input/output and FP32 arithmetic.

`bench_fa_gate.py` first enumerates all finite BF16 gate encodings with attention
equal to one. It then compares all outputs at 4096/8192/65536 tokens and two
seeds, including large finite gates. Only a variant passing both exhaustive
sigmoid and full output exactness receives eager/graph timings. It never relaxes
the gate, modifies scheduler batches, or treats these numbers as model TTFT.

```sh
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 timeout 180s \
python3 /bench/patch/ttft_fa_gate/bench_fa_gate.py \
  --output /bench/result/ttft/issue3-attention/fa-gate.json
```

## BW100 screen and selected implementation

`result/ttft/issue3-attention/fa-gate.json` verifies all 65,280 finite BF16 gate
encodings and every element of all three shapes at both seeds. Both expressions
were exact; `tl_sigmoid` was faster and is the only production candidate.

| Tokens | Baseline eager ms | Fused eager ms | Baseline graph ms | Fused graph ms |
| --- | --- | --- | --- | --- |
| 4096 | 0.163984 | 0.095528 | 0.162106 | 0.091711 |
| 8192 | 0.321033 | 0.182984 | 0.319042 | 0.179133 |
| 65536 | 2.515677 | 1.427483 | 2.513833 | 1.423918 |

This is gate-stage evidence, not measured model TTFT or task-quality acceptance.
The integration helper calls the identical tested kernel; its arithmetic was
not changed when the serving guard was added.

## Guarded integration candidate

The Dockerfile derives from `sglang-bw100:ttft-numa4-issue3`. `install.py` checks
the exact parent `qwen3_5.py` SHA256 in `source-manifest.json`, then replaces only
the two-operation gate block with a guarded call and the original fallback.
It retains the existing QK fusion, both MLP hooks, and GDN changes. The rollback
file is `qwen3_5.py.ttft_fa_gate_original`, which the Python model scanner will
not import. No normalization, projection, scheduler, or GEMM code is replaced.

The serving guard requires HCU, dense Qwen27B geometry, TP2 Q12/KV2/H256,
BF16 contiguous matching `[T,3072]` tensors, `4096 <= T <= 65536`, ordinary
`ForwardMode.EXTEND`, no speculative metadata, and no graph capture. Decode,
small health/warmup requests, and all unsupported inputs retain the original
implementation. Small requests remain legal but are intentionally not fused.

```sh
docker build -t sglang-bw100:ttft-fa-gate-issue3 patch/ttft_fa_gate
```

The image enables `SGLANG_QWEN27B_FA_GATE=1`. Set it to `0` before server import
to disable only this candidate. Root owns GPU execution, image construction,
service lifecycle, and unchanged-case model validation. This delivery itself
does not claim that the installed service has been tested.

## 组合模型记录

与NUMA父配置组合后的 `sglang-bw100:ttft-postops-issue3` 固定suite268秒，短TTFT14.303938s（相对安擎原生提升19.47%）、长10.273190s（提升35.39%）。实际分派日志命中，130个输出hash与NUMA父候选逐个一致。结果 `result/quick/ttft-postops/20261015-192750-UC3IQs/`；该组合差异不能单独归因于此一项，短case仍未达标，完整任务精度与重复A/B另验。
