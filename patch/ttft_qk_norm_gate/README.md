# Qwen27B BF16 QK norm and gate extraction

Port of official SGLang [PR #27656](https://github.com/sgl-project/sglang/pull/27656),
commit `0da18f8d916eb2d99b7382851d19426554a74cf3`, plus the zero-grid guard from
[PR #31794](https://github.com/sgl-project/sglang/pull/31794), commit
`6314e9e4f5bdadd975969e8781dd1edcb27bb4a7`. The source manifest records the upstream
file hash and both audited vendor hashes. Upstream's model/hardware performance
numbers are not BW100 evidence and are not used as acceptance results here.

The current full-attention path splits a `[T,7168]` QKV tensor into interleaved
Q/gate width 6144 and K/V widths 512. It materializes flat Q and gate before
normalization; flattening K across tokens also requires a copy at this layout.
The port reads token/head strides directly and writes contiguous Q norm, K norm,
and gate in one kernel. V remains the original view. It retains ordinary RoPE,
attention, `torch.sigmoid(gate)`, BF16 multiply, and output projection unchanged.
It does **not** port upstream's fused sigmoid-multiply operation or FP8 variants.

Unlike the previously rejected native GemmaNorm/quant candidate, the existing
QK norm reference is itself visible Triton source. Its arithmetic is the same as
the port: FP32 Q/K and original weight offset, sum of squares / 256, reciprocal
square root, multiplication by FP32 `(weight+1)`, then one BF16 output cast.
Expression association and the default Triton launch settings are preserved.
Actual all-element equality still gates every measured candidate.

The model hook is opt-in and restricted to HCU, dense `qwen3_5_text`, hidden size
5120, TP2, Q12/KV2/H256, BF16 input/weights, nonempty ordinary `ForwardMode.EXTEND`,
no speculative metadata, and no separate fused-RoPE flag. Other modes run the
original branch. Decode and graph-capture paths retain their original behavior.
The narrow installer indents the original branch under an `else`; it preserves
the parent image's GDN views/prepare and both decoder MLP hooks.

Run the standalone device screen before building/enabling a service candidate:

```sh
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 timeout 180s \
python3 /bench/patch/ttft_qk_norm_gate/bench_qk_gate.py \
  --output /bench/result/ttft/issue3-attention/qk-norm-gate.json
```

The screen compares the real vendor split/reshape/QK norm path against the port
at 4096, 8192, and 32768 tokens with two seeds, plus padded row-stride and empty
input cases. Q, K, V, and gate must be exactly equal and V must retain its view.
Only exact production-shaped cases receive eager/graph timing. Those timings
cover preparation only, not complete attention or model TTFT.

```sh
docker build -t sglang-bw100:ttft-qk-norm-gate-issue3 patch/ttft_qk_norm_gate
```

The default parent is `sglang-bw100:ttft-prefill-release-issue3`; the audited
`sglang-bw100:ttft-current-gemm-issue3` is also usable via `BASE_IMAGE` when its
SHA checks match. The Dockerfile explicitly enables this candidate. Roll back
only this path by setting `SGLANG_QWEN27B_QK_NORM_GATE=0` before server import.
An installer hash mismatch is a stop condition, not permission to overwrite
other patches. Fixed-case service validation and correctness remain mandatory.

## Installer correction

The first service startup failed because the rollback file was named
`qwen3_5_before_qk_norm_gate.py`. SGLang scans Python files in the model directory,
so this backup registered the same Qwen model implementations a second time.
The installer now writes `qwen3_5.py.ttft_qk_original`, which is not a Python
module candidate. Rebuild from the clean audited parent; this correction changes
only the backup filename, not the vendor baseline or helper arithmetic.

## BW100 设备验证

Root已在保留的GPU4串行运行真实布局micro：0/129（padding64）/4096/8192/32768/65536，非空各两个seed。Q、K、V、gate全量逐值一致、V别名保持，任何失败记录后终止。8192阶段eager耗时与其余结果见原始JSON；32768从3.036592ms到1.681268ms，65536从6.045929ms到3.351328ms。该结果仅包含QKV拆分、QK归一化和gate提取，不能换算为模型TTFT提升。

证据：`result/ttft/issue3-attention/qk-norm-gate.json`、`qk-norm-gate-64k.json`。实际组合镜像构建使用 `--build-arg BASE_IMAGE=sglang-bw100:ttft-current-parts-issue3`，保留父层的调度开关和9形状GEMM白名单。服务测试设置chunk65536/parts8/mem_fraction0.80，模型已确认命中，固定suite269秒：短TTFT14.386166s（提升19.01%），长10.319667s（提升35.10%）。正式65输出hash与32K父候选相同，预热59/65相同；完整任务精度与最终重复验收尚未完成。64K分块使短档均值17.324586s高于32K父候选16.023994s，但仍略好于原生17.746731s；不能将P50改善等同均值改善。结果 `result/quick/ttft-qk64-v2/20261015-185539-CLjdvC/`。
