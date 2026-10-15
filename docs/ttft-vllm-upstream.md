# vLLM 官方优化与 BW100 镜像缺口核查

核查日期：2026-10-15。对应 LoopX goal `q38-bw100-gdn-fusion`、[项目 Issue #3](https://github.com/egtoken-ai/q38-bw100-sglang/issues/3)。本轮只读核查官方仓库、官方 Issues/PR 和现有容器源码，没有运行 GPU 实验。候选仅考虑 BF16、INT8 及 FP32 累加；用户所说 BF32 仍需由具体底层库确认，不能直接视作 TF32 或新的 PyTorch dtype。

本地参考 checkout `/Users/max/Project/local-inference-lab/vllm` 的 HEAD 为 `47ccf6c57d92f03630ebcbad3809450545825488`，存在未跟踪用户文档，未修改该 checkout。该 HEAD 仅作为已读源码快照，不称为远端最新 main。上游 Issue/PR 状态通过 GitHub API 实时核对。镜像路径均相对 `/usr/local/lib/python3.10/dist-packages/sglang/srt/`，容器为 `qwen38_bw100_sglang_opt`。

## 优先候选

| 优先级 | 官方来源与状态 | 镜像缺口 | 实施边界与收益判断 |
| --- | --- | --- | --- |
| P1 | [Issue #60499](https://github.com/vllm-project/vllm/issues/60499)，Open，2026-10-07 更新 | Qwen MLP 未调用已存在的 SiLU×mul+INT8 量化参数通路；RMSNorm→INT8 同样可融合 | 优先测试大 M；上游报告现有 fused CUDA 在 M4096 比分立调用更慢，不能凭 fused 名称启用。当前 BW100 vendor kernel 已由主任务另行测量 |
| P1 | [PR #37813](https://github.com/vllm-project/vllm/pull/37813)，merged，`4eefbf9609e5ddb996e3ac37e192e92466ec35cc` | GDN conv 后仍分别做 split/view、gating、Q/K L2Norm；可把布局与归一化一起合并 | 纯 Triton 实现可作为移植基础。另一路 agent 负责 `patch/ttft_gdn_views/`，本报告不重复实现 |
| P2 | [PR #44700](https://github.com/vllm-project/vllm/pull/44700)，merged，`fa27d4e9cf3c8d8a5a143f38c346b27c02b2c2e3` | 镜像 `GDNBackend.forward_extend` 没有显式将混合 batch 中 decode 前段分离到 recurrent kernel | 先确认当前调度确实生成 mixed batch。仅启用 chunked prefill 不足以证明此路径有收益；普通独立 prefill/decode 无收益 |
| P2 | 同一 vLLM GDN metadata builder 的 CPU 预计算设计 | 镜像 `prepare_chunk_indices` 对 GPU 长度 tensor 调 `.tolist()`，导致每个新 metadata 的同步 | 镜像有 tensor identity cache，不能声称每层都同步。可用已有 CPU 长度生成并异步拷贝 chunk indices，作为低风险小补丁 |
| P3 | [PR #47070](https://github.com/vllm-project/vllm/pull/47070)，merged，`b6cc46ec3b903c71405f4355c1e9ecb47ae54bb2` | 该 PR 放宽 MoE sequence parallel 对 DP 的要求 | 当前是 dense 27B，不能直接套用该 PR 或其速度数字；需独立重排 dense TP 的 reduce-scatter/norm/all-gather，工程量与正确性风险较大 |

## INT8 激活融合：官方尚未落地的建议

[#60499](https://github.com/vllm-project/vllm/issues/60499) 提出动态 per-token INT8 模型缺少 RMSNorm 量化融合 pass，并给出 add+RMSNorm+INT8、SiLU×mul+INT8 两类 Triton 实验。报告来自 RTX 4090 和 Qwen3-1.7B，不是 BW100 或本模型验证；其 prefill 收益不能作为当前验收结果。

本地 `vllm/compilation/passes/fusion/rms_quant_fusion.py` 的 `FUSED_OPS` 和注册逻辑确实以 FP8 为主，不能把已有 `rms_norm_dynamic_per_token_quant(..., torch.int8)` 算子入口等同为模型自动命中。相关 [PR #38026](https://github.com/vllm-project/vllm/pull/38026) 已关闭且未合入；其 INT8 decomposition 包含 NVIDIA PTX cast，不能原样移到 HCU。

镜像已有 plumbing：

- `layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py:274` 的 `apply_weights` 接受 `input_quant_args` 与 `silu_quant_args`；只有相应环境开关和参数同时存在才跳过 `per_token_quant_int8`。
- 因此应在 Qwen dense MLP 和 norm→linear 调用点建立严格的 W8A8/BF16 guard，而不是笼统打开环境变量。
- 数值验证必须覆盖 BF16 中间舍入、INT8 舍入与饱和、per-token scale、Gemma 的 `1+weight`、残差就地更新及别名；融合数学表达式相同也不保证 INT8 payload 位相同。
- 单独比较算子的大 M4096/M8192 耗时后，再跑固定两个正式 case；小 batch 融合有效不代表 prefill 有效。

## GDN conv 后融合：具体可移植入口

上游文件：[fused_gdn_prefill_post_conv.py](https://github.com/vllm-project/vllm/blob/4eefbf9609e5ddb996e3ac37e192e92466ec35cc/vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py)。入口 `fused_post_conv_prep`，kernel `_fused_post_conv_kernel`。它直接读 conv 输出，写连续的 Q/K/V 和 FP32 的 g/beta，同时做 Q/K L2Norm、softplus、sigmoid。许可证为 Apache-2.0，应保留声明。

镜像对应链：

1. `layers/attention/linear/gdn_backend.py:474` split mixed_qkv，随后 view Q/K/V。
2. 同文件 `:503` 单独 `fused_gdn_gating`。
3. `layers/attention/linear/kernels/gdn_triton.py:168` 传 `use_qk_l2norm_in_kernel=True`。
4. `layers/attention/fla/chunk.py:101` 和 `:102` 分别执行 `l2norm_fwd`。外层 `input_guard` 还可能为不连续的 Q/K/V 做复制。

当前每个 TP rank 的形状为 H=8、HV=24、K=V=128。移植需注意上游 kernel 仅显式使用 token stride，默认最后一维连续；镜像 conv 后 transpose 的 stride 必须实测或额外传入 feature stride。融合后必须禁止后续重复 L2Norm。还需保持 beta dtype 与旧路径一致、检查 `rsqrt`/`sqrt`、epsilon 和 softplus threshold 差异，并验证 chunk 输出及 SSM 状态，而非只比较准备出的 Q/K/V。

上游 PR 自身的端到端示例改善有限，不能据此承诺补足当前短输入 case 所需的约 17 个百分点。该候选适合与减少多余布局复制、INT8 激活融合叠加。

## GDN mixed batch 与 metadata

[PR #44700](https://github.com/vllm-project/vllm/pull/44700) 把混合 batch 中单 token decode 分离，避免每条 decode 在 chunk 路径占一个 64-token chunk。它同时在 metadata builder 预计算 prefill 的 indices/offsets，并重定位 prefill tail。这里只能迁移算法思想；上游 B200/MoE 的数据不能代替 dense BW100 测量。

镜像 `layers/attention/fla/index.py:21` 的 `prepare_chunk_indices` 调 `triton.cdiv(...).tolist()`；`layers/attention/fla/utils.py:103` 的 `tensor_cache` 按 Python tensor identity 缓存最近四组输入，因此同 batch 跨层可复用。不要把这一处写成每层一次同步。vLLM 当前 `vllm/v1/attention/backends/gdn_attn.py:373` 在 CPU 构造 metadata，再通过 `async_tensor_h2d` 传入；镜像已有 `extend_seq_lens_cpu` 可用来实现同类方法。

[Ascend PR #7967](https://github.com/vllm-project/vllm-ascend/pull/7967) 以 Triton 清 SSM 状态替代 host-bound boolean index update；镜像 GPU GDN 路径直接传 SSM pool 和 cache indices，未发现同一条 vLLM `initial_state[~mask]=0` 语句，所以不把该修复列为当前缺口。

## 当前不采用的方向

- Blackwell FlashInfer/CuTeDSL GDN、FP4/FP8 attention 或 GEMM 不符合本设备与 dtype 约束。
- [Issue #54919](https://github.com/vllm-project/vllm/issues/54919) 的长上下文 decode 饥饿属于诊断线索，环境是另一种多节点/模型组合，且报告含本地 scheduler 修改。不能将其视作已定位的当前根因。
- [Issue #36627](https://github.com/vllm-project/vllm/issues/36627) 提到冗余 zero init；主任务已审查并优化旧 GDN 初始化。不能机械地把所有 `zeros` 换为 `empty`，必须证明每个被读取元素已写入，特别是 graph padding 和 masked state。
- 上游 sequence-parallel、通信融合数据大多来自 MoE、FP8 或特定 AMD/NVIDIA 通信 kernel；当前 BW100 的 BF16 QuickReduce 已有主任务微测，无明显 prefill 优势，不能把相邻技术的宣传速度移作证据。

## 相同历史的数值比较工具

新增 `scripts/check_ttft_teacher_forcing.py`，不自动启动服务或改变容器：

```bash
python3 scripts/check_ttft_teacher_forcing.py \
  --snapshot result/ttft/native-model-outputs.json \
  --output result/ttft/native-teacher-forcing.json

python3 scripts/check_ttft_teacher_forcing.py \
  --snapshot result/ttft/native-model-outputs.json \
  --reference result/ttft/native-teacher-forcing.json \
  --output result/ttft/candidate-teacher-forcing.json
```

两次均输入相同随机 prompt 和原生 16-token continuation，通过 `/generate` 的 input logprobs 比较相同历史条件下的分数，避免 greedy 分叉之后比较不同历史。工具请求 `logprob_start_len=prompt_length-1`，严格校验返回 17 项、首项 None、完整 token IDs、prompt 长度，再取后 16 项。该偏移已对照镜像 `scheduler_output_processor_mixin.py:720/732` 核查。结果保存每 token 差、最大绝对差、平均绝对差和平均有符号差，不设置伪精度通过阈值。

注意 teacher forcing 会让 continuation 作为 prefill 处理，不能直接与旧 greedy decode logprobs 相减并称作优化误差。因此原生和候选都要用同工具测量。工具已通过 dry-run、离线错位拒绝和差值统计检查；本 agent 未向服务发起推理请求。完整任务精度仍需独立真实数据评测。
