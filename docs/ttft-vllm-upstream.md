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

## 补充核查：Dense Qwen27B sequence-parallel norm 未采用

LoopX goal `q38-bw100-gdn-fusion` / Issue #3，2026-10-15。本项仅做源码核查和接线设计，没有实现或启用模型补丁，没有执行 GPU 实验。

结论：镜像没有可直接给当前 dense Qwen3.8-27B、HCU TP2 安全启用的现成开关。BF16 reduce-scatter → 局部 GemmaRMSNorm → BF16 all-gather 可以单独实现，不改变 TP 权重和 GEMM 的 token 输入；但它不减少 ring all-reduce 的理论通信字节，只把 norm 工作减半。按已有 4K C1 profiler 的 GPU 事件和估算，理想收益约 1.89%，且未计额外 launch/分配成本，不能单独支撑短 case 提升 20%。

## 已有代码及不能直接启用的原因

源码路径相对 `/usr/local/lib/python3.10/dist-packages/sglang/srt`：

| 入口 | 实际条件 | 当前模型结果 |
| --- | --- | --- |
| `server_args.py:7124 --enable-attn-tp-input-scattered` | `layers/communicator.py:268 AttnTpContext.init_context` 要求 CUDA 或 NPU、`q_lora_rank != None`、非 NSA、TP>1、非 DP attention、MoE a2a none、非 dense fully-DP、禁用 piecewise graph、非 EAGLE3 | HCU 不满足平台 guard；Qwen3_5 也没有 MLA q_lora_rank。模型未调用 `init_context` 或 `maybe_input_scattered` |
| `AttentionInputs.fetch_hidden_states/fetch_qkv_latent` | 为上述 MLA scattered 输入在 consumer 之前 all-gather | Qwen3_5 的 Full Attention 直接 `qkv_proj(hidden_states)`，GDN 直接 `_forward_input_proj`，不经过此 consumer；硬开 context 会使 token 维与 positions/cache metadata 不一致 |
| 两类 Qwen3_5 decoder 的 `LayerCommunicator(...allow_reduce_scatter=True)` | 只是能力声明，实际由 `should_use_reduce_scatter` 选择；dense scatter mode 默认 FULL，DP1 时 FULL 与 TP_ATTN_FULL group size 相同 | 当前不会自动进入所需 sequence-parallel norm |
| `moe_dense_tp_size=1` | `enable_moe_dense_fully_dp` 使 dense mode SCATTERED | 不能把它当本模型开关；Qwen dense MLP 的 Column/RowParallelLinear 仍按 TP 切权重，需另行处理权重/布局，不满足本次“不复制权重”的边界 |
| `--minimax-opt` | 明确 MiniMax M2 专用 sequence-parallel prefill | 不用于 Qwen；此前 packed attention 已用独立窄 guard 接线 |
| `enable_sp` / `enable_mlp_tp_scatter` | 镜像 `server_args.py` 未发现这些字段 | 不存在可直接启用的相应开关 |

已有 `CommunicateWithAllReduceAndLayerNormFn._tp_all_reduce_with_scattered_residual` 也不等于这里的方案：它把局部 residual 加入全 token partial buffer，再 all-reduce 得到 full residual，随后做 full norm（`:1186–1190`），不是 RS→半 token norm→AG。

## 性能上界

已有 `result/ttft/issue3-profile/4k-c1/kernel-summary-tp0.json`：

- GPU 事件耗时和 607.31 ms。
- Gemma `fused_add_rms_kernel_opt` 22.96 ms，占 3.78%。即使完全线性减半，最多省 11.48 ms，约事件和 1.89%。这只是估算，没有扣除额外通信调用、分配、局部 slice/copy 成本。
- RCCL 事件约 20.5%，不能算作该方案可消除的时间。Ring AR 可看作 RS+AG；TP2、full tensor 大小 S 时，每 rank 的理论发送量都是约 S。独立调用 RS/AG 还可能比库内 AR 更慢。
- profiler 的事件和含重叠，可能含额外流水 decode；4K C1 也不是正式 C64。因此上述百分比不是 TTFT 可证明上界，只用于判断候选优先级，正式收益必须测固定 case。

INT8 AG 可以进一步减少 norm 后传输量，但涉及量化 consumer 接线。GDN 的 ba 是未量化 BF16 consumer，不能直接用 INT8 输入替代，也不能为它再 AG 整个 BF16 hidden 而隐瞒额外通信。保持精度需要局部 ba 投影及小输出 AG 等额外设计。该方案不属于当前最小补丁。已有有损 QuickReduce INT8 实验也不能用作替代。

## 最小模型接线设计

只针对 `Qwen3_5LinearDecoderLayer` 与 `Qwen3_5AttentionDecoderLayer` 的 communicator 实例采用专用 subclass/wrapper，其他模型保持原逻辑。不要全局篡改 `AttnTpContext` 或 generic LayerCommunicator。

一个 prefill step 内保持以下不变量：

- 每个 ColumnParallel GEMM、attention、GDN 都仍接收 full-token hidden `[T,5120]`。
- RowParallel 的 output 为各 rank 的 full-token partial `[T,5120]`。
- residual 仅存本 rank 固定连续 token 段 `[T/2,5120]`，按 TP rank 顺序。不要按 sequence 边界分配 shard；norm 是逐 token 操作，token shard 可以穿过 sequence 边界，attention/GDN 在 AG 后仍使用原 metadata。
- 全程 BF16 hidden/residual，Gemma norm 保持 vendor 原算子；不改量化权重、输入量化舍入或 accumulation 规则。

接线顺序：

1. 首层 `prepare_attn`：embedding 已完成 all-reduce，是 full replicated hidden。只取本 rank token slice，保留其 residual，然后局部无残差 Gemma norm、AG normalized hidden。**禁止对 embedding 再 RS，否则会把相同值累加 TP 次。**
2. 所有层 `prepare_mlp`：对 attention/GDN `reduce_results=False` 的 partial output 做 BF16 RS；与局部 residual 调用原 Gemma residual norm；AG normalized hidden 给原 MLP。保留更新后的局部 residual。
3. MLP 调用 `use_reduce_scatter=True`，使现有 `Qwen2MoeMLP.forward` 通过 `down_proj(...skip_all_reduce=True)` 返回 partial output。这个 flag 本身并不执行 RS。
4. 非末层 `postprocess_layer` 保留 partial hidden/full T 与局部 residual，交给下一层 `prepare_attn` 做 RS→local norm→AG。必须禁止与旧 allreduce-fusion 标记叠加。
5. 末层恢复原模型 final norm 需要的 full hidden/full residual。最保守做法是对末层 MLP partial output 做原 BF16 AR，并把局部 residual AG 到 full，随后保留 `Qwen3_5Model.forward` 现有最终 Gemma norm。这仅在末层多出一次 residual AG；不可每层都 AG residual，否则持续增加通信量。

初始化 guard 应精确检查 dense `qwen3_5_text`、hidden5120、64层、TP2、DP1、PP1、BF16、GemmaRMSNorm、当前 W8A8 checkpoint、两类 decoder、MLP down_proj reduce_results=True、attention out_proj reduce_results=False。模型层面一次决定启用，不能运行到中间层才因 shape 不符回退，因为 residual 已经是 shard。

每个 forward 的 guard 需基于所有层共享的信息：普通 non-spec extend、无 TBO、无 CUDA graph capture、无 aux/last-layer capture、无 deepstack/multimodal additions、总 T>阈值且可被2整除。不满足时整个 step 使用原路径；decode 必须回退。初版建议不支持 tail odd T，不做隐藏 token padding。

## 验证要求

先两卡微测完整 AR+norm 与 RS+local norm+AG，在同样 full partial inputs、复制 residual 上比较输出。RS 与 AR 的 BF16 规约顺序可能不同，数学等价不代表位相同；记录 normalized hidden 和 residual 的误差，不只比较形状。多层 synthetic 链必须测试初层、非末层、末层恢复，以及不整除 token/decode 的整体回退。

若微测没有明确优势，不制作新的模型镜像。若有优势再接模型，验证全部层实际路径、相同历史 logprobs、真实任务精度，并对固定两个 case 做 matched A/B。由于预计只消除少量 norm 重复计算，该候选优先级低于已经找到的更大 GDN/Attention/调度瓶颈，不能借改并发、输出长度或模型精度放大表面收益。

最终取舍：主任务决定不实现 BF16 sequence-parallel norm。现有配置不能安全启用，预计收益小、涉及跨层 residual 语义，当前优先级不足；这不是证明其在 BW100 一定无收益。

## 补充核查：hipBLASLt 真实 dispatch 与未枚举的算法空间

本项核查了镜像 Python 包装、`lmslimquant` native 符号与函数反汇编、DTK 头文件、`hipblaslt-bench --help`。只按主任务授权把不可执行的 bench 原文件复制至容器 `/tmp/q38-hipblaslt-bench-help` 并为副本增加执行权限；没有更改原安装文件，没有运行 GPU 性能测试。

之前 `patch/ttft_prefill/bench_gemm.py` 中的 `gemm_ops.hipblaslt_w8a8_gemm` **确实调用 hipBLASLt，不是在 Python 层回落到 lightop ASM**：

```text
lightop.gemm_ops.hipblaslt_w8a8_gemm
  → lightop._lmslim_native.quantize.quant_ops.hipblaslt_w8a8_gemm
  → torch.ops.lmslim.blaslt_w8a8_bf16_gemm
  → lmslimquant.blaslt_w8a8_bf16_gemm
  → hipblaslt_gemm::w8a8_bf16_gemm
  → hipblasLtMatmulAlgoGetHeuristic → hipblasLtMatmul
```

native 函数起始地址为该镜像 `lmslimquant.cpython-310-x86_64-linux-gnu.so` 的 `0x27153e0`。`0x271587c` 调用 heuristic，`0x27159c6` 调用 matmul；未见调用 `gemm_w8a8_smooth`。这证明不同 dispatch API，但仍可能选择性能相近的底层 ISA kernel，不能只凭名字认定不同算法已全面比较。

更关键的缺口是该 native wrapper 只请求 **1 个 heuristic solution**，workspace preference 为零，matmul 的 workspace pointer/size 也为零。Python API 不暴露 solution index 或 workspace。已测结果只代表这一选择，没有遍历 hipBLASLt 算法空间。另一个 `hipblaslt_w8a8_channelwise_gemm` v2 wrapper 也请求单个零 workspace heuristic，不应包装成已经完成的全面算法搜索。

精确类型配置也不能误写：旧 wrapper 使用 A/B=INT8、C=INT32、D=BF16、compute=INT32、scale type=FP32、A/B outer vector scales。反汇编 `BlasltMatmulDesc` 初始化是 compute enum 9、scalar enum 0，C layout enum 10、D layout enum 14；对应 DTK `hipblas.h` 和 `hip/library_types.h` 的枚举。DTK `hipblaslt_common.yaml:309` 同样列出此组合。`C/D=BF16 + compute=FP32` 是另一个组合，不是旧微测的精确复现。

已验证本容器 bench 的 `--help` 支持：

- `--scaleA 2 --scaleB 2`：向量 scale。
- `--algo_method heuristic --requested_solution 32`：有界请求32解；`--algo_method all` 也存在，但初筛不建议无界枚举。
- `--algo_method index --solution_index ID`：指定 solution（帮助里的“algo_method 2”是旧描述，枚举值列表支持 index）。
- `--workspace`、`--cold_iters`、`--iters`、`--use_gpu_timer`、`--print_kernel_info`。
- `--api_method mix/cpp` 时还暴露 `--splitk`、`--wgm`；先测基础算法，不混入额外调参。

可由主任务在 GPU 空闲时执行的第一组独立 micro 命令如下。这里交换了矩阵 M/N：原 PyTorch 输出 `[8192,17408]` 为 row-major，等价于 BLAS column-major `[17408,8192]`，weight 作为 A、activation 作为 B：

```bash
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
HIPBLASLT_TUNING_USER_MAX_WORKSPACE=0 \
HIPBLASLT_TUNING_FILE=/bench/result/ttft/hipblaslt-ws0-tuning.txt \
timeout 120 /tmp/q38-hipblaslt-bench-help \
  -m 17408 -n 8192 -k 5120 \
  --transA T --transB N --lda 5120 --ldb 5120 --ldc 17408 --ldd 17408 \
  --a_type i8_r --b_type i8_r --c_type i32_r --d_type bf16_r \
  --compute_type i32_r --scale_type f32_r --scaleA 2 --scaleB 2 \
  --alpha 1 --beta 0 --initialization rand_int \
  --api_method c --algo_method heuristic --requested_solution 32 \
  --workspace 0 --cold_iters 3 --iters 5 --use_gpu_timer --print_kernel_info
```

第二轮可把 workspace 和 `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` 都改为 `134217728`，使用独立 tuning 文件。若仅128MiB workspace 的算法有收益，不能把它写入现有零 workspace wrapper 的 override 后直接称作可用；必须新建显式分配并传入 workspace 的 native 接口。更改 problem shape 时同步改变 lda/ldb/ldc/ldd，不能仅修改 M/N。

镜像 libhipblaslt 的可读字符串包含 `HIPBLASLT_TUNING_OVERRIDE_FILE`、`HIPBLASLT_TUNING_FILE`、`HIPBLASLT_TUNING_USER_MAX_WORKSPACE`、`TENSILE_SOLUTION_INDEX` 和日志变量。其用途与 [AMD 官方离线调优文档](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-6.4.1/how-to-use-hipblaslt-offline-tuning.html) 一致：bench 保存最优选择，override 文件用于运行时读取。**本项只确认库包含相应配置支持，尚未做 runtime 命中验证。** 零 workspace winner 可先在独立 Python micro 进程加载 override，打开 kernel/solution 日志验证真实命中；不能仅因为设置环境变量就宣称生效。

算法筛选必须在同一 INT8 数据、per-token/per-channel scale 和 BF16 输出上重新做数值检查，再做 model A/B。bench 内部纯 GEMM 时延也不含原 Python wrapper 每次重建 descriptor/heuristic 的 host 成本，不能与端到端 TTFT直接换算。现有 ASM mode0–10 已测收益仅约噪声级，不重复该枚举。

标量 ABI 更正：descriptor 的 `HIP_R_32F` scale type 不意味着这个 DTK `COMPUTE_32I` 路径的 alpha/beta 指针应指向float。镜像 `quant_ops.py:716–718` 明确构造 CPU INT32 的1和0传入native，bench符号也有 INT8/BF16/int compute 特化。首版缓存原型误用float标量，已被独立micro的全输出检查拒绝；两处标量改为INT32，向量scale仍FP32，未降低正确性要求。

## 恢复后新增候选：仅保留两个明确源码缺口

以下核查只读取恢复后的 `qwen38_bw100_ttft_prepare`，没有运行GPU或改服务。固定两个正式case、关闭prefix cache等验收条件不变。

**候选一：CPU预建GDN chunk metadata，消除step中的第一次同步。** 官方 [vLLM PR #38361](https://github.com/vllm-project/vllm/pull/38361) 已合并，commit `cb10b7e80b138f876079c2cd697df29f0965b885`；`GDNAttentionMetadataBuilder` 从已有CPU长度计算indices/offsets，异步H2D并作为显式参数穿过FLA链。当前镜像 `layers/attention/fla/index.py:27` 仍在GPU长度上 `.tolist()`，`chunk.py:104`仍在算子内构造；identity cache只避免同step后续重复同步，不能消除第一次。可同时把 `gdn_backend.py:418` 每个GDN层重复的 `extend_prefix_lens > 0` 提前到共享metadata。

此候选是明确消除CPU/GPU串行屏障，不是已有TTFT收益证明：上游PR正式prefill测试Mean TTFT由14.020s到14.274s，作者认为属于噪声；不能拿其局部trace差异承诺当前提升20%。移植需对当前vendor chunk64索引逐元素相同，并确保 `chunk_fwd_o` 等全部consumer接收新metadata，否则仍会走旧缓存路径。非GDN、spec与不支持的mode保留原fallback。

**候选二：把paged KV→varlen的逐请求Python pack改为一个直接gather kernel。** 当前 `pack_paged_kv_to_varlen.py:107–119` 每层按请求循环，反复page indices转long、K/V `index_select`、layout转换/reshape、最后`cat`。各层KV内容不同，不能跳过实际读取；但可以复用该step的page/sequence映射并合并内存遍历，减少CPU launch和临时tensor。

官方参考为 [vLLM ROCm AITER FA 的 cp_mha_gather_cache_kernel](https://github.com/vllm-project/vllm/blob/47ccf6c57d92f03630ebcbad3809450545825488/vllm/v1/attention/backends/rocm_aiter_fa.py)，来源 [PR #25763](https://github.com/vllm-project/vllm/pull/25763)，合并commit `dc937175d496a801f175b5dcce6a8157506bae52`。它在一个Triton kernel里通过token→batch、cu_seqlens、block_table定位并直接写连续K/V。可借鉴算法，但不能原样复制NHD/SHUFFLE布局：当前HCU是K `[pages,Hkv,page,D]`、V `[pages,Hkv,D,page]`，V feature stride尤其不同，必须显式传全部stride。固定BF16，不做dequant或dtype转换；packed输出可要求位相同。需要测随机page顺序、尾页、不同seq长度与前缀，再确认模型原有varlen attention路径不变。

第二项只移动/复用metadata、减少pack开销，没有跳过prefill或隐藏prefix cache命中，适合当前短case的多请求batch；仍须端到端A/B，尚无BW100收益数字。不要把上游AMD backend支持写成该HCU移植已验证。
