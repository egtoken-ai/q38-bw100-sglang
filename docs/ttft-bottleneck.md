# BW100 Qwen3.8-27B TTFT 瓶颈与 20% 优化目标

跟踪：Issue #3。目标是在同口径原生基线下，使 4K/1K C64、64K/1K C1 的 TTFT P50
均降低至少 20%。本文件记录诊断和候选选择，不表示目标已达成。

## 已测到的瓶颈

2026-10-15，在当前 GDN 清零优化镜像上，通过 SGLang `/start_profile` 采集 CPU/GPU
事件。设备为 BW100 gfx936，GPU 4/5、TP=2；真实 INT8 checkpoint，BF16 计算/KV cache，
Triton attention，chunked-prefill-size=8192，无 radix cache。模型为 48 层 GDN、16 层
Full Attention；TP 后全注意力为 12 个 Q heads、2 个 KV heads、head_dim=256。

诊断请求只输出 1 token，先预热后采集，目的是观察 Prefill；不能将其作为 1K 输出验收分数。重叠调度可能仍产生一个额外 decode step，不能把事件总和当成纯 Prefill 墙钟时间。
下表为 TP0 的 kernel、GPU memcpy/memset 事件累计时长占比，跨 stream 可能重叠，
不是严格互斥的墙钟耗时分解，也不代表 TTFT 可按占比直接缩短。

| 设备事件类别 | 4K C1 | 64K C1 |
| --- | ---: | ---: |
| Full Attention `_fwd_kernel` | 5.59% | 48.74% |
| INT8 GEMM 主 kernel | 36.03% | 21.92% |
| RCCL kernels | 20.55% | 11.89% |
| GDN chunk kernels（状态更新、KKT/求逆、W/U、输出等） | 9.45% | 5.48% |
| 其他（归一化、量化、激活、复制等） | 28.38% | 11.97% |

64K trace 中 Full Attention 累计约 7.756 秒，INT8 GEMM 约 3.489 秒。
KKT＋求逆 kernel 仅占约 1.14%，所以只继续优化这个小阶段无法解释或支持 20% 整体目标。
4K C1 用来观察短输入算子结构，不能代替 C64 的调度/排队分析。

证据位于忽略目录 `result/ttft/issue3-profile/`，含两 rank 的压缩 trace、请求记录和配置。
4K C64 的首次高并发 trace 采集在启动 profiler 后服务无响应，已终止并单独标记；
该次诊断不是有效性能结果，不能混入均值。

## 候选优先级

1. **64K：Full Attention。** 检查镜像自带 Hygon FlashAttention 的 H256/GQA 正确性和耗时，
   与当前 Triton Extend、不同 tile 配置直接对照。保留 prefix 的右下角 causal 语义。
2. **短输入：INT8 GEMM、RCCL 与 Prefill 调度。** 测量模型 TP2 真实投影形状，比较当前
   lightop、hipBLASLt、rocBLAS、Triton 路径；再评估 chunk 大小，避免只改善长输入。
3. **剩余开销：量化/归一化融合及 GDN。** 按实测占比决定是否继续投入，不以局部微基准
   代替模型结果。不得采用改变模型结构、截断上下文或增加缓存命中的方式达标。

参考材料已通过 LoopX doc registry 登记：SGLang 官方 Benchmark and Profiling、
vLLM 官方 Optimization and Tuning、FlashAttention-2（arXiv:2307.08691）、
Gated Delta Networks（arXiv:2412.06464）。论文和通用实现仅提供候选依据；
CUDA 专用实现、其他 GPU 测量不能直接视为 BW100 收益。

## 验收

使用 `scripts/run_bench_quick.sh` 的两个固定 case，每档预热 1 轮、正式 1 轮，
请求数=并发数，每次快速筛选目标 8 分钟以内。原生快速基线已补齐（20261015-144145-arnCYw）：
4K C64 TTFT P50=17.778秒，64K C1=15.907秒；最终按相同工作负载交替重复至少 3 次，检查 TTFT、TPS/TPOT 和正确性。
单轮细小变化、profile 下的延迟或此前 3 轮完整基线与 1 轮 quick 的差异都不能证明达标。

## 初步算子筛选结果

- 注意力：TP2 H256/GQA6、query8192/prefix57344，Triton中位数113.65ms，Hygon FlashAttention39.25ms。5种形状的输出对照通过；小形状另用FP32 SDPA参考。仅是算子收益，不计为模型收益。
- GEMM：按真实投影形状及三档M，hipBLASLt与原生lightop接近；正确匹配布局后的rocBLAS与纯Triton更慢，暂不替换。
- 原始实验与异常记录在 `result/ttft/issue3-attention/`。候选镜像 `sglang-bw100:ttft-fa-issue3` 已构建，尚待模型验证。

- BF16 QuickReduce FP（关闭BF16→FP16转换）与RCCL在GPU4/5的四种张量大小上逐值一致；8192×5120的80MiB张量约1.85ms vs1.87ms，收益不足，因此第一轮模型候选保持RCCL不变。

## 第一轮模型候选：仅切换 fa3

两档均完成，但 TTFT 未改善：4K C64 17.867秒 vs原生17.778秒；64K C1 16.268秒 vs原生15.907秒。64K输出TPS有所提高，但这不满足TTFT目标。

接入检查发现，镜像内的 packed paged-KV→连续FlashAttention路径额外要求 `minimax_opt`。Qwen并未启用这一模型专属模式，所以原语测量的高速路径没有进入该候选。下一步在不打开MiniMax其他行为的前提下，为Qwen TP2 H256/GQA6增加显式开关及形状/阶段保护，并检查分页顺序、尾页和因果语义。

模型输出对照：4个基础问答一致且答案正确；4K/64K随机token输入的greedy输出序列不同，尚不能宣称模型精度回归通过，后续需进一步比较真实文本及teacher-forced logprob。


## 第二轮模型候选：Qwen 专属 packed Prefill

将现有 vendor paged-KV 打包路径从 MiniMax 专属入口扩展到显式开启的 Qwen TP2 H256/GQA6 Prefill；不启用 MiniMax 的其他模型行为。补丁位于 `patch/ttft_prefill/`，镜像 `sglang-bw100:ttft-packed-issue3`。

| 固定 case | 安擎基线 TTFT P50(s) | packed TTFT P50(s) | TTFT 提升 | 验收 |
| --- | ---: | ---: | ---: | --- |
| 4K输入/1K输出 C64 | 17.778396 | 17.239791 | 3.03% | 未达到20% |
| 64K输入/1K输出 C1 | 15.906568 | 10.943413 | 31.20% | 单轮达到20%，待重复验证 |

两档输入/输出长度和成功请求数均通过。该表仅为单轮筛选，不代表 Goal 达成；随机 token greedy 输出仍不完全一致，需共同历史的 teacher-forced logprob 对照及进一步精度验证。原始结果：`result/quick/ttft-packed/20261015-150851-5I2oGQ/`。

## 上游移植和短输入筛选

按用户授权，在同一 LoopX Goal 内由三个 subagent 分别核实 SGLang 上游、vLLM 上游、镜像融合路径；GPU 4/5 实验由主 agent 串行管理。

- vendor SiLU+量化首轮 BF16 数值逐值一致，但大 M=4096/8192 的单次事件计时未见收益；补充循环/graph 测量后再决定是否接入，不因算子名含 fused 就启用。
- vendor Gemma RMSNorm+量化大 M 未见收益，且量化值约7%不同，暂不接入。
- 后续候选仅用 BF16/INT8，FP32 用于累加或独立参考；不采用 FP4/FP8/FP16。用户所称 BF32 尚未映射成框架 dtype，不能当作已验证的计算模式。
- 16K prefill chunk 完成固定两档筛选：4K C64 17.005514s（提升4.35%）、64K C1 10.731927s（提升32.53%），仍未整体达标。仅改变服务分块，未改变输入、输出、并发、轮次或缓存设置。


## 上游 GDN views 的设备验证

`patch/ttft_gdn_views/` 移植官方 #36267/#38806 的投影视图与3D Z归一化读法，限制HCU、TP2、16/48 heads、D128、非speculative BF16 Prefill。未照搬gfx950调参。镜像 `sglang-bw100:ttft-gdn-views-issue3` 已构建。

BW100上长度[63,129]、[4096]、[4096,4096]、[8192]的投影拆分、gating、两种归一化顺序、causal conv输出与最终状态检查通过；归一化输出最大绝对差为0。8192-token准备阶段约0.762ms→0.299ms。此计时不包含完整GDN或模型，不能换算为TTFT已达标。

RCCL显式Ring/Simple和默认几乎一致，Tree/Simple更慢：8192×5120 BF16 all-reduce分别1.828、1.829、1.978ms；全部精确求和检查通过。保持默认通信参数。注意profiler的AllGather命名kernel主要属于all_reduce实现，不能据kernel名字认定模型做了大规模logits gather或INT8量化通信。
