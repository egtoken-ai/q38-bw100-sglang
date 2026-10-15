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


## 后续筛选的保留与拒绝

- INT8 ASM 实际入口对 M8192 的4种投影形状、11个vendor合法mode逐一隔离验证，共44项数值检查通过；最优mode8相对当前默认约-0.1%至+1.3%，不足以证明收益，不安装替换配置。
- SiLU+INT8多调用/graph复测：8192 tokens约0.7191→0.7159ms，低于1%；GemmaNorm+INT8约0.7500→0.7705ms且量化不一致，均不启用。
- QuickReduce INT8采用原生BF16、关闭FP16转换，目标三档全部更慢；8192 tokens RCCL1.8329ms、Quick INT8 2.0440ms，相对L2误差约0.00914。拒绝该有损候选，不进入模型。
- GDN post-conv prepare融合在6种布局/长度上通过检查；token-major8192阶段0.3751→0.1469ms，Q/K最大绝对差0.001953125，V/g/beta逐值一致，小序列chunk输出最大绝对差0.00012207、state差0。候选采用4warps，补丁位于 `patch/ttft_gdn_prepare/`。
- 更新后的views微测确认真实GPU conv返回QKV两臂stride均(5120,1)，未产生布局退化；8192 tokens含split/conv/gating/norm约1.292→0.847ms，仍不包含完整recurrence。

真实模型组合候选 `sglang-bw100:ttft-gdn-prepare-w4-issue3` 固定套件已通过长度/成功数检查，用时275秒；原生GEMM/RCCL保留，仅叠加packed、GDN views/prepare及32K服务分块。原始记录仍在 `result/ttft/issue3-attention/` 和 `result/quick/`，未纳入Git。


## 组合候选模型结果和命中证据

| 固定 case | 安擎基线 TTFT P50(s) | GDN组合 TTFT P50(s) | TTFT 提升 | TPS | TPOT(ms/token) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/1K C64 | 17.778396 | 16.392254 | 7.80% | 692.312 | 76.328 |
| 64K/1K C1 | 15.906568 | 10.380586 | 34.74% | 25.556 | 29.020 |

结果目录：`result/quick/ttft-gdn32k/20261015-155735-OZY6Pf/`。部署快照保存实际argv/env和镜像ID；未达到逐case20%的总体目标。

4K C1诊断trace确认48次 `fused_gdn_prefill_prepare_kernel` 和16次Hygon FlashAttention，证明模型实际进入移植路径。累计GPU kernel时间543.6ms，当前INT8 GEMM占40.24%，RCCL主kernel占22.30%；它仍可能包含一次额外decode，不能直接当C64墙钟分解。

4个简单确定性问答仍与原生相同且答案正确；4K/64K随机token的greedy序列不相同。共同历史teacher-forcing原生参考已采集。组合候选随机4K/64K的平均绝对logprob差分别约0.334/0.186、最大约1.575/0.544；packed单独候选4K出现更大单点差。原生prefill与自身逐token decode也有明显差异，不能把这种诊断直接当成任务精度分数；完整精度回归仍未通过验收。

一阶段BF16 custom all-reduce通过真实双向P2P/IPC和四档精确求和，但8192tokens约2.196ms vs RCCL1.835ms，更慢；不接入服务。保持真实PCIe拓扑，未伪造full_nvlink。


## 有界 GEMM 与通信边界筛选

- hipBLASLt实际原生调用只请求一个heuristic且workspace为0。4个投影形状分别枚举32算法、workspace0/128MiB后，增加workspace未显示明显收益。新C ABI prototype曾因alpha/beta scalar ABI错误失败，已改为INT32(1/0)并在4种形状与ASM逐值一致；FP32 vector scales不变。修正后的原型与既有hipBLASLt速度一致，不安装额外native库。
- 既有hipBLASLt在M8192的up/QKV/out形状约比ASM快3%–4%，down略慢；不能把该小范围GEMM收益声称为模型20%。
- RS→局部GemmaNorm/INT8 quant→AG INT8+FP32 scale边界在三档全部逐值一致，边界自身提升仅2.07%/6.27%/11.12%（M4096/8192/16384）。考虑GDN未量化BA消费者和跨层残差接入成本，暂不进行复杂模型改造。
- 新Triton SiLU原型首轮遗漏SiLU后、乘up前的BF16舍入，已通过独立探针定位；不会放宽阈值接入错误公式。保留原失败数据，修正后必须quant与scale全量逐值一致。


## 环境恢复后的验证（2026-10-15）

Docker环境短暂不可用导致有界Triton GEMM父流程中断；恢复后镜像、容器、模型和项目目录均已返回。重新测原生镜像，避免将不同环境的结果直接相除。固定两档和轮次不变，新原生基线用时303秒：

| case | 安擎基线 TTFT(s) | TPS | TPOT(ms/token) |
| --- | ---: | ---: | ---: |
| 4K/1K C64 | 17.762641 | 659.018 | 79.725 |
| 64K/1K C1 | 15.901140 | 19.954 | 34.619 |

记录：`result/quick/ttft-native-recovery/20261015-171727-99PbBj/`，实际镜像/argv/env快照在deployment.json。

- 修正SiLU两次BF16舍入后，四档量化/scale全量逐值一致，但8192阶段0.7395ms vs原生0.7169ms，拒绝接入。
- MLP-only输出AR重叠三档提升−8.48%/−0.54%/+0.68%，拒绝接入。
- 加入输入AR、真实GemmaNorm和残差的完整MLP边界，在32768 tokens四子批提升9.87%（38.857→35.023ms），输出/norm/residual逐值一致；这只支持继续原型接入，不代表模型TTFT提升。
- 新直接KV打包移除逐请求index_select/permute复制/cat，12组布局/长度/tile全量K/V逐值一致。8×4096打包1.052→0.489ms；64K单序列1.317→0.974ms。候选位于`patch/ttft_kv_pack/`，需要模型测试。
- GDN CPU元数据候选参照官方vLLM已合并#38361，提前构造chunk索引/offset和initial-state mask，避免旧GPU长度转CPU及逐层重复构造；官方记录本身没有证明TTFT改善，因此本项目独立验证。

整体Goal仍未达到每个case≥20%的要求；恢复前7.80%/34.74%只作为历史结果保留。


## 直接KV打包与CPU元数据组合模型

| 固定case | 安擎基线 TTFT(s) | 优化 TTFT(s) | TTFT提升 | 优化TPS | 优化TPOT(ms/token) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/1K C64 | 17.762641 | 16.179167 | 8.91% | 691.267 | 76.704 |
| 64K/1K C1 | 15.901140 | 10.380501 | 34.72% | 25.618 | 28.925 |

结果`result/quick/ttft-metadata-pack/20261015-173302-OByAtv/`，镜像`sglang-bw100:ttft-metadata-pack-issue3`。CPU metadata日志实际命中；K/V及metadata局部逐值检查均通过。6项输出回归与上一GDN32K父候选相同，但与原生的完整精度差异尚需实测任务评估。当前短档仍不达标，不将此前原生随机token数值差异掩盖为精度通过。


## 四compute-stream MLP候选的模型反例

真实两类decoder层边界微测输出/残差/norm逐值一致且提升9.62%/9.72%，但完整模型短输入TTFT16.179→54.584s，长输入（guard回退）10.371s。固定套件372秒完成，保存失败数据`result/quick/ttft-mlp-pipeline/20261015-175041-GvKM3e/`。该版本拒绝作为可用优化。

另行CPU-only诊断使用9×4096输入/1输出，仅用于定位，绝不替代固定验收case。两个TP rank的32K step主机提交耗时10.45/10.99s，其中aten::empty累计9.91/10.47s、最长单次883/1044ms；allreduce API CPU累计仅78–82ms。注意这些inclusive CPU事件嵌套，不能将算子总计相加。证据`result/ttft/pipeline-cpu-diagnostic/analysis.json`。

结合完整服务约93%设备显存占用、微测每轮synchronize，优先验证跨stream分配/延迟复用导致的压力，而不是把微测收益直接外推。后续先审查单compute-stream异步通信或严格双向fence的lifetime修正；保留原失败版本和原case要求。


## 当前stream修正与GEMM组合

连续64层、8GiB可用余量微测确认旧版每轮10/11次分配重试与全流同步，修正版0次；最终输出/残差均逐值一致。12组实际INT8模块全量exact，9组满足8轮全快且中位提升≥2%的白名单门槛。组合镜像`sglang-bw100:ttft-current-gemm-issue3`已真实命中两条路径。

270秒固定suite结果：短TTFT15.091597s（提升15.04%）、长10.335549s（提升35.00%）；TPS710.645/25.561，TPOT75.236/29.056ms。结果`result/quick/ttft-current-gemm/20261015-181937-dugWWz/`。短输入尚差约0.881秒才到20%，继续验证连续Prefill先处理输出、内部chunk/子批选择，不改case。全部130个请求输出hash与metadata+KV父候选相同；这不替代与原生模型的任务精度回归。


## 连续Prefill结果处理顺序

镜像已有官方 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 选项：连续Extend时先处理上一批结果，再提交下一批。保持Decode overlap、32K chunk和9形状GEMM白名单。实际模型环境确认为1，固定suite270秒完成。

| 固定case | 安擎基线TTFT(s) | 优化TTFT(s) | TTFT提升 | 优化TPS | 优化TPOT(ms/token) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/1K C64 | 17.762641 | 14.714893 | 17.16% | 708.211 | 75.895 |
| 64K/1K C1 | 15.901140 | 10.350391 | 34.91% | 25.501 | 29.133 |

结果 `result/quick/ttft-prefill-release/20261015-183754-FJen1s/`；正式测量65/65输出hash与metadata+KV父候选相同，预热59/65相同（短档6个不同），不能称全部130请求一致。相对current-stream+GEMM父配置短TTFT进一步提升约2.50%，但单次测量不证明全部来自该调度开关；需要最终重复A/B。每个case20%的总体Goal仍未达成，短档距阈值14.210112s还差约0.505秒。


## 64K分块、8子批与QK归一化融合

官方SGLang #27656的BF16 QK Norm+gate拆分融合，在真实stride、两seed、0/129/4K/8K/32K/64K上逐值一致；32K准备阶段3.0366→1.6813ms，64K6.0459→3.3513ms。64K/8真实两类MLP边界也全量逐值一致。组合镜像 `sglang-bw100:ttft-qk-gate-64k-v2-issue3` 的实际模型已命中两路径。v1因Python后缀模型备份被注册器扫描失败，v2改为非Python后缀，失败启动日志保留，无v1性能数据。

固定suite269秒：短TTFT14.386166s（提升19.01%）、长10.319667s（提升35.10%）；TPS716.696/25.571、TPOT75.162/29.056ms。内部chunk/max-prefill65536、parts8、mem_fraction_static0.80，KV容量852608tokens；case与精度不变。正式65输出hash与32K父候选一致，预热59/65一致。证据 `result/quick/ttft-qk64-v2/20261015-185539-CLjdvC/`。目标仍未达成，短档还差0.176054s。

必须保留批次阶梯的统计取舍：短档均值17.324586s，原生17.746731s，上一32K候选16.023994s；P95则从上一候选28.747709到27.893919s。64K批次为1+16+16+16+15，排序第32/33请求都落在第二大批。P50改善不代表均值同比改善，不按首批人数操纵分位数；首批聚合草稿已暂停。后续优先改善相同批次的真实执行时间，例如已确认GPU4/5同属NUMA4的本地绑定。


## NUMA与后续有界筛选

GPU4=PCI98:00.0、GPU5=9b:00.0，均属NUMA4；显式 `--numa-node 4 4`、`SGLANG_NUMA_BIND_V2=0` 在模型/池初始化前使用preferred策略。两rank实际CPU集合和numa_maps均验证。268秒固定suite短14.364530s（提升19.13%）、长10.300595s（提升35.22%），全部130输出与qk64父候选一致；额外短档0.15%单轮差异不作为稳定NUMA收益。

64K/16子批相对当前已安装GEMM的8子批，两类层分别慢1.61%/1.90%，5/5轮均更慢，虽然所有输出/残差/norm逐值一致，仍拒绝生产。新GDN按token归一化和融合量化也被拒绝：逐值一致版本64K更慢，四warp量化融合有数值差异。FA输出sigmoid×attention融合保留中间BF16舍入，有限BF16编码穷举及三档两seed全量一致，64K局部2.515677→1.427483ms；仅与已通过门槛的64K输出投影GEMM一起进入新模型候选，尚不把局部收益当作TTFT。

一次17请求/单输出token的诊断误用profile API活动名CUDA，只产生CPU事件；保留 `result/ttft/numa64-prefill-profile/`，不能据此报告GPU kernel占比。已按当前vendor映射修正为GPU，后续单独采集，不替代固定case。


## 后处理组合模型

`sglang-bw100:ttft-postops-issue3` 在同样64K/8和NUMA配置上，新增精确FA gate与仅一个64K out GEMM形状。268秒固定suite：短TTFT14.303938s（提升19.47%）、长10.273190s（提升35.39%），TPS716.360/25.628、TPOT75.288/29.015ms。两TP rank日志确认两个新路径命中；130个预热/正式输出hash全与NUMA父候选相同。短档还差0.093825s，整体Goal未完成。记录 `result/quick/ttft-postops/20261015-192750-UC3IQs/`。

后续验证固定生产8子批下的默认/4/8 RCCL通道，比较真实Norm/MLP边界绝对耗时，检查通信与GEMM竞争；不改batch组成或统计口径。原生OpenCompass API客户端已隔离准备，官方源码SHA、评分器已知答案、reader64及CLI均验证；同样本64题模型A/B尚在执行，不把客户端可用当作模型精度通过，也不把子集当完整1319题或四数据集准入。
