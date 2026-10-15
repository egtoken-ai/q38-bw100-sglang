# BW100 Qwen3.8-27B TTFT 瓶颈与 20% 优化目标

跟踪：Issue #3。目标是在同口径原生基线下，使 4K/1K C64、64K/1K C1 的 TTFT P50
均降低至少 20%。本文件记录诊断和候选选择，不表示目标已达成。

## 已测到的瓶颈

2026-10-15，在当前 GDN 清零优化镜像上，通过 SGLang `/start_profile` 采集 CPU/GPU
事件。设备为 BW100 gfx936，GPU 4/5、TP=2；真实 INT8 checkpoint，BF16 计算/KV cache，
Triton attention，chunked-prefill-size=8192，无 radix cache。模型为 48 层 GDN、16 层
Full Attention；TP 后全注意力为 12 个 Q heads、2 个 KV heads、head_dim=256。

诊断请求只输出 1 token，先预热后采集，目的是隔离 Prefill；不能将其作为 1K 输出验收分数。
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
