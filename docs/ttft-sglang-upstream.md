# BW100 TTFT：SGLang 上游候选与镜像差异

2026-10-15，LoopX goal `q38-bw100-gdn-fusion` / Issue #3。目标是固定 4K 输入、1K 输出、C64 和 64K 输入、1K 输出、C1 两个 case 各自达到 TTFT ≥20% 提升；此文不把其他设备的结果计入验收。只读研究未启动 GPU 任务、未修改运行容器。新增代码位于 `patch/ttft_gdn_views/`，交由主 agent 统一构建、验证。

本地 `/Users/max/Project/sglang` 为 main，head `77ba54f7df57fa47048177fc861757709a1ff6cb`（2026-10-10）；工作树有原有未跟踪文档，未修改。官方 API 当前 main 为 `a776851c97253bf6bb43e9de9dc543cd22d8de73`（2026-10-15 07:16:56Z），因此本地快照不冒充最新 main。镜像代码直接读取 `root@bw100` 的 `qwen38_bw100_sglang_opt` sleep 容器。

| 优先级 | 候选 | 官方证据与状态 | 镜像缺口 | 可落地动作 |
| --- | --- | --- | --- | --- |
| 1 | GDN projection views + 3D Z norm | [PR #36267](https://github.com/sgl-project/sglang/pull/36267)，已合并 `07199fa220cc353a2b5230b8bf47a0dead7ea96c` | qwen3_5.py:475 只允许 ratio 1/2/4 fused unpack；27B ratio3 每层 B/A contiguous、QKV cat、Z reshape copy | 已生成 `patch/ttft_gdn_views` 最小补丁；BF16、HCU、non-spec extend、16/48 heads、D128 gate，独立验证 stride/conv/state/norm |
| 2 | SiLU+mul+动态 INT8 quant | [Issue #31504](https://github.com/sgl-project/sglang/issues/31504) 的 producer→quant 思路，仍 Open；原提案是 FP8，不可直接用 | qwen2_moe.py:195 的 dense MLP 总是 act_fn 后 down_proj；vendor linear/quant 层已有 lightop INT8 融合接口但模型没接 | 以当前 INT8 数据流接线，保留 BF16 中间舍入，对比 fused vendor 算子与基线；由其他 agent 负责实现 |
| 3 | GDN fused prepare：split+L2norm+gating | [PR #38806](https://github.com/sgl-project/sglang/pull/38806)，Open，head `62e3bdfb899cf158d1aecd2c070d3f1b3a4264da` | gdn_backend.py:474 仍 split/view，随后 fused_gdn_gating；chunk 内另做两次 Q/K norm | 移植独立 Triton prepare，再通过 `use_qk_l2norm_in_kernel=False` 防止重复归一化；必须处理 conv 后转置布局，不只 row stride |
| 4 | RMSNorm+INT8 quant、GDN gated norm+INT8 quant | [Issue #31504](https://github.com/sgl-project/sglang/issues/31504) 提出 norm/activation/通信 epilogue 融合 | Qwen GemmaRMSNorm→quant 和 gated norm→out_proj quant 分离 | 逐个 sole-consumer 点接线；Gemma 的 1+weight、residual 别名、量化scale、BF16 rounding 都需要匹配 |
| 5 | Prefill 批量/CPU调度 | [Issue #35851](https://github.com/sgl-project/sglang/issues/35851) 报告另一部署 breakable graph 每层中断造成 GPU idle | 当前镜像未启用同样 breakable graph，不能据此归因；4K C64 admission/queue 时间仍需检查 | 从现有 trace 做 CPU/GPU union 和实际 batch tokens 归因；可测更大 prefill chunk，但固定测试 case 不变 |

上游 #36267 先在 CUDA 启用 views，并调整 gated norm 读 3D Z；其 FlashInfer QKV prepare 部分不属于当前 Triton GDN 后端。当前移植保持 vendor norm 的行 tile 和数学表达式，未直接替换整个上游文件。镜像 `causal_conv1d_triton.py` 按 x.stride 读取，`fused_gdn_gating.py` 按 A/B 行 stride 读取；独立 BF16 验证覆盖这些消费者。可选 HCU 二进制 conv 的 stride 兼容性尚未验证，不能自动扩展结论。[上游改动](https://github.com/sgl-project/sglang/commit/07199fa220cc353a2b5230b8bf47a0dead7ea96c)

#38806 的 fused prepare 保留 V，Q/K 用 FP32 L2 归一化，g/beta 按既有 gate 公式生成；其 AMD chunk_h/chunk_o 调参专门面向 gfx950。BW100 gfx936 不能直接套用性能结论。该 PR 上游使用其他模型、量化和设备的测试，不能证明本项目达到 20%。当前先移植 views，随后若 profile 表明 prepare 成本值得投入，再独立移植和测量。[PR #38806](https://github.com/sgl-project/sglang/pull/38806)

不采用的方向：当前上游 `Qwen2MoeMLP._silu_fp4_quant_fused` 依赖 NVFP4；FP8 producer fusion 实现也不能直接运行于用户限制下。只迁移消除中间读写的机制，用 BF16/INT8 实现。用户“BF32”的库类型尚需主 agent 核实，不等同于私自启用 FP16 或 CUDA TF32。

TP 通信方面，最新 [RFC #43820](https://github.com/sgl-project/sglang/issues/43820) 用复制 MoE 权重配合 sequence parallel，面向小 TP 稀疏 MoE；本模型是 dense 27B，复制 MLP 的权重与计算成本完全不同，不能直接移植其收益。已有 QuickReduce 微测由主 agent 完成，不重复占用 GPU。

当前代码验证：`python3 -m py_compile` 对新 installer、模型、norm 和 microbench 通过；未进行 GPU 或模型验证，未修改 LoopX 的任务状态或宣布目标完成。
