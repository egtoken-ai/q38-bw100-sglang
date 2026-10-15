# TP2 dense MLP / RCCL overlap feasibility

先验证硬件是否有可利用的计算/通信重叠窗口，不接服务、不改模型、权重精度或测试case。

## 官方能力与当前缺口

2026-10-15 重新访问 [SGLang 官方 TBO operations strategy](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/batch_overlap/operations_strategy.py)：支持列表包含 DeepSeek、Qwen3MoE、MiMo 等专用层路径，没有 Qwen3_5 dense decoder；DeepSeek 路径还显式断言 sparse layer。不能直接开启 TBO flag 宣称本模型支持。先前从 vendor 镜像读取并保存的 `patch/ttft_gdn_views/qwen3_5.original.py` 也没有对应 TBO operation 接线。

[vLLM DBO 官方设计](https://docs.vllm.ai/en/latest/design/dbo/) 描述的重叠点主要在 MoE 的 dispatch/combine，按两个 microbatch 保存独立 forward context 和 metadata；其公开运行示例为 DP+EP。可借鉴调度机制，但 dense TP2 all-reduce 需要独立适配，不能把其他拓扑的收益迁移为 BW100 结论。

环境读取限制已解除：主 agent 恢复了 sleep-only `qwen38_bw100_ttft_prepare`。本次通过只读命令核实该容器的 `operations_strategy.py` 同样仅识别 DeepSeekV2、Qwen3MoE、MiMoV2，没有 Qwen3_5 dense；同时读了实际 Qwen decoder、MLP、communicator 和 hybrid backend。没有启动 GPU 或修改运行服务。

## 微测内容

`bench_tp2_mlp_overlap.py` 使用已验证的 vendor `lightop.gemm_ops.triton_scaled_mm`、per-token INT8 quant、BF16 SiLU。随机 INT8 分片权重使用真实 TP2 dense MLP 形状：gate/up N17408,K5120；down N5120,K8704。每个子批包括输入 BF16 all-reduce、MLP、输出 BF16 all-reduce。

比较三种执行方式，相同总tokens、输入和权重：

1. 单batch处理全部token。
2. 两个microbatch串行执行。
3. 两个microbatch分阶段执行：AR(inA)、AR(inB)、compute(A)、AR(outA)、compute(B)、AR(outB)。

第三种使用两个计算stream和同一个RCCL process group。两rank的 collective 发起顺序完全一致；`Work.wait()` 在消费数据的stream执行。compute(B) 等待 compute(A) 的完成事件，因而只研究通信/计算重叠，不引入同时执行两个大GEMM的竞争。outA 的通信可以与 compute(B) 重叠；最终默认stream等待两计算stream，再拼接输出。

pipeline 与 serial 要求逐位一致；serial 与 single 另比较 BF16误差、relative L2，允许不同M选择GEMM实现的微小末位差异，超界则两rank一起停止，不记录性能收益。所有计时使用相同固定迭代数，防止rank因自适应计时执行不同 collective 数量。记录最慢rank的GPU事件时间和wall时间，并轮换三种模式顺序。没有使用多线程，没有KV/SSM缓存，所以这只能证明MLP/通信的重叠潜力。

```bash
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_overlap/bench_tp2_mlp_overlap.py \
  --output /bench/result/ttft/issue3-attention/mlp-overlap.json
```

默认 micro tokens=2048、4096、8192，对应总tokens=4096、8192、16384。可先 `--micro-tokens 4096 --iterations 3 --repeats 3`。`--trace-dir` 可额外输出4096子批的单次串行/重叠CPU+GPU trace，trace不计入性能样本。必须由主 agent 确认GPU空闲后串行启动。

## 若微测通过，服务集成仍需独立完成

只有相对“单batch”也有收益才有继续实现的依据；仅比串行拆批快可能只是抵消拆批开销。服务接入必须为两个独立请求集合分别保存/切换 attention backend `forward_metadata`，确保KV及SSM slot互不重叠，保证每层每rank collective顺序一致，并处理生命周期、残差和异步stream的tensor存储释放。Qwen3.8的64K单请求存在GDN状态递推依赖，不能把其连续token段当成两个独立请求；应保持现有路径。此micro完全不验证上述服务行为。

当前仅语法检查通过；未运行GPU，未接服务，不宣称存在BW100实测收益。


## 更小的候选：只重叠 MLP 输出通信

新增 `bench_mlp_only_overlap.py`，不修改原全边界微测。它没有输入 all-reduce；两个 rank 使用相同 BF16 输入，模拟已经完成 attention AR 和 post-attention norm 的状态。三个对照仍为单batch、串行双batch、重叠双batch；候选只把 AR(outA) 与 MLP(B) 重叠。TP 权重保持分片，输出拼接保持原 token 顺序，数学上不需要在请求边界拆分逐token的MLP。

```bash
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_overlap/bench_mlp_only_overlap.py \
  --micro-tokens 4096 --iterations 3 --repeats 3 \
  --output /bench/result/ttft/issue3-attention/mlp-only-overlap.json
```

实际镜像源码给出的最小可行集成范围：

| 路线 | 源码接入点 | 元数据/状态影响 | 预计范围与门槛 |
| --- | --- | --- | --- |
| 只重叠 MLP 输出 AR | `qwen2_moe.py:Qwen2MoeMLP.forward`；两个 Qwen3_5 decoder 的 dense MLP 调用处传入受限启用条件；独立 helper 管理 streams/works | Attention/GDN 仍整批执行，完全不拆 `ForwardBatch`、KV、SSM；residual 不参与 MLP，原样交回 postprocess | 约3处源码接线，外加一个helper，预计200–400行原型；先验证MLP-only micro相对单batch有净收益 |
| post-attention AR+norm+MLP 一起重叠 | 两个 decoder 的 `prepare_mlp` 边界；复用实际 GemmaRMSNorm；MLP output显式异步AR | 仍可保持整批Attention/GDN；必须保存每个半批的residual，并保持norm的原位更新/舍入语义 | 额外拆解communicator当前分支，较上一方案复杂；全边界micro没有norm，不能当作精确服务收益 |
| 完整两路decoder/layer交错 | `Qwen3_5ForCausalLM.forward` 层循环、两类decoder、ForwardBatch切分、HybridLinearAttnBackend、GDN/FA backend metadata | 要管理每子批metadata与KV/SSM slots，处理异步存储和capture等状态 | 明显超过小补丁范围；除非较小方案已证明硬件重叠有实测价值，否则不优先 |

源码位置与关键语义已经核实：

- `qwen3_5.py:253–263` 的 GDN `out_proj` 和 `:767–776` 的 FullAttn `o_proj` 都是 `reduce_results=False`。attention 输出AR在 `LayerCommunicator.prepare_mlp` 选择的 `_gather_hidden_states_and_residual` 中执行，不能误以为 projection 内已经完成。
- `communicator.py:_gather_hidden_states_and_residual` 在 TP2、DP1 普通路径先 `attention_tensor_model_parallel_all_reduce`，再 `layernorm(hidden_states,residual)`。若是特殊融合、quant communications、DP/scatter路径，不能直接绕过。
- `Qwen2MoeMLP.forward` 调用 `down_proj(... skip_all_reduce=should_allreduce_fusion or use_reduce_scatter)`；普通dense路径则由 `RowParallelLinear.forward` 做 TP all-reduce。MLP-only helper 必须显式 `skip_all_reduce=True` 并自行提交且等待输出AR，不能保留原AR后再做一次。
- 最小候选只能在 `should_allreduce_fusion=False`、`use_reduce_scatter=False`、DP1、TP2、非spec eager prefill、非tuple BF16输入下命中；其余走原路径。为保持64K C1行为，可在decoder以 `forward_batch.batch_size>=2` 控制启用。这是启用策略，不是MLP数学上要求独立请求边界。
- 复用 `get_tp_group().device_group` 提交异步AR需与两rank命中条件完全一致，并确认没有其他同group异步collective穿插；不能另取默认WORLD组。输入/输出/残差tensor的所有者须活到消费stream完成。
- 完整layer交错时，`HybridLinearAttnBackend.init_forward_metadata` 会依次覆盖两个子backend的 `forward_metadata`；Mamba metadata包含 `query_start_loc`、`mamba_cache_indices`、track indices。仅复制外层 `ForwardBatch` 不能隔离这些共享可变状态。

结论：MLP-only 在结构上可实现、无需更改attention状态管理，是本轮最小可行路线；实际能否抵消拆批引起的GEMM效率损失仍由micro决定。原全边界微测若比serial快、却不比single快，也不能据此进入服务集成。当前新增脚本仅语法检查通过，未运行GPU。


## GPU 首轮筛选结果：暂不接入 MLP-only

主 agent 已在 BW100 GPU4/5 串行执行并回收结果；以下为局部计算段相对同总token单batch的延迟提升，不是模型TTFT提升。

| 总tokens | MLP-only overlap 提升 | 无norm双边界 overlap 提升 |
| --- | --- | --- |
| 4096 | −8.48% | −3.85% |
| 8192 | −0.54% | +4.54% |
| 16384 | +0.68% | +6.24% |

证据：`result/ttft/issue3-attention/mlp-only-overlap-recovery.json`、`mlp-overlap-recovery.json`。MLP-only 在8192总tokens仍无净收益，16384也只有0.68%，不足以承担新服务路径的复杂度，因此不接入。双边界微测说明某些shape确有重叠窗口，但4.54%/6.24%是该局部段的结果，不能声称补齐每case≥20%的端到端目标。相对串行拆批有提升也不能替代与单batch比较。

## 下一阶段：包含原 GemmaNorm 和 residual 的微测

`bench_norm_mlp_overlap.py` 直接使用镜像 `lightop.norm.gemma_fused_add_rmsnorm`，与当前 `GemmaRMSNorm.forward_hip` 的 HCU、非AITER、residual非空分支一致：传入原 BF16 weight 和 epsilon=1e-6，不预先把 `1+weight` 舍入为另一份gamma。输入为各rank独立的attention partial output，residual和norm weight在两rank一致；输入AR后进行原GemmaNorm、INT8 MLP及输出AR。所有模式共享输入/residual工作buffer，每次公平执行相同两次reset copy。

比较单batch、串行2/4子批、流水2/4子批。流水的同group通信顺序在两rank均为 `in0...inN-1,out0...outN-1`，用event保证各子批norm/GEMM按序执行，使其主要探索通信重叠；输出拼接不打乱token顺序。Norm输出与residual输出都要求相对单batch逐位一致；流水最终MLP输出相对对应串行拆批逐位一致；不同M相对单batch还检查BF16容差及relative L2。额外验证residual确实相对初值发生更新，避免无效检查。

```bash
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_overlap/bench_norm_mlp_overlap.py \
  --total-tokens 8192 16384 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/norm-mlp-overlap.json
```

该新微测只通过语法检查，未运行GPU。它尚不验证scheduler、KV/SSM、真实权重准确性或完整模型收益。服务接入仍以公平单batch对照和两case端到端验收为准。


## GemmaNorm + residual + MLP 实测与32768定点补测

主 agent 已完成 `result/ttft/issue3-attention/norm-mlp-overlap-recovery.json`：两rank、所有模式的norm、residual、MLP输出校验均通过，记录的mismatch为0；residual相对输入确实更新。局部计时如下：

| 总tokens | 单batch ms | pipeline 2 ms | 对单batch提升 | pipeline 4 ms | 对单batch提升 |
| --- | --- | --- | --- | --- | --- |
| 8192 | 9.8798 | 9.6886 | 1.93% | 10.0249 | −1.47% |
| 16384 | 19.5436 | 18.7222 | 4.20% | 18.1922 | 6.91% |

这些是包含输入AR、原GemmaNorm、residual更新、MLP、输出AR及公平buffer reset的局部结果。16384时4子批更好，但尚未达到本轮10%的局部筛选门槛，也不等于模型TTFT提升。服务尚未接入。

实际候选服务当前使用 chunk32768，因此仅在允许集合中增加32768，默认测试集合保持8192/16384；不泛扫其他批大小。补测命令：

```bash
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_overlap/bench_norm_mlp_overlap.py \
  --total-tokens 32768 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/norm-mlp-overlap-32768.json
```

32768局部提升若达到10%，才考虑受限prepare_mlp+MLP服务原型；是否满足用户每case TTFT≥20%，仍必须经过固定模型case的匹配A/B验收。新增尺寸只通过语法检查，GPU补测由主 agent 串行执行。
