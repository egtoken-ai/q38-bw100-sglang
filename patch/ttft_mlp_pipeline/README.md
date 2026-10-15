# Qwen27B TP2 four-way prefill MLP boundary pipeline

**本目录根部的四private-compute-stream原型已在完整模型失败，不作为可用优化。** 短case TTFT达到54.584s；失败证据保留。可用修正位于 `current_stream/`，后续可选4/8子批位于 `current_stream_parts/`。以下原型描述与微测用于追溯，不能代表服务收益。

该候选来自当前 BW100 实测：32768 总 tokens 的原 GemmaNorm + residual + 双边界 AR + INT8 MLP 微测，single=38.857ms、pipeline2=36.856ms（提升5.15%）、pipeline4=35.023ms（提升9.87%），所有输出逐位一致。证据 `result/ttft/issue3-attention/norm-mlp-overlap-32k.json`。这是局部收益，不是模型TTFT结果。

## 最小改动

`qwen3_5.py` 只在 `Qwen3_5LinearDecoderLayer.forward` 和 `Qwen3_5AttentionDecoderLayer.forward` 的 attention 输出之后、原 `prepare_mlp` 之前增加可选入口。helper 返回 None 就继续原代码；命中时执行原 post-attention norm 和原 `Qwen2MoeMLP`，输出交回原 postprocess。

不修改 attention/GDN kernel、backend metadata、ForwardBatch、KV/SSM slots、精度或模型权重。MLP 是逐token算子，按token行连续四分后按原顺序拼回；不拆GDN序列。64K C1和decode明确不命中。

四子批共用同一个 `get_tp_group().device_group`，两rank同序提交 `in0,in1,in2,in3,out0,out1,out2,out3`。各子批的norm/GEMM通过event串行，允许通信与下一子批计算重叠。每个子批调用原 `layer.post_attention_layernorm`，因此保留原Gemma gamma/残差舍入语义。原MLP使用 `use_reduce_scatter=True` **仅用于已审查源码中的skip_all_reduce逻辑**：`Qwen2MoeMLP.forward` 不执行scatter，只把该布尔值传成 `down_proj.skip_all_reduce`；本helper随后显式执行且等待唯一一次输出AR。安装器锁定该MLP和linear源码，防止后续语义漂移造成重复或漏AR。

Residual输出若仍别名原分片，直接返回原residual tensor；否则按token顺序拼接实际norm返回值。不假设返回值永远别名。输入、残差和输出显式记录stream生命周期，默认stream在拼接/返回前等待四个stream。

## Guard 与回退

全部guard成功后才进行分配或collective。主要条件：

- 环境 `SGLANG_QWEN27B_MLP_PIPELINE=true`、HCU、TP2/DP1/PP1，无CP/DP attention；ordinary non-spec eager extend。
- token数≥32768且被4整除，batch_size≥2，hidden=5120，连续BF16输入及独立residual。
- 真实dense `Qwen2MoeMLP`、原 `GemmaRMSNorm`、动态对称channel W8A8 INT8、无bias；真实projection TP2。
- communicator必须是已审查的 `_gather_hidden_states_and_residual` + TP_ATTN_FULL residual，postprocess必须为`_trivial`；没有scatter、deferred/boundary AR fusion、norm/quant融合、量化通信或LoRA。
- 禁用torch compile、piecewise graph、CUDA capture、嵌套TBO和blocking NCCL wait；attention和MLP的TP ranks一致。

回退：`SGLANG_QWEN27B_MLP_PIPELINE=0`。Guard不满足会直接走原路径，不在发生GPU/通信错误后静默回退。命中首例日志为 `QWEN27B_MLP_PIPELINE_HIT`；真实模型验证必须检查该日志，避免把未命中的测试当作优化证据。

## 构建与实际层验证

```bash
docker build -t sglang-bw100:ttft-mlp-pipeline-issue3 patch/ttft_mlp_pipeline
# 主 agent 确认GPU空闲后，在sleep容器运行：
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /opt/ttft-mlp-pipeline/bench_actual_layer.py \
  --model /model --tokens 32768 \
  --output /bench/result/ttft/issue3-attention/mlp-pipeline-actual-layer.json
```

默认父镜像 `sglang-bw100:ttft-gdn-prepare-w4-issue3`。安装器对qwen3_5和所依赖的qwen2_moe、linear、layernorm、communicator、INT8 scheme逐个校验SHA256；保存修改前qwen文件，拒绝覆盖已有安装。

`bench_actual_layer.py` 构造真实 `Qwen3_5LinearDecoderLayer` 和 `Qwen3_5AttentionDecoderLayer`，从真实config读取量化设置，使用它们原生的 INT8 权重参数、weight processing、MLP、norm、communicator，对比原边界与新helper。只随机初始化要执行的MLP权重和norm，不加载完整checkpoint，也不执行attention。它不是完整模型准确性测试。校验output、residual及single-request/decode回退guard；计时使用两rank固定次数和最慢rank，轮换顺序。最终仍须固定两case的实际模型A/B与准确性检查。

服务必须显式保持 eager（例如 `--disable-piecewise-cuda-graph`），其余启动参数沿用当前best，包括chunk32768。此目录不修改启动脚本或LoopX状态。

本地检查：全部Python语法检查通过；AST确认只改两个decoder的forward，GDN/attention实现没有变化。随后主 agent 完成下述真实层GPU验证；完整模型结果仍需单独验收。subagent 未自行启动GPU或服务。


## 真实层 BW100 GPU 证据

主 agent 已回收 `result/ttft/issue3-attention/mlp-pipeline-actual-layer.json`。M32768、TP2 的两类真实decoder边界均通过：两rank的最终MLP输出、residual以及原位norm缓冲区全部逐值一致，max-abs=0、mismatch=0。residual相对输入有167510765个元素更新，检查不是比较未写入的空缓冲区。

| 真实层类型 | 原边界 ms | 四子批 pipeline ms | 局部延迟提升 |
| --- | --- | --- | --- |
| Qwen3_5LinearDecoderLayer | 38.9333 | 35.1876 | 9.62% |
| Qwen3_5AttentionDecoderLayer | 38.9702 | 35.1834 | 9.72% |

这里执行了真实norm、量化linear、MLP和communicator，但使用随机权重且未执行attention。因此可证明目标设备上的边界实现与数值一致性，不能替代完整checkpoint准确性、两case TTFT或模型dispatch命中验证。主 agent 正在验证默认四子批模型。

## 八子批的有界后续诊断

新增独立 `bench_parts.py`，只允许 M32768 和 parts∈{4,8}，默认parts=4；不修改已构建helper的四子批策略，也不修改Dockerfile/服务。

八子批的数学和依赖图上可行：每子批4096行，都是逐token norm/MLP；row-parallel权重仍在TP两rank分片，各行仍只有一次输入及输出AR。两rank同序提交全部input AR再全部output AR，compute按event顺序执行，没有输入通信依赖后续输出的循环。默认stream在拼接之前等待所有stream；norm/residual仍使用原BF16语义。该结论是源码/CPU审查结论，不能代替设备验证。

有合理但有限的性能动机：4→8可缩短首个输入AR和末尾输出AR的填充/排空时间，4096行的MLP形状此前已经被微测覆盖。但同总tokens的collective数量8→16、norm/GEMM调用数翻倍，消息变小，GEMM效率及Python调度开销可能抵消收益。八个stream的缓存分配也可能增加峰值内存，脚本记录每rank每次计时的peak allocated及相对起点增量。不能假定子批越多越快。

```bash
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_mlp_pipeline/bench_parts.py \
  --total-tokens 32768 --parts 4 8 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-pipeline-parts48.json
```

这个独立micro沿用原GemmaNorm+MLP对照实现：单batch、串行4/8和流水4/8均使用相同总tokens、输入、随机权重与公平reset；norm/residual对单batch要求逐值一致，流水输出对相应串行拆批逐值一致，记录最慢rank的GPU/wall时间。必须先相对单batch有收益，再比较八子批是否优于四子批。当前只通过语法检查，未运行八子批GPU测试。


## 追加实际层诊断：只测M32768/M65536 × parts4/8

针对实际服务chunk候选，新增 `bench_actual_parts.py` 与仅供该诊断导入的 `parts_pipeline.py`。它们不进入Dockerfile安装列表，不替换当前已构建helper。默认仍为M32768/parts4，只允许两个总token数和4/8子批。

这个版本构造真实两类Qwen decoder、真实量化linear/norm/communicator。每个候选的output/residual/norm缓冲区都要求相对同总tokens单batch逐值一致；四子批还与已安装服务helper进行逐值控制检查。所有模式共享公平reset，按最慢rank计时，记录峰值allocated及相对起点增量。诊断用的batch_size取M/4096，与固定4K输入case的批量token组成一致，仍不执行attention。

```bash
# 在现有已安装四子批helper的sleep容器中，主 agent 独占GPU串行执行：
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_mlp_pipeline/bench_actual_parts.py \
  --tokens 32768 --parts 4 8 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-actual-parts-32k.json

HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_mlp_pipeline/bench_actual_parts.py \
  --tokens 65536 --parts 4 8 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-actual-parts-64k.json
```

M65536/parts8保持8192行子批，有可能减少完整模型prefill调度次数，同时保持此前有效的子批形状；这只是待测推断。其单batch参考以及候选需要更大的activation临时空间，不能与占用GPU的推理服务同时运行。若随后服务验证改用chunk65536并将mem_fraction_static设为0.80，应独立记录该配置变动、KV容量、峰值显存与两case结果。这里不修改服务配置，不把chunk变化当作benchmark输入/输出case变化。

新增实际层诊断只通过CPU语法/源码一致性检查，尚无八子批或M65536设备结果。

## 完整模型反例与当前状态

四compute-stream初版在固定suite中失败：短TTFT54.584s（上一候选16.179s），长C1回退10.371s；372秒完成。真实模型命中日志已确认，但不能采用这次退化版本。证据`result/quick/ttft-mlp-pipeline/20261015-175041-GvKM3e/`。

CPU-only诊断（9×4K/1输出，非验收case）看到32K step主机耗时10.45/10.99s，`aten::empty`累计9.91/10.47s，allreduce API仅78–82ms；inclusive事件嵌套，不能相加。完整模型约93%显存占用，而早期单层微测每轮synchronize，跨stream分配的延迟复用压力没有在原micro中体现。保留初版与失败数据，先验证单compute-stream/连续多层受限显存方案，不盲目把四子批扩成八子批。
