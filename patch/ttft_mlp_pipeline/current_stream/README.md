# Current-stream allocator repair candidate

四个private compute streams的完整模型候选失败，不能采用其微测收益代表服务收益。主 agent 的CPU trace显示32K prefill host耗时10.45/10.99s，其中`aten::empty`累计9.91/10.47s，而c10d all-reduce仅78–82ms。原实现的跨stream分配/回收是当前首要假设；该证据还不能把所有等待精确归因于某一个`record_stream`调用。

本候选只替换pipeline helper，不改变decoder接线、模型、精度、attention、KV/SSM或guard。保留原失败helper和结果。所有norm/GEMM本来已通过event串行，因此把计算全部保留在调用方的真实model_stream，使用ProcessGroupNCCL自己的通信stream进行async AR即可，无需四个private compute streams。

## DAG 与显存生命周期

顺序：在当前model_stream提交全部inAR，逐子批执行`in_work.wait → 原norm → 原MLP(skip原outAR) → async outAR`，最后逐个`out_work.wait → cat → 原postprocess`。inAR的NCCL依赖事件在任何子批compute前记录；outAR(A)等待compute(A)，但后续compute(B)只等待更早入队的inAR(B)，不会等待outAR(A)。通信和compute可以重叠，依赖图没有环。

| 张量 | allocation owner / 使用方 | 释放或复用安全性 |
| --- | --- | --- |
| hidden及其xs视图 | 原attention projection在model_stream分配；PG通信stream做输入AR | async Work保存引用；在原norm写入该视图前`in_work.wait()`把NCCL完成事件接回model_stream；父tensor仍由helper持有 |
| residual及rs视图 | model_stream产生，原norm在同stream读写 | 没有新的非owner计算stream，原位或新输出都保留原norm语义 |
| 模型weights | 加载时创建，模型整个生命周期持有；只在model_stream读 | 不在batch间释放，原模型初始化的同步契约未变 |
| norm/quant/gate/activation临时张量 | 全部在model_stream分配并在同stream消费 | 原stream内的缓存复用保持device队列顺序，不需要跨stream延迟回收 |
| 每子批MLP output | model_stream分配；PG通信stream做outAR | outputs列表保持引用；`out_work.wait()`在cat及释放前接回model_stream；PG同步后解除内部stash |
| 拼接output/residual | model_stream分配或保留原residual别名 | 下一层继续使用同model_stream，原scheduler跨stream输出处理未修改 |

[PyTorch 2.10 record_stream 文档](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.record_stream.html) 说明跨stream记录会影响内存复用，并允许在显式依赖成立时手动管理生命周期。[ProcessGroupNCCL v2.10.0源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp) 的`WorkNCCL::synchronizeStream`把完成事件接入当前stream后unstash；async collective保存input/output引用。本候选保留全部Work.wait，只移除不再存在的private compute streams及用户侧record_stream。

镜像`model_runner.py:526`创建非默认forward_stream，scheduler通过forward_stream上下文执行模型。新helper不假设current stream是默认stream；诊断输出实际stream ID与线程信息。微测也显式使用非默认model_stream。

此前建议的“四side streams + tail fence”理论上可通过双向event证明生命周期，但需要逐一处理owner stream与缓存arena；current-stream方案更小，当前优先验证它，不实现或启用八子批服务。

## 构建、保留失败版本与诊断

```bash
docker build -t sglang-bw100:ttft-mlp-current-issue3 patch/ttft_mlp_pipeline/current_stream
```

父镜像为失败的`sglang-bw100:ttft-mlp-pipeline-issue3`。installer校验原helper SHA256 `d4f525cf184bf9472f3f54a750d2e5bc5616a8423da3f2bfe4ed35a5d1b1658a`，保存`qwen27b_mlp_pipeline_failed.py`后才安装新helper。原仓库`../mlp_pipeline.py`未修改，已有失败镜像未覆盖。

可选探针源文件为`../diagnostics/pipeline_probe.py`，本构建目录包含其相同副本。默认采样前2个eligible prefill batch、每层一次JSONL；环境变量`SGLANG_MLP_DIAG_MAX_BATCHES`最大4。记录CPU提交分段、线程/stream、allocator allocated/active/reserved/inactive_split及alloc_retries、num_sync_all_streams等计数。没有GPU同步、Tensor.item/cpu或逐层GPU event计时；这是host/allocator telemetry，不是GPU阶段耗时。`SGLANG_MLP_DIAG_ENABLED=0`关闭探针。

## 连续多层与受限显存验证

```bash
# 只能在模型服务停止、GPU4/5空闲时，由主 agent 串行运行。
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /opt/ttft-mlp-current/bench_stress.py --model /model \
  --tokens 32768 --layers 64 --headroom-gib 8 \
  --modes baseline current failed --repeats 2 \
  --output /bench/result/ttft/issue3-attention/mlp-current-stress-8g.json
```

压力脚本使用真实Qwen decoder边界，重复64层；整条链只在末尾同步，层间不synchronize。非默认model_stream运行，全链输出与residual对同样baseline逐值验证。持有一个活跃uint8张量使空闲显存降至指定headroom，允许4–32GiB；先8GiB，再按主 agent 判断收紧到4GiB。默认modes为baseline/current，失败模式需要显式加入。逐层记录CPU提交和allocator变化，并报告全链GPU时间。

为了只检验pipeline/allocator，脚本重复使用同一真实MLP权重，每层用`h*0.5`生成新的attention partial占位张量，TP2输入AR还原该值；没有运行真实attention，也没有KV/SSM池或真实服务的内存碎片结构。因此它是连续依赖与显存压力复现工具，不是完整27B模型模拟或TTFT验收。64层压力测试不能与运行中的模型共用GPU。真实模型仍需在明确记录mem_fraction_static等参数后验证固定case。

当前stream修正已通过下述GPU压力验证和实际模型测试；旧四stream完整模型失败结论仍保留。

## Root连续压力验证

在GPU4/5、64个连续真实MLP边界、非默认model stream、8GiB可用余量条件下，所有模式两rank最终输出和残差均逐值一致。每层不做GPU同步。

| 模式 | 两轮完整链GPU耗时(ms) | 主机提交耗时(ms) | 每rank分配重试/全stream同步次数 |
| --- | --- | --- | --- |
| 原生边界 | 2447.9 / 2632.5 | 62.8 / 272.7 | 0 / 1 |
| 当前stream修正 | 2149.7 / 2185.5 | 222.2 / 219.5 | 0 / 0 |
| 失败的四compute-stream版本 | 4430.5 / 6379.6 | 4295.4 / 6275.9 | 10 / 11 |

证据`result/ttft/issue3-attention/mlp-current-stress.json`。验证支持跨stream分配压力的诊断，修正版没有发生分配重试；这仍是连续MLP链，完整模型结果另验。

## 实际模型验证

与9个已验证GEMM形状组合后，固定套件270秒完成。4K/1K C64 TTFT从17.762641到15.091597s（提升15.04%）；64K/1K C1从15.901140到10.335549s（提升35.00%，该case不命中MLP拆批）。全部130个请求的输出hash与metadata+KV父候选逐个相同。模型前两个32K预热批次的分配重试和全stream同步均为0，证明旧失败机制没有重现；第二批单层仍有约930ms主机提交等待，需进一步分析，不能继续称为分配重试。结果见 `result/quick/ttft-current-gemm/20261015-181937-dugWWz/` 和 `result/ttft/mlp-allocator-diagnostic/`。完整原生任务精度与每个case至少20%的目标仍未通过。
