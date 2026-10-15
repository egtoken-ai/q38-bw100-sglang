# Current-compute-stream parts4/8：独立候选与真实层微测

本目录只扩展已经修正的current-stream结构，不使用旧`parts_pipeline.py`的private compute streams，也不复用旧结构的性能结论。默认配置保持4子批；本次文档更新未修改服务。

`current_parts.py:try_mlp_pipeline_parts` 默认parts=4，仅允许parts4/8及总tokens32768/65536。所有norm、GEMM、activation allocation继续位于实际caller stream；PG NCCL独立通信stream执行AR，全部inAR先提交、计算串行、全部outAR完成事件接回caller后cat。没有创建private compute streams、CUDA events或调用用户侧record_stream。Guard与修正后的服务helper保持一致。

八子批仍无依赖环：inAR8个事件均在compute之前记录，compute(i)只等其更早的inAR(i)，outAR(i)等compute(i)。outAR不会成为后续inAR的前置条件。输出总量不变、单个GEMM临时张量更小，但collective和norm/GEMM调用次数增加，必须实测而非推断提升。

## 64K实际层设备结果与候选选择

主agent已完成M65536、parts4/8设备微测，原始结果见 [mlp-current-parts-64k.json](../../../result/ttft/issue3-attention/mlp-current-parts-64k.json)。同一current-stream实现、同总tokens单batch参考；每项为5次采样的最慢rank耗时中位数，单位ms。下表提升按`(参考耗时-候选耗时)/参考耗时`计算。

| 实际层边界 | 单batch | 4子批 | 8子批 | 8子批相对单batch提升 | 8子批相对4子批提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN层的MLP边界 | 77.4842 | 68.0120 | 66.3167 | 14.41% | 2.49% |
| Full Attention层的MLP边界 | 77.6203 | 68.2219 | 66.2451 | 14.65% | 2.90% |

两类层、两rank、4/8子批的output、residual、inplace norm缓冲区均逐值一致（max_abs=0，mismatches=0）；每rank residual相对输入确实改变335,022,614个元素，校验非空操作。已安装4子批控制组也完全一致。

据此选定**内部chunk65536＋8子批（每子批8192 tokens）进入完整模型固定case验证**。M32768的8子批尚未测量，不能把64K结果推广到32K/8，也不能拿旧private-stream结构的结果替代。本结果只覆盖真实层的prepare_mlp AR＋Norm＋MLP＋outAR边界，使用随机权重；不含attention/GDN计算、完整checkpoint、调度、KV压力或服务TTFT。局部14%左右提升不是模型TTFT提升，更不代表两条固定case已达到20%目标。模型验证仍保持4096/1024/C64与65536/1024/C1、warmup1＋measure1，记录chunk、parts、mem_fraction及调度flag后单独判断。

## 独立实际层验证

以下是脚本参数示例；执行时使用[AUDIT_AND_64K.md](AUDIT_AND_64K.md)中的显式9形状白名单、外层timeout和失败即停命令。仅由主agent确认GPU4/5空闲后，在已经安装current-stream修正helper的sleep容器中串行运行：

```bash
HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_mlp_pipeline/current_stream_parts/bench_actual_parts.py \
  --tokens 32768 --parts 4 8 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-current-parts-32k.json

HIP_VISIBLE_DEVICES=4,5 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_mlp_pipeline/current_stream_parts/bench_actual_parts.py \
  --tokens 65536 --parts 4 8 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-current-parts-64k.json
```

默认tokens32768/parts4。脚本构造真实两类Qwen decoder边界和量化权重参数，随机权重；同总tokens单batch是基准。每个parts候选的output、residual、norm缓冲区都要求逐值一致；与已安装helper当前选定parts相同的候选额外作控制对照，并记录installed_control_parts。最慢rank计时，记录峰值allocated及相对增量。启动时检查已安装helper，若仍有private Stream/record_stream调用就拒绝，防止误测旧失败镜像。

这仍是边界微测，不是完整checkpoint准确性、模型TTFT或高显存连续链验证。M65536单batch临时空间更大，只能在独占空闲GPU测试。64K微测已支持8子批进入完整模型固定case验证；本目录不自动安装到服务。

本次CPU检查通过：rejection_reason AST与current-stream helper一致，循环/chunk均使用parts，helper没有Stream/Event/record_stream/synchronize调用；语法检查通过。主agent随后完成的64K设备结果见上方记录；本次子agent更新未运行GPU。


## 独立可构建selector候选：默认仍4，64K/8进入模型验证

已新增生产入口文件`mlp_pipeline.py`、`install.py`、`Dockerfile`和`current-stream-parts.patch`，与前面的参数化诊断函数分开。默认父镜像为`sglang-bw100:ttft-current-gemm-issue3`，包含GEMM补丁但白名单默认为空；测试时须显式传入已批准的9形状。主agent另以`sglang-bw100:ttft-prefill-release-issue3`为父构建`sglang-bw100:ttft-current-parts-issue3`，继承该父镜像的白名单和调度flag，两种来源应分别记录。本installer只修改pipeline helper，不触碰GEMM代码或配置。

```bash
docker build -t sglang-bw100:ttft-current-gemm-parts-issue3 \
  patch/ttft_mlp_pipeline/current_stream_parts
```

安装器要求父helper SHA256为`b066831d162186cd635197703be8293f6c527e5bb0eafc5d3f0f95c1e100a68f`，已与live current-gemm镜像和本地原current helper逐字核对；保存`qwen27b_mlp_pipeline_before_parts.py`后才替换。项目原current_stream文件、当前容器与服务没有修改。

唯一新增生产配置：`SGLANG_QWEN27B_MLP_PIPELINE_PARTS`，默认`4`，只接受字面值`4`或`8`。其他值在模块导入时抛出明确ValueError；不会偷偷改回4。选择8时，如果token数不能被8整除，helper返回None，由原decoder路径继续。原有全部guard和每个子批的input/output Work.wait保留；仍没有private compute streams或record_stream。

根据已完成的65536×4/8微测，已选定64K/8进入模型验证，由主agent在新容器中显式设置`SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8`。默认镜像不会自动启用8。回到4使用同变量值4；关闭整个pipeline仍使用原`SGLANG_QWEN27B_MLP_PIPELINE=0`。完整模型比较应记录该新变量、chunk大小及mem_fraction_static；不更改固定benchmark case。

CPU检查已覆盖：未设置时为4、4/8接受、0/2/16/04/空值/带空格/auto拒绝；8子批+32772 tokens在任何import/collective前返回None；原rejection_reason AST不变，input和output Work.wait均保留，helper无private Stream/Event/record_stream/synchronize。上述是CPU检查范围；设备结果与模型验证状态单独记录，不由CPU检查推断。

CPU审计结论、已修复的installed-control标注问题及待执行64K chunk固定case命令见 [AUDIT_AND_64K.md](AUDIT_AND_64K.md)。
