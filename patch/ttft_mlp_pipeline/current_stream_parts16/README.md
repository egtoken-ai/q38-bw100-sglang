# 仅诊断：64K current-stream MLP的8/16子批比较

这是独立微测目录，没有生产helper、安装器或Dockerfile。复用已审查的current-stream真实decoder层边界验证，严格限定总tokens=65536、parts=[8,16]、TP2、BF16 activation＋原INT8 channel量化。没有改动`current_stream_parts/`、生产helper、服务或admission草稿；子agent未运行GPU；主agent随后完成的设备结果如下。

目的只是在相同64K总tokens下比较8192和4096 tokens/子批，看更细分块能否改善通信与计算重叠。16子批同时增加collective、norm和GEMM launch数量，且所有输入AR仍先提交，可能因通信启动开销和小GEMM而变慢，不能预先假设收益。当前9形状GEMM白名单没有4096×17408×5120，因此16子批的gate-up会按既有分派回原路径；不能为此悄悄增加未经验证形状。

## 实测结论：拒绝16子批进入生产

主agent已在安装GEMM分派的`qk64_v2`环境执行，原始结果为[mlp-current-parts-8vs16-64k.json](../../../result/ttft/issue3-attention/mlp-current-parts-8vs16-64k.json)。两层、两rank、8/16子批完整output、residual及inplace norm均逐值exact；8子批也通过已安装parts8 helper的控制对照，残差相对初值确实改变。

下表耗时为5轮max-rank的中位数，单位ms；提升按`(8子批耗时-16子批耗时)/8子批耗时`计算，负值表示16子批更慢。

| 层边界 | 同tokens单batch | 8子批 | 16子批 | 16相对8提升 |
| --- | ---: | ---: | ---: | ---: |
| GDN层MLP边界 | 77.67305 | 64.40658 | 65.44402 | -1.61% |
| Full Attention层MLP边界 | 77.49262 | 64.41602 | 65.63825 | -1.90% |

原始5轮max-rank时间如下，每轮均是3次迭代平均，测试模式顺序轮换：

| 层与模式 | 第1轮 | 第2轮 | 第3轮 | 第4轮 | 第5轮 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN层MLP边界 pipeline_8 | 64.42882 | 64.54380 | 64.25314 | 64.40658 | 64.38758 |
| GDN层MLP边界 pipeline_16 | 65.44402 | 65.43874 | 65.34269 | 65.51219 | 65.52514 |
| Full Attention层MLP边界 pipeline_8 | 64.27793 | 64.41602 | 64.44663 | 64.10487 | 64.48188 |
| Full Attention层MLP边界 pipeline_16 | 65.52972 | 65.60765 | 65.63825 | 65.87069 | 65.72871 |

两层均5/5轮16子批慢于同轮8子批；每层16子批的最快一轮也慢于8子批最慢一轮，故该微测中结果一致，不是中位数偶然跨过一次噪声。保留8子批，16子批不生成生产安装器、不进入完整模型测试。

跨轮环境边界：此前`mlp-current-parts-64k.json`的4/8微测使用`ttft_current`容器，尚未安装GEMM分派补丁；即使传入9形状ENV也不能证明实际命中。此次`qk64_v2`已安装该分派，因此旧8子批约66.3ms与此次约64.4ms不是纯CPU或NUMA对照，不能把差值归为绑定收益。本JSON记录白名单是配置证据，仍不等于独立的逐kernel命中计数。

本轮是随机权重的真实层边界微测，不含attention/GDN、请求调度、完整checkpoint或高显存连续模型压力；不能据此给出模型TTFT，也不能将16子批变慢唯一归因于通信或GEMM。当前可确认的结论仅为：在这次固定环境、M65536、两类边界中，16子批没有优于8子批。

## 文件与校验边界

- `current_parts16.py`：仅诊断helper，默认parts8；只允许M65536、parts8或16。原完整rejection_reason保持不变；所有compute和activation分配仍在实际caller stream，PG NCCL自身通信stream执行async AR，input/output Work.wait与cat顺序保持。没有创建private compute streams或手工record_stream。
- `bench_actual_parts16.py`：构造真实`Qwen3_5LinearDecoderLayer`与`Qwen3_5AttentionDecoderLayer`、原GemmaNorm、compressed-tensors INT8权重分派和communicator；随机权重，不加载完整checkpoint、不执行attention/GDN。每种parts与同总tokens单batch参考比较完整output、residual、inplace norm，必须两rank逐值exact，并验证residual相对初值确实变化。8子批还与已安装helper的output/residual/norm作完整exact控制比较；已安装模块实际parts不是8就拒绝。

三模式均包含原prepare_mlp AR＋Norm＋MLP＋outAR边界与相同输入reset。每轮轮换测试顺序，5轮、每轮3次，保存每rank耗时、max-rank耗时及其中位数；保留peak allocated和相对分配增量。输出JSON额外记录GEMM白名单、NCCL blocking wait、installed_control_parts与16相对8提升。检查仍在原流式布局与原dtype上，没有放宽容差。

本次CPU检查通过：语法检查、完整rejection_reason AST相同、计算/生命周期函数体AST除日志标签外与已审查8子批helper相同；没有private Stream/Event/record_stream/synchronize调用。CPU解析检查确认拒绝M32768、parts4、只测parts8、反序16/8以及非5轮；默认恰为M65536和8/16两候选。主agent设备验证已经完成，结果见上方实测记录。

微测空闲GPU的结果不能替代高显存连续模型执行，也不是TTFT数据。只有全量exact且16对8有稳定局部收益，才值得另行审查生产selector；本目录不提供16子批服务安装路径。

## 主agent串行执行命令（保留复现，不安排重跑）

主agent先完成NUMA固定suite、确认GPU4/5无服务/其他任务，使用已安装current-stream parts selector和9形状GEMM的sleep-only容器。不要直接在推理服务仍占GPU时运行。以下在宿主项目目录执行；先同步本目录到服务器项目同路径。

```bash
set -eu
: "${DIAG_CONTAINER:?填入主agent已确认GPU独占的sleep-only容器名}"
APPROVED_SHAPES=4096x8192x5120,8192x17408x5120,8192x8192x5120,8192x5120x3072,16384x17408x5120,16384x8192x5120,16384x5120x3072,32768x8192x5120,32768x5120x3072
if docker exec -w /bench \
  -e HIP_VISIBLE_DEVICES=4,5 \
  -e TORCH_NCCL_BLOCKING_WAIT=0 \
  -e Q38_HIPBLASLT_SHAPES="$APPROVED_SHAPES" \
  -e SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8 \
  "$DIAG_CONTAINER" \
  timeout --signal=TERM --kill-after=15s 180s \
  torchrun --standalone --nproc-per-node=2 \
  /bench/patch/ttft_mlp_pipeline/current_stream_parts16/bench_actual_parts16.py \
  --model /model --tokens 65536 --parts 8 16 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-current-parts-8vs16-64k.json; then
  echo '诊断完成，先审查两层两rank全部exact，再比较5轮max-rank时间'
else
  rc=$?
  echo "诊断失败或超时 rc=$rc；停止后续GPU任务，先核查残留worker" >&2
  exit "$rc"
fi
```

timeout只限制等待，不证明所有torchrun子进程已退出；任何失败必须先由主agent核查残留进程及GPU占用。保持既有固定case不变，不通过修改初始请求分批或统计口径来补目标。
