# CPU审计与下一轮64K chunk命令

当前已验证模型候选为current-stream+9-shape GEMM，两固定case TTFT提升15.04%/35.00%，130请求输出hash与parent一致。此处只审计parts4/8及准备待执行命令；没有操作GPU、容器或服务，没有访问/修改prefill_release目录。

审计结论：未发现当前受限TP2路径的阻断问题。生产和诊断helper的`rejection_reason` AST一致；从probe导入开始的整个compute/lifetime函数体在忽略日志标签后AST一致。因此诊断不是另一个数学实现或private-stream实现。每个子批先等待输入AR再norm/GEMM，输出AR全部等待后才cat，输入/输出owner和consumer都回到caller stream；没有新增record_stream或GPU同步。source中的32K阈值、C1/decode fallback、无scatter/ARfusion、动态对称INT8和原GemmaNorm约束均保持。双rankcollective同序，row顺序保持。

发现并修复一个诊断标注问题：原诊断把已安装helper一律视为4子批；若容器设置PARTS=8，控制组会实际运行8却被当作4。现在读取已安装模块的`_PIPELINE_PARTS`（旧current模块没有该字段时为4），在JSON记录`installed_control_parts`，只对匹配parts的候选做安装版本控制检查。不会改服务环境。

安装校验自洽：`mlp_pipeline.original.py`的SHA256与installer要求的`b066831d162186cd635197703be8293f6c527e5bb0eafc5d3f0f95c1e100a68f`相同。Dockerfile继承current-gemm，只替换helper并保存before_parts备份；默认PARTS=4。非法PARTS值明确报错，8且token数非8倍数在collective前返回原路径。

CPU检查包括语法、guard AST、完整计算体AST、SHA一致性和固定benchmark脚本dry-run。dry-run确认仍是4096/1024/C64与65536/1024/C1，各warmup1+measure1、seed20260912，没有变更case。

## 64K实际层设备结果与候选选择

主agent已完成M65536、parts4/8设备微测，原始结果见 [mlp-current-parts-64k.json](../../../result/ttft/issue3-attention/mlp-current-parts-64k.json)。同一current-stream实现、同总tokens单batch参考；每项为5次采样的最慢rank耗时中位数，单位ms。下表提升按`(参考耗时-候选耗时)/参考耗时`计算。

| 实际层边界 | 单batch | 4子批 | 8子批 | 8子批相对单batch提升 | 8子批相对4子批提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN层的MLP边界 | 77.4842 | 68.0120 | 66.3167 | 14.41% | 2.49% |
| Full Attention层的MLP边界 | 77.6203 | 68.2219 | 66.2451 | 14.65% | 2.90% |

两类层、两rank、4/8子批的output、residual、inplace norm缓冲区均逐值一致（max_abs=0，mismatches=0）；每rank residual相对输入确实改变335,022,614个元素，校验非空操作。已安装4子批控制组也完全一致。

据此选定**内部chunk65536＋8子批（每子批8192 tokens）进入完整模型固定case验证**。M32768的8子批尚未测量，不能把64K结果推广到32K/8，也不能拿旧private-stream结构的结果替代。本结果只覆盖真实层的prepare_mlp AR＋Norm＋MLP＋outAR边界，使用随机权重；不含attention/GDN计算、完整checkpoint、调度、KV压力或服务TTFT。局部14%左右提升不是模型TTFT提升，更不代表两条固定case已达到20%目标。模型验证仍保持4096/1024/C64与65536/1024/C1、warmup1＋measure1，记录chunk、parts、mem_fraction及调度flag后单独判断。

## 执行顺序与命令（由主agent执行）

主 agent 完成当前调度策略试验、释放GPU4/5后，再按微测结果选择是否执行。64K×parts4/8微测已完成，8子批胜出，选定64K/8进入模型验证；32K/8尚未测。下面保留有界诊断复现命令，不要求为此自动补跑32K或重复已完成64K。不要拿旧private-stream微测结果代替这一轮。

同步本目录并在服务器构建：

```bash
rsync -a patch/ttft_mlp_pipeline/current_stream_parts/ \
  root@bw100:/data/tools/q38-bw100-sglang/patch/ttft_mlp_pipeline/current_stream_parts/

ssh root@bw100 'cd /data/tools/q38-bw100-sglang && docker build \
  -t sglang-bw100:ttft-current-gemm-parts-issue3 \
  patch/ttft_mlp_pipeline/current_stream_parts'
```

Dockerfile默认FROM `sglang-bw100:ttft-current-gemm-issue3`，该父镜像包含GEMM补丁，但`Q38_HIPBLASLT_SHAPES`为空，不能称为默认启用9形状。以下服务和诊断命令均显式传入已批准的9形状及`TORCH_NCCL_BLOCKING_WAIT=0`。

主agent另在构建`sglang-bw100:ttft-current-parts-issue3`，使用`--build-arg BASE_IMAGE=sglang-bw100:ttft-prefill-release-issue3`，会继承独立调度策略flag和白名单。若选这个镜像，应把下面的`NEXT_IMAGE`改成该tag、显式设置`PREFILL_RELEASE=1`，并记录父镜像ID；不要把两种镜像来源和调度策略混为同一次parts A/B。

下面在bw100服务器的`/data/tools/q38-bw100-sglang`执行。`PREFILL_RELEASE`默认0，对应已知current-gemm parent行为；只有当前独立调度策略试验被选定后才显式改1，并记录这个额外因素。这里没有修改该策略或它的文件。

```bash
NEXT_CONTAINER=qwen38_bw100_ttft_parts64k
NEXT_LABEL=current-gemm-chunk64k-parts8-mem80
NEXT_IMAGE=sglang-bw100:ttft-current-gemm-parts-issue3
APPROVED_SHAPES=4096x8192x5120,8192x17408x5120,8192x8192x5120,8192x5120x3072,16384x17408x5120,16384x8192x5120,16384x5120x3072,32768x8192x5120,32768x5120x3072
PREFILL_RELEASE=${PREFILL_RELEASE:-0}

# start_docker.sh会覆盖同名容器，先用检查保护已有实例。
if docker container inspect "$NEXT_CONTAINER" >/dev/null 2>&1; then
  echo "容器名已存在，选择新名称后再运行: $NEXT_CONTAINER" >&2
  exit 1
fi
CONTAINER_NAME="$NEXT_CONTAINER" \
IMAGE_NAME="$NEXT_IMAGE" \
MODEL_PATH=/data/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
BENCH_DIR=/data/tools/q38-bw100-sglang \
  bash scripts/start_docker.sh

# 原启动脚本/镜像wrapper沿用FA、packed路径；9-shape GEMM在此显式启用。
# 只更改已列出的chunk、parts、mem_fraction；GPU固定4,5。
docker exec -w /bench \
  -e HIP_VISIBLE_DEVICES=4,5 \
  -e MODEL_PATH=/model \
  -e TORCH_NCCL_BLOCKING_WAIT=0 \
  -e Q38_HIPBLASLT_SHAPES="$APPROVED_SHAPES" \
  -e CHUNKED_PREFILL_SIZE=65536 \
  -e MAX_PREFILL_TOKENS=65536 \
  -e MEM_FRACTION_STATIC=0.80 \
  -e SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8 \
  -e SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP="$PREFILL_RELEASE" \
  "$NEXT_CONTAINER" bash /opt/ttft-prefill/start_server.sh --daemon

# 在Docker host保存真实argv、白名单env和immutable image ID。
python3 scripts/capture_ttft_deployment.py \
  --container "$NEXT_CONTAINER" \
  --output "result/ttft/${NEXT_LABEL}-deployment.json"

docker exec -w /bench "$NEXT_CONTAINER" \
  bash scripts/run_bench_quick.sh "$NEXT_LABEL"
```

启动后必须确认服务guard命中日志包含`QWEN27B_MLP_CURRENT_STREAM_HIT ... tokens=65536 parts=8`，真实argv为chunk/max-prefill65536、mem_fraction0.80，进程环境记录PARTS=8及9-shape白名单；C1和decode应走原路径。启动脚本不直接接受任意额外CLI参数，eager状态依赖当前模型ServerArgs处理并由已有helper guard验证；若没有HIT，不能把该轮标为8子批优化结果。

此轮仍跑完整两case、warmup1+measure1，输出长度不缩短。保存当前配置的结果与snapshot，核对请求总数130、token长度、输出hash、TTFT/TPS/TPOT、实际KV容量和峰值显存。mem_fraction_static=0.80是有意改变的服务参数，不能隐去。如果同时启用PREFILL_RELEASE=1，也必须在对比表中列明；有收益后才进行重复匹配A/B。

## 有界实际层诊断：按顺序执行，失败即停

先由主agent在宿主机确认GPU4/5没有服务、benchmark或其他任务占用，再选择已有的sleep-only current-stream+GEMM容器。此处不自动停止容器，不凭容器内部进程列表判断GPU空闲。两项顺序运行；各项外层GNU timeout为180秒，TERM后15秒仍未退出则KILL。任一项失败或超时立即结束流程，由主agent确认torchrun及两rank均已退出、GPU已释放之后才能安排下一项。超时本身不等于已经确认所有后代进程退出。

下面在服务器项目目录执行，先把`DIAG_CONTAINER`设置为主agent已核实的sleep-only容器名；目录需已同步。每项同时比较parts4和8，控制组使用同一白名单，不能用旧private-stream控制组。2×195秒是诊断的有界等待上限，不是固定验收suite的实际耗时。

```bash
set -eu
: "${DIAG_CONTAINER:?先设置已核实且GPU独占的sleep-only容器名}"
APPROVED_SHAPES=4096x8192x5120,8192x17408x5120,8192x8192x5120,8192x5120x3072,16384x17408x5120,16384x8192x5120,16384x5120x3072,32768x8192x5120,32768x5120x3072
for tokens in 32768 65536; do
  if docker exec -w /bench \
    -e HIP_VISIBLE_DEVICES=4,5 \
    -e TORCH_NCCL_BLOCKING_WAIT=0 \
    -e Q38_HIPBLASLT_SHAPES="$APPROVED_SHAPES" \
    -e SGLANG_QWEN27B_MLP_PIPELINE_PARTS=4 \
    "$DIAG_CONTAINER" \
    timeout --signal=TERM --kill-after=15s 180s \
    torchrun --standalone --nproc-per-node=2 \
    /bench/patch/ttft_mlp_pipeline/current_stream_parts/bench_actual_parts.py \
    --model /model --tokens "$tokens" --parts 4 8 --iterations 3 --repeats 5 \
    --output "/bench/result/ttft/issue3-attention/current-parts-${tokens}-gemm.json"; then
    echo "完成 tokens=$tokens；下一项仍按顺序运行"
  else
    rc=$?
    echo "诊断失败或超时 rc=$rc；停止后续任务，先核查残留worker与GPU占用" >&2
    exit "$rc"
  fi
done
```

微测JSON必须通过逐值output/residual/norm对照，再比较同tokens的单batch、parts4、parts8。64K与32K是内部chunk诊断，不改变两条固定验收case。微测优势只用于选型，最终结论仍以完整模型固定suite为准。
