# 已安装8子批MLP：默认/4/8通信通道独立进程诊断

此目录只有诊断脚本和说明，不安装或修改任何生产helper。`bench_channel_overlap.py`只调用已安装的`try_mlp_pipeline`，要求其实际`_PIPELINE_PARTS=8`；M固定65536，真实GDN/Full Attention两种decoder层的prepare_mlp AR＋原GemmaNorm＋原INT8 MLP＋outAR边界，BF16 activation。原生单batch边界作为参考，不重测16子批。

待检验假设：较少通信通道可能缓解通信kernel与GEMM并行时的设备资源竞争，但也可能增加AR时延。此前裸RCCL微测几乎无收益，不能直接说明GEMM重叠场景；本诊断也不能预先证明有效。

仅允许以下三个独立进程配置；`--channels`必须与实际环境完全一致，default要求两变量真正unset，空字符串不等于unset：

| 参数 | NCCL_MIN_NCHANNELS | NCCL_MAX_NCHANNELS |
| --- | --- | --- |
| `--channels default` | unset | unset |
| `--channels 4` | `4` | `4` |
| `--channels 8` | `8` | `8` |

另有NCCL_MIN/MAX_CTAS或旧NCCL_MIN/MAX_NRINGS显式设置则拒绝，防止另一套通道控制混入。JSON记录实际MIN/MAX_NCHANNELS、上述替代变量、ALGO/PROTO/CONFIG_FILE、blocking wait、HIP设备，以及GEMM白名单和已安装helper SHA。两rank环境也作一致性检查。NCHANNELS环境是请求配置，不等于runtime communicator实际采用的通道数；若要证明后者，需另保留相同NCCL/RCCL日志级别下的初始化日志。配置文件里的设置也须在三轮保持一致，default含义为不施加这两个env覆盖，不宣称所有库配置都为空。

每种配置内部比较baseline与installed8，两类层/两rank完整output、residual、inplace norm要求逐值exact，且residual相对初值必须变化。随机权重、固定种子，使用真实compressed-tensors量化分派；不加载完整checkpoint、不执行attention/GDN。每模式预热3次，然后5轮×3次，轮换模式顺序，记录每rank与max-rank时间、中位数、peak allocated及增量。

跨通道选择应比较**installed8的绝对max-rank耗时分布**，同时检查baseline是否随通道配置变化。不能只看每个配置内部installed8对baseline的百分比，因为分母也可能变化。只有正确性通过、两类层有一致收益，才值得另行模型固定case验证；本微测不是模型TTFT或高显存连续层压力测试。

## CPU检查与未执行边界

已完成语法和CPU配置检查：只接受M65536、5×3和default/4/8；验证unset/4/8、单变量缺失、混合4/8、空字符串和CTA覆盖的接受/拒绝；baseline函数AST与已验证真实层微测相同，candidate只调用生产helper，没有引入新的pipeline实现。没有运行GPU、build镜像或操作服务。

## 主agent串行执行（尚未执行）

先完成当前suite并确认GPU4/5空闲，再用同一个已安装8part生产helper的sleep-only镜像；三次必须是新torchrun进程，不能在已经初始化的进程组内改变变量。固定同一GEMM白名单/NUMA等其余环境。下面在宿主项目目录运行，`MATCHED_GEMM_SHAPES`从选定父配置读取，可为原9或已批准10项，但三次必须完全相同，不在此扩展形状。

```bash
set -eu
: "${DIAG_CONTAINER:?填入主agent确认GPU独占的sleep-only容器名}"
: "${MATCHED_GEMM_SHAPES:?填入同一父配置的实际GEMM白名单}"
for channels in default 4 8; do
  channel_env=()
  if [ "$channels" != default ]; then
    channel_env=("NCCL_MIN_NCHANNELS=$channels" "NCCL_MAX_NCHANNELS=$channels")
  fi
  if docker exec -w /bench \
    -e HIP_VISIBLE_DEVICES=4,5 \
    -e TORCH_NCCL_BLOCKING_WAIT=0 \
    -e SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8 \
    -e Q38_HIPBLASLT_SHAPES="$MATCHED_GEMM_SHAPES" \
    "$DIAG_CONTAINER" \
    env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS "${channel_env[@]}" \
    timeout --signal=TERM --kill-after=10s 120s \
    torchrun --standalone --nproc-per-node=2 \
    /bench/patch/ttft_mlp_pipeline/channel_overlap/bench_channel_overlap.py \
    --model /model --tokens 65536 --channels "$channels" --iterations 3 --repeats 5 \
    --output "/bench/result/ttft/issue3-attention/mlp-installed8-channels-${channels}.json"; then
    echo "完成 channels=$channels，下一配置仍顺序运行"
  else
    rc=$?
    echo "失败/超时 rc=$rc；停止后续配置，先确认全部worker退出与GPU释放" >&2
    exit "$rc"
  fi
done
```

每项120秒加10秒终止宽限，三项最大等待约390秒；timeout返回本身不能证明所有后代进程已退出。任何失败均由主agent核查残留任务后再调度。若需要固定CPU/NUMA亲和，须在三次使用完全相同的进程启动绑定；仅镜像存在NUMA服务脚本不代表直接torchrun微测已经绑定。后续服务A/B仍保留原请求case、轮次、缓存和统计口径。
