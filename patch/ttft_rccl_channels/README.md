# RCCL 8 通道（8 子批 MLP 重叠场景）

只加环境变量的镜像层：在 `sglang-bw100:ttft-postops-issue3` 之上固定 `NCCL_MIN_NCHANNELS=8`、`NCCL_MAX_NCHANNELS=8`，不改 Python 或 kernel 源码。输出 tag：`sglang-bw100:ttft-rccl8-issue3`。

## 依据

profile（`result/ttft/postops-prefill-gpu-profile/analysis.md`）显示，16×4K 批中 RCCL 与 GEMM 并行时抢占 CU，gate_up/down/norm 合计约 453ms 争用开销。此前孤立 AR 微测里改通道数没有收益；重叠场景下结论不同。

`patch/ttft_mlp_pipeline/channel_overlap/bench_channel_overlap.py`（M65536、已安装 8 子批 helper、两类 decoder 层、两 rank 逐值 exact、5×3 计时）在 GPU 4/5 上的 installed8 中位数（ms）：

| 通道 | GDN 层 | FA 层 | 不重叠 baseline |
| --- | ---: | ---: | ---: |
| default | 64.30 | 64.43 | 77.7 |
| 4 | 63.85 | 63.93 | 102.8 |
| 6 | 61.26 | 61.43 | 88.9 |
| **8** | **60.08 / 60.18** | **60.29 / 60.60** | 83.0 |
| 12 | 61.45 | 61.75 | 78.1 |
| 16 | 63.98 | 64.17 | 77.5 |

所有配置 output/residual/norm 均逐值一致（TP2 两路求和与通道数无关）。原始 JSON：`result/ttft/issue3-attention/mlp-installed8-channels-*.json`（第二轮为 `*-sweep2.json`）。

## 模型固定快测（单轮，2026-10-16）

用 postops 镜像加 `-e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8`，其余服务参数与 postops 相同（64K chunk、8 子批、mem 0.80、NUMA 4）。结果：`result/quick/ttft-postops-ch8/20261016-111159-z7dqyj`。

| 场景 | 安擎基线 | postops | postops+ch8 |
| --- | ---: | ---: | ---: |
| 4K C64 TTFT P50 | 17.763s | 14.304s (+19.47%) | 13.838s (+22.10%) |
| 4K C64 平均 TTFT | — | 17.226s | 16.645s |
| 64K C1 TTFT P50 | 15.901s | 10.273s (+35.39%) | 10.621s (+33.21%) |
| 4K C64 TPOT P50 | 79.725ms | 75.288ms | 74.970ms |
| 64K C1 TPOT P50 | 34.619ms | 29.015ms | 29.109ms |

130 个预热及正式输出 SHA256 与 postops 父候选全部一致。64K C1 单请求不走 MLP 重叠，AR 完全暴露，8 通道使其慢约 3.4%，仍高于 20% 门槛。该单轮期间 GPU 6/7 有精度评测服务在运行（同 NUMA 侧），属于筛选口径；最终重复 A/B 须在 6/7 空闲时进行。

## 构建与启动

```bash
# BW100 上（项目同步到 /data/tools/q38-bw100-sglang）
docker build --build-arg BASE_IMAGE=sglang-bw100:ttft-postops-issue3 \
  -t sglang-bw100:ttft-rccl8-issue3 patch/ttft_rccl_channels
```

服务启动沿用 postops 方式（sleep 容器 + `/opt/ttft-numa/start_server.sh --daemon`），并显式设置 `CHUNKED_PREFILL_SIZE=65536 MAX_PREFILL_TOKENS=65536 MEM_FRACTION_STATIC=0.80 SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8 TORCH_NCCL_BLOCKING_WAIT=0`、`HIP_VISIBLE_DEVICES=4,5`。回滚：去掉该层或启动时 `-e NCCL_MIN_NCHANNELS= ...` 不可用（空字符串≠unset），应直接使用父镜像 `ttft-postops-issue3`。
