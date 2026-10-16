# C1：8 子批 MLP pipeline 去掉 `torch.cat`

来源：`result/ttft/postops-prefill-gpu-profile/analysis.md` 候选 1。每层 MLP 8 子批结束后的 `torch.cat` 每次 3.03ms，64 层合计约 194ms（每个 16×4K 批次）。

## 改了什么

父 helper：`sglang/srt/layers/qwen27b_mlp_pipeline.py`，SHA256 `a946a802abb6a6faeb84108e4ccb10db957bc553bfc573333d026cc681febbad`。它与 `patch/ttft_mlp_pipeline/current_stream_parts/mlp_pipeline.py` 逐字一致，并已在镜像 `ttft-postops-issue3` / `ttft-qk-gate-64k-v2-issue3` 内核对。
新 helper：`mlp_pipeline.py`，SHA256 `1796bea1dbde29d135a6f013f63a39331ca8f38583ae09c4b983144687e217a7`。

- 新增 `SGLANG_QWEN27B_MLP_CATFREE`，在 import 时读取，只接受字面值 `0` 或 `1`，其他值抛出 ValueError。helper 默认取 `0`，本 Dockerfile 层设为 `1`。设为 `0` 时走原路径：每个 part 一个 `all_reduce`，最后做 `torch.cat`。
- `try_mlp_pipeline` 只转发到 `run_mlp_pipeline(..., catfree=_CATFREE)`。`rejection_reason`、`_parts_from_env`、input AR、norm、MLP（含 quant 和 GEMM，形状与调用顺序不变）、residual 处理和 probe 都没有改动。
- catfree=1 时：
  - 进入循环前，在 caller stream 上 `torch.empty_like(hidden_states)` 分配完整输出，按行 chunk 成 8 个连续切片。
  - 每个 part 的 down_proj 局部和 `y` 先做 `reduce_scatter_tensor`，把本 rank 那一半直接规约进输出切片对应的位置；再在同一切片上做原地 `all_gather_into_tensor`，满足 NCCL in-place 条件 `sendbuff == recvbuff + rank*count`。
  - 循环结束后对每个 part 的两个 Work 依次 `wait()`，等待位置与原路径相同，然后直接使用预分配的输出，不再拼接。
  - 不新建 stream 或 event，不调用 record_stream 或 synchronize。

为什么不用 GEMM 写入切片：lightop 的 `lightop_channel_int8_mm` 和 `hipblaslt_w8a8_gemm` 都只返回新张量，没有 `out=` 参数。in-place 的 `all_reduce` 又必须作用在 `y` 自身上。另一种做法是每次 wait 之后再复制进切片，但搬运的字节数与 cat 相同，而且仍在计算 stream 上，省不下时间。TP2 下 RS+AG 等价于一次 ring all-reduce，每个元素仍只做一次“两个 rank 局部和相加”。IEEE 加法满足交换律，因此输出应与原 AR 逐位一致。**这一点必须由下面的 GPU 脚本实测确认。**

可选的第二步（隐藏尾部 out-AR(7)/首部 in-AR(0)）**没有实现**。要隐藏尾部，下一层的 input norm 必须按 part 执行，这会改动 decoder forward；如果改用不等长子批来缩短尾部，GEMM 的 M 会变，无法保证逐位一致。两者都不满足“简单且逐位一致”的要求。

## 预期收益与风险

- 每层省掉 3.03ms 的 cat，每个 16×4K 批次约 −190ms，占 6843ms step 的 2.8%。P50 包含两个这样的批次，约 −0.38s。64K C1 单请求批次被 guard 拒绝，不受影响。
- 代价：每个 part 多一次 collective launch，每层多 8 次，大部分与计算重叠。尾部多暴露一次 AG 的启动延迟，估计每层约 0.02–0.05ms，64 层约 1–3ms。
- 峰值显存：完整输出改为循环前分配，因此最后一个 part 的中间张量与它同时存活。M65536 时峰值估计比原路径高约 0.3–0.6GB，以脚本报告的 `catfree_peak_delta_vs_old8_bytes` 为准。注意力段的全 M 中间张量更大，预计不会改变整步峰值，需在模型运行中确认。
- 如果 RS/AG 与 AR 的 RCCL 内核在 bf16 求和时舍入路径不同，就可能出现非零 mismatch。这种情况下判 C1 失败，用 `SGLANG_QWEN27B_MLP_CATFREE=0` 回退。

## 文件

| 文件 | 作用 |
| --- | --- |
| `mlp_pipeline.py` | 新 helper |
| `install.py` | 校验父 SHA 与候选 SHA。备份为 `qwen27b_mlp_pipeline.py.ttft_catfree_original`（非 .py 后缀）。备份已存在或父文件漂移时拒绝安装；安装后复核两个哈希 |
| `Dockerfile` | `ARG BASE_IMAGE=sglang-bw100:ttft-postops-issue3`，运行 installer，`ENV SGLANG_QWEN27B_MLP_CATFREE=1`。不写入 NCCL 通道 |
| `bench_catfree.py` | GPU 验证（TP2）|
| `cpu_check.py` | CPU 检查（无 GPU）|

## CPU 检查（已完成，未使用 GPU 或容器）

用 `docker exec qwen38_bw100_ttft_qk64_v2 cat …/qwen27b_mlp_pipeline.py` 取回真实 helper 文本，SHA 为 a946a802…，然后运行：

```bash
python3.11 patch/ttft_mlp_catfree/cpu_check.py --old-helper <取回的 helper 文件>
```

结果全部通过：

1. AST：`rejection_reason` 和 `_parts_from_env` 与原 helper 完全一致；没有 Stream/Event/record_stream/synchronize/wait_stream 调用；三个 .py 都通过 py_compile。
2. 模拟安装：在临时 layers 目录中，对真实 helper 文本运行 `install.py`。哈希正确，备份没有 .py 后缀，也没有多出 .py 文件。重复安装和父文件漂移都被拒绝。
3. 功能模拟：sglang 打桩，用单进程模拟 TP2 的 dist。分别模拟 rank 0 和 rank 1，旧 helper、新 helper（catfree=0）、新 helper（catfree=1）三者的 output、residual、norm 缓冲逐值相等。AG 确为 in-place，所有 Work 都被 wait。

## 主 agent 执行的命令

在 bw100 的 `/data/tools/q38-bw100-sglang` 下执行，目录需先同步。

**1. 构建镜像**

```bash
docker build --build-arg BASE_IMAGE=sglang-bw100:ttft-postops-issue3 \
  -t sglang-bw100:ttft-catfree-issue3 patch/ttft_mlp_catfree
```

**2. GPU 验证**

先确认 GPU4/5 空闲。可以在新镜像的 sleep-only 容器里运行，也可以直接在现有的 `qwen38_bw100_ttft_qk64_v2` 里运行。

脚本会从显式文件加载两个 helper，并分别校验 SHA：

- 旧 helper：优先取 `.ttft_catfree_original` 备份；没有备份时取仍为旧版的已安装 helper。
- 新 helper：取 `/bench/patch/ttft_mlp_catfree/mlp_pipeline.py`。

因此两种容器都能运行。脚本内部强制 parts=8、M=65536。通道设置与服务保持一致：

```bash
docker exec -w /bench -e HIP_VISIBLE_DEVICES=4,5 -e TORCH_NCCL_BLOCKING_WAIT=0 \
  -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8 \
  "$DIAG_CONTAINER" timeout --signal=TERM --kill-after=15s 240s \
  torchrun --standalone --nproc-per-node=2 /bench/patch/ttft_mlp_catfree/bench_catfree.py \
  --model /model --tokens 65536 --iterations 3 --repeats 5 \
  --output /bench/result/ttft/issue3-attention/mlp-catfree-64k-ch8.json
```

脚本内容：

- 对 GDN 层和 Full-Attention 层，使用真实的 compressed-tensors INT8 分派。
- guard 检查：M4096 bs1、M65536 bs1、DECODE 三种情况下，旧、新 helper 的拒绝原因相同，三种调用都返回 None，输入未被改动。
- 正确性：`old8 / new8_cat / new8_catfree` 分别对 `baseline` 和 `old8` 比较 output、residual、in-place norm 缓冲，要求全部 exact，并做 residual 非空检查。
- 计时：baseline、old8、new8_catfree 三种模式，预热 3 次，再交替顺序跑 5 轮，每轮 3 次，取最慢 rank 的中位数和峰值增量。
- 输出字段 `catfree_saving_vs_old8_ms/pct` 和 `catfree_peak_delta_vs_old8_bytes`，任何失败都会立即停止。

**3. 服务**

用新镜像，环境与当前最佳配置相同：`NCCL_MIN_NCHANNELS=8`、`NCCL_MAX_NCHANNELS=8`、`SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8`、chunked prefill 65536。`SGLANG_QWEN27B_MLP_CATFREE=1` 已写入镜像；回退时用 `-e SGLANG_QWEN27B_MLP_CATFREE=0`。

日志里必须出现 `QWEN27B_MLP_CURRENT_STREAM_HIT ... tokens=65536 parts=8 catfree=1`。之后：

- 重新 profile，确认 cat kernel 已消失，同时检查 RCCL 尾部；
- 跑固定套件；
- 核对 130 个输出 hash。
