# C2: causal_conv1d prefill launch tuning

## 路径确认

GDN prefill（`gdn_backend.py::forward_extend`，非 target-verify）调用
`layers/attention/mamba/causal_conv1d_triton.py::causal_conv1d_fn`（镜像未设
`SGLANG_USE_CAUSAL_CONV1D`，HCU 不走 CUDA/HCU 扩展）。profile 中 `_causal_conv1d_fwd_kernel`
grid `[16,512,20]` / `[1,8192,20]`、block 256，即厂商 launch BLOCK_M=8、BLOCK_N=256、
num_warps 默认4、num_stages=2。生产布局：`x = projected_qkvz[:, :5120].T`
（dim stride 1，token stride 8192，来自 GDN views patch），weight `[5120,4]`，bias 无，
SiLU，conv_states 是 `[slots,5120,3]` BF16 层切片，`has_initial_state` BOOL、
`query_start_loc`/`cache_indices` int32。conv_dim/rank = (2·16·128+48·128)/2 = 5120。
已有 patch 均未改动该文件；qk64-v2 镜像中 SHA256
`e87f568526b175f9e7e49e742ae2e6e0b36093d455882ed7a45b5593743d5413`，postops 后续层（numa/fa-gate/gemm）也不涉及它，安装器 SHA 校验为最终依据。

## 改动

- `orig`：原 `_causal_conv1d_fwd_kernel` 源码不变，只改 BLOCK_M/BLOCK_N/num_warps/num_stages。
- `tile2d`：`causal_conv1d_ttft.py::_causal_conv1d_fwd_kernel_tile2d`。前半段（初始状态读取、
  conv_state 更新）逐字复制厂商 kernel；只把逐 token 串行循环换成一个 [BLOCK_M,BLOCK_N]
  tile。每个输出仍是 FP32 起点（bias 或 0）依次 `+= x[t-3]*w0, x[t-2]*w1, x[t-1]*w2, x[t]*w3`
  （BF16 乘积、同顺序）、同一 SiLU 表达式、store 时同一 BF16 cast；chunk 前 3 个先前 token
  用同一组寄存器值（chunk0 来自 conv_state/0，其余来自 x）。仅 width 4、BF16。
- `install.py`：SHA256 校验原文件；只加一个 import 和 launch 处分支——`select_launch` 返回
  None（`SGLANG_QWEN27B_CONV1D_TUNE=0`，或非 BF16/width4/channel-last/有 state+indices
  的调用）时执行原 launch 文本（逐字保留），否则用 config 的 kernel 与参数。备份为
  `causal_conv1d_triton.py.ttft_conv1d_original`（非 .py）。config 从 `config.json` 嵌入 helper；
  `status` 不是 `selected` 时拒绝安装，防止误建空候选。
- 本地已做：AST 检查、对镜像真实文本的模拟安装与 diff、numpy 模拟（BLOCK_M 4–64 的
  orig/tile2d 分块索引与状态更新和 BLOCK_M=8 逐位一致）。未在 GPU 运行。

## Root 运行命令（GPU4，预计 1–2 分钟，主要是 ~27 个配置的 Triton 编译）

```bash
ssh root@bw100
docker run -d --name q38_c2c3_bench --network host --ipc host --shm-size 16g --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video \
  --security-opt seccomp=unconfined -u root \
  -v /opt/hyhal:/opt/hyhal:ro -v /data/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8:/model:ro \
  -v /data/tools/q38-bw100-sglang:/bench --entrypoint /bin/bash \
  sglang-bw100:ttft-postops-issue3 -lc 'sleep infinity'

docker exec -e HIP_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4 q38_c2c3_bench timeout 300s \
  python3 /bench/patch/ttft_conv1d_tune/bench_conv1d.py \
  --output /bench/result/ttft/issue3-conv1d/conv1d-sweep.json \
  --write-config /bench/result/ttft/issue3-conv1d/conv1d-selected.json
```

案例：`prod_16x4096`、`prod_65536`（计时，has_init 全 False）、`mixed_16x4096`（交替 has_init）、
`init_65536`、`odd_63_129`、`short_1_2_3_5_4096`（seqlen<3 的移位路径），全部非零随机初始
conv_state、乱序 cache slot、生产 stride；前 `--top-extra 4` 个最快配置再跑 `odd_bias`（带 bias）
和 `contig`（连续布局）。输出与整个 conv_state 池均按 int16 位视图比较，参考 kernel 先自检确定性。
打印排序表（中位数 ms、相对厂商 launch 加速）与 JSON（`recommended` 为两类案例均逐位一致的最快配置）。
`--grid-json FILE` 换网格，`--only-config FILE` 只验某个配置。

选定后：把 `conv1d-selected.json` 内容复制到 `patch/ttft_conv1d_tune/config.json`（`status: selected`），然后

```bash
docker build -t sglang-bw100:ttft-conv1d-issue3 patch/ttft_conv1d_tune   # BASE_IMAGE 默认 ttft-postops-issue3
# 在新镜像的 sleep 容器（同上 docker run，换镜像名/容器名）中复核已安装模块与备份逐位一致：
docker exec -e HIP_VISIBLE_DEVICES=4 <container> timeout 300s python3 \
  /bench/patch/ttft_conv1d_tune/bench_conv1d.py --installed \
  --output /bench/result/ttft/issue3-conv1d/conv1d-installed.json
```

回滚：服务启动前 `SGLANG_QWEN27B_CONV1D_TUNE=0`。微测不替代模型 TTFT 与输出 hash 验收。
