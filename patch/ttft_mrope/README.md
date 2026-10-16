# C3: token-tiled mRoPE kernel

## 路径确认

模型为 `Qwen3_5ForConditionalGeneration`，config `rope_parameters` 含 `mrope_section [11,11,10]`、
`mrope_interleaved: true`，因此 `get_rope` 构造 `MRotaryEmbedding`；纯文本请求也使用 3 行
`mrope_positions`（三行相同、int64、`[3,T]` 连续），走 `forward_triton` →
`rotary_embedding/triton_kernels.py::triton_mrope_fused` → `_triton_mrope_forward_fused`
（profile 中 grid `[65536]`/`[4096]`，block 256）。生产 q/k 来自 QK norm/gate patch，
q `[T,3072]`、k `[T,512]` 连续 BF16；cos_sin_cache 被 `_match_cos_sin_cache_dtype` 转为 BF16
`[262144,64]`。每 rank 12 Q 头、2 KV 头、head 256、rotary 64（partial 0.25），neox。
厂商 kernel 每 token 一个 program，tile [16,128]，只 12×32 lane 有效（≈81% mask）。
原文件 SHA256 `49b3a2ca2784b4ffb0773947d2e26055f55a520c9cd847b86bcc3f6c2bf94d05`，已有 patch 未改。

## 改动

- `mrope_tiled.py::_triton_mrope_forward_tiled`：每 program 处理 BLOCK_T 个 token × 实际
  rotary 半宽（32 lane），静态循环实际头（`PER_HEAD=true` 时改为 grid 第二维每头一个 program）。
  逐元素运算不变：同样的 interleaved t/h/w mask 公式、三次 `other=0` masked load 后按
  `t+h+w` 在 cache dtype 中相加、q/k `.to(sin_row.dtype)`、`x1*cos-x2*sin` / `x2*cos+x1*sin`、
  原 dtype 无 cast store。只接管 neox + interleaved（非 GLM）+ BF16 + 行内连续 + cache stride
  等于 rotary_dim 的调用，其它情况及 0 token 回退原 launch。
- `install.py`：SHA256 校验；只加 import 和在原 launch 前插入 `if _ttft_try_mrope_tiled(...): return`，
  原 launch 逐字保留。备份 `triton_kernels.py.ttft_mrope_original`（非 .py）。config 从
  `config.json` 嵌入；`status` 非 `selected` 拒绝安装。
- 本地已做：AST、真实文本模拟安装与 diff、numpy 模拟 cos/sin 行（lane<32 与厂商逐位一致）。未在 GPU 运行。

## Root 运行命令（GPU4，预计 < 1 分钟）

沿用 C2 README 中的 `q38_c2c3_bench` sleep 容器（`sglang-bw100:ttft-postops-issue3`）：

```bash
docker exec -e HIP_VISIBLE_DEVICES=4 -e CUDA_VISIBLE_DEVICES=4 q38_c2c3_bench timeout 300s \
  python3 /bench/patch/ttft_mrope/bench_mrope.py \
  --output /bench/result/ttft/issue3-mrope/mrope-sweep.json \
  --write-config /bench/result/ttft/issue3-mrope/mrope-selected.json
```

案例：`prod_16x4096`（16 段 0..4095）、`prod_65536`、`bs1_4096`（计时）、`decode_16`、
`odd_63_129_prefix`（起点 100/70000）、`one_token`、`random_thw_777`（三行不同的随机位置，
覆盖 h/w mask），以及 `strided_*`（q/k 为 `[T,7168]` 缓冲的切片视图，比较整个缓冲区）。
q/k 全张量（含非 rotary 维）按 int16 位视图比较；参考 kernel 先自检确定性。约 19 个配置
（BLOCK_T 4–128 × num_warps 1–8，含 per-head 变体），打印排序表与 JSON。

选定后复制到 `patch/ttft_mrope/config.json`（`status: selected`），再

```bash
docker build -t sglang-bw100:ttft-mrope-issue3 patch/ttft_mrope   # 可 --build-arg BASE_IMAGE=sglang-bw100:ttft-conv1d-issue3 叠加 C2
docker exec -e HIP_VISIBLE_DEVICES=4 <new-image container> timeout 300s python3 \
  /bench/patch/ttft_mrope/bench_mrope.py --installed \
  --output /bench/result/ttft/issue3-mrope/mrope-installed.json
```

两个安装器互不触及对方文件，可任意顺序叠加。回滚：`SGLANG_QWEN27B_MROPE_TILED=0`。
decode/graph capture 也会走新 kernel（形状静态，exactness 已覆盖 16/1 token）；若需只限
prefill 再加 token 门限。微测不替代模型 TTFT 与输出 hash 验收。
