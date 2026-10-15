# W8A8 原生 hipBLASLt 限定形状分派候选

该候选使用镜像已有 `lightop.gemm_ops.hipblaslt_w8a8_gemm`，不加载此前 cached C ABI，不增加 workspace，不更换量化算法。12个实际模块形状在BW100上均通过全量输出逐值一致与native入口命中检查，其中9个满足8轮均更快、中位提升≥2%的白名单门槛。组合当前stream MLP修正的镜像 `sglang-bw100:ttft-current-gemm-issue3` 已完成固定两case测试，TTFT分别提升15.04%/35.00%；这不能单独归因于GEMM分派。两case全部130个预热/测量请求输出hash与metadata+KV父候选相同；原生模型完整任务精度验收仍未完成。

## 实际模块和布局

源快照来自容器 `qwen38_bw100_ttft_prepare` 的 `sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`，原文件与 SHA256 在 `vendor/`、`source-manifest.json`。

`CompressedTensorsW8A8Int8.create_weights` 创建连续 INT8 `[N,K]` 和 FP32 `[N,1]` channel scales；strategy 1 的 `process_weights_after_loading` 保存 `weight.t()` 为 `[K,N]`、stride `(1,K)`。候选在 `apply_weights` 已完成原来的 per-token quant（包括原有 fused quant 参数选择）后插入分派。传给 native 的 `weight.t()` 只是连续 `[N,K]` 视图；输入 INT8 `[M,K]`、原 FP32 `[M,1]`/`[N,1]` scales 不变，逻辑布局 `NT`，BF16 输出。已有 native 实现使用 INT32 accumulation；不经过 BF16 权重或 FP32 GEMM。

仅允许 gfx936 / **实际 64 CU**、strategy 1、动态对称 channel quant、无 bias、原布局和 contiguous scales。允许候选 M 为 4096/8192/16384/32768，N/K 为 17408/5120（up）、8192/5120（QKV）、5120/3072（out）。Down 5120/8704、decode、其他 M/N/K、其他策略/布局/精度均走原分派。`Q38_HIPBLASLT_SHAPES` 默认为空，因此所有形状默认关闭；将来只能填入已经 exact 且重复有稳定收益的精确 `MxNxK` 项。此候选没有改变进程级 `W8A8_SUPPORT_METHODS`，没有 strategy 3 全局切换。

## 有界验证

```bash
python3 patch/ttft_gemm_dispatch/bench_dispatch.py \
  --output result/ttft/gemm-dispatch.json
```

父进程按 12 个形状依次启动独立 worker，每个最多 35 秒，总执行预算 450 秒；fault、超时和失败均记录，不允许沿用上次 worker JSON。单 shape 调试可使用 `--worker --shape 32768 17408 5120 --output ...`，但直接 worker 本身不提供外层超时，需调用者加 `timeout`。CPU 检查可用 `--dry-run`。

脚本读取并校验安装模块 SHA，内存中编译一份 candidate 类，调用真实 `create_weights`、`process_weights_after_loading`、`apply_weights`；只用 stub 屏蔽 process_weights 的 JSON 写盘和可选配置 warmup。没有修改安装文件。真实模型若使用 fused quant 输入仍由原方法选择，micro 使用该方法原生动态 per-token quant 路径。

每组检查：

- 权重处理后 shape/stride、原始 data_ptr、transpose 连续性；decode/bias fallback，down 不在允许集合。
- 候选确实调用一次 native hipBLASLt，避免未命中回退造成假 PASS。
- 整个 `[M,N]` BF16 输出与原 ASM 模块 `torch.equal`；任何差异直接失败，不放宽阈值。
- 独立 CPU FP64 小片整数累加与 scales 参考（此处容许 BF16 舍入误差，不替代全量 ASM exact）。
- 先 warmup，再 8 轮交替 AB/BA，每段 3 次完整模块调用（包括 quant），CUDA events 计时；报告原始样本、中位数、逐轮提升和全部轮次是否更快。编译/初始化不计入性能样本，但受 worker 时间预算约束。

本脚本不输出自动安装白名单。白名单必须逐 shape 满足全量 exact、真实 native 命中、全部 8 轮 paired faster 且中位数提升 ≥2%；未测、失败、慢或非目标形状不启用。大 M 如果无稳定收益就拒绝这些形状；各形状结果不稳定或总体无收益则不安装本候选。即使 micro 通过，也还需真实模型数值和固定两 case TTFT 验证。CPU 检查不证明 GPU 可用或性能改善。

## 候选镜像安装与回退

```bash
python3 patch/ttft_gemm_dispatch/install.py --check
# 构建默认关闭分派的候选：
docker build -t sglang-bw100:ttft-gemm-dispatch-issue3 patch/ttft_gemm_dispatch
```

Dockerfile 默认父镜像 `sglang-bw100:ttft-metadata-pack-issue3`。安装器验证实际与 bundled vendor SHA、唯一插入锚点、helper/backup 不存在，保存原文件备份后写入补丁。helper 安装到包内唯一模块 `sglang.srt.layers.quantization.compressed_tensors.schemes.q38_bw100_legacy_gemm_dispatch`；真实模块通过完整包名导入，不依赖工作目录或裸模块 import。micro 在独立进程 `sys.modules` 中提供这个包名，从而验证相同 import 而不修改安装文件。

镜像显式 `Q38_HIPBLASLT_SHAPES=""`。服务使用 `approved-shapes.json` 中9个精确形状，以逗号分隔的 `MxNxK` 字符串传入；撤销环境变量或设为空即可回到原分派。安装器不会操作服务、GPU 或 LoopX。CPU 临时目录验证已覆盖只读检查、实际安装内容和拒绝重复安装，不能代替镜像构建验证。

## Root完整模块筛选

原独立worker的35秒限制被CPU冷导入触发，保留TIMEOUT数据，不将其当算子慢。改用同进程复用导入后，12组实际模块在36.57秒完成，全量输出逐值一致、native命中均为1。此模式只用于已执行过的native API，遇到首个异常立即终止，并由外层450秒timeout约束；不用于未知kernel调参。

| M | N | K | 模块耗时提升 | 8轮均更快 | 启用资格 |
| ---: | ---: | ---: | ---: | --- | --- |
| 4096 | 17408 | 5120 | 0.74% | 是 | 否 |
| 4096 | 8192 | 5120 | 2.03% | 是 | 是 |
| 4096 | 5120 | 3072 | -3.59% | 否 | 否 |
| 8192 | 17408 | 5120 | 3.98% | 是 | 是 |
| 8192 | 8192 | 5120 | 4.48% | 是 | 是 |
| 8192 | 5120 | 3072 | 2.10% | 是 | 是 |
| 16384 | 17408 | 5120 | 3.75% | 是 | 是 |
| 16384 | 8192 | 5120 | 3.29% | 是 | 是 |
| 16384 | 5120 | 3072 | 3.75% | 是 | 是 |
| 32768 | 17408 | 5120 | 1.78% | 是 | 否 |
| 32768 | 8192 | 5120 | 2.56% | 是 | 是 |
| 32768 | 5120 | 3072 | 3.83% | 是 | 是 |

白名单见`approved-shapes.json`，默认环境仍为空。结果`result/ttft/gemm-dispatch-shared-import.json`；该模块筛选不等于模型TTFT提升。实际模型已记录分派命中；固定case结果与部署快照见 `result/quick/ttft-current-gemm/20261015-181937-dugWWz/`。
