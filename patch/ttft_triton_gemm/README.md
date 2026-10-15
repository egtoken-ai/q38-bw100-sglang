# 有界 Triton INT8 GEMM 参数验证

对应 LoopX goal `q38-bw100-gdn-fusion` / Issue #3。此目录当前是独立 micro 工具，没有修改模型 dispatch，也没有由本 agent 执行 GPU 测试。

发现：旧 `patch/ttft_prefill/bench_gemm.py` 的纯 Triton 入口调用 `matmul_int8(..., None)`，M>1024 固定使用 BM256/BN256/BK64、warps8、stages0、GROUP8、SPLIT_K1，没有读 vendor shape JSON。所以之前“慢2–3倍”是这个默认配置的结果，不能推广为全部已调优 Triton 配置。

镜像实际路径：`lightop._lmslim_native.layers.gemm.int8_utils.matmul_int8`（:557）接受显式配置，并原样传给 `matmul_kernel`。支持 BM/BN/BK、GROUP_SIZE_M、SPLIT_K、num_warps、num_stages。`lightop.gemm_ops.triton_scaled_mm` 的 `best_config` 只在 ASM 没命中时才传到这里；本工具直接调用 underlying `matmul_int8`，不会误测 ASM。

实际目标 GPU 为 **gfx936/64CU**，依据此前 ASM worker 的 `actual_cus=64, selected_cus=64`。脚本只允许此实际设备。下表的80CU JSON仅为跨CU数的候选参数，不能称为当前64CU设备已调优配置。随后只读核查确认，这四组N/K的 `W8A8_<N>_<K>_gfx936_64cu.json` 全部不存在；实际64CU的目标形状仍存在配置缺口。

镜像 `lightop/configs/w8a8` 与 SGLang `W8a8GetCacheJSON` 的默认读取目录相同；`lightop.config.get_w8a8_config_dir()` 返回这个目录。只读核查如下：

| N/K | gfx936_80cu JSON | M8192 参数 |
| --- | --- | --- |
| 17408/5120 | 存在 | BM256 BN128 BK64 GROUP4 SPLIT1 stages1 warps4 |
| 5120/8704 | 存在 | BM128 BN256 BK64 GROUP8 SPLIT1 stages2 warps4 |
| 8192/5120 | 不存在 | 未发现匹配文件 |
| 5120/3072 | 不存在 | 未发现匹配文件 |

因此不做无边界泛扫，只先验证 M8192/N17408/K5120 的8个配置，包括80CU up JSON候选、旧默认、80CU down JSON候选，以及最多5个相邻 tile/stage 组合。所有配置 SPLIT_K=1，保持 INT32完整累加，输出BF16，per-token/channel scales仍FP32。

```bash
# 此命令需由主任务在GPU空闲时执行。
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
python3 patch/ttft_triton_gemm/bench_tuned.py \
  --output result/ttft/triton-gemm-bounded.json
```

约束与验证：

- 每个配置独立进程/进程组，50秒超时；整个脚本450秒截止，最多8组。超时终止包括编译器子进程。为确保外部调度上限，调用方也可使用 `timeout 480`。
- `kernel.warmup` 只编译，编译后读取实际 `metadata.shared`，大于64KiB标记SKIP，不launch该kernel；编译失败记录FAIL和错误。没有根据粗略 tile 大小假装证明LDS合法。
- 全量输出对照当前ASM，CPU FP64独立核验8×16小切片，记录 exact_fraction、max_abs_error、relative_l2_error；容差只是算子筛选，不是完整模型精度通过。
- 每种实现预热3次，再取5组×3调用的CUDA event中位数，编译不计入性能。每个worker独立重测ASM基线。
- 同时报告原 `matmul_int8` API（含 `torch.zeros` 分配初始化）与预分配输出的 kernel-only；两者输出必须位相同。后者只是分离初始化成本的诊断，不能冒充已安装模型路径。
- 每个worker保留JSON和日志；必须新output文件名，防止混读旧结果。不会写入安装目录或覆盖vendor JSON。

初版误将80CU配置文件名当作当前设备CU数，已在GPU执行前纠正为gfx936/64CU guard；配置数量没有扩大。已通过本地 py_compile 与无torch导入的 dry-run。GPU正确性、编译是否成功、吞吐收益均待主任务执行后确定。

## 已收集结果：前7组汇总完成，父流程中断

恢复后只读核查服务器 `/data/tools/q38-bw100-sglang/result/ttft/triton-gemm-bounded*`。本轮没有重跑 GPU。正式父汇总 `triton-gemm-bounded.json` 仅包含 config0–6，父日志末尾为 `No such exec instance`，因此不能标记整次运行完整成功。

| config | ASM ms | Triton API ms | kernel-only ms | 已见算子数值状态 |
| --- | ---: | ---: | ---: | --- |
| 0：80CU up候选 | 3.292 | 7.019 | 6.773 | PASS |
| 1：旧默认 | 3.304 | 8.058 | 7.832 | PASS |
| 2：80CU down候选 | 3.293 | 5.412 | 5.169 | PASS |
| 3：128×128 K128 | 3.302 | 5.379 | 5.126 | PASS |
| 4：256×128 K128 | 3.297 | 5.090 | 4.840 | PASS |
| 5：128×256 K128 | 3.299 | 4.691 | 4.444 | PASS |
| 6：128×128 K256 | 3.296 | 6.838 | 6.579 | PASS |

前7组全部慢于 ASM，最快config5的原API耗时仍约为ASM的1.42倍，即使只算预分配kernel也约1.35倍，因此不采用这些配置。

第8组（config7）没有被父流程收集，属于环境中断处。但独立 `triton-gemm-bounded-config7.json` 和对应日志实际存在完成记录：ASM3.297ms、Triton API6.335ms、kernel-only6.065ms，算子status=PASS。该独立artifact同样没有显示性能优势，但不能据此回填父进程正常退出或宣称整个runner完成。所有PASS仅表示脚本的算子检查通过，不是模型精度或两case性能验收通过。
