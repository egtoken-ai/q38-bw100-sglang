# BW100 INT8 ASM mode screening

此目录对应 LoopX goal `q38-bw100-gdn-fusion` / Issue #3。只提供隔离的微测工具，不修改模型或服务配置。本 agent 未运行 GPU 测试，尚无性能结果。

镜像源码显示真实调用链：

`lightop.gemm_ops.triton_scaled_mm` → `_lmslim_native.quantize.quant_ops.triton_scaled_mm` → `torch.ops.lmslim.lightop_channel_int8_mm` → `gemmopt.gemm_w8a8_smooth` → `op.gemm_w8a8_smooth_asm`。

`gemmopt.py:56` 的 `w8a8_smooth_asm_config(M,N,K)` 查 `config_smooth_gemm.py` 的 CU 专属表；若形状没有配置，返回 `(False,255)`。`gemmopt.py:101` 将这两个参数直接传给 native 算子。源码中的变量拼写是 `isTunning_slide_block` 和 `tunning_slide_block`。不同 mode 的具体底层 tile 映射在 native 二进制内，未拿到对应 HIP 源码，不能杜撰编号与 tile 的映射。

核查结果：80/72/64/48 CU 的表均缺少 M4096/8192 对应以下四组 N/K 的全部八项：17408/5120、5120/8704、8192/5120、5120/3072。80 CU 表出现的 mode 是 0–10。`lightop.utils` 会检测 CU 并设置 `LIGHTOP_GPU_CUS`；脚本在实际运行时再次要求实际 GPU CU 等于所选表，并要求测试 mode 存在于该表。上述表外形状当前走默认选择，但不能据此断言默认选择不是最优。

SGLang 传入 `best_config` 只影响 ASM 返回失败后的纯 Triton fallback，对已命中的 ASM 不生效。没有在已读 Python 包装中找到动态 per-token 量化与该 GEMM 合成一次调用的入口；当前接口显式接收 INT8 A 和已算好的 scale_a。不要把目录中的 FP8、MoE grouped GEMM 或 w8a16 接口误当作本任务的现成替代。

建议先在主 agent 确认 GPU 空闲后运行三组大形状：

```bash
HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 patch/ttft_gemm_tuning/bench_tuned.py \
  --m 8192 --shapes 0 1 2 --output result/ttft/asm-mode-m8192.json
```

脚本默认每个 mode 单独启动进程、120 秒超时，避免某个非法配置污染后续模式的 GPU context。每个 mode、每个形状均先测当前 public baseline，再调 native ASM 指定 mode。先 warmup 三次，再取七组、每组三次调用的 CUDA event 中位数，warmup/编译不计时。输出包括实际 CU、源码 hash、默认配置、每次时延、baseline 倍率、最大绝对误差、相对 L2 和逐元素完全相等比例。每个 mode 有独立 JSON 和日志；必须使用新的 output 文件名以免读入旧结果。

正确性检查包括完整输出与 public baseline 的 `rtol=.02/atol=.01` 对比，以及 8×16 输出切片的 CPU FP64 独立点积参考。容差仅用于排除明显错误，不代表模型精度合格；融合入模型前应优先选位相同的模式，并完成真实模型的相同历史数值比较与精度验证。

有收益后再跑默认八组形状，重复并交错核实，最后以精确 M/N/K/设备 guard 接入 `w8a8_smooth_asm_config`。该函数有 LRU cache，安装配置必须在模型首次执行前完成或明确清除 cache。不要扩大到未验证 decode、小 M、其它模型、其它 CU 数或其它 dtype。

仅 `python3 bench_tuned.py --output /tmp/unused.json --dry-run` 和 `py_compile` 已在本地通过；dry-run 不 import torch/lightop、不访问 GPU。模型的两个固定 case 仍必须分别满足 TTFT 提升 20%，微测改善不能代替验收。
