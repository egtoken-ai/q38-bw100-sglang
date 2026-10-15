# BW100 GDN launch 配置诊断

镜像源码可以做小范围实验，无需改动算法或生产文件：

- `fla/chunk_delta_h.py`：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 单配置 autotune，BV=32、warps=4、stages=2。
- `fla/chunk_o.py`：`chunk_fwd_kernel_o` 固定 BK=128、BV=64、warps=4、stages=2。
- `fla/chunk.py` 将这两个阶段串起来；实验比较完整 `chunk_gated_delta_rule` 输出和最终 state。

[上游 SGLang #38806](https://github.com/sgl-project/sglang/pull/38806) 提供 gfx950 的调参方向，本脚本只把这些设置当候选，在 BW100 上重新测量。默认测试 Hq/Hk=8、Hv=24、K=V=128，BF16 QKV、原接口 FP32 g/beta/state，BT 保持64。序列为192、4096、4096+4096、8192；`--include-long` 可另测64K，默认不运行。

```bash
HIP_VISIBLE_DEVICES=4 python3 /bench/patch/ttft_gdn_tuning/bench_launch.py \
  --output /bench/result/ttft/issue3-attention/gdn-launch.json
```

脚本仅临时包装当前 Python 进程的 kernel launcher，结束恢复原对象，不修改镜像文件。chunk_h 绕过多配置 autotune，每个候选只有单次指定 launch；每次完整调用前把 state 恢复同一随机初值，避免候选测试污染 state。两臂计时均包含相同 state reset。先用显式原配置验证 wrapper 与部署基线一致，再分别试4个 h 候选、3个 o 候选，最后只组合通过且更快的候选。记录完整 output/state max-abs、relative L2，失败候选不计收益；设备执行错误立即停止。

必须由主 agent 在独占 GPU 时间串行运行。微测收益不能替代固定两个模型 case 的 TTFT验收，也不自动选择生产配置。当前只通过本地语法检查，未启动 GPU。
