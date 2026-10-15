# 连续 Prefill 先处理上一批输出

这是镜像已有功能的配置候选，不声称新移植了调度实现。
`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1` 在连续两个 Extend 批次时，
使 `event_loop_overlap` 先 `pop_and_process` 上一批，再 `run_batch` 当前批。
Decode→Decode 仍保持原 overlap。当前 HCU 镜像的 `scheduler.py:1709`
已包含该判断及 DP rank 一致性逻辑；本项目运行 TP2/DP1。

官方当前实现也保留该功能：[固定版本源码](https://github.com/sgl-project/sglang/blob/a776851c97253bf6bb43e9de9dc543cd22d8de73/python/sglang/srt/managers/scheduler.py#L2050)、
[DP一致性修复 #20853](https://github.com/sgl-project/sglang/pull/20853)。
[Issue #10514](https://github.com/sgl-project/sglang/issues/10514) 描述过下一批提交阻塞上一批输出的类似问题；其其他硬件测量不能外推到 BW100。

候选保留当前stream四子批、32K prefill chunk、BF16/INT8和已测9个GEMM形状，
只改变连续Prefill的结果处理顺序。64K C1 内部两个chunk也会触发，可能因CPU空洞
损失吞吐，必须独立测量。不能把单层930ms阻塞直接换算为C64整体TTFT收益。

```bash
docker build -t sglang-bw100:ttft-prefill-release-issue3 patch/ttft_prefill_release
```

回退：启动服务时设置 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=0`，
或使用父镜像并沿用相同的9-shape白名单。此目录没有修改SGLang Python源文件。
固定两个case、一轮预热/一轮测量已完成，用时270秒。短TTFT14.714893s，相对恢复后的安擎基线提升17.16%；长TTFT10.350391s，提升34.91%。短档仍未达到20%，不宣称Goal完成。正式测量65/65输出hash与metadata+KV父候选相同；预热59/65相同，短档预热6个输出不同；完整原生任务精度另验。结果与实际环境快照：`result/quick/ttft-prefill-release/20261015-183754-FJen1s/`。
