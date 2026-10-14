# BW100 GDN KKT + 求逆输出缓冲区优化

跟踪任务：[Issue #1](https://github.com/egtoken-ai/q38-bw100-sglang/issues/1)。源码基线来自 `root@bw100` 的 **`sglang-bw100`** 容器。

## 改动

基础镜像已经融合 KKT 预计算和下三角求逆。本补丁保持计算和精度路径，针对输出缓冲区做进一步优化：

- 原实现先用 `torch.zeros` 初始化整个 `[B,T,H,64]` 输出，再由融合 kernel 写入 10 个下三角子块。
- 新实现使用 `torch.empty`，融合 kernel 写入原来的 10 个子块，同时把 6 个严格上三角子块写为零，移除独立清零操作及下三角区域的重复写入。
- 尾块存储仍受每条序列的局部长度约束，避免覆盖后续打包序列。
- 仅应用于实际使用的 `C=64`；其他 chunk size 走原函数，不新增其支持承诺。
- 原文件保留为 `chunk_fwd_baseline.py`，设置 `SGLANG_GDN_ZERO_INIT_BASELINE=1` 并重启进程可回退。

`chunk_fwd.py` 是完整优化源码；`changes.patch` 是相对容器基线的最小差异。保留上游版权声明。

## 构建与验证

将本目录复制到 BW100 主机后执行：

```bash
bash build_and_verify.sh
```

脚本默认读取 `sglang-bw100` 的基础镜像和源码 SHA256，拒绝不匹配的基线；构建 `sglang-bw100:gdn-kkt-solve-issue1`，使用带设备映射的临时容器 `sglang-bw100-gdn-verify` 验证，结束后自动删除临时容器。原来的 `sglang-bw100` 保持不变。默认使用设备 0；通过 `DEVICE` 选择空闲设备，`RESULT_DIR` 指定结果目录。

基础环境需要主机 `/opt/hyhal` 驱动库和 `/dev/kfd`、`/dev/mkfd`、`/dev/dri`；原浏览容器没有这些映射，不能直接承担设备验证。

```bash
# 只做正确性验证
bash build_and_verify.sh --correctness-only

# 调整重复测量次数
DEVICE=0 bash build_and_verify.sh --rounds 9 --repeats 40
```

构建上下文被 `.dockerignore` 限制为必要源码、安装程序、验证程序和 Dockerfile，不会包含项目 `data/`。

## 验证覆盖与解释

- BF16/FP16、等长/变长、1/15/16/17/63/64/65/127/128/129-token 边界。
- CPU FP64 独立下三角求逆参考；原融合实现与新实现的 A/W/U 对照。
- NaN 预填充输出，检查完整写入与严格上三角零值。
- Qwen GDN TP=2 形状：8 个 key heads、24 个 value heads、K/V 维度 128。
- 完整 GDR Prefill 对照：非零初始状态、非连续状态槽索引、无关状态槽不变。
- 纯 kernel、含分配的 KKT+求逆、包含 W/U 的块内阶段分别计时；同时记录 eager 和 graph，交替测量顺序，保留每轮样本。
- 分离 KKT+求逆只用于解释既有融合的价值；本补丁的收益必须相对当前融合基线计算。

这些是算子验证，不等同于真实模型的 TTFT/TPOT 或精度准入。模型级验证与结果解释见实际验证报告；未完成的验证不得写成通过。
