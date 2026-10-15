# BW100 GDN post-conv preparation fusion

独立于 `ttft_gdn_views` 的第二阶段候选：融合 conv 后 Q/K/V split 和连续化、Q/K L2 norm、GDN gating。来源是 [SGLang PR #38806](https://github.com/sgl-project/sglang/pull/38806)，审查 head `62e3bdfb899cf158d1aecd2c070d3f1b3a4264da`（Open）；并对照 [vLLM PR #37813](https://github.com/vllm-project/vllm/pull/37813)，commit `4eefbf9609e5ddb996e3ac37e192e92466ec35cc`。helper 是前者的 Triton kernel 加本项目 BF16/shape guard；没有采用 gfx950 recurrence launch 调参。

## 数据和数学约束

- 只在 HCU non-spec extend、Triton GDN、local Q/K heads=8、V heads=24、D=128、BF16 QKV/A/B 时启用。
- 输入同时传 token stride 和 feature stride，支持真实 conv 后的 transpose tensor；V 原值复制。
- Q/K 在 FP32 做平方和、`x / sqrt(sum(x*x)+1e-6)`，结果写回 BF16，再交给 chunk recurrence。
- `g=-exp(A_log)*where(x<=20, log(1+exp(x)), x)`，`x=a+dt_bias`，保持 vendor 公式。
- beta 先 sigmoid 后转输入 B 的 BF16，再存 FP32，与 vendor 一致。g/beta 的 FP32 存储是原有接口；没有将它们改为其他精度。
- `TritonGDNKernel.extend` 新参数默认 True；只在 fused prepare 成功的分支传 False，防止重复 Q/K norm。其他路径不变。
- 规约的 Triton tile 可能产生末位差异；必须通过真实设备、chunk state、模型输出和精度验证。vLLM 原版使用不同 softplus 数值表达式且 beta 不做 BF16 中间舍入，因此未直接复制其数学逻辑。

## 构建与诊断

```bash
docker build -t sglang-bw100:ttft-gdn-prepare-issue3 patch/ttft_gdn_prepare
# 由主 agent 选择 sleep 容器、确认 GPU 空闲后运行：
HIP_VISIBLE_DEVICES=4 python3 /opt/ttft-gdn-prepare/bench_prepare.py \
  --output /bench/result/ttft/issue3-attention/gdn-prepare.json
```

默认基于 `sglang-bw100:ttft-gdn-views-issue3`；不修改 views 源文件。安装器校验原 `gdn_backend.py` 和 `gdn_triton.py` SHA256，预编译后才写入，保留 `.py.ttft_prepare_original`。已有 helper 则拒绝覆盖。

回退：`SGLANG_QWEN27B_GDN_FUSED_PREPARE=0`，仍可保留 views 优化。新容器默认值 True。

`bench_prepare.py` 使用 BF16，覆盖 T=192/4096/8192、token-major 与 feature-major、strided BA；Q/K 对比实际部署的 `l2norm_fwd`，V/g/beta 要求逐位相同。小序列还比较 chunk 输出和最终 state。微测探索 num_warps=4/8/16，设备实测 token-major 布局下4优于8/16，服务默认选择4；feature-major仍作为诊断边界，未以其数据替代真实布局。局部计时不代表模型 TTFT。

本地仅 `py_compile` 通过；未启动 GPU 或服务，未声称模型准确性和收益已经验证。

BW100 首轮6种token数/布局检查通过，Q/K最大绝对差≤0.001953125，V/g/beta逐值一致；192-token完整chunk输出最大绝对差0.00012207、最终state差0。8192 token-major准备阶段0.375ms→0.147ms（4 warps）；feature-major明显更慢，需确认模型实际输出stride。结果位于忽略目录 result/ttft/issue3-attention/gdn-prepare.json。仍需模型验证。
