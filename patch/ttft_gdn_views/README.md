# Qwen3.8-27B BW100 GDN prefill projection views

来源：[SGLang PR #36267](https://github.com/sgl-project/sglang/pull/36267)，已合并 commit `07199fa220cc353a2b5230b8bf47a0dead7ea96c`；[HIP 扩展 PR #38806](https://github.com/sgl-project/sglang/pull/38806)，2026-10-15 API 核实仍 Open，head `62e3bdfb899cf158d1aecd2c070d3f1b3a4264da`。这里只移植 projection view 和 3D Z norm，不采用 gfx950 launch 参数或新增 fused prepare。保留 vendor Apache-2.0 声明。

镜像的 27B GDN value/key head ratio=3，不在旧 fused-unpack `[1,2,4]` 列表，每层 prefill 因而复制 B/A、拼接 QKV，随后 Z reshape 再复制。本补丁让非 speculative extend 直接传入 projection 的 strided views，并使原 Triton gated norm 直接消费 3D Z。改动不改变 BF16 数据、权重、累计公式。3D Z 不能交给 vendor 可选 lightop norm，因而该路径使用现有 Triton norm；需要实测确认净收益。

`install.py` 对两个原 vendor 源文件校验 SHA256，先校验全部文件再写入，并备份 `.py.ttft_views_original`。完整原文件保存在本目录供审查；最小差异在 `gdn-prefill-views.patch`。

```bash
docker build -t sglang-bw100:ttft-gdn-views-issue3 patch/ttft_gdn_views
# 在指定 sleep 容器内串行运行，不与推理服务/其他 GPU 实验重叠：
HIP_VISIBLE_DEVICES=4 python3 /opt/ttft-gdn-views/bench_views.py \
  --output /bench/result/ttft/issue3-attention/gdn-views.json
```

默认 base 是 `sglang-bw100:ttft-packed-issue3`；GPU 与容器调度由主 agent 控制。`SGLANG_QWEN27B_GDN_PREFILL_VIEWS=0` 恢复旧模型路径。Decode、target verify、其他 GDN head geometry、非 BF16 路径均走原实现。新路径保留原 residual、state、cache 参数。

`bench_views.py` 覆盖 TP2 的 8 K / 24 V heads、128 head dim、短尾/双序列/4096/8192 tokens：projection 数值精确，strided gating 精确，生产 Triton conv 输出和 state 精确，gate 在 norm 前后的 BF16 一致性，以及 unpack+gating+norm 整链计时。计时不含 conv，因此应视为局部诊断；它不能替代固定 4K/C64 和 64K/C1 模型 A/B。若启用了 `SGLANG_USE_CAUSAL_CONV1D` 的 HCU 二进制替代实现，必须额外验证该实现的 strides，当前微测覆盖默认 Triton conv。

本次交付仅本地语法检查通过；GPU 正确性、端到端性能、模型准确性由主 agent 串行验证，尚不宣称达标。
