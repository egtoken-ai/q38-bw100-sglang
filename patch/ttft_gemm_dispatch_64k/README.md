# 仅 M65536 的独立 native hipBLASLt 筛选

用于内部 prefill chunk 65536 候选，保持固定外部测试 case 不变。只测试 `(M,N,K)=(65536,17408,5120)/(65536,8192,5120)/(65536,5120,3072)`；不重跑原 12 组，不覆盖 `result/ttft/gemm-dispatch-shared-import.json`。主 agent 已完成 GPU 验证，结果同步到 `result/ttft/gemm-dispatch-m65536.json`。三组均全量 exact、native hit=1；仅 out 满足性能筛选条件。本目录记录 micro 选项，不修改现有服务白名单。

```bash
python3 patch/ttft_gemm_dispatch_64k/bench_64k.py \
  --timeout 300 --output result/ttft/gemm-dispatch-m65536.json
```

父进程只启动一个共享导入的 child，三组顺序运行，默认硬超时 300 秒（可设定，但最大 450 秒），失败即停止后续 GPU 工作。整个 child 初始化也计入期限；不重复旧版每组 35 秒导入超时路径。日志保存于输出 JSON 同名 `.log`，三组各有独立 JSON。`--dry-run` 不导入 torch、不访问 GPU。

`module_worker.py` 复用已验证的真实 `create_weights/process_weights_after_loading/apply_weights` 测量逻辑，并保留 SHA 核验、full output exact、native hit=1、CPU 小片参考、4 次 warmup 和 8 轮 AB/BA（三次完整模块调用/段）。唯一入口调整是独立 micro 可以读取原始或精确已打 dispatch 补丁的已安装模块，在进程内从 SHA 校验后的原始 vendor 源加载 ASM baseline，然后动态编译 candidate。这样可在已有 dispatch 镜像的独立进程验证，不改已安装文件、不影响运行中的服务进程。安装源不匹配这两种精确 SHA 时立即拒绝。

本目录 helper 的候选集合增加 M65536，供独立 micro 使用；原目录 helper、镜像、服务和原白名单均未修改。`allowlist-extension.json` 现标明 `MICRO_VALIDATED_REQUIRES_MODEL_VALIDATION`，仅 out `[65536,5120,3072]` 入选，并记录原始结果 SHA256 和逐形状判断；它不是安装或启动文件。micro 输出中的 `approved_shapes` 只表示该次结果满足：full exact、native hit=1、全部 8 轮 paired faster、median 提升 ≥2%；是否采用仍由主 agent 审阅，并保留原 9 个已验证 shape 的证据。未测试/慢/失败的 M65536 shape 不应加到模型白名单。Down、decode 和其他形状仍不在本次新候选中。

M65536/up 单份 BF16 输出约 2.13 GiB，全量 exact 同时持有 baseline/candidate，需主 agent 确保 GPU 空闲及足够显存。这里只陈述脚本实际分配，不改变服务内存配置。8 parts 的 MLP 子批若为8192，可继续用已有8192证据；65536 筛选主要补齐未拆分的投影形状证据，不能代替最终模型 TTFT。

## 2026-10-15 实测与筛选

结果文件 `result/ttft/gemm-dispatch-m65536.json` 显示 suite 完成、进程退出码 0，三组全部全输出逐值一致、native hit=1、CPU 小片 max abs=0。共享导入内部耗时 33.981 秒，本次主 agent 设定硬超时 180 秒。原 12 组证据保留。

| M / N / K | ASM 中位数 ms | hipBLASLt 中位数 ms | 中位数提升 | 最小 paired 提升 | 决策 |
| --- | --- | --- | --- | --- | --- |
| 65536 / 17408 / 5120（up） | 30.083073 | 29.388809 | 2.3078% | -0.9908% | 拒绝：一轮更慢，继续 ASM |
| 65536 / 8192 / 5120（QKV） | 14.306594 | 14.129421 | 1.2384% | 0.6596% | 拒绝：中位数未达到 2%，继续 ASM |
| 65536 / 5120 / 3072（out） | 5.758984 | 5.538105 | 3.8354% | 3.7482% | 入选模型验证；8 轮全部更快 |

CPU 复核已从 8 轮原始样本重新计算 medians、paired 提升及筛选集合，与结果文件一致；检查脚本的源码 SHA、真分派断言、exact 门槛、父进程硬超时及失败停止路径，未发现需要变更的功能问题。没有重跑 GPU，也没有把 micro 改善当作模型 TTFT 或完整精度验收。只有完整 suite 成功且该 shape 满足筛选条件的结果可作为入选依据；超时或失败的部分输出不能替代完整证据。
