# 直接分页 KV 打包候选

Issue #3 / LoopX `q38-bw100-gdn-fusion`。父镜像为已完成设备验证的
`sglang-bw100:ttft-gdn-prepare-w4-issue3`；本目录独立构建候选，默认镜像
通过 `SGLANG_QWEN27B_FUSED_KV_PACK=true` 启用，设置为 false 回到原打包代码。

原 packed 路径对每个请求分别执行 page index 转换、K/V index_select、
permute/reshape 和最后 cat。新 kernel 直接读取 page table，将
K[P,H,T,D]、V[P,H,D,T] 写入连续的 K/V[token,H,D]，不改变缓存内容、
token 顺序、因果 mask 或 attention 数学。每个 sequence 一个独立 grid 维度，
使用现有 GPU cu_seqlens_k 计算目标偏移，不需要另建 token→sequence 表。

适用范围严格限制 BF16、TP2 KV heads=2、head dimension=256、page size=64。
沿用父路径的非 speculative Extend、全局 attention 等保护条件。
保留 CPU seq_lens 求和用于输出分配；未声称消除所有 CPU 工作。

`install.py` 用保存的父源文件 SHA-256 比较，拒绝安装到不同源码上。
`bench_pack.py` 包含乱序分页、尾页、非连续 page table、不同长度、
8/64 个短请求及64K长请求；K/V必须与旧实现逐值一致，然后才计时。
微测不代表模型 TTFT 或整体精度验收。

```bash
docker build -t sglang-bw100:ttft-kv-pack-issue3 patch/ttft_kv_pack
# 仅在 GPU4 空闲时执行，真实设备验证由主任务串行管理。
HIP_VISIBLE_DEVICES=4 python3 patch/ttft_kv_pack/bench_pack.py \
  --output result/ttft/issue3-attention/kv-pack.json
```

BW100 GPU4实测：6组长度/布局、2组tile，K/V全部逐值一致。
8×4096 tokens打包阶段约1.052→0.489ms，64K单序列约1.317→0.974ms。
选择block=1024；结果在 `result/ttft/issue3-attention/kv-pack-recovery.json`。
这仅是打包阶段收益，尚未证明模型TTFT提升。父调用方负责保证GPU
cu_seqlens与CPU长度一致；helper不增加GPU→CPU同步去重复读取这些值。

组合模型筛选（与CPU GDN metadata共同加入）：短输入TTFT16.179167s、长10.380501s，恢复后原生对应17.762641/15.901140s。不能将组合收益全部归因于KV打包；短档8.91%仍不达20%目标。6项输出快照与前一GDN32K候选一致，完整任务精度另验。
