# Qwen27B GDN CPU metadata 候选

对应 LoopX goal `q38-bw100-gdn-fusion` / Issue #3。借鉴 [vLLM #38361](https://github.com/vllm-project/vllm/pull/38361) 的已合并方案（`cb10b7e80b138f876079c2cd697df29f0965b885`），从已有CPU长度构建chunk indices/offsets，避免GDN prefill第一次 `prepare_chunk_indices` 的GPU→CPU `.tolist()` 同步；同时把48个GDN层重复的 `extend_prefix_lens > 0` 合为每step一次CPU构建/异步H2D。

上游正式TTFT没有证明改善，本候选也尚未完成模型性能验证。它不改变输入/输出长度、prefix cache、量化精度、attention或GDN数学kernel。

## 基础镜像与安装边界

Dockerfile默认 `FROM sglang-bw100:ttft-gdn-prepare-w4-issue3`。`vendor/` 保留从该基础镜像的临时CPU容器读取的8个原文件，`vendor-sha256.json` 保存实际SHA256。恢复后的prepare容器与此基础镜像的这8个文件相同；既有NT_BUCKET等代码保持原样，不纳入本补丁。

`patched/` 只修改5个文件：GDN backend、Triton GDN dispatcher、FLA chunk、chunk_delta_h、chunk_o。增加独立helper `gdn_cpu_metadata.py`。安装器先核对全部8个依赖文件SHA并编译检查全部候选，再写替换与 `.py.ttft_metadata_original` 备份；无法匹配时拒绝安装，不尝试模糊patch。

KV-pack父镜像若不改变这些GDN文件，可直接叠加：

```bash
docker build --build-arg BASE_IMAGE=你的KV-pack父镜像 \
  -t sglang-bw100:ttft-gdn-metadata-issue3 patch/ttft_gdn_metadata
```

`SGLANG_QWEN27B_GDN_CPU_METADATA=1` 为启用开关，设0回退旧metadata构建方式。新API参数都是optional；非Triton backend不传这些新kwargs。后续FLA链会复用已有indices至chunk_o，数学结果不变；该少量Python缓存查询复用在flag关闭时也存在，严格旧source比较请使用基础镜像或保存的vendor源。

## 可信CPU metadata与guard

镜像 `ForwardBatch.init_new` 同时从scheduler的 `batch.extend_seq_lens`、`batch.extend_prefix_lens` 创建GPU tensors，并把同源Python lists放到 `extend_seq_lens_cpu`、`extend_prefix_lens_cpu`。本helper仅使用这些CPU lists，不从GPU补做 `.cpu()` / `.item()` / `.tolist()`。

模型guard：HCU、Triton GDN、TP2、dense `qwen3_5_text`、hidden5120、64层、linear K heads16/V heads48。每step还要求普通non-spec extend、无spec_info/TBO/capture，正整数extend lengths、非负prefix lengths、总token≥64且<2^31，并且batch数、extend_num_tokens、input_ids长度、query_start_loc形状一致。任何不满足者整条metadata候选回退原路径。零长度/padding请求、小于64token、draft/target verify/decode不使用新路径。

CPU indices精确复现原 `arange(chunks) → eq(0).cumsum` 的序列编号；offsets使用每请求ceil(length/64)的前缀和。indices为INT32，与原 `.to(query_start_loc)`一致；offsets为INT64，与原Torch cumsum默认提升一致；has_initial为BOOL。分别用pinned CPU tensor异步H2D，保留源buffer引用至本step metadata，不用全局identity缓存。

传递链：

```text
GDNAttnBackend.init_forward_metadata
 → ForwardMetadata.gdn_cpu_chunk_indices/offsets/has_initial_states
 → GDNAttnBackend.forward_extend（只向Triton加kwargs）
 → TritonGDNKernel.extend
 → chunk_gated_delta_rule / autograd Function / fwd
 → cumsum + intra + chunk_delta_h + chunk_o
```

chunk_o原来按tiny-T动态选择BT；新路径要求总T≥64，确保BT64相同；其它调用仍保留原动态BT fallback。原GPU query_start_loc保留，未替换其语义或缓存映射。首次实际命中会记录 `Qwen27B GDN CPU metadata active: ...`。

## 验证

已完成：纯CPU108组长度/前缀/尾chunk对比，非法metadata拒绝与decode回退，8文件SHA guard，全部Python编译检查。没有由本agent执行GPU测试。

```bash
# 无torch/GPU导入的CPU检查
python3 patch/ttft_gdn_metadata/verify_metadata.py

# 主任务在新镜像、GPU空闲时执行
python3 /opt/ttft-gdn-metadata/verify_metadata.py --device \
  --output /bench/result/ttft/gdn-cpu-metadata-verify.json
```

设备验证包含：[1,63]、[63,64,65,129]、多4096请求、8192/32768/65536及尾1token的metadata逐值比较；BF16和FP32初始SSM state下的GDN输出、全部中间h和SSM pool逐值相同检查。新路径执行时把各FLA模块的 `prepare_chunk_indices/offsets` 替换成抛错函数，以防漏传导致实际仍走GPU metadata fallback。测试不会放宽误差容差：这些比较要求 `torch.equal`。

通过后仍需模型真实命中日志、相同历史logprob/实际任务精度及固定两个case的TTFT A/B。不能把取消同步本身当作达成20%目标。

## Root设备与模型验证

GPU4：8组metadata逐值一致；3组变长GDN分别使用BF16/FP32 state，输出、中间h及全部SSM pool逐值一致；禁止fallback探针通过。证据`result/ttft/gdn-cpu-metadata-verify.json`。

叠加直接KV打包的镜像`sglang-bw100:ttft-metadata-pack-issue3`已完成固定两档模型筛选，短TTFT16.179167s、长10.380501s，对恢复后原生17.762641/15.901140s分别提升8.91%/34.72%。真实模型日志确认CPU metadata命中；6项输出回归与上一GDN32K父候选完全一致（4个简单问答及两档16token随机续写）。仍不是完整任务精度通过，短输入未满足20%。见`result/quick/ttft-metadata-pack/20261015-173302-OByAtv/`。
