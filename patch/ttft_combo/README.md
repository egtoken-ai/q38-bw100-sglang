# Qwen3.8 BW100 TTFT Combo 统一优化补丁 (Release)

本目录收敛了海光 BW100 双卡平台针对 Qwen3.8-27B-Channel-INT8 模型经压测验证的全部有效核心优化。

## 核心优化模块清单

1. **Cat-Free 8-Part MLP 流水线 (`qwen27b_mlp_pipeline.py`)**：
   - 采用 Reduce-Scatter + 原地 All-Gather 机制，消除子批微流水线尾部的 `torch.cat` 显存重分配开销。
   - 优化通信计算重叠，小批次与大批次全覆盖。

2. **hipBLASLt 10 形状白名单派发 (`compressed_tensors_w8a8_int8.py`, `q38_bw100_legacy_gemm_dispatch.py`)**：
   - 针对 BW100 DCU 矩阵核固化 10 个核心形状派发白名单，涵盖常规与 64K 超长上下文（包含 `(65536, 5120, 3072)`）。

3. **GDN 线性注意力加速 (`gdn_backend.py`, `gdn_triton.py`, `chunk*.py`, `layernorm_gated.py`)**：
   - Fused GDN Prepare 4-warp Triton 算子，GDN 状态计算视图优化与提前批元数据预计算。

4. **Paged KV 打包优化 (`pack_paged_kv_to_varlen.py`)**：
   - Prefill 阶段对离散 Paged KV 进行重排打包，消除注意力访存碎片。

5. **图折叠与算子融合 (`qwen3_5.py`)**：
   - Attention 输出门控 Sigmoid 计算与注意力计算图折叠。
   - BF16 QK Normalization 与门控特征向量融合提取。

6. **Causal Conv1d 调优 (`causal_conv1d_triton.py`, `causal_conv1d_ttft.py`)**：
   - Triton Launch 网格参数优化（`BLOCK_M=64`, `BLOCK_N=512`），显著提升卷积核吞吐。

7. **Token-Tiled mRoPE (`triton_kernels.py`, `mrope_tiled.py`)**：
   - 旋转位置编码分块向量化加载，实现多模态与长文本下的高效连续访存。

8. **NUMA 与通信调度 (`start_server.sh`)**：
   - 双卡硬绑定至 NUMA node 4，消除跨 NUMA 内存访问延迟。
   - 限制 `NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8`，降低 GEMM 算力争用。
   - 最佳分块调度参数：`CHUNKED_PREFILL_SIZE=16384`，`MAX_PREFILL_TOKENS=32768`。

## 构建方式

```bash
# 从 Issue 1 镜像构建最终组合镜像
docker build -t sglang-bw100:ttft-combo patch/ttft_combo
```
