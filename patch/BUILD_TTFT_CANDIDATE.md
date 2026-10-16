# BW100 TTFT 统一优化镜像构建指南 (Release)

本项目已将经过充分压测验证的所有有效优化，全部收敛整合为 **`patch/ttft_combo`** 单一优化包，直接基于海光官方原生镜像一步完成构建，无需任何中间镜像层。

## 架构组成

**`patch/ttft_combo`** 包含本次挑战赛全部采纳的核心优化项：
1. **GDN KKT 融合求解内核** (`chunk_fwd.py`)：消除逆缓冲区冗余初始化
2. **Cat-Free 8-Part MLP 流水线** (`qwen27b_mlp_pipeline.py`)：RS + in-place AG 消除 `torch.cat`
3. **hipBLASLt 10 形状白名单派发** (`compressed_tensors_w8a8_int8.py`, `q38_bw100_legacy_gemm_dispatch.py`)：支持 64K 超长上下文
4. **Fused GDN Prepare 4-warp 优化与批元数据预计算** (`gdn_backend.py`, `gdn_triton.py`, `chunk*.py`, `layernorm_gated.py`)
5. **Paged KV 变长连续打包加速** (`pack_paged_kv_to_varlen.py`)
6. **BF16 QK Normalization 与 Attention 门控融合折叠** (`qwen3_5.py`)
7. **Causal Conv1d 发射调优** (`causal_conv1d_triton.py`, `causal_conv1d_ttft.py`)：`BLOCK_M=64, BLOCK_N=512`
8. **Token-Tiled mRoPE 向量化连续访存算子** (`triton_kernels.py`, `mrope_tiled.py`)
9. **NUMA node 4 硬绑定与 8-channel RCCL 拓扑调优** (`start_server.sh`)

## 一键构建镜像

```bash
# 从海光官方原生镜像一步直接构建包含全部补丁的最终镜像
bash patch/build_ttft_candidate.sh

# 仅预检查看构建命令 (Dry Run)
bash patch/build_ttft_candidate.sh --dry-run
```
