# BW100 TTFT 候选镜像构建指南 (Release)

本项目已将经过充分压测验证的所有有效优化，收敛整合为 **`patch/ttft_combo`** 统一优化补丁，直接消除历史开发过程中产生的 10 余个中间过渡层与覆盖目录。

## 架构组成

1. **`patch/gdn_kkt_solve`**：Issue 1 GDN KKT 融合求解内核。
2. **`patch/ttft_combo`**：Issue 3 TTFT 组合优化统一补丁包：
   - Cat-Free 8-Part MLP 流水线（RS + in-place AG）
   - hipBLASLt 10 形状白名单派发（支持 64K 超长上下文）
   - Fused GDN Prepare 4-warp 优化与状态元数据预计算
   - Paged KV 打包加速
   - BF16 QK Normalization 与 Attention 门控融合折叠
   - Causal Conv1d 发射调优（BLOCK_M=64, BLOCK_N=512）
   - Token-Tiled mRoPE 向量化连续访存算子
   - NUMA node 4 硬绑定与 8-channel RCCL 拓扑调优

## 一键构建镜像

```bash
# 方式 1: 从现有 Issue 1 镜像构建最终 Release 镜像（推荐，耗时仅数秒）
bash patch/build_ttft_candidate.sh --from issue1

# 方式 2: 从海光官方原生镜像全量构建
bash patch/build_ttft_candidate.sh --from native

# 方式 3: 仅预检查看构建命令 (Dry Run)
bash patch/build_ttft_candidate.sh --dry-run
```
