# Qwen3.8 BW100 统一优化补丁 (Release)

本目录收敛了海光 BW100 双卡平台针对 Qwen3.8-27B-Channel-INT8 模型经压测验证的全部有效核心优化。所有优化代码直接覆盖安装至 SGLang 官方原生镜像，无需任何中间多阶段过渡镜像。

## 目录结构

```text
patch/
├── Dockerfile              # 优化镜像构建文件 (基于海光官方原生镜像一步构建)
├── README.md               # 补丁与构建说明文档
├── build_ttft_candidate.sh # 镜像自动化构建辅助脚本
├── start_server.sh         # NUMA 绑定与优化参数服务启动脚本
└── sglang/                 # 核心修改的 SGLang 源码文件
    └── srt/
        ├── models/
        │   └── qwen3_5.py                # Attention 门控 Sigmoid 融合与 BF16 QK Norm 图折叠
        └── layers/
            ├── qwen27b_mlp_pipeline.py   # Cat-Free 8-Part MLP 流水线 (RS + in-place AG)
            ├── qwen27b_mlp_probe.py      # 流水线激活探针与调度保护
            ├── attention/
            │   ├── pack_paged_kv_to_varlen.py  # Paged KV 离散重排打包
            │   ├── mamba/
            │   │   ├── causal_conv1d_triton.py # Causal Conv1d Triton 调度网格调优 (64x512)
            │   │   └── causal_conv1d_ttft.py
            │   ├── linear/
            │   │   ├── gdn_backend.py          # GDN 4-warp 融合 Prepare 与批元数据预计算
            │   │   └── kernels/gdn_triton.py
            │   └── fla/
            │       ├── chunk_fwd.py            # GDN KKT 融合求解内核 (消除逆缓冲区冗余初始化)
            │       ├── chunk.py                # GDN Chunk 状态计算优化
            │       ├── chunk_delta_h.py
            │       ├── chunk_o.py
            │       └── layernorm_gated.py
            ├── quantization/compressed_tensors/schemes/
            │   ├── compressed_tensors_w8a8_int8.py    # hipBLASLt 10 形状固化派发白名单 (含 64K)
            │   └── q38_bw100_legacy_gemm_dispatch.py
            └── rotary_embedding/
                ├── triton_kernels.py                  # Token-Tiled 向量化连续访存 mRoPE
                └── mrope_tiled.py
```

## 一键构建镜像

```bash
# 方式 1: 直接使用 Docker 命令构建
docker build -t sglang-bw100:ttft-combo-issue3 patch/

# 方式 2: 使用辅助构建脚本
bash patch/build_ttft_candidate.sh

# 预览构建命令 (Dry Run)
bash patch/build_ttft_candidate.sh --dry-run
```
