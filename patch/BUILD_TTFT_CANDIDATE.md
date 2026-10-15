# BW100 TTFT 候选镜像构建链

`build_ttft_candidate.sh` 从 Issue 1 镜像或指定原生镜像按依赖顺序构建到 `current-gemm` / `prefill-release`。它只调用 Docker build；不启动推理服务、不运行 GPU 验证、不推送镜像、不修改 Git。每层显式设置 `BASE_IMAGE`，尤其补上各 Dockerfile 默认值没有表达的 KV→metadata→pipeline→current→GEMM 依赖。

```bash
# 本机只做 CPU 源码检查并打印完整构建命令，不需要 Docker daemon。
bash patch/build_ttft_candidate.sh --dry-run --from native --target prefill-release

# 在已同步项目的 BW100 服务器执行构建，从现有 Issue 1 开始。
bash patch/build_ttft_candidate.sh --from issue1 --target prefill-release

# 从指定原生镜像构建；后续输出用独立 suffix，避免混淆已有试验 tag。
bash patch/build_ttft_candidate.sh --from native --target current-gemm \
  --issue1-image sglang-bw100:gdn-kkt-solve-repro \
  --tag-suffix repro
```

默认原生输入为 `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`；可用 `--native-image` 指定已核验 digest。默认 Issue 1 输入为 `sglang-bw100:gdn-kkt-solve-issue1`。`--image-prefix` 改输出 repository，`--tag-suffix` 改后续 tag 后缀，`--issue1-image` 在 issue1 模式为输入、native 模式为该层输出。默认 tag 与现有试验 tag 相同，实际构建会更新这些 tag。脚本在任意工作目录可用，context 仅是相应 patch 子目录，不发送项目 data 目录。

## 精确层顺序与 override

以下 tag 省略共同前缀 `sglang-bw100:`。2026-10-15 在 BW100 只读核对 image inspect，表中相邻镜像的 RootFS layer 列表均为父层前缀；这是现存链的佐证，不保证重新构建产生同一 image ID。

| 输出 tag | Dockerfile/context | 显式 BASE_IMAGE | 当前短 ID |
| --- | --- | --- | --- |
| gdn-kkt-solve-issue1（可选） | gdn_kkt_solve | 指定原生镜像 | a8df71ccac44 |
| ttft-fa-issue3 | ttft_prefill/Dockerfile | gdn-kkt-solve-issue1 | 4171296e18f8 |
| ttft-packed-issue3 | ttft_prefill/Dockerfile.packed | ttft-fa-issue3 | f899cde4c0ae |
| ttft-gdn-views-issue3 | ttft_gdn_views | ttft-packed-issue3 | 260deea16218 |
| ttft-gdn-prepare-w4-issue3 | ttft_gdn_prepare | ttft-gdn-views-issue3 | 7c7b02cffabe |
| ttft-kv-pack-issue3 | ttft_kv_pack | ttft-gdn-prepare-w4-issue3 | d673daa3d05f |
| ttft-metadata-pack-issue3 | ttft_gdn_metadata | **ttft-kv-pack-issue3** | 69e1a34f2d45 |
| ttft-mlp-pipeline-issue3 | ttft_mlp_pipeline | **ttft-metadata-pack-issue3** | de17bea2fde8 |
| ttft-mlp-current-stream-issue3 | ttft_mlp_pipeline/current_stream | ttft-mlp-pipeline-issue3 | 7973cfb3af12 |
| ttft-current-gemm-issue3 | ttft_gemm_dispatch | **ttft-mlp-current-stream-issue3** | 7f1ad09c9bcd |
| ttft-prefill-release-issue3（可选终点） | ttft_prefill_release | ttft-current-gemm-issue3 | 58b0ce4a9324 |

`ttft_gdn_metadata/Dockerfile`、`ttft_mlp_pipeline/Dockerfile` 默认仍是 prepare-w4；直接单独 build 会丢失前面附加的层。`ttft_gemm_dispatch/Dockerfile` 默认是 metadata-pack；这里必须 override 到 current-stream，才能保留正确的 MLP 路径。所有实际 Dockerfile 均支持 `ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`，脚本在 CPU preflight 再次检查。

## w4、current stream 和白名单

当前 `ttft_gdn_prepare/gdn_prefill_prepare.py` 的 `fused_gdn_prefill_prepare` 默认 `num_warps=4`，launch `num_stages=3`；backend 调用不覆盖 warps。脚本核验这些条件，直接使用当前源码构建 w4 tag，不需要重建历史 w8 prepare 镜像再覆盖 helper。这复现当前 w4 源码和配置，不声称复现历史镜像的逐层字节。也不加入 `ttft_gdn_tuning` 的其它 launch 候选。

`ttft-mlp-pipeline-issue3` 是失败的 private-stream **中间依赖**，不是最终优化镜像。后面的 current-stream 安装器先严格验证该旧 helper SHA `d4f525cf184bf9472f3f54a750d2e5bc5616a8423da3f2bfe4ed35a5d1b1658a`，保存失败源为备份，然后替换实际运行 helper。因此构建链保留旧层仅为安装器依赖，任何失败都使脚本停止。最终使用 current-compute-stream 四子批；不加入 `current_stream_parts` 或八子批试验。

`current-gemm` 镜像的 `Q38_HIPBLASLT_SHAPES` 默认为空，启动时才可显式设置已验证的 9 个形状。`prefill-release` 最后一层固化以下已测 9-shape 白名单，同时设置 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1`：

```text
4096x8192x5120,8192x17408x5120,8192x8192x5120,8192x5120x3072,16384x17408x5120,16384x8192x5120,16384x5120x3072,32768x8192x5120,32768x5120x3072
```

该构建链不加入 M65536 新形状，也不把镜像名的 release 解释成完整精度或性能验收。32K chunk、模型路径、GPU4/5、TP2、禁用 piecewise graph 等仍由现有启动脚本参数控制，不在构建脚本中暗改。两种终点均保留父镜像的 sleep entrypoint；服务由项目 scripts 启动。

## 本次检查范围

已完成 Bash 语法、native→prefill-release 与 issue1→current-gemm 两条 dry-run、逐层命令/父 tag 核对、w4 默认配置及失败 helper SHA 依赖检查。只读查询了 BW100 现存 tag、layer 前缀和最终环境变量。没有实际执行此次 build，也没有 GPU 或服务操作；镜像构建是否通过仍由各层安装器的实际 SHA guard 检验。

## 64K/8子批与官方QK融合候选

新增 `--target qk-gate`：在prefill-release之后依次构建 `current_stream_parts`、`ttft_qk_norm_gate`，两个父镜像均显式override，最终tag为 `ttft-qk-gate-<suffix>`。可运行 `bash patch/build_ttft_candidate.sh --dry-run --from native --target qk-gate`。服务另行显式设置 `CHUNKED_PREFILL_SIZE=65536 MAX_PREFILL_TOKENS=65536 SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8 MEM_FRACTION_STATIC=0.80` 和GPU4/5；构建不会偷偷更改这些服务参数。生产helper默认仍4子批。

该组合已构建为 `sglang-bw100:ttft-qk-gate-64k-v2-issue3`。v1在启动时发现Python后缀模型备份被自动扫描，v2将备份改为 `.py.ttft_qk_original`，保留失败记录；v1没有性能结果。64K/8真实MLP边界与QK融合各自全量逐值检查通过，但组合模型结果独立验收。
