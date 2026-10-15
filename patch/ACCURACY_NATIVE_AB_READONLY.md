# 原生 OpenCompass 64 样本 A/B：准备状态与实际证据

**评测客户端已准备完成，尚未由本分工执行模型评测。** BW100 sleep 容器中的 `/bench/.venv-accuracy` 已安装固定源码的 OpenCompass 0.5.4 和兼容 CPU 依赖；官方评分器、原生 reader、CLI、两份配置及本地 tokenizer 检查通过。最终状态记录于 `result/ttft/accuracy-setup/preparation-status.json`：`READY_NATIVE_API_64_SUBSET`。准备结束时没有 pip/build 进程，没有模型请求或 GPU kernel 执行，系统 GPU 栈不变。

完整启动说明见 [ttft_accuracy_client/README.md](ttft_accuracy_client/README.md)。不要沿用早期失败的 git+file 或隔离 setup_requires 方案。

## 已验证配置与可运行命令

双方设置均为 `reader_cfg.test_range='[:64]'`、thinking=false、concurrency=32、max_out_len=512、temperature=0、seed=20261012；本地 tokenizer `/model`，服务地址 `http://127.0.0.1:18580/v1`，模型 alias `Qwen3.8-27B-Channel-INT8-w8a8`。两份配置已生成并经 MMEngine 原生解析，Qwen2TokenizerFast 在禁止 socket 连接时成功编码。

由主 agent 确认待测服务身份并调度，不能混用原生与优化目录：

```bash
cd /bench
bash patch/ttft_accuracy_client/opencompass_api.sh \
  result/accuracy-native/gsm8k64-native/configs/gsm8k.py \
  --work-dir result/accuracy-native/gsm8k64-native/opencompass/gsm8k \
  --reuse native --mode all

# 仅在优化服务处于待测状态时运行。
bash patch/ttft_accuracy_client/opencompass_api.sh \
  result/accuracy-native/gsm8k64-optimized/configs/gsm8k.py \
  --work-dir result/accuracy-native/gsm8k64-optimized/opencompass/gsm8k \
  --reuse native --mode all
```

启动器自带 CPU0–3、nice19、OMP1、MAX_JOBS2、CMAKE2、GPU不可见、HF offline、本机 NO_PROXY 和本地无鉴权接口所需 EMPTY key 默认值；原样调用官方 OpenCompass CLI。只有客户端进程通过明确命名的 finder 禁用可选 `vllm` / `vllm_hcu`，使上游已有 ImportError 分支生效；子进程继承已验证。不 mock API/评分器、不修改 OpenCompass 或 GSM8K 源码，不改变系统 sys.path/site-packages。

64 题是官方 test 文件的固定前 64 行，不是完整 GSM8K 验收。结果分别在各自 `opencompass/gsm8k/native/predictions/q38/gsm8k.json` 与 `results/q38/gsm8k.json`。现有项目 `audit_oc` 强制全部 1319 题；小样本必须与其隔离，不能伪称完整准入 PASS。追加范围后不要调用 wrapper `run`，它会重写配置；使用上述原生 CLI。

## 实际安装和验证

- 源码 commit：`1543bd270b6fbabb1a118307f78d4144e8296467`。保全到 `/bench/result/ttft/accuracy-setup/source`，不含 `.git`；3627 个 tracked 文件经原 Git blob ID 核对。完整 SHA256 清单和 provenance 已保存。
- 在独立 venv 中构建官方 wheel，3043 个打包 Python 文件全部匹配原源码，安装后再次匹配。wheel SHA256：`ceabe9640de63be2bf60a1ddaf1b9a63d46828e83b284b4a371c4d2c3c1c321c`。上游打包自然省略 4 个非 GSM8K 文件，清单在 `oc-wheel-verification.json`。
- 官方 Gsm8kEvaluator 对 `[42,7,-3]` / `[42,8,-3]` 返回 66.6666666667%，逐题 `[true,false,true]`；这是已知答案 CPU sanity，不是模型精度结果。
- 原生 reader `load_partial_dataset(...,'[:64]')` 选择原始索引 0…63；字符串范围使用上游确定切片，而不是整数抽样。
- `api-client-validation.json` 记录评分器、原生 reader、源码 SHA、两处 ImportError catch、子进程继承、GPU 栈与 CPU 限制。guard SHA256 为 `05b68d5a42fa846f7a26997d27cac1c750401d0da6f3838efe7108b255bbc508`。
- `native64-config-validation.json`、`cli-help.log`、`config-generation.log` 记录配置与入口验证；本分工未执行 `--mode all` 请求模型。

最终 CPU 包版本：transformers 4.51.3、sentence-transformers 3.4.1、tokenizers 0.21.4、datasets 3.6.0、huggingface_hub 0.36.2、mmengine-lite 0.11.0rc3、nltk 3.8、setuptools 80.10.2。完整独立环境记录为 `venv-freeze.txt`。系统 torch/torchvision/torchaudio/triton 的原始版本和路径在 `vendor-stack-before.json`，最终验证保持不变。

经主 agent 授权，将上游未锁定 Git master 的 rouge_chinese 依赖替换为明确的 PyPI `rouge-chinese==1.0.3`；wheel SHA256 `afda22cb64f9a0f1d7a19e88d7cd9acfd2c1f5d45bf463ade7ecadf87f30624d` 与 PyPI 元数据一致。GSM8K 仍执行官方 evaluator，不调用中文 ROUGE。

纯目录构建后 pip direct_url 如实指向本地 wheel，不伪造 VCS metadata。原项目完整 runner 的 VCS 来源断言不能因此称为通过；本次用官方原生命令，并以 Git 来源保全、文件 SHA 和 wheel SHA 单独证明源码身份。

## 数据与环境边界

远端已有 `/data/tools/q38-bw100-sglang/data/accuracy/gsm8k/test.jsonl`（1319 行，SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`）及 `train.jsonl`（7473 行，SHA256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`）。没有同步、下载或修改 data。

容器项目挂载 `/bench`、模型只读挂载 `/model`。venv 使用 system-site-packages，仅供相同镜像的客户端使用，不应在宿主直接执行。当前客户端需持续存活，不能随待测服务切换而中断评测进程。

## 准备过程和失败路径（历史，不是当前操作指令）

1. 镜像缺少 ensurepip，标准 venv 创建失败；使用现有 virtualenv 21.7.4，以 `--system-site-packages --no-download` 建立 `/bench/.venv-accuracy`，没有 apt 或系统安装。
2. 初次 PyPI dry-run 停滞于隔离 `nltk==3.8`；清华重试又遇到 GitHub GnuTLS -110。本地 partial/shallow git+file 复用触发全历史打包及 promisor 递归子进程，已立即终止精确进程组；主 agent 随后重启这个 sleep 容器清理僵尸进程。该路线不再使用。
3. 先保全纯源码目录，再在 CPU0–3/nice19、有限线程、有界超时条件下安装 CPU 构建依赖。完整 runtime plan 确认无新增 GPU 栈，44 个待安装 artifact 逐项记录 HTTPS URL 和 SHA256，并用 `--no-deps --no-build-isolation` 安装已解析集合。
4. 继承系统 HF kernels 的 setuptools 插件与旧 hub 冲突：用只包含 venv site 路径的 CPU build backend 构建 wheel，未改源。Transformers 4.57 聚合导入同一冲突，进一步固定兼容的 4.51.3 / sentence-transformers 3.4.1 / tokenizers 0.21.4，先检查仅三项 CPU 包的计划再安装。
5. OpenCompass 聚合导入系统 vendor vLLM 时试图探测 GPU，被 GPU 不可见环境阻止；经授权使用本目录 API-only finder，触发官方已有 ImportError fallback。实际评测模块不被屏蔽或替换。

准备日志、起止时间、CPU策略、依赖 plan/来源差异、freeze 及最终 PASS 证据均在 `result/ttft/accuracy-setup/`。上述准备已完成，最后状态不是“仍未安装”或“未执行 resolve”。

## 官方来源

- [DatasetReader / load_partial_dataset](https://github.com/open-compass/opencompass/blob/1543bd270b6fbabb1a118307f78d4144e8296467/opencompass/openicl/icl_dataset_reader.py#L199)：原生确定范围。
- [GSM8K 配置](https://github.com/open-compass/opencompass/blob/1543bd270b6fbabb1a118307f78d4144e8296467/opencompass/configs/datasets/gsm8k/gsm8k_gen_1d7fe4.py)：沿用 prompt、postprocess 和 evaluator。
- [setup.py](https://github.com/open-compass/opencompass/blob/1543bd270b6fbabb1a118307f78d4144e8296467/setup.py)、[runtime 依赖](https://github.com/open-compass/opencompass/blob/1543bd270b6fbabb1a118307f78d4144e8296467/requirements/runtime.txt)：安装范围和依赖记录。
