# 原生 OpenCompass API 客户端：固定 GSM8K 64 样本 A/B

评测环境已准备在 BW100 的 `/bench/.venv-accuracy`，复用镜像系统 torch/torchvision/torchaudio/triton，未改 GPU 栈或服务。已通过 CPU 源码、官方评分器、原生 reader、CLI 和配置检查；随后经明确授权完成一次 optimized GSM8K64 原生模型评测，官方 61/64（95.3125%），详见下方。64 题子集只能作为相同样本回归，不能替代 GSM8K 1319 题或完整四项精度验收。

## 现成运行命令（由主 agent 调度）

在项目挂载为 `/bench`、模型挂载为 `/model` 的持续存活客户端容器内执行。先确保目标原生服务已由主 agent 启动且验证身份，再运行第一条；切换并核实优化服务后运行第二条。两边使用同一 API 地址 `http://127.0.0.1:18580/v1`、模型 alias、原始 test 文件前 64 行、thinking=false、concurrency=32、max output=512、temperature=0、seed=20261012。

```bash
cd /bench
bash patch/ttft_accuracy_client/opencompass_api.sh \
  result/accuracy-native/gsm8k64-native/configs/gsm8k.py \
  --work-dir result/accuracy-native/gsm8k64-native/opencompass/gsm8k \
  --reuse native --mode all

# 由主 agent 调度，在优化服务待测时执行；不得与上条混用预测目录。
bash patch/ttft_accuracy_client/opencompass_api.sh \
  result/accuracy-native/gsm8k64-optimized/configs/gsm8k.py \
  --work-dir result/accuracy-native/gsm8k64-optimized/opencompass/gsm8k \
  --reuse native --mode all
```

启动器自动设置 CPU0–3、nice19、OMP1、MAX_JOBS2、CMAKE_BUILD_PARALLEL_LEVEL2，隐藏 HIP/CUDA/ROCR GPU，设置 HF Hub/Datasets offline、本机 NO_PROXY，并在没有 API key 时使用本地无鉴权接口占位值 EMPTY。现有配置指定本地 `/model` tokenizer；已验证 Qwen2TokenizerFast 能在禁止 socket 连接时编码。本启动器原样调用官方 `.venv-accuracy/bin/opencompass`，没有自制请求循环、评分或答案后处理。

`ACCURACY_VENV`、`ACCURACY_CPU_AFFINITY` 可用于明确变更客户端路径/CPU；当前验收证据对应默认 CPU0–3。不要在宿主直接运行依赖容器 system-site-packages 的 venv。客户端容器停止会终止其评测进程，因此不要把客户端和被切换服务的生命周期混在一起。

## 隔离范围

`client_env.sh` 只作用于启动器及子进程，通过进程环境 PYTHONPATH 加载 `import_guard/sitecustomize.py`。启用开关是 `Q38_OC_API_ONLY=1`，finder **仅对 vllm、vllm_hcu 及其子模块抛出 ModuleNotFoundError**；其它模块原样查找，不提供 mock 模块。

已用 AST 核实固定 OpenCompass 的 `models/vllm.py` 和 `models/vllm_with_tf_above_v4_33.py` 原本就有 `except ImportError` 回退。该隔离使未使用的本地 GPU 后端处于官方“未安装”分支，避免镜像 vendor vLLM 在 import 时探测 GPU。OpenAISDK、Gsm8kEvaluator、prompt、postprocess 和数据选择代码都没有修改；不是给失败评分制造成功结果。没有全局 sitecustomize 或系统安装修改，子进程继承已实测。

## 环境与证据

- 固定 OpenCompass commit：`1543bd270b6fbabb1a118307f78d4144e8296467`，package 0.5.4。从官方 checkout 保全不含 `.git` 的源码；wheel 中 3043 个 Python 文件逐个 SHA256 匹配该源码，安装后再次匹配。
- wheel SHA256：`ceabe9640de63be2bf60a1ddaf1b9a63d46828e83b284b4a371c4d2c3c1c321c`。官方 wheel 自然省略 4 个非 GSM8K 文件，清单保存在证据中。
- CPU 客户端兼容版本：transformers 4.51.3、sentence-transformers 3.4.1、tokenizers 0.21.4、datasets 3.6.0、huggingface_hub 0.36.2、mmengine-lite 0.11.0rc3、nltk 3.8、setuptools 80.10.2。
- 经主 agent 授权，仅把上游未锁定 Git master 的 rouge_chinese 依赖替换为 PyPI `rouge-chinese==1.0.3`。wheel SHA256 `afda22cb64f9a0f1d7a19e88d7cd9acfd2c1f5d45bf463ade7ecadf87f30624d` 与 PyPI 元数据一致；GSM8K 不调用中文 ROUGE。
- 本地目录构建避免了系统 HF kernels 的 setuptools 插件冲突。源码未改；pip direct_url 如实指向本地 wheel，不伪造 VCS 元数据。因此原项目全量编排器的严格 VCS 来源检查不能直接视为通过；这里使用官方 CLI 和独立来源校验。
- `result/ttft/accuracy-setup/api-client-validation.json`：官方 Gsm8kEvaluator 对 `[42,7,-3]` / `[42,8,-3]` 得到 `66.6666666667%`，逐题 `[true,false,true]`，证明真实评分代码执行；原生 `load_partial_dataset(...,'[:64]')` 精确选择 0…63。
- `native64-config-validation.json`：双方配置相同、32 并发、512 输出、thinking=false，且本地 tokenizer 检查通过。`cli-help.log` 记录原生 CLI 可运行。
- `vendor-stack-before.json`、`runtime-plan-check.json`、`oc-wheel-verification.json`、`directory-source-provenance.json`、`source-file-sha256.json`、CPU 亲和策略、安装日志和 freeze 保存完整准备边界。安装前 plan 拒绝新增 GPU 栈包，实际 runtime 按已解析 HTTPS artifact+SHA 安装。

## 只生成配置或重复 CPU 检查

`prepare_native64.sh` 只调用现有 `run_accuracy_native.sh config`，再追加官方 `reader_cfg.test_range='[:64]'`；不下载/准备 data，不请求模型。若目标目录已有 OpenCompass run，脚本拒绝覆盖，应选新的 `ACCURACY_OUTPUT_ROOT`。不要再对这两份配置调用 wrapper `run`，它会重新生成配置并清除子集字段。

```bash
# 仅 fresh output-root 时使用。
taskset -c 0-3 nice -n 19 bash patch/ttft_accuracy_client/prepare_native64.sh

# CPU-only sanity，子进程继承隔离；不请求模型。
source patch/ttft_accuracy_client/client_env.sh
taskset -c 0-3 nice -n 19 .venv-accuracy/bin/python \
  patch/ttft_accuracy_client/verify_client.py --evidence-dir result/ttft/accuracy-setup

taskset -c 0-3 nice -n 19 .venv-accuracy/bin/python \
  patch/ttft_accuracy_client/verify_configs.py \
  --output-root result/accuracy-native --evidence-dir result/ttft/accuracy-setup
```

模型运行后，结果位于各自 `opencompass/gsm8k/native/predictions/q38/gsm8k.json` 与 `results/q38/gsm8k.json`。必须核对双方完整 64 题、错误/空响应及逐题对应。现有 `audit_oc` 强制 1319 题，因此子集结果与全量审计隔离，不把子集分数记作完整准入通过。

## 已执行的优化服务子集结果

2026-10-15 对 `sglang-bw100:ttft-postops-issue3` 执行一次原生 CLI，19:35:28–19:36:50（82 秒），退出 0；64 条完整预测，官方正确 61/64，accuracy 95.3125%，错误原始索引 12/37/46。空输出和 API 失败标记均为 0。索引 12/37 的原始输出经同一 tokenizer 重编码达到512 token，存在触顶截断迹象；native artifact 不保留 finish_reason/usage，不能把“无明确 finish_reason”写成“无截断”。没有提高上限或重跑。原生服务 A/B 待主 agent 调度，64子集不是完整验收。

完整证据保存在 `result/accuracy-native/gsm8k64-optimized/`，摘要 `execution-evidence/summary.md`、官方 predictions/results、部署和始末日志均已同步本地。环境准备阶段的 `model_requests_sent=false` 是19:27准备完成时的历史边界，不适用于本次另行授权的模型评测。
