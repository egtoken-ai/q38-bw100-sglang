# OpenCompass / VLMEvalKit 原生配置与命令评测

新增入口：`bash scripts/run_accuracy_native.sh`，或 `bash scripts/run.sh accuracy-native`。旧版 `run_accuracy.sh` 保留。

这个版本**真正启动上游 CLI 子进程**，提示词渲染、模型请求、并发重试、原始预测保存、断点恢复和评分都由原生框架负责。外层 Python 只负责数据准备、配置生成、命令启动、结果完整性检查及阈值报告，不调用旧版的 `generate`、`run_one`、`score_benchmark`。

| 基准 | 原生执行方式 | 固定协议 |
|---|---|---|
| GSM8K | `opencompass <配置.py> --mode all` | 官方 test 1319 题；`gsm8k_gen_1d7fe4`，4-shot CoT，`Gsm8kEvaluator` |
| LiveCodeBench v6 | `opencompass <配置.py> --mode all` | 仅 `test6.jsonl` 175 题；`n=1`；原生 `LCBCodeGenerationEvaluator`，`pass@1` |
| RealWorldQA | `python <上游run.py> --config <配置.json>` | 标准 TSV 765 题；原生 `ImageMCQDataset`，`exact_matching` |
| MathVision Without CI | `python <上游run.py> --config <配置.json>` | `MathVision_MINI` / testmini 304 题；原生 `MathVision.evaluate` 和 LLM Judge；不提供 Code Interpreter |

源码配置模板位于 `config/accuracy_native/`，生成后的具体配置保存在输出目录的 `configs/`。视觉部分使用 OpenCompass 团队的独立工具 VLMEvalKit，不宣称全部四项都由 OpenCompass 文本 CLI 执行。

## 环境准备

使用独立的 **Linux x86_64 / Python 3.10** 评测环境，复用已有 `requirements-accuracy.txt`，无需修改海光 SGLang 服务环境：

```bash
python3.10 -m venv .venv-accuracy
.venv-accuracy/bin/python -m pip install --upgrade pip 'setuptools<81' wheel
.venv-accuracy/bin/python -m pip install -r requirements-accuracy.txt
```

脚本默认选择 `.venv-accuracy/bin/python`，也支持 `ACCURACY_PYTHON=/path/to/python`。OpenCompass 使用该解释器同目录的 `opencompass` 可执行文件。运行时检查两个框架的安装 commit，要求与依赖文件一致。

## 下载数据与原生入口

```bash
bash scripts/run_accuracy_native.sh prepare
# 缓存齐全后可离线验证，不再下载
bash scripts/run_accuracy_native.sh prepare --offline
```

沿用已有的 `data/accuracy/` 缓存，不重复下载四个测试集。原生 GSM8K 加载器还要求 `train.jsonl`，因此首次增加下载官方 7473 条训练数据；4-shot 示例仍来自上游固定提示词，并不从测试集中抽取。

VLMEvalKit 的 pip 包未包含仓库根目录 `run.py`，脚本自动下载固定 commit 的**未修改原始入口文件**到 `data/accuracy/native_tools/`，并验证固定 SHA256，不需要额外克隆整个仓库。

LiveCodeBench 的原生 Evaluator 会自行重新加载数据，因此需要同时约束推理和评分的来源。脚本将一个仅暴露 `v6/test6.jsonl` 的 Hugging Face 本地数据适配文件放入 `native_oc/data/code_generation_lite/`，通过原生 `COMPASS_DATA_CACHE` 路径机制让两阶段使用相同的 175 题。该适配文件不实现提示词或评分；测试用例解码、代码提取及执行均由上游负责。不能删除或替换它后仍宣称评测子集相同。

视觉数据站证书过期时，可使用有效镜像 `--vlm-data-base-url https://mirror/VLMEval`，或显式使用 `prepare --insecure-dataset-download`；行为与旧版下载器一致，只影响经固定校验和验证的两个视觉 TSV，不影响模型 API 的 TLS。

## 完整运行

先启动支持图像输入的 SGLang 服务，再配置 MathVision 使用的固定 Judge：

```bash
export ACCURACY_API_KEY='EMPTY'               # 被测服务无鉴权时可省略
export ACCURACY_JUDGE_API_KEY='your-judge-key' # Judge 无鉴权时可省略

bash scripts/run_accuracy_native.sh run \
  --base-url http://127.0.0.1:18580/v1 \
  --model-revision q38-int8-v1 \
  --judge-base-url http://judge-host:8000/v1 \
  --judge-model fixed-judge \
  --output-dir result/accuracy-native/int8-v1
```

Judge 参数也可以通过 `ACCURACY_JUDGE_BASE_URL`、`ACCURACY_JUDGE_MODEL` 设置。`--judge-model` 请使用服务端不含 `/` 的别名，因为上游会将模型名写入中间文件名。原生 MathVision 会将题干和模型输出发给指定 Judge 提取答案；Judge 不向被测模型反馈答案。

**没有 Judge 配置时**，完整运行会继续其他三项，并将 MathVision 标记为 `ERROR`；不会自动调用公共付费服务，也不会改用本地评分规则。可先执行 `infer` 保存预测，再补齐 Judge 执行 `eval`。

GSM8K 默认保留上游 `max_out_len=512`；其余基准默认 32768。长推理模型可显式设置 `--gsm8k-max-tokens 32768`。所有参数都保存在配置中，改变 token 上限可能改变精度，不能把不同配置混作同一次对比。默认温度 0、top_p 1、并发 4；thinking 设置沿用服务端，可用 `--extra-body '{"chat_template_kwargs":{"enable_thinking":true}}'` 显式指定。

OpenCompass 使用原生 `OpenAISDK`，配置原生 `extract_non_reasoning_content` 后处理，再执行数据集自己的答案提取。默认 `gpt-4` tokenizer 仅用于该 API 适配器的长度估计，不会调用 GPT-4；可通过 `--tokenizer-path /path/to/tokenizer` 指定被测模型的本地 tokenizer。第一次使用 tokenizer 可能需要下载资源；`--offline` 只约束本脚本的数据/入口下载。

## 分阶段与原生命令

只生成具体配置和命令，无需安装评测依赖，也不发出推理请求：

```bash
bash scripts/run_accuracy_native.sh config \
  --model-revision q38-int8-v1 \
  --judge-base-url http://judge-host:8000/v1 --judge-model fixed-judge \
  --output-dir result/accuracy-native/int8-v1
```

得到 `configs/*.py`、`configs/*.json`、`commands.json` 和包含必要环境变量的 **`commands.sh`**。完成 `prepare` 后，可直接执行 `bash result/accuracy-native/int8-v1/commands.sh`。命令文件使用环境变量读取 Key，不嵌入 Key 值。

文本部分实际执行的命令形如：

```bash
opencompass result/accuracy-native/int8-v1/configs/gsm8k.py \
  --work-dir result/accuracy-native/int8-v1/opencompass/gsm8k \
  --reuse native --mode all
```

视觉部分实际执行固定版本的原始 `run.py --config ... --mode infer`，再执行 `--mode eval --reuse`。这使被测服务与 Judge 可以使用不同 Key；两阶段仍全部由 VLMEvalKit 完成。

也可以由统一入口分阶段调度：

```bash
# 只跑推理，可以暂不设置 Judge
bash scripts/run_accuracy_native.sh infer --output-dir result/accuracy-native/staged
# 补齐 Judge，使用原生缓存进行评分
bash scripts/run_accuracy_native.sh eval \
  --judge-base-url http://judge-host:8000/v1 --judge-model fixed-judge \
  --output-dir result/accuracy-native/staged
# 单独重建准入报告
bash scripts/run_accuracy_native.sh report --output-dir result/accuracy-native/staged
```

分阶段命令必须保持同样的模型、数据路径和生成参数。允许从未配置 Judge 补充为指定 Judge；更换已经指定的 Judge、模型权重标识或生成参数要求使用新输出目录。改变服务端量化、权重或启动配置时，应更换 `--model-revision` 和输出目录。

重复 `run` 自动使用 OpenCompass `--reuse native` 和 VLMEvalKit `--reuse`；具体任务/样本的恢复规则由上游实现决定。`eval` 不重新请求被测模型；原生结果已经存在时，上游可能直接复用结果，不保证强制重新打分。

支持 `--benchmarks gsm8k livecodebench` 等单项选择；在同一配置目录可分批补齐四项。没有额外的自定义小样本测试模式，防止将试跑结果当作完整准入结果。

## 报告与限制

原生输出保留在 `opencompass/<benchmark>/native/` 和 `vlmeval/<benchmark>/q38/T*/`，包括配置快照、预测、原始评分、中间 Judge 结果和日志。顶层另外生成：

- `summary.md`：基线、阈值、实际分数、降幅、题数、PASS/FAIL/ERROR。
- `summary.csv`：UTF-8 BOM，便于 Excel 和程序读写。
- `summary.json`：未舍入数值、状态及原生评分文件来源。
- `suite_config.json`、`commands.json`、`commands.sh`、`environment.txt`：复现信息。

汇总检查原生预测的完整数量/ID、空输出、API 失败标记、评分状态，以及 MathVision Judge 是否失败；不只检查进程退出码。RealWorldQA 原生分数为 0–1，汇总换算成百分数；另外三项原生分数已经是百分数。仅四项完整且达到 95.0 / 88.3 / 83.9 / 88.0 才整体通过。原生 API 适配器没有在预测文件完整保留 HTTP finish_reason，因此本版本**不能像旧版自定义 HTTP 客户端那样严格识别所有 token 截断**。

`prepare`、`config`、成功的 `infer` 返回 0；完整评分四项通过返回 0，四项完成但分数不达标返回 1，缺项/运行错误返回 2，中断返回 130。`infer` 成功不代表准入通过，报告仍为 `INCOMPLETE`。失败后应重新执行相应 `run`/`eval` 修复；`report` 不会把记录过的失败任务仅凭旧文件改为通过。

使用原生命令提高了复现便利性，但仍需与主办方确认提示词、thinking、生成参数和 Judge 口径。LiveCodeBench 会执行生成的 Python，建议沿用独立隔离评测环境。

上游依据：[OpenCompass 配置与完整评测](https://opencompass.readthedocs.io/en/latest/user_guides/config_based_evaluation.html)、[VLMEvalKit 配置系统](https://github.com/open-compass/VLMEvalKit/blob/21e52b8df8ce1f628e93c82de547edb2c985f6ac/docs/en/ConfigSystem.md)。

## 本次验证范围

`python3 -m unittest discover -s tests -p 'test_accuracy*.py' -v` 的 17 项测试通过，包含原生命令生成、失败后继续其他基准、分数单位换算、缺题/失败 Judge 拦截和配置复用检查。另在独立 Python 3.10 环境中使用真实 `mmengine-lite==0.11.0rc3` 解析了两个文本配置并保存/重新加载快照；使用真实 `datasets==3.6.0` 加载本地 LCB builder，验证得到 175 个唯一 question_id。四个测试集、GSM8K 训练文件与固定版本 VLMEvalKit 原始入口已完成准备。

这些检查不等于执行了完整原生 CLI 推理/评分。当前机器没有运行中的被测 SGLang/Judge 服务，也没有安装整套 Linux 评测依赖，因此尚无真实模型精度成绩。
