# 精度准入验证

入口为 `bash scripts/run_accuracy.sh {prepare|run|report}`，也可以使用 `bash scripts/run.sh accuracy ...`。评测客户端通过 HTTP 访问已有的 SGLang 服务，默认 `http://127.0.0.1:18580/v1`，模型名与 `start_server.sh` 一致。

脚本使用 **OpenCompass 和 OpenCompass 团队的 VLMEvalKit Python API**：沿用上游提示词、代码执行与评分实现，共用本项目的请求、数据缓存、逐题续跑和报告逻辑。它不调用 `opencompass run.py`，也不依赖性能压测脚本。

## 评测口径

| Benchmark | 数据集 | 生成/评分方法 | 基线 % | 阈值 % |
|---|---|---|---:|---:|
| GSM8K | 官方 test，1319 题 | OpenCompass `gsm8k_gen_1d7fe4`，**4-shot + CoT**，数值答案匹配 | 97.0 | 95.0 |
| LiveCodeBench v6 | **仅 `test6.jsonl`，175 题** | 上游 zero-shot 提示词，Python，单次生成 `pass@1`，运行全部公开和私有测试用例 | 90.3 | 88.3 |
| RealWorldQA | VLMEvalKit `RealWorldQA.tsv`，765 题 | zero-shot，图像和选择题提示词，`exact_matching` | 85.9 | 83.9 |
| MathVision (Without CI) | `MathVision_MINI.tsv`，对应 **testmini，304 题** | zero-shot CoT，图像输入，不提供 Code Interpreter；规则或独立 LLM 提取答案，VLMEvalKit 数学匹配 | 90.0 | 88.0 |

默认 `temperature=0`、`top_p=1`、每题生成一次、`max_tokens=32768`、并发 4。生成长度覆盖长推理模型；它覆盖了 GSM8K 上游示例配置的 512-token 限制。CoT 提示词与模型自身的 thinking 开关是不同设置；脚本默认沿用服务端 thinking 设置，可通过 `--extra-body` 明确指定。若服务器未分离 reasoning，脚本只对 `</think>` 后的最终文本评分，同时保留完整 API 响应。

所有阈值以**百分点**判断，使用未四舍五入的分数，四项全部完整通过才能得到整体 `PASS`。少跑、漏题、HTTP 错误、空答案、输出被 token 上限截断等均不能通过；脚本会保留错误并继续其他基准。模型答错属于有效回答，正常计为错题。

**可比性边界：**用户提供了基线分数与子集，但未提供基线的完整生成参数、thinking 设置、提示词及 Judge 配置。因此报告是上述明确协议下的阈值验证，不能据此宣称严格复现了官方基线。提交比赛时应使用主办方最终确认的配置。模型需支持图像输入；纯文本服务无法完成后两项。

## 安装评测环境

在 **Linux + Python 3.10** 的独立评测环境执行。客户端不需要 DCU/GPU，可以放在能访问服务的其他机器上；依赖包含通用 PyTorch 等组件，勿在海光 SGLang 服务环境中直接安装或替换它们。

```bash
python3.10 -m venv .venv-accuracy
.venv-accuracy/bin/python -m pip install --upgrade pip 'setuptools<81' wheel
.venv-accuracy/bin/python -m pip install -r requirements-accuracy.txt
```

依赖固定 OpenCompass / VLMEvalKit 的 Git commit，执行时检查安装来源，并在报告目录记录全部依赖版本。`pyext==0.7` 用于 LiveCodeBench 的代码测试；选用 Python 3.10 以兼容其旧版 API。若使用其他环境，设置 `ACCURACY_PYTHON=/path/to/python`。

LiveCodeBench 会在评测机运行模型生成的 Python 代码；请在无业务凭据的隔离容器/评测机运行，子进程超时机制并不是安全沙箱。MathVision 的 Without CI 指不给被测模型提供代码执行工具，与 LiveCodeBench 的答案判题执行不是同一件事。

## 下载与复用数据

仅下载无需安装 OpenCompass，也不需要启动模型服务：

```bash
python3 scripts/accuracy.py prepare
# 自定义共享缓存目录
python3 scripts/accuracy.py prepare --data-dir /dataset/accuracy
# 验证离线缓存，禁止下载
python3 scripts/accuracy.py prepare --data-dir /dataset/accuracy --offline
```

目录结构如下，可将已有的原始文件直接放在对应位置：

```text
data/accuracy/
├── manifest.json                    # 来源、SHA256、题数
├── gsm8k/test.jsonl
├── livecodebench/test6.jsonl
└── vlm/
    ├── RealWorldQA.tsv
    ├── MathVision_MINI.tsv
    └── images/                      # 首次推理时由 VLMEvalKit 解码并缓存
```

下载写入 `.part`，题数/字段/校验和检查通过后才成为正式缓存。文件已存在则直接校验复用，不重复下载；损坏文件报错，不自动覆盖。LiveCodeBench 固定官方 HF revision 和 SHA256，视觉 TSV 使用上游发布的 MD5。GSM8K 记录实际 SHA256。`--offline` 限制本脚本的数据下载，推理仍需访问所指定的模型 API。

开发时实测视觉数据站点存在 TLS 证书过期问题。如果仍遇到此问题，优先通过 `--vlm-data-base-url https://your-mirror/VLMEval` 使用保存相同 TSV 的有效 HTTPS 镜像，或复用本地数据。也可显式选择以下兼容下载方式：

```bash
python3 scripts/accuracy.py prepare --insecure-dataset-download
```

该选项仅跳过两个公开视觉 TSV 的 TLS 证书验证，文件仍必须匹配固定的上游 MD5；GSM8K、LiveCodeBench 以及所有模型/Judge API 连接继续验证 TLS。它不修改系统证书配置。localhost API 请求默认绕过系统代理；连接远程内网服务时，如有代理请自行设置 `NO_PROXY`。

本实现选择 VLMEvalKit 自带的 RealWorldQA.tsv 作为常规标准化版本，因此已有的其他格式需事先转换或取得该标准文件，不会静默更换数据版本。MathVision 是 **MathVision**，不是 MathVista；LiveCodeBench 不使用累计的 `release_v6` 或包含多轮采样的 academic 配置。

## 运行全部基准

先按主 README 启动 SGLang，再执行：

```bash
bash scripts/run_accuracy.sh run \
  --base-url http://127.0.0.1:18580/v1 \
  --data-dir /dataset/accuracy \
  --model-revision q38-int8-baseline-server-config-v1 \
  --output-dir result/accuracy/int8-v1
```

缺失数据会自动下载。相同命令再次运行时复用成功的逐题响应，只重试失败/截断的请求，再重新评分。改变数据、模型名、模型版本标识、生成参数、工具版本或脚本内容后必须换一个输出目录；同一目录有进程锁，不能并行写入。**服务地址/模型名不变而权重、量化或启动配置变更时，务必更换 `--model-revision` 和输出目录**，脚本无法仅凭 API 模型名识别这些变更。

API Key 从环境读取，不写入配置/报告：

```bash
export ACCURACY_API_KEY='your-server-key'  # 无鉴权可不设置
bash scripts/run_accuracy.sh run --output-dir result/accuracy/run-02
```

单项和冒烟测试：

```bash
# 先各跑 2 题，验证文本、图像与评分依赖
bash scripts/run_accuracy.sh run --limit 2 --output-dir result/accuracy/smoke
# 单跑 GSM8K 和 LiveCodeBench
bash scripts/run_accuracy.sh run --benchmarks gsm8k livecodebench \
  --output-dir result/accuracy/text-only
# 显式开启模型 thinking（服务需支持该字段）
bash scripts/run_accuracy.sh run \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}' \
  --output-dir result/accuracy/thinking
```

冒烟测试永远标记 `SMOKE`。只选择部分基准时，未运行项为 `MISSING`，整体为 `INCOMPLETE`，不会误报四项准入通过。`--max-tokens`、`--concurrency`、`--timeout`、`--eval-workers`、`--code-timeout` 可调整；完整参数见 `--help`。

允许在同一输出目录分批运行不同基准，前提是目录内的 `suite_config.json` 完全一致；已完成的其他基准可一起汇总。更换模型/协议后不能把旧目录结果混入新报告。

## MathVision 答案提取

默认 `--math-judge local` 无需额外 API：使用 balanced-brace `\\boxed{...}` / final-answer / 最后一行规则提取最终答案，再调用 VLMEvalKit `post_check` / `MATH_V_acc`；未能正确提取会计为错题。这是明确的本地评分变体，**不同于 VLMEvalKit 默认的 LLM 提取方式**。

如果有固定的独立 Judge 服务，可使用上游 `MATH_V_auxeval` 先规则预检、再 LLM 提取，最后仍使用上游数学匹配。Judge 只提取答案，不向被测模型提供工具或反馈，不把参考答案写入提取提示词。API 失败直接使该基准报错，不静默改用另一种评分方式。

```bash
export ACCURACY_JUDGE_API_KEY='your-judge-key'
bash scripts/run_accuracy.sh run \
  --math-judge api \
  --judge-base-url http://judge-host:8000/v1 \
  --judge-model fixed-judge-model \
  --output-dir result/accuracy/with-judge
```

`api` 模式会将 MathVision 的题干与被测模型输出发往指定 Judge 地址；若指定外部服务，会使用该服务的配额。比较两个被测版本时应固定 Judge 和评分模式。

## 报告与退出码

```text
result/accuracy/int8-v1/
├── run.json                         # 命令、时间、服务/模型信息
├── environment.txt                  # 已安装依赖版本
├── summary.md                       # 可读表格与错误原因
├── summary.csv                      # UTF-8 BOM，便于 Excel/程序读取
├── summary.json                     # 原始精度、drop_pp、状态
└── <benchmark>/
    ├── config.json                  # 数据哈希、推理配置、工具 commit
    ├── predictions.jsonl            # 逐题最终答案、完整 API 响应、usage、错误、耗时
    ├── result.json                  # 该基准的最终分数和完成状态
    └── evaluation.*                 # OpenCompass/VLMEvalKit 逐题判分及原生结果
```

实时进度输出到终端，可使用 shell 的 `tee` 保存日志。请求中不包含参考答案；逐题文件保存提示词哈希，提示词可由固定的代码、数据、工具版本重建。JSONL 追加日志可能包含同一题的失败记录和后续成功记录，续跑以最后一条成功记录为准。中断留下的不完整最后一行会被截断恢复。

可随时从各项 `result.json` 重建报告：

```bash
bash scripts/run_accuracy.sh report --output-dir result/accuracy/int8-v1
```

退出码：`0` 四项全部通过（或 prepare 成功）；`1` 四项完整但存在不达标；`2` 配置/依赖/数据/请求/评分失败、缺项或冒烟测试；`130` 手动中断。报告重建只读取当前目录的结果，不执行模型推理。

## 上游实现来源与本地验证

- [OpenCompass GSM8K 配置](https://github.com/open-compass/opencompass/blob/1543bd270b6fbabb1a118307f78d4144e8296467/opencompass/configs/datasets/gsm8k/gsm8k_gen_1d7fe4.py)
- [OpenCompass LiveCodeBench 评分](https://github.com/open-compass/opencompass/blob/1543bd270b6fbabb1a118307f78d4144e8296467/opencompass/datasets/livecodebench/evaluator.py)：直接调用 `codegen_metrics`，避免高层 evaluator 再次加载其他 release 数据。
- [LiveCodeBench test6.jsonl](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl)
- [VLMEvalKit MathVision](https://github.com/open-compass/VLMEvalKit/blob/21e52b8df8ce1f628e93c82de547edb2c985f6ac/vlmeval/dataset/image_vqa.py) / [数学匹配](https://github.com/open-compass/VLMEvalKit/blob/21e52b8df8ce1f628e93c82de547edb2c985f6ac/vlmeval/dataset/utils/mathv.py)
- [VLMEvalKit RealWorldQA 数据和 MCQ 评分](https://github.com/open-compass/VLMEvalKit/blob/21e52b8df8ce1f628e93c82de547edb2c985f6ac/vlmeval/dataset/image_mcq.py)

本地回归检查：`python3 -m unittest discover -s tests -p 'test_accuracy.py'`。这些检查验证报告判定、HTTP 处理、缓存复用和断点恢复，不替代 Linux 客户端上的真实模型评测。

本次开发验证：10 项回归测试通过；四个真实数据集完成下载、题数及可用校验和检查，并验证离线复用；175 道 LiveCodeBench 的全部公开/私有用例成功解码。Linux x86_64 / Python 3.10 的依赖解析通过，但未在本机安装并运行整套评测依赖。当前机器未提供运行中的 SGLang 服务，因此未执行真实模型推理或四项完整评分，不包含实测精度结论。
