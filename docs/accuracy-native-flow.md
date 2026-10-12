# 原生精度评测脚本调用流程

[run_accuracy_native.sh](../scripts/run_accuracy_native.sh) 负责选择 Python 并转发参数，主要调度逻辑位于 [accuracy_native.py](../scripts/accuracy_native.py) 的 `main()`。安装和使用方法见 [原生评测说明](accuracy-native.md)。

## 入口与子命令分支

```mermaid
flowchart TD
    A["run_accuracy_native.sh 参数"] --> B["选择 Python<br/>ACCURACY_PYTHON 或 .venv-accuracy<br/>不可执行则回退 python3"]
    B --> C["exec python -u accuracy_native.py<br/>原样传递参数"]
    C --> D["解析参数、规范化路径和 URL、校验参数"]
    D --> E{"子命令"}

    E -->|prepare| P["复用或下载数据<br/>准备 LCB 本地加载器<br/>下载固定版本 VLMEvalKit run.py"]
    P --> P1["校验数据并保存 native_manifest.json"]

    E -->|其他命令| L["锁定输出目录<br/>防止并发写入"]
    L --> F{"子命令"}

    F -->|report| R["读取已有结果<br/>按条件复核原生输出"]
    R --> R1["生成 Markdown / CSV / JSON 报告"]

    F -->|config / run / infer / eval| G["检查配置是否允许复用<br/>保存 suite_config.json"]
    G --> H["生成框架配置<br/>commands.json 和 commands.sh"]
    H --> I{"子命令"}

    I -->|config| J["打印原生命令并结束<br/>不执行推理"]
    I -->|run / infer / eval| K["逐个执行所选 Benchmark"]
    K --> R1
```

参数无效、无法取得目录锁或整体配置不兼容时，会结束本次调用并返回错误；这些检查发生在逐项评测循环之前。

## run 完整执行流程

不同 Benchmark 按顺序运行；单个 Benchmark 内部的请求并发由原生框架管理。外层脚本通过 `execute()` 启动原生命令，不自行发送模型推理请求或计算基准分数。

```mermaid
flowchart TD
    A["run：记录运行信息<br/>将所选基准标记为 running"] --> B{"还有未处理的基准？"}
    B -->|有| C["取下一个 Benchmark"]
    B -->|无| Z["生成最终准入报告<br/>返回退出码"]

    C --> D{"MathVision<br/>是否缺少 Judge 配置？"}
    D -->|是| ERR["记录 ERROR"]
    D -->|否或其他基准| E["检查框架安装版本"]
    E --> F["prepare：缓存存在则校验复用<br/>不存在则下载；offline 模式下报错"]
    F --> G["核对本次数据与历史数据指纹"]
    G --> H{"基准类型"}

    H -->|GSM8K / LiveCodeBench| OC["启动 OpenCompass CLI<br/>--reuse native --mode all"]
    OC --> OC1["原生推理流程<br/>构造提示词 → 调用 SGLang API"]
    OC1 --> OC2["原生 Evaluator 评分<br/>GSM8K 数值匹配 / LCB 代码测试"]

    H -->|RealWorldQA / MathVision| VL["启动 VLMEvalKit run.py<br/>--mode infer --reuse"]
    VL --> VL1["原生图文推理<br/>发送图片和问题 → SGLang API"]
    VL1 --> VL2["再次启动原生 run.py<br/>--mode eval --reuse"]
    VL2 --> VJ{"视觉基准"}
    VJ -->|RealWorldQA| RW["exact_matching 评分"]
    VJ -->|MathVision| MV["Judge API 提取答案<br/>原生数学匹配评分"]

    OC2 --> AUDIT["audit：检查题数与 ID、空输出、失败标记<br/>评分状态、Judge 结果和分数单位"]
    RW --> AUDIT
    MV --> AUDIT

    AUDIT --> SAVE["保存该基准 result.json<br/>更新汇总报告"]
    ERR --> SAVE
    SAVE --> B

    E -.->|步骤异常| ERR
    F -.->|步骤异常| ERR
    G -.->|步骤异常| ERR
    OC -.->|命令失败| ERR
    VL -.->|命令失败| ERR
    VL2 -.->|命令失败| ERR
    AUDIT -.->|检查失败| ERR
```

图中的推理、评分节点表示原生框架的职责；已有有效缓存时，框架可按 `--reuse` 规则跳过已完成工作。逐项循环内捕获的异常会保存为该项 `ERROR`，然后继续其他基准；手动中断则结束整个运行。

## 子命令与输出

| 命令 | 实际行为 |
|---|---|
| `prepare` | 下载或复用数据，准备本地加载器及原生入口，不执行推理 |
| `config` | 生成具体配置、`commands.json` 和可执行的 `commands.sh`，不执行推理 |
| `run` | 推理、评分、完整性检查及准入报告 |
| `infer` | 只运行原生推理，成功标记为 `inferred`，不判定准入通过 |
| `eval` | 使用已有预测运行原生评分，再检查结果并生成准入报告 |
| `report` | 读取已有结果并按条件复核原生输出，重新汇总，不发起推理或评分 |

`report` 不会仅凭旧输出将已记录为 `error` 或 `running` 的任务改为通过；这类任务需要通过 `run` / `eval` 恢复。

输出目录顶层保存 `summary.md`、`summary.csv`、`summary.json`；原生工作目录分别位于 `opencompass/<benchmark>/native/` 和 `vlmeval/<benchmark>/q38/T*/`。四项完整且分别达到 95.0、88.3、83.9、88.0，整体判定才为 `PASS`。

| 退出码 | 含义 |
|---|---|
| `0` | 完整评测四项全部通过；或 `prepare`、`config`、所选项的 `infer` 成功 |
| `1` | 四项评分完成，但存在精度不达标 |
| `2` | 参数、配置或执行错误，或完整评测存在缺项 |
| `130` | 手动中断 |

`infer` 返回 0 仅表示推理阶段成功，准入报告仍为 `INCOMPLETE`。
