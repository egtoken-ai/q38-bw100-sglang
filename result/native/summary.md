| Benchmark | 基线 % | 阈值 % | 实测 % | 降幅 pp | 完成题数 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| GSM8K | 97.0 | 95.0 | 95.2995 | 1.7005 | 1319/1319 | PASS |
| LiveCodeBench v6 | 90.3 | 88.3 | 63.4286 | 26.8714 | 175/175 | FAIL |
| RealWorldQA | 85.9 | 83.9 | 56.9935 | 28.9065 | 765/765 | FAIL |
| MathVision (Without CI) | 90.0 | 88.0 | — | — | 0/304 | ERROR |

整体判定：**INCOMPLETE**

分数由 OpenCompass / VLMEvalKit 原生命令生成；本报告仅校验完整性、转换单位和判断阈值。
准入使用未舍入的分数。MathVision 使用原生 LLM Judge，无 Code Interpreter。
官方基线的完整参数尚未提供；原生命令不等同于已经复现官方基线。

- MathVision (Without CI): Native evaluation not complete: done; mode_infer
