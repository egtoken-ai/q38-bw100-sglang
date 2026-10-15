# Qwen3.8-27B-Channel-INT8-w8a8 推理优化工程 (海光 BW100 双卡)

本项目面向社区 Qwen3.8 推理性能优化挑战赛（子赛道 B：SGLang），针对海光 BW100 双卡平台与 Qwen3.8-27B-Channel-INT8-w8a8 模型，提供完整的容器配置、推理服务启动脚本以及标准评测执行脚本。

精度准入验证见 [OpenCompass / VLMEvalKit 使用说明](docs/accuracy.md)：支持 GSM8K、LiveCodeBench v6（175 题）、RealWorldQA 和 MathVision testmini（304 题），包含自动下载、缓存复用、逐题续跑及 Markdown/CSV/JSON 报告。入口：`bash scripts/run_accuracy.sh run`，需先按说明安装独立评测环境并启动模型服务。

新增 **[原生配置＋命令版本](docs/accuracy-native.md)**：`bash scripts/run_accuracy_native.sh run`，真正调用 OpenCompass CLI 和 VLMEvalKit `run.py`，由框架执行推理、评分与续跑。MathVision 完整评分需配置独立 Judge API；原来的 Python API 版本仍保留。

---

## 一、目录结构

```text
.
├── env_declare.yaml         # 环境声明配置
├── requirements.txt         # 测试客户端依赖
├── README.md                # 部署与复现指导
├── config/                  # 自定义配置文件目录
├── logs/                    # 服务运行日志目录
├── patch/                   # 框架/算子优化补丁目录
├── result/                  # 评测输出结果目录
│   ├── fixed/               # 固定长度测试结果 (4K/64K)
│   └── sharegpt/            # ShareGPT 测试结果
└── scripts/
    ├── benchmark.py         # 固定长度压测客户端脚本
    ├── run.sh               # 一键式统一评测入口脚本
    ├── run_bench_fixed.sh   # 固定长度自动化测试脚本
    ├── run_bench_sharegpt.sh# ShareGPT 自动化测试脚本
    ├── start_docker.sh      # 官方容器启动脚本
    └── start_server.sh      # SGLang 服务启动脚本
```

---

## 二、环境准备

### 1. 镜像地址
官方指定测试镜像：
`harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`

### 2. 启动容器
执行如下命令启动容器并挂载模型与项目目录：

```bash
# 环境变量可根据机器实际路径调整
export MODEL_PATH="/workspace/work/Qwen3.8-27B-Channel-INT8-w8a8"
export DATASET_PATH="/workspace/dataset"

bash scripts/start_docker.sh
```

进入容器：
```bash
docker exec -it qwen38_bw100_sglang bash
cd /bench
```

### 3. 安装依赖
在容器内安装压测客户端依赖：
```bash
pip install -r requirements.txt
```

---

## 三、服务启动

### 1. 使用一键脚本启动 (推荐)
在容器内执行：
```bash
# 后台启动服务并等待就绪
bash scripts/run.sh start-server
```

### 2. 独立脚本启动
如果需要自定义参数或前台观察输出：
```bash
bash scripts/start_server.sh
```

后台启动：
```bash
bash scripts/start_server.sh --daemon
```

核心环境变量与启动参数说明：
- `HIP_VISIBLE_DEVICES=0,1`：使用双卡 BW100。
- `HSA_FORCE_FINE_GRAIN_PCIE=1`：开启细粒度 PCIe 优化。
- `--tp-size 2`：双卡张量并行。
- `--dtype bfloat16` 与 `--kv-cache-dtype bfloat16`：计算与缓存数据类型。
- `--context-length 131072`：最大上下文长度支持 128K。
- `--port 18580`：HTTP 推理服务端口，健康检查路径为 `http://127.0.0.1:18580/health`。

---

## 四、压测脚本使用方法与性能评测

### 1. 一键全流程评测
运行全量测试流水线（服务健康状态校验、固定长度测试、ShareGPT 测试）：
```bash
bash scripts/run.sh all
```

---

### 2. 固定长度请求测试 (4K 与 64K)

固定长度客户端使用随机 Token 构造输入，设置 `ignore_eos=True` 强制输出 1024 个 Token；配置 2 轮预热、3 轮正式测试，每轮请求数为 $\max(8, 2 \times \text{并发})$。

#### (1) 测试档位规范
- **4K 输入 / 1K 输出**：C1、C8、C16、C32、C64、C128
- **64K 输入 / 1K 输出**：C1、C8、C16、C32

#### (2) 自动化执行命令
使用自动化调度脚本 `scripts/run_bench_fixed.sh`：
```bash
# 执行全部固定长度档位 (4K 与 64K 所有档位)
bash scripts/run_bench_fixed.sh all
# 或使用主控入口：
bash scripts/run.sh bench-fixed all

# 仅测试 4K/1K 场景 (C1 至 C128)
bash scripts/run_bench_fixed.sh 4k

# 仅测试 64K/1K 场景 (C1 至 C32)
bash scripts/run_bench_fixed.sh 64k
```

#### (3) 单档位底层命令执行示例 (benchmark.py)
如需单独调试或复现某一特定并发（例如 4K 输入、1K 输出、并发 32）：
```bash
python3 -u scripts/benchmark.py \
  --base-url http://127.0.0.1:18580 \
  --model Qwen3.8-27B-Channel-INT8-w8a8 \
  --output-dir result/fixed/4096in_1024out_c32 \
  --input-len 4096 \
  --output-len 1024 \
  --concurrency 32 \
  --warmup-rounds 2 \
  --measure-rounds 3 \
  --requests-per-round 64 \
  --timeout 3600
```
结果将自动汇总生成在 `result/fixed/4096in_1024out_c32/summary.json`。

---

### 快速优化测试（2 个 case）

`scripts/run_bench_quick.sh` 用于优化迭代筛选。从原有四个 case 中保留
**4K/1K C64、64K/1K C1**，分别覆盖并发吞吐和长输入单请求性能，case 的输入长度、
输出长度和并发参数不变。每档 **1 轮预热 + 1 轮正式测试**，每轮请求数等于并发数，
共 130 个请求（含预热）。保留 1024-token 输出和固定随机种子。
按已有基线的一波请求耗时估算约 5 分钟（两档各约 98 秒、51 秒，每档两轮），
目标控制在 8 分钟以内；实际受待测版本、编译和机器负载影响。

在已启动目标服务的容器内执行：

```bash
cd /bench
# 预览两个命令，不发送请求、不创建结果目录
bash scripts/run_bench_quick.sh base --dry-run
# 原生服务上采集快速基线
bash scripts/run_bench_quick.sh base
# 切换为待验证的优化服务后，用相同配置测试
bash scripts/run_bench_quick.sh gdn-opt
```

脚本检查服务和模型名称，不自动启动或切换服务。支持 `BASE_URL`、`MODEL_NAME`、
`RESULT_ROOT` 环境变量。每次运行保存到独立的 `result/quick/<标签>/<时间戳>-<随机后缀>/`，
包含每档日志、原始请求、进度、TPS/TTFT P50/TPOT P50 汇总 `summary.csv` 与 `summary.json`，
以及脚本指纹和运行配置。`exit_code=0` 表示全部通过；请求失败或 token 长度不符立即停止后续档位。

原生与优化版本应使用相同快速脚本、硬件、服务参数和空闲机器比较。单轮结果用于筛选，
尤其 C1 只有一个正式请求，波动较大；不能将其与完整压测的多轮、多波次结果视为等价。
确认收益后仍运行 `run_bench_fixed.sh all` 做完整性能回归；性能 PASS 不代表模型精度通过。

### 3. ShareGPT 真实对话数据集测试

基于官方标准文件 `ShareGPT_V3_unfiltered_cleaned_split.json`，每档进行 1000 个正式请求，使用 $2 \times \text{并发}$ 个请求进行预热，执行 3 轮正式测试。

#### (1) 测试档位规范
- 并发档位：C1、C4、C16、C32、C64

#### (2) 自动化执行命令
使用自动化脚本 `scripts/run_bench_sharegpt.sh`：
```bash
bash scripts/run_bench_sharegpt.sh
# 或使用主控入口：
bash scripts/run.sh bench-sharegpt
```

可通过环境变量覆盖参数（如数据集路径）：
```bash
export DATASET_PATH="/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
bash scripts/run_bench_sharegpt.sh
```

#### (3) 单档位底层命令执行示例 (sglang.bench_serving)
以并发 32 为例，单轮执行命令为：
```bash
python3 -u -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url http://127.0.0.1:18580 \
  --dataset-name sharegpt \
  --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /model \
  --served-model-name Qwen3.8-27B-Channel-INT8-w8a8 \
  --sharegpt-context-len 131072 \
  --num-prompts 1000 \
  --max-concurrency 32 \
  --request-rate inf \
  --warmup-requests 64 \
  --seed 20260912 \
  --disable-tqdm \
  --output-details \
  --output-file result/sharegpt/sharegpt-c32/round-01.jsonl
```

---

### 4. 停止推理服务
测试完成后，可使用主控脚本安全停止后台服务：
```bash
bash scripts/run.sh stop-server
```
