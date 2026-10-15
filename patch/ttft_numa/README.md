# BW100 GPU4/5 显式 NUMA4 绑定候选

这是未测性能的启动配置候选，目标是改善scheduler、CPU launch、host metadata与传输的NUMA locality。没有更改模型、BF16/INT8精度、请求、缓存或输出长度；不能预先声称补足短case尚差的0.505秒。此次只读检查和文档编写没有改变服务、进程affinity或GPU状态。

## 已确认硬件与当前缺口

2026-10-15在宿主运行 `/opt/hyhal/bin/hy-smi --showbus --showtoponuma`，并交叉检查PCI sysfs，得到：

| 物理设备 | PCI BDF | NUMA node / affinity | 本地CPU |
| --- | --- | --- | --- |
| HCU4 | 0000:98:00.0 | 4 / 4 | 64–79,192–207 |
| HCU5 | 0000:9b:00.0 | 4 / 4 | 64–79,192–207 |

`/sys/bus/pci/devices/0000:98:00.0/{numa_node,local_cpulist}`与`0000:9b:00.0`结果一致，`/sys/devices/system/node/node4/cpulist`同样为`64-79,192-207`。这不是由卡序号猜测的映射。

当前重建服务`qwen38_bw100_ttft_qk64_v2`的scheduler PID193、194实查affinity均为CPU0–255；日志ServerArgs为`numa_node=None, base_gpu_id=0, gpu_id_step=1`。白名单环境`SGLANG_SET_CPU_AFFINITY`与`SGLANG_NUMA_BIND_V2`均unset。后者**有效默认值是True**，见vendor `environ.py:601`，不能把unset当False。

镜像源码根目录为`/usr/local/lib/python3.10/dist-packages/sglang/srt`。关键路径：

- `utils/numa_utils.py:158–163`：自动检测遇到非CUDA直接返回False；`:196–239`的设备查询也采用NVML。`utils/common.py:134–135`的CUDA判断要求`torch.version.cuda is not None`。HCU/HIP自动绑定不能依靠这条路径。
- `utils/numa_utils.py:103–104`：显式`server_args.numa_node[gpu_id]`先返回，绕过上述自动检测限制。
- `server_args.py:7105–7109`：`--numa-node`的类型为int、`nargs='+'`。`entrypoints/engine.py:603–606`计算的是可见设备逻辑ID；当前base0/step1/TP2对应0、1，配合`HIP_VISIBLE_DEVICES=4,5`，应传两个值：`--numa-node 4 4`。不要传物理GPU序号列表。

官方也讨论过手动NUMA映射和host跨节点开销，见[官方Issue #20073](https://github.com/sgl-project/sglang/issues/20073)及[NUMA工具源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/numa_utils.py)。Issue于2026-05-10因无活动自动关闭，不能把“Closed”解读为BW100已修复或有性能验证。此处实际分派判断以vendor源码为准。

## 首选有界方案与分配时序

下一轮采用`SGLANG_NUMA_BIND_V2=0`加`--numa-node 4 4`，保持`SGLANG_SET_CPU_AFFINITY=0`。这会在scheduler内调用`numa_run_on_node(4)`及`numa_set_preferred(4)`（`numa_utils.py:133–140`）：scheduler CPU限定为node4，后续host内存优先本地分配，允许回落其他node。

时序已经确认：`run_scheduler_process:4187`先调用`configure_scheduler_process`，其`:4165–4168`完成绑定；然后`:4212`才构造Scheduler；Scheduler`:444`初始化model worker；`tp_worker.py:350`创建ModelRunner；`model_runner.py:667`加载模型、`:761`建立memory pool。因此该方案覆盖后续模型加载、CPU staging和pool初始化时的host分配，并非在池建好以后才改CPU。GPU VRAM仍在指定GPU上，NUMA设置不是把显存移到host；已有host页也不会因为set_preferred被自动迁移。Python导入及在绑定前创建的线程/页不保证被重新绑定或迁移。

默认V2=True时则在`configure_subprocess:27–32`使用`numactl --cpunodebind=4 --membind=4`，在子进程启动前严格绑定host内存。它可能限制单node容量，不与首选preferred方案混测。不要另加`SGLANG_SET_CPU_AFFINITY=1`：`utils/common.py:2229–2248`只是按全机CPU数均分，不考虑GPU PCI拓扑，且与NUMA设置顺序叠加可能产生误解。

## 可构建启动候选（本次未build、未启动）

目录包含guarded原脚本`start_server.original.sh`、候选`start_server.sh`、`numa-startup.patch`、CPU检查器和Dockerfile。原脚本来自当前项目`scripts/start_server.sh`，SHA256为`fa4ca0aa1fd7e62e43e0c567ad4d152092e180efad193ed3d9dc2e80418063a2`。候选仅以下精确差异：

1. HIP_VISIBLE_DEVICES默认值从`0,1`改为`4,5`；若外部显式指定其他设备则不偷偷覆盖。
2. 在原LAUNCH_CMD构造完成后、NUMA参数注入及Python服务启动前，要求实际`HIP_VISIBLE_DEVICES=4,5`且`TP_SIZE=2`，否则exit 2；通过后加入`LAUNCH_CMD+=(--numa-node 4 4)`。
3. `PROJECT_DIR=/bench`：脚本放到`/opt/ttft-numa`后，原来按脚本父目录计算会错误得到`/opt`；显式路径确保日志和结果仍写`/bench/logs`。

`check_startup.py`按固定SHA验证原文件，并重建上述包括设备默认值与scope guard的预期内容，逐字节比较候选、运行bash语法检查。已在CPU上通过，带`--reference scripts/start_server.sh`也通过。Docker构建阶段运行同一guard；运行前可额外验证挂载的项目脚本未漂移。项目原`scripts/start_server.sh`没有修改。

默认FROM `sglang-bw100:ttft-qk-gate-64k-v2-issue3`。Dockerfile仅复制launcher与guard，并设置V2=0；同时把旧`/opt/ttft-prefill` wrapper原有FA默认值`ATTENTION_BACKEND=fa3`及`SGLANG_KV_LAYOUT_HCU_FA=true`显式保留。父镜像GDN、GEMM白名单、QK和pipeline代码/环境均继承，不再安装或改写。所有启动参数仍可由主agent用docker exec环境覆盖。

仅在主agent完成当前固定suite、确定有必要测试后执行构建：

```bash
# 在宿主项目目录执行；本次没有运行此命令。
docker build -t sglang-bw100:ttft-numa-issue3 patch/ttft_numa
```

先用原项目start_docker.sh按主agent流程创建该镜像的新sleep-only容器，并确认GPU4/5独占。候选服务必须复用刚测过的父服务chunk/parts/mem_fraction、9形状白名单、QK和调度flag；单独改变NUMA，不能同时更换64K/8与32K/4。以下命令要求从parent部署记录填写变量，无隐含调参：

```bash
set -eu
: "${NUMA_CONTAINER:?填入新sleep-only候选容器名}"
: "${MATCHED_CHUNK:?从parent部署记录填写chunk}"
: "${MATCHED_MAX_PREFILL:?从parent部署记录填写max-prefill-tokens}"
: "${MATCHED_PARTS:?从parent部署记录填写parts}"
: "${MATCHED_MEM_FRACTION:?从parent部署记录填写mem_fraction_static}"
# 同一父镜像继承9形状及QK等flags；若父服务曾用exec env覆盖，必须同样传入。
docker exec "$NUMA_CONTAINER" python3 /opt/ttft-numa/check_startup.py \
  --reference /bench/scripts/start_server.sh
docker exec -w /bench \
  -e HIP_VISIBLE_DEVICES=4,5 \
  -e CHUNKED_PREFILL_SIZE="$MATCHED_CHUNK" \
  -e MAX_PREFILL_TOKENS="$MATCHED_MAX_PREFILL" \
  -e SGLANG_QWEN27B_MLP_PIPELINE_PARTS="$MATCHED_PARTS" \
  -e MEM_FRACTION_STATIC="$MATCHED_MEM_FRACTION" \
  -e SGLANG_NUMA_BIND_V2=0 -e SGLANG_SET_CPU_AFFINITY=0 \
  "$NUMA_CONTAINER" bash /opt/ttft-numa/start_server.sh --daemon
```

现有`/bench/scripts/start_server.sh`只识别`--daemon`，不会透传任意额外参数；不能把`--numa-node 4 4`直接附在旧wrapper尾部。

原项目start_docker.sh带`--privileged`及seccomp unconfined；仍需核实启动日志无libnuma/权限错误。Vendor V1未检查两个libnuma调用的返回值，故“ServerArgs里有numa_node”本身不是绑定成功证据。

在候选容器确认scheduler实际CPU集合（只读）：

```bash
python3 - <<'PY'
from pathlib import Path
import os
for p in Path('/proc').glob('[0-9]*'):
    try:
        if not (p/'comm').read_text().startswith('sglang::schedul'):
            continue
        allowed = sorted(os.sched_getaffinity(int(p.name)))
        expected = list(range(64,80)) + list(range(192,208))
        print('scheduler', p.name, 'cpus', allowed, 'expected_match', allowed == expected)
        counts = {}
        for line in (p/'numa_maps').read_text().splitlines():
            parts = line.split()
            if len(parts) > 1:
                counts[parts[1]] = counts.get(parts[1], 0) + 1
        print('numa_maps_policy_counts', counts)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
PY
```

需要CPU集合等于node4并在后续映射中看到`prefer:4`策略（旧映射可以仍是default）；NUMA maps条数不是字节数，也不代表全部host页已在node4。可另保存/proc各scheduler task的Cpus_allowed_list检查继承线程，避免把主线程结果泛化所有导入前线程。记录有效flag、ServerArgs、宿主拓扑、父镜像ID和chunk等其余参数，再运行原固定两case/warmup1+measure1；不调整输入、输出、并发、seed或缓存。TTFT仍需每case分别比较，亦记录TPOT/TPS、输出hash及suite时长。

## 回退与筛选边界

停止候选必须由主agent按原调度顺序进行。本文件不包含自动停止服务命令。回退使用同一父镜像和原`/opt/ttft-prefill/start_server.sh --daemon`启动新服务，移除`--numa-node`及候选显式NUMA/affinity环境；不要只在现有进程里unset变量，它不会撤销已生效affinity或内存策略。项目原脚本和模型代码未改；使用父镜像及原入口即可回退，不需要反向模型补丁。

没有为了凑第二项推荐PrefillDelayer：vendor `prefill_delayer.py:199–237`主要面向decode完成导致的小队列碎片，首次不等待，队列触发还要求running_batch>0；`:302`每轮引入all-gather。当前固定C64突发、多条完整4K排队，首批running=0不命中queue触发；128-running < max_prefill_bs的slot条件也不成立。DP1能执行delayer，但全局prefillable只有all/none，不进入mixed分支的max_delay_passes逻辑。因此现成delayer不能提供首批2–5ms聚合；若要实现必须另加有界admission debounce并计入真实TTFT，不能声称现有flag已支持。`enable_dynamic_chunking`在scheduler`:1122–1124`要求PP>1，当前PP1不命中；MQ broadcaster环境默认True，也不是尚未启用的优化。本次仅保留上述一个有实查缺口的候选。

## BW100固定case记录

268秒固定suite完成，短TTFT14.364530s（相对安擎原生提升19.13%）、长10.300595s（提升35.22%），仍未达到每档20%。相对未绑定qk64父候选，短档仅约0.15%的单轮差异，不据此宣称稳定NUMA收益。全部130个预热/正式输出hash与该父候选相同。结果、实际镜像/argv/env见 `result/quick/ttft-numa4/20261015-190749-9CUqxW/`。

`capture_binding.py` 在Docker宿主读取两个scheduler的CPU集合与numa_maps策略，实测都为 `64-79,192-207`，并存在 `prefer:4`，证据 `result/ttft/numa4-binding.json`。这验证进程绑定生效，不证明所有旧映射或每个线程均已迁移。
