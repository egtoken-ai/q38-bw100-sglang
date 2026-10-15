# 单形状生产候选：M65536 out投影白名单扩展

仅将`(65536, 5120, 3072)`加入已部署GEMM helper的`SHAPES`，并把运行环境从原9个已批准形状扩展为10个。计算公式、量化、BF16输出、scale、原生API调用、硬件/布局guard以及其他函数逐字保持。没有纳入65536×17408×5120或65536×8192×5120；没有复制父目录允许三个实验形状的诊断helper。

## 已有设备证据与限制

[原始微测](../../../result/ttft/gemm-dispatch-m65536.json)中65536×5120×3072满足全部门槛：335,544,320个输出元素全量exact、native dispatch hits=1、实际64 CU、8/8配对更快。原ASM中位数5.758984ms，候选5.538105ms，提升3.83538%；8轮配对提升范围3.74822%–4.02371%。

另两形状明确保持原分派：65536×17408×5120首轮提升为−0.99077%，不满足8轮一致；65536×8192×5120中位数提升1.23840%，未达到既定2%门槛。它们虽全量exact，也不因此放入生产ENV。

这是单模块设备证据，不是模型TTFT或完整checkpoint精度证据。完整模型需确认新形状实际命中日志，再按原两条固定case验收；没有改变输入/输出、并发、缓存、dtype或分位数口径。

## 安装与CPU校验

`install.py`只读取并修改：

```
/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/compressed_tensors/schemes/q38_bw100_legacy_gemm_dispatch.py
```

原helper SHA256必须为`302417813ae667e02ed8c7a48784482043049d91b68e0b157f4302596928d428`，已与项目`patch/ttft_gemm_dispatch/gemm_dispatch.py`及当前容器`qwen38_bw100_ttft_qk64_v2`内文件交叉核实。只在唯一SHAPES定义后插入一行`SHAPES.add((65536, 5120, 3072))`；任何其他源码漂移或已有备份均拒绝。它不会重新改写compressed-tensors原算子、其他model文件或已有FA/GDN/MLP/NUMA补丁。

备份为`q38_bw100_legacy_gemm_dispatch.py.before_64k_out`，最终扩展名不是`.py`，不会被模块自动发现扫描为额外Python模型。安装前先compile校验；`--check`不写文件。默认父镜像`sglang-bw100:ttft-numa4-issue3`，允许BASE_IMAGE覆盖，与后续FA gate镜像串联；父镜像仍须有完全匹配的原GEMM helper。

已运行`python3 patch/ttft_gemm_dispatch_64k/install_extension/check_cpu.py`，通过真实临时目录安装与以下检查：

- candidate删除唯一新增tuple行后与原文件逐字相同，全部函数AST相同。
- `SHAPES`只新增一个tuple，ENV恰为原9形状加这一项。
- `approved_shapes`接受10项/原9项/空集，拒绝另两65536实验形状及down投影。
- `--check`无写入；实际临时安装、非.py备份、拒绝重复安装、拒绝SHA漂移、无关源文件不变。

本次没有执行Docker build、GPU任务或服务操作。准备构建命令（由主agent串行调度）：

```bash
docker build -t sglang-bw100:ttft-numa4-gemm64k-out-issue3 \
  patch/ttft_gemm_dispatch_64k/install_extension

# 如需叠加另一已选FA gate父镜像，显式传入其完整tag；不能省略父镜像记录。
# docker build --build-arg BASE_IMAGE=<已选父镜像完整tag> \
#   -t <新候选tag> patch/ttft_gemm_dispatch_64k/install_extension
```

服务启动沿用所选父镜像入口与全部环境；若父镜像包含NUMA，继续使用`/opt/ttft-numa/start_server.sh`，不能因为GEMM扩展切回其他入口。新Docker ENV提供10形状，若主agent启动时额外传入`Q38_HIPBLASLT_SHAPES`，需保存实际值并确认未被旧9形状覆盖。核查日志新增`QWEN27B_HIPBLASLT_HIT M=65536 N=5120 K=3072 BF16 INT8`；不能把仅ENV含有它当成实际命中。

运行时回退：重启同一镜像服务并把ENV设回原9形状（或用原父镜像），该tuple虽仍在硬限制集合里，但不再被运行白名单选择：

```bash
Q38_HIPBLASLT_SHAPES=4096x8192x5120,8192x17408x5120,8192x8192x5120,8192x5120x3072,16384x17408x5120,16384x8192x5120,16384x5120x3072,32768x8192x5120,32768x5120x3072
```

不要在运行中的Python进程只修改宿主环境并宣称回退已生效。对比必须记录父镜像ID、启动入口、实际白名单和相同其余参数；本目录没有自动停止或启动服务的命令。

## 组合模型记录

与NUMA父配置组合后的 `sglang-bw100:ttft-postops-issue3` 固定suite268秒，短TTFT14.303938s（相对安擎原生提升19.47%）、长10.273190s（提升35.39%）。实际分派日志命中，130个输出hash与NUMA父候选逐个一致。结果 `result/quick/ttft-postops/20261015-192750-UC3IQs/`；该组合差异不能单独归因于此一项，短case仍未达标，完整任务精度与重复A/B另验。
