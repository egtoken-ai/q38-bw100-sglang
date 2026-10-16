ARG BASE_IMAGE=sglang-bw100:gdn-kkt-solve-issue1
FROM ${BASE_IMAGE}

# Copy the consolidated, validated TTFT combo patches into SGLang
COPY sglang/ /usr/local/lib/python3.10/dist-packages/sglang/

# Byte-compile updated Python modules
RUN python3 -m compileall -q /usr/local/lib/python3.10/dist-packages/sglang/srt

# Copy NUMA-aware service startup script
COPY start_server.sh /opt/ttft-numa/start_server.sh
RUN chmod +x /opt/ttft-numa/start_server.sh

# Environment parameters validated on BW100 GPU 4/5 TP2
ENV Q38_HIPBLASLT_SHAPES=4096x8192x5120,8192x17408x5120,8192x8192x5120,8192x5120x3072,16384x17408x5120,16384x8192x5120,16384x5120x3072,32768x8192x5120,32768x5120x3072,65536x5120x3072
ENV SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1
ENV SGLANG_QWEN27B_MLP_PIPELINE=1
ENV SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8
ENV SGLANG_QWEN27B_MLP_CATFREE=1
ENV SGLANG_QWEN27B_CONV1D_TUNE=1
ENV SGLANG_QWEN27B_MROPE_TILED=1
ENV SGLANG_NUMA_BIND_V2=0
ENV NCCL_MIN_NCHANNELS=8
ENV NCCL_MAX_NCHANNELS=8
