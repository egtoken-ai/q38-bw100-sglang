# Fix RCCL to 8 channels for the 8-part overlapped MLP pipeline.
# Env-only layer: no Python or kernel source changes.
ARG BASE_IMAGE=sglang-bw100:ttft-postops-issue3
FROM ${BASE_IMAGE}
ENV NCCL_MIN_NCHANNELS=8 \
    NCCL_MAX_NCHANNELS=8
