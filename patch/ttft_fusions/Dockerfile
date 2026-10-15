ARG BASE_IMAGE=sglang-bw100:ttft-packed-issue3
FROM ${BASE_IMAGE}
COPY install.py qwen27b_int8_mlp.py /opt/ttft-fusions/
RUN python3 /opt/ttft-fusions/install.py
ENV SGLANG_QWEN27B_FUSED_SILU_QUANT=false
ENV SGLANG_USE_FUSED_SILU_MUL_QUANT=true
