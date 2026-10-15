ARG BASE_IMAGE=sglang-bw100:ttft-gdn-prepare-w4-issue3
FROM ${BASE_IMAGE}
COPY . /opt/ttft-kv-pack/
RUN python3 /opt/ttft-kv-pack/install.py
ENV SGLANG_QWEN27B_FUSED_KV_PACK=true
