ARG BASE_IMAGE=sglang-bw100:ttft-gdn-prepare-w4-issue3
FROM ${BASE_IMAGE}
COPY . /opt/ttft-gdn-metadata/
RUN python3 /opt/ttft-gdn-metadata/install.py
ENV SGLANG_QWEN27B_GDN_CPU_METADATA=1
