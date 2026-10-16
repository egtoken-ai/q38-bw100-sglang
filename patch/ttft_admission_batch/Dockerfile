ARG BASE_IMAGE=sglang-bw100:ttft-qk-gate-64k-v2-issue3
FROM ${BASE_IMAGE}
COPY admission_batch.py install.py source-manifest.json /opt/ttft-admission-batch/
RUN python3 /opt/ttft-admission-batch/install.py
ENV SGLANG_QWEN27B_ADMISSION_DELAY_MS=0
