ARG BASE_IMAGE=sglang-bw100:ttft-gdn-views-issue3
FROM ${BASE_IMAGE}
COPY gdn_backend.py gdn_triton.py gdn_prefill_prepare.py install.py bench_prepare.py /opt/ttft-gdn-prepare/
RUN python3 /opt/ttft-gdn-prepare/install.py
ENV SGLANG_QWEN27B_GDN_FUSED_PREPARE=true
