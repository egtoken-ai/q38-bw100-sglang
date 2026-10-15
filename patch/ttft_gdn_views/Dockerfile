ARG BASE_IMAGE=sglang-bw100:ttft-packed-issue3
FROM ${BASE_IMAGE}
COPY qwen3_5.py layernorm_gated.py install.py bench_views.py /opt/ttft-gdn-views/
RUN python3 /opt/ttft-gdn-views/install.py
ENV SGLANG_QWEN27B_GDN_PREFILL_VIEWS=true
