ARG BASE_IMAGE=sglang-bw100:ttft-gdn-prepare-w4-issue3
FROM ${BASE_IMAGE}
COPY mlp_pipeline.py qwen3_5.py install.py bench_actual_layer.py /opt/ttft-mlp-pipeline/
RUN python3 /opt/ttft-mlp-pipeline/install.py
ENV SGLANG_QWEN27B_MLP_PIPELINE=true
