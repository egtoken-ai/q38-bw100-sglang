ARG BASE_IMAGE=sglang-bw100:ttft-current-gemm-issue3
FROM ${BASE_IMAGE}
COPY mlp_pipeline.py install.py current_parts.py bench_actual_parts.py /opt/ttft-mlp-current-parts/
RUN python3 /opt/ttft-mlp-current-parts/install.py
ENV SGLANG_QWEN27B_MLP_PIPELINE_PARTS=4
