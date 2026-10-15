ARG BASE_IMAGE=sglang-bw100:ttft-mlp-pipeline-issue3
FROM ${BASE_IMAGE}
COPY mlp_pipeline.py pipeline_probe.py install.py bench_stress.py /opt/ttft-mlp-current/
RUN python3 /opt/ttft-mlp-current/install.py
ENV SGLANG_MLP_DIAG_ENABLED=1
ENV SGLANG_MLP_DIAG_MAX_BATCHES=2
