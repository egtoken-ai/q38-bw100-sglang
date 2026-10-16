ARG BASE_IMAGE=sglang-bw100:ttft-postops-issue3
FROM ${BASE_IMAGE}
COPY mlp_pipeline.py install.py bench_catfree.py /opt/ttft-mlp-catfree/
RUN python3 /opt/ttft-mlp-catfree/install.py
# Default on in this layer; =0 falls back to the original per-part all_reduce + torch.cat.
ENV SGLANG_QWEN27B_MLP_CATFREE=1
