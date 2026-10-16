ARG BASE_IMAGE=sglang-bw100:ttft-postops-issue3
FROM ${BASE_IMAGE}
COPY causal_conv1d_ttft.py install.py config.json /opt/ttft-conv1d-tune/
RUN python3 /opt/ttft-conv1d-tune/install.py
ENV SGLANG_QWEN27B_CONV1D_TUNE=1
