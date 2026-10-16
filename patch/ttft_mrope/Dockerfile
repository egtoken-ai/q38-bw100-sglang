ARG BASE_IMAGE=sglang-bw100:ttft-postops-issue3
FROM ${BASE_IMAGE}
COPY mrope_tiled.py install.py config.json /opt/ttft-mrope/
RUN python3 /opt/ttft-mrope/install.py
ENV SGLANG_QWEN27B_MROPE_TILED=1
