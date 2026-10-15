ARG BASE_IMAGE=sglang-bw100:ttft-prefill-release-issue3
FROM ${BASE_IMAGE}
COPY qk_norm_gate.py install.py source-manifest.json /opt/ttft-qk-norm-gate/
RUN python3 /opt/ttft-qk-norm-gate/install.py
ENV SGLANG_QWEN27B_QK_NORM_GATE=1
