ARG BASE_IMAGE=sglang-bw100:ttft-numa4-issue3
FROM ${BASE_IMAGE}
COPY fa_gate.py install.py source-manifest.json /opt/ttft-fa-gate/
RUN python3 /opt/ttft-fa-gate/install.py
ENV SGLANG_QWEN27B_FA_GATE=1
