ARG BASE_IMAGE=sglang-bw100:ttft-qk-gate-64k-v2-issue3
FROM ${BASE_IMAGE}
COPY start_server.original.sh start_server.sh check_startup.py /opt/ttft-numa/
RUN python3 /opt/ttft-numa/check_startup.py
# The original /opt/ttft-prefill wrapper provided these defaults. The new
# launcher retains them; root can still override every startup environment.
ENV ATTENTION_BACKEND=fa3 \
    SGLANG_KV_LAYOUT_HCU_FA=true \
    SGLANG_NUMA_BIND_V2=0
