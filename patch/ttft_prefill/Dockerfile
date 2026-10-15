ARG BASE_IMAGE=sglang-bw100:gdn-kkt-solve-issue1
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="BW100 Qwen27B TTFT attention candidate"
LABEL org.opencontainers.image.description="Issue 3: target-validated Hygon FlashAttention prefill candidate"
COPY start_server.sh /opt/ttft-prefill/start_server.sh
ENV ATTENTION_BACKEND=fa3
ENV SGLANG_KV_LAYOUT_HCU_FA=true
# The serving process still uses the project script and explicit model/GPU env.
# Keep the container idle so baseline/candidate trials can share the same GPUs.
ENTRYPOINT ["/bin/sleep"]
CMD ["infinity"]
