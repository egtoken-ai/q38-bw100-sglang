ARG BASE_IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="SGLang BW100 GDN complete-store validation"
LABEL org.opencontainers.image.description="Issue 1: remove redundant GDN inverse buffer initialization"
COPY chunk_fwd.py install.py verify.py /opt/gdn-kkt-solve/
RUN python3 /opt/gdn-kkt-solve/install.py
ENTRYPOINT ["/bin/sleep"]
CMD ["infinity"]
