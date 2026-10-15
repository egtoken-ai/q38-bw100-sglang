ARG BASE_IMAGE=sglang-bw100:ttft-metadata-pack-issue3
FROM ${BASE_IMAGE}
COPY gemm_dispatch.py install.py source-manifest.json /opt/ttft-gemm-dispatch/
COPY vendor/ /opt/ttft-gemm-dispatch/vendor/
RUN python3 /opt/ttft-gemm-dispatch/install.py
ENV Q38_HIPBLASLT_SHAPES=""
