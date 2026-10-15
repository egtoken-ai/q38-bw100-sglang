#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_path=${1:-"${source_dir}/libq38_hipblaslt_cached.so"}
# CPU-only C++ compilation/linking. No HIP kernels or GPU execution at build time.
${CXX:-g++} -std=c++17 -O3 -fPIC -shared -D__HIP_PLATFORM_AMD__ \
  -I/opt/dtk/include -I/opt/dtk/include/hipblas \
  "${source_dir}/hipblaslt_cached.cpp" \
  -L/opt/dtk/lib -Wl,-rpath,/opt/dtk/lib -lhipblaslt -lamdhip64 \
  -o "${output_path}"
