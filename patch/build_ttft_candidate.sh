#!/usr/bin/env bash
# Build source-guarded layers only. No service launch or GPU validation.
set -euo pipefail

patch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
start_from=issue1
target=prefill-release
image_prefix=sglang-bw100
tag_suffix=issue3
issue1_image=''
native_image=harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10
dry_run=0
usage() {
  cat <<'HELP'
Usage: bash patch/build_ttft_candidate.sh [options]
  --dry-run                    Validate local source; print build commands only
  --from issue1|native         Start with Issue 1 image, or build it (default issue1)
  --target current-gemm|prefill-release|qk-gate|postops   Last layer (default prefill-release)
  --image-prefix NAME          Output repository (default sglang-bw100)
  --tag-suffix TEXT            Issue 3 tag suffix (default issue3)
  --issue1-image IMAGE         Issue 1 input/output tag (default PREFIX:gdn-kkt-solve-issue1)
  --native-image IMAGE         Exact vendor image used with --from native
  --help                      Show this help
Without --dry-run this builds images using the local Docker daemon.
It does not launch services, run GPU benchmarks, push images, or change Git.
HELP
}
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --from|--target|--image-prefix|--tag-suffix|--issue1-image|--native-image)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      option=$1; value=$2; shift 2
      case "$option" in
        --from) start_from=$value ;;
        --target) target=$value ;;
        --image-prefix) image_prefix=$value ;;
        --tag-suffix) tag_suffix=$value ;;
        --issue1-image) issue1_image=$value ;;
        --native-image) native_image=$value ;;
      esac ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$start_from" == issue1 || "$start_from" == native ]] || { printf 'Invalid --from\n' >&2; exit 2; }
[[ "$target" == current-gemm || "$target" == prefill-release || "$target" == qk-gate || "$target" == postops ]] || { printf 'Invalid --target\n' >&2; exit 2; }
[[ -n "$image_prefix" && -n "$tag_suffix" ]] || { printf 'Empty image prefix/suffix\n' >&2; exit 2; }
issue1_image=${issue1_image:-"${image_prefix}:gdn-kkt-solve-issue1"}

# Pure CPU preflight also runs in dry-run. Do not import torch/sglang.
python3 - "$patch_dir" <<'PY'
import ast
import hashlib
from pathlib import Path
import sys
root=Path(sys.argv[1])
dockerfiles=['gdn_kkt_solve/Dockerfile','ttft_prefill/Dockerfile','ttft_prefill/Dockerfile.packed',
 'ttft_gdn_views/Dockerfile','ttft_gdn_prepare/Dockerfile','ttft_kv_pack/Dockerfile',
 'ttft_gdn_metadata/Dockerfile','ttft_mlp_pipeline/Dockerfile',
 'ttft_mlp_pipeline/current_stream/Dockerfile','ttft_gemm_dispatch/Dockerfile',
 'ttft_prefill_release/Dockerfile','ttft_mlp_pipeline/current_stream_parts/Dockerfile',
 'ttft_qk_norm_gate/Dockerfile','ttft_numa/Dockerfile','ttft_fa_gate/Dockerfile',
 'ttft_gemm_dispatch_64k/install_extension/Dockerfile']
for name in dockerfiles:
 text=(root/name).read_text()
 assert 'ARG BASE_IMAGE=' in text and 'FROM ${BASE_IMAGE}' in text, f'BASE_IMAGE override unavailable: {name}'
prepare=ast.parse((root/'ttft_gdn_prepare/gdn_prefill_prepare.py').read_text())
fn=next(n for n in prepare.body if isinstance(n,ast.FunctionDef) and n.name=='fused_gdn_prefill_prepare')
defaults=dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],fn.args.defaults))
assert ast.literal_eval(defaults['num_warps'])==4, 'w4 tag requires actual num_warps=4 default'
assert any(isinstance(n,ast.keyword) and n.arg=='num_stages' and isinstance(n.value,ast.Constant) and n.value.value==3 for n in ast.walk(fn)), 'prepare stages differ'
backend=ast.parse((root/'ttft_gdn_prepare/gdn_backend.py').read_text())
calls=[n for n in ast.walk(backend) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fused_gdn_prefill_prepare']
assert len(calls)==1 and not any(k.arg=='num_warps' for k in calls[0].keywords), 'backend must use w4 default'
installer=ast.parse((root/'ttft_mlp_pipeline/current_stream/install.py').read_text())
expected=next(ast.literal_eval(n.value) for n in installer.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
actual=hashlib.sha256((root/'ttft_mlp_pipeline/mlp_pipeline.py').read_bytes()).hexdigest()
assert actual==expected, 'private-stream intermediate no longer matches current-stream repair SHA dependency'
print('CPU preflight PASS: BASE_IMAGE overrides, prepare w4/stages3, current-stream parent SHA',file=sys.stderr)
PY

run() {
  printf '%q ' "$@"
  printf '\n'
  if (( ! dry_run )); then "$@"; fi
}
build_layer() {
  local parent=$1 tag=$2 context=$3 dockerfile=${4:-Dockerfile}
  run docker build --build-arg "BASE_IMAGE=$parent" -t "$tag" -f "$patch_dir/$context/$dockerfile" "$patch_dir/$context"
}
image() { printf '%s:ttft-%s-%s' "$image_prefix" "$1" "$tag_suffix"; }
if [[ "$start_from" == native ]]; then
  build_layer "$native_image" "$issue1_image" gdn_kkt_solve
fi
parent=$issue1_image
for stage in fa packed gdn-views gdn-prepare-w4 kv-pack metadata-pack mlp-pipeline mlp-current-stream current-gemm; do
  tag=$(image "$stage")
  case "$stage" in
    fa) build_layer "$parent" "$tag" ttft_prefill ;;
    packed) build_layer "$parent" "$tag" ttft_prefill Dockerfile.packed ;;
    gdn-views) build_layer "$parent" "$tag" ttft_gdn_views ;;
    gdn-prepare-w4) build_layer "$parent" "$tag" ttft_gdn_prepare ;;
    kv-pack) build_layer "$parent" "$tag" ttft_kv_pack ;;
    metadata-pack) build_layer "$parent" "$tag" ttft_gdn_metadata ;;
    mlp-pipeline) build_layer "$parent" "$tag" ttft_mlp_pipeline ;;
    mlp-current-stream) build_layer "$parent" "$tag" ttft_mlp_pipeline/current_stream ;;
    current-gemm) build_layer "$parent" "$tag" ttft_gemm_dispatch ;;
  esac
  parent=$tag
done
if [[ "$target" == prefill-release || "$target" == qk-gate || "$target" == postops ]]; then
  tag=$(image prefill-release)
  build_layer "$parent" "$tag" ttft_prefill_release
  parent=$tag
fi
if [[ "$target" == qk-gate || "$target" == postops ]]; then
  tag=$(image current-parts)
  build_layer "$parent" "$tag" ttft_mlp_pipeline/current_stream_parts
  parent=$tag
  tag=$(image qk-gate)
  build_layer "$parent" "$tag" ttft_qk_norm_gate
  parent=$tag
fi
if [[ "$target" == postops ]]; then
  tag=$(image numa4)
  build_layer "$parent" "$tag" ttft_numa
  parent=$tag
  tag=$(image fa-gate)
  build_layer "$parent" "$tag" ttft_fa_gate
  parent=$tag
  tag=$(image postops)
  build_layer "$parent" "$tag" ttft_gemm_dispatch_64k/install_extension
  parent=$tag
fi
printf 'Final candidate: %s\n' "$parent" >&2
