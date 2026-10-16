#!/usr/bin/env bash
# Build release candidate image directly from Issue 1 or vendor base.
set -euo pipefail

patch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
start_from=issue1
image_prefix=sglang-bw100
tag_suffix=issue3
issue1_image=''
native_image=harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10
dry_run=0

usage() {
  cat <<'HELP'
Usage: bash patch/build_ttft_candidate.sh [options]
  --dry-run                    Print build commands only without building
  --from issue1|native         Start with Issue 1 image or vendor base (default: issue1)
  --image-prefix NAME          Output repository (default: sglang-bw100)
  --tag-suffix TEXT            Image tag suffix (default: issue3)
  --issue1-image IMAGE         Issue 1 image tag (default: PREFIX:gdn-kkt-solve-issue1)
  --native-image IMAGE         Vendor base image
  --help                       Show this help
HELP
}

while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --from|--image-prefix|--tag-suffix|--issue1-image|--native-image)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      option=$1; value=$2; shift 2
      case "$option" in
        --from) start_from=$value ;;
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
issue1_image=${issue1_image:-"${image_prefix}:gdn-kkt-solve-issue1"}
final_image="${image_prefix}:ttft-combo-${tag_suffix}"

run() {
  printf '%q ' "$@"
  printf '\n'
  if (( ! dry_run )); then "$@"; fi
}

if [[ "$start_from" == native ]]; then
  echo ">>> [Stage 1/2] Building Issue 1 GDN KKT solver layer from native image..."
  run docker build -t "$issue1_image" -f "$patch_dir/gdn_kkt_solve/Dockerfile" "$patch_dir/gdn_kkt_solve"
fi

echo ">>> [Stage 2/2] Building consolidated TTFT Combo release image from $issue1_image..."
run docker build --build-arg "BASE_IMAGE=$issue1_image" -t "$final_image" -f "$patch_dir/ttft_combo/Dockerfile" "$patch_dir/ttft_combo"

printf '\nFinal candidate successfully built: %s\n' "$final_image" >&2
