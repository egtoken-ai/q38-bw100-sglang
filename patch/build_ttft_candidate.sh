#!/usr/bin/env bash
# Build release candidate image directly from vendor base.
set -euo pipefail

patch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image_prefix=sglang-bw100
tag_suffix=issue3
native_image=harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10
dry_run=0

usage() {
  cat <<'HELP'
Usage: bash patch/build_ttft_candidate.sh [options]
  --dry-run                    Print build commands only without building
  --image-prefix NAME          Output repository (default: sglang-bw100)
  --tag-suffix TEXT            Image tag suffix (default: issue3)
  --native-image IMAGE         Vendor base image
  --help                       Show this help
HELP
}

while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --image-prefix|--tag-suffix|--native-image)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      option=$1; value=$2; shift 2
      case "$option" in
        --image-prefix) image_prefix=$value ;;
        --tag-suffix) tag_suffix=$value ;;
        --native-image) native_image=$value ;;
      esac ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

final_image="${image_prefix}:ttft-combo-${tag_suffix}"

run() {
  printf '%q ' "$@"
  printf '\n'
  if (( ! dry_run )); then "$@"; fi
}

echo ">>> Building consolidated release candidate image directly from native base..."
run docker build --build-arg "BASE_IMAGE=$native_image" -t "$final_image" -f "$patch_dir/ttft_combo/Dockerfile" "$patch_dir/ttft_combo"

printf '\nFinal candidate successfully built: %s\n' "$final_image" >&2
