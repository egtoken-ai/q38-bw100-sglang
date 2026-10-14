#!/usr/bin/env bash
set -euo pipefail

# Run on the BW100 host. The build context contains only this patch directory.
PATCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_CONTAINER=${SOURCE_CONTAINER:-sglang-bw100}
IMAGE=${IMAGE:-sglang-bw100:gdn-kkt-solve-issue1}
VERIFY_CONTAINER=${VERIFY_CONTAINER:-sglang-bw100-gdn-verify}
DEVICE=${DEVICE:-0}
RESULT_DIR=${RESULT_DIR:-/root/gdn-kkt-solve-validation/results}
DRIVER_DIR=${DRIVER_DIR:-/opt/hyhal}
BASE_IMAGE=$(docker inspect "$SOURCE_CONTAINER" --format '{{.Config.Image}}')
SOURCE_IMAGE_ID=$(docker inspect "$SOURCE_CONTAINER" --format '{{.Image}}')
TAG_IMAGE_ID=$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')
test "$SOURCE_IMAGE_ID" = "$TAG_IMAGE_ID" || {
  echo 'Source container and base image tag differ; pin the correct image first.' >&2
  exit 1
}
BASE_SOURCE=/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/fla/chunk_fwd.py
EXPECTED=76680de79e5b078e2321fc243382285af8396c91ab47bafa7be0a0738ad91a1d
ACTUAL=$(docker exec "$SOURCE_CONTAINER" sha256sum "$BASE_SOURCE")
test "${ACTUAL%% *}" = "$EXPECTED" || {
  echo 'Source container fingerprint differs from the audited baseline.' >&2
  exit 1
}
mkdir -p "$RESULT_DIR"
docker build --pull=false --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" "$PATCH_DIR"
docker image inspect "$IMAGE" > "$RESULT_DIR/image-inspect.json"
docker run --rm --name "$VERIFY_CONTAINER" \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  --group-add video --security-opt seccomp=unconfined --shm-size 2g \
  -e "HIP_VISIBLE_DEVICES=$DEVICE" \
  -e TRITON_CACHE_DIR=/results/triton-cache \
  -v "$DRIVER_DIR:/opt/hyhal:ro" -v "$RESULT_DIR:/results" \
  --entrypoint python3 "$IMAGE" /opt/gdn-kkt-solve/verify.py \
  --output /results/validation.json "$@" 2>&1 | tee "$RESULT_DIR/validation.log"
