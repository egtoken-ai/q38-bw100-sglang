# Source only in the API client shell; never in a serving process.
q38_client_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export Q38_OC_API_ONLY=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONPATH="${q38_client_dir}/import_guard${PYTHONPATH:+:${PYTHONPATH}}"
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:-},localhost,127.0.0.1,::1"
unset DATASET_SOURCE LOCAL_LLM RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
