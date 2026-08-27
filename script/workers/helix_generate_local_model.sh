#!/bin/bash
# Exclusive one-node TP8 H200 generation worker.
set -euo pipefail
umask 077
ulimit -c 0

[[ "$#" -eq 7 ]] || { echo "usage: helix_generate_local_model.sh ROUTE REPO_ROOT RUN_ROOT GEMINI_KEY_FILE PYTHON_MODULE CUDA_MODULE CONCURRENCY" >&2; exit 64; }
readonly ROUTE="$1"
readonly REPO_ROOT="$2"
readonly RUN_ROOT="$3"
readonly GEMINI_KEY_FILE="$4"
readonly PYTHON_MODULE="$5"
readonly CUDA_MODULE="$6"
readonly CONCURRENCY="$7"
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-single" ]] || { echo "generation requires partition gpu-single" >&2; exit 72; }
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || { echo "generation requires exactly one node" >&2; exit 72; }
[[ "${SLURM_CPUS_PER_TASK:-}" == "64" ]] || { echo "generation requires exactly 64 CPUs" >&2; exit 72; }
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || { echo "requeued jobs are forbidden" >&2; exit 72; }
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && (( CONCURRENCY >= 1 && CONCURRENCY <= 64 )) || {
  echo "coordinate concurrency must be an integer between 1 and 64" >&2
  exit 64
}
[[ -f "${GEMINI_KEY_FILE}" && ! -L "${GEMINI_KEY_FILE}" ]] || {
  echo "Gemini key file is absent, unsafe, or not visible on the compute node" >&2
  exit 66
}
key_owner="$(stat -c '%u' -- "${GEMINI_KEY_FILE}")"
readonly key_owner
key_mode="$(stat -c '%a' -- "${GEMINI_KEY_FILE}")"
readonly key_mode
if [[ "${key_owner}" != "$(id -u)" || "${key_mode}" != "600" ]]; then
  echo "Gemini key file must be owned by the job user with exact mode 0600" >&2
  exit 77
fi
if [[ "${ROUTE}" == "kimi_k3" ]]; then
  echo "Kimi K3 generation is blocked: one eight-H200 node cannot satisfy the official runtime footprint" >&2
  exit 65
fi

if ! type module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -r "${module_init}" ]]; then
      # shellcheck source=/dev/null
      source "${module_init}"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || { echo "environment modules are unavailable" >&2; exit 69; }

module --force purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"

case "${ROUTE}" in
  kimi_k26|kimi_k25|kimi_k2_thinking|qwen38_27b) readonly VLLM_VERSION=0.25.1 ;;
  *) echo "unsupported generation route: ${ROUTE}" >&2; exit 64 ;;
esac
readonly VENV="${RUN_ROOT}/envs/${ROUTE}-vllm-${VLLM_VERSION}"
readonly PYTHON="${VENV}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "prepared venv is absent" >&2; exit 66; }

# Hash every checkpoint shard and compare it with the signed preparation
# authority before vLLM can load or serve the local snapshot.
"${PYTHON}" -m hsle.helix_local verify-prepared \
  --route "${ROUTE}" --workspace "${RUN_ROOT}"

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "${gpu_count}" -eq 8 ]] || { echo "exactly eight visible GPUs are required" >&2; exit 72; }
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -v 'H200' >/dev/null; then
  echo "all eight allocated GPUs must be H200" >&2
  exit 72
fi

readarray -t model_meta < <("${PYTHON}" - "${RUN_ROOT}/control/${ROUTE}/preparation.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["model_id"])
print(value["model_revision"])
print(value["model"]["root"])
print(value["model_modality"])
PY
)
[[ "${#model_meta[@]}" -eq 4 ]] || { echo "preparation model metadata differs" >&2; exit 65; }
readonly MODEL_ID="${model_meta[0]}"
readonly MODEL_REVISION="${model_meta[1]}"
readonly MODEL_ROOT="${model_meta[2]}"
readonly MODEL_MODALITY="${model_meta[3]}"
[[ -d "${MODEL_ROOT}" ]] || { echo "pinned model snapshot is absent" >&2; exit 66; }

readonly CACHE_ROOT="${RUN_ROOT}/cache/${ROUTE}/runtime-${SLURM_JOB_ID}"
mkdir -p "${CACHE_ROOT}"/{vllm,triton,torch,xdg} "${RUN_ROOT}/logs" "${RUN_ROOT}/need_to_be_judged/${ROUTE}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HSLE_GEMINI_KEY_FILE="${GEMINI_KEY_FILE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${RUN_ROOT}/cache/${ROUTE}/huggingface"
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torch" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN

server_args=(
  --model "${MODEL_ROOT}"
  --tokenizer "${MODEL_ROOT}"
  --revision "${MODEL_REVISION}"
  --tokenizer-revision "${MODEL_REVISION}"
  --served-model-name "${MODEL_ID}"
  --host 127.0.0.1 --port 8000
  --tensor-parallel-size 8
  --distributed-executor-backend mp
  --dtype bfloat16
  --gpu-memory-utilization 0.94
  --max-model-len 32768
  --max-num-seqs 8
  --max-num-batched-tokens 32768
  --disable-custom-all-reduce
  --trust-remote-code
  --seed 0
  --no-enable-log-requests
)
if [[ "${ROUTE}" == "qwen38_27b" ]]; then
  server_args+=(--reasoning-parser qwen3)
else
  server_args+=(--reasoning-parser kimi_k2)
fi
if [[ "${MODEL_MODALITY}" == "multimodal" ]]; then
  # After data/image -> data/images normalization and byte-alias deduplication,
  # each target/example has at most one image: three in two-shot/LFE history.
  server_args+=(
    --limit-mm-per-prompt '{"image":3}'
    --mm-encoder-tp-mode data
  )
fi
readonly SERVER_OUT="${RUN_ROOT}/logs/vllm-${ROUTE}-${SLURM_JOB_ID}.out"
readonly SERVER_ERR="${RUN_ROOT}/logs/vllm-${ROUTE}-${SLURM_JOB_ID}.err"
"${PYTHON}" -m vllm.entrypoints.openai.api_server "${server_args[@]}" \
  >"${SERVER_OUT}" 2>"${SERVER_ERR}" &
readonly SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 1440); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "local vLLM server exited during model loading; see ${SERVER_ERR}" >&2
    exit 70
  fi
  if "${PYTHON}" -B - <<'PY' >/dev/null 2>&1
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    ready=1
    break
  fi
  sleep 15
done
[[ "${ready}" -eq 1 ]] || { echo "local vLLM server did not become ready within six hours" >&2; exit 70; }

"${PYTHON}" -m hsle.helix_local run \
  --route "${ROUTE}" --workspace "${RUN_ROOT}" --endpoint http://127.0.0.1:8000 \
  --concurrency "${CONCURRENCY}"
