#!/usr/bin/env bash
# Submit one pinned HSLE local-model preparation/generation pair on Helix.
set -euo pipefail
umask 077

readonly ROUTE="${1:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly REPO_ROOT="${repo_root}"

distributed_handoff() {
  local minimum_nodes model_contract runtime_contract
  case "${ROUTE}" in
    kimi_k3)
      minimum_nodes=2
      model_contract='moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569; 1,560,936,091,448 checkpoint bytes; at least 16 H200 GPUs'
      runtime_contract='official multi-node CUDA 13 container/nightly recipe with driver R580+'
      ;;
    *) return 0 ;;
  esac

  echo "Distributed Helix handoff required for ${ROUTE}." >&2
  echo "Pinned model: ${model_contract}." >&2
  echo "Runtime contract: ${runtime_contract}." >&2
  echo "The one-node gpu-single launcher will not be used for this route." >&2
  echo "A site-specific Helix implementation must provide all of:" >&2
  echo "  HSLE_HELIX_MULTINODE_OPT_IN=I_ACKNOWLEDGE_DISTRIBUTED_CAPACITY" >&2
  echo "  HSLE_HELIX_MULTINODE_PARTITION=<site multi-node GPU partition>" >&2
  echo "  HSLE_HELIX_MULTINODE_NODES>=${minimum_nodes}" >&2
  echo "  HSLE_HELIX_GPUS_PER_NODE=8" >&2
  echo "  HSLE_HELIX_VLLM_RUNTIME=<pinned official runtime/container>" >&2
  echo "See docs/HELIX_LOCAL.md for rank orchestration and scientific invariants." >&2

  if [[ "${HSLE_HELIX_MULTINODE_OPT_IN:-}" == "I_ACKNOWLEDGE_DISTRIBUTED_CAPACITY" ]]; then
    [[ -n "${HSLE_HELIX_MULTINODE_PARTITION:-}" ]] || {
      echo "Missing HSLE_HELIX_MULTINODE_PARTITION." >&2
      exit 64
    }
    [[ "${HSLE_HELIX_MULTINODE_NODES:-}" =~ ^[0-9]+$ ]] \
      && (( HSLE_HELIX_MULTINODE_NODES >= minimum_nodes )) || {
      echo "HSLE_HELIX_MULTINODE_NODES is below the verified capacity floor." >&2
      exit 64
    }
    [[ "${HSLE_HELIX_GPUS_PER_NODE:-}" == "8" ]] || {
      echo "HSLE_HELIX_GPUS_PER_NODE must be exactly 8 for this handoff." >&2
      exit 64
    }
    [[ -n "${HSLE_HELIX_VLLM_RUNTIME:-}" ]] || {
      echo "Missing HSLE_HELIX_VLLM_RUNTIME pin." >&2
      exit 64
    }
    echo "Distributed configuration acknowledged. Automatic submission remains guarded until the site-specific multi-node worker is implemented and audited." >&2
  fi
  exit 78
}

distributed_handoff
case "${ROUTE}" in
  kimi_k26|kimi_k25|kimi_k2_thinking|qwen38_27b) ;;
  *) echo "Unknown Helix route: ${ROUTE:-<blank>}" >&2; exit 64 ;;
esac

command -v sbatch >/dev/null || { echo "sbatch is required on the Helix login node" >&2; exit 69; }
: "${HSLE_GEMINI_KEY_FILE:?Set HSLE_GEMINI_KEY_FILE to a user-owned mode-0600 key file for inline LFE feedback}"
[[ -f "${HSLE_GEMINI_KEY_FILE}" && ! -L "${HSLE_GEMINI_KEY_FILE}" ]] || {
  echo "HSLE_GEMINI_KEY_FILE must be a regular non-symlink file" >&2
  exit 66
}
key_directory="$(cd "$(dirname "${HSLE_GEMINI_KEY_FILE}")" && pwd -P)"
readonly key_directory
key_basename="$(basename "${HSLE_GEMINI_KEY_FILE}")"
readonly key_basename
readonly GEMINI_KEY_FILE="${key_directory}/${key_basename}"
key_owner="$(stat -c '%u' -- "${GEMINI_KEY_FILE}")"
readonly key_owner
key_mode="$(stat -c '%a' -- "${GEMINI_KEY_FILE}")"
readonly key_mode
if [[ "${key_owner}" != "$(id -u)" || "${key_mode}" != "600" ]]; then
  echo "HSLE_GEMINI_KEY_FILE must be owned by the submitting user with exact mode 0600" >&2
  exit 77
fi
awk 'NR == 1 { if ($0 ~ /^[[:space:]]*$/) exit 1; next } { exit 1 } END { if (NR != 1) exit 1 }' \
  "${GEMINI_KEY_FILE}" || {
  echo "HSLE_GEMINI_KEY_FILE must contain exactly one nonblank line" >&2
  exit 65
}

readonly PYTHON_MODULE="${HSLE_HELIX_PYTHON_MODULE:-devel/python}"
readonly CUDA_MODULE="${HSLE_HELIX_CUDA_MODULE:-devel/cuda}"
readonly CONCURRENCY="${HSLE_HELIX_CONCURRENCY:-16}"
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && (( CONCURRENCY >= 1 && CONCURRENCY <= 64 )) || {
  echo "HSLE_HELIX_CONCURRENCY must be an integer between 1 and 64" >&2
  exit 64
}

workspace="${HSLE_HELIX_WORKSPACE:-}"
if [[ -n "${workspace}" ]]; then
  [[ -d "${workspace}" ]] || { echo "HSLE_HELIX_WORKSPACE is not a directory" >&2; exit 72; }
  workspace="$(cd "${workspace}" && pwd -P)"
else
  command -v ws_find >/dev/null || { echo "ws_find is required unless HSLE_HELIX_WORKSPACE is set" >&2; exit 69; }
  command -v ws_allocate >/dev/null || { echo "ws_allocate is required unless HSLE_HELIX_WORKSPACE is set" >&2; exit 69; }
  readonly workspace_name="${HSLE_HELIX_WORKSPACE_NAME:-hsle-benchmark-${USER:-user}}"
  workspace="$(ws_find "${workspace_name}" 2>/dev/null || true)"
  if [[ ! -d "${workspace}" ]]; then
    # Helix workspaces provide the official 10-TB workspace-backed storage class.
    workspace="$(ws_allocate "${workspace_name}" 30)"
  fi
  [[ -d "${workspace}" ]] || {
    echo "Could not resolve the Helix workspace path; set HSLE_HELIX_WORKSPACE explicitly" >&2
    exit 72
  }
  workspace="$(cd "${workspace}" && pwd -P)"
fi

readonly RUN_ROOT="${workspace}/hsle-local-benchmark"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/control" "${RUN_ROOT}/need_to_be_judged/${ROUTE}"

readonly PREP_WORKER="${REPO_ROOT}/script/workers/helix_prepare_local_model.sh"
readonly GPU_WORKER="${REPO_ROOT}/script/workers/helix_generate_local_model.sh"
[[ -f "${PREP_WORKER}" && -f "${GPU_WORKER}" ]] || {
  echo "Helix worker scripts are missing from the checkout" >&2
  exit 66
}

account_args=()
if [[ -n "${HSLE_HELIX_ACCOUNT:-}" ]]; then
  account_args=(--account="${HSLE_HELIX_ACCOUNT}")
fi

prep_job_id="$(sbatch --parsable \
  --job-name="hsle-prep-${ROUTE}" \
  --partition="${HSLE_HELIX_CPU_PARTITION:-cpu-single}" \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G \
  --time=120:00:00 --no-requeue --export=NONE \
  --output="${RUN_ROOT}/logs/%x_%j.out" \
  --error="${RUN_ROOT}/logs/%x_%j.err" \
  "${account_args[@]}" \
  "${PREP_WORKER}" "${ROUTE}" "${REPO_ROOT}" "${RUN_ROOT}" "${PYTHON_MODULE}")"
[[ "${prep_job_id}" =~ ^[0-9]+$ ]] || { echo "Unexpected preparation job ID: ${prep_job_id}" >&2; exit 70; }

gpu_job_id="$(sbatch --parsable \
  --job-name="hsle-gen-${ROUTE}" \
  --partition=gpu-single \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=2100G \
  --gres=gpu:H200:8 --time=120:00:00 --exclusive --no-requeue --export=NONE \
  --dependency="afterok:${prep_job_id}" --kill-on-invalid-dep=yes \
  --output="${RUN_ROOT}/logs/%x_%j.out" \
  --error="${RUN_ROOT}/logs/%x_%j.err" \
  "${account_args[@]}" \
  "${GPU_WORKER}" "${ROUTE}" "${REPO_ROOT}" "${RUN_ROOT}" \
  "${GEMINI_KEY_FILE}" "${PYTHON_MODULE}" "${CUDA_MODULE}" "${CONCURRENCY}")"
[[ "${gpu_job_id}" =~ ^[0-9]+$ ]] || { echo "Unexpected generation job ID: ${gpu_job_id}" >&2; exit 70; }

printf 'Preparation job: %s\n' "${prep_job_id}"
printf 'Dependent H200 generation job: %s\n' "${gpu_job_id}"
printf 'Workspace: %s\n' "${RUN_ROOT}"
printf 'Results: %s\n' "${RUN_ROOT}/need_to_be_judged/${ROUTE}"
