#!/bin/bash
# CPU-only preparation worker. This script is submitted; it is not run on login.
set -euo pipefail
umask 077
ulimit -c 0

[[ "$#" -eq 4 ]] || { echo "usage: helix_prepare_local_model.sh ROUTE REPO_ROOT RUN_ROOT PYTHON_MODULE" >&2; exit 64; }
readonly ROUTE="$1"
readonly REPO_ROOT="$2"
readonly RUN_ROOT="$3"
readonly PYTHON_MODULE="$4"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "preparation must run inside Slurm" >&2; exit 72; }
if [[ "${ROUTE}" == "kimi_k3" ]]; then
  echo "Kimi K3 preparation is blocked: the exact official multi-node CUDA 13 container runtime is not implemented for Helix" >&2
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

case "${ROUTE}" in
  kimi_k26|kimi_k25|kimi_k2_thinking|qwen38_27b) readonly VLLM_VERSION=0.25.1 ;;
  *) echo "unsupported preparation route: ${ROUTE}" >&2; exit 64 ;;
esac

readonly VENV="${RUN_ROOT}/envs/${ROUTE}-vllm-${VLLM_VERSION}"
mkdir -p "${RUN_ROOT}/envs" "${RUN_ROOT}/cache/${ROUTE}" "${RUN_ROOT}/control/${ROUTE}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
readonly PYTHON="${VENV}/bin/python"
readonly PIP="${VENV}/bin/pip"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${RUN_ROOT}/cache/${ROUTE}/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN

runtime_packages=(
  "vllm==${VLLM_VERSION}"
  "huggingface-hub==1.4.1"
  "pandas==2.2.3"
  "PyYAML==6.0.2"
)
if [[ "${ROUTE}" == "qwen38_27b" ]]; then
  runtime_packages+=("transformers>=5.8.0")
fi
"${PIP}" install --disable-pip-version-check "${runtime_packages[@]}"
"${PIP}" install --disable-pip-version-check --no-deps --editable "${REPO_ROOT}"
"${PYTHON}" -m hsle.helix_local prepare \
  --route "${ROUTE}" --repo-root "${REPO_ROOT}" --workspace "${RUN_ROOT}"

echo "Pinned Helix preparation complete for ${ROUTE}"
