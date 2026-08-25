#!/usr/bin/env bash

set -euo pipefail
umask 077

MODE=${1:-}
PROJECT_ROOT=${HSLE_PUBLIC_PROJECT_ROOT:?HSLE_PUBLIC_PROJECT_ROOT is required}
OUTPUT_ROOT=${HSLE_PUBLIC_OUTPUT_ROOT:?HSLE_PUBLIC_OUTPUT_ROOT is required}
INPUT_ROOT=${HSLE_INPUT_ROOT:?HSLE_INPUT_ROOT is required}
PYTHON=${PROJECT_ROOT}/.venv-openrouter-benchmark/bin/python
if [[ ! -x ${PYTHON} ]]; then
  echo "The OpenRouter benchmark virtual environment is absent." >&2
  exit 2
fi
if [[ ${MODE} == worker ]]; then
  API_KEY_ENV_NAME=${HSLE_PUBLIC_API_KEY_ENV_NAME:?HSLE_PUBLIC_API_KEY_ENV_NAME is required}
  ROUTE=${HSLE_PUBLIC_ROUTE:?HSLE_PUBLIC_ROUTE is required}
  SHARD_COUNT=${HSLE_PUBLIC_SHARD_COUNT:?HSLE_PUBLIC_SHARD_COUNT is required}
  SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
  if [[ ${API_KEY_ENV_NAME} == HF_TOKEN || ${API_KEY_ENV_NAME} == HUGGINGFACE_TOKEN || ${API_KEY_ENV_NAME} == HUGGINGFACE_HUB_TOKEN ]]; then
    echo "A Hugging Face credential cannot be used as the OpenRouter key." >&2
    exit 2
  fi
  if [[ ! ${API_KEY_ENV_NAME} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || [[ -z ${!API_KEY_ENV_NAME:-} ]]; then
    echo "The named OpenRouter credential is unavailable to this Slurm task." >&2
    exit 2
  fi
  while IFS= read -r ENVIRONMENT_NAME; do
    if [[ ${ENVIRONMENT_NAME} != "${API_KEY_ENV_NAME}" ]] && \
       [[ ${ENVIRONMENT_NAME} =~ (_API_KEY|_TOKEN)$ || ${ENVIRONMENT_NAME} == HF_TOKEN || ${ENVIRONMENT_NAME} == HUGGINGFACE_TOKEN || ${ENVIRONMENT_NAME} == HUGGINGFACE_HUB_TOKEN ]]; then
      echo "An alternate credential survived the explicit Slurm export: ${ENVIRONMENT_NAME}" >&2
      exit 2
    fi
  done < <(compgen -e)
  exec "${PYTHON}" -m hsle.public_openrouter_resume worker \
    --project-root "${PROJECT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --api-key-env "${API_KEY_ENV_NAME}" \
    --route "${ROUTE}" \
    --shard-count "${SHARD_COUNT}" \
    --shard-index "${SHARD_INDEX}"
elif [[ ${MODE} == finalize ]]; then
  ROUTE_SELECTION=${HSLE_PUBLIC_ROUTE_SELECTION:-all}
  exec "${PYTHON}" -m hsle.public_openrouter_resume finalize \
    --project-root "${PROJECT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --route-selection "${ROUTE_SELECTION}"
else
  echo "Worker mode must be 'worker' or 'finalize'." >&2
  exit 2
fi
