#!/usr/bin/env bash
# One public entry point for the five release-frozen stopped OpenRouter routes.
# Usage (the API key value is never an argument):
#   bash   script/run_remaining_openrouter_benchmarks.sh API_KEY_ENV PARTITION
#   sbatch script/run_remaining_openrouter_benchmarks.sh API_KEY_ENV PARTITION

set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 API_KEY_ENVIRONMENT_NAME SLURM_PARTITION" >&2
  exit 2
fi

API_KEY_ENV_NAME=$1
PARTITION_NAME=$2
if [[ ! ${API_KEY_ENV_NAME} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "The API key environment-variable name is malformed." >&2
  exit 2
fi
if [[ ${API_KEY_ENV_NAME} == HF_TOKEN || ${API_KEY_ENV_NAME} == HUGGINGFACE_TOKEN || ${API_KEY_ENV_NAME} == HUGGINGFACE_HUB_TOKEN ]]; then
  echo "A Hugging Face credential cannot be used as the OpenRouter key." >&2
  exit 2
fi
if [[ ! ${PARTITION_NAME} =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "The Slurm partition name is malformed." >&2
  exit 2
fi
if [[ -z ${!API_KEY_ENV_NAME:-} ]]; then
  echo "The named API key environment variable is empty: ${API_KEY_ENV_NAME}" >&2
  exit 2
fi
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  # Slurm executes a private spool copy, so BASH_SOURCE no longer identifies
  # the clone.  The documented sbatch form must be launched from the clone
  # root, which Slurm records without exposing any credential.
  if [[ -z ${SLURM_SUBMIT_DIR:-} || ${SLURM_SUBMIT_DIR} != /* ]]; then
    echo "SLURM_SUBMIT_DIR is required for a scheduled controller." >&2
    exit 2
  fi
  PROJECT_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)
  SCRIPT_DIRECTORY=${PROJECT_ROOT}/script
  PROJECT_CONTROLLER=${SCRIPT_DIRECTORY}/run_remaining_openrouter_benchmarks.sh
  if [[ ! -f ${PROJECT_CONTROLLER} ]] || ! cmp -s -- "${BASH_SOURCE[0]}" "${PROJECT_CONTROLLER}"; then
    echo "Run sbatch from the root of the exact clean HSLE clone." >&2
    exit 2
  fi
else
  SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
  PROJECT_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)
fi
if [[ ${PROJECT_ROOT} == *","* || ${PROJECT_ROOT} == *$'\n'* || ${PROJECT_ROOT} == *$'\r'* ]]; then
  echo "The project path cannot contain a comma or newline." >&2
  exit 2
fi
HF_DATASET_REPO_ID=shashankskagnihotri/humanitys-second-last-exam
HF_DATASET_ARCHIVE=openrouter/hsle_public_openrouter_resume_v2.tar.gz
HF_DATASET_ARCHIVE_SHA256=226ba161608182f37ee4310bd8d3cb32457604603f272ad3a88ee5d0666ecd23
HF_DATASET_REVISION=6861ef237eb9501b8fda3d4fe61788154e143c22
INPUT_DOWNLOAD_REQUIRED=0
if [[ -n ${HSLE_INPUT_ROOT:-} ]]; then
  INPUT_ROOT=${HSLE_INPUT_ROOT}
else
  if [[ -n ${XDG_CACHE_HOME:-} && ${XDG_CACHE_HOME} == /* ]]; then
    DEFAULT_INPUT_ROOT=${XDG_CACHE_HOME}/hsle/public-openrouter-resume-v2
  else
    DEFAULT_INPUT_ROOT=${HOME:?HOME is required}/.cache/hsle/public-openrouter-resume-v2
  fi
  if [[ -d ${DEFAULT_INPUT_ROOT} ]]; then
    INPUT_ROOT=${DEFAULT_INPUT_ROOT}
  elif [[ -d ${PROJECT_ROOT}/.hsle_public_resume_inputs_v2 ]]; then
    INPUT_ROOT=${PROJECT_ROOT}/.hsle_public_resume_inputs_v2
  else
    INPUT_ROOT=${DEFAULT_INPUT_ROOT}
    INPUT_DOWNLOAD_REQUIRED=1
  fi
fi
if [[ ${INPUT_ROOT} != /* ]]; then
  echo "HSLE_INPUT_ROOT/cache must be an absolute path." >&2
  exit 2
fi
if [[ ${INPUT_ROOT} == *","* || ${INPUT_ROOT} == *$'\n'* || ${INPUT_ROOT} == *$'\r'* ]]; then
  echo "HSLE_INPUT_ROOT cannot contain a comma or newline." >&2
  exit 2
fi
if [[ ${INPUT_DOWNLOAD_REQUIRED} == 0 && ! -d ${INPUT_ROOT} ]]; then
  echo "HSLE_INPUT_ROOT/cache must name an existing directory." >&2
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required on the submitting cluster." >&2
  exit 2
fi

VENV_ROOT=${PROJECT_ROOT}/.venv-public-openrouter-resume
OUTPUT_ROOT=${PROJECT_ROOT}/needs_to_be_judged
WORKER_SCRIPT=${PROJECT_ROOT}/script/workers/run_public_openrouter_resume_shard.sh
SHARD_COUNT=${HSLE_PUBLIC_RESUME_SHARDS:-8}
ARRAY_LIMIT=${HSLE_PUBLIC_RESUME_ARRAY_LIMIT:-2}

if [[ ! ${SHARD_COUNT} =~ ^[1-9][0-9]*$ ]]; then
  echo "HSLE_PUBLIC_RESUME_SHARDS must be a positive integer." >&2
  exit 2
fi
if [[ ! ${ARRAY_LIMIT} =~ ^[1-9][0-9]*$ ]]; then
  echo "HSLE_PUBLIC_RESUME_ARRAY_LIMIT must be a positive integer." >&2
  exit 2
fi
if [[ ! -x ${WORKER_SCRIPT} ]]; then
  echo "The release worker is absent or not executable: ${WORKER_SCRIPT}" >&2
  exit 2
fi

if [[ ${HSLE_PUBLIC_RESUME_SKIP_INSTALL:-0} != 1 ]]; then
  if [[ ! -x ${VENV_ROOT}/bin/python ]]; then
    env -u "${API_KEY_ENV_NAME}" -u HF_TOKEN -u HUGGINGFACE_TOKEN -u HUGGINGFACE_HUB_TOKEN \
      python3 -m venv "${VENV_ROOT}"
  fi
  env -u "${API_KEY_ENV_NAME}" -u HF_TOKEN -u HUGGINGFACE_TOKEN -u HUGGINGFACE_HUB_TOKEN \
    "${VENV_ROOT}/bin/python" -m pip install --disable-pip-version-check -e "${PROJECT_ROOT}"
elif [[ ! -x ${VENV_ROOT}/bin/python ]]; then
  echo "HSLE_PUBLIC_RESUME_SKIP_INSTALL=1 requires an existing release venv." >&2
  exit 2
fi

PYTHON=${VENV_ROOT}/bin/python
if [[ ${INPUT_DOWNLOAD_REQUIRED} == 1 ]]; then
  env -u "${API_KEY_ENV_NAME}" -u HF_TOKEN -u HUGGINGFACE_TOKEN -u HUGGINGFACE_HUB_TOKEN \
    "${PYTHON}" - \
    "${HF_DATASET_REPO_ID}" \
    "${HF_DATASET_REVISION}" \
    "${HF_DATASET_ARCHIVE}" \
    "${HF_DATASET_ARCHIVE_SHA256}" \
    "${INPUT_ROOT}" <<'PY'
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

from huggingface_hub import hf_hub_download


repo_id, revision, filename, expected_sha256, target_text = sys.argv[1:]
target = Path(target_text)
target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
lock_path = target.parent / f".{target.name}.download.lock"
with lock_path.open("a+b") as lock:
    os.chmod(lock_path, 0o600)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if target.is_dir():
        raise SystemExit(0)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"input-cache target exists but is not a directory: {target}")

    archive = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=False,
        )
    )
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("downloaded HSLE input archive differs from the release SHA-256")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.extract-", dir=target.parent))
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members: list[tuple[tarfile.TarInfo, Path]] = []
            seen: set[Path] = set()
            for member in bundle.getmembers():
                source = PurePosixPath(member.name)
                if source.is_absolute() or ".." in source.parts or "\\" in member.name:
                    raise RuntimeError("HSLE input archive contains an unsafe path")
                relative = Path(*(part for part in source.parts if part not in {"", "."}))
                if not relative.parts:
                    if not member.isdir():
                        raise RuntimeError("HSLE input archive has an invalid root member")
                    continue
                if relative in seen:
                    raise RuntimeError("HSLE input archive contains a duplicate path")
                seen.add(relative)
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError("HSLE input archive contains a non-regular member")
                members.append((member, relative))

            for member, relative in members:
                destination = temporary / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            for member, relative in members:
                if not member.isfile():
                    continue
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source_handle = bundle.extractfile(member)
                if source_handle is None:
                    raise RuntimeError("HSLE input archive regular file cannot be read")
                with source_handle, destination.open("xb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                os.chmod(destination, 0o600)
        if not (temporary / "INPUT_MANIFEST.json").is_file():
            raise RuntimeError("HSLE input archive lacks INPUT_MANIFEST.json")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Downloaded frozen HSLE inputs from {repo_id}@{revision} into {target}")
PY
fi
if [[ ! -d ${INPUT_ROOT} ]]; then
  echo "HSLE_INPUT_ROOT/cache must name an existing directory." >&2
  exit 2
fi
INPUT_ROOT=$(cd -- "${INPUT_ROOT}" && pwd -P)
export HSLE_INPUT_ROOT=${INPUT_ROOT}

"${PYTHON}" -m hsle.public_openrouter_resume prepare \
  --project-root "${PROJECT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --api-key-env "${API_KEY_ENV_NAME}" \
  --partition "${PARTITION_NAME}" \
  --shard-count "${SHARD_COUNT}"

# HF_TOKEN is accepted only by the optional prepare-time official-source
# validation.  It and alternate Hugging Face credentials cannot reach Slurm.
unset HF_TOKEN HUGGINGFACE_TOKEN HUGGINGFACE_HUB_TOKEN

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/control"
declare -a ROUTES=(kimi_k2_thinking kimi_k25 kimi_k26 kimi_k3 qwen38_max)
declare -a JOB_IDS=()
for ROUTE in "${ROUTES[@]}"; do
  declare -a SUBMIT=(
    sbatch
    --parsable
    --partition "${PARTITION_NAME}"
    --job-name "hsle-or-${ROUTE}"
    --array "0-$((SHARD_COUNT - 1))%${ARRAY_LIMIT}"
    --cpus-per-task 1
    --mem 8G
    --time 2-00:00:00
    --output "${OUTPUT_ROOT}/logs/%x-%A_%a.out"
    --error "${OUTPUT_ROOT}/logs/%x-%A_%a.err"
    --export "${API_KEY_ENV_NAME},PATH=/usr/bin:/bin,HSLE_INPUT_ROOT=${INPUT_ROOT},HSLE_PUBLIC_API_KEY_ENV_NAME=${API_KEY_ENV_NAME},HSLE_PUBLIC_PROJECT_ROOT=${PROJECT_ROOT},HSLE_PUBLIC_OUTPUT_ROOT=${OUTPUT_ROOT},HSLE_PUBLIC_ROUTE=${ROUTE},HSLE_PUBLIC_SHARD_COUNT=${SHARD_COUNT}"
    "${WORKER_SCRIPT}"
    worker
  )
  if [[ ${HSLE_PUBLIC_RESUME_DRY_RUN:-0} == 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${SUBMIT[@]}"
    printf '\n'
    JOB_IDS+=("dry-run-${ROUTE}")
  else
    JOB_ID=$("${SUBMIT[@]}")
    JOB_ID=${JOB_ID%%;*}
    if [[ ! ${JOB_ID} =~ ^[0-9]+$ ]]; then
      echo "sbatch returned an invalid job id for ${ROUTE}." >&2
      exit 1
    fi
    JOB_IDS+=("${JOB_ID}")
    echo "Submitted ${ROUTE}: ${JOB_ID}"
  fi
done

if [[ ${HSLE_PUBLIC_RESUME_DRY_RUN:-0} == 1 ]]; then
  echo "DRY RUN: finalizer would wait for all five arrays."
  exit 0
fi

DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
FINALIZER_JOB_ID=$(sbatch \
  --parsable \
  --partition "${PARTITION_NAME}" \
  --job-name hsle-or-finalize \
  --dependency "afterany:${DEPENDENCY}" \
  --cpus-per-task 1 \
  --mem 4G \
  --time 02:00:00 \
  --output "${OUTPUT_ROOT}/logs/%x-%j.out" \
  --error "${OUTPUT_ROOT}/logs/%x-%j.err" \
  --export "PATH=/usr/bin:/bin,HSLE_INPUT_ROOT=${INPUT_ROOT},HSLE_PUBLIC_PROJECT_ROOT=${PROJECT_ROOT},HSLE_PUBLIC_OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "${WORKER_SCRIPT}" finalize)
FINALIZER_JOB_ID=${FINALIZER_JOB_ID%%;*}
if [[ ! ${FINALIZER_JOB_ID} =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an invalid finalizer job id." >&2
  exit 1
fi
echo "Submitted transfer-manifest finalizer: ${FINALIZER_JOB_ID}"
echo "Results will be collected in: ${OUTPUT_ROOT}"
