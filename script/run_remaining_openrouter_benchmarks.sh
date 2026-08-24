#!/usr/bin/env bash
#SBATCH --no-requeue
# One entry point for the six release-frozen remaining OpenRouter routes.
# Preferred usage (the API key value is never an argument):
#   export HSLE_OPENROUTER_KEY_ENV=MY_OPENROUTER_KEY
#   export HSLE_SLURM_PARTITION=PARTITION
#   bash script/run_remaining_openrouter_benchmarks.sh
# Two positional arguments remain supported: API_KEY_ENV PARTITION.

set -euo pipefail
umask 077

if [[ $# -eq 0 ]]; then
  API_KEY_ENV_NAME=${HSLE_OPENROUTER_KEY_ENV:-}
  PARTITION_NAME=${HSLE_SLURM_PARTITION:-}
elif [[ $# -eq 2 ]]; then
  API_KEY_ENV_NAME=$1
  PARTITION_NAME=$2
else
  echo "Usage: $0 [API_KEY_ENVIRONMENT_NAME SLURM_PARTITION]" >&2
  exit 2
fi
if [[ ! ${API_KEY_ENV_NAME} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "The API key environment-variable name is malformed." >&2
  exit 2
fi
case ${API_KEY_ENV_NAME} in
  HF_TOKEN|HUGGINGFACE_TOKEN|HUGGINGFACE_HUB_TOKEN|PATH|HOME|XDG_CACHE_HOME|HSLE_OPENROUTER_KEY_ENV|HSLE_SLURM_PARTITION|HSLE_DATASET_ROOT|HSLE_INPUT_ROOT|HSLE_OUTPUT_ROOT|HSLE_PUBLIC_RESUME_SHARDS|HSLE_PUBLIC_RESUME_SKIP_INSTALL|HSLE_VALIDATE_OFFICIAL_HF|HSLE_PUBLIC_API_KEY_ENV_NAME|HSLE_PUBLIC_PROJECT_ROOT|HSLE_PUBLIC_OUTPUT_ROOT|HSLE_PUBLIC_ROUTE|HSLE_PUBLIC_SHARD_COUNT|SLURM_*|SBATCH_*)
    echo "The OpenRouter key name collides with a reserved controller or Slurm variable." >&2
    exit 2
    ;;
esac
if [[ ! ${PARTITION_NAME} =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "The Slurm partition name is malformed." >&2
  exit 2
fi
if [[ -z ${!API_KEY_ENV_NAME:-} ]]; then
  echo "The named API key environment variable is empty: ${API_KEY_ENV_NAME}" >&2
  exit 2
fi

# Setup and public-data downloads do not need provider credentials. Record only
# environment-variable names here; secret values are never copied into shell
# arguments or written to disk.
declare -a DETECTED_CREDENTIAL_ENVIRONMENT_NAMES=("${API_KEY_ENV_NAME}")
while IFS= read -r ENVIRONMENT_NAME; do
  case ${ENVIRONMENT_NAME^^} in
    *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*)
      DETECTED_CREDENTIAL_ENVIRONMENT_NAMES+=("${ENVIRONMENT_NAME}")
      ;;
  esac
done < <(compgen -e)

run_without_credentials() (
  for ENVIRONMENT_NAME in "${DETECTED_CREDENTIAL_ENVIRONMENT_NAMES[@]}"; do
    unset "${ENVIRONMENT_NAME}"
  done
  exec "$@"
)

run_with_openrouter_and_optional_hf_only() (
  for ENVIRONMENT_NAME in "${DETECTED_CREDENTIAL_ENVIRONMENT_NAMES[@]}"; do
    if [[ ${ENVIRONMENT_NAME} != "${API_KEY_ENV_NAME}" && ${ENVIRONMENT_NAME} != HF_TOKEN ]]; then
      unset "${ENVIRONMENT_NAME}"
    fi
  done
  exec "$@"
)
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
HF_DATASET_ARCHIVE=openrouter/hsle_openrouter_single_dispatch_v3.tar.gz
HF_DATASET_ARCHIVE_SHA256=ced06f31b7d82a58db28391f6e9bf09293a88933480f6b5354784ce98d3ede5f
HF_DATASET_REVISION=aeda08b2536a19e698d027fd4f701eea78c9171d
INPUT_DOWNLOAD_REQUIRED=0
if [[ -n ${HSLE_DATASET_ROOT:-} ]]; then
  DATASET_ROOT=${HSLE_DATASET_ROOT}
else
  if [[ -n ${XDG_CACHE_HOME:-} && ${XDG_CACHE_HOME} == /* ]]; then
    DATASET_ROOT=${XDG_CACHE_HOME}/hsle/huggingface-dataset-v3
  else
    DATASET_ROOT=${HOME:?HOME is required}/.cache/hsle/huggingface-dataset-v3
  fi
fi
if [[ ${DATASET_ROOT} != /* ]]; then
  echo "HSLE_DATASET_ROOT/cache must be an absolute path." >&2
  exit 2
fi
if [[ ${DATASET_ROOT} == *","* || ${DATASET_ROOT} == *$'\n'* || ${DATASET_ROOT} == *$'\r'* ]]; then
  echo "HSLE_DATASET_ROOT cannot contain a comma or newline." >&2
  exit 2
fi
if [[ -n ${HSLE_INPUT_ROOT:-} ]]; then
  INPUT_ROOT=${HSLE_INPUT_ROOT}
else
  if [[ -n ${XDG_CACHE_HOME:-} && ${XDG_CACHE_HOME} == /* ]]; then
    DEFAULT_INPUT_ROOT=${XDG_CACHE_HOME}/hsle/openrouter-single-dispatch-v3
  else
    DEFAULT_INPUT_ROOT=${HOME:?HOME is required}/.cache/hsle/openrouter-single-dispatch-v3
  fi
  if [[ -d ${DEFAULT_INPUT_ROOT} ]]; then
    INPUT_ROOT=${DEFAULT_INPUT_ROOT}
  elif [[ -d ${PROJECT_ROOT}/.hsle_openrouter_single_dispatch_inputs_v3 ]]; then
    INPUT_ROOT=${PROJECT_ROOT}/.hsle_openrouter_single_dispatch_inputs_v3
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

for SLURM_COMMAND in sbatch scontrol scancel; do
  if ! command -v "${SLURM_COMMAND}" >/dev/null 2>&1; then
    echo "${SLURM_COMMAND} is required on the submitting cluster." >&2
    exit 2
  fi
done

VENV_ROOT=${PROJECT_ROOT}/.venv-openrouter-benchmark
if [[ -n ${HSLE_OUTPUT_ROOT:-} ]]; then
  OUTPUT_ROOT=${HSLE_OUTPUT_ROOT}
else
  OUTPUT_ROOT=${PROJECT_ROOT}/need_to_be_judged
fi
if [[ ${OUTPUT_ROOT} != /* || ${OUTPUT_ROOT} == *","* || ${OUTPUT_ROOT} == *$'\n'* || ${OUTPUT_ROOT} == *$'\r'* ]]; then
  echo "HSLE_OUTPUT_ROOT must be an absolute path without commas or newlines." >&2
  exit 2
fi
WORKER_SCRIPT=${PROJECT_ROOT}/script/workers/run_public_openrouter_resume_shard.sh
SHARD_COUNT=${HSLE_PUBLIC_RESUME_SHARDS:-8}
if [[ ! ${SHARD_COUNT} =~ ^[1-9][0-9]*$ ]]; then
  echo "HSLE_PUBLIC_RESUME_SHARDS must be a positive integer." >&2
  exit 2
fi
if [[ ! -x ${WORKER_SCRIPT} ]]; then
  echo "The release worker is absent or not executable: ${WORKER_SCRIPT}" >&2
  exit 2
fi

if [[ ${HSLE_PUBLIC_RESUME_SKIP_INSTALL:-0} != 1 ]]; then
  if [[ ! -x ${VENV_ROOT}/bin/python ]]; then
    run_without_credentials python3 -m venv "${VENV_ROOT}"
  fi
  run_without_credentials \
    "${VENV_ROOT}/bin/python" -m pip install --disable-pip-version-check -e "${PROJECT_ROOT}"
elif [[ ! -x ${VENV_ROOT}/bin/python ]]; then
  echo "HSLE_PUBLIC_RESUME_SKIP_INSTALL=1 requires an existing release venv." >&2
  exit 2
fi

PYTHON=${VENV_ROOT}/bin/python
run_without_credentials "${PYTHON}" - \
  "${HF_DATASET_REPO_ID}" \
  "${HF_DATASET_REVISION}" \
  "${DATASET_ROOT}" <<'PY'
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download
from hsle.download_data import validate_dataset


repo_id, revision, target_text = sys.argv[1:]
target = Path(target_text)
target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
lock_path = target.parent / f".{target.name}.download.lock"
with lock_path.open("a+b") as lock:
    os.chmod(lock_path, 0o600)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=target,
        token=False,
    )
    remote_files = set(
        HfApi(token=False).list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    )
    if (
        len(remote_files) != 274
        or "openrouter/hsle_openrouter_single_dispatch_v3.tar.gz" not in remote_files
    ):
        raise RuntimeError("pinned Hugging Face revision has an unexpected file inventory")
    actual_files: set[str] = set()
    for path in target.rglob("*"):
        relative = path.relative_to(target)
        if relative.parts[:1] == (".cache",):
            continue
        if path.is_symlink():
            raise RuntimeError(f"downloaded Hugging Face snapshot contains a symlink: {relative}")
        if path.is_file():
            actual_files.add(relative.as_posix())
    if actual_files != remote_files:
        missing = sorted(remote_files - actual_files)[:5]
        unexpected = sorted(actual_files - remote_files)[:5]
        raise RuntimeError(
            "local Hugging Face snapshot is incomplete or contaminated: "
            f"missing={missing}, unexpected={unexpected}"
        )
    validate_dataset(target)
    print(f"Downloaded the complete pinned HSLE dataset from {repo_id}@{revision} into {target}")
PY
if [[ ${INPUT_DOWNLOAD_REQUIRED} == 1 ]]; then
  run_without_credentials "${PYTHON}" - \
    "${HF_DATASET_REPO_ID}" \
    "${HF_DATASET_REVISION}" \
    "${HF_DATASET_ARCHIVE}" \
    "${HF_DATASET_ARCHIVE_SHA256}" \
    "${INPUT_ROOT}" \
    "${DATASET_ROOT}" <<'PY'
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

repo_id, revision, filename, expected_sha256, target_text, dataset_text = sys.argv[1:]
target = Path(target_text)
dataset_root = Path(dataset_text)
target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
lock_path = target.parent / f".{target.name}.download.lock"
with lock_path.open("a+b") as lock:
    os.chmod(lock_path, 0o600)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if target.is_dir():
        raise SystemExit(0)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"input-cache target exists but is not a directory: {target}")

    archive = dataset_root / filename
    if not archive.is_file() or archive.is_symlink():
        raise RuntimeError("complete Hugging Face snapshot lacks the pinned runner input archive")
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
DATASET_ROOT=$(cd -- "${DATASET_ROOT}" && pwd -P)
INPUT_ROOT=$(cd -- "${INPUT_ROOT}" && pwd -P)
run_without_credentials "${PYTHON}" - \
  "${PROJECT_ROOT}" \
  "${DATASET_ROOT}" \
  "${INPUT_ROOT}" \
  "${VENV_ROOT}" \
  "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import sys


project, dataset, inputs, venv, output = (Path(value).resolve() for value in sys.argv[1:])
if output == project or project.is_relative_to(output):
    raise RuntimeError("HSLE_OUTPUT_ROOT cannot equal or contain the project root")
if output.is_relative_to(project) and output != project / "need_to_be_judged":
    raise RuntimeError("inside the project, HSLE_OUTPUT_ROOT must be need_to_be_judged")
for label, protected in (("dataset", dataset), ("input", inputs), ("venv", venv)):
    if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
        raise RuntimeError(f"HSLE_OUTPUT_ROOT overlaps the protected {label} root")
print("Validated disjoint project, dataset, input, environment, and output roots")
PY
export HSLE_INPUT_ROOT=${INPUT_ROOT}

run_with_openrouter_and_optional_hf_only \
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
declare -a ROUTES=(kimi_k2_thinking kimi_k25 kimi_k26 kimi_k3 qwen38_max minimax_m25)
declare -a JOB_IDS=()
declare -a SUBMITTED_JOB_IDS=()
SUBMISSION_RELEASED=0
cancel_unreleased_submission() {
  local exit_status=$?
  if [[ ${SUBMISSION_RELEASED} == 0 && ${#SUBMITTED_JOB_IDS[@]} -gt 0 ]]; then
    scancel "${SUBMITTED_JOB_IDS[@]}" >/dev/null 2>&1 || true
  fi
  return "${exit_status}"
}
trap cancel_unreleased_submission EXIT

GATE_JOB_ID=$(sbatch \
  --parsable \
  --hold \
  --partition "${PARTITION_NAME}" \
  --job-name hsle-or-release-gate \
  --no-requeue \
  --cpus-per-task 1 \
  --mem 64M \
  --time 00:05:00 \
  --output "${OUTPUT_ROOT}/logs/%x-%j.out" \
  --error "${OUTPUT_ROOT}/logs/%x-%j.err" \
  --export "PATH=/usr/bin:/bin" \
  --wrap /usr/bin/true)
GATE_JOB_ID=${GATE_JOB_ID%%;*}
if [[ ! ${GATE_JOB_ID} =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an invalid release-gate job id." >&2
  exit 1
fi
SUBMITTED_JOB_IDS+=("${GATE_JOB_ID}")
echo "Staged held release gate: ${GATE_JOB_ID}"

for ROUTE in "${ROUTES[@]}"; do
  case ${ROUTE} in
    kimi_k3)
      ROUTE_ARRAY_LIMIT=1
      ;;
    *)
      ROUTE_ARRAY_LIMIT=4
      ;;
  esac
  declare -a SUBMIT=(
    sbatch
    --parsable
    --partition "${PARTITION_NAME}"
    --job-name "hsle-or-${ROUTE}"
    --no-requeue
    --dependency "afterok:${GATE_JOB_ID}"
    --array "0-$((SHARD_COUNT - 1))%${ROUTE_ARRAY_LIMIT}"
    --cpus-per-task 1
    --mem 8G
    --time 2-00:00:00
    --output "${OUTPUT_ROOT}/logs/%x-%A_%a.out"
    --error "${OUTPUT_ROOT}/logs/%x-%A_%a.err"
    --export "${API_KEY_ENV_NAME},PATH=/usr/bin:/bin,HSLE_INPUT_ROOT=${INPUT_ROOT},HSLE_PUBLIC_API_KEY_ENV_NAME=${API_KEY_ENV_NAME},HSLE_PUBLIC_PROJECT_ROOT=${PROJECT_ROOT},HSLE_PUBLIC_OUTPUT_ROOT=${OUTPUT_ROOT},HSLE_PUBLIC_ROUTE=${ROUTE},HSLE_PUBLIC_SHARD_COUNT=${SHARD_COUNT}"
    "${WORKER_SCRIPT}"
    worker
  )
  JOB_ID=$("${SUBMIT[@]}")
  JOB_ID=${JOB_ID%%;*}
  if [[ ! ${JOB_ID} =~ ^[0-9]+$ ]]; then
    echo "sbatch returned an invalid job id for ${ROUTE}." >&2
    exit 1
  fi
  JOB_IDS+=("${JOB_ID}")
  SUBMITTED_JOB_IDS+=("${JOB_ID}")
  echo "Staged ${ROUTE} behind release gate: ${JOB_ID}"
done

DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
FINALIZER_JOB_ID=$(sbatch \
  --parsable \
  --partition "${PARTITION_NAME}" \
  --job-name hsle-or-finalize \
  --no-requeue \
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
SUBMITTED_JOB_IDS+=("${FINALIZER_JOB_ID}")
if ! scontrol release "${GATE_JOB_ID}"; then
  echo "Failed to release the staged OpenRouter campaign; cancelling exact submitted jobs." >&2
  exit 1
fi
SUBMISSION_RELEASED=1
echo "Submitted transfer-manifest finalizer: ${FINALIZER_JOB_ID}"
echo "Released complete six-route campaign through gate: ${GATE_JOB_ID}"
echo "Results will be collected in: ${OUTPUT_ROOT}"
