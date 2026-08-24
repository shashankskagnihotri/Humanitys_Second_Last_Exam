"""Portable, judge-free recovery for the five stopped OpenRouter routes.

This module is intentionally separate from the live production controllers.  It
consumes an authorized, release-frozen input root, writes a durable intent before
every provider dispatch, and never automatically redraws an ambiguous paid request.
Static coordinates can finish in one pass.  Learning-from-experience (LFE)
coordinates stop after each solved-example response and emit a Gemini 3.5 Flash
feedback request; the same command resumes only after a bound decision is added
to ``lfe_feedback_decisions``.  No judge endpoint is present in this module.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from hsle._public_openrouter_core import (
    AdaptiveMessage,
    ContextExample,
    FeedbackMode,
    PromptEnvelope,
    PromptSource,
    RecoveryCoordinate,
    RecoveryTask,
    _append_canonical_message,
    _lfe_question_message,
    _source_transcript,
    build_attempt_request,
    canonical_json_bytes,
    canonical_json_sha256,
    generation_response_is_usable,
    lfe_feedback,
    load_prompt_source_tables,
    preflight_prompt_source,
)


SCHEMA_VERSION = 1
INPUT_DIRECTORY_NAME = ".hsle_public_resume_inputs_v2"
INPUT_MANIFEST_FILENAME = "INPUT_MANIFEST.json"
EXPECTED_INPUT_MANIFEST_FILE_SHA256 = (
    "fc1e7f134626df95c2092ca07c5f3932a47b59ea8660633202e307e5cdb508dd"
)
EXPECTED_INPUT_FILE_COUNT = 270
EXPECTED_TARGET_COUNT = 491
EXPECTED_CONTEXT_COUNT = 982
EXPECTED_IMAGE_FILE_COUNT = 258
EXPECTED_CORRECTION_COUNTS = {
    "correction_applied": 172,
    "apply_question_correction": 22,
    "apply_answer_correction": 93,
    "apply_rationale_correction": 160,
    "semantic_answer_change": 91,
}
EXPECTED_GROUND_TRUTH_VERSIONS = {
    "hle_verified_pinned_2026_02",
    "hsle_merged_source",
}
OFFICIAL_HLE_DATASET = "cais/hle"
OFFICIAL_HLE_SPLIT = "test"
OFFICIAL_HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
OFFICIAL_HLE_ROW_COUNT = 2_500
HF_VALIDATION_ENVIRONMENT_NAME = "HSLE_VALIDATE_OFFICIAL_HF"
HF_CREDENTIAL_ENVIRONMENT_NAME = "HF_TOKEN"
RESERVED_HF_CREDENTIAL_NAMES = frozenset({"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"})
OUTPUT_DIRECTORY_NAME = "needs_to_be_judged"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FEEDBACK_PROVIDER = "gemini"
FEEDBACK_MODEL = "gemini-3.5-flash"
MAX_ANSWER_ATTEMPTS = 6
MAX_OPERATIONAL_DISPATCHES = 12
DEFAULT_SHARD_COUNT = 8

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PARTITION = re.compile(r"[A-Za-z0-9_.-]+\Z")
_EVALUATION_KEY = re.compile(r"eval_[0-9a-f]{24}\Z")
_ROUTE_KEY = re.compile(r"[a-z0-9_]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PublicResumeError(RuntimeError):
    """A fail-closed public-resume validation error."""


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route_key: str
    scientific_model_id: str
    provider_requested_model_id: str
    catalog_model_id: str
    provider_tag: str
    provider_display_name: str
    input_modalities: tuple[str, ...]
    context_length: int
    max_tokens: int
    prompt_price_per_million: str
    completion_price_per_million: str
    reasoning_effort: str = ""
    predecessor_failed_request_model_id: str = ""

    @property
    def accepted_response_models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.catalog_model_id, self.provider_requested_model_id)))


ROUTES: dict[str, RouteSpec] = {
    "kimi_k2_thinking": RouteSpec(
        route_key="kimi_k2_thinking",
        scientific_model_id="moonshotai/Kimi-K2-Thinking",
        catalog_model_id="moonshotai/kimi-k2-thinking",
        provider_requested_model_id="moonshotai/kimi-k2-thinking-20251106",
        provider_tag="novita/bf16",
        provider_display_name="Novita",
        input_modalities=("text",),
        context_length=262_144,
        max_tokens=16_384,
        prompt_price_per_million="0.6",
        completion_price_per_million="2.5",
    ),
    "kimi_k25": RouteSpec(
        route_key="kimi_k25",
        scientific_model_id="moonshotai/Kimi-K2.5",
        catalog_model_id="moonshotai/kimi-k2.5",
        provider_requested_model_id="moonshotai/kimi-k2.5-0127",
        provider_tag="deepinfra/fp4",
        provider_display_name="DeepInfra",
        input_modalities=("text", "image"),
        context_length=262_144,
        max_tokens=16_384,
        prompt_price_per_million="0.45",
        completion_price_per_million="2.25",
    ),
    "kimi_k26": RouteSpec(
        route_key="kimi_k26",
        scientific_model_id="moonshotai/Kimi-K2.6",
        catalog_model_id="moonshotai/kimi-k2.6",
        provider_requested_model_id="moonshotai/kimi-k2.6",
        provider_tag="deepinfra/fp4",
        provider_display_name="DeepInfra",
        input_modalities=("text", "image"),
        context_length=262_144,
        max_tokens=16_384,
        prompt_price_per_million="0.75",
        completion_price_per_million="3.5",
        predecessor_failed_request_model_id="moonshotai/kimi-k2.6-20260420",
    ),
    "kimi_k3": RouteSpec(
        route_key="kimi_k3",
        scientific_model_id="moonshotai/Kimi-K3",
        catalog_model_id="moonshotai/kimi-k3",
        provider_requested_model_id="moonshotai/kimi-k3-20260715",
        provider_tag="morph/fp4",
        provider_display_name="Morph",
        input_modalities=("text", "image", "video"),
        context_length=1_048_576,
        max_tokens=16_384,
        prompt_price_per_million="2.8",
        completion_price_per_million="14.0",
        reasoning_effort="max",
    ),
    "qwen38_max": RouteSpec(
        route_key="qwen38_max",
        scientific_model_id="qwen/qwen3.8-max",
        catalog_model_id="qwen/qwen3.8-max",
        provider_requested_model_id="qwen/qwen3.8-max-20260803",
        provider_tag="alibaba",
        provider_display_name="Alibaba",
        input_modalities=("text", "image", "video"),
        context_length=1_000_000,
        max_tokens=16_384,
        prompt_price_per_million="2.0",
        completion_price_per_million="6.0",
        reasoning_effort="xhigh",
    ),
}

EXPECTED_VECTOR_COUNTS = {
    "kimi_k2_thinking": 1_438,
    "kimi_k25": 2_246,
    "kimi_k26": 2_421,
    "kimi_k3": 1_975,
    "qwen38_max": 2_047,
}
EXPECTED_VECTOR_SHA256 = {
    "kimi_k2_thinking": "9e11b590f0befb71749ebe29de9250b19260d257f6875585c606dad1e781690a",
    "kimi_k25": "a4e590facd7f8e35ff2bf48fe17abd5638ad95ea1c87f50ab5b8425e01d6349f",
    "kimi_k26": "1a53d559b206d44fe88e86b2b2fca31324423fa985af101d40e13a6f45281c57",
    "kimi_k3": "9971ff8400aedd9393e112926aff3aae6adb5bb1a429ce3404a15d1ec76b2b43",
    "qwen38_max": "9ecbe9c0fa15dda259626243d48b57df5d2fab3fe5172dceec7e20d1795f846d",
}
# These hashes bind the provider-visible prompt envelopes (including ordered
# image bytes) independently of the absolute location of the authorized root.
EXPECTED_PROMPT_VECTOR_SHA256 = {
    "kimi_k2_thinking": "a07275bc7db67ca320802544a5d9aefee7c74b52b4ae8312971a3b93b32bb41a",
    "kimi_k25": "122c88fb6c85f970c7e40bb258b07819699b470b2e443d2ab9f406f9da39afca",
    "kimi_k26": "75f80ee83c2b2594787839dec37fff7a2903904f4def01ecaf233322cd3476e5",
    "kimi_k3": "f6502602635c15f3e339536f3865ebd07e073ba6fdbbd6a661074b6bd92cc3e6",
    "qwen38_max": "5d65e8fc80bf6178dfcf7dae659d58ba24c6e66e7051d3885f3214dffb4b0bf5",
}
EXPECTED_PARTITION_CONTRACTS = {
    "kimi_k2_thinking": {
        "canonical_count": 2_085,
        "canonical_evaluation_key_vector_sha256": "6eac1029d3ace647791f6d5ca625babecaf20ef00191d66bab760e500c2e7efd",
        "settled_count": 640,
        "settled_evaluation_key_vector_sha256": "cf641f9abb71949991339d4bd3635282e8c39e5c7fff4c1f6410b1365912f465",
        "paid_no_replay_count": 7,
        "paid_no_replay_evaluation_key_vector_sha256": "1bf07ea8bd1f1bfd37b7286e6546494b8f5b496c274e212ae801e9afbfea2279",
        "callable_count": 1_438,
        "callable_evaluation_key_vector_sha256": "9e11b590f0befb71749ebe29de9250b19260d257f6875585c606dad1e781690a",
    },
    "kimi_k25": {
        "canonical_count": 2_455,
        "canonical_evaluation_key_vector_sha256": "056b10a24bb520a9cb8d2d985027bc3326c1a336e1fa7e81db2434de825c54de",
        "settled_count": 203,
        "settled_evaluation_key_vector_sha256": "2026fc60cca630a7e19991b87a885efee23a9111355826b090c665b9fff34410",
        "paid_no_replay_count": 6,
        "paid_no_replay_evaluation_key_vector_sha256": "5a872f0e0f4374bf46cda2a1ee90a6a3d04afc50a3f1bfb6d067b1508978e624",
        "callable_count": 2_246,
        "callable_evaluation_key_vector_sha256": "a4e590facd7f8e35ff2bf48fe17abd5638ad95ea1c87f50ab5b8425e01d6349f",
    },
    "kimi_k26": {
        "canonical_count": 2_455,
        "canonical_evaluation_key_vector_sha256": "868ef44bb0df43e2226189ef2fd865247bac85f45ba0c52f517d6738955de04c",
        "settled_count": 30,
        "settled_evaluation_key_vector_sha256": "1970b464ef96d37d2b664dd572034993bf1af3b8c92e6cfd9d5d52e8af3aa633",
        "paid_no_replay_count": 4,
        "paid_no_replay_evaluation_key_vector_sha256": "4a00355acdf73ebe0bf788d605cf6eacba0a4fec37be4f1f3504c72616cc1dce",
        "callable_count": 2_421,
        "callable_evaluation_key_vector_sha256": "1a53d559b206d44fe88e86b2b2fca31324423fa985af101d40e13a6f45281c57",
    },
    "kimi_k3": {
        "canonical_count": 2_455,
        "canonical_evaluation_key_vector_sha256": "3b2e009837b8ac2278a56f1e6e867a3da42fe92fe28ddcc01c52cd0468f30800",
        "settled_count": 442,
        "settled_evaluation_key_vector_sha256": "d421a408e5e664d069a4effbe355dabe0d4d6a2939010a188d88b88e822fdebb",
        "paid_no_replay_count": 38,
        "paid_no_replay_evaluation_key_vector_sha256": "8d54a6310c29e3ffa6e8f91c15bdac0af19c52e9c462e08ee09732630a63eb86",
        "callable_count": 1_975,
        "callable_evaluation_key_vector_sha256": "9971ff8400aedd9393e112926aff3aae6adb5bb1a429ce3404a15d1ec76b2b43",
    },
    "qwen38_max": {
        "canonical_count": 2_455,
        "canonical_evaluation_key_vector_sha256": "252d0674cd4f95c07efd972494f638891bf32576991385e7d9a1908bee0c32f9",
        "settled_count": 403,
        "settled_evaluation_key_vector_sha256": "c6954c3ab2d8bfdc9c0c334913f7b2b38dae01bd613593190d17121f76c246d4",
        "paid_no_replay_count": 5,
        "paid_no_replay_evaluation_key_vector_sha256": "1b6f42ac1b9c00869a405dc51586fdfd72e741bdc4baf6ee689f5b0dc9e9094d",
        "callable_count": 2_047,
        "callable_evaluation_key_vector_sha256": "9ecbe9c0fa15dda259626243d48b57df5d2fab3fe5172dceec7e20d1795f846d",
    },
}
EXPECTED_ROUTE_REQUEST_CONTRACTS = {
    route_key: {
        "scientific_model_id": route.scientific_model_id,
        "provider_requested_model_id": route.provider_requested_model_id,
        "provider_tag": route.provider_tag,
        "provider_display_name": route.provider_display_name,
        "predecessor_failed_request_model_id": route.predecessor_failed_request_model_id,
    }
    for route_key, route in ROUTES.items()
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "record_sha256" in result:
        raise PublicResumeError("record payload already has record_sha256")
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, add_record_hash: bool = True
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = _record_payload(payload) if add_record_hash else dict(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        raw = canonical_json_bytes(result) + b"\n"
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    _fsync_directory(path.parent)
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicResumeError(f"JSON object required: {path}")
    if "record_sha256" in value:
        unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
        if value["record_sha256"] != canonical_json_sha256(unsigned):
            raise PublicResumeError(f"signed JSON record differs: {path}")
    return value


def read_signed_json(path: Path) -> dict[str, Any]:
    """Read one runner-owned canonical record and require its self-hash."""

    if path.is_symlink() or not path.is_file():
        raise PublicResumeError(f"signed JSON record is absent or non-regular: {path}")
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PublicResumeError(f"signed JSON record cannot be read: {path}") from error
    record_sha256 = value.get("record_sha256")
    if not isinstance(record_sha256, str) or _SHA256.fullmatch(record_sha256) is None:
        raise PublicResumeError(f"signed JSON record lacks a record SHA-256: {path}")
    if path.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise PublicResumeError(f"signed JSON record is not canonical: {path}")
    return value


def validate_api_key_environment_name(name: str, *, require_value: bool) -> str:
    if _ENV_NAME.fullmatch(name) is None:
        raise PublicResumeError("API key environment name is malformed")
    if name in RESERVED_HF_CREDENTIAL_NAMES:
        raise PublicResumeError(
            "A Hugging Face credential name cannot be used as the OpenRouter key"
        )
    if require_value:
        value = os.environ.get(name, "")
        if not value.strip():
            raise PublicResumeError(f"Required environment variable {name!r} is empty")
    return name


def validate_partition(partition: str) -> str:
    if _PARTITION.fullmatch(partition) is None:
        raise PublicResumeError("Slurm partition name is malformed")
    return partition


def _root_path_is_export_safe(path: Path) -> None:
    raw = str(path)
    if not raw or "," in raw or "\n" in raw or "\r" in raw:
        raise PublicResumeError(
            "HSLE input root cannot be represented by the explicit Slurm export"
        )


def authorized_input_candidates(project_root: Path) -> tuple[Path, ...]:
    """Return documented warm-cache candidates without creating either one."""

    candidates: list[Path] = []
    cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache_home).expanduser() if cache_home else None
    if base is not None and base.is_absolute():
        candidates.append(base / "hsle" / "public-openrouter-resume-v2")
    else:
        home = os.environ.get("HOME", "").strip()
        if home and Path(home).expanduser().is_absolute():
            candidates.append(
                Path(home).expanduser() / ".cache" / "hsle" / "public-openrouter-resume-v2"
            )
    candidates.append(project_root / INPUT_DIRECTORY_NAME)
    return tuple(candidates)


def resolve_authorized_input_root(project_root: Path) -> Path:
    """Resolve a caller-provided or already-warm authorized input directory."""

    configured_raw = os.environ.get("HSLE_INPUT_ROOT", "")
    configured = configured_raw.strip()
    if configured_raw and configured != configured_raw:
        raise PublicResumeError("HSLE_INPUT_ROOT cannot have boundary whitespace")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise PublicResumeError("HSLE_INPUT_ROOT must be an absolute path")
        candidates = (candidate,)
    else:
        candidates = authorized_input_candidates(project_root)
    selected = next((path for path in candidates if path.is_dir()), None)
    if selected is None:
        searched = ", ".join(str(path) for path in candidates)
        raise PublicResumeError(
            f"authorized HSLE input root is absent; set HSLE_INPUT_ROOT or warm the documented cache ({searched})"
        )
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise PublicResumeError("authorized HSLE input root cannot be resolved") from error
    if not resolved.is_dir():
        raise PublicResumeError("authorized HSLE input root is not a directory")
    _root_path_is_export_safe(resolved)
    return resolved


def _safe_manifest_relative_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise PublicResumeError("input manifest path is malformed")
    relative = Path(raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != raw
    ):
        raise PublicResumeError("input manifest path is unsafe")
    return relative


def _verify_manifest_inventory(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = root / INPUT_MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PublicResumeError("authorized input root lacks a regular input manifest")
    if file_sha256(manifest_path) != EXPECTED_INPUT_MANIFEST_FILE_SHA256:
        raise PublicResumeError("input manifest SHA-256 differs from the release pin")
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PublicResumeError("input manifest cannot be read") from error
    if manifest.get("kind") != "hsle_public_openrouter_resume_inputs_v2":
        raise PublicResumeError("input manifest kind differs")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PublicResumeError("input manifest schema version differs")
    if manifest.get("route_vector_sha256s") != EXPECTED_VECTOR_SHA256:
        raise PublicResumeError("input manifest route vectors differ from the release")
    if manifest.get("route_partition_contracts") != EXPECTED_PARTITION_CONTRACTS:
        raise PublicResumeError("input manifest route partitions differ from the release")
    if manifest.get("route_request_contracts") != EXPECTED_ROUTE_REQUEST_CONTRACTS:
        raise PublicResumeError("input manifest route request contracts differ")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_INPUT_FILE_COUNT:
        raise PublicResumeError("input manifest file count differs from the release")

    expected_paths: set[str] = set()
    verified_hashes: dict[str, str] = {}
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise PublicResumeError("input manifest file row is malformed")
        relative = _safe_manifest_relative_path(row["path"])
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            raise PublicResumeError("input manifest contains a duplicate path")
        expected_paths.add(relative_text)
        expected_sha256 = row["sha256"]
        expected_size = row["size"]
        if (
            not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise PublicResumeError("input manifest file evidence is malformed")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PublicResumeError(f"input file is absent: {relative_text}")
        if path.stat().st_size != expected_size:
            raise PublicResumeError(f"input size differs: {relative_text}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise PublicResumeError(f"input SHA-256 differs: {relative_text}")
        verified_hashes[relative_text] = observed_sha256

    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PublicResumeError("authorized input root contains a symbolic link")
        if path.is_file():
            relative_text = path.relative_to(root).as_posix()
            if relative_text != INPUT_MANIFEST_FILENAME:
                actual_paths.add(relative_text)
    if actual_paths != expected_paths:
        raise PublicResumeError("authorized input file inventory differs from manifest")
    return manifest, verified_hashes


def verify_input_directory(
    root: Path, *, full_scientific_validation: bool = True
) -> dict[str, Any]:
    """Authenticate all bytes, then optionally preflight the scientific inputs."""

    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise PublicResumeError("authorized HSLE input root cannot be resolved") from error
    _root_path_is_export_safe(root)
    manifest, verified_hashes = _verify_manifest_inventory(root)
    audit: dict[str, Any] = {
        "input_manifest_file_sha256": EXPECTED_INPUT_MANIFEST_FILE_SHA256,
        "input_manifest_payload_sha256": canonical_json_sha256(manifest),
        "file_count": len(verified_hashes),
        "byte_inventory_sha256": canonical_json_sha256(verified_hashes),
    }
    if full_scientific_validation:
        audit.update(_verify_scientific_inputs(root, verified_hashes))
    return audit


def ensure_inputs(
    project_root: Path, *, full_scientific_validation: bool = True
) -> tuple[Path, dict[str, Any]]:
    root = resolve_authorized_input_root(project_root)
    return root, verify_input_directory(root, full_scientific_validation=full_scientific_validation)


def _task_from_row(row: Mapping[str, str], route: RouteSpec) -> RecoveryTask:
    required_controls = {
        "max_retries": "5",
        "max_total_attempts": "6",
        "attempts_1_to_4_input_profile": "full",
        "attempt_5_input_profile": "smart_truncate_v1",
        "attempt_6_input_profile": "smart_truncate_v2",
        "terminal_failure_status": "nonresponsive",
        "terminal_hle_correctness": "incorrect",
    }
    for name, expected in required_controls.items():
        if str(row.get(name, "")) != expected:
            raise PublicResumeError(f"task {name} differs for {route.route_key}")
    if float(str(row.get("terminal_closeness_score", "nan"))) != 0.0:
        raise PublicResumeError("terminal_closeness_score must be zero")
    evaluation_key = str(row.get("evaluation_key", ""))
    if _EVALUATION_KEY.fullmatch(evaluation_key) is None:
        raise PublicResumeError("task evaluation key is malformed")
    if row.get("model_id") != route.scientific_model_id:
        raise PublicResumeError("task scientific model identity differs")
    modality: Literal["text_only", "multimodal"] = (
        "multimodal" if "image" in route.input_modalities else "text_only"
    )
    if row.get("model_modality") != modality:
        raise PublicResumeError("task modality differs from exact route")
    coordinate = RecoveryCoordinate(
        model_id=route.scientific_model_id,
        requested_model_id=route.scientific_model_id,
        model_modality=modality,
        evaluation_setting=str(row.get("evaluation_setting", "")),
        concrete_variant=str(row.get("concrete_variant", "")),
        original_question_id=str(row.get("original_question_id", "")),
        setting_instance_id=str(row.get("setting_instance_id", "")),
        evaluation_key=evaluation_key,
    )
    return RecoveryTask(coordinate=coordinate)


def load_route_tasks(inputs: Path, route_key: str) -> tuple[RecoveryTask, ...]:
    route = ROUTES[route_key]
    path = inputs / "tasks" / f"{route_key}.csv"
    if not path.is_file():
        raise PublicResumeError(f"frozen route vector is absent: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tasks = tuple(_task_from_row(row, route) for row in rows)
    keys = [task.coordinate.evaluation_key for task in tasks]
    if len(keys) != len(set(keys)) or len(keys) != EXPECTED_VECTOR_COUNTS[route_key]:
        raise PublicResumeError(f"frozen task count/uniqueness differs for {route_key}")
    if canonical_json_sha256(sorted(keys)) != EXPECTED_VECTOR_SHA256[route_key]:
        raise PublicResumeError(f"frozen task vector SHA-256 differs for {route_key}")
    return tasks


def verify_route_partition(
    inputs: Path,
    route_key: str,
    tasks: Sequence[RecoveryTask],
) -> dict[str, Any]:
    """Prove callable, settled, and paid/no-replay form the exact route universe."""

    path = inputs / "exclusions" / f"{route_key}.json"
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PublicResumeError(f"route partition cannot be read: {route_key}") from error
    route = ROUTES[route_key]
    if (
        value.get("kind") != "hsle_public_openrouter_resume_exclusions_v2"
        or value.get("route_key") != route_key
        or value.get("scientific_model_id") != route.scientific_model_id
    ):
        raise PublicResumeError(f"route partition identity differs: {route_key}")

    def exact_keys(field: str) -> list[str]:
        raw = value.get(field)
        if not isinstance(raw, list) or any(
            not isinstance(key, str) or _EVALUATION_KEY.fullmatch(key) is None for key in raw
        ):
            raise PublicResumeError(f"route partition {field} is malformed: {route_key}")
        if raw != sorted(raw) or len(raw) != len(set(raw)):
            raise PublicResumeError(f"route partition {field} is not sorted/unique: {route_key}")
        return raw

    settled = exact_keys("complete_evaluation_keys")
    no_replay = exact_keys("paid_nonreplay_evaluation_keys")
    callable_keys = [task.coordinate.evaluation_key for task in tasks]
    settled_set = set(settled)
    no_replay_set = set(no_replay)
    callable_set = set(callable_keys)
    if settled_set & no_replay_set or settled_set & callable_set or no_replay_set & callable_set:
        raise PublicResumeError(f"route partition overlaps: {route_key}")
    canonical = settled_set | no_replay_set | callable_set
    observed = {
        "canonical_count": len(canonical),
        "canonical_evaluation_key_vector_sha256": canonical_json_sha256(sorted(canonical)),
        "settled_count": len(settled),
        "settled_evaluation_key_vector_sha256": canonical_json_sha256(settled),
        "paid_no_replay_count": len(no_replay),
        "paid_no_replay_evaluation_key_vector_sha256": canonical_json_sha256(no_replay),
        "callable_count": len(callable_keys),
        "callable_evaluation_key_vector_sha256": canonical_json_sha256(sorted(callable_keys)),
    }
    if observed != EXPECTED_PARTITION_CONTRACTS[route_key]:
        raise PublicResumeError(f"route partition differs from canonical authority: {route_key}")
    expected_fields = {
        "canonical_coordinate_count": observed["canonical_count"],
        "canonical_evaluation_key_vector_sha256": observed[
            "canonical_evaluation_key_vector_sha256"
        ],
        "complete_count": observed["settled_count"],
        "complete_evaluation_key_vector_sha256": observed["settled_evaluation_key_vector_sha256"],
        "paid_nonreplay_count": observed["paid_no_replay_count"],
        "paid_nonreplay_vector_sha256": observed["paid_no_replay_evaluation_key_vector_sha256"],
        "callable_count": observed["callable_count"],
        "callable_evaluation_key_vector_sha256": observed["callable_evaluation_key_vector_sha256"],
        "partition_contract_sha256": canonical_json_sha256(observed),
    }
    for field, expected in expected_fields.items():
        if value.get(field) != expected:
            raise PublicResumeError(f"route partition field differs ({field}): {route_key}")
    variants = dict(sorted(Counter(task.coordinate.concrete_variant for task in tasks).items()))
    if value.get("callable_variant_counts") != variants:
        raise PublicResumeError(f"route callable variant counts differ: {route_key}")
    return observed


def _strict_boolean(value: object, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise PublicResumeError(f"canonical input {field} is not an exact boolean")


def _strict_cell(value: object, *, field: str, nonblank: bool = False) -> str:
    if not isinstance(value, str):
        raise PublicResumeError(f"canonical input {field} is not text")
    if nonblank and not value.strip():
        raise PublicResumeError(f"canonical input {field} is blank")
    return value


def _strict_image_paths(
    raw_value: object,
    *,
    has_image: object,
    inputs: Path,
    field: str,
) -> tuple[Path, ...]:
    raw = _strict_cell(raw_value, field=field)
    declared = _strict_boolean(has_image, field=f"{field}_has_image")
    parts = raw.split(";") if raw else []
    paths: list[Path] = []
    for item in parts:
        relative = _safe_manifest_relative_path(item)
        if relative.parts[:2] != ("data", "images"):
            raise PublicResumeError(f"canonical input {field} leaves data/images")
        path = inputs / relative
        if not path.is_file() or path.is_symlink():
            raise PublicResumeError(f"canonical input image is absent: {relative}")
        paths.append(path.resolve(strict=True))
    if declared != bool(paths):
        raise PublicResumeError(f"canonical input {field} image flag differs")
    return tuple(paths)


def load_public_prompt_source(
    task: RecoveryTask,
    *,
    originals_by_id: Any,
    links_by_id: Any,
    inputs: Path,
) -> PromptSource:
    """Load one source while forbidding the legacy external-path fallbacks."""

    coordinate = task.coordinate
    try:
        original = originals_by_id.loc[coordinate.original_question_id]
        link = links_by_id.loc[coordinate.original_question_id]
    except KeyError as error:
        raise PublicResumeError("task question is absent from canonical inputs") from error
    if getattr(original, "ndim", 1) != 1 or getattr(link, "ndim", 1) != 1:
        raise PublicResumeError("canonical prompt input contains duplicate targets")
    target_images = _strict_image_paths(
        original.get("image_paths_or_ids", ""),
        has_image=original.get("has_image", ""),
        inputs=inputs,
        field="target_image_paths_or_ids",
    )
    examples: list[ContextExample] = []
    for index in (1, 2):
        prefix = f"example_{index}"
        examples.append(
            ContextExample(
                question_id=_strict_cell(
                    link.get(f"{prefix}_question_id", ""),
                    field=f"{prefix}_question_id",
                    nonblank=True,
                ),
                question=_strict_cell(
                    link.get(f"{prefix}_question", ""),
                    field=f"{prefix}_question",
                    nonblank=True,
                ),
                answer=_strict_cell(
                    link.get(f"{prefix}_answer", ""),
                    field=f"{prefix}_answer",
                    nonblank=True,
                ),
                rationale=_strict_cell(
                    link.get(f"{prefix}_rationale", ""),
                    field=f"{prefix}_rationale",
                ),
                has_image=_strict_boolean(
                    link.get(f"{prefix}_has_image", ""),
                    field=f"{prefix}_has_image",
                ),
                image_paths=tuple(
                    str(path)
                    for path in _strict_image_paths(
                        link.get(f"{prefix}_image_paths_or_ids", ""),
                        has_image=link.get(f"{prefix}_has_image", ""),
                        inputs=inputs,
                        field=f"{prefix}_image_paths_or_ids",
                    )
                ),
            )
        )
    expected_instance = {
        "zero_shot": "",
        "one_shot_a": examples[0].question_id,
        "one_shot_b": examples[1].question_id,
        "two_shot": f"{examples[0].question_id};{examples[1].question_id}",
        "learning_from_experience": (f"{examples[0].question_id};{examples[1].question_id}"),
    }[coordinate.concrete_variant]
    if coordinate.setting_instance_id != expected_instance:
        raise PublicResumeError("task linkage differs from canonical examples")
    return PromptSource(
        target_question=_strict_cell(
            original.get("question", ""), field="target_question", nonblank=True
        ),
        corrected_answer=_strict_cell(
            original.get("answer", ""), field="target_answer", nonblank=True
        ),
        target_rationale=_strict_cell(original.get("rationale", ""), field="target_rationale"),
        target_image_paths=target_images,
        examples=(examples[0], examples[1]),
    )


def _portable_image_record(
    path: Path, *, inputs: Path, verified_hashes: Mapping[str, str]
) -> dict[str, Any]:
    try:
        relative = path.resolve(strict=True).relative_to(inputs).as_posix()
    except (OSError, ValueError) as error:
        raise PublicResumeError("prompt image leaves authorized input root") from error
    digest = verified_hashes.get(relative)
    if digest is None:
        raise PublicResumeError("prompt image is absent from authenticated inventory")
    return {"path": relative, "sha256": digest}


def compute_prompt_vector_sha256(
    inputs: Path,
    route_key: str,
    *,
    originals_by_id: Any | None = None,
    links_by_id: Any | None = None,
    verified_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Hash every source and provider-visible envelope without dispatching."""

    if originals_by_id is None or links_by_id is None:
        originals_by_id, links_by_id = load_prompt_source_tables(
            inputs / "processed" / "hsle_original_questions.csv",
            inputs / "processed" / "hsle_question_example_links.csv",
        )
    if verified_hashes is None:
        _manifest, verified_hashes = _verify_manifest_inventory(inputs)
    vector: list[dict[str, Any]] = []
    envelope_count = 0
    for task in load_route_tasks(inputs, route_key):
        source = load_public_prompt_source(
            task,
            originals_by_id=originals_by_id,
            links_by_id=links_by_id,
            inputs=inputs,
        )
        envelopes = preflight_prompt_source(task, source)
        envelope_count += len(envelopes)
        source_record = {
            "target_question": source.target_question,
            "corrected_answer": source.corrected_answer,
            "target_rationale": source.target_rationale,
            "target_images": [
                _portable_image_record(path, inputs=inputs, verified_hashes=verified_hashes)
                for path in source.target_image_paths
            ],
            "examples": [
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "answer": example.answer,
                    "rationale": example.rationale,
                    "images": [
                        _portable_image_record(
                            Path(path),
                            inputs=inputs,
                            verified_hashes=verified_hashes,
                        )
                        for path in example.image_paths
                    ],
                }
                for example in source.examples
            ],
        }
        envelope_records = []
        for envelope in envelopes:
            envelope_records.append(
                {
                    "turn": envelope.turn,
                    "target_question": envelope.target_question,
                    "target_images": [
                        _portable_image_record(
                            path,
                            inputs=inputs,
                            verified_hashes=verified_hashes,
                        )
                        for path in envelope.target_image_paths
                    ],
                    "messages": [
                        {
                            "role": message.role,
                            "content": message.content,
                            "reasoning_content": message.reasoning_content,
                            "semantic_kind": message.semantic_kind,
                            "images": [
                                _portable_image_record(
                                    path,
                                    inputs=inputs,
                                    verified_hashes=verified_hashes,
                                )
                                for path in message.image_paths
                            ],
                        }
                        for message in envelope.messages
                    ],
                }
            )
        vector.append(
            {
                "coordinate": asdict(task.coordinate),
                "source_sha256": canonical_json_sha256(source_record),
                "envelopes_sha256": canonical_json_sha256(envelope_records),
            }
        )
    return {
        "task_count": len(vector),
        "envelope_count": envelope_count,
        "prompt_vector_sha256": canonical_json_sha256(vector),
    }


def _collect_referenced_images(originals: Any, links: Any, inputs: Path) -> set[str]:
    referenced: set[str] = set()
    for _index, row in originals.iterrows():
        for path in _strict_image_paths(
            row["image_paths_or_ids"],
            has_image=row["has_image"],
            inputs=inputs,
            field="target_image_paths_or_ids",
        ):
            referenced.add(path.relative_to(inputs).as_posix())
    for _index, row in links.iterrows():
        for number in (1, 2):
            for path in _strict_image_paths(
                row[f"example_{number}_image_paths_or_ids"],
                has_image=row[f"example_{number}_has_image"],
                inputs=inputs,
                field=f"example_{number}_image_paths_or_ids",
            ):
                referenced.add(path.relative_to(inputs).as_posix())
    return referenced


def _verify_scientific_inputs(inputs: Path, verified_hashes: Mapping[str, str]) -> dict[str, Any]:
    try:
        originals, links = load_prompt_source_tables(
            inputs / "processed" / "hsle_original_questions.csv",
            inputs / "processed" / "hsle_question_example_links.csv",
        )
        if len(originals) != EXPECTED_TARGET_COUNT or len(links) != EXPECTED_TARGET_COUNT:
            raise PublicResumeError("canonical target/linkage row count differs")
        target_ids = set(originals.index)
        if set(links.index) != target_ids:
            raise PublicResumeError("canonical target/linkage ID sets differ")
        if set(originals["row_role"]) != {"original"}:
            raise PublicResumeError("canonical target row roles differ")
        if set(originals["is_added"]) != {"False"}:
            raise PublicResumeError("canonical target added-row flags differ")
        if set(originals["ground_truth_version"]) != EXPECTED_GROUND_TRUTH_VERSIONS:
            raise PublicResumeError("canonical ground-truth versions differ")
        if "example_count" not in links or set(links["example_count"]) != {"2"}:
            raise PublicResumeError("canonical linkage example count differs")
        context_ids = [
            _strict_cell(
                row[f"example_{number}_question_id"],
                field=f"example_{number}_question_id",
                nonblank=True,
            )
            for _index, row in links.iterrows()
            for number in (1, 2)
        ]
        if (
            len(context_ids) != EXPECTED_CONTEXT_COUNT
            or len(set(context_ids)) != EXPECTED_CONTEXT_COUNT
            or target_ids.intersection(context_ids)
        ):
            raise PublicResumeError("canonical 982-context linkage differs")
        for field, expected_count in EXPECTED_CORRECTION_COUNTS.items():
            if field not in originals:
                raise PublicResumeError(f"canonical correction field is absent: {field}")
            values = [_strict_boolean(value, field=field) for value in originals[field]]
            if sum(values) != expected_count:
                raise PublicResumeError(f"canonical correction count differs: {field}")
        manifest_images = {
            path for path in verified_hashes if Path(path).parts[:2] == ("data", "images")
        }
        if len(manifest_images) != EXPECTED_IMAGE_FILE_COUNT:
            raise PublicResumeError("canonical image-file count differs")
        if _collect_referenced_images(originals, links, inputs) != manifest_images:
            raise PublicResumeError("canonical image references differ from inventory")

        prompt_hashes: dict[str, str] = {}
        envelope_counts: dict[str, int] = {}
        partition_contracts: dict[str, dict[str, Any]] = {}
        for route_key in ROUTES:
            tasks = load_route_tasks(inputs, route_key)
            partition_contracts[route_key] = verify_route_partition(inputs, route_key, tasks)
            prompt_audit = compute_prompt_vector_sha256(
                inputs,
                route_key,
                originals_by_id=originals,
                links_by_id=links,
                verified_hashes=verified_hashes,
            )
            observed = prompt_audit["prompt_vector_sha256"]
            if observed != EXPECTED_PROMPT_VECTOR_SHA256[route_key]:
                raise PublicResumeError(f"provider-visible prompt vector differs for {route_key}")
            prompt_hashes[route_key] = observed
            envelope_counts[route_key] = int(prompt_audit["envelope_count"])
    except PublicResumeError:
        raise
    except Exception as error:
        raise PublicResumeError("canonical scientific input preflight failed") from error
    return {
        "target_count": EXPECTED_TARGET_COUNT,
        "context_count": EXPECTED_CONTEXT_COUNT,
        "image_file_count": EXPECTED_IMAGE_FILE_COUNT,
        "correction_counts": EXPECTED_CORRECTION_COUNTS,
        "route_task_counts": EXPECTED_VECTOR_COUNTS,
        "route_vector_sha256s": EXPECTED_VECTOR_SHA256,
        "route_partition_contracts": partition_contracts,
        "route_prompt_vector_sha256s": prompt_hashes,
        "route_envelope_counts": envelope_counts,
    }


def validate_optional_official_hle_access() -> dict[str, Any]:
    """Optionally authenticate access to the pinned official cais/hle revision."""

    requested = os.environ.get(HF_VALIDATION_ENVIRONMENT_NAME, "0")
    if requested not in {"0", "1"}:
        raise PublicResumeError(f"{HF_VALIDATION_ENVIRONMENT_NAME} must be 0 or 1")
    if requested == "0":
        return {"requested": False, "credential_exported_to_children": False}
    token = os.environ.get(HF_CREDENTIAL_ENVIRONMENT_NAME, "")
    if not token.strip():
        raise PublicResumeError(
            f"{HF_CREDENTIAL_ENVIRONMENT_NAME} is required for official validation"
        )
    try:
        from huggingface_hub import HfApi

        information = HfApi(token=token).dataset_info(
            OFFICIAL_HLE_DATASET,
            revision=OFFICIAL_HLE_REVISION,
        )
    except Exception as error:
        raise PublicResumeError("pinned official HLE access validation failed") from error
    if getattr(information, "sha", None) != OFFICIAL_HLE_REVISION:
        raise PublicResumeError("official HLE revision differs from the release pin")
    return {
        "requested": True,
        "dataset": OFFICIAL_HLE_DATASET,
        "revision": OFFICIAL_HLE_REVISION,
        "split": OFFICIAL_HLE_SPLIT,
        "expected_row_count": OFFICIAL_HLE_ROW_COUNT,
        "credential_environment_name": HF_CREDENTIAL_ENVIRONMENT_NAME,
        "credential_value_persisted": False,
        "credential_exported_to_children": False,
    }


def _read_unpriced_openrouter_json(
    request: Request, *, credential: str = "", purpose: str
) -> dict[str, Any]:
    """Read one non-generation OpenRouter response without persisting its body."""

    try:
        with urlopen(request, timeout=30) as response:
            status = int(response.status)
            raw = response.read(8 * 1024 * 1024 + 1)
    except HTTPError as error:
        # Consume the body so the connection can close, but never persist or
        # echo an authenticated error response.
        error.read(8 * 1024 * 1024 + 1)
        raise PublicResumeError(
            f"OpenRouter {purpose} preflight returned HTTP {int(error.code)}"
        ) from None
    except (URLError, TimeoutError, OSError, ValueError) as error:
        raise PublicResumeError(f"OpenRouter {purpose} preflight was unavailable") from error
    if status != 200:
        raise PublicResumeError(f"OpenRouter {purpose} preflight returned HTTP {status}")
    if len(raw) > 8 * 1024 * 1024:
        raise PublicResumeError(f"OpenRouter {purpose} preflight response was oversized")
    if credential and _response_contains_credential(raw, credential):
        raise PublicResumeError(
            f"OpenRouter {purpose} preflight echoed the credential; body discarded"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicResumeError(f"OpenRouter {purpose} preflight did not return JSON") from error
    if not isinstance(value, dict):
        raise PublicResumeError(f"OpenRouter {purpose} preflight JSON is not an object")
    return value


def _per_million_price(raw: object, *, route_key: str, field: str) -> Decimal:
    try:
        value = Decimal(str(raw)) * Decimal(1_000_000)
    except (InvalidOperation, ValueError) as error:
        raise PublicResumeError(
            f"OpenRouter endpoint price is malformed for {route_key}/{field}"
        ) from error
    if not value.is_finite() or value < 0:
        raise PublicResumeError(f"OpenRouter endpoint price is invalid for {route_key}/{field}")
    return value


def validate_openrouter_live_contract(api_key: str) -> dict[str, Any]:
    """Fail closed on bad auth or an unavailable/drifted exact endpoint.

    These GET requests are catalog/authentication checks only.  They never call
    chat completions and therefore cannot consume a benchmark coordinate.
    """

    if not api_key or api_key != api_key.strip() or "\r" in api_key or "\n" in api_key:
        raise PublicResumeError("OpenRouter credential has boundary whitespace or is empty")
    api_base = OPENROUTER_URL.rsplit("/chat/completions", 1)[0]
    auth = _read_unpriced_openrouter_json(
        Request(
            f"{api_base}/auth/key",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            method="GET",
        ),
        credential=api_key,
        purpose="credential",
    )
    if not isinstance(auth.get("data"), dict):
        raise PublicResumeError("OpenRouter credential preflight lacks account data")

    endpoint_evidence: dict[str, dict[str, Any]] = {}
    for route_key, route in ROUTES.items():
        try:
            author, slug = route.provider_requested_model_id.split("/", 1)
        except ValueError as error:
            raise PublicResumeError(f"OpenRouter model ID is malformed for {route_key}") from error
        catalog = _read_unpriced_openrouter_json(
            Request(
                f"{api_base}/models/{quote(author, safe='')}/{quote(slug, safe='')}/endpoints",
                headers={"Accept": "application/json"},
                method="GET",
            ),
            purpose=f"endpoint {route_key}",
        )
        data = catalog.get("data")
        if not isinstance(data, dict) or data.get("id") != route.catalog_model_id:
            raise PublicResumeError(f"OpenRouter catalog identity differs for {route_key}")
        endpoints = data.get("endpoints")
        matches = (
            [
                row
                for row in endpoints
                if isinstance(row, dict) and row.get("tag") == route.provider_tag
            ]
            if isinstance(endpoints, list)
            else []
        )
        if len(matches) != 1:
            raise PublicResumeError(
                f"OpenRouter exact endpoint tag is absent or duplicated for {route_key}"
            )
        endpoint = matches[0]
        status = endpoint.get("status")
        context_length = endpoint.get("context_length")
        max_completion_tokens = endpoint.get("max_completion_tokens")
        supported = endpoint.get("supported_parameters")
        required_parameters = {"reasoning", "include_reasoning", "max_tokens", "seed"}
        if route.reasoning_effort:
            required_parameters.add("reasoning_effort")
        if endpoint.get("provider_name") != route.provider_display_name:
            raise PublicResumeError(f"OpenRouter provider identity differs for {route_key}")
        if isinstance(status, bool) or not isinstance(status, int) or status != 0:
            raise PublicResumeError(
                f"OpenRouter exact endpoint is not healthy for {route_key} (status {status!r})"
            )
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or context_length < route.context_length
        ):
            raise PublicResumeError(f"OpenRouter endpoint context is insufficient for {route_key}")
        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens < route.max_tokens
        ):
            raise PublicResumeError(
                f"OpenRouter endpoint output limit is insufficient for {route_key}"
            )
        if not isinstance(supported, list) or not required_parameters.issubset(
            {str(item) for item in supported}
        ):
            raise PublicResumeError(f"OpenRouter endpoint parameters differ for {route_key}")
        pricing = endpoint.get("pricing")
        if not isinstance(pricing, dict):
            raise PublicResumeError(f"OpenRouter endpoint pricing is absent for {route_key}")
        observed_prompt = _per_million_price(
            pricing.get("prompt"), route_key=route_key, field="prompt"
        )
        observed_completion = _per_million_price(
            pricing.get("completion"), route_key=route_key, field="completion"
        )
        expected_prompt = Decimal(route.prompt_price_per_million)
        expected_completion = Decimal(route.completion_price_per_million)
        if observed_prompt != expected_prompt or observed_completion != expected_completion:
            raise PublicResumeError(f"OpenRouter endpoint price differs for {route_key}")
        endpoint_evidence[route_key] = {
            "catalog_model_id": route.catalog_model_id,
            "provider_requested_model_id": route.provider_requested_model_id,
            "provider_tag": route.provider_tag,
            "provider_display_name": route.provider_display_name,
            "status": status,
            "context_length": context_length,
            "max_completion_tokens": max_completion_tokens,
            "required_parameters": sorted(required_parameters),
            "prompt_price_per_million": str(expected_prompt),
            "completion_price_per_million": str(expected_completion),
        }
    return {
        "kind": "hsle_public_openrouter_unpriced_live_preflight_v1",
        "checked_at_utc": utc_now(),
        "credential_authenticated": True,
        "credential_value_persisted": False,
        "generation_calls_made": 0,
        "endpoints": endpoint_evidence,
    }


def stable_shard(task: RecoveryTask, shard_count: int) -> int:
    material = list(task.shard_material)
    digest = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _image_data_url(path: Path, expected_sha256: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise PublicResumeError(f"prompt image is absent: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PublicResumeError(f"prompt image changed: {path}")
    mime, _encoding = mimetypes.guess_type(path.name)
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise PublicResumeError(f"unsupported prompt image MIME: {path}")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def openrouter_messages(attempt: Any) -> list[dict[str, Any]]:
    evidence_by_message: dict[int, list[Any]] = {}
    for evidence in attempt.image_evidence:
        evidence_by_message.setdefault(evidence.message_index, []).append(evidence)
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(attempt.messages):
        evidence = sorted(
            evidence_by_message.get(index, ()),
            key=lambda item: item.image_index_within_message,
        )
        if evidence:
            content: str | list[dict[str, Any]] = [{"type": "text", "text": message.content}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(item.path), item.sha256)},
                }
                for item in evidence
            )
        else:
            content = message.content
        row: dict[str, Any] = {"role": message.role, "content": content}
        if message.reasoning_content:
            if message.role != "assistant":
                raise PublicResumeError("only assistant messages may carry reasoning")
            row["reasoning_content"] = message.reasoning_content
        messages.append(row)
    return messages


def build_payload(attempt: Any, route: RouteSpec) -> dict[str, Any]:
    reasoning = {"effort": route.reasoning_effort} if route.reasoning_effort else {"enabled": True}
    return {
        "model": route.provider_requested_model_id,
        "messages": openrouter_messages(attempt),
        "max_tokens": route.max_tokens,
        "seed": 0,
        "include_reasoning": True,
        "reasoning": reasoning,
        "usage": {"include": True},
        "provider": {
            "order": [route.provider_tag],
            "only": [route.provider_tag],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {
                "prompt": float(route.prompt_price_per_million),
                "completion": float(route.completion_price_per_million),
            },
        },
    }


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    return ""


def _response_contains_credential(raw: bytes, credential: str) -> bool:
    """Detect literal and JSON-unicode-escaped credential representations."""

    text = raw.decode("utf-8", "replace")
    escaped_lower = "".join(f"\\u{ord(character):04x}" for character in credential)
    escaped_upper = "".join(f"\\u{ord(character):04X}" for character in credential)
    return any(value and value in text for value in (credential, escaped_lower, escaped_upper))


def _response_fields(body: Mapping[str, Any], route: RouteSpec) -> dict[str, Any]:
    response_id = body.get("id")
    model = body.get("model")
    choices = body.get("choices")
    usage = body.get("usage")
    if not isinstance(response_id, str) or not response_id.strip():
        raise PublicResumeError("accepted response lacks provider response id")
    if model not in route.accepted_response_models:
        raise PublicResumeError(f"accepted response model differs: {model!r}")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise PublicResumeError("accepted response lacks first choice")
    if not isinstance(usage, dict):
        raise PublicResumeError("accepted response lacks usage")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise PublicResumeError("accepted response lacks assistant message")
    text = _content_text(message.get("content"))
    reasoning = _content_text(message.get("reasoning_content", message.get("reasoning", "")))
    provider = body.get("provider")
    if provider not in (None, "", route.provider_display_name):
        raise PublicResumeError(f"accepted response provider differs: {provider!r}")
    return {
        "response_id": response_id,
        "response_model": model,
        "provider": provider or route.provider_display_name,
        "text": text,
        "reasoning": reasoning,
        "finish_reason": choices[0].get("finish_reason", ""),
        "usage": usage,
    }


def _dispatch_backoff(status: int, dispatch_ordinal: int, evaluation_key: str) -> float:
    jitter = int(hashlib.sha256(evaluation_key.encode()).hexdigest()[:8], 16)
    if status == 429:
        return float(60 + jitter % 61)
    base = (45, 90, 180, 300)[min(dispatch_ordinal - 1, 3)]
    return float(base + jitter % 21)


@dataclass(frozen=True, slots=True)
class TurnResult:
    status: Literal["success", "exhausted", "ambiguous", "blocked"]
    text: str = ""
    reasoning: str = ""
    successful_attempt: int | None = None
    successful_input_profile: str = ""
    response_record: str = ""
    reason: str = ""


def _existing_dispatch_state(path: Path) -> str:
    settlements = [
        name
        for name in ("response.json", "ambiguity.json", "rejection.json")
        if (path / name).exists() or (path / name).is_symlink()
    ]
    if len(settlements) > 1:
        raise PublicResumeError(f"dispatch has conflicting settlement records: {path}")
    if settlements:
        return settlements[0].removesuffix(".json")
    intent = path / "intent.json"
    if intent.exists() or intent.is_symlink():
        return "dangling_intent"
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise PublicResumeError(f"dispatch path is non-regular: {path}")
    return "empty"


def run_turn(
    *,
    route: RouteSpec,
    task: RecoveryTask,
    envelope: PromptEnvelope,
    api_key: str,
    output_root: Path,
) -> TurnResult:
    raw_root = (
        output_root / "raw_api" / route.route_key / task.coordinate.evaluation_key / envelope.turn
    )
    for attempt_number in range(1, MAX_ANSWER_ATTEMPTS + 1):
        attempt = build_attempt_request(envelope, attempt_number)
        payload = build_payload(attempt, route)
        request_sha256 = canonical_json_sha256(payload)
        definite_blank = False
        for dispatch_ordinal in range(1, MAX_OPERATIONAL_DISPATCHES + 1):
            dispatch_root = raw_root / (
                f"attempt-{attempt_number:02d}-{attempt.input_profile}/dispatch-{dispatch_ordinal:02d}"
            )
            state = _existing_dispatch_state(dispatch_root)
            if state in {"dangling_intent", "ambiguity"}:
                if state == "dangling_intent":
                    atomic_write_json(
                        dispatch_root / "ambiguity.json",
                        {
                            "kind": "hsle_public_openrouter_paid_ambiguity_v1",
                            "recorded_at_utc": utc_now(),
                            "evaluation_key": task.coordinate.evaluation_key,
                            "route_key": route.route_key,
                            "turn": envelope.turn,
                            "answer_attempt": attempt_number,
                            "operational_dispatch": dispatch_ordinal,
                            "request_sha256": request_sha256,
                            "automatic_replay_authorized": False,
                            "reason": "durable_intent_without_settlement_on_resume",
                        },
                    )
                return TurnResult(
                    status="ambiguous",
                    reason="paid request has no definitive settlement; no replay",
                )
            if state == "response":
                response_relative = (dispatch_root / "response.json").relative_to(output_root)
                fields, _validated_attempt, _validated_profile = _validated_response_record(
                    output_root=output_root,
                    route=route,
                    task=task,
                    raw_relative=response_relative.as_posix(),
                    expected_turn=envelope.turn,
                    expected_request_sha256=request_sha256,
                )
                if generation_response_is_usable(fields.get("text", "")):
                    return TurnResult(
                        status="success",
                        text=str(fields["text"]),
                        reasoning=str(fields.get("reasoning", "")),
                        successful_attempt=attempt_number,
                        successful_input_profile=attempt.input_profile,
                        response_record=str(
                            (dispatch_root / "response.json").relative_to(output_root)
                        ),
                    )
                definite_blank = True
                break
            if state == "rejection":
                rejection_kind, status = _validated_rejection_record(
                    output_root=output_root,
                    route=route,
                    task=task,
                    raw_relative=(dispatch_root / "rejection.json")
                    .relative_to(output_root)
                    .as_posix(),
                    expected_turn=envelope.turn,
                    expected_request_sha256=request_sha256,
                )
                if rejection_kind == "hsle_public_openrouter_accepted_error_v1" or status in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    continue
                return TurnResult(status="blocked", reason=f"definitive HTTP {status}")

            intent = atomic_write_json(
                dispatch_root / "intent.json",
                {
                    "kind": "hsle_public_openrouter_dispatch_intent_v1",
                    "recorded_at_utc": utc_now(),
                    "evaluation_key": task.coordinate.evaluation_key,
                    "coordinate_key": task.coordinate.key,
                    "route_key": route.route_key,
                    "scientific_model_id": route.scientific_model_id,
                    "provider_requested_model_id": route.provider_requested_model_id,
                    "provider_tag": route.provider_tag,
                    "turn": envelope.turn,
                    "answer_attempt": attempt_number,
                    "input_profile": attempt.input_profile,
                    "operational_dispatch": dispatch_ordinal,
                    "request_sha256": request_sha256,
                    "credential_environment_name_persisted": False,
                    "credential_value_persisted": False,
                    "automatic_replay_authorized": False,
                },
            )
            body_bytes: bytes | None = None
            try:
                request = Request(
                    OPENROUTER_URL,
                    data=canonical_json_bytes(payload),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/hsle-benchmark/public-resume",
                        "X-Title": "HSLE public stopped-route resume",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=1800) as response:
                    http_status = int(response.status)
                    body_bytes = response.read()
            except HTTPError as error:
                error_body = error.read()
                rejection = atomic_write_json(
                    dispatch_root / "rejection.json",
                    {
                        "kind": "hsle_public_openrouter_definitive_rejection_v1",
                        "recorded_at_utc": utc_now(),
                        "intent_record_sha256": intent["record_sha256"],
                        "evaluation_key": task.coordinate.evaluation_key,
                        "route_key": route.route_key,
                        "turn": envelope.turn,
                        "answer_attempt": attempt_number,
                        "operational_dispatch": dispatch_ordinal,
                        "request_sha256": request_sha256,
                        "http_status": int(error.code),
                        "response_body_sha256": hashlib.sha256(error_body).hexdigest(),
                        "response_body_persisted": False,
                    },
                )
                status = int(rejection["http_status"])
                if status not in {429, 500, 502, 503, 504}:
                    return TurnResult(status="blocked", reason=f"definitive HTTP {status}")
                if dispatch_ordinal == MAX_OPERATIONAL_DISPATCHES:
                    return TurnResult(status="blocked", reason=f"HTTP {status} retry limit")
                time.sleep(
                    _dispatch_backoff(status, dispatch_ordinal, task.coordinate.evaluation_key)
                )
                continue
            except (URLError, TimeoutError, OSError) as error:
                atomic_write_json(
                    dispatch_root / "ambiguity.json",
                    {
                        "kind": "hsle_public_openrouter_paid_ambiguity_v1",
                        "recorded_at_utc": utc_now(),
                        "intent_record_sha256": intent["record_sha256"],
                        "evaluation_key": task.coordinate.evaluation_key,
                        "route_key": route.route_key,
                        "turn": envelope.turn,
                        "answer_attempt": attempt_number,
                        "operational_dispatch": dispatch_ordinal,
                        "request_sha256": request_sha256,
                        "automatic_replay_authorized": False,
                        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                        "reason": "transport outcome after durable intent is not provably uncharged",
                    },
                )
                return TurnResult(status="ambiguous", reason="transport settlement ambiguous")

            assert body_bytes is not None
            if _response_contains_credential(body_bytes, api_key):
                atomic_write_json(
                    dispatch_root / "ambiguity.json",
                    {
                        "kind": "hsle_public_openrouter_paid_ambiguity_v1",
                        "recorded_at_utc": utc_now(),
                        "intent_record_sha256": intent["record_sha256"],
                        "evaluation_key": task.coordinate.evaluation_key,
                        "route_key": route.route_key,
                        "turn": envelope.turn,
                        "answer_attempt": attempt_number,
                        "operational_dispatch": dispatch_ordinal,
                        "request_sha256": request_sha256,
                        "raw_response_sha256": hashlib.sha256(body_bytes).hexdigest(),
                        "automatic_replay_authorized": False,
                        "reason": "provider response contained credential value and was not persisted",
                    },
                )
                return TurnResult(
                    status="ambiguous",
                    reason="credential-bearing provider response was not persisted",
                )
            raw_path = dispatch_root / "raw_response.bin"
            descriptor = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                offset = 0
                while offset < len(body_bytes):
                    offset += os.write(descriptor, body_bytes[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(dispatch_root)
            try:
                body = json.loads(body_bytes)
                if not isinstance(body, dict):
                    raise PublicResumeError("accepted response is not an object")
                if isinstance(body.get("error"), dict) and not body.get("id"):
                    atomic_write_json(
                        dispatch_root / "rejection.json",
                        {
                            "kind": "hsle_public_openrouter_accepted_error_v1",
                            "recorded_at_utc": utc_now(),
                            "intent_record_sha256": intent["record_sha256"],
                            "evaluation_key": task.coordinate.evaluation_key,
                            "route_key": route.route_key,
                            "turn": envelope.turn,
                            "answer_attempt": attempt_number,
                            "operational_dispatch": dispatch_ordinal,
                            "request_sha256": request_sha256,
                            "http_status": http_status,
                            "response_body_sha256": hashlib.sha256(body_bytes).hexdigest(),
                            "provider_error": body["error"],
                        },
                    )
                    if dispatch_ordinal == MAX_OPERATIONAL_DISPATCHES:
                        return TurnResult(status="blocked", reason="accepted error retry limit")
                    time.sleep(
                        _dispatch_backoff(500, dispatch_ordinal, task.coordinate.evaluation_key)
                    )
                    continue
                fields = _response_fields(body, route)
            except Exception as error:
                atomic_write_json(
                    dispatch_root / "ambiguity.json",
                    {
                        "kind": "hsle_public_openrouter_paid_ambiguity_v1",
                        "recorded_at_utc": utc_now(),
                        "intent_record_sha256": intent["record_sha256"],
                        "evaluation_key": task.coordinate.evaluation_key,
                        "route_key": route.route_key,
                        "turn": envelope.turn,
                        "answer_attempt": attempt_number,
                        "operational_dispatch": dispatch_ordinal,
                        "request_sha256": request_sha256,
                        "raw_response_sha256": hashlib.sha256(body_bytes).hexdigest(),
                        "automatic_replay_authorized": False,
                        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                        "reason": str(error),
                    },
                )
                return TurnResult(status="ambiguous", reason="accepted response parse ambiguous")
            response_record = atomic_write_json(
                dispatch_root / "response.json",
                {
                    "kind": "hsle_public_openrouter_response_v1",
                    "recorded_at_utc": utc_now(),
                    "intent_record_sha256": intent["record_sha256"],
                    "evaluation_key": task.coordinate.evaluation_key,
                    "route_key": route.route_key,
                    "turn": envelope.turn,
                    "answer_attempt": attempt_number,
                    "input_profile": attempt.input_profile,
                    "operational_dispatch": dispatch_ordinal,
                    "request_sha256": request_sha256,
                    "http_status": http_status,
                    "raw_response_path": str(raw_path.relative_to(output_root)),
                    "raw_response_sha256": hashlib.sha256(body_bytes).hexdigest(),
                    "parsed_fields": fields,
                },
            )
            if generation_response_is_usable(fields["text"]):
                return TurnResult(
                    status="success",
                    text=str(fields["text"]),
                    reasoning=str(fields["reasoning"]),
                    successful_attempt=attempt_number,
                    successful_input_profile=attempt.input_profile,
                    response_record=str((dispatch_root / "response.json").relative_to(output_root)),
                )
            definite_blank = True
            _ = response_record
            break
        if not definite_blank:
            return TurnResult(status="blocked", reason="operational dispatches exhausted")
    return TurnResult(
        status="exhausted",
        reason="six settled accepted responses had no usable visible answer",
    )


def _scientific_result(
    *,
    route: RouteSpec,
    task: RecoveryTask,
    turn: TurnResult,
    turn_attempt_counts: Mapping[str, int],
    feedback_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    success = turn.status == "success"
    response_id = (
        "resp_public_"
        + hashlib.sha256(
            f"{route.route_key}\0{task.coordinate.evaluation_key}".encode()
        ).hexdigest()[:32]
    )
    result = {
        "response_id": response_id,
        "model_id": route.scientific_model_id,
        "requested_model_id": route.scientific_model_id,
        "model_modality": task.coordinate.model_modality,
        "evaluation_setting": task.coordinate.evaluation_setting,
        "concrete_variant": task.coordinate.concrete_variant,
        "original_question_id": task.coordinate.original_question_id,
        "setting_instance_id": task.coordinate.setting_instance_id,
        "evaluation_key": task.coordinate.evaluation_key,
        "coordinate_key": task.coordinate.key,
        "generation_status": "success" if success else "nonresponsive",
        "model_parsed_answer": "",
        "model_raw_output": turn.text if success else "",
        "model_raw_output_sha256": hashlib.sha256(turn.text.encode()).hexdigest(),
        "model_reasoning_content": turn.reasoning if success else "",
        "model_reasoning_content_sha256": hashlib.sha256(turn.reasoning.encode()).hexdigest(),
        "successful_attempt_number": turn.successful_attempt,
        "successful_input_profile": turn.successful_input_profile,
        "turn_attempt_counts": dict(turn_attempt_counts),
        "turn_attempt_counts_json": json.dumps(
            dict(turn_attempt_counts), sort_keys=True, separators=(",", ":")
        ),
        "feedback_evidence": list(feedback_evidence),
        "feedback_evidence_json": json.dumps(
            list(feedback_evidence), sort_keys=True, separators=(",", ":")
        ),
        "max_retries": 5,
        "max_total_attempts": 6,
        "attempts_1_to_4_input_profile": "full",
        "attempt_5_input_profile": "smart_truncate_v1",
        "attempt_6_input_profile": "smart_truncate_v2",
        "failed_turn": "" if success else "generation",
        "terminal_reason": "" if success else turn.reason,
        "score_origin": (
            "pending_gemini35_judging" if success else "generation_nonresponsive_policy"
        ),
        "terminal_hle_correctness": None if success else "incorrect",
        "terminal_closeness_score": None if success else 0,
        "gemini_judge_call_required": success,
    }
    return {
        "kind": "hsle_public_openrouter_generation_result_v1",
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        "route_key": route.route_key,
        "scientific_model_id": route.scientific_model_id,
        "provider_requested_model_id": route.provider_requested_model_id,
        "provider_tag": route.provider_tag,
        "provider_display_name": route.provider_display_name,
        "input_manifest_file_sha256": EXPECTED_INPUT_MANIFEST_FILE_SHA256,
        "route_vector_sha256": EXPECTED_VECTOR_SHA256[route.route_key],
        "route_prompt_vector_sha256": EXPECTED_PROMPT_VECTOR_SHA256[route.route_key],
        "response_record": turn.response_record,
        "result": result,
        "final_hle_and_closeness_judging_complete": False,
        "judge_calls_made_by_public_resume": 0,
    }


def _feedback_request(
    *,
    route: RouteSpec,
    task: RecoveryTask,
    turn_name: str,
    question: str,
    corrected_answer: str,
    model_response: str,
    transcript: str,
    turn: TurnResult,
) -> dict[str, Any]:
    return {
        "kind": "hsle_public_lfe_feedback_request_v1",
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now(),
        "route_key": route.route_key,
        "evaluation_key": task.coordinate.evaluation_key,
        "coordinate_key": task.coordinate.key,
        "turn": turn_name,
        "judge_provider_required": FEEDBACK_PROVIDER,
        "judge_model_required": FEEDBACK_MODEL,
        "question": question,
        "corrected_answer": corrected_answer,
        "model_response": model_response,
        "source_transcript": transcript,
        "source_response_record": turn.response_record,
        "source_response_sha256": hashlib.sha256(model_response.encode()).hexdigest(),
    }


def _load_feedback_decision(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    turn_name: str,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = (
        output_root
        / "lfe_feedback_decisions"
        / route.route_key
        / task.coordinate.evaluation_key
        / f"{turn_name}.json"
    )
    if not path.is_file():
        return None
    decision = read_json(path)
    expected_request_sha = canonical_json_sha256(
        {key: value for key, value in request.items() if key != "record_sha256"}
    )
    if (
        decision.get("kind") != "hsle_public_lfe_feedback_decision_v1"
        or not isinstance(decision.get("record_sha256"), str)
        or decision.get("feedback_request_record_sha256") != request.get("record_sha256")
        or decision.get("feedback_request_payload_sha256") != expected_request_sha
        or decision.get("judge_provider") != FEEDBACK_PROVIDER
        or decision.get("judge_model") != FEEDBACK_MODEL
        or not isinstance(decision.get("is_correct"), bool)
        or decision.get("evaluation_key") != task.coordinate.evaluation_key
        or decision.get("turn") != turn_name
    ):
        raise PublicResumeError(f"feedback decision does not bind request: {path}")
    return decision


def _safe_output_file(output_root: Path, raw_relative: object) -> Path:
    if not isinstance(raw_relative, str) or not raw_relative or "\\" in raw_relative:
        raise PublicResumeError("output record path is malformed")
    relative = Path(raw_relative)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != raw_relative
    ):
        raise PublicResumeError("output record path is unsafe")
    try:
        root = output_root.resolve(strict=True)
        path = (output_root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise PublicResumeError(f"output record path cannot be resolved: {raw_relative}") from error
    if (output_root / relative).is_symlink() or not path.is_file():
        raise PublicResumeError(f"output record is absent or non-regular: {raw_relative}")
    return path


def _validated_dispatch_record_location(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    raw_relative: object,
    expected_turn: str,
    expected_filename: str,
) -> tuple[Path, int, str, int]:
    path = _safe_output_file(output_root, raw_relative)
    relative = path.relative_to(output_root.resolve(strict=True))
    parts = relative.parts
    if (
        len(parts) != 7
        or parts[:4]
        != (
            "raw_api",
            route.route_key,
            task.coordinate.evaluation_key,
            expected_turn,
        )
        or parts[-1] != expected_filename
    ):
        raise PublicResumeError("dispatch record path differs from its coordinate")
    attempt_match = re.fullmatch(
        r"attempt-([0-9]{2})-(full|smart_truncate_v1|smart_truncate_v2)", parts[4]
    )
    dispatch_match = re.fullmatch(r"dispatch-([0-9]{2})", parts[5])
    if attempt_match is None or dispatch_match is None:
        raise PublicResumeError("dispatch record attempt/dispatch path is malformed")
    attempt_number = int(attempt_match.group(1))
    input_profile = attempt_match.group(2)
    dispatch_ordinal = int(dispatch_match.group(1))
    expected_profile = (
        "full"
        if attempt_number <= 4
        else "smart_truncate_v1"
        if attempt_number == 5
        else "smart_truncate_v2"
    )
    if (
        not 1 <= attempt_number <= MAX_ANSWER_ATTEMPTS
        or not 1 <= dispatch_ordinal <= MAX_OPERATIONAL_DISPATCHES
        or input_profile != expected_profile
    ):
        raise PublicResumeError("dispatch record attempt policy differs")
    return path, attempt_number, input_profile, dispatch_ordinal


def _validated_dispatch_intent(
    *,
    path: Path,
    route: RouteSpec,
    task: RecoveryTask,
    expected_turn: str,
    attempt_number: int,
    input_profile: str,
    dispatch_ordinal: int,
    request_sha256: str,
    expected_record_sha256: object,
) -> dict[str, Any]:
    intent = read_signed_json(path.parent / "intent.json")
    if (
        expected_record_sha256 != intent.get("record_sha256")
        or intent.get("kind") != "hsle_public_openrouter_dispatch_intent_v1"
        or intent.get("evaluation_key") != task.coordinate.evaluation_key
        or intent.get("coordinate_key") != task.coordinate.key
        or intent.get("route_key") != route.route_key
        or intent.get("scientific_model_id") != route.scientific_model_id
        or intent.get("provider_requested_model_id") != route.provider_requested_model_id
        or intent.get("provider_tag") != route.provider_tag
        or intent.get("turn") != expected_turn
        or intent.get("answer_attempt") != attempt_number
        or intent.get("input_profile") != input_profile
        or intent.get("operational_dispatch") != dispatch_ordinal
        or intent.get("request_sha256") != request_sha256
        or intent.get("credential_environment_name_persisted") is not False
        or intent.get("credential_value_persisted") is not False
        or intent.get("automatic_replay_authorized") is not False
    ):
        raise PublicResumeError("dispatch intent binding differs")
    return intent


def _validated_response_record(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    raw_relative: object,
    expected_turn: str,
    expected_request_sha256: str | None = None,
) -> tuple[dict[str, Any], int, str]:
    """Validate a settled response, its intent, and its immutable raw body."""

    path, attempt_number, input_profile, dispatch_ordinal = _validated_dispatch_record_location(
        output_root=output_root,
        route=route,
        task=task,
        raw_relative=raw_relative,
        expected_turn=expected_turn,
        expected_filename="response.json",
    )

    response = read_signed_json(path)
    request_sha256 = response.get("request_sha256")
    status = response.get("http_status")
    if (
        response.get("kind") != "hsle_public_openrouter_response_v1"
        or response.get("evaluation_key") != task.coordinate.evaluation_key
        or response.get("route_key") != route.route_key
        or response.get("turn") != expected_turn
        or response.get("answer_attempt") != attempt_number
        or response.get("input_profile") != input_profile
        or response.get("operational_dispatch") != dispatch_ordinal
        or not isinstance(request_sha256, str)
        or _SHA256.fullmatch(request_sha256) is None
        or (expected_request_sha256 is not None and request_sha256 != expected_request_sha256)
        or isinstance(status, bool)
        or not isinstance(status, int)
        or not 200 <= status < 300
    ):
        raise PublicResumeError("response record core invariants differ")

    _validated_dispatch_intent(
        path=path,
        route=route,
        task=task,
        expected_turn=expected_turn,
        attempt_number=attempt_number,
        input_profile=input_profile,
        dispatch_ordinal=dispatch_ordinal,
        request_sha256=request_sha256,
        expected_record_sha256=response.get("intent_record_sha256"),
    )

    expected_raw_relative = (
        (path.parent / "raw_response.bin").relative_to(output_root.resolve(strict=True)).as_posix()
    )
    if response.get("raw_response_path") != expected_raw_relative:
        raise PublicResumeError("response raw-body path differs")
    raw_path = _safe_output_file(output_root, expected_raw_relative)
    raw_sha256 = file_sha256(raw_path)
    if response.get("raw_response_sha256") != raw_sha256:
        raise PublicResumeError("response raw-body SHA-256 differs")
    try:
        raw_body = json.loads(raw_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicResumeError("response raw body is not JSON") from error
    if not isinstance(raw_body, dict):
        raise PublicResumeError("response raw body is not an object")
    parsed_fields = response.get("parsed_fields")
    if not isinstance(parsed_fields, dict) or parsed_fields != _response_fields(raw_body, route):
        raise PublicResumeError("response parsed fields differ from its raw body")
    return parsed_fields, attempt_number, input_profile


def _validated_rejection_record(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    raw_relative: object,
    expected_turn: str,
    expected_request_sha256: str,
) -> tuple[str, int]:
    """Validate a definitive rejection or accepted provider error before reuse."""

    path, attempt_number, input_profile, dispatch_ordinal = _validated_dispatch_record_location(
        output_root=output_root,
        route=route,
        task=task,
        raw_relative=raw_relative,
        expected_turn=expected_turn,
        expected_filename="rejection.json",
    )
    rejection = read_signed_json(path)
    kind = rejection.get("kind")
    request_sha256 = rejection.get("request_sha256")
    status = rejection.get("http_status")
    body_sha256 = rejection.get("response_body_sha256")
    if (
        kind
        not in {
            "hsle_public_openrouter_definitive_rejection_v1",
            "hsle_public_openrouter_accepted_error_v1",
        }
        or rejection.get("evaluation_key") != task.coordinate.evaluation_key
        or rejection.get("route_key") != route.route_key
        or rejection.get("turn") != expected_turn
        or rejection.get("answer_attempt") != attempt_number
        or rejection.get("operational_dispatch") != dispatch_ordinal
        or not isinstance(request_sha256, str)
        or request_sha256 != expected_request_sha256
        or _SHA256.fullmatch(request_sha256) is None
        or isinstance(status, bool)
        or not isinstance(status, int)
        or not isinstance(body_sha256, str)
        or _SHA256.fullmatch(body_sha256) is None
    ):
        raise PublicResumeError("rejection record core invariants differ")
    _validated_dispatch_intent(
        path=path,
        route=route,
        task=task,
        expected_turn=expected_turn,
        attempt_number=attempt_number,
        input_profile=input_profile,
        dispatch_ordinal=dispatch_ordinal,
        request_sha256=request_sha256,
        expected_record_sha256=rejection.get("intent_record_sha256"),
    )

    raw_path = path.parent / "raw_response.bin"
    if kind == "hsle_public_openrouter_definitive_rejection_v1":
        if (
            not 300 <= status < 600
            or rejection.get("response_body_persisted") is not False
            or raw_path.exists()
            or raw_path.is_symlink()
        ):
            raise PublicResumeError("definitive rejection invariants differ")
    else:
        if not 200 <= status < 300:
            raise PublicResumeError("accepted provider error HTTP status differs")
        raw_relative_path = raw_path.relative_to(output_root.resolve(strict=True)).as_posix()
        verified_raw_path = _safe_output_file(output_root, raw_relative_path)
        if file_sha256(verified_raw_path) != body_sha256:
            raise PublicResumeError("accepted provider error raw-body SHA-256 differs")
        try:
            body = json.loads(verified_raw_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicResumeError("accepted provider error raw body is not JSON") from error
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("error"), dict)
            or body.get("id")
            or rejection.get("provider_error") != body.get("error")
        ):
            raise PublicResumeError("accepted provider error body differs")
    return kind, status


def _validated_envelope_response_record(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    raw_relative: object,
    envelope: PromptEnvelope,
) -> tuple[dict[str, Any], int, str]:
    """Bind a stored response to the exact provider-visible request envelope."""

    _path, attempt_number, _profile, _dispatch = _validated_dispatch_record_location(
        output_root=output_root,
        route=route,
        task=task,
        raw_relative=raw_relative,
        expected_turn=envelope.turn,
        expected_filename="response.json",
    )
    expected_request_sha256 = canonical_json_sha256(
        build_payload(build_attempt_request(envelope, attempt_number), route)
    )
    return _validated_response_record(
        output_root=output_root,
        route=route,
        task=task,
        raw_relative=raw_relative,
        expected_turn=envelope.turn,
        expected_request_sha256=expected_request_sha256,
    )


@dataclass(frozen=True, slots=True)
class _ValidatedFeedbackEvidence:
    response_record: str
    response_fields: dict[str, Any]
    attempt_number: int
    input_profile: str
    decision: dict[str, Any]


def _validate_feedback_evidence(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    source: PromptSource,
    evidence: Mapping[str, Any],
    expected_turn: str,
) -> _ValidatedFeedbackEvidence:
    request_path = (
        output_root
        / "lfe_feedback_requests"
        / route.route_key
        / task.coordinate.evaluation_key
        / f"{expected_turn}.json"
    )
    request = read_signed_json(request_path)
    index = 0 if expected_turn == "example_1" else 1
    example = source.examples[index]
    if (
        request.get("kind") != "hsle_public_lfe_feedback_request_v1"
        or request.get("schema_version") != SCHEMA_VERSION
        or request.get("route_key") != route.route_key
        or request.get("evaluation_key") != task.coordinate.evaluation_key
        or request.get("coordinate_key") != task.coordinate.key
        or request.get("turn") != expected_turn
        or request.get("judge_provider_required") != FEEDBACK_PROVIDER
        or request.get("judge_model_required") != FEEDBACK_MODEL
        or request.get("question") != example.question
        or request.get("corrected_answer") != example.answer
        or not isinstance(request.get("model_response"), str)
        or request.get("source_response_sha256")
        != hashlib.sha256(str(request.get("model_response", "")).encode()).hexdigest()
    ):
        raise PublicResumeError("LFE feedback request differs from its coordinate")
    fields, attempt_number, input_profile = _validated_response_record(
        output_root=output_root,
        route=route,
        task=task,
        raw_relative=request.get("source_response_record"),
        expected_turn=expected_turn,
    )
    if request.get("model_response") != fields.get("text") or not generation_response_is_usable(
        str(fields.get("text", ""))
    ):
        raise PublicResumeError("LFE feedback request differs from provider response")
    decision = _load_feedback_decision(
        output_root=output_root,
        route=route,
        task=task,
        turn_name=expected_turn,
        request=request,
    )
    decision_path = (
        output_root
        / "lfe_feedback_decisions"
        / route.route_key
        / task.coordinate.evaluation_key
        / f"{expected_turn}.json"
    )
    if decision is None or (
        evidence.get("turn") != expected_turn
        or evidence.get("judge_provider") != FEEDBACK_PROVIDER
        or evidence.get("judge_model") != FEEDBACK_MODEL
        or evidence.get("is_correct") is not decision.get("is_correct")
        or evidence.get("feedback_request_record_sha256") != request.get("record_sha256")
        or evidence.get("feedback_decision_file_sha256") != file_sha256(decision_path)
    ):
        raise PublicResumeError("LFE feedback evidence differs from bound decision")
    response_record = request.get("source_response_record")
    if not isinstance(response_record, str):
        raise PublicResumeError("LFE feedback response path is malformed")
    return _ValidatedFeedbackEvidence(
        response_record=response_record,
        response_fields=fields,
        attempt_number=attempt_number,
        input_profile=input_profile,
        decision=decision,
    )


def _validated_lfe_envelope_for_turn(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    source: PromptSource,
    turn_name: str,
    prior_feedback: Sequence[_ValidatedFeedbackEvidence],
) -> PromptEnvelope:
    """Rebuild one LFE turn while proving every prior response request."""

    if turn_name not in {"example_1", "example_2", "target"}:
        raise PublicResumeError("LFE turn name is malformed")
    expected_prior_count = {"example_1": 0, "example_2": 1, "target": 2}[turn_name]
    if len(prior_feedback) != expected_prior_count:
        raise PublicResumeError("LFE prior-feedback count differs from terminal turn")

    use_images = task.coordinate.model_modality == "multimodal"
    messages: list[AdaptiveMessage] = []
    for index, example in enumerate(source.examples, start=1):
        current_turn = f"example_{index}"
        paths = tuple(Path(path) for path in example.image_paths) if use_images else ()
        _append_canonical_message(
            messages,
            _lfe_question_message(
                example.question,
                paths,
                image_label="solved example 1",
            ),
        )
        envelope = PromptEnvelope(
            coordinate=task.coordinate,
            turn=current_turn,
            messages=tuple(messages),
            target_question=example.question,
            target_image_paths=paths,
        )
        if current_turn == turn_name:
            return envelope

        prior = prior_feedback[index - 1]
        fields, attempt_number, input_profile = _validated_envelope_response_record(
            output_root=output_root,
            route=route,
            task=task,
            raw_relative=prior.response_record,
            envelope=envelope,
        )
        if (
            fields != prior.response_fields
            or attempt_number != prior.attempt_number
            or input_profile != prior.input_profile
        ):
            raise PublicResumeError("LFE prior response differs during prompt reconstruction")
        _append_canonical_message(
            messages,
            AdaptiveMessage(
                role="assistant",
                content=str(fields["text"]),
                reasoning_content=str(fields["reasoning"]),
                semantic_kind="assistant_response",
            ),
        )
        _append_canonical_message(
            messages,
            AdaptiveMessage(
                role="user",
                content=lfe_feedback(
                    bool(prior.decision["is_correct"]),
                    example.answer,
                    example.rationale,
                    FeedbackMode.BINARY_ONLY,
                ),
                semantic_kind="feedback",
            ),
        )

    target_paths = source.target_image_paths if use_images else ()
    _append_canonical_message(
        messages,
        _lfe_question_message(
            source.target_question,
            target_paths,
            image_label="target question",
            semantic_kind="target_question",
        ),
    )
    return PromptEnvelope(
        coordinate=task.coordinate,
        turn="target",
        messages=tuple(messages),
        target_question=source.target_question,
        target_image_paths=target_paths,
    )


def _validate_nonresponsive_attempt_chain(
    *,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    envelope: PromptEnvelope,
) -> None:
    """Prove exactly six accepted blank attempts and forbid an attempt seven."""

    turn_root = (
        output_root / "raw_api" / route.route_key / task.coordinate.evaluation_key / envelope.turn
    )
    if turn_root.is_symlink() or not turn_root.is_dir():
        raise PublicResumeError("nonresponsive turn WAL is absent or non-regular")
    expected_attempt_names = {
        f"attempt-{attempt_number:02d}-"
        + (
            "full"
            if attempt_number <= 4
            else "smart_truncate_v1"
            if attempt_number == 5
            else "smart_truncate_v2"
        )
        for attempt_number in range(1, MAX_ANSWER_ATTEMPTS + 1)
    }
    actual_attempts = {path.name for path in turn_root.iterdir()}
    if actual_attempts != expected_attempt_names:
        raise PublicResumeError("nonresponsive turn must contain exactly attempts one through six")

    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt_number in range(1, MAX_ANSWER_ATTEMPTS + 1):
        attempt = build_attempt_request(envelope, attempt_number)
        attempt_root = turn_root / f"attempt-{attempt_number:02d}-{attempt.input_profile}"
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise PublicResumeError("nonresponsive attempt WAL is non-regular")
        dispatch_entries = list(attempt_root.iterdir())
        dispatch_ordinals: list[int] = []
        for dispatch_path in dispatch_entries:
            match = re.fullmatch(r"dispatch-([0-9]{2})", dispatch_path.name)
            if match is None or dispatch_path.is_symlink() or not dispatch_path.is_dir():
                raise PublicResumeError("nonresponsive dispatch WAL is malformed")
            dispatch_ordinals.append(int(match.group(1)))
        dispatch_ordinals.sort()
        if (
            not dispatch_ordinals
            or len(dispatch_ordinals) > MAX_OPERATIONAL_DISPATCHES
            or dispatch_ordinals != list(range(1, len(dispatch_ordinals) + 1))
        ):
            raise PublicResumeError("nonresponsive dispatch sequence is not contiguous")

        expected_request_sha256 = canonical_json_sha256(build_payload(attempt, route))
        for dispatch_ordinal in dispatch_ordinals:
            dispatch_root = attempt_root / f"dispatch-{dispatch_ordinal:02d}"
            state = _existing_dispatch_state(dispatch_root)
            is_last = dispatch_ordinal == dispatch_ordinals[-1]
            if is_last:
                if state != "response":
                    raise PublicResumeError(
                        "nonresponsive attempt lacks its terminal accepted response"
                    )
                fields, observed_attempt, observed_profile = _validated_response_record(
                    output_root=output_root,
                    route=route,
                    task=task,
                    raw_relative=(dispatch_root / "response.json")
                    .relative_to(output_root)
                    .as_posix(),
                    expected_turn=envelope.turn,
                    expected_request_sha256=expected_request_sha256,
                )
                if (
                    observed_attempt != attempt_number
                    or observed_profile != attempt.input_profile
                    or generation_response_is_usable(str(fields.get("text", "")))
                ):
                    raise PublicResumeError(
                        "nonresponsive terminal response is not a bound blank settlement"
                    )
                continue
            if state != "rejection":
                raise PublicResumeError(
                    "nonresponsive attempt has a non-rejection before settlement"
                )
            kind, status = _validated_rejection_record(
                output_root=output_root,
                route=route,
                task=task,
                raw_relative=(dispatch_root / "rejection.json").relative_to(output_root).as_posix(),
                expected_turn=envelope.turn,
                expected_request_sha256=expected_request_sha256,
            )
            if (
                kind != "hsle_public_openrouter_accepted_error_v1"
                and status not in retryable_statuses
            ):
                raise PublicResumeError(
                    "nonresponsive attempt contains a definitive pre-settlement rejection"
                )


def _validate_generation_result(
    *,
    path: Path,
    output_root: Path,
    route: RouteSpec,
    task: RecoveryTask,
    source: PromptSource,
) -> dict[str, Any]:
    record = read_signed_json(path)
    result = record.get("result")
    if (
        record.get("kind") != "hsle_public_openrouter_generation_result_v1"
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("route_key") != route.route_key
        or record.get("scientific_model_id") != route.scientific_model_id
        or record.get("provider_requested_model_id") != route.provider_requested_model_id
        or record.get("provider_tag") != route.provider_tag
        or record.get("provider_display_name") != route.provider_display_name
        or record.get("input_manifest_file_sha256") != EXPECTED_INPUT_MANIFEST_FILE_SHA256
        or record.get("route_vector_sha256") != EXPECTED_VECTOR_SHA256[route.route_key]
        or record.get("route_prompt_vector_sha256")
        != EXPECTED_PROMPT_VECTOR_SHA256[route.route_key]
        or record.get("final_hle_and_closeness_judging_complete") is not False
        or record.get("judge_calls_made_by_public_resume") != 0
        or not isinstance(result, dict)
    ):
        raise PublicResumeError(f"generation result release/route invariants differ: {path}")

    coordinate = task.coordinate
    expected_coordinate_fields = {
        "model_id": route.scientific_model_id,
        "requested_model_id": route.scientific_model_id,
        "model_modality": coordinate.model_modality,
        "evaluation_setting": coordinate.evaluation_setting,
        "concrete_variant": coordinate.concrete_variant,
        "original_question_id": coordinate.original_question_id,
        "setting_instance_id": coordinate.setting_instance_id,
        "evaluation_key": coordinate.evaluation_key,
        "coordinate_key": coordinate.key,
    }
    if any(result.get(name) != value for name, value in expected_coordinate_fields.items()):
        raise PublicResumeError(f"generation result coordinate differs: {path}")
    expected_response_id = (
        "resp_public_"
        + hashlib.sha256(f"{route.route_key}\0{coordinate.evaluation_key}".encode()).hexdigest()[
            :32
        ]
    )
    output = result.get("model_raw_output")
    reasoning = result.get("model_reasoning_content")
    turn_counts = result.get("turn_attempt_counts")
    feedback = result.get("feedback_evidence")
    if (
        result.get("response_id") != expected_response_id
        or result.get("model_parsed_answer") != ""
        or not isinstance(output, str)
        or result.get("model_raw_output_sha256") != hashlib.sha256(output.encode()).hexdigest()
        or not isinstance(reasoning, str)
        or result.get("model_reasoning_content_sha256")
        != hashlib.sha256(reasoning.encode()).hexdigest()
        or not isinstance(turn_counts, dict)
        or result.get("turn_attempt_counts_json")
        != json.dumps(turn_counts, sort_keys=True, separators=(",", ":"))
        or not isinstance(feedback, list)
        or result.get("feedback_evidence_json")
        != json.dumps(feedback, sort_keys=True, separators=(",", ":"))
        or result.get("max_retries") != 5
        or result.get("max_total_attempts") != 6
        or result.get("attempts_1_to_4_input_profile") != "full"
        or result.get("attempt_5_input_profile") != "smart_truncate_v1"
        or result.get("attempt_6_input_profile") != "smart_truncate_v2"
    ):
        raise PublicResumeError(f"generation result content hashes/policy differ: {path}")
    allowed_turns = (
        {"target"}
        if coordinate.evaluation_setting != "learning_from_experience"
        else {"example_1", "example_2", "target"}
    )
    if set(turn_counts) - allowed_turns or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6
        for value in turn_counts.values()
    ):
        raise PublicResumeError(f"generation result turn attempts differ: {path}")
    if coordinate.evaluation_setting != "learning_from_experience" and feedback:
        raise PublicResumeError(f"static generation result contains LFE evidence: {path}")
    if len(feedback) > 2 or any(not isinstance(row, dict) for row in feedback):
        raise PublicResumeError(f"generation result feedback evidence is malformed: {path}")
    validated_feedback = [
        _validate_feedback_evidence(
            output_root=output_root,
            route=route,
            task=task,
            source=source,
            evidence=row,
            expected_turn=f"example_{index}",
        )
        for index, row in enumerate(feedback, start=1)
    ]

    status = result.get("generation_status")
    if status == "success":
        target_envelope = (
            _validated_lfe_envelope_for_turn(
                output_root=output_root,
                route=route,
                task=task,
                source=source,
                turn_name="target",
                prior_feedback=validated_feedback,
            )
            if coordinate.evaluation_setting == "learning_from_experience"
            else preflight_prompt_source(task, source)[0]
        )
        fields, attempt_number, input_profile = _validated_envelope_response_record(
            output_root=output_root,
            route=route,
            task=task,
            raw_relative=record.get("response_record"),
            envelope=target_envelope,
        )
        if (
            not generation_response_is_usable(output)
            or fields.get("text") != output
            or fields.get("reasoning") != reasoning
            or result.get("successful_attempt_number") != attempt_number
            or result.get("successful_input_profile") != input_profile
            or turn_counts.get("target") != attempt_number
            or result.get("failed_turn") != ""
            or result.get("terminal_reason") != ""
            or result.get("score_origin") != "pending_gemini35_judging"
            or result.get("terminal_hle_correctness") is not None
            or result.get("terminal_closeness_score") is not None
            or result.get("gemini_judge_call_required") is not True
            or (
                coordinate.evaluation_setting == "learning_from_experience"
                and (
                    len(feedback) != 2
                    or set(turn_counts) != allowed_turns
                    or any(
                        turn_counts.get(f"example_{index}") != evidence.attempt_number
                        for index, evidence in enumerate(validated_feedback, start=1)
                    )
                )
            )
            or (
                coordinate.evaluation_setting != "learning_from_experience"
                and set(turn_counts) != {"target"}
            )
        ):
            raise PublicResumeError(f"successful generation result invariants differ: {path}")
    elif status == "nonresponsive":
        if (
            record.get("response_record") != ""
            or output != ""
            or reasoning != ""
            or result.get("successful_attempt_number") is not None
            or result.get("successful_input_profile") != ""
            or result.get("failed_turn") != "generation"
            or result.get("terminal_reason")
            != "six settled accepted responses had no usable visible answer"
            or result.get("score_origin") != "generation_nonresponsive_policy"
            or result.get("terminal_hle_correctness") != "incorrect"
            or result.get("terminal_closeness_score") != 0
            or result.get("gemini_judge_call_required") is not False
        ):
            raise PublicResumeError(f"terminal generation result invariants differ: {path}")
        if coordinate.evaluation_setting == "learning_from_experience":
            terminal_turn = ("example_1", "example_2", "target")[len(validated_feedback)]
            expected_count_keys = set(
                ("example_1", "example_2", "target")[: len(validated_feedback) + 1]
            )
            if (
                set(turn_counts) != expected_count_keys
                or turn_counts.get(terminal_turn) != MAX_ANSWER_ATTEMPTS
                or any(
                    turn_counts.get(f"example_{index}") != evidence.attempt_number
                    for index, evidence in enumerate(validated_feedback, start=1)
                )
            ):
                raise PublicResumeError(f"terminal LFE turn accounting differs: {path}")
            terminal_envelope = _validated_lfe_envelope_for_turn(
                output_root=output_root,
                route=route,
                task=task,
                source=source,
                turn_name=terminal_turn,
                prior_feedback=validated_feedback,
            )
        else:
            if set(turn_counts) != {"target"} or turn_counts.get("target") != MAX_ANSWER_ATTEMPTS:
                raise PublicResumeError(f"terminal static turn accounting differs: {path}")
            terminal_envelope = preflight_prompt_source(task, source)[0]
        _validate_nonresponsive_attempt_chain(
            output_root=output_root,
            route=route,
            task=task,
            envelope=terminal_envelope,
        )
    else:
        raise PublicResumeError(f"generation result status differs: {path}")
    return record


def run_coordinate(
    *,
    route: RouteSpec,
    task: RecoveryTask,
    source: PromptSource,
    api_key: str,
    output_root: Path,
) -> str:
    result_path = (
        output_root
        / "generation_results"
        / route.route_key
        / f"{task.coordinate.evaluation_key}.json"
    )
    if result_path.exists() or result_path.is_symlink():
        _validate_generation_result(
            path=result_path,
            output_root=output_root,
            route=route,
            task=task,
            source=source,
        )
        return "already_complete"
    lock_path = (
        output_root
        / "control"
        / "locks"
        / (f"{route.route_key}-{task.coordinate.evaluation_key}.lock")
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if result_path.exists() or result_path.is_symlink():
            _validate_generation_result(
                path=result_path,
                output_root=output_root,
                route=route,
                task=task,
                source=source,
            )
            return "already_complete"
        if task.coordinate.evaluation_setting != "learning_from_experience":
            envelope = preflight_prompt_source(task, source)[0]
            turn = run_turn(
                route=route,
                task=task,
                envelope=envelope,
                api_key=api_key,
                output_root=output_root,
            )
            if turn.status in {"ambiguous", "blocked"}:
                return turn.status
            atomic_write_json(
                result_path,
                _scientific_result(
                    route=route,
                    task=task,
                    turn=turn,
                    turn_attempt_counts={"target": turn.successful_attempt or MAX_ANSWER_ATTEMPTS},
                    feedback_evidence=(),
                ),
            )
            return "complete"

        use_images = task.coordinate.model_modality == "multimodal"
        messages: list[AdaptiveMessage] = []
        feedback_rows: list[Mapping[str, Any]] = []
        counts: dict[str, int] = {}
        for index, example in enumerate(source.examples, start=1):
            turn_name = f"example_{index}"
            paths = tuple(Path(path) for path in example.image_paths) if use_images else ()
            _append_canonical_message(
                messages,
                _lfe_question_message(
                    example.question,
                    paths,
                    image_label="solved example 1",
                ),
            )
            envelope = PromptEnvelope(
                coordinate=task.coordinate,
                turn=turn_name,
                messages=tuple(messages),
                target_question=example.question,
                target_image_paths=paths,
            )
            turn = run_turn(
                route=route,
                task=task,
                envelope=envelope,
                api_key=api_key,
                output_root=output_root,
            )
            counts[turn_name] = turn.successful_attempt or MAX_ANSWER_ATTEMPTS
            if turn.status in {"ambiguous", "blocked"}:
                return turn.status
            if turn.status == "exhausted":
                atomic_write_json(
                    result_path,
                    _scientific_result(
                        route=route,
                        task=task,
                        turn=turn,
                        turn_attempt_counts=counts,
                        feedback_evidence=feedback_rows,
                    ),
                )
                return "complete"
            _append_canonical_message(
                messages,
                AdaptiveMessage(
                    role="assistant",
                    content=turn.text,
                    reasoning_content=turn.reasoning,
                    semantic_kind="assistant_response",
                ),
            )
            request_path = (
                output_root
                / "lfe_feedback_requests"
                / route.route_key
                / task.coordinate.evaluation_key
                / f"{turn_name}.json"
            )
            if request_path.is_file():
                feedback_request = read_json(request_path)
                if (
                    feedback_request.get("evaluation_key") != task.coordinate.evaluation_key
                    or feedback_request.get("turn") != turn_name
                    or feedback_request.get("source_response_sha256")
                    != hashlib.sha256(turn.text.encode()).hexdigest()
                    or feedback_request.get("judge_provider_required") != FEEDBACK_PROVIDER
                    or feedback_request.get("judge_model_required") != FEEDBACK_MODEL
                ):
                    raise PublicResumeError(f"existing feedback request differs: {request_path}")
            else:
                feedback_request = atomic_write_json(
                    request_path,
                    _feedback_request(
                        route=route,
                        task=task,
                        turn_name=turn_name,
                        question=example.question,
                        corrected_answer=example.answer,
                        model_response=turn.text,
                        transcript=_source_transcript(messages),
                        turn=turn,
                    ),
                )
            decision = _load_feedback_decision(
                output_root=output_root,
                route=route,
                task=task,
                turn_name=turn_name,
                request=feedback_request,
            )
            if decision is None:
                return f"waiting_feedback_{index}"
            feedback_rows.append(
                {
                    "turn": turn_name,
                    "judge_provider": decision["judge_provider"],
                    "judge_model": decision["judge_model"],
                    "is_correct": decision["is_correct"],
                    "feedback_request_record_sha256": feedback_request["record_sha256"],
                    "feedback_decision_file_sha256": file_sha256(
                        output_root
                        / "lfe_feedback_decisions"
                        / route.route_key
                        / task.coordinate.evaluation_key
                        / f"{turn_name}.json"
                    ),
                }
            )
            _append_canonical_message(
                messages,
                AdaptiveMessage(
                    role="user",
                    content=lfe_feedback(
                        bool(decision["is_correct"]),
                        example.answer,
                        example.rationale,
                        FeedbackMode.BINARY_ONLY,
                    ),
                    semantic_kind="feedback",
                ),
            )

        target_paths = source.target_image_paths if use_images else ()
        _append_canonical_message(
            messages,
            _lfe_question_message(
                source.target_question,
                target_paths,
                image_label="target question",
                semantic_kind="target_question",
            ),
        )
        target = run_turn(
            route=route,
            task=task,
            envelope=PromptEnvelope(
                coordinate=task.coordinate,
                turn="target",
                messages=tuple(messages),
                target_question=source.target_question,
                target_image_paths=target_paths,
            ),
            api_key=api_key,
            output_root=output_root,
        )
        counts["target"] = target.successful_attempt or MAX_ANSWER_ATTEMPTS
        if target.status in {"ambiguous", "blocked"}:
            return target.status
        atomic_write_json(
            result_path,
            _scientific_result(
                route=route,
                task=task,
                turn=target,
                turn_attempt_counts=counts,
                feedback_evidence=feedback_rows,
            ),
        )
        return "complete"


def prepare(
    *,
    project_root: Path,
    output_root: Path,
    api_key_environment_name: str,
    partition: str,
    shard_count: int,
) -> dict[str, Any]:
    validate_api_key_environment_name(api_key_environment_name, require_value=True)
    validate_partition(partition)
    if shard_count <= 0:
        raise PublicResumeError("shard count must be positive")
    inputs, input_audit = ensure_inputs(project_root)
    official_hle_validation = validate_optional_official_hle_access()
    route_rows = {}
    for route_key in ROUTES:
        tasks = load_route_tasks(inputs, route_key)
        sizes = [
            sum(stable_shard(task, shard_count) == i for task in tasks) for i in range(shard_count)
        ]
        route_rows[route_key] = {
            "task_count": len(tasks),
            "evaluation_key_vector_sha256": EXPECTED_VECTOR_SHA256[route_key],
            "shard_sizes": sizes,
            "scientific_model_id": ROUTES[route_key].scientific_model_id,
            "provider_requested_model_id": (ROUTES[route_key].provider_requested_model_id),
            "provider_tag": ROUTES[route_key].provider_tag,
            "predecessor_failed_request_model_id": (
                ROUTES[route_key].predecessor_failed_request_model_id
            ),
        }
    api_key = os.environ[api_key_environment_name]
    try:
        openrouter_live_validation = validate_openrouter_live_contract(api_key)
    finally:
        api_key = ""
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for relative in (
        "raw_api",
        "generation_results",
        "lfe_feedback_requests",
        "lfe_feedback_decisions",
        "control/locks",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
    record = atomic_write_json(
        output_root / "control" / "prepare.json",
        {
            "kind": "hsle_public_openrouter_resume_prepare_v1",
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": utc_now(),
            "output_directory_name": output_root.name,
            "partition": partition,
            "shard_count": shard_count,
            "api_key_environment_name": api_key_environment_name,
            "api_key_value_persisted": False,
            "judge_calls_authorized": False,
            "routes": route_rows,
            "input_source": "authorized_hsle_input_root",
            "input_root_path_persisted": False,
            "input_audit": input_audit,
            "official_hle_access_validation": official_hle_validation,
            "openrouter_live_validation": openrouter_live_validation,
        },
    )
    return record


def worker(
    *,
    project_root: Path,
    output_root: Path,
    api_key_environment_name: str,
    route_key: str,
    shard_count: int,
    shard_index: int,
) -> dict[str, Any]:
    validate_api_key_environment_name(api_key_environment_name, require_value=True)
    if route_key not in ROUTES or _ROUTE_KEY.fullmatch(route_key) is None:
        raise PublicResumeError("route key is not one of the five frozen routes")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise PublicResumeError("invalid shard index/count")
    api_key = os.environ[api_key_environment_name]
    inputs, _audit = ensure_inputs(project_root, full_scientific_validation=False)
    tasks = tuple(
        task
        for task in load_route_tasks(inputs, route_key)
        if stable_shard(task, shard_count) == shard_index
    )
    originals, links = load_prompt_source_tables(
        inputs / "processed" / "hsle_original_questions.csv",
        inputs / "processed" / "hsle_question_example_links.csv",
    )
    counts: dict[str, int] = {}
    route = ROUTES[route_key]
    for task in tasks:
        source = load_public_prompt_source(
            task,
            originals_by_id=originals,
            links_by_id=links,
            inputs=inputs,
        )
        status = run_coordinate(
            route=route,
            task=task,
            source=source,
            api_key=api_key,
            output_root=output_root,
        )
        counts[status] = counts.get(status, 0) + 1
    api_key = ""
    record = atomic_write_json(
        output_root
        / "control"
        / "worker_summaries"
        / route_key
        / f"shard-{shard_index:02d}-of-{shard_count:02d}.json",
        {
            "kind": "hsle_public_openrouter_resume_worker_summary_v1",
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": utc_now(),
            "route_key": route_key,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "owned_task_count": len(tasks),
            "status_counts": counts,
            "api_key_environment_name": api_key_environment_name,
            "api_key_value_persisted": False,
            "judge_call_count": 0,
        },
    )
    return record


def finalize(*, project_root: Path, output_root: Path) -> dict[str, Any]:
    inputs, _audit = ensure_inputs(project_root, full_scientific_validation=False)
    try:
        output_root = output_root.resolve(strict=True)
    except OSError as error:
        raise PublicResumeError("public resume output root is absent") from error
    originals, links = load_prompt_source_tables(
        inputs / "processed" / "hsle_original_questions.csv",
        inputs / "processed" / "hsle_question_example_links.csv",
    )
    generation_root = output_root / "generation_results"
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise PublicResumeError("generation-results directory is absent or non-regular")
    unexpected_route_directories = {
        path.name for path in generation_root.iterdir() if path.name not in ROUTES
    }
    if unexpected_route_directories:
        raise PublicResumeError(
            "generation-results contains unexpected route entries: "
            + ", ".join(sorted(unexpected_route_directories))
        )
    status: dict[str, dict[str, int]] = {}
    for route_key, route in ROUTES.items():
        route_status = {
            "frozen_callable": EXPECTED_VECTOR_COUNTS[route_key],
            "generation_results": 0,
            "waiting_feedback_example_1": 0,
            "waiting_feedback_example_2": 0,
            "paid_ambiguity": 0,
            "without_generation_result": 0,
        }
        tasks = load_route_tasks(inputs, route_key)
        tasks_by_key = {task.coordinate.evaluation_key: task for task in tasks}
        result_directory = generation_root / route_key
        if result_directory.exists() or result_directory.is_symlink():
            if result_directory.is_symlink() or not result_directory.is_dir():
                raise PublicResumeError(
                    f"generation-results route directory is non-regular: {route_key}"
                )
            expected_names = {f"{key}.json" for key in tasks_by_key}
            actual_names = {path.name for path in result_directory.iterdir()}
            unexpected = actual_names - expected_names
            if unexpected:
                raise PublicResumeError(
                    f"generation-results contains unexpected files for {route_key}: "
                    + ", ".join(sorted(unexpected)[:5])
                )
        for task in tasks:
            key = task.coordinate.evaluation_key
            result_path = result_directory / f"{key}.json"
            source: PromptSource | None = None
            if result_path.exists() or result_path.is_symlink():
                source = load_public_prompt_source(
                    task,
                    originals_by_id=originals,
                    links_by_id=links,
                    inputs=inputs,
                )
                _validate_generation_result(
                    path=result_path,
                    output_root=output_root,
                    route=route,
                    task=task,
                    source=source,
                )
                route_status["generation_results"] += 1
            else:
                for index in (2, 1):
                    turn_name = f"example_{index}"
                    request_path = (
                        output_root
                        / "lfe_feedback_requests"
                        / route_key
                        / key
                        / f"{turn_name}.json"
                    )
                    if not (request_path.exists() or request_path.is_symlink()):
                        continue
                    request = read_signed_json(request_path)
                    if (
                        task.coordinate.evaluation_setting != "learning_from_experience"
                        or request.get("kind") != "hsle_public_lfe_feedback_request_v1"
                        or request.get("schema_version") != SCHEMA_VERSION
                        or request.get("route_key") != route_key
                        or request.get("evaluation_key") != key
                        or request.get("coordinate_key") != task.coordinate.key
                        or request.get("turn") != turn_name
                        or request.get("judge_provider_required") != FEEDBACK_PROVIDER
                        or request.get("judge_model_required") != FEEDBACK_MODEL
                    ):
                        raise PublicResumeError(
                            f"waiting LFE feedback request differs: {request_path}"
                        )
                    route_status[f"waiting_feedback_example_{index}"] += 1
                    break
            ambiguities = sorted(
                (output_root / "raw_api" / route_key / key).glob("**/ambiguity.json")
            )
            for ambiguity_path in ambiguities:
                ambiguity = read_signed_json(ambiguity_path)
                if (
                    ambiguity.get("kind") != "hsle_public_openrouter_paid_ambiguity_v1"
                    or ambiguity.get("route_key") != route_key
                    or ambiguity.get("evaluation_key") != key
                    or ambiguity.get("automatic_replay_authorized") is not False
                ):
                    raise PublicResumeError(f"paid-ambiguity record differs: {ambiguity_path}")
            if ambiguities:
                route_status["paid_ambiguity"] += 1
        route_status["without_generation_result"] = (
            route_status["frozen_callable"] - route_status["generation_results"]
        )
        status[route_key] = route_status
    inventory = []
    for path in sorted(output_root.rglob("*")):
        if path.is_symlink():
            raise PublicResumeError(f"public resume output contains a symbolic link: {path}")
        if path.is_file() and path.name != "RUN_MANIFEST.json":
            inventory.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    record = atomic_write_json(
        output_root / "RUN_MANIFEST.json",
        {
            "kind": "hsle_public_openrouter_resume_transfer_manifest_v1",
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": utc_now(),
            "routes": {key: asdict(value) for key, value in ROUTES.items()},
            "status": status,
            "files": inventory,
            "judge_call_count": 0,
            "contains_openrouter_api_key": False,
            "transfer_instruction": "Share this entire needs_to_be_judged directory.",
        },
    )
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "worker", "finalize"):
        child = sub.add_parser(name)
        child.add_argument("--project-root", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        if name in {"prepare", "worker"}:
            child.add_argument("--api-key-env", required=True)
        if name == "prepare":
            child.add_argument("--partition", required=True)
            child.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
        if name == "worker":
            child.add_argument("--route", choices=tuple(ROUTES), required=True)
            child.add_argument("--shard-count", type=int, required=True)
            child.add_argument("--shard-index", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            project_root=args.project_root,
            output_root=args.output_root,
            api_key_environment_name=args.api_key_env,
            partition=args.partition,
            shard_count=args.shard_count,
        )
    elif args.command == "worker":
        result = worker(
            project_root=args.project_root,
            output_root=args.output_root,
            api_key_environment_name=args.api_key_env,
            route_key=args.route,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    else:
        result = finalize(project_root=args.project_root, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublicResumeError as error:
        print(f"public resume error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
