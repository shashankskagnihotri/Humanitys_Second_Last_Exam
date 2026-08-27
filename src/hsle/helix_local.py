"""Pinned, no-replay HSLE generation against a local Helix vLLM service."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import mimetypes
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

from hsle.download_data import (
    CONSOLIDATED_DATASET_SHA256,
    DATA_PATTERNS,
    PUBLIC_DATASET_REPO_ID,
    PUBLIC_DATASET_REVISION,
    validate_dataset,
)
from hsle.prompts import (
    ContextExample,
    one_shot_prompt,
    render_hle_judge_prompt,
    two_shot_prompt,
    zero_shot_prompt,
)


SCHEMA_VERSION = 1
MAX_OUTPUT_TOKENS = 4096
MAX_MODEL_LEN = 32768
SEED = 0
TEMPERATURE = 0.0
TOP_P = 1.0
GEMINI_FEEDBACK_MODEL = "gemini-3.5-flash"
EXPECTED_IMAGE_COUNT = 258
GEMINI_FEEDBACK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "strict": {"type": "boolean"},
    },
    "required": [
        "extracted_final_answer",
        "reasoning",
        "correct",
        "confidence",
        "strict",
    ],
    "additionalProperties": False,
}
GEMINI_BLOCK_NONE_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)


@dataclass(frozen=True)
class ModelSpec:
    route: str
    model_id: str
    revision: str
    modality: str
    expected_coordinates: int
    vllm_version: str
    family: str
    checkpoint_weight_bytes: int | None = None
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    reasoning_effort: str | None = None
    preserve_thinking: bool = False


MODEL_SPECS: dict[str, ModelSpec] = {
    "kimi_k2_thinking": ModelSpec(
        route="kimi_k2_thinking",
        model_id="moonshotai/Kimi-K2-Thinking",
        revision="a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55",
        modality="text_only",
        expected_coordinates=2_085,
        vllm_version="0.25.1",
        family="kimi",
    ),
    "kimi_k25": ModelSpec(
        route="kimi_k25",
        model_id="moonshotai/Kimi-K2.5",
        revision="4d01dfe0332d63057c186e0b262165819efb6611",
        modality="multimodal",
        expected_coordinates=2_455,
        vllm_version="0.25.1",
        family="kimi",
    ),
    "kimi_k26": ModelSpec(
        route="kimi_k26",
        model_id="moonshotai/Kimi-K2.6",
        revision="7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        modality="multimodal",
        expected_coordinates=2_455,
        vllm_version="0.25.1",
        family="kimi",
    ),
    "kimi_k3": ModelSpec(
        route="kimi_k3",
        model_id="moonshotai/Kimi-K3",
        revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        modality="multimodal",
        expected_coordinates=2_455,
        vllm_version="0.27.1",
        family="kimi",
        checkpoint_weight_bytes=1_560_936_091_448,
        reasoning_effort="max",
        preserve_thinking=True,
    ),
    "qwen38_27b": ModelSpec(
        route="qwen38_27b",
        model_id="Qwen/Qwen3.8-27B",
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        modality="multimodal",
        expected_coordinates=2_455,
        vllm_version="0.25.1",
        family="qwen",
        checkpoint_weight_bytes=55_563_006_776,
        reasoning_effort="xhigh",
        preserve_thinking=True,
    ),
}


class HelixBenchmarkError(RuntimeError):
    """A pinned input, runtime, or no-replay invariant differs."""


class CapturedRequestError(HelixBenchmarkError):
    """A single HTTP response was unusable but its received bytes were retained."""

    def __init__(self, message: str, raw_response: object) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _captured_wire_body(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "body_encoding": "base64",
            "body_bytes": len(raw),
            "body_base64": base64.b64encode(raw).decode("ascii"),
        }


def _read_json_http_response(response: Any, context: str) -> object:
    raw = response.read()
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapturedRequestError(
            f"{context} returned malformed JSON", _captured_wire_body(raw)
        ) from exc


def _captured_http_error(error: urllib.error.HTTPError) -> object:
    return {
        "http_status": error.code,
        "http_reason": str(error.reason),
        "response_body": _captured_wire_body(error.read()),
    }


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    return {**unsigned, "record_sha256": _sha256_bytes(_canonical(unsigned))}


def _write_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _signed(value)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise HelixBenchmarkError(f"refusing to replace immutable artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return payload


def _load_signed(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HelixBenchmarkError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise HelixBenchmarkError(f"JSON artifact is not an object: {path}")
    expected = value.get("record_sha256")
    if expected != _signed(value)["record_sha256"]:
        raise HelixBenchmarkError(f"record SHA-256 mismatch: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() == "true"


def _raw_path_values(value: object) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise HelixBenchmarkError("image-path JSON must be a list")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.replace("|", ";").split(";") if item.strip()]


def _normalized_image_paths(data_root: Path, value: object) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in _raw_path_values(value):
        if raw.startswith("data/image/"):
            raw = "data/images/" + raw.removeprefix("data/image/")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise HelixBenchmarkError(f"unsafe image reference: {raw!r}")
        parts = relative.parts[1:] if relative.parts[:1] == ("data",) else relative.parts
        if not parts or parts[0] != "images":
            raise HelixBenchmarkError(f"image reference leaves images/: {raw!r}")
        candidate = data_root.joinpath(*parts).resolve(strict=False)
        try:
            candidate.relative_to(data_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise HelixBenchmarkError(f"image reference leaves dataset root: {raw!r}") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise HelixBenchmarkError(f"referenced image is absent or unsafe: {raw!r}")
        candidate = candidate.resolve(strict=True)
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return tuple(resolved)


def _image_records(paths: Sequence[Path], data_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(data_root.resolve(strict=True)).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _validate_inputs(data_root: Path) -> dict[str, Any]:
    validate_dataset(data_root)
    originals_path = data_root / "processed" / "hsle_original_questions.csv"
    contexts_path = data_root / "processed" / "hsle_context_examples.csv"
    links_path = data_root / "processed" / "hsle_question_example_links.csv"
    originals = _read_csv(originals_path)
    contexts = _read_csv(contexts_path)
    links = _read_csv(links_path)
    if (len(originals), len(contexts), len(links)) != (491, 982, 491):
        raise HelixBenchmarkError("HSLE row counts differ from 491/982/491")

    originals_by_id = {row["original_question_id"]: row for row in originals}
    contexts_by_id = {row["question_id"]: row for row in contexts}
    links_by_id = {row["original_question_id"]: row for row in links}
    if not (
        len(originals_by_id) == 491
        and len(contexts_by_id) == 982
        and len(links_by_id) == 491
        and set(originals_by_id) == set(links_by_id)
    ):
        raise HelixBenchmarkError("target/context/link ID uniqueness or coverage differs")

    context_counts: dict[str, int] = {}
    for row in contexts:
        target_id = row["original_question_id"]
        context_counts[target_id] = context_counts.get(target_id, 0) + 1
    if set(context_counts.values()) != {2} or set(context_counts) != set(originals_by_id):
        raise HelixBenchmarkError("every target must have exactly two context rows")

    for target_id, link in links_by_id.items():
        for index in (1, 2):
            context_id = link[f"example_{index}_question_id"]
            context = contexts_by_id.get(context_id)
            if context is None or context["original_question_id"] != target_id:
                raise HelixBenchmarkError(f"missing linked context: {target_id} example {index}")
            for link_field, context_field in (
                (f"example_{index}_question", "question"),
                (f"example_{index}_answer", "answer"),
                (f"example_{index}_rationale", "rationale"),
                (f"example_{index}_has_image", "has_image"),
                (f"example_{index}_image_paths_or_ids", "image_paths_or_ids"),
            ):
                if link[link_field] != context[context_field]:
                    raise HelixBenchmarkError(
                        f"context linkage field differs: {target_id} {link_field}"
                    )

    references: set[Path] = set()
    target_image_rows = 0
    context_image_rows = 0
    for row in originals:
        paths = _normalized_image_paths(data_root, row["image_paths_or_ids"])
        if _truthy(row["has_image"]) != bool(paths):
            raise HelixBenchmarkError("target image flag differs from normalized paths")
        target_image_rows += bool(paths)
        references.update(paths)
    for row in contexts:
        paths = _normalized_image_paths(data_root, row["image_paths_or_ids"])
        if _truthy(row["has_image"]) != bool(paths):
            raise HelixBenchmarkError("context image flag differs from normalized paths")
        context_image_rows += bool(paths)
        references.update(paths)
    actual_images = {
        path.resolve(strict=True)
        for path in (data_root / "images").iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if len(actual_images) != EXPECTED_IMAGE_COUNT or references != actual_images:
        raise HelixBenchmarkError(
            "normalized image universe differs from the exact 258-file dataset inventory"
        )
    image_authority = sorted(
        _image_records(tuple(actual_images), data_root), key=lambda item: item["path"]
    )
    return {
        "originals": originals,
        "contexts": contexts,
        "links": links,
        "originals_by_id": originals_by_id,
        "contexts_by_id": contexts_by_id,
        "links_by_id": links_by_id,
        "authority": {
            "consolidated_sha256": CONSOLIDATED_DATASET_SHA256,
            "originals_count": len(originals),
            "contexts_count": len(contexts),
            "links_count": len(links),
            "target_image_rows": target_image_rows,
            "target_text_rows": len(originals) - target_image_rows,
            "context_image_rows": context_image_rows,
            "image_count": len(actual_images),
            "image_authority_sha256": _sha256_bytes(_canonical(image_authority)),
            "processed_files": {
                path.name: _sha256(path)
                for path in (originals_path, contexts_path, links_path)
            },
        },
    }


def _snapshot_weight_inventory(snapshot: Path) -> tuple[list[dict[str, Any]], int]:
    weights = sorted(
        (
            path
            for path in snapshot.rglob("*.safetensors")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(snapshot).as_posix(),
    )
    inventory = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in weights
    ]
    return inventory, sum(item["bytes"] for item in inventory)


def _validate_prepared_model_snapshot(
    spec: ModelSpec, preparation: Mapping[str, Any]
) -> Path:
    model = preparation.get("model")
    if not isinstance(model, Mapping):
        raise HelixBenchmarkError("preparation authority lacks model metadata")
    try:
        snapshot = Path(str(model["root"])).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise HelixBenchmarkError("prepared model snapshot is absent") from exc
    for required in ("config.json", "tokenizer_config.json"):
        path = snapshot / required
        if not path.is_file():
            raise HelixBenchmarkError(f"prepared model snapshot lacks {required}")
    weights, weight_bytes = _snapshot_weight_inventory(snapshot)
    expected_inventory = model.get("weight_file_inventory")
    if weights != expected_inventory:
        raise HelixBenchmarkError("prepared checkpoint weight digests changed")
    if len(weights) != model.get("weight_file_count"):
        raise HelixBenchmarkError("prepared checkpoint weight-file count changed")
    if weight_bytes != model.get("weight_bytes"):
        raise HelixBenchmarkError("prepared checkpoint weight byte count changed")
    if _sha256_bytes(_canonical(weights)) != model.get("weight_inventory_sha256"):
        raise HelixBenchmarkError("prepared checkpoint inventory authority changed")
    if _sha256(snapshot / "config.json") != model.get("config_sha256"):
        raise HelixBenchmarkError("prepared model config digest changed")
    if _sha256(snapshot / "tokenizer_config.json") != model.get(
        "tokenizer_config_sha256"
    ):
        raise HelixBenchmarkError("prepared tokenizer config digest changed")
    if spec.checkpoint_weight_bytes is not None and weight_bytes != spec.checkpoint_weight_bytes:
        raise HelixBenchmarkError(
            f"{spec.model_id} weight bytes {weight_bytes} != {spec.checkpoint_weight_bytes}"
        )
    return snapshot


def prepare_workspace(spec: ModelSpec, workspace: Path, repo_root: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise HelixBenchmarkError("huggingface-hub is required in the preparation venv") from exc

    workspace = workspace.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    data_root = workspace / "dataset" / PUBLIC_DATASET_REVISION
    snapshot = workspace / "models" / spec.route / spec.revision
    control = workspace / "control" / spec.route
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    dataset_path = Path(
        snapshot_download(
            repo_id=PUBLIC_DATASET_REPO_ID,
            repo_type="dataset",
            revision=PUBLIC_DATASET_REVISION,
            local_dir=data_root,
            allow_patterns=list(DATA_PATTERNS),
            token=False,
        )
    ).resolve(strict=True)
    input_authority = _validate_inputs(dataset_path)["authority"]

    info = HfApi(token=False).model_info(spec.model_id, revision=spec.revision)
    if str(info.sha) != spec.revision:
        raise HelixBenchmarkError(
            f"model revision resolved to {info.sha}, expected {spec.revision}"
        )
    snapshot_path = Path(
        snapshot_download(
            repo_id=spec.model_id,
            repo_type="model",
            revision=spec.revision,
            local_dir=snapshot,
            token=False,
        )
    ).resolve(strict=True)
    for required in ("config.json", "tokenizer_config.json"):
        if not (snapshot_path / required).is_file():
            raise HelixBenchmarkError(f"pinned model snapshot lacks {required}")
    weights, weight_bytes = _snapshot_weight_inventory(snapshot_path)
    if not weights:
        raise HelixBenchmarkError("pinned model snapshot contains no safetensors weights")
    if spec.checkpoint_weight_bytes is not None and weight_bytes != spec.checkpoint_weight_bytes:
        raise HelixBenchmarkError(
            f"{spec.model_id} weight bytes {weight_bytes} != {spec.checkpoint_weight_bytes}"
        )

    manifest_path = control / "preparation.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hsle_helix_local_preparation_v1",
        "created_at": _now(),
        "route": spec.route,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "model_modality": spec.modality,
        "expected_coordinates": spec.expected_coordinates,
        "vllm_version": spec.vllm_version,
        "transformers_version": importlib.metadata.version("transformers"),
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "dataset": {
            "repo_id": PUBLIC_DATASET_REPO_ID,
            "revision": PUBLIC_DATASET_REVISION,
            "root": str(dataset_path),
            "authority": input_authority,
        },
        "model": {
            "root": str(snapshot_path),
            "weight_file_count": len(weights),
            "weight_bytes": weight_bytes,
            "weight_file_inventory": weights,
            "weight_inventory_sha256": _sha256_bytes(_canonical(weights)),
            "config_sha256": _sha256(snapshot_path / "config.json"),
            "tokenizer_config_sha256": _sha256(snapshot_path / "tokenizer_config.json"),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "node": os.environ.get("SLURMD_NODENAME", ""),
        },
        "credentials_persisted": False,
    }
    if manifest_path.exists():
        existing = _load_signed(manifest_path)
        stable_fields = (
            "route",
            "model_id",
            "model_revision",
            "model_modality",
            "expected_coordinates",
            "vllm_version",
            "transformers_version",
            "workspace",
            "dataset",
            "model",
        )
        if any(existing.get(field) != manifest.get(field) for field in stable_fields):
            raise HelixBenchmarkError("existing preparation authority differs")
        return existing
    return _write_once(manifest_path, manifest)


def verify_prepared_workspace(spec: ModelSpec, workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    preparation_path = workspace / "control" / spec.route / "preparation.json"
    preparation = _load_signed(preparation_path)
    if (
        preparation.get("route") != spec.route
        or preparation.get("model_id") != spec.model_id
        or preparation.get("model_revision") != spec.revision
        or preparation.get("vllm_version") != spec.vllm_version
    ):
        raise HelixBenchmarkError("preparation authority differs from requested route")
    snapshot = _validate_prepared_model_snapshot(spec, preparation)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id.isdigit():
        raise HelixBenchmarkError("prepared-model verification must run inside Slurm")
    verification = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hsle_helix_model_snapshot_verification_v1",
        "created_at": _now(),
        "route": spec.route,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "model_root": str(snapshot),
        "preparation_record_sha256": preparation["record_sha256"],
        "weight_inventory_sha256": preparation["model"]["weight_inventory_sha256"],
        "config_sha256": preparation["model"]["config_sha256"],
        "tokenizer_config_sha256": preparation["model"]["tokenizer_config_sha256"],
        "slurm_job_id": slurm_job_id,
        "slurm_node": os.environ.get("SLURMD_NODENAME", ""),
    }
    verification_path = (
        workspace
        / "control"
        / spec.route
        / "model_verifications"
        / f"slurm_{slurm_job_id}.json"
    )
    if verification_path.exists():
        existing = _load_signed(verification_path)
        stable_fields = tuple(key for key in verification if key != "created_at")
        if any(existing.get(field) != verification.get(field) for field in stable_fields):
            raise HelixBenchmarkError("existing model-snapshot verification differs")
        return existing
    return _write_once(verification_path, verification)


def _evaluation_key(model_id: str, setting: str, target_id: str, instance: str) -> str:
    raw = json.dumps(
        [model_id, setting, target_id, instance],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return "eval_" + hashlib.sha256(raw).hexdigest()[:24]


def _coordinates(spec: ModelSpec, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    coordinates: list[dict[str, Any]] = []
    originals = list(inputs["originals"])
    if spec.modality == "text_only":
        originals = [row for row in originals if not _truthy(row["has_image"])]
    for original in originals:
        target_id = original["original_question_id"]
        link = inputs["links_by_id"][target_id]
        example_1 = link["example_1_question_id"]
        example_2 = link["example_2_question_id"]
        joined = f"{example_1};{example_2}"
        variants = (
            ("zero_shot", "zero_shot", "", ""),
            ("one_shot_a", "one_shot", example_1, "1"),
            ("one_shot_b", "one_shot", example_2, "2"),
            ("two_shot", "two_shot", joined, ""),
            ("learning_from_experience", "learning_from_experience", joined, ""),
        )
        for variant, setting, instance, shot_index in variants:
            coordinates.append(
                {
                    "evaluation_key": _evaluation_key(
                        spec.model_id, setting, target_id, instance
                    ),
                    "evaluation_setting": setting,
                    "concrete_variant": variant,
                    "setting_instance_id": instance,
                    "shot_index": shot_index,
                    "original": original,
                    "link": link,
                }
            )
    if len(coordinates) != spec.expected_coordinates:
        raise HelixBenchmarkError(
            f"coordinate count {len(coordinates)} != {spec.expected_coordinates}"
        )
    keys = [row["evaluation_key"] for row in coordinates]
    if len(keys) != len(set(keys)):
        raise HelixBenchmarkError("evaluation keys are not unique")
    return coordinates


def _text_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return ""


class LocalVLLMClient:
    """One HTTP request per generate call; this class has no retry path."""

    def __init__(self, endpoint: str, spec: ModelSpec) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.spec = spec
        self.model_id = spec.model_id

    def authenticate_identity(self) -> None:
        request = urllib.request.Request(f"{self.endpoint}/v1/models", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise HelixBenchmarkError("local vLLM identity endpoint is unavailable") from exc
        ids = {
            str(item.get("id", ""))
            for item in payload.get("data", [])
            if isinstance(item, Mapping)
        }
        if ids != {self.model_id}:
            raise HelixBenchmarkError(
                f"local vLLM served identity {sorted(ids)} != {[self.model_id]}"
            )

    @staticmethod
    def _image_data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        wire_messages: list[dict[str, Any]] = []
        for message in messages:
            images = tuple(message.get("images", ()))
            text = str(message.get("text", ""))
            if images:
                content: object = [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_uri(Path(path))},
                    }
                    for path in images
                ]
                content.append({"type": "text", "text": text})  # type: ignore[union-attr]
            else:
                content = text
            wire_message = {"role": str(message["role"]), "content": content}
            reasoning_content = str(message.get("reasoning_content", "")).strip()
            if reasoning_content:
                wire_message["reasoning_content"] = reasoning_content
            wire_messages.append(wire_message)
        payload = {
            "model": self.model_id,
            "messages": wire_messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "seed": SEED,
            "stream": False,
        }
        for name in (
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        ):
            value = getattr(self.spec, name)
            if value is not None:
                payload[name] = value
        if self.spec.reasoning_effort is not None:
            payload["reasoning_effort"] = self.spec.reasoning_effort
        if self.spec.preserve_thinking:
            payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "preserve_thinking": True,
            }
        raw = _canonical(payload)
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=7_200) as response:
                result = _read_json_http_response(response, "local vLLM")
        except urllib.error.HTTPError as exc:
            raise CapturedRequestError(
                f"local vLLM returned HTTP {exc.code}", _captured_http_error(exc)
            ) from exc
        if not isinstance(result, dict):
            raise CapturedRequestError("local vLLM response is not an object", result)
        return result


def _public_messages(
    messages: Sequence[Mapping[str, Any]], data_root: Path
) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for message in messages:
        public.append(
            {
                "role": str(message["role"]),
                "text": str(message.get("text", "")),
                "reasoning_content": str(message.get("reasoning_content", "")),
                "images": _image_records(
                    tuple(Path(path) for path in message.get("images", ())), data_root
                ),
            }
        )
    return public


def _model_call(
    *,
    client: LocalVLLMClient,
    route_root: Path,
    data_root: Path,
    spec: ModelSpec,
    evaluation_key: str,
    call_name: str,
    messages: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    wal = route_root / "wal" / evaluation_key
    intent_path = wal / f"{call_name}.model.intent.json"
    result_path = wal / f"{call_name}.model.result.json"
    if result_path.is_file():
        if not intent_path.is_file():
            raise HelixBenchmarkError(f"model result lacks its intent: {result_path}")
        _load_signed(intent_path)
        return _load_signed(result_path)
    if intent_path.is_file():
        intent = _load_signed(intent_path)
        return _write_once(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "hsle_helix_model_call_result_v1",
                "created_at": _now(),
                "route": spec.route,
                "evaluation_key": evaluation_key,
                "call_name": call_name,
                "attempt_count": 1,
                "status": "ambiguous_first_attempt_no_replay",
                "content": "",
                "reasoning_content": "",
                "raw_response": None,
                "usage": {},
                "execution_runtime": intent.get("execution_runtime", {}),
                "error": "write-ahead intent exists without a result; replay is forbidden",
            },
        )

    public_messages = _public_messages(messages, data_root)
    generation = {
        "max_tokens": MAX_OUTPUT_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "seed": SEED,
    }
    for name in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        value = getattr(spec, name)
        if value is not None:
            generation[name] = value
    if spec.reasoning_effort is not None:
        generation["reasoning_effort"] = spec.reasoning_effort
    generation["preserve_thinking"] = spec.preserve_thinking
    _write_once(
        intent_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "hsle_helix_model_call_intent_v1",
            "created_at": _now(),
            "route": spec.route,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "evaluation_key": evaluation_key,
            "call_name": call_name,
            "attempt_number": 1,
            "messages": public_messages,
            "prompt_sha256": _sha256_bytes(_canonical(public_messages)),
            "generation": generation,
            "execution_runtime": dict(runtime),
        },
    )
    raw_response: object | None = None
    try:
        raw_response = client.generate(messages)
        if raw_response.get("model") != spec.model_id:
            raise HelixBenchmarkError(
                "local response model identity differs from the exact requested model"
            )
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise HelixBenchmarkError("local response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
            raise HelixBenchmarkError("local response choice lacks a message")
        message = choice["message"]
        content = _text_content(message.get("content"))
        reasoning = _text_content(
            message.get("reasoning_content", message.get("reasoning", ""))
        )
        usage = raw_response.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        status_value = "real_nonblank" if content else "terminal_incorrect_first_response"
        error = "" if content else "first model response content was blank or malformed"
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hsle_helix_model_call_result_v1",
            "created_at": _now(),
            "route": spec.route,
            "evaluation_key": evaluation_key,
            "call_name": call_name,
            "attempt_count": 1,
            "status": status_value,
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": str(choice.get("finish_reason", "")),
            "served_model": str(raw_response.get("model", "")),
            "raw_response": raw_response,
            "usage": dict(usage),
            "execution_runtime": dict(runtime),
            "error": error,
        }
    except Exception as exc:  # the first failed request is terminal by contract
        if raw_response is None and isinstance(exc, CapturedRequestError):
            raw_response = exc.raw_response
        served_model = (
            str(raw_response.get("model", ""))
            if isinstance(raw_response, Mapping)
            else ""
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hsle_helix_model_call_result_v1",
            "created_at": _now(),
            "route": spec.route,
            "evaluation_key": evaluation_key,
            "call_name": call_name,
            "attempt_count": 1,
            "status": "terminal_incorrect_first_response",
            "content": "",
            "reasoning_content": "",
            "served_model": served_model,
            "raw_response": raw_response,
            "usage": {},
            "execution_runtime": dict(runtime),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return _write_once(result_path, result)


def _validate_hle_feedback(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(
        GEMINI_FEEDBACK_RESPONSE_SCHEMA["required"]
    ):
        raise HelixBenchmarkError("Gemini feedback keys differ from the exact HLE schema")
    if not isinstance(value["extracted_final_answer"], str):
        raise HelixBenchmarkError("Gemini extracted_final_answer must be a string")
    if not isinstance(value["reasoning"], str):
        raise HelixBenchmarkError("Gemini reasoning must be a string")
    if not isinstance(value["correct"], str) or value["correct"] not in {"yes", "no"}:
        raise HelixBenchmarkError("Gemini correct must be exactly yes or no")
    confidence = value["confidence"]
    if type(confidence) is not int or not 0 <= confidence <= 100:
        raise HelixBenchmarkError("Gemini confidence must be an integer from 0 through 100")
    if value["strict"] is not True:
        raise HelixBenchmarkError("Gemini strict must be true")
    return dict(value)


def _gemini_feedback_call(
    *,
    route_root: Path,
    spec: ModelSpec,
    evaluation_key: str,
    call_name: str,
    question: str,
    response: str,
    answer: str,
    api_key: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    wal = route_root / "wal" / evaluation_key
    intent_path = wal / f"{call_name}.feedback.intent.json"
    result_path = wal / f"{call_name}.feedback.result.json"
    if result_path.is_file():
        if not intent_path.is_file():
            raise HelixBenchmarkError(f"feedback result lacks its intent: {result_path}")
        _load_signed(intent_path)
        return _load_signed(result_path)
    if intent_path.is_file():
        intent = _load_signed(intent_path)
        return _write_once(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "hsle_helix_feedback_call_result_v1",
                "created_at": _now(),
                "route": spec.route,
                "evaluation_key": evaluation_key,
                "call_name": call_name,
                "attempt_count": 1,
                "status": "ambiguous_first_attempt_no_replay",
                "verdict": "",
                "judgment": {},
                "request_contract": intent.get("request_contract", {}),
                "raw_response": None,
                "execution_runtime": intent.get("execution_runtime", {}),
                "error": "feedback intent exists without a result; replay is forbidden",
            },
        )
    prompt = render_hle_judge_prompt(question, response, answer)
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_FEEDBACK_MODEL}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "seed": 0,
            "candidateCount": 1,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseJsonSchema": GEMINI_FEEDBACK_RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
        "safetySettings": [
            {"category": category, "threshold": "BLOCK_NONE"}
            for category in GEMINI_BLOCK_NONE_CATEGORIES
        ],
    }
    request_contract = {
        "endpoint": endpoint,
        "method": "POST",
        "judge_model": GEMINI_FEEDBACK_MODEL,
        "accepted_model_versions": [GEMINI_FEEDBACK_MODEL],
        "study_feedback_identity_label": "medium",
        "dispatched_thinking_level": "LOW",
        "request_attempts": 1,
        "automatic_retries": 0,
        "response_mime_type": "application/json",
        "response_json_schema": GEMINI_FEEDBACK_RESPONSE_SCHEMA,
        "seed": 0,
        "candidate_count": 1,
        "max_output_tokens": 8192,
        "sampling_knobs": "omitted",
        "safety_threshold": "BLOCK_NONE",
        "safety_categories": list(GEMINI_BLOCK_NONE_CATEGORIES),
    }
    _write_once(
        intent_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "hsle_helix_feedback_call_intent_v1",
            "created_at": _now(),
            "route": spec.route,
            "evaluation_key": evaluation_key,
            "call_name": call_name,
            "judge_model": GEMINI_FEEDBACK_MODEL,
            "attempt_number": 1,
            "prompt": prompt,
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "request_payload": payload,
            "request_payload_sha256": _sha256_bytes(_canonical(payload)),
            "request_contract": request_contract,
            "execution_runtime": dict(runtime),
            "purpose": "inline_binary_learning_from_experience_feedback_only",
        },
    )
    request = urllib.request.Request(
        endpoint,
        data=_canonical(payload),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    raw_response: object | None = None
    try:
        try:
            with urllib.request.urlopen(request, timeout=300) as response_handle:
                raw_response = _read_json_http_response(response_handle, "Gemini feedback")
        except urllib.error.HTTPError as exc:
            raise CapturedRequestError(
                f"Gemini feedback returned HTTP {exc.code}", _captured_http_error(exc)
            ) from exc
        if not isinstance(raw_response, dict):
            raise CapturedRequestError(
                "Gemini feedback response is not an object", raw_response
            )
        if raw_response.get("modelVersion") != GEMINI_FEEDBACK_MODEL:
            raise HelixBenchmarkError(
                "Gemini feedback modelVersion differs from gemini-3.5-flash"
            )
        candidates = raw_response.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise HelixBenchmarkError("Gemini feedback must contain exactly one candidate")
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise HelixBenchmarkError("Gemini feedback candidate is not an object")
        content = candidate.get("content")
        if not isinstance(content, Mapping) or not isinstance(content.get("parts"), list):
            raise HelixBenchmarkError("Gemini feedback candidate lacks content parts")
        text = "".join(
            str(part.get("text", ""))
            for part in content["parts"]
            if isinstance(part, Mapping)
        ).strip()
        try:
            judgment = _validate_hle_feedback(json.loads(text))
        except json.JSONDecodeError as exc:
            raise HelixBenchmarkError("Gemini feedback is not valid JSON") from exc
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hsle_helix_feedback_call_result_v1",
            "created_at": _now(),
            "route": spec.route,
            "evaluation_key": evaluation_key,
            "call_name": call_name,
            "attempt_count": 1,
            "status": "complete",
            "verdict": judgment["correct"],
            "judgment": judgment,
            "model_version": raw_response["modelVersion"],
            "request_contract": request_contract,
            "raw_response": raw_response,
            "execution_runtime": dict(runtime),
            "error": "",
        }
    except Exception as exc:
        if raw_response is None and isinstance(exc, CapturedRequestError):
            raw_response = exc.raw_response
        model_version = (
            str(raw_response.get("modelVersion", ""))
            if isinstance(raw_response, Mapping)
            else ""
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hsle_helix_feedback_call_result_v1",
            "created_at": _now(),
            "route": spec.route,
            "evaluation_key": evaluation_key,
            "call_name": call_name,
            "attempt_count": 1,
            "status": "terminal_feedback_failure_no_replay",
            "verdict": "",
            "judgment": {},
            "model_version": model_version,
            "request_contract": request_contract,
            "raw_response": raw_response,
            "execution_runtime": dict(runtime),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return _write_once(result_path, result)


def _target_images(
    spec: ModelSpec, data_root: Path, original: Mapping[str, str]
) -> tuple[Path, ...]:
    if spec.modality != "multimodal":
        return ()
    return _normalized_image_paths(data_root, original.get("image_paths_or_ids", ""))


def _example_images(
    spec: ModelSpec, data_root: Path, link: Mapping[str, str], index: int
) -> tuple[Path, ...]:
    if spec.modality != "multimodal":
        return ()
    return _normalized_image_paths(
        data_root, link.get(f"example_{index}_image_paths_or_ids", "")
    )


def _record_base(
    *,
    spec: ModelSpec,
    coordinate: Mapping[str, Any],
    data_root: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    original = coordinate["original"]
    link = coordinate["link"]
    target_id = original["original_question_id"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "hsle_helix_coordinate_result_v1",
        "created_at": _now(),
        "run_id": f"helix_{spec.route}_{spec.revision[:12]}",
        "response_id": "response_" + hashlib.sha256(
            coordinate["evaluation_key"].encode("utf-8")
        ).hexdigest()[:24],
        "evaluation_key": coordinate["evaluation_key"],
        "provider": "local_vllm",
        "execution_provider": "local_vllm",
        "model_family": spec.family,
        "model_id": spec.model_id,
        "requested_model_id": spec.model_id,
        "requested_registry_revision": spec.revision,
        "model_version": spec.revision,
        "model_modality": spec.modality,
        "evaluation_setting": coordinate["evaluation_setting"],
        "concrete_variant": coordinate["concrete_variant"],
        "setting_instance_id": coordinate["setting_instance_id"],
        "shot_index": coordinate["shot_index"],
        "question_id": target_id,
        "original_question_id": target_id,
        "example_ids_used": coordinate["setting_instance_id"],
        "question": original["question"],
        "source_question": original["question"],
        "has_image": original["has_image"],
        "image_paths_or_ids": ";".join(
            path.relative_to(data_root).as_posix()
            for path in _target_images(spec, data_root, original)
        ),
        "ground_truth_answer": original["answer"],
        "ground_truth_rationale": original["rationale"],
        "ground_truth_version": original.get("ground_truth_version", ""),
        "linked_example_1_id": link["example_1_question_id"],
        "linked_example_2_id": link["example_2_question_id"],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "seed": SEED,
        "execution_runtime": dict(runtime),
        "final_hle_judging_deferred": True,
        "final_closeness_judging_deferred": True,
    }


def _static_messages(
    spec: ModelSpec,
    coordinate: Mapping[str, Any],
    data_root: Path,
) -> list[dict[str, Any]]:
    original = coordinate["original"]
    link = coordinate["link"]
    question = original["question"]
    target_images = _target_images(spec, data_root, original)
    examples = [
        ContextExample(link[f"example_{index}_question"], link[f"example_{index}_answer"])
        for index in (1, 2)
    ]
    variant = coordinate["concrete_variant"]
    if variant == "zero_shot":
        prompt = zero_shot_prompt(question)
        images = target_images
    elif variant in {"one_shot_a", "one_shot_b"}:
        index = 1 if variant == "one_shot_a" else 2
        prompt = one_shot_prompt(examples[index - 1], question)
        images = _example_images(spec, data_root, link, index) + target_images
    elif variant == "two_shot":
        prompt = two_shot_prompt(examples, question)
        images = (
            _example_images(spec, data_root, link, 1)
            + _example_images(spec, data_root, link, 2)
            + target_images
        )
    else:
        raise HelixBenchmarkError(f"not a static coordinate: {variant}")
    return [{"role": "user", "text": prompt, "images": images}]


def _usage(result: Mapping[str, Any]) -> tuple[object, object, object]:
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        return "", "", ""
    return (
        usage.get("prompt_tokens", ""),
        usage.get("completion_tokens", ""),
        usage.get("total_tokens", ""),
    )


def _coordinate_from_model_result(
    *,
    base: dict[str, Any],
    messages: Sequence[Mapping[str, Any]],
    data_root: Path,
    result: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    public_messages = _public_messages(messages, data_root)
    content = str(result.get("content", ""))
    reasoning = str(result.get("reasoning_content", ""))
    combined = "\n\n".join(value for value in (reasoning, content) if value)
    result_status = str(result.get("status", ""))
    valid = result_status == "real_nonblank" and bool(content.strip())
    terminal = result_status == "terminal_incorrect_first_response"
    coordinate_status = (
        "complete" if valid else "terminal_incorrect" if terminal else "operationally_incomplete"
    )
    completion_type = (
        "real_nonblank"
        if valid
        else "terminal_incorrect_first_response"
        if terminal
        else result_status or "operationally_incomplete_unknown_model_result"
    )
    input_tokens, output_tokens, total_tokens = _usage(result)
    return {
        **base,
        "prompt": "\n\n".join(
            f"{item['role'].upper()}: {item['text']}" for item in public_messages
        ),
        "messages": public_messages,
        "source_prompt_hash": _sha256_bytes(_canonical(public_messages)),
        "model_parsed_answer": content,
        "model_content": content,
        "model_reasoning_content": reasoning,
        "model_raw_output": combined,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "status": coordinate_status,
        "generation_completion_type": completion_type,
        "terminal_status": result_status,
        "error_message": str(result.get("error", "")),
        "model_calls": list(calls),
        "feedback_calls": list(feedback),
    }


def _run_static_coordinate(
    *,
    spec: ModelSpec,
    coordinate: Mapping[str, Any],
    data_root: Path,
    route_root: Path,
    client: LocalVLLMClient,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    messages = _static_messages(spec, coordinate, data_root)
    result = _model_call(
        client=client,
        route_root=route_root,
        data_root=data_root,
        spec=spec,
        evaluation_key=coordinate["evaluation_key"],
        call_name="static_target",
        messages=messages,
        runtime=runtime,
    )
    return _coordinate_from_model_result(
        base=_record_base(
            spec=spec, coordinate=coordinate, data_root=data_root, runtime=runtime
        ),
        messages=messages,
        data_root=data_root,
        result=result,
        calls=[result],
    )


def _failed_lfe_record(
    *,
    base: dict[str, Any],
    messages: Sequence[Mapping[str, Any]],
    data_root: Path,
    calls: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    status_value: str,
    error: str,
) -> dict[str, Any]:
    public_messages = _public_messages(messages, data_root)
    return {
        **base,
        "prompt": "\n\n".join(
            f"{item['role'].upper()}: {item['text']}" for item in public_messages
        ),
        "messages": public_messages,
        "source_prompt_hash": _sha256_bytes(_canonical(public_messages)),
        "model_parsed_answer": "",
        "model_content": "",
        "model_reasoning_content": "",
        "model_raw_output": "",
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
        "status": (
            "terminal_incorrect"
            if status_value.startswith("terminal_incorrect")
            else "operationally_incomplete"
        ),
        "generation_completion_type": status_value,
        "terminal_status": status_value,
        "error_message": error,
        "model_calls": list(calls),
        "feedback_calls": list(feedback),
    }


def _run_lfe_coordinate(
    *,
    spec: ModelSpec,
    coordinate: Mapping[str, Any],
    data_root: Path,
    route_root: Path,
    client: LocalVLLMClient,
    gemini_key: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    original = coordinate["original"]
    link = coordinate["link"]
    messages: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    feedback_calls: list[dict[str, Any]] = []
    base = _record_base(
        spec=spec, coordinate=coordinate, data_root=data_root, runtime=runtime
    )
    for index in (1, 2):
        messages.append(
            {
                "role": "user",
                "text": zero_shot_prompt(link[f"example_{index}_question"]),
                "images": _example_images(spec, data_root, link, index),
            }
        )
        result = _model_call(
            client=client,
            route_root=route_root,
            data_root=data_root,
            spec=spec,
            evaluation_key=coordinate["evaluation_key"],
            call_name=f"lfe_example_{index}",
            messages=messages,
            runtime=runtime,
        )
        calls.append(result)
        if result.get("status") != "real_nonblank" or not str(
            result.get("content", "")
        ).strip():
            result_status = str(result.get("status", ""))
            status_value = (
                f"terminal_incorrect_lfe_example_{index}_first_response"
                if result_status == "terminal_incorrect_first_response"
                else f"operationally_incomplete_lfe_example_{index}_no_replay"
            )
            return _failed_lfe_record(
                base=base,
                messages=messages,
                data_root=data_root,
                calls=calls,
                feedback=feedback_calls,
                status_value=status_value,
                error=str(result.get("error", "first example response was unusable")),
            )
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "text": str(result["content"]),
            "images": (),
        }
        if spec.preserve_thinking:
            assistant_message["reasoning_content"] = str(
                result.get("reasoning_content", "")
            )
        messages.append(assistant_message)
        feedback = _gemini_feedback_call(
            route_root=route_root,
            spec=spec,
            evaluation_key=coordinate["evaluation_key"],
            call_name=f"lfe_example_{index}",
            question=link[f"example_{index}_question"],
            response=str(result["content"]),
            answer=link[f"example_{index}_answer"],
            api_key=gemini_key,
            runtime=runtime,
        )
        feedback_calls.append(feedback)
        if feedback.get("status") != "complete":
            return _failed_lfe_record(
                base=base,
                messages=messages,
                data_root=data_root,
                calls=calls,
                feedback=feedback_calls,
                status_value=f"operationally_incomplete_lfe_feedback_{index}",
                error=str(feedback.get("error", "binary feedback was unavailable")),
            )
        feedback_text = (
            "Your previous answer was correct."
            if feedback["verdict"] == "yes"
            else "Your previous answer was incorrect."
        )
        messages.append({"role": "user", "text": feedback_text, "images": ()})

    messages.append(
        {
            "role": "user",
            "text": zero_shot_prompt(original["question"]),
            "images": _target_images(spec, data_root, original),
        }
    )
    target = _model_call(
        client=client,
        route_root=route_root,
        data_root=data_root,
        spec=spec,
        evaluation_key=coordinate["evaluation_key"],
        call_name="lfe_target",
        messages=messages,
        runtime=runtime,
    )
    calls.append(target)
    return _coordinate_from_model_result(
        base=base,
        messages=messages,
        data_root=data_root,
        result=target,
        calls=calls,
        feedback=feedback_calls,
    )


CSV_FIELDS = (
    "response_id",
    "evaluation_key",
    "run_id",
    "provider",
    "execution_provider",
    "model_family",
    "model_id",
    "requested_model_id",
    "requested_registry_revision",
    "model_version",
    "model_modality",
    "max_output_tokens",
    "max_model_len",
    "temperature",
    "top_p",
    "seed",
    "evaluation_setting",
    "concrete_variant",
    "setting_instance_id",
    "shot_index",
    "question_id",
    "original_question_id",
    "example_ids_used",
    "question",
    "source_question",
    "has_image",
    "image_paths_or_ids",
    "prompt",
    "messages_json",
    "source_prompt_hash",
    "model_parsed_answer",
    "model_content",
    "model_reasoning_content",
    "model_raw_output",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "status",
    "generation_completion_type",
    "terminal_status",
    "error_message",
    "created_at",
    "ground_truth_answer",
    "ground_truth_rationale",
    "ground_truth_version",
    "runtime_json",
    "call_ledger_json",
    "final_hle_judging_deferred",
    "final_closeness_judging_deferred",
)


def _runtime_inventory(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for record in records:
        candidates: list[object] = [record.get("execution_runtime")]
        for ledger_name in ("model_calls", "feedback_calls"):
            ledger = record.get(ledger_name, [])
            if isinstance(ledger, list):
                candidates.extend(
                    call.get("execution_runtime")
                    for call in ledger
                    if isinstance(call, Mapping)
                )
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate:
                continue
            runtime = dict(candidate)
            digest = _sha256_bytes(_canonical(runtime))
            inventory[digest] = {
                "runtime_sha256": digest,
                "runtime": runtime,
            }
    return [inventory[digest] for digest in sorted(inventory)]


def _export_csv(route_root: Path, runtime: Mapping[str, Any]) -> None:
    records = [
        _load_signed(path)
        for path in sorted((route_root / "generation_results").glob("eval_*.json"))
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {field: record.get(field, "") for field in CSV_FIELDS}
        row["messages_json"] = json.dumps(
            record.get("messages", []), sort_keys=True, ensure_ascii=False
        )
        row["runtime_json"] = json.dumps(
            {
                "coordinate_recording_runtime": record.get(
                    "execution_runtime", dict(runtime)
                ),
                "call_runtime_inventory": _runtime_inventory([record]),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        row["call_ledger_json"] = json.dumps(
            {
                "model_calls": record.get("model_calls", []),
                "feedback_calls": record.get("feedback_calls", []),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        rows.append(row)
    destination = route_root / "responses.csv"
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".responses.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_gemini_key(path_value: str) -> str:
    if not path_value:
        raise HelixBenchmarkError(
            "HSLE_GEMINI_KEY_FILE is required for inline Gemini 3.5 Flash LFE feedback"
        )
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise HelixBenchmarkError("HSLE_GEMINI_KEY_FILE must name a regular non-symlink file")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise HelixBenchmarkError("HSLE_GEMINI_KEY_FILE must be user-owned with exact mode 0600")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value:
        raise HelixBenchmarkError("HSLE_GEMINI_KEY_FILE must contain exactly one nonblank key")
    return value


def _unexpected_coordinate_failure(
    *,
    spec: ModelSpec,
    coordinate: Mapping[str, Any],
    route_root: Path,
    error: Exception,
) -> dict[str, Any]:
    """Record a run-level exception while leaving uncalled work resumable."""

    wal_root = route_root / "wal" / coordinate["evaluation_key"]
    wal_inventory = [
        {
            "path": path.relative_to(route_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(wal_root.glob("*.json"))
        if path.is_file() and not path.is_symlink()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "hsle_helix_coordinate_execution_error_v1",
        "created_at": _now(),
        "route": spec.route,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "evaluation_key": coordinate["evaluation_key"],
        "concrete_variant": coordinate["concrete_variant"],
        "status": "run_failed_coordinate_left_resumable",
        "error": f"{type(error).__name__}: {error}",
        "wal_inventory": wal_inventory,
        "request_replay_policy": "existing intents are never replayed; uncalled steps may resume",
    }


def run_generation(
    spec: ModelSpec, workspace: Path, endpoint: str, concurrency: int
) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    preparation = _load_signed(workspace / "control" / spec.route / "preparation.json")
    if (
        preparation.get("route") != spec.route
        or preparation.get("model_id") != spec.model_id
        or preparation.get("model_revision") != spec.revision
        or preparation.get("vllm_version") != spec.vllm_version
    ):
        raise HelixBenchmarkError("preparation authority differs from requested route")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id.isdigit():
        raise HelixBenchmarkError("local generation must run inside Slurm")
    preparation_model = preparation.get("model")
    if not isinstance(preparation_model, Mapping):
        raise HelixBenchmarkError("preparation authority lacks model metadata")
    verification = _load_signed(
        workspace
        / "control"
        / spec.route
        / "model_verifications"
        / f"slurm_{slurm_job_id}.json"
    )
    if (
        verification.get("route") != spec.route
        or verification.get("model_id") != spec.model_id
        or verification.get("model_revision") != spec.revision
        or verification.get("preparation_record_sha256")
        != preparation.get("record_sha256")
        or verification.get("weight_inventory_sha256")
        != preparation_model.get("weight_inventory_sha256")
        or verification.get("slurm_job_id") != slurm_job_id
    ):
        raise HelixBenchmarkError("current-job model-snapshot verification differs")
    data_root = Path(preparation["dataset"]["root"]).resolve(strict=True)
    inputs = _validate_inputs(data_root)
    if inputs["authority"] != preparation["dataset"]["authority"]:
        raise HelixBenchmarkError("dataset authority changed after preparation")
    installed_vllm = importlib.metadata.version("vllm")
    if installed_vllm != spec.vllm_version:
        raise HelixBenchmarkError(
            f"vLLM {installed_vllm} != pinned {spec.vllm_version}"
        )
    installed_transformers = importlib.metadata.version("transformers")
    if installed_transformers != preparation.get("transformers_version"):
        raise HelixBenchmarkError(
            "installed Transformers version differs from preparation authority"
        )
    client = LocalVLLMClient(endpoint, spec)
    client.authenticate_identity()
    if concurrency < 1 or concurrency > 64:
        raise HelixBenchmarkError("concurrency must be between 1 and 64")
    gemini_key = _load_gemini_key(os.environ.get("HSLE_GEMINI_KEY_FILE", ""))
    route_root = workspace / "need_to_be_judged" / spec.route
    route_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime = {
        "execution_provider": "local_vllm",
        "vllm_version": installed_vllm,
        "transformers_version": installed_transformers,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "tensor_parallel_size": 8,
        "gpu_type": "H200",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "seed": SEED,
        "coordinate_concurrency": concurrency,
        "reasoning_effort": spec.reasoning_effort or "provider_default",
        "preserve_thinking_history": spec.preserve_thinking,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_node": os.environ.get("SLURMD_NODENAME", ""),
        "final_judging": "deferred",
    }
    coordinates = _coordinates(spec, inputs)
    completed = 0
    pending: list[dict[str, Any]] = []
    run_attempt_id = (
        f"slurm_{slurm_job_id}" if slurm_job_id.isdigit() else f"process_{os.getpid()}"
    )
    for coordinate in coordinates:
        destination = route_root / "generation_results" / (
            coordinate["evaluation_key"] + ".json"
        )
        if destination.is_file():
            existing = _load_signed(destination)
            if (
                existing.get("model_id") != spec.model_id
                or existing.get("requested_registry_revision") != spec.revision
                or existing.get("evaluation_key") != coordinate["evaluation_key"]
            ):
                raise HelixBenchmarkError(f"existing coordinate differs: {destination}")
            completed += 1
            continue
        pending.append(coordinate)

    def execute(coordinate: Mapping[str, Any]) -> dict[str, Any]:
        if coordinate["concrete_variant"] == "learning_from_experience":
            return _run_lfe_coordinate(
                spec=spec,
                coordinate=coordinate,
                data_root=data_root,
                route_root=route_root,
                client=client,
                gemini_key=gemini_key,
                runtime=runtime,
            )
        return _run_static_coordinate(
            spec=spec,
            coordinate=coordinate,
            data_root=data_root,
            route_root=route_root,
            client=client,
            runtime=runtime,
        )

    # Concurrency is between independent coordinates only. Each LFE future
    # retains strict example-1 -> feedback-1 -> example-2 -> feedback-2 ->
    # target order, and every call has its own immutable write-ahead path.
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="hsle-coordinate") as pool:
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
            pool.submit(execute, coordinate): coordinate for coordinate in pending
        }
        for future in as_completed(futures):
            coordinate = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # One coordinate cannot cancel the remaining futures. Any call
                # already attempted remains protected by WAL, while a step with
                # no intent remains callable on a later job rather than being
                # mislabeled as a generation result.
                error_record = _unexpected_coordinate_failure(
                    spec=spec,
                    coordinate=coordinate,
                    route_root=route_root,
                    error=exc,
                )
                error_path = route_root / "run_errors" / run_attempt_id / (
                    coordinate["evaluation_key"] + ".json"
                )
                _write_once(error_path, error_record)
                continue
            destination = route_root / "generation_results" / (
                coordinate["evaluation_key"] + ".json"
            )
            _write_once(destination, result)
            completed += 1
            if completed % 50 == 0:
                _export_csv(route_root, runtime)
    _export_csv(route_root, runtime)
    records = [
        _load_signed(path)
        for path in sorted((route_root / "generation_results").glob("eval_*.json"))
    ]
    if len(records) != spec.expected_coordinates:
        raise HelixBenchmarkError("final coordinate-record count differs")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hsle_helix_generation_summary_v1",
        "created_at": _now(),
        "route": spec.route,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "expected_coordinates": spec.expected_coordinates,
        "record_count": len(records),
        "status_counts": {
            status_value: sum(record.get("status") == status_value for record in records)
            for status_value in ("complete", "terminal_incorrect", "operationally_incomplete")
        },
        "response_csv": str(route_root / "responses.csv"),
        "response_csv_sha256": _sha256(route_root / "responses.csv"),
        "final_hle_judging_deferred": True,
        "final_closeness_judging_deferred": True,
        "finalizer_runtime": runtime,
        "runtime_inventory": _runtime_inventory(records),
    }
    summary_path = route_root / "generation_summary.json"
    if summary_path.exists():
        final_summary = _load_signed(summary_path)
        if (
            final_summary.get("response_csv_sha256") != summary["response_csv_sha256"]
            or final_summary.get("status_counts") != summary["status_counts"]
        ):
            raise HelixBenchmarkError("existing final summary differs")
    else:
        final_summary = _write_once(summary_path, summary)
    operational_count = int(
        final_summary.get("status_counts", {}).get("operationally_incomplete", 0)
    )
    if operational_count:
        raise HelixBenchmarkError(
            f"{operational_count} coordinates are operationally incomplete; "
            "terminal first-response failures remain valid incorrect settlements"
        )
    return final_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download and validate pinned inputs")
    prepare.add_argument("--route", choices=sorted(MODEL_SPECS), required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, required=True)
    verify = subparsers.add_parser(
        "verify-prepared", help="cryptographically verify a prepared model snapshot"
    )
    verify.add_argument("--route", choices=sorted(MODEL_SPECS), required=True)
    verify.add_argument("--workspace", type=Path, required=True)
    run = subparsers.add_parser("run", help="execute the exact no-replay coordinate vector")
    run.add_argument("--route", choices=sorted(MODEL_SPECS), required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:8000")
    run.add_argument("--concurrency", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = MODEL_SPECS[args.route]
    if args.command == "prepare":
        result = prepare_workspace(spec, args.workspace, args.repo_root)
        print(
            f"Prepared {result['model_id']}@{result['model_revision']} in "
            f"{result['workspace']}"
        )
        return
    if args.command == "verify-prepared":
        result = verify_prepared_workspace(spec, args.workspace)
        print(
            f"Verified {result['model_id']}@{result['model_revision']} for "
            f"Slurm job {result['slurm_job_id']}"
        )
        return
    result = run_generation(spec, args.workspace, args.endpoint, args.concurrency)
    print(
        f"Recorded {result['record_count']}/{result['expected_coordinates']} coordinates; "
        "final HLE and closeness judging remain deferred"
    )


if __name__ == "__main__":
    main()
