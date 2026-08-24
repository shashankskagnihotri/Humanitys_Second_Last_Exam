"""Private exact protocol primitives for the public OpenRouter resume runner.

This flat helper keeps the public release's existing hsle.benchmark and
hsle.prompts modules unshadowed. It contains only the prompt construction,
single-dispatch request, source-loading, and validation primitives used by
hsle.public_openrouter_resume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

import pandas as pd


ADAPTIVE_RECOVERY_PROTOCOL_VERSION = "hsle-openrouter-single-dispatch-v1"
MAX_RETRIES = 0
MAX_TOTAL_ATTEMPTS = 1
FULL_INPUT_PROFILE = "full"
TERMINAL_FAILURE_STATUS = "nonresponsive"
TERMINAL_HLE_CORRECTNESS = "incorrect"
TERMINAL_CLOSENESS_SCORE = 0

VALID_SETTINGS = frozenset(
    {
        "zero_shot",
        "one_shot",
        "two_shot",
        "learning_from_experience",
    }
)
VALID_CONCRETE_VARIANTS = frozenset(
    {
        "zero_shot",
        "one_shot_a",
        "one_shot_b",
        "two_shot",
        "learning_from_experience",
    }
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_NAME = re.compile(r"record-([0-9]{12})\.json\Z")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value in the protocol's unique byte representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text_sha256 requires a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string")


def _require_nonblank_source_text(name: str, value: str) -> None:
    """Validate verbatim source text without silently normalizing its bytes.

    Canonical HLE question text can legitimately retain boundary CR/LF or
    spaces from the frozen source.  Prompt provenance, request hashes, and
    feedback-judge evidence all depend on preserving those exact bytes.  A
    source question therefore has a different invariant from identifiers:
    it must contain non-whitespace text, but it must *not* be stripped merely
    to satisfy an identifier-oriented validator.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-whitespace source text")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        data = canonical_json_bytes(dict(payload)) + b"\n"
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@dataclass(frozen=True, slots=True)
class AdaptiveRecoveryPolicy:
    """Immutable one-dispatch policy for every benchmark turn."""

    max_retries: int = MAX_RETRIES
    max_total_attempts: int = MAX_TOTAL_ATTEMPTS
    input_profile_name: str = FULL_INPUT_PROFILE
    terminal_failure_status: str = TERMINAL_FAILURE_STATUS
    terminal_hle_correctness: str = TERMINAL_HLE_CORRECTNESS
    terminal_closeness_score: int = TERMINAL_CLOSENESS_SCORE

    def __post_init__(self) -> None:
        expected = {
            "max_retries": MAX_RETRIES,
            "max_total_attempts": MAX_TOTAL_ATTEMPTS,
            "input_profile_name": FULL_INPUT_PROFILE,
            "terminal_failure_status": TERMINAL_FAILURE_STATUS,
            "terminal_hle_correctness": TERMINAL_HLE_CORRECTNESS,
            "terminal_closeness_score": TERMINAL_CLOSENESS_SCORE,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"The versioned recovery protocol requires {name}={value!r}")
        if self.max_total_attempts != 1 + self.max_retries:
            raise ValueError("max_total_attempts must equal one initial attempt plus max_retries")

    def input_profile(self, attempt_number: int) -> str:
        if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
            raise TypeError("attempt_number must be an integer")
        if not 1 <= attempt_number <= self.max_total_attempts:
            raise ValueError(f"attempt_number must be in [1, {self.max_total_attempts}]")
        return self.input_profile_name

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(
            {
                "protocol_version": ADAPTIVE_RECOVERY_PROTOCOL_VERSION,
                **asdict(self),
            }
        )


@dataclass(frozen=True, slots=True)
class RecoveryCoordinate:
    model_id: str
    requested_model_id: str
    model_modality: Literal["multimodal", "text_only"]
    evaluation_setting: str
    concrete_variant: str
    original_question_id: str
    setting_instance_id: str
    evaluation_key: str

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "requested_model_id",
            "evaluation_setting",
            "concrete_variant",
            "original_question_id",
            "evaluation_key",
        ):
            _require_nonempty(name, getattr(self, name))
        if self.model_modality not in {"multimodal", "text_only"}:
            raise ValueError(f"Unsupported modality: {self.model_modality!r}")
        if self.evaluation_setting not in VALID_SETTINGS:
            raise ValueError(f"Unsupported evaluation_setting: {self.evaluation_setting!r}")
        if self.concrete_variant not in VALID_CONCRETE_VARIANTS:
            raise ValueError(f"Unsupported concrete_variant: {self.concrete_variant!r}")
        expected_setting = (
            "one_shot"
            if self.concrete_variant in {"one_shot_a", "one_shot_b"}
            else self.concrete_variant
        )
        if self.evaluation_setting != expected_setting:
            raise ValueError(
                "evaluation_setting and concrete_variant are inconsistent: "
                f"{self.evaluation_setting!r}, {self.concrete_variant!r}"
            )
        if self.evaluation_setting == "zero_shot" and self.setting_instance_id:
            raise ValueError("zero_shot setting_instance_id must be blank")
        if self.evaluation_setting != "zero_shot" and not self.setting_instance_id:
            raise ValueError(f"{self.evaluation_setting} requires a setting_instance_id")

    @property
    def key(self) -> str:
        return canonical_json_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class AdaptiveMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    image_paths: tuple[Path, ...] = ()
    reasoning_content: str = ""
    semantic_kind: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if not isinstance(self.reasoning_content, str):
            raise TypeError("reasoning_content must be a string")
        if self.reasoning_content and self.role != "assistant":
            raise ValueError("reasoning_content is valid only for assistant messages")
        if any(not isinstance(path, Path) for path in self.image_paths):
            raise TypeError("image_paths must contain pathlib.Path values")


@dataclass(frozen=True, slots=True)
class PromptEnvelope:
    coordinate: RecoveryCoordinate
    turn: str
    messages: tuple[AdaptiveMessage, ...]
    target_question: str
    target_image_paths: tuple[Path, ...] = ()
    static_examples: tuple[ContextExample, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("turn", self.turn)
        _require_nonblank_source_text("target_question", self.target_question)
        if not self.messages:
            raise ValueError("PromptEnvelope requires at least one message")
        if self.messages[-1].role != "user":
            raise ValueError("The current provider-visible message must be a user turn")
        if self.target_question not in self.messages[-1].content:
            raise ValueError("The final user message does not contain the exact target question")
        final_images = tuple(self.messages[-1].image_paths)
        target_images = tuple(self.target_image_paths)
        if target_images and final_images[-len(target_images) :] != target_images:
            raise ValueError("The final user message must end with the exact ordered target images")


@dataclass(frozen=True, slots=True)
class ImageEvidence:
    ordinal: int
    message_index: int
    image_index_within_message: int
    byte_count: int
    sha256: str
    path: str

    def __post_init__(self) -> None:
        if not _HEX_SHA256.fullmatch(self.sha256):
            raise ValueError("Image evidence requires a SHA-256 digest")

    def hash_payload(self) -> dict[str, Any]:
        """Return path-independent image identity used in request hashes."""

        return {
            "ordinal": self.ordinal,
            "message_index": self.message_index,
            "image_index_within_message": self.image_index_within_message,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AttemptRequest:
    protocol_version: str
    policy_sha256: str
    coordinate: RecoveryCoordinate
    turn: str
    attempt_number: int
    input_profile: str
    messages: tuple[AdaptiveMessage, ...]
    image_evidence: tuple[ImageEvidence, ...]
    target_question_sha256: str
    target_image_sha256s: tuple[str, ...]
    canonical_request_sha256: str
    request_sha256: str
    prompt_sha256: str

    def manifest(self, *, include_paths: bool = True) -> dict[str, Any]:
        messages = [
            {
                "role": message.role,
                "content": message.content,
                "reasoning_content": message.reasoning_content,
                "semantic_kind": message.semantic_kind,
                "image_ordinals": [
                    evidence.ordinal
                    for evidence in self.image_evidence
                    if evidence.message_index == index
                ],
            }
            for index, message in enumerate(self.messages)
        ]
        images = [
            asdict(evidence) if include_paths else evidence.hash_payload()
            for evidence in self.image_evidence
        ]
        return {
            "protocol_version": self.protocol_version,
            "policy_sha256": self.policy_sha256,
            "coordinate": asdict(self.coordinate),
            "coordinate_key": self.coordinate.key,
            "turn": self.turn,
            "attempt_number": self.attempt_number,
            "input_profile": self.input_profile,
            "messages": messages,
            "images": images,
            "target_question_sha256": self.target_question_sha256,
            "target_image_sha256s": list(self.target_image_sha256s),
            "canonical_request_sha256": self.canonical_request_sha256,
            "request_sha256": self.request_sha256,
            "prompt_sha256": self.prompt_sha256,
            "truncation_applied": False,
            "truncation_events": [],
        }


def attachment_legend(examples: Sequence[ContextExample], target_paths: Sequence[Path]) -> str:
    """Render the exact image-order legend used by the canonical benchmark."""

    labels: list[str] = []
    image_number = 1
    for index, example in enumerate(examples, start=1):
        if example.image_paths:
            stop = image_number + len(example.image_paths) - 1
            span = str(image_number) if stop == image_number else f"{image_number}-{stop}"
            labels.append(f"image(s) {span}: solved example {index}")
            image_number = stop + 1
    if target_paths:
        stop = image_number + len(target_paths) - 1
        span = str(image_number) if stop == image_number else f"{image_number}-{stop}"
        labels.append(f"image(s) {span}: target question")
    if not labels:
        return ""
    return "\n\nATTACHED IMAGE ORDER:\n" + "\n".join(labels)


def _static_prompt(
    coordinate: RecoveryCoordinate,
    examples: Sequence[ContextExample],
    target_question: str,
    target_image_paths: Sequence[Path],
) -> tuple[str, tuple[Path, ...]]:
    use_images = coordinate.model_modality == "multimodal"
    if coordinate.evaluation_setting == "zero_shot":
        if examples:
            raise ValueError("zero_shot prompt may not contain examples")
        prompt = zero_shot_prompt(target_question)
    elif coordinate.evaluation_setting == "one_shot":
        if len(examples) != 1:
            raise ValueError("one_shot requires exactly one selected example")
        prompt = one_shot_prompt(examples[0], target_question)
    elif coordinate.evaluation_setting == "two_shot":
        if len(examples) != 2:
            raise ValueError("two_shot requires exactly two selected examples")
        prompt = two_shot_prompt(list(examples), target_question)
    else:
        raise ValueError("_static_prompt does not render LFE chat histories")
    image_paths: tuple[Path, ...] = ()
    if use_images:
        image_paths = (
            *(Path(path) for example in examples for path in example.image_paths),
            *tuple(target_image_paths),
        )
        prompt += attachment_legend(examples, target_image_paths)
    return prompt, image_paths


def build_static_prompt_envelope(
    coordinate: RecoveryCoordinate,
    *,
    target_question: str,
    target_image_paths: Sequence[Path],
    examples: Sequence[ContextExample],
) -> PromptEnvelope:
    if coordinate.evaluation_setting == "learning_from_experience":
        raise ValueError("Use a chat PromptEnvelope for learning_from_experience")
    selected = tuple(examples)
    prompt, all_images = _static_prompt(coordinate, selected, target_question, target_image_paths)
    return PromptEnvelope(
        coordinate=coordinate,
        turn="target",
        messages=(
            AdaptiveMessage(
                role="user",
                content=prompt,
                image_paths=all_images,
                semantic_kind="target_with_static_examples",
            ),
        ),
        target_question=target_question,
        target_image_paths=tuple(target_image_paths),
        static_examples=selected,
    )


def _image_evidence(messages: Sequence[AdaptiveMessage]) -> tuple[ImageEvidence, ...]:
    evidence: list[ImageEvidence] = []
    ordinal = 0
    for message_index, message in enumerate(messages):
        for image_index, path in enumerate(message.image_paths):
            candidate = Path(path)
            if not candidate.is_file():
                raise FileNotFoundError(f"Prompt image does not exist: {candidate}")
            evidence.append(
                ImageEvidence(
                    ordinal=ordinal,
                    message_index=message_index,
                    image_index_within_message=image_index,
                    byte_count=candidate.stat().st_size,
                    sha256=file_sha256(candidate),
                    path=str(candidate),
                )
            )
            ordinal += 1
    return tuple(evidence)


def _request_hash_payload(
    messages: Sequence[AdaptiveMessage], images: Sequence[ImageEvidence]
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "reasoning_content": message.reasoning_content,
                "semantic_kind": message.semantic_kind,
                "image_ordinals": [
                    evidence.ordinal for evidence in images if evidence.message_index == index
                ],
            }
            for index, message in enumerate(messages)
        ],
        "images": [evidence.hash_payload() for evidence in images],
    }


def _prompt_transcript(messages: Sequence[AdaptiveMessage], images: Sequence[ImageEvidence]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        parts.append(f"{message.role.upper()}:\n{message.content}")
        if message.reasoning_content:
            parts.append("ASSISTANT_REASONING_CONTENT:\n" + message.reasoning_content)
        message_images = [evidence.sha256 for evidence in images if evidence.message_index == index]
        if message_images:
            parts.append("ORDERED_IMAGE_SHA256S:\n" + "\n".join(message_images))
    return "\n\n".join(parts)


def build_attempt_request(
    envelope: PromptEnvelope,
    attempt_number: int,
    *,
    policy: AdaptiveRecoveryPolicy | None = None,
) -> AttemptRequest:
    active_policy = policy or AdaptiveRecoveryPolicy()
    profile = active_policy.input_profile(attempt_number)
    messages = envelope.messages
    if envelope.target_question not in messages[-1].content:
        raise ValueError("Single-dispatch request changed or removed the target question")
    if messages[-1].image_paths != envelope.messages[-1].image_paths:
        raise ValueError("Single-dispatch request changed target image bindings")

    canonical_images = _image_evidence(envelope.messages)
    attempt_images = _image_evidence(messages)
    if [item.hash_payload() for item in canonical_images] != [
        item.hash_payload() for item in attempt_images
    ]:
        raise ValueError("Single-dispatch request changed ordered image identities")
    canonical_payload = _request_hash_payload(envelope.messages, canonical_images)
    request_payload = _request_hash_payload(messages, attempt_images)
    target_image_count = len(envelope.target_image_paths)
    target_image_hashes = (
        tuple(evidence.sha256 for evidence in attempt_images[-target_image_count:])
        if target_image_count
        else ()
    )
    return AttemptRequest(
        protocol_version=ADAPTIVE_RECOVERY_PROTOCOL_VERSION,
        policy_sha256=active_policy.sha256,
        coordinate=envelope.coordinate,
        turn=envelope.turn,
        attempt_number=attempt_number,
        input_profile=profile,
        messages=messages,
        image_evidence=attempt_images,
        target_question_sha256=text_sha256(envelope.target_question),
        target_image_sha256s=target_image_hashes,
        canonical_request_sha256=canonical_json_sha256(canonical_payload),
        request_sha256=canonical_json_sha256(request_payload),
        prompt_sha256=text_sha256(_prompt_transcript(messages, attempt_images)),
    )


MISSING_STRINGS = {"", "nan", "none", "null", "<na>"}


def clean_optional_str(value: Any) -> str:
    """Convert pandas/JSON missing values to a stable empty string."""

    if value is None:
        return ""
    text = str(value)
    if text.lower() in MISSING_STRINGS:
        return ""
    return text


def generation_response_is_usable(
    response_text: Any,
    *,
    error_message: Any = "",
) -> bool:
    """Apply the benchmark's canonical generated-response validity rule."""

    if clean_optional_str(error_message):
        return False
    answer = clean_optional_str(response_text).strip()
    return bool(answer and not answer.startswith("[ERROR]") and answer != "[No answer]")


def bool_or_none(value: Any) -> bool | None:
    """Parse common boolean encodings without guessing arbitrary text."""

    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in MISSING_STRINGS:
        return None
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


class FeedbackMode(StrEnum):
    BINARY_ONLY = "binary_only"
    BINARY_PLUS_ANSWER = "binary_plus_answer"
    BINARY_PLUS_RATIONALE = "binary_plus_rationale"


ANSWER_INSTRUCTION = (
    "Give the final answer clearly. If the question has answer choices, include the option letter."
)


@dataclass(frozen=True)
class ContextExample:
    question_id: str
    question: str
    answer: str
    rationale: str = ""
    has_image: bool = False
    image_paths: tuple[str, ...] = ()


def zero_shot_prompt(question: str) -> str:
    return f"{ANSWER_INSTRUCTION}\n\nQUESTION:\n{clean_optional_str(question)}"


def one_shot_prompt(example: ContextExample, question: str) -> str:
    return (
        f"{ANSWER_INSTRUCTION}\n\n"
        "Here is one solved example.\n\n"
        f"EXAMPLE QUESTION:\n{clean_optional_str(example.question)}\n\n"
        f"EXAMPLE ANSWER:\n{clean_optional_str(example.answer)}\n\n"
        f"QUESTION:\n{clean_optional_str(question)}"
    )


def two_shot_prompt(examples: list[ContextExample], question: str) -> str:
    if len(examples) != 2:
        raise ValueError(f"two_shot_prompt requires exactly two examples; got {len(examples)}")
    parts = [ANSWER_INSTRUCTION, "", "Here are two solved examples."]
    for idx, example in enumerate(examples, start=1):
        parts.extend(
            [
                "",
                f"EXAMPLE {idx} QUESTION:",
                clean_optional_str(example.question),
                "",
                f"EXAMPLE {idx} ANSWER:",
                clean_optional_str(example.answer),
            ]
        )
    parts.extend(["", "QUESTION:", clean_optional_str(question)])
    return "\n".join(parts)


def lfe_question_prompt(question: str) -> str:
    return f"{ANSWER_INSTRUCTION}\n\nQUESTION:\n{clean_optional_str(question)}"


def lfe_feedback(is_correct: bool | None, answer: str, rationale: str, mode: FeedbackMode) -> str:
    if is_correct is True:
        base = "Your previous answer was correct."
    elif is_correct is False:
        base = "Your previous answer was incorrect."
    else:
        base = "Your previous answer could not be graded deterministically."
    if mode == FeedbackMode.BINARY_ONLY:
        return base
    if mode == FeedbackMode.BINARY_PLUS_ANSWER:
        return f"{base}\nCorrect answer: {clean_optional_str(answer)}"
    if mode == FeedbackMode.BINARY_PLUS_RATIONALE:
        return (
            f"{base}\nCorrect answer: {clean_optional_str(answer)}\n"
            f"Reference rationale: {clean_optional_str(rationale)}"
        )
    raise ValueError(f"Unsupported feedback mode: {mode}")


@dataclass(frozen=True, slots=True)
class RecoveryTask:
    coordinate: RecoveryCoordinate
    provider: str = ""
    model_family: str = ""
    requested_registry_revision: str = ""
    availability_status: str = ""

    @property
    def shard_material(self) -> tuple[str, str, str]:
        """Keep both one-shot repetitions for a question on the same owner."""

        return (
            self.coordinate.model_id,
            self.coordinate.evaluation_setting,
            self.coordinate.original_question_id,
        )


@dataclass(frozen=True, slots=True)
class PromptSource:
    target_question: str
    corrected_answer: str
    target_rationale: str
    target_image_paths: tuple[Path, ...]
    examples: tuple[ContextExample, ContextExample]


def load_prompt_source_tables(
    originals_csv: Path,
    links_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = pd.read_csv(originals_csv, dtype=str, keep_default_na=False)
    links = pd.read_csv(links_csv, dtype=str, keep_default_na=False)
    required_original = {
        "original_question_id",
        "question",
        "answer",
        "rationale",
        "has_image",
        "image_paths_or_ids",
    }
    required_links = {
        "original_question_id",
        *{
            f"example_{index}_{field}"
            for index in (1, 2)
            for field in (
                "question_id",
                "question",
                "answer",
                "rationale",
                "has_image",
                "image_paths_or_ids",
            )
        },
    }
    missing_original = sorted(required_original - set(originals.columns))
    missing_links = sorted(required_links - set(links.columns))
    if missing_original or missing_links:
        raise ValueError(
            "Canonical prompt inputs lack columns: "
            f"originals={missing_original}, links={missing_links}"
        )
    if originals["original_question_id"].duplicated().any():
        raise ValueError("Processed originals contain duplicate question IDs")
    if links["original_question_id"].duplicated().any():
        raise ValueError("Example links contain duplicate question IDs")
    return (
        originals.set_index("original_question_id", drop=False),
        links.set_index("original_question_id", drop=False),
    )


def _selected_static_examples(
    task: RecoveryTask, source: PromptSource
) -> tuple[ContextExample, ...]:
    variant = task.coordinate.concrete_variant
    if variant == "zero_shot":
        return ()
    if variant == "one_shot_a":
        return (source.examples[0],)
    if variant == "one_shot_b":
        return (source.examples[1],)
    if variant == "two_shot":
        return source.examples
    raise ValueError("LFE does not use a static prompt")


def _lfe_question_message(
    question: str,
    image_paths: Sequence[Path],
    *,
    image_label: str,
    semantic_kind: str = "example_question",
) -> AdaptiveMessage:
    prompt = lfe_question_prompt(question)
    if image_paths:
        count = len(image_paths)
        span = "1" if count == 1 else f"1-{count}"
        prompt += f"\n\nATTACHED IMAGE ORDER:\nimage(s) {span}: {image_label}"
    return AdaptiveMessage(
        role="user",
        content=prompt,
        image_paths=tuple(image_paths),
        semantic_kind=semantic_kind,
    )


def _source_transcript(messages: Sequence[AdaptiveMessage]) -> str:
    """Match the exact historical uppercase-role feedback-judge transcript."""

    return "\n\n".join(f"{message.role.upper()}: {message.content}" for message in messages)


def _append_canonical_message(messages: list[AdaptiveMessage], message: AdaptiveMessage) -> None:
    """Apply the project's exact adjacent-same-role chat normalization."""

    if messages and messages[-1].role == message.role:
        previous = messages[-1]
        messages[-1] = AdaptiveMessage(
            role=previous.role,
            content=f"{previous.content}\n\n{message.content}",
            image_paths=(*previous.image_paths, *message.image_paths),
            reasoning_content=(previous.reasoning_content or message.reasoning_content),
            # The newest question determines which complete user turn is
            # current.  Earlier feedback remains byte-exact in the merged text.
            semantic_kind=message.semantic_kind or previous.semantic_kind,
        )
    else:
        messages.append(message)


def _require_nonblank_prompt_source(name: str, value: str) -> None:
    """Require substantive source text while preserving its exact bytes."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain non-whitespace source text")


def preflight_prompt_source(
    task: RecoveryTask,
    source: PromptSource,
) -> tuple[PromptEnvelope, ...]:
    """Build every provider-visible question envelope without dispatching it.

    This preflight deliberately preserves the verbatim frozen question and
    answer strings.  Boundary CR/LF and spaces are source bytes, not missing
    content.  Constructing all three LFE turns also exercises adjacent-user
    message normalization and ordered image bindings before any paid or local
    generation call is made.
    """

    coordinate = task.coordinate
    _require_nonblank_prompt_source("target corrected answer", source.corrected_answer)
    for index, example in enumerate(source.examples, start=1):
        _require_nonblank_prompt_source(f"context example {index} corrected answer", example.answer)

    if coordinate.evaluation_setting != "learning_from_experience":
        return (
            build_static_prompt_envelope(
                coordinate,
                target_question=source.target_question,
                target_image_paths=(
                    source.target_image_paths if coordinate.model_modality == "multimodal" else ()
                ),
                examples=_selected_static_examples(task, source),
            ),
        )

    use_images = coordinate.model_modality == "multimodal"
    messages: list[AdaptiveMessage] = []
    envelopes: list[PromptEnvelope] = []
    for index, example in enumerate(source.examples, start=1):
        example_paths = tuple(Path(path) for path in example.image_paths) if use_images else ()
        _append_canonical_message(
            messages,
            _lfe_question_message(
                example.question,
                example_paths,
                image_label="solved example 1",
            ),
        )
        envelopes.append(
            PromptEnvelope(
                coordinate=coordinate,
                turn=f"example_{index}",
                messages=tuple(messages),
                target_question=example.question,
                target_image_paths=example_paths,
            )
        )
        # These messages are never dispatched.  They reproduce the roles and
        # merge boundaries of a successful LFE history so the next question's
        # complete provider-visible envelope can be validated offline.
        _append_canonical_message(
            messages,
            AdaptiveMessage(
                role="assistant",
                content="[prompt-source preflight response]",
                semantic_kind="preflight_assistant_response",
            ),
        )
        _append_canonical_message(
            messages,
            AdaptiveMessage(
                role="user",
                content=lfe_feedback(
                    False,
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
    envelopes.append(
        PromptEnvelope(
            coordinate=coordinate,
            turn="target",
            messages=tuple(messages),
            target_question=source.target_question,
            target_image_paths=target_paths,
        )
    )
    return tuple(envelopes)
