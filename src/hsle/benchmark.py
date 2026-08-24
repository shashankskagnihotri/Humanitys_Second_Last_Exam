"""Generate HSLE responses for the four evaluation settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from hsle.config import load_yaml, resolve_path
from hsle.judge import DEFAULT_JUDGE_MODEL, RateLimiter, grade_hle_response
from hsle.prompts import (
    ContextExample,
    lfe_feedback,
    one_shot_prompt,
    two_shot_prompt,
    zero_shot_prompt,
)
from hsle.providers import Generation, Message, Provider, build_provider


SETTINGS = (
    "zero_shot",
    "one_shot",
    "two_shot",
    "learning_from_experience",
)


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _paths(value: object) -> tuple[Path, ...]:
    text = str(value).strip()
    if not text:
        return ()
    values: list[str]
    if text.startswith("["):
        parsed = json.loads(text)
        values = [str(item) for item in parsed]
    else:
        values = [item for item in text.replace(";", "|").split("|") if item.strip()]
    resolved = tuple(resolve_path(item.strip()) for item in values)
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing referenced images: {missing}")
    return resolved


def _model_metadata(model_id: str) -> tuple[str, str]:
    payload = load_yaml("configs/models.yaml")
    for row in payload.get("models", []):
        if str(row.get("model_id", "")).strip() == model_id:
            return str(row["family"]), str(row["modality"])
    raise ValueError(f"{model_id!r} is not in configs/models.yaml; pass --family and --modality.")


def _append(messages: list[Message], message: Message) -> None:
    """Coalesce adjacent roles exactly as the study's chat topology requires."""

    if messages and messages[-1].role == message.role:
        prior = messages[-1]
        messages[-1] = Message(
            role=prior.role,
            text=f"{prior.text}\n\n{message.text}",
            images=prior.images + message.images,
        )
    else:
        messages.append(message)


def _transcript(messages: list[Message]) -> str:
    return "\n\n".join(f"{message.role.upper()}: {message.text}" for message in messages)


def _response_id(
    model_id: str,
    setting: str,
    question_id: str,
    setting_instance_id: str,
) -> str:
    payload = "\0".join((model_id, setting, question_id, setting_instance_id))
    return "response_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _generate(
    provider: Provider,
    messages: list[Message],
    model: str,
    *,
    attempts: int,
    retry_delay: float,
    limiter: RateLimiter,
) -> Generation:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            limiter.wait()
            generation = provider.generate(messages, model)
            if not isinstance(generation.text, str) or not generation.text.strip():
                raise ValueError("Provider returned a blank generation")
            return generation
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(60.0, retry_delay * (2**attempt)))
    assert last_error is not None
    raise last_error


def _examples(link: pd.Series) -> list[ContextExample]:
    return [
        ContextExample(
            question=str(link[f"example_{index}_question"]).strip(),
            answer=str(link[f"example_{index}_answer"]).strip(),
        )
        for index in (1, 2)
    ]


def _example_images(link: pd.Series, index: int, multimodal: bool) -> tuple[Path, ...]:
    if not multimodal or not _truthy(link.get(f"example_{index}_has_image", "")):
        return ()
    return _paths(link.get(f"example_{index}_image_paths_or_ids", ""))


def _static_plans(
    setting: str,
    original: pd.Series,
    link: pd.Series,
    multimodal: bool,
) -> list[tuple[str, str, list[Message]]]:
    question = str(original["question"]).strip()
    target_images = (
        _paths(original.get("image_paths_or_ids", ""))
        if multimodal and _truthy(original.get("has_image", ""))
        else ()
    )
    examples = _examples(link)
    if setting == "zero_shot":
        return [
            (
                "zero_shot",
                "",
                [Message("user", zero_shot_prompt(question), target_images)],
            )
        ]
    if setting == "one_shot":
        plans: list[tuple[str, str, list[Message]]] = []
        for index, suffix in ((1, "a"), (2, "b")):
            images = _example_images(link, index, multimodal) + target_images
            plans.append(
                (
                    f"one_shot_{suffix}",
                    str(link[f"example_{index}_question_id"]).strip(),
                    [Message("user", one_shot_prompt(examples[index - 1], question), images)],
                )
            )
        return plans
    if setting == "two_shot":
        images = (
            _example_images(link, 1, multimodal)
            + _example_images(link, 2, multimodal)
            + target_images
        )
        instance = ";".join(str(link[f"example_{index}_question_id"]).strip() for index in (1, 2))
        return [
            (
                "two_shot",
                instance,
                [Message("user", two_shot_prompt(examples, question), images)],
            )
        ]
    raise ValueError(f"Static plan does not support {setting}")


def _run_lfe(
    original: pd.Series,
    link: pd.Series,
    provider: Provider,
    request_model: str,
    *,
    multimodal: bool,
    judge_model: str,
    attempts: int,
    retry_delay: float,
    limiter: RateLimiter,
) -> tuple[Generation, str, str]:
    examples = _examples(link)
    messages: list[Message] = []
    for index, example in enumerate(examples, start=1):
        _append(
            messages,
            Message(
                "user",
                zero_shot_prompt(example.question),
                _example_images(link, index, multimodal),
            ),
        )
        generation = _generate(
            provider,
            messages,
            request_model,
            attempts=attempts,
            retry_delay=retry_delay,
            limiter=limiter,
        )
        _append(messages, Message("assistant", generation.text))
        limiter.wait()
        judgment = grade_hle_response(
            example.question,
            generation.text,
            example.answer,
            model=judge_model,
            attempts=attempts,
            retry_delay=retry_delay,
        )
        _append(messages, Message("user", lfe_feedback(judgment.correct == "yes")))

    target_images = (
        _paths(original.get("image_paths_or_ids", ""))
        if multimodal and _truthy(original.get("has_image", ""))
        else ()
    )
    _append(
        messages,
        Message("user", zero_shot_prompt(str(original["question"])), target_images),
    )
    visible_transcript = _transcript(messages)
    result = _generate(
        provider,
        messages,
        request_model,
        attempts=attempts,
        retry_delay=retry_delay,
        limiter=limiter,
    )
    instance = ";".join(str(link[f"example_{index}_question_id"]).strip() for index in (1, 2))
    return result, visible_transcript, instance


def run_benchmark(
    *,
    provider_name: str,
    request_model: str,
    model_id: str,
    family: str,
    modality: str,
    setting: str,
    originals_csv: Path,
    links_csv: Path,
    output_csv: Path,
    judge_model: str,
    requests_per_minute: float,
    attempts: int,
    retry_delay: float,
    limit: int | None,
    dry_run: bool,
    fail_fast: bool,
) -> pd.DataFrame:
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}")
    if modality not in {"multimodal", "text_only"}:
        raise ValueError("modality must be multimodal or text_only")
    originals = pd.read_csv(originals_csv, dtype=str, keep_default_na=False)
    links = pd.read_csv(links_csv, dtype=str, keep_default_na=False)
    links_by_id = links.set_index("original_question_id", verify_integrity=True)
    if modality == "text_only":
        originals = originals.loc[~originals["has_image"].map(_truthy)].copy()
    if limit is not None:
        originals = originals.head(limit).copy()

    existing = pd.DataFrame()
    if output_csv.is_file() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    reusable_statuses = {"complete", "dry_run"} if dry_run else {"complete"}
    if "status" in existing:
        existing = existing.loc[existing["status"].astype(str).isin(reusable_statuses)].copy()
    else:
        existing = existing.iloc[0:0].copy()
    if "generation_completion_type" not in existing:
        existing["generation_completion_type"] = ""
    complete = existing.get("status", pd.Series("", index=existing.index)).eq("complete")
    parsed_nonblank = (
        existing.get("model_parsed_answer", pd.Series("", index=existing.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    raw_nonblank = (
        existing.get("model_raw_output", pd.Series("", index=existing.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    invalid_complete = complete & (
        ~existing["generation_completion_type"].eq("real_nonblank")
        | ~(parsed_nonblank | raw_nonblank)
    )
    if invalid_complete.any():
        raise ValueError(
            "Existing complete rows must have generation_completion_type=real_nonblank "
            "and a nonblank response; refusing ambiguous reuse"
        )
    records = existing.to_dict(orient="records")
    done = set(existing.get("response_id", pd.Series(dtype=str)))
    provider = None if dry_run else build_provider(provider_name)
    limiter = RateLimiter(requests_per_minute)

    for _, original in originals.iterrows():
        question_id = str(original["original_question_id"]).strip()
        if question_id not in links_by_id.index:
            raise ValueError(f"No context-example link for {question_id}")
        link = links_by_id.loc[question_id]
        if isinstance(link, pd.DataFrame):
            raise ValueError(f"Duplicate context-example links for {question_id}")
        plans: list[tuple[str, str, list[Message]]]
        if setting == "learning_from_experience":
            examples = _examples(link)
            instance = ";".join(
                str(link[f"example_{index}_question_id"]).strip() for index in (1, 2)
            )
            plans = [
                (
                    "learning_from_experience",
                    instance,
                    [
                        Message("user", zero_shot_prompt(examples[0].question)),
                        Message("assistant", "<MODEL_RESPONSE_TO_EXAMPLE_1>"),
                        Message("user", "<BINARY_HLE_FEEDBACK_1>"),
                        Message("user", zero_shot_prompt(examples[1].question)),
                        Message("assistant", "<MODEL_RESPONSE_TO_EXAMPLE_2>"),
                        Message("user", "<BINARY_HLE_FEEDBACK_2>"),
                        Message("user", zero_shot_prompt(str(original["question"]))),
                    ],
                )
            ]
        else:
            plans = _static_plans(
                setting,
                original,
                link,
                modality == "multimodal",
            )

        for setting_instance_id, example_ids, messages in plans:
            response_id = _response_id(
                model_id,
                setting,
                question_id,
                setting_instance_id,
            )
            if response_id in done:
                continue
            record: dict[str, Any] = {
                "response_id": response_id,
                "provider": provider_name,
                "request_model": request_model,
                "model_id": model_id,
                "model_family": family,
                "model_modality": modality,
                "evaluation_setting": setting,
                "setting_instance_id": setting_instance_id,
                "example_ids": example_ids,
                "original_question_id": question_id,
                "question": str(original["question"]).strip(),
                "ground_truth_answer": str(original["answer"]).strip(),
                "ground_truth_rationale": str(original.get("rationale", "")).strip(),
                "prompt_transcript": _transcript(messages),
                "model_parsed_answer": "",
                "model_raw_output": "",
                "generation_completion_type": "",
                "source_reported_model_version": "",
                "input_tokens": "",
                "output_tokens": "",
                "status": "dry_run" if dry_run else "pending",
                "error_message": "",
            }
            if not dry_run:
                assert provider is not None
                try:
                    if setting == "learning_from_experience":
                        generation, transcript, lfe_instance = _run_lfe(
                            original,
                            link,
                            provider,
                            request_model,
                            multimodal=modality == "multimodal",
                            judge_model=judge_model,
                            attempts=attempts,
                            retry_delay=retry_delay,
                            limiter=limiter,
                        )
                        record["prompt_transcript"] = transcript
                        record["example_ids"] = lfe_instance
                    else:
                        generation = _generate(
                            provider,
                            messages,
                            request_model,
                            attempts=attempts,
                            retry_delay=retry_delay,
                            limiter=limiter,
                        )
                    record.update(
                        {
                            "model_parsed_answer": generation.text,
                            "model_raw_output": generation.text,
                            "generation_completion_type": "real_nonblank",
                            "source_reported_model_version": generation.model_version,
                            "input_tokens": generation.input_tokens or "",
                            "output_tokens": generation.output_tokens or "",
                            "status": "complete",
                        }
                    )
                except Exception as exc:
                    record["status"] = "failed"
                    record["error_message"] = f"{type(exc).__name__}: {exc}"
                    records.append(record)
                    done.add(response_id)
                    _atomic_csv(pd.DataFrame(records), output_csv)
                    if fail_fast:
                        raise
                    continue
            records.append(record)
            done.add(response_id)
            _atomic_csv(pd.DataFrame(records), output_csv)
    return pd.DataFrame(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True, help="Provider-facing model ID")
    parser.add_argument("--model-id", help="Analytical model ID; defaults to --model")
    parser.add_argument("--family")
    parser.add_argument("--modality", choices=["multimodal", "text_only"])
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument(
        "--originals",
        type=Path,
        default=Path("data/processed/hsle_original_questions.csv"),
    )
    parser.add_argument(
        "--links",
        type=Path,
        default=Path("data/processed/hsle_question_example_links.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--requests-per-minute", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_id = args.model_id or args.model
    if args.family and args.modality:
        family, modality = args.family, args.modality
    elif args.family or args.modality:
        raise ValueError("--family and --modality must be supplied together")
    else:
        family, modality = _model_metadata(model_id)
    result = run_benchmark(
        provider_name=args.provider,
        request_model=args.model,
        model_id=model_id,
        family=family,
        modality=modality,
        setting=args.setting,
        originals_csv=resolve_path(args.originals),
        links_csv=resolve_path(args.links),
        output_csv=resolve_path(args.output),
        judge_model=args.judge_model,
        requests_per_minute=args.requests_per_minute,
        attempts=args.attempts,
        retry_delay=args.retry_delay,
        limit=args.limit,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
    )
    print(f"Wrote {len(result):,} rows to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
