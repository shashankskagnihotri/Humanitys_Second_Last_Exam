"""Gemini HLE correctness and 0--10 closeness evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hsle.config import require_key, resolve_path
from hsle.prompts import render_closeness_prompt, render_hle_judge_prompt


DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"


class HLEJudgment(BaseModel):
    """Structured output compatible with the official HLE judge fields."""

    model_config = ConfigDict(extra="forbid")

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int = Field(ge=0, le=100)
    strict: bool


def _safety_settings(types: Any) -> list[Any]:
    categories = [
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    ]
    return [
        types.SafetySetting(
            category=category,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
        for category in categories
    ]


def _hle_config() -> object:
    from google.genai import types

    return types.GenerateContentConfig(
        seed=0,
        candidate_count=1,
        max_output_tokens=65536,
        response_mime_type="application/json",
        response_json_schema=HLEJudgment.model_json_schema(),
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        safety_settings=_safety_settings(types),
    )


def _closeness_config() -> object:
    from google.genai import types

    return types.GenerateContentConfig(
        seed=0,
        candidate_count=1,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
        response_mime_type="text/plain",
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        safety_settings=_safety_settings(types),
    )


def _gemini_client() -> object:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install google-genai to run the evaluators.") from exc
    return genai.Client(
        api_key=require_key("GEMINI_API_KEY"),
        enterprise=False,
        vertexai=False,
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    for value in (
        getattr(exc, "retry_after", None),
        getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
        if getattr(exc, "response", None) is not None
        else None,
    ):
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def _call_with_retries(action: Any, attempts: int, base_delay: float) -> object:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            retry_after = _retry_after_seconds(exc)
            delay = retry_after if retry_after is not None else min(60.0, base_delay * (2**attempt))
            delay *= 0.9 + random.Random(attempt).random() * 0.2
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def grade_hle_response(
    question: str,
    response: str,
    correct_answer: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    client: object | None = None,
    attempts: int = 6,
    retry_delay: float = 2.0,
) -> HLEJudgment:
    """Apply the exact HLE prompt and return one validated judgment."""

    judge_client = client or _gemini_client()
    prompt = render_hle_judge_prompt(question, response, correct_answer)
    raw = _call_with_retries(
        lambda: judge_client.models.generate_content(
            model=model,
            contents=prompt,
            config=_hle_config(),
        ),
        attempts,
        retry_delay,
    )
    parsed = getattr(raw, "parsed", None)
    if isinstance(parsed, HLEJudgment):
        result = parsed
    elif isinstance(parsed, BaseModel):
        result = HLEJudgment.model_validate(parsed.model_dump())
    elif isinstance(parsed, dict):
        result = HLEJudgment.model_validate(parsed)
    else:
        result = HLEJudgment.model_validate(json.loads(getattr(raw, "text", "")))
    if result.strict is not True:
        raise ValueError("HLE judge returned strict != true")
    return result


def grade_closeness(
    *,
    question: str,
    ground_truth_answer: str,
    ground_truth_rationale: str,
    model_answer: str,
    model_raw_output: str,
    model_explanation: str,
    model: str = DEFAULT_JUDGE_MODEL,
    client: object | None = None,
    attempts: int = 6,
    retry_delay: float = 2.0,
) -> int:
    """Apply the study's exact closeness prompt and parse one integer."""

    judge_client = client or _gemini_client()
    prompt = render_closeness_prompt(
        question=question,
        ground_truth_answer=ground_truth_answer,
        ground_truth_rationale=ground_truth_rationale,
        model_answer=model_answer,
        model_raw_output=model_raw_output,
        model_explanation=model_explanation,
    )
    raw = _call_with_retries(
        lambda: judge_client.models.generate_content(
            model=model,
            contents=prompt,
            config=_closeness_config(),
        ),
        attempts,
        retry_delay,
    )
    match = re.fullmatch(r"\s*(10|[0-9])\s*", getattr(raw, "text", "") or "")
    if match is None:
        raise ValueError(f"Invalid closeness response: {getattr(raw, 'text', '')!r}")
    return int(match.group(1))


class RateLimiter:
    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self.last_started = 0.0

    def wait(self) -> None:
        remaining = self.interval - (time.monotonic() - self.last_started)
        if remaining > 0:
            time.sleep(remaining)
        self.last_started = time.monotonic()


def _text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


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


def judge_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    metric: str,
    judge_model: str,
    requests_per_minute: float,
    attempts: int,
    retry_delay: float,
    limit: int | None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    if "status" in frame:
        complete_mask = frame["status"].astype(str).str.strip().eq("complete")
        skipped = int((~complete_mask).sum())
        if skipped:
            print(
                f"Skipping {skipped:,} non-complete generation rows; "
                "they remain missing coordinates for the metric policy."
            )
        frame = frame.loc[complete_mask].copy()
    if limit is not None:
        frame = frame.head(limit).copy()
    if "response_id" not in frame:
        frame["response_id"] = [f"row_{index:08d}" for index in range(len(frame))]
    if frame["response_id"].eq("").any() or frame["response_id"].duplicated().any():
        raise ValueError("response_id must be nonblank and unique")

    completed = pd.DataFrame()
    if output_csv.is_file() and output_csv.stat().st_size > 0:
        completed = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    if not completed.empty:
        hle_done = (
            completed.get("hle_correct", pd.Series("", index=completed.index))
            .astype(str)
            .str.strip()
            .ne("")
        )
        closeness_done = (
            completed.get("closeness_score", pd.Series("", index=completed.index))
            .astype(str)
            .str.strip()
            .ne("")
        )
        if metric == "hle":
            reusable = hle_done
        elif metric == "closeness":
            reusable = closeness_done
        else:
            reusable = hle_done & closeness_done
        completed = completed.loc[reusable].copy()
    done = set(completed.get("response_id", pd.Series(dtype=str)))
    records = completed.to_dict(orient="records")
    client = _gemini_client()
    limiter = RateLimiter(requests_per_minute)

    for _, row in frame.iterrows():
        response_id = row["response_id"]
        if response_id in done:
            continue
        question = _text(row, "question")
        truth = _text(row, "ground_truth_answer", "answer", "correct_answer")
        parsed_answer = _text(row, "model_parsed_answer", "parsed_answer", "model_answer")
        raw_answer = _text(row, "model_raw_output", "raw_output", "response")
        model_response = raw_answer or parsed_answer
        if not question or not truth:
            raise ValueError(f"{response_id} lacks question or ground-truth answer")

        result: dict[str, Any] = row.to_dict()
        result["judge_model"] = judge_model
        if metric in {"hle", "both"}:
            limiter.wait()
            judgment = grade_hle_response(
                question,
                model_response,
                truth,
                model=judge_model,
                client=client,
                attempts=attempts,
                retry_delay=retry_delay,
            )
            result.update(
                {
                    "hle_extracted_final_answer": judgment.extracted_final_answer,
                    "hle_reasoning": judgment.reasoning,
                    "hle_correct": judgment.correct,
                    "hle_confidence": judgment.confidence,
                }
            )
        if metric in {"closeness", "both"}:
            limiter.wait()
            result["closeness_score"] = grade_closeness(
                question=question,
                ground_truth_answer=truth,
                ground_truth_rationale=_text(row, "ground_truth_rationale", "rationale"),
                model_answer=parsed_answer,
                model_raw_output=raw_answer,
                model_explanation=_text(row, "model_explanation", "extracted_explanation"),
                model=judge_model,
                client=client,
                attempts=attempts,
                retry_delay=retry_delay,
            )
        records.append(result)
        done.add(response_id)
        _atomic_csv(pd.DataFrame(records), output_csv)
    return pd.DataFrame(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=["hle", "closeness", "both"], default="both")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--requests-per-minute", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = judge_csv(
        resolve_path(args.input),
        resolve_path(args.output),
        metric=args.metric,
        judge_model=args.judge_model,
        requests_per_minute=args.requests_per_minute,
        attempts=args.attempts,
        retry_delay=args.retry_delay,
        limit=args.limit,
    )
    print(f"Wrote {len(result):,} judged rows to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
