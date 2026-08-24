"""Aggregate local judged-response rows into model-setting metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from hsle.config import load_yaml, resolve_path


SETTING_ORDER = (
    "zero_shot",
    "one_shot",
    "two_shot",
    "learning_from_experience",
)

_AUDITED_TOTALS = {
    "expected_coordinates": 77_315,
    "real_nonblank_responses": 63_769,
    "terminal_policy_settlements": 325,
    "callable_coordinates": 13_148,
    "paid_no_replay_coordinates": 73,
}
_AUDITED_DERIVED_TOTALS = {
    "scientifically_settled_coordinates": 64_094,
    "unresolved_coordinates": 13_221,
}
_AUDITED_VARIANTS = {
    "zero_shot": (26, 12_998, 2_450, 15),
    "one_shot_a": (26, 13_017, 2_430, 16),
    "one_shot_b": (26, 12_999, 2_444, 20),
    "two_shot": (26, 13_002, 2_445, 16),
    "learning_from_experience": (24, 12_078, 3_379, 6),
}
_AUDITED_REMAINING_ROUTES = {
    "public_openrouter_runner": {
        "moonshotai/Kimi-K2-Thinking",
        "moonshotai/Kimi-K2.5",
        "moonshotai/Kimi-K2.6",
        "moonshotai/Kimi-K3",
        "qwen/qwen3.8-max",
    },
    "outside_public_runner": {"moonshotai/Kimi-K2-Instruct"},
    "cap_held_requires_credit_or_new_authority": {"MiniMaxAI/MiniMax-M2.5"},
    "local_cluster_routes": {
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    },
}
_STATUS_COUNTS = {
    "strict_complete": 23,
    "terminal_complete": 1,
    "incomplete": 9,
}
_PARTITION_FIELDS = (
    "real_responses",
    "terminal_settlements",
    "callable_coordinates",
    "paid_no_replay_coordinates",
)
_RESPONSE_COLUMNS = (
    "model_parsed_answer",
    "model_raw_output",
    "parsed_answer",
    "raw_output",
    "model_answer",
    "response",
)


def _variant(row: pd.Series) -> str:
    setting = str(row["evaluation_setting"]).strip()
    if setting != "one_shot":
        return setting
    value = str(row.get("setting_instance_id", "")).strip().casefold()
    if value.endswith("_a") or "example_1" in value or value.endswith("__1") or value == "1":
        return "one_shot_a"
    if value.endswith("_b") or "example_2" in value or value.endswith("__2") or value == "2":
        return "one_shot_b"
    raise ValueError(f"Cannot identify one-shot A/B variant from {value!r}")


def _correct(value: object) -> float:
    normalized = str(value).strip().casefold()
    if normalized in {"yes", "true", "1", "correct"}:
        return 1.0
    if normalized in {"no", "false", "0", "incorrect"}:
        return 0.0
    raise ValueError(f"Invalid HLE correctness value: {value!r}")


def _registry_integer(row: dict[str, Any], field: str, model_id: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"configs/models.yaml {model_id!r} field {field!r} must be a non-negative integer"
        )
    return value


def _load_registry() -> list[dict[str, Any]]:
    payload = load_yaml("configs/models.yaml")
    if not isinstance(payload, dict):
        raise ValueError("configs/models.yaml must contain a mapping")
    raw_snapshot = payload.get("audited_generation_snapshot")
    if not isinstance(raw_snapshot, dict):
        raise ValueError("configs/models.yaml lacks audited_generation_snapshot")
    for field, expected in _AUDITED_TOTALS.items():
        value = raw_snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"configs/models.yaml audited_generation_snapshot.{field} must be "
                f"{expected}, found {value!r}"
            )
    for field, expected in _AUDITED_DERIVED_TOTALS.items():
        value = raw_snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"configs/models.yaml audited_generation_snapshot.{field} must be "
                f"{expected}, found {value!r}"
            )

    raw_variants = payload.get("concrete_variant_census")
    if not isinstance(raw_variants, dict) or set(raw_variants) != set(_AUDITED_VARIANTS):
        raise ValueError("configs/models.yaml has an invalid concrete_variant_census scope")
    variant_fields = ("complete_models", "settled", "callable", "paid_no_replay")
    for variant, expected in _AUDITED_VARIANTS.items():
        raw_variant = raw_variants.get(variant)
        if not isinstance(raw_variant, dict):
            raise ValueError(f"configs/models.yaml census for {variant!r} must be a mapping")
        found = tuple(raw_variant.get(field) for field in variant_fields)
        if found != expected:
            raise ValueError(
                f"configs/models.yaml census for {variant!r} must be {expected}, found {found}"
            )
        if sum(found[1:]) != 15_463:
            raise ValueError(f"configs/models.yaml census for {variant!r} does not close")

    raw_routes = payload.get("remaining_work_routing")
    if not isinstance(raw_routes, dict) or set(raw_routes) != set(_AUDITED_REMAINING_ROUTES):
        raise ValueError("configs/models.yaml has an invalid remaining_work_routing scope")
    routed_models: set[str] = set()
    for route, expected in _AUDITED_REMAINING_ROUTES.items():
        raw_model_ids = raw_routes.get(route)
        if not isinstance(raw_model_ids, list) or set(raw_model_ids) != expected:
            raise ValueError(
                f"configs/models.yaml remaining route {route!r} must be {sorted(expected)}"
            )
        if len(raw_model_ids) != len(set(raw_model_ids)) or routed_models & set(raw_model_ids):
            raise ValueError(
                "configs/models.yaml remaining-work routes overlap or duplicate models"
            )
        routed_models.update(raw_model_ids)

    raw_rows = payload.get("models")
    if not isinstance(raw_rows, list) or len(raw_rows) != 33:
        found = len(raw_rows) if isinstance(raw_rows, list) else type(raw_rows).__name__
        raise ValueError(f"configs/models.yaml must contain exactly 33 models; found {found}")

    registry: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, dict):
            raise ValueError(f"configs/models.yaml model row {index} must be a mapping")
        model_id = str(raw_row.get("model_id", "")).strip()
        display_name = str(raw_row.get("display_name", "")).strip()
        family = str(raw_row.get("family", "")).strip()
        modality = str(raw_row.get("modality", "")).strip()
        status = str(raw_row.get("generation_status", "")).strip()
        if not model_id or not display_name or not family:
            raise ValueError(f"configs/models.yaml model row {index} has blank identity metadata")
        if modality not in {"multimodal", "text_only"}:
            raise ValueError(f"configs/models.yaml {model_id!r} has invalid modality {modality!r}")
        if status not in _STATUS_COUNTS:
            raise ValueError(
                f"configs/models.yaml {model_id!r} has invalid generation_status {status!r}"
            )
        counts = {field: _registry_integer(raw_row, field, model_id) for field in _PARTITION_FIELDS}
        expected = 2_455 if modality == "multimodal" else 2_085
        if sum(counts.values()) != expected:
            raise ValueError(
                f"configs/models.yaml {model_id!r} four-way partition sums to "
                f"{sum(counts.values())}, expected {expected}"
            )
        real = counts["real_responses"]
        terminal = counts["terminal_settlements"]
        callable_count = counts["callable_coordinates"]
        paid_no_replay = counts["paid_no_replay_coordinates"]
        if status == "strict_complete" and (
            real != expected or terminal != 0 or callable_count != 0 or paid_no_replay != 0
        ):
            raise ValueError(f"configs/models.yaml {model_id!r} violates strict_complete semantics")
        if status == "terminal_complete" and (
            real + terminal != expected
            or terminal == 0
            or callable_count != 0
            or paid_no_replay != 0
        ):
            raise ValueError(
                f"configs/models.yaml {model_id!r} violates terminal_complete semantics"
            )
        if status == "incomplete" and real + terminal >= expected:
            raise ValueError(f"configs/models.yaml {model_id!r} violates incomplete semantics")
        registry.append(
            {
                "model_id": model_id,
                "model_display_name": display_name,
                "model_family": family,
                "model_modality": modality,
                "generation_status": status,
                **counts,
            }
        )

    model_ids = [row["model_id"] for row in registry]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("configs/models.yaml contains duplicate model IDs")
    display_names = [row["model_display_name"] for row in registry]
    if len(set(display_names)) != len(display_names):
        raise ValueError("configs/models.yaml contains duplicate display names")
    families = {row["model_family"] for row in registry}
    if len(families) != 13:
        raise ValueError(f"configs/models.yaml must contain 13 families; found {len(families)}")
    modality_counts = {
        modality: sum(row["model_modality"] == modality for row in registry)
        for modality in ("multimodal", "text_only")
    }
    if modality_counts != {"multimodal": 23, "text_only": 10}:
        raise ValueError(
            "configs/models.yaml modality census must be 23 multimodal and 10 text-only; "
            f"found {modality_counts}"
        )
    status_counts = {
        status: sum(row["generation_status"] == status for row in registry)
        for status in _STATUS_COUNTS
    }
    if status_counts != _STATUS_COUNTS:
        raise ValueError(
            f"configs/models.yaml generation-status census must be {_STATUS_COUNTS}; "
            f"found {status_counts}"
        )
    incomplete_models = {
        row["model_id"] for row in registry if row["generation_status"] == "incomplete"
    }
    if routed_models != incomplete_models:
        raise ValueError(
            "configs/models.yaml remaining-work routing must cover exactly the incomplete models"
        )
    totals = {
        "expected_coordinates": sum(
            2_455 if row["model_modality"] == "multimodal" else 2_085 for row in registry
        ),
        "real_nonblank_responses": sum(row["real_responses"] for row in registry),
        "terminal_policy_settlements": sum(row["terminal_settlements"] for row in registry),
        "callable_coordinates": sum(row["callable_coordinates"] for row in registry),
        "paid_no_replay_coordinates": sum(row["paid_no_replay_coordinates"] for row in registry),
    }
    if totals != _AUDITED_TOTALS:
        raise ValueError(
            f"configs/models.yaml model rows total {totals}, expected {_AUDITED_TOTALS}"
        )
    return registry


def _expected_keys(
    registry: list[dict[str, Any]],
    originals: pd.DataFrame,
) -> set[tuple[str, str, str, str]]:
    image = originals["has_image"].astype(str).str.casefold().isin({"1", "true", "yes"})
    all_ids = tuple(originals["original_question_id"].astype(str))
    text_ids = tuple(originals.loc[~image, "original_question_id"].astype(str))
    expected: set[tuple[str, str, str, str]] = set()
    for model in registry:
        ids = all_ids if model["model_modality"] == "multimodal" else text_ids
        for setting in SETTING_ORDER:
            variants = ("one_shot_a", "one_shot_b") if setting == "one_shot" else (setting,)
            expected.update(
                (model["model_id"], setting, question_id, variant)
                for question_id in ids
                for variant in variants
            )
    return expected


def aggregate(
    judged: pd.DataFrame,
    originals: pd.DataFrame,
    *,
    missing_policy: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "model_id",
        "evaluation_setting",
        "setting_instance_id",
        "original_question_id",
        "generation_completion_type",
        "closeness_score",
    }
    missing = sorted(required - set(judged.columns))
    if missing:
        raise ValueError(f"Judged rows lack columns: {missing}")
    correct_column = next(
        (
            column
            for column in ("hle_correct", "judge_correct", "official_hle_correct")
            if column in judged
        ),
        None,
    )
    if correct_column is None:
        raise ValueError("Judged rows lack an HLE correctness column")
    registry = _load_registry()
    registry_lookup = {row["model_id"]: row for row in registry}

    work = judged.copy()
    work["model_id"] = work["model_id"].astype(str).str.strip()
    work["evaluation_setting"] = work["evaluation_setting"].astype(str).str.strip()
    work["original_question_id"] = work["original_question_id"].astype(str).str.strip()
    work["generation_completion_type"] = (
        work["generation_completion_type"].fillna("").astype(str).str.strip()
    )
    unknown = sorted(set(work["model_id"]) - set(registry_lookup))
    if unknown:
        raise ValueError(f"Judged rows contain models outside the release scope: {unknown}")
    work["variant"] = work.apply(_variant, axis=1)
    work["hle_value"] = work[correct_column].map(_correct)
    work["closeness_value"] = pd.to_numeric(work["closeness_score"], errors="raise")
    if (
        not work["closeness_value"].between(0, 10).all()
        or not work["closeness_value"].mod(1).eq(0).all()
    ):
        raise ValueError("Closeness scores must be integers from 0 through 10")
    keys = ["model_id", "evaluation_setting", "original_question_id", "variant"]

    completion_types = set(work["generation_completion_type"])
    allowed_completion_types = {"real_nonblank", "terminal_policy"}
    invalid_completion_types = sorted(completion_types - allowed_completion_types)
    if invalid_completion_types:
        raise ValueError(
            "generation_completion_type must be real_nonblank or terminal_policy; "
            f"found {invalid_completion_types}"
        )
    response_columns = [column for column in _RESPONSE_COLUMNS if column in work.columns]
    if not response_columns:
        raise ValueError(
            f"Judged rows lack a response column; expected one of {list(_RESPONSE_COLUMNS)}"
        )
    response_nonblank = pd.DataFrame(
        {
            column: work[column].fillna("").astype(str).str.strip().ne("")
            for column in response_columns
        }
    ).any(axis=1)
    real = work["generation_completion_type"].eq("real_nonblank")
    terminal = work["generation_completion_type"].eq("terminal_policy")
    invalid_real = work.loc[real & ~response_nonblank, keys]
    if not invalid_real.empty:
        raise ValueError(
            "real_nonblank rows must contain a nonblank response; sample="
            f"{invalid_real.head(10).to_dict(orient='records')}"
        )
    invalid_terminal_response = work.loc[terminal & response_nonblank, keys]
    if not invalid_terminal_response.empty:
        raise ValueError(
            "terminal_policy rows must have blank response fields; sample="
            f"{invalid_terminal_response.head(10).to_dict(orient='records')}"
        )
    invalid_terminal_scores = work.loc[
        terminal & (~work["hle_value"].eq(0.0) | ~work["closeness_value"].eq(0.0)),
        keys,
    ]
    if not invalid_terminal_scores.empty:
        raise ValueError(
            "terminal_policy rows must be HLE incorrect/0 with closeness 0; sample="
            f"{invalid_terminal_scores.head(10).to_dict(orient='records')}"
        )

    duplicates = work.loc[work.duplicated(keys, keep=False), keys]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate response coordinates: "
            f"{duplicates.drop_duplicates().head(10).to_dict(orient='records')}"
        )

    expected = _expected_keys(registry, originals)
    actual = set(map(tuple, work[keys].itertuples(index=False, name=None)))
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"Unexpected response coordinates: {extra[:10]}")
    missing_keys = sorted(expected - actual)
    if missing_policy != "none":
        raise ValueError(f"Unknown missing policy: {missing_policy}")
    if missing_keys:
        raise ValueError(
            f"Missing response coordinates are not allowed; sample={missing_keys[:10]}"
        )

    metadata = pd.DataFrame(registry)
    registry_columns = [column for column in metadata.columns if column != "model_id"]
    work = work.drop(columns=registry_columns, errors="ignore").merge(
        metadata, on="model_id", validate="many_to_one"
    )
    question_layer = (
        work.groupby(
            [
                "model_id",
                "model_display_name",
                "model_family",
                "model_modality",
                "generation_status",
                "evaluation_setting",
                "original_question_id",
            ],
            sort=False,
            dropna=False,
        )
        .agg(
            hle_question_score=("hle_value", "mean"),
            closeness_question_score=("closeness_value", "mean"),
        )
        .reset_index()
    )
    grouped = question_layer.groupby(
        [
            "model_id",
            "model_display_name",
            "model_family",
            "model_modality",
            "generation_status",
            "evaluation_setting",
        ],
        sort=False,
        dropna=False,
    )
    metrics = grouped.agg(
        hle_accuracy_pct=(
            "hle_question_score",
            lambda values: 100.0 * float(values.mean()),
        ),
        closeness_mean=("closeness_question_score", "mean"),
        eligible_questions=("original_question_id", "nunique"),
    ).reset_index()
    closeness_std = (
        grouped.agg(
            closeness_std=(
                "closeness_question_score",
                lambda values: (float(values.std(ddof=1)) if len(values) > 1 else 0.0),
            ),
            eligible_questions=("original_question_id", "nunique"),
        )
        .reset_index()
        .drop(columns=["model_modality"])
    )
    expected_rows = len(registry) * len(SETTING_ORDER)
    if len(metrics) != expected_rows or len(closeness_std) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} model-setting rows, found "
            f"{len(metrics)} metrics and {len(closeness_std)} standard deviations"
        )
    for row in metrics.itertuples(index=False):
        expected_questions = 491 if row.model_modality == "multimodal" else 417
        if row.eligible_questions != expected_questions:
            raise ValueError(
                f"{row.model_id}/{row.evaluation_setting} has "
                f"{row.eligible_questions} questions, expected {expected_questions}"
            )
        if not (math.isfinite(row.hle_accuracy_pct) and math.isfinite(row.closeness_mean)):
            raise ValueError("Non-finite aggregate metric")
    order = {setting: index for index, setting in enumerate(SETTING_ORDER)}
    model_order = {row["model_id"]: index for index, row in enumerate(registry)}
    for frame in (metrics, closeness_std):
        frame["_model_order"] = frame["model_id"].map(model_order)
        frame["_setting_order"] = frame["evaluation_setting"].map(order)
        frame.sort_values(["_model_order", "_setting_order"], inplace=True, kind="mergesort")
        frame.drop(columns=["_model_order", "_setting_order"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return metrics, closeness_std


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--originals",
        type=Path,
        default=Path("data/processed/hsle_original_questions.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/metrics/model_setting_metrics.csv"),
    )
    parser.add_argument(
        "--std-output",
        type=Path,
        default=Path("outputs/metrics/model_setting_closeness_std.csv"),
    )
    parser.add_argument(
        "--missing-policy",
        choices=["none"],
        default="none",
        help="require a complete 33-model judged matrix; synthetic scores are forbidden",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    judged = pd.concat(
        [pd.read_csv(resolve_path(path), dtype=str, keep_default_na=False) for path in args.input],
        ignore_index=True,
    )
    originals = pd.read_csv(resolve_path(args.originals), dtype=str, keep_default_na=False)
    metrics, closeness_std = aggregate(
        judged,
        originals,
        missing_policy=args.missing_policy,
    )
    metrics_path = resolve_path(args.metrics_output)
    std_path = resolve_path(args.std_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    std_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    closeness_std.to_csv(std_path, index=False)
    print(f"Wrote {len(metrics):,} metric rows to {metrics_path}")
    print(f"Wrote {len(closeness_std):,} standard-deviation rows to {std_path}")


if __name__ == "__main__":
    main()
