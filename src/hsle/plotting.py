"""Render the exact current 25-model HSLE all-model and family PDF release.

The input is the coordinate-level scored layer, not a pre-aggregated metric
table. One-shot A and B are paired within each model-question unit before any
setting mean is calculated. All-model figures use each model's native question
universe. A family containing any text-only model uses the shared 417-question
text cohort; an all-multimodal family uses all 491 questions.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from hsle.config import load_yaml, resolve_path


SETTING_ORDER = (
    "zero_shot",
    "one_shot",
    "two_shot",
    "learning_from_experience",
)
VARIANT_ORDER = (
    "zero_shot",
    "one_shot_a",
    "one_shot_b",
    "two_shot",
    "learning_from_experience",
)
FAMILY_ORDER = (
    "claude",
    "deepseek",
    "gemini",
    "gemma",
    "gpt",
    "internlm",
    "kimi",
    "llama",
    "llava",
    "minimax",
    "mistral",
    "qwen",
)
HLE_STEM = "hle_eval_all_selected_open_and_closed_source_models_by_prompt_setting"
CLOSENESS_STEM = (
    "gemini-3.5-flash_judged_closeness_all_selected_open_and_closed_source_"
    "models_mean_with_std_by_prompt_setting"
)
RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "hsle-historical-all-available-20260728",
    "hatch.linewidth": 0.7,
}


@dataclass(frozen=True)
class Model:
    model_id: str
    display_name: str
    family: str
    modality: str
    order: int


@dataclass(frozen=True)
class PlotContract:
    model_ids: tuple[str, ...]
    lower_bound_model_ids: frozenset[str]
    family_order: tuple[str, ...]
    setting_labels: dict[str, str]
    setting_colors: dict[str, str]
    setting_hatches: dict[str, str]
    family_colors: dict[str, str]
    expected_coordinate_count: int
    expected_generation_observed_count: int
    expected_hle_observed_count: int
    expected_closeness_observed_count: int
    expected_global_logical_question_units: int
    expected_family_coordinate_count: int
    expected_family_logical_question_units: int
    output_directory: Path
    output_dpi: int


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _require_string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicates")
    return result


def _require_string_mapping(
    value: Any,
    label: str,
    expected_keys: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    mapping = _require_mapping(value, label)
    if set(mapping) != set(expected_keys):
        raise ValueError(
            f"{label} must contain exactly {list(expected_keys)}; found={list(mapping)}"
        )
    result: dict[str, str] = {}
    for key in expected_keys:
        item = mapping[key]
        if not isinstance(item, str) or (not allow_empty and not item.strip()):
            raise ValueError(f"{label}.{key} must be a string")
        result[key] = item
    return result


def _require_positive_integer(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive integer")
    return value


def _load_plot_contract() -> PlotContract:
    payload = load_yaml("configs/plotting.yaml")
    release = _require_mapping(payload.get("release"), "plotting.release")
    model_ids = _require_string_list(release.get("model_ids"), "plotting.release.model_ids")
    if len(model_ids) != 25:
        raise ValueError("plotting.release.model_ids must contain exactly 25 models")
    lower_bound_model_ids = frozenset(
        _require_string_list(
            release.get("lower_bound_model_ids"),
            "plotting.release.lower_bound_model_ids",
        )
    )
    if not lower_bound_model_ids or not lower_bound_model_ids <= set(model_ids):
        raise ValueError("plotting lower-bound models must be a non-empty subset of model_ids")

    settings = _require_mapping(payload.get("settings"), "plotting.settings")
    setting_order = _require_string_list(settings.get("order"), "plotting.settings.order")
    if setting_order != SETTING_ORDER:
        raise ValueError(
            "plotting.settings.order differs from the four-setting publication contract"
        )
    setting_labels = _require_string_mapping(
        settings.get("labels"), "plotting.settings.labels", SETTING_ORDER
    )
    setting_colors = _require_string_mapping(
        settings.get("colors"), "plotting.settings.colors", SETTING_ORDER
    )
    setting_hatches = _require_string_mapping(
        settings.get("hatches"),
        "plotting.settings.hatches",
        SETTING_ORDER,
        allow_empty=True,
    )
    invalid_setting_colors = [
        setting
        for setting, color in setting_colors.items()
        if not mcolors.is_color_like(color)
    ]
    if invalid_setting_colors:
        raise ValueError(f"Invalid setting colors: {invalid_setting_colors}")

    family_order = _require_string_list(payload.get("family_order"), "plotting.family_order")
    if family_order != FAMILY_ORDER:
        raise ValueError("plotting.family_order differs from the current 12-family release")
    family_colors = _require_string_mapping(
        payload.get("family_colors"), "plotting.family_colors", FAMILY_ORDER
    )
    invalid_family_colors = [
        family for family, color in family_colors.items() if not mcolors.is_color_like(color)
    ]
    if invalid_family_colors:
        raise ValueError(f"Invalid family colors: {invalid_family_colors}")

    output = _require_mapping(payload.get("output"), "plotting.output")
    directory = output.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("plotting.output.directory must be a non-empty string")
    if output.get("format") != "pdf":
        raise ValueError("plotting.output.format must be 'pdf'")
    if output.get("titles") is not False:
        raise ValueError("plotting.output.titles must remain false")
    output_dpi = _require_positive_integer(output, "dpi", "plotting.output")
    if output_dpi != 300:
        raise ValueError("plotting.output.dpi must remain 300 for the current release")

    return PlotContract(
        model_ids=model_ids,
        lower_bound_model_ids=lower_bound_model_ids,
        family_order=family_order,
        setting_labels=setting_labels,
        setting_colors=setting_colors,
        setting_hatches=setting_hatches,
        family_colors=family_colors,
        expected_coordinate_count=_require_positive_integer(
            release, "expected_coordinate_count", "plotting.release"
        ),
        expected_generation_observed_count=_require_positive_integer(
            release, "expected_generation_observed_count", "plotting.release"
        ),
        expected_hle_observed_count=_require_positive_integer(
            release, "expected_hle_observed_count", "plotting.release"
        ),
        expected_closeness_observed_count=_require_positive_integer(
            release, "expected_closeness_observed_count", "plotting.release"
        ),
        expected_global_logical_question_units=_require_positive_integer(
            release,
            "expected_global_logical_question_units",
            "plotting.release",
        ),
        expected_family_coordinate_count=_require_positive_integer(
            release, "expected_family_coordinate_count", "plotting.release"
        ),
        expected_family_logical_question_units=_require_positive_integer(
            release,
            "expected_family_logical_question_units",
            "plotting.release",
        ),
        output_directory=Path(directory),
        output_dpi=output_dpi,
    )


def _load_models(contract: PlotContract) -> list[Model]:
    payload = load_yaml("configs/models.yaml")
    rows = payload.get("models")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("configs/models.yaml must contain exactly 32 model rows")
    registry: dict[str, Model] = {}
    registry_order: list[str] = []
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"models[{index}]")
        values: dict[str, str] = {}
        for field in ("model_id", "display_name", "family", "modality"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"models[{index}].{field} must be a non-empty string")
            values[field] = value
        model_id = values["model_id"]
        if model_id in registry:
            raise ValueError(f"Duplicate model_id in configs/models.yaml: {model_id}")
        if values["modality"] not in {"multimodal", "text_only"}:
            raise ValueError(f"Unsupported modality for {model_id}: {values['modality']}")
        registry[model_id] = Model(
            model_id=model_id,
            display_name=values["display_name"],
            family=values["family"],
            modality=values["modality"],
            order=index,
        )
        registry_order.append(model_id)

    unknown = sorted(set(contract.model_ids) - set(registry))
    if unknown:
        raise ValueError(f"Plot contract contains unknown models: {unknown}")
    expected_order = tuple(model_id for model_id in registry_order if model_id in contract.model_ids)
    if contract.model_ids != expected_order:
        raise ValueError("Plot model order differs from configs/models.yaml publication order")
    models = [registry[model_id] for model_id in contract.model_ids]
    modality_counts = {
        modality: sum(model.modality == modality for model in models)
        for modality in ("multimodal", "text_only")
    }
    if modality_counts != {"multimodal": 19, "text_only": 6}:
        raise ValueError(f"Current plot modality census differs: {modality_counts}")
    if {model.family for model in models} != set(contract.family_order):
        raise ValueError("Current plot family scope differs from plotting.family_order")
    return models


def _explicit_bool(value: object, label: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} must be an explicit boolean; found {value!r}")


def _hle_score(value: object) -> float:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "1.0", "yes", "correct"}:
        return 1.0
    if normalized in {"false", "0", "0.0", "no", "incorrect"}:
        return 0.0
    raise ValueError(f"hle_correct must be binary; found {value!r}")


def _optional_hle_score(value: object) -> float:
    if str(value).strip() == "":
        return math.nan
    return _hle_score(value)


def _load_scored_coordinates(
    paths: Sequence[Path],
    contract: PlotContract,
    models: Sequence[Model],
) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one scored-coordinate input is required")
    frames = [
        pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in paths
    ]
    work = pd.concat(frames, ignore_index=True)
    required = {
        "model_id",
        "evaluation_setting",
        "concrete_variant",
        "original_question_id",
        "hle_correct",
        "closeness_score",
        "operational_missing",
        "hle_metric_missing",
        "closeness_metric_missing",
    }
    missing_columns = sorted(required - set(work.columns))
    if missing_columns:
        raise ValueError(f"Scored-coordinate input lacks columns: {missing_columns}")
    for column in (
        "model_id",
        "evaluation_setting",
        "concrete_variant",
        "original_question_id",
    ):
        work[column] = work[column].astype(str).str.strip()
        if work[column].eq("").any():
            raise ValueError(f"Scored-coordinate input has blank {column} values")
    for column in ("operational_missing", "hle_metric_missing", "closeness_metric_missing"):
        work[column] = work[column].map(lambda value, name=column: _explicit_bool(value, name))
    work["hle_value"] = work["hle_correct"].map(_optional_hle_score)
    work["closeness_value"] = pd.to_numeric(work["closeness_score"], errors="coerce")
    invalid_hle = ~work["hle_metric_missing"] & work["hle_value"].isna()
    invalid_closeness = ~work["closeness_metric_missing"] & (
        work["closeness_value"].isna()
        | ~work["closeness_value"].between(0, 10)
        | ~work["closeness_value"].mod(1).eq(0)
    )
    if invalid_hle.any():
        raise ValueError("Every non-missing HLE coordinate must have a binary score")
    if invalid_closeness.any():
        raise ValueError(
            "Every non-missing closeness coordinate must have an integer score from 0 through 10"
        )
    work.loc[work["hle_metric_missing"], "hle_value"] = 0.0
    work.loc[work["closeness_metric_missing"], "closeness_value"] = 0.0

    if len(work) != contract.expected_coordinate_count:
        raise ValueError(
            "Scored-coordinate count differs from the current release: "
            f"expected={contract.expected_coordinate_count}, found={len(work)}"
        )
    if set(work["model_id"]) != set(contract.model_ids):
        raise ValueError("Scored-coordinate model scope differs from the current 25-model release")
    keys = ["model_id", "concrete_variant", "original_question_id"]
    if work.duplicated(keys).any():
        sample = work.loc[work.duplicated(keys, keep=False), keys].head(10)
        raise ValueError(f"Duplicate scored coordinates: {sample.to_dict(orient='records')}")
    if set(work["evaluation_setting"]) != set(SETTING_ORDER):
        raise ValueError("Scored-coordinate settings differ from the four-setting contract")
    if set(work["concrete_variant"]) != set(VARIANT_ORDER):
        raise ValueError("Scored-coordinate variants differ from the five-variant contract")
    expected_settings = work["concrete_variant"].replace(
        {"one_shot_a": "one_shot", "one_shot_b": "one_shot"}
    )
    if not work["evaluation_setting"].eq(expected_settings).all():
        raise ValueError("evaluation_setting and concrete_variant disagree")

    operational_missing = int(work["operational_missing"].sum())
    hle_missing = int(work["hle_metric_missing"].sum())
    closeness_missing = int(work["closeness_metric_missing"].sum())
    expected_operational_missing = (
        contract.expected_coordinate_count - contract.expected_generation_observed_count
    )
    expected_hle_missing = contract.expected_coordinate_count - contract.expected_hle_observed_count
    expected_closeness_missing = (
        contract.expected_coordinate_count - contract.expected_closeness_observed_count
    )
    if (
        operational_missing != expected_operational_missing
        or hle_missing != expected_hle_missing
        or closeness_missing != expected_closeness_missing
    ):
        raise ValueError(
            "Missing-evidence census differs from the current release: "
            f"operational={operational_missing}, HLE={hle_missing}, "
            f"closeness={closeness_missing}"
        )
    if not (
        work.loc[work["operational_missing"], "hle_metric_missing"].all()
        and work.loc[work["operational_missing"], "closeness_metric_missing"].all()
    ):
        raise ValueError("Operationally missing coordinates must be missing both metrics")
    actual_lower_bound_models = frozenset(
        work.loc[
            work["hle_metric_missing"] | work["closeness_metric_missing"],
            "model_id",
        ]
    )
    if actual_lower_bound_models != contract.lower_bound_model_ids:
        raise ValueError(
            "Lower-bound labels differ from models with missing score evidence: "
            f"expected={sorted(contract.lower_bound_model_ids)}, "
            f"found={sorted(actual_lower_bound_models)}"
        )

    model_lookup = {model.model_id: model for model in models}
    native_question_sets: dict[str, set[str]] = {}
    for model_id in contract.model_ids:
        model = model_lookup[model_id]
        expected_questions = 417 if model.modality == "text_only" else 491
        selected = work[work["model_id"].eq(model_id)]
        variant_counts = selected["concrete_variant"].value_counts().to_dict()
        if len(selected) != 5 * expected_questions or variant_counts != {
            variant: expected_questions for variant in VARIANT_ORDER
        }:
            raise ValueError(f"Native coordinate universe differs for {model_id}")
        question_sets = {
            variant: set(
                selected.loc[
                    selected["concrete_variant"].eq(variant), "original_question_id"
                ]
            )
            for variant in VARIANT_ORDER
        }
        if (
            len({frozenset(values) for values in question_sets.values()}) != 1
            or any(len(values) != expected_questions for values in question_sets.values())
        ):
            raise ValueError(f"Question vectors differ across variants for {model_id}")
        native_question_sets[model_id] = question_sets["zero_shot"]

    text_sets = {
        frozenset(native_question_sets[model.model_id])
        for model in models
        if model.modality == "text_only"
    }
    multimodal_sets = {
        frozenset(native_question_sets[model.model_id])
        for model in models
        if model.modality == "multimodal"
    }
    if len(text_sets) != 1 or len(multimodal_sets) != 1:
        raise ValueError("Models do not share the canonical native question vectors")
    text_universe = set(next(iter(text_sets)))
    multimodal_universe = set(next(iter(multimodal_sets)))
    if not (
        len(text_universe) == 417
        and len(multimodal_universe) == 491
        and text_universe < multimodal_universe
    ):
        raise ValueError("The canonical 417 text questions are not a subset of all 491 targets")
    return work


def _family_question_universes(
    layer: pd.DataFrame,
    models: Sequence[Model],
    contract: PlotContract,
) -> dict[str, set[str]]:
    universes: dict[str, set[str]] = {}
    coordinate_total = 0
    for family in contract.family_order:
        family_models = [model for model in models if model.family == family]
        if not family_models:
            raise ValueError(f"Current release has no models in family {family}")
        model_sets = [
            set(
                layer.loc[
                    layer["model_id"].eq(model.model_id)
                    & layer["concrete_variant"].eq("zero_shot"),
                    "original_question_id",
                ]
            )
            for model in family_models
        ]
        common = set.intersection(*model_sets)
        expected = 417 if any(model.modality == "text_only" for model in family_models) else 491
        if len(common) != expected:
            raise ValueError(f"Shared question universe differs for family {family}")
        universes[family] = common
        coordinate_total += 5 * expected * len(family_models)
    if coordinate_total != contract.expected_family_coordinate_count:
        raise ValueError(
            "Family-comparable coordinate census differs: "
            f"expected={contract.expected_family_coordinate_count}, found={coordinate_total}"
        )
    return universes


def _logical_units(
    selected: pd.DataFrame,
    *,
    model_id: str,
    setting: str,
    expected_questions: int,
) -> pd.DataFrame:
    expected_variants = {"one_shot_a", "one_shot_b"} if setting == "one_shot" else {setting}
    rows: list[dict[str, Any]] = []
    for question_id, group in selected.groupby("original_question_id", sort=True):
        if len(group) != len(expected_variants) or set(group["concrete_variant"]) != expected_variants:
            raise ValueError(f"Question bundle differs: {model_id}/{setting}/{question_id}")
        rows.append(
            {
                "original_question_id": str(question_id),
                "hle_correct": float(group["hle_value"].mean()),
                "closeness_score": float(group["closeness_value"].mean()),
            }
        )
    units = pd.DataFrame(rows)
    if len(units) != expected_questions:
        raise ValueError(
            f"Logical question count differs for {model_id}/{setting}: "
            f"expected={expected_questions}, found={len(units)}"
        )
    return units


def _aggregate(
    layer: pd.DataFrame,
    models: Sequence[Model],
    contract: PlotContract,
    *,
    family_universes: Mapping[str, set[str]] | None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model in models:
        question_universe = None if family_universes is None else family_universes[model.family]
        expected_questions = (
            417 if model.modality == "text_only" else 491
        ) if question_universe is None else len(question_universe)
        for setting in SETTING_ORDER:
            selected = layer.loc[
                layer["model_id"].eq(model.model_id)
                & layer["evaluation_setting"].eq(setting)
            ].copy()
            if question_universe is not None:
                selected = selected.loc[
                    selected["original_question_id"].isin(question_universe)
                ].copy()
            units = _logical_units(
                selected,
                model_id=model.model_id,
                setting=setting,
                expected_questions=expected_questions,
            )
            records.append(
                {
                    "model_id": model.model_id,
                    "evaluation_setting": setting,
                    "question_count": expected_questions,
                    "official_hle_accuracy": float(units["hle_correct"].mean()),
                    "mean_closeness_score": float(units["closeness_score"].mean()),
                    "std_closeness_score": float(units["closeness_score"].std(ddof=1)),
                }
            )
    summary = pd.DataFrame(records)
    expected_units = (
        contract.expected_global_logical_question_units
        if family_universes is None
        else contract.expected_family_logical_question_units
    )
    if (
        len(summary) != 4 * len(models)
        or not summary.groupby("model_id").size().eq(4).all()
        or int(summary["question_count"].sum()) != expected_units
    ):
        raise ValueError("Current 25-model logical summary census differs")
    return summary


def _family_spans(models: Sequence[Model]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    current = models[0].family
    for index, model in enumerate(models[1:], start=1):
        if model.family != current:
            spans.append((current, start, index - 1))
            start = index
            current = model.family
    spans.append((current, start, len(models) - 1))
    return spans


def _display_name(model: Model, contract: PlotContract) -> str:
    suffix = " (lower bound)" if model.model_id in contract.lower_bound_model_ids else ""
    return f"{model.display_name}{suffix}"


def _lower_legend_until_disjoint(
    figure: plt.Figure,
    axis: plt.Axes,
    legend: Any,
) -> float:
    anchor_y = -0.22
    anchor_floor = -0.82
    while anchor_y >= anchor_floor - 1e-9:
        legend.set_bbox_to_anchor((0.5, anchor_y), transform=axis.transAxes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        label_bboxes = [
            tick.get_window_extent(renderer=renderer)
            for tick in axis.get_xticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        if not any(label_bbox.overlaps(legend_bbox) for label_bbox in label_bboxes):
            return anchor_y
        anchor_y = round(anchor_y - 0.05, 10)
    raise ValueError("Unable to place the four-setting legend below all model labels")


def _clean_grouped_plot(
    frame: pd.DataFrame,
    models: Sequence[Model],
    contract: PlotContract,
    *,
    value_col: str,
    ylabel: str,
    percent: bool,
    y_limits: tuple[float, float],
    error_col: str | None = None,
) -> plt.Figure:
    width = 0.19
    count = len(models)
    figure_width = max(10.5, min(25.0, 0.78 * count + 6.5))
    figure, axis = plt.subplots(figsize=(figure_width, 8.4))
    x = np.arange(count, dtype=float)
    model_ids = [model.model_id for model in models]
    display_labels = [_display_name(model, contract) for model in models]
    needs_deep_bottom = count > 10 or max(map(len, display_labels)) >= 24
    work = frame.loc[frame["model_id"].isin(model_ids)].copy()
    if len(work) != 4 * count:
        raise ValueError("Plot frame does not contain the exact model-setting grid")

    if percent:
        observed_max = float(pd.to_numeric(work[value_col], errors="raise").max())
        y_limits = (y_limits[0], min(1.05, max(y_limits[1], observed_max * 1.08 + 0.025)))

    for band_index, (family, start, end) in enumerate(_family_spans(models)):
        if band_index % 2:
            axis.axvspan(
                start - 0.5,
                end + 0.5,
                color=contract.family_colors[family],
                alpha=0.045,
                zorder=0,
            )
        if start > 0:
            axis.axvline(start - 0.5, color="#D5D5D5", linewidth=0.8, zorder=0)

    ymin, ymax = y_limits
    for setting_index, setting in enumerate(SETTING_ORDER):
        offset = (setting_index - 1.5) * width
        setting_rows = (
            work.loc[work["evaluation_setting"].eq(setting)]
            .set_index("model_id")
            .reindex(model_ids)
        )
        if setting_rows[value_col].isna().any():
            raise ValueError(f"Plot values are missing for {setting}")
        for model_index, model_id in enumerate(model_ids):
            row = setting_rows.loc[model_id]
            value = float(row[value_col])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite plot value for {model_id}/{setting}")
            xpos = x[model_index] + offset
            axis.bar(
                xpos,
                value,
                width=width * 0.92,
                color=contract.setting_colors[setting],
                alpha=0.83,
                edgecolor="white",
                linewidth=0.45,
                hatch=contract.setting_hatches[setting],
                zorder=3,
            )
            if error_col is not None:
                error = float(row[error_col])
                if not math.isfinite(error) or error < 0.0:
                    raise ValueError(f"Invalid error bar for {model_id}/{setting}")
                lower = min(error, value - ymin)
                upper = min(error, ymax - value)
                axis.errorbar(
                    xpos,
                    value,
                    yerr=np.array([[max(0.0, lower)], [max(0.0, upper)]]),
                    fmt="none",
                    color="#303030",
                    capsize=2.0,
                    linewidth=0.8,
                    zorder=5,
                )

    axis.set_ylabel(ylabel)
    axis.set_xticks(x)
    axis.set_xticklabels(display_labels, rotation=38, ha="right", rotation_mode="anchor")
    for tick, model in zip(axis.get_xticklabels(), models, strict=True):
        tick.set_color(contract.family_colors[model.family])
        tick.set_fontweight("semibold")
    axis.set_xlim(-0.65, max(0.65, count - 0.35))
    axis.set_ylim(ymin, ymax)
    if percent:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", alpha=0.28, linewidth=0.7)
    axis.grid(axis="x", visible=False)
    handles = [
        Patch(
            facecolor=contract.setting_colors[setting],
            edgecolor="#444444",
            hatch=contract.setting_hatches[setting],
            label=contract.setting_labels[setting],
            alpha=0.83,
        )
        for setting in SETTING_ORDER
    ]
    legend = axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.39 if needs_deep_bottom else 0.28, top=0.97)
    _lower_legend_until_disjoint(figure, axis, legend)
    return figure


def _setup_style(contract: PlotContract) -> None:
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({**RC_PARAMS, "savefig.dpi": contract.output_dpi})


def _expected_pdf_relatives(contract: PlotContract) -> tuple[Path, ...]:
    values = [Path(f"{HLE_STEM}.pdf"), Path(f"{CLOSENESS_STEM}.pdf")]
    for family in contract.family_order:
        values.extend(
            (
                Path("families") / f"{family}_family_accuracy_by_prompt_setting.pdf",
                Path("families")
                / (
                    f"{family}_family_closeness_gemini-3.5-flash_judged_"
                    "mean_with_std_by_prompt_setting.pdf"
                ),
            )
        )
    if len(values) != 26 or len(set(values)) != 26:
        raise ValueError("Expected PDF inventory differs from the exact 26-file contract")
    return tuple(values)


def _assert_clean_figure(
    figure: plt.Figure,
    models: Sequence[Model],
    contract: PlotContract,
) -> None:
    if len(figure.axes) != 1 or figure.texts:
        raise ValueError("Figure contains extra axes or explanatory prose")
    axis = figure.axes[0]
    if any(axis.get_title(loc=location) for location in ("center", "left", "right")):
        raise ValueError("Figure contains a title or subtitle")
    legend = axis.get_legend()
    expected_legend = [contract.setting_labels[setting] for setting in SETTING_ORDER]
    if legend is None or [text.get_text() for text in legend.get_texts()] != expected_legend:
        raise ValueError("Figure legend differs from the four-setting contract")
    expected_labels = [_display_name(model, contract) for model in models]
    if [tick.get_text() for tick in axis.get_xticklabels()] != expected_labels:
        raise ValueError("Figure model labels differ from the current release")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer)
    overlaps = [
        tick.get_text()
        for tick in axis.get_xticklabels()
        if tick.get_window_extent(renderer=renderer).overlaps(legend_bbox)
    ]
    if overlaps:
        raise ValueError(f"Figure legend overlaps model labels: {overlaps}")
    all_text = "\n".join(text.get_text() for text in figure.findobj(match=plt.Text))
    forbidden = (
        "†",
        "dagger",
        "footnote",
        "subtitle",
        "audit",
        "canonical coverage",
        "missing evidence:",
        "partial denominator",
        "N/A",
    )
    if any(token.casefold() in all_text.casefold() for token in forbidden):
        raise ValueError("Figure contains forbidden explanatory prose")


def _save_pdf(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError(f"Refusing to write through a symlinked directory: {path.parent}")
    figure.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        metadata={
            "Title": None,
            "Subject": None,
            "Keywords": None,
            "Author": "Shashank Agnihotri",
            "Creator": "Shashank Agnihotri / HSLE exact 25-model renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def _render(
    output_dir: Path,
    global_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    models: Sequence[Model],
    contract: PlotContract,
) -> None:
    jobs: list[
        tuple[Path, pd.DataFrame, Sequence[Model], str, str, bool, tuple[float, float], str | None]
    ] = [
        (
            Path(f"{HLE_STEM}.pdf"),
            global_summary,
            models,
            "official_hle_accuracy",
            "Official HLE row accuracy",
            True,
            (0.0, 0.65),
            None,
        ),
        (
            Path(f"{CLOSENESS_STEM}.pdf"),
            global_summary,
            models,
            "mean_closeness_score",
            "Question-balanced mean closeness score (0–10)",
            False,
            (0.0, 10.0),
            "std_closeness_score",
        ),
    ]
    for family in contract.family_order:
        family_models = [model for model in models if model.family == family]
        jobs.extend(
            (
                (
                    Path("families") / f"{family}_family_accuracy_by_prompt_setting.pdf",
                    family_summary,
                    family_models,
                    "official_hle_accuracy",
                    "Official HLE row accuracy",
                    True,
                    (0.0, 0.65),
                    None,
                ),
                (
                    Path("families")
                    / (
                        f"{family}_family_closeness_gemini-3.5-flash_judged_"
                        "mean_with_std_by_prompt_setting.pdf"
                    ),
                    family_summary,
                    family_models,
                    "mean_closeness_score",
                    "Question-balanced mean closeness score (0–10)",
                    False,
                    (0.0, 10.0),
                    "std_closeness_score",
                ),
            )
        )
    if len(jobs) != 26:
        raise ValueError("Render job count differs from the current 26-PDF release")
    for relative, frame, selected_models, value, ylabel, percent, limits, error in jobs:
        figure = _clean_grouped_plot(
            frame,
            selected_models,
            contract,
            value_col=value,
            ylabel=ylabel,
            percent=percent,
            y_limits=limits,
            error_col=error,
        )
        try:
            _assert_clean_figure(figure, selected_models, contract)
            _save_pdf(figure, output_dir / relative)
        finally:
            plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_pdf_inventory(output_dir: Path, contract: PlotContract) -> dict[str, str]:
    expected = set(_expected_pdf_relatives(contract))
    entries = list(output_dir.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Plot output contains symlinks")
    files = {path.relative_to(output_dir) for path in entries if path.is_file()}
    directories = {path.relative_to(output_dir) for path in entries if path.is_dir()}
    if files != expected or directories != {Path("families")}:
        raise ValueError(
            "Plot output must contain exactly 26 PDFs and one families directory: "
            f"missing={sorted(map(str, expected - files))}, "
            f"extra={sorted(map(str, files - expected))}"
        )
    manifest: dict[str, str] = {}
    for relative in sorted(expected):
        path = output_dir / relative
        payload = path.read_bytes()
        if (
            not payload.startswith(b"%PDF-")
            or b"%%EOF" not in payload[-1024:]
            or len(re.findall(rb"/Type\s*/Page\b", payload)) != 1
        ):
            raise ValueError(f"PDF structure or one-page contract differs: {relative}")
        manifest[str(relative)] = _sha256(path)
    return manifest


def _require_safe_output_scope(output_dir: Path, contract: PlotContract) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"Refusing to write to a symlinked output directory: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir():
        raise ValueError(f"Plot output is not a directory: {output_dir}")
    expected = set(_expected_pdf_relatives(contract))
    entries = list(output_dir.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Existing plot output contains symlinks")
    files = {path.relative_to(output_dir) for path in entries if path.is_file()}
    directories = {path.relative_to(output_dir) for path in entries if path.is_dir()}
    if not files <= expected or not directories <= {Path("families")}:
        raise ValueError("Existing output contains files outside the exact PDF contract")


def generate(scored_paths: Sequence[Path], output_dir: Path | None = None) -> dict[str, str]:
    contract = _load_plot_contract()
    models = _load_models(contract)
    paths = [resolve_path(path) for path in scored_paths]
    layer = _load_scored_coordinates(paths, contract, models)
    family_universes = _family_question_universes(layer, models, contract)
    global_summary = _aggregate(
        layer,
        models,
        contract,
        family_universes=None,
    )
    family_summary = _aggregate(
        layer,
        models,
        contract,
        family_universes=family_universes,
    )
    target = resolve_path(output_dir or contract.output_directory)
    _require_safe_output_scope(target, contract)
    _setup_style(contract)
    _render(target, global_summary, family_summary, models, contract)
    return _validate_pdf_inventory(target, contract)


def build_parser() -> argparse.ArgumentParser:
    contract = _load_plot_contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="coordinate-level scored CSV file(s) for the exact current 25-model cohort",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=contract.output_directory,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate(args.input, args.output_dir)
    print(f"Wrote and validated {len(manifest)} PDFs in {resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()
