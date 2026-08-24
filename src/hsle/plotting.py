"""Create the 28 HLE-accuracy and closeness PDFs from local metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

from hsle.config import load_yaml, resolve_path


_EXPECTED_SETTING_ORDER = (
    "zero_shot",
    "one_shot",
    "two_shot",
    "learning_from_experience",
)
_EXPECTED_FAMILY_ORDER = (
    "claude",
    "gemini",
    "gpt",
    "deepseek",
    "gemma",
    "internlm",
    "kimi",
    "llama",
    "llava",
    "minimax",
    "mistral",
    "nemotron",
    "qwen",
)


@dataclass(frozen=True)
class PlotContract:
    setting_order: tuple[str, ...]
    setting_labels: dict[str, str]
    setting_hatches: dict[str, str]
    family_order: tuple[str, ...]
    family_colors: dict[str, str]
    output_directory: Path
    output_format: str
    output_dpi: int
    show_titles: bool


@dataclass(frozen=True)
class ModelRegistry:
    order: dict[str, int]
    display_names: dict[str, str]
    families: dict[str, str]
    modalities: dict[str, str]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


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
            raise ValueError(f"{label}.{key} must be a non-empty string")
        result[key] = item
    return result


def _load_plot_contract() -> PlotContract:
    payload = load_yaml("configs/plotting.yaml")
    settings = _require_mapping(payload.get("settings"), "plotting.settings")
    setting_order_value = settings.get("order")
    if not isinstance(setting_order_value, list) or not all(
        isinstance(item, str) for item in setting_order_value
    ):
        raise ValueError("plotting.settings.order must be a list of strings")
    setting_order = tuple(setting_order_value)
    if setting_order != _EXPECTED_SETTING_ORDER:
        raise ValueError(
            "plotting.settings.order differs from the four-setting publication contract: "
            f"expected={list(_EXPECTED_SETTING_ORDER)}, found={list(setting_order)}"
        )
    setting_labels = _require_string_mapping(
        settings.get("labels"),
        "plotting.settings.labels",
        setting_order,
    )
    setting_hatches = _require_string_mapping(
        settings.get("hatches"),
        "plotting.settings.hatches",
        setting_order,
        allow_empty=True,
    )

    family_order_value = payload.get("family_order")
    if not isinstance(family_order_value, list) or not all(
        isinstance(item, str) for item in family_order_value
    ):
        raise ValueError("plotting.family_order must be a list of strings")
    family_order = tuple(family_order_value)
    if family_order != _EXPECTED_FAMILY_ORDER:
        raise ValueError(
            "plotting.family_order differs from the 13-family publication contract: "
            f"expected={list(_EXPECTED_FAMILY_ORDER)}, found={list(family_order)}"
        )
    family_colors = _require_string_mapping(
        payload.get("family_colors"),
        "plotting.family_colors",
        family_order,
    )
    invalid_colors = [
        family for family, color in family_colors.items() if not mcolors.is_color_like(color)
    ]
    if invalid_colors:
        raise ValueError(f"Invalid family colors for: {invalid_colors}")

    output = _require_mapping(payload.get("output"), "plotting.output")
    directory = output.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("plotting.output.directory must be a non-empty string")
    output_format = output.get("format")
    if output_format != "pdf":
        raise ValueError("plotting.output.format must be 'pdf' for the 28-PDF contract")
    output_dpi = output.get("dpi")
    if isinstance(output_dpi, bool) or not isinstance(output_dpi, int) or output_dpi <= 0:
        raise ValueError("plotting.output.dpi must be a positive integer")
    show_titles = output.get("titles")
    if not isinstance(show_titles, bool):
        raise ValueError("plotting.output.titles must be a boolean")
    return PlotContract(
        setting_order=setting_order,
        setting_labels=setting_labels,
        setting_hatches=setting_hatches,
        family_order=family_order,
        family_colors=family_colors,
        output_directory=Path(directory),
        output_format=output_format,
        output_dpi=output_dpi,
        show_titles=show_titles,
    )


def _load_model_registry(contract: PlotContract) -> ModelRegistry:
    rows = load_yaml("configs/models.yaml").get("models")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("configs/models.yaml must contain exactly 32 model rows")
    order: dict[str, int] = {}
    display_names: dict[str, str] = {}
    families: dict[str, str] = {}
    modalities: dict[str, str] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"models[{index}]")
        values: dict[str, str] = {}
        for field in ("model_id", "display_name", "family", "modality"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"models[{index}].{field} must be a non-empty string")
            values[field] = value
        model_id = values["model_id"]
        if model_id in order:
            raise ValueError(f"Duplicate model_id in configs/models.yaml: {model_id}")
        if values["modality"] not in {"multimodal", "text_only"}:
            raise ValueError(f"Unsupported modality for {model_id}: {values['modality']}")
        order[model_id] = index
        display_names[model_id] = values["display_name"]
        families[model_id] = values["family"]
        modalities[model_id] = values["modality"]
    observed_family_order = tuple(dict.fromkeys(families.values()))
    if observed_family_order != contract.family_order:
        raise ValueError(
            "configs/models.yaml family blocks differ from plotting.family_order: "
            f"expected={list(contract.family_order)}, "
            f"found={list(observed_family_order)}"
        )
    return ModelRegistry(
        order=order,
        display_names=display_names,
        families=families,
        modalities=modalities,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _setup_style(output_dpi: int) -> None:
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": output_dpi,
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.0,
        }
    )


def _lighten(color: str, amount: float) -> str:
    rgb = mcolors.to_rgb(color)
    return mcolors.to_hex(tuple((1 - amount) * component + amount for component in rgb))


def _model_colors(
    frame: pd.DataFrame,
    registry: ModelRegistry,
    contract: PlotContract,
) -> dict[str, str]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in frame[["model_family", "model_id"]].drop_duplicates().itertuples(index=False):
        by_family[row.model_family].append(row.model_id)
    colors: dict[str, str] = {}
    for family, models in by_family.items():
        models.sort(key=lambda model: registry.order[model])
        base = contract.family_colors[family]
        for index, model in enumerate(models):
            amount = 0.45 if len(models) == 1 else 0.5 * (1 - index / (len(models) - 1))
            colors[model] = _lighten(base, amount)
    return colors


def _bar_label(value_column: str, value: float) -> str:
    if value_column == "accuracy":
        percentage = (Decimal(str(value)) * Decimal(100)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        return f"{percentage:.1f}%"
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}"


def _axis_upper(value_column: str, maximum: float, label_offset: float) -> float:
    if value_column == "accuracy":
        return min(1.0, max(0.05, maximum + 4 * label_offset) * 1.12)
    return max(10.0, maximum + 4 * label_offset)


def _plot(
    frame: pd.DataFrame,
    *,
    value_column: str,
    error_column: str | None,
    ylabel: str,
    path: Path,
    registry: ModelRegistry,
    contract: PlotContract,
) -> None:
    models = sorted(frame["model_id"].unique(), key=lambda value: registry.order[value])
    expected = {(model, setting) for model in models for setting in contract.setting_order}
    observed = set(
        map(
            tuple,
            frame[["model_id", "evaluation_setting"]].itertuples(index=False, name=None),
        )
    )
    if observed != expected:
        raise ValueError(
            f"{path.name} lacks an exact model-setting grid; "
            f"missing={sorted(expected - observed)[:5]}"
        )
    colors = _model_colors(frame, registry, contract)
    x_positions = range(len(models))
    width = 0.20
    figure_width = max(13.5, len(models) * 2.0)
    figure, axis = plt.subplots(figsize=(figure_width, 10.0))
    plot_data: list[tuple[str, float, list[float], list[float], list[str]]] = []
    maximum = 0.0
    for offset_index, setting in enumerate(contract.setting_order):
        offset = (offset_index - (len(contract.setting_order) - 1) / 2) * width
        heights: list[float] = []
        errors: list[float] = []
        bar_colors: list[str] = []
        for model in models:
            row = frame[frame["model_id"].eq(model) & frame["evaluation_setting"].eq(setting)]
            height = float(row[value_column].iloc[0])
            error = float(row[error_column].iloc[0]) if error_column else 0.0
            if not math.isfinite(height) or not math.isfinite(error) or error < 0:
                raise ValueError(f"Invalid plotted value for {model}/{setting}")
            heights.append(height)
            errors.append(error)
            bar_colors.append(colors[model])
        maximum = max(
            maximum,
            *(height + error for height, error in zip(heights, errors)),
        )
        plot_data.append((setting, offset, heights, errors, bar_colors))
    label_offset = max(
        maximum * (0.035 if value_column == "accuracy" else 0.020),
        0.003 if value_column == "accuracy" else 0.08,
    )
    upper = _axis_upper(value_column, maximum, label_offset)
    for setting, offset, heights, errors, bar_colors in plot_data:
        bars = axis.bar(
            [x + offset for x in x_positions],
            heights,
            width=width,
            label=contract.setting_labels[setting],
            color=bar_colors,
            edgecolor="#222222",
            linewidth=0.5,
            hatch=contract.setting_hatches[setting],
            yerr=errors if error_column else None,
            capsize=2 if error_column else 0,
            error_kw={
                "elinewidth": 0.8,
                "ecolor": "#222222",
                "alpha": 0.75,
            },
        )
        for bar, height, error in zip(bars, heights, errors):
            if height > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(
                        height + (error if error_column else 0.0) + label_offset,
                        upper * 0.985,
                    ),
                    _bar_label(value_column, height),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    rotation=90,
                )
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(
        [registry.display_names[model] for model in models],
        rotation=35,
        ha="right",
    )
    axis.set_ylim(0, upper)
    if value_column == "accuracy":
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.margins(x=0.01)
    if contract.show_titles:
        axis.set_title(f"{ylabel} by Prompt Setting")
    figure.subplots_adjust(top=0.86, bottom=0.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format=contract.output_format,
        dpi=contract.output_dpi,
        metadata={
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _load(
    metrics_path: Path,
    std_path: Path,
    contract: PlotContract,
) -> tuple[pd.DataFrame, ModelRegistry]:
    metrics = pd.read_csv(metrics_path)
    std = pd.read_csv(std_path)
    required = {
        "model_id",
        "model_family",
        "model_modality",
        "evaluation_setting",
        "hle_accuracy_pct",
        "closeness_mean",
        "eligible_questions",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Metric table lacks columns: {missing}")
    std_required = {
        "model_id",
        "model_family",
        "evaluation_setting",
        "closeness_std",
        "eligible_questions",
    }
    std_missing = sorted(std_required - set(std.columns))
    if std_missing:
        raise ValueError(f"Closeness-STD table lacks columns: {std_missing}")
    if metrics.duplicated(["model_id", "evaluation_setting"]).any():
        raise ValueError("Metric table contains duplicate model-setting rows")
    if std.duplicated(["model_id", "evaluation_setting"]).any():
        raise ValueError("Closeness-STD table contains duplicate model-setting rows")
    merged = metrics.merge(
        std,
        on=["model_id", "model_family", "evaluation_setting"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_std"),
    )
    if (
        merged["closeness_std"].isna().any()
        or not merged["eligible_questions"].eq(merged["eligible_questions_std"]).all()
    ):
        raise ValueError("Metric and closeness-STD tables do not align")
    registry = _load_model_registry(contract)
    if set(merged["model_id"]) != set(registry.order):
        raise ValueError("Metric model scope differs from configs/models.yaml")
    if set(merged["model_family"]) != set(contract.family_order):
        raise ValueError("Metric family scope differs from the color contract")
    if set(merged["evaluation_setting"]) != set(contract.setting_order):
        raise ValueError("Metric settings differ from the four-setting contract")
    if len(merged) != len(registry.order) * len(contract.setting_order):
        raise ValueError("Metric table does not contain the complete grid")
    for row in merged.itertuples(index=False):
        model_id = str(row.model_id)
        if row.model_family != registry.families[model_id]:
            raise ValueError(f"Model family differs from configs/models.yaml for {model_id}")
        if row.model_modality != registry.modalities[model_id]:
            raise ValueError(f"Model modality differs from configs/models.yaml for {model_id}")
        expected = 491 if registry.modalities[model_id] == "multimodal" else 417
        if int(row.eligible_questions) != expected:
            raise ValueError(f"Question denominator differs for {model_id}")
    merged["accuracy"] = pd.to_numeric(merged["hle_accuracy_pct"], errors="raise") / 100.0
    merged["mean_closeness_score"] = pd.to_numeric(merged["closeness_mean"], errors="raise")
    merged["std_closeness_score"] = pd.to_numeric(merged["closeness_std"], errors="raise")
    return merged, registry


def generate(
    metrics_path: Path,
    std_path: Path,
    output_dir: Path | None = None,
) -> dict[str, str]:
    contract = _load_plot_contract()
    if output_dir is None:
        output_dir = resolve_path(contract.output_directory)
    frame, registry = _load(metrics_path, std_path, contract)
    _setup_style(contract.output_dpi)
    expected_paths = {
        output_dir / "hle_eval_accuracy_all_models_by_prompt_setting.pdf",
        output_dir / "closeness_score_all_models_by_prompt_setting.pdf",
    }
    for family in contract.family_order:
        expected_paths.update(
            {
                output_dir / "families" / f"{family}_hle_eval_accuracy_by_prompt_setting.pdf",
                output_dir / "families" / f"{family}_closeness_score_by_prompt_setting.pdf",
            }
        )
    for path in expected_paths:
        if path.exists() and not path.is_file():
            raise ValueError(f"Refusing to replace non-file output: {path}")
    _plot(
        frame,
        value_column="accuracy",
        error_column=None,
        ylabel="HLE Eval Accuracy",
        path=output_dir / "hle_eval_accuracy_all_models_by_prompt_setting.pdf",
        registry=registry,
        contract=contract,
    )
    _plot(
        frame,
        value_column="mean_closeness_score",
        error_column="std_closeness_score",
        ylabel="Mean Closeness Score",
        path=output_dir / "closeness_score_all_models_by_prompt_setting.pdf",
        registry=registry,
        contract=contract,
    )
    for family in contract.family_order:
        subset = frame.loc[frame["model_family"].eq(family)].copy()
        _plot(
            subset,
            value_column="accuracy",
            error_column=None,
            ylabel="HLE Eval Accuracy",
            path=output_dir / "families" / f"{family}_hle_eval_accuracy_by_prompt_setting.pdf",
            registry=registry,
            contract=contract,
        )
        _plot(
            subset,
            value_column="mean_closeness_score",
            error_column="std_closeness_score",
            ylabel="Mean Closeness Score",
            path=output_dir / "families" / f"{family}_closeness_score_by_prompt_setting.pdf",
            registry=registry,
            contract=contract,
        )
    actual_paths = set(output_dir.rglob("*.pdf"))
    if actual_paths != expected_paths:
        raise ValueError(
            "Output PDF set differs from the 28-file contract: "
            f"extra={sorted(map(str, actual_paths - expected_paths))}, "
            f"missing={sorted(map(str, expected_paths - actual_paths))}"
        )
    manifest = {str(path.relative_to(output_dir)): _sha256(path) for path in sorted(actual_paths)}
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    contract = _load_plot_contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/metrics/model_setting_metrics.csv"),
    )
    parser.add_argument(
        "--closeness-std",
        type=Path,
        default=Path("outputs/metrics/model_setting_closeness_std.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=contract.output_directory,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = resolve_path(args.output_dir)
    manifest = generate(
        resolve_path(args.metrics),
        resolve_path(args.closeness_std),
        output,
    )
    print(f"Wrote {len(manifest)} PDFs to {output}")


if __name__ == "__main__":
    main()
