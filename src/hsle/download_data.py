"""Download and validate the separately hosted HSLE dataset snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from hsle.config import load_environment, resolve_path


DATA_PATTERNS = ("processed/**", "corrections/**", "images/**")
PUBLIC_DATASET_REPO_ID = "shashankskagnihotri/humanitys-second-last-exam"
PUBLIC_DATASET_REVISION = "582207e5bd95b4f4e2948887c2d613398e98a17e"
REQUIRED_PROCESSED_FILES = (
    "hsle_all_rows.csv",
    "hsle_context_examples.csv",
    "hsle_image_manifest.csv",
    "hsle_original_questions.csv",
    "hsle_question_example_links.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_path(data_dir: Path, recorded_path: str) -> Path:
    relative = Path(recorded_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe path in dataset metadata: {recorded_path!r}")
    parts = relative.parts[1:] if relative.parts[:1] == ("data",) else relative.parts
    return data_dir.joinpath(*parts)


def _image_references(frame: pd.DataFrame, column: str) -> set[str]:
    references: set[str] = set()
    for raw_value in frame[column].astype(str):
        raw = raw_value.strip()
        if not raw:
            continue
        values = (
            json.loads(raw)
            if raw.startswith("[")
            else [item for item in raw.replace(";", "|").split("|") if item.strip()]
        )
        for value in values:
            relative = Path(str(value).strip())
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image reference: {value!r}")
            parts = relative.parts[1:] if relative.parts[:1] == ("data",) else relative.parts
            references.add(Path(*parts).as_posix())
    return references


def validate_dataset(data_dir: Path) -> None:
    """Validate file hashes, row counts, and the complete image-reference set."""

    metadata_path = data_dir / "processed" / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    recorded_hashes = metadata.get("processed_artifact_sha256", {})
    for filename in REQUIRED_PROCESSED_FILES:
        path = data_dir / "processed" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed dataset file: {path}")
        expected = str(recorded_hashes.get(filename, "")).strip()
        if not expected or _sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")

    for section in ("ground_truth_corrections", "context_example_corrections"):
        payload = metadata.get(section, {})
        manifest = _bundle_path(data_dir, str(payload.get("manifest", "")))
        expected = str(payload.get("manifest_sha256", "")).strip()
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing correction manifest: {manifest}")
        if not expected or _sha256(manifest) != expected:
            raise ValueError(f"SHA-256 mismatch for {manifest}")

    originals = pd.read_csv(
        data_dir / "processed" / "hsle_original_questions.csv",
        dtype=str,
        keep_default_na=False,
    )
    contexts = pd.read_csv(
        data_dir / "processed" / "hsle_context_examples.csv",
        dtype=str,
        keep_default_na=False,
    )
    links = pd.read_csv(
        data_dir / "processed" / "hsle_question_example_links.csv",
        dtype=str,
        keep_default_na=False,
    )
    if (len(originals), len(contexts), len(links)) != (491, 982, 491):
        raise ValueError(
            "Unexpected dataset row counts: "
            f"originals={len(originals)}, contexts={len(contexts)}, links={len(links)}"
        )
    if originals["original_question_id"].nunique() != 491:
        raise ValueError("Target question IDs are not unique")
    if links["original_question_id"].nunique() != 491:
        raise ValueError("Target-to-example links are not unique")
    if set(pd.to_numeric(links["example_count"], errors="raise")) != {2}:
        raise ValueError("Every target must have exactly two linked examples")

    references = set()
    references.update(_image_references(originals, "image_paths_or_ids"))
    references.update(_image_references(contexts, "image_paths_or_ids"))
    references.update(_image_references(links, "example_1_image_paths_or_ids"))
    references.update(_image_references(links, "example_2_image_paths_or_ids"))
    images = {
        path.relative_to(data_dir).as_posix()
        for path in (data_dir / "images").iterdir()
        if path.is_file()
    }
    if references != images:
        raise ValueError(
            "Image bundle differs from the references: "
            f"missing={sorted(references - images)[:5]}, "
            f"unreferenced={sorted(images - references)[:5]}"
        )
    expected_images = int(metadata.get("bundled_image_blob_count", -1))
    if len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {len(images)}")


def download_dataset(
    *,
    repo_id: str,
    revision: str | None,
    output_dir: Path,
) -> Path:
    """Download one Hugging Face dataset revision and validate its contents."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to download the dataset.") from exc

    load_environment()
    token = os.environ.get("HF_TOKEN", "").strip() or None
    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=output_dir,
            allow_patterns=list(DATA_PATTERNS),
            token=token,
        )
    )
    validate_dataset(downloaded)
    return downloaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        help=(
            "Hugging Face dataset repository; defaults to HSLE_DATASET_REPO "
            f"or {PUBLIC_DATASET_REPO_ID}"
        ),
    )
    parser.add_argument(
        "--revision",
        help=(
            "Pinned dataset revision; defaults to HSLE_DATASET_REVISION or "
            f"{PUBLIC_DATASET_REVISION}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Local Git-ignored dataset directory (default: data)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_environment()
    repo_id = (
        args.repo_id or os.environ.get("HSLE_DATASET_REPO", "") or PUBLIC_DATASET_REPO_ID
    ).strip()
    revision = (
        args.revision or os.environ.get("HSLE_DATASET_REVISION", "") or PUBLIC_DATASET_REVISION
    ).strip()
    output_dir = resolve_path(args.output_dir)
    downloaded = download_dataset(
        repo_id=repo_id,
        revision=revision,
        output_dir=output_dir,
    )
    print(f"Downloaded and validated the HSLE dataset at {downloaded}")


if __name__ == "__main__":
    main()
