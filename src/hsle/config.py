"""Portable project configuration and credential loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return this checkout's root, optionally overridden for automation."""

    override = os.environ.get("HSLE_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    """Resolve a user path relative to the checkout, never a machine-specific root."""

    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else project_root() / candidate


@lru_cache(maxsize=1)
def load_environment() -> None:
    """Load a simple ignored .env file without overriding exported variables."""

    path = project_root() / ".env"
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env assignment at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "A").isalnum():
            raise ValueError(f"Invalid .env variable name at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_key(variable: str) -> str:
    load_environment()
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(
            f"{variable} is not set. Copy .env.example to .env and set it there, "
            "or export it in the shell."
        )
    return value


def load_yaml(relative_path: str | Path) -> dict[str, Any]:
    path = resolve_path(relative_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload
