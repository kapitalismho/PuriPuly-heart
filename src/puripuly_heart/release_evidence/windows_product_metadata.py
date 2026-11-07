from __future__ import annotations

import tomllib
from pathlib import Path

PRODUCT_NAME = "PuriPuly <3"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def read_project_version(repo_root: Path | None = None) -> str:
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_from_here()
    pyproject_path = root / "pyproject.toml"
    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"project version source not found: {pyproject_path}") from exc
    try:
        version = payload["project"]["version"]
    except KeyError as exc:
        raise RuntimeError(f"project version missing in {pyproject_path}") from exc
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"project version is blank in {pyproject_path}")
    return version.strip()


def version_tuple(version: str) -> tuple[int, int, int, int]:
    parts = version.strip().split(".")
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"version must have 1-4 numeric parts: {version!r}")
    numbers: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"version part is not numeric: {part!r} in {version!r}")
        value = int(part)
        if not 0 <= value <= 65535:
            raise ValueError(f"version part out of range 0-65535: {part!r} in {version!r}")
        numbers.append(value)
    while len(numbers) < 4:
        numbers.append(0)
    return (numbers[0], numbers[1], numbers[2], numbers[3])
