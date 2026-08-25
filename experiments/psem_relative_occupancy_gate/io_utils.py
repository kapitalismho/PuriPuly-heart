from __future__ import annotations

import hashlib
import json
import os
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
FROZEN_DATA_RELATIVE_PATH = Path("experiments/psem_training_strategy_gate/data/v2")


class ExperimentError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"invalid JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ExperimentError(f"JSONL rows must be objects: {path}")
    return rows


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(canonical_json(dict(row)) for row in rows)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + ("\n" if payload else ""))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def config() -> dict[str, Any]:
    value = load_json(CONFIG_PATH)
    if not isinstance(value, dict):
        raise ExperimentError("config must be an object")
    return value


def data_dir() -> Path:
    configured = Path(str(config()["dataset"]["data_dir"]))
    if configured != FROZEN_DATA_RELATIVE_PATH:
        raise ExperimentError("dataset data_dir is not the pinned V2 repository path")
    expected = (REPOSITORY_ROOT / FROZEN_DATA_RELATIVE_PATH).resolve()
    if not expected.is_dir() or expected == REPOSITORY_ROOT:
        raise ExperimentError("pinned V2 data directory is unavailable")
    return expected


def required_external_root(name: str, explicit: Path | None = None) -> Path:
    raw = str(explicit) if explicit is not None else os.environ.get(name)
    if not raw:
        raise ExperimentError(f"{name} is required")
    path = Path(raw).resolve()
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path == REPOSITORY_ROOT
        or REPOSITORY_ROOT in path.parents
    ):
        raise ExperimentError(f"{name} must be an absolute path outside the repository")
    return path


def safe_child(root: Path, relative: str | Path, field: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or any(part == ".." for part in value.parts):
        raise ExperimentError(f"{field} must be a relative path inside its pinned root")
    resolved_root = root.resolve()
    path = (resolved_root / value).resolve()
    if path == resolved_root or resolved_root not in path.parents:
        raise ExperimentError(f"{field} escapes its pinned root")
    return path


def safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    frozen = data_dir()
    if resolved == frozen or frozen in resolved.parents:
        raise ExperimentError("experiment outputs cannot modify the immutable V2 dataset")
    return resolved


def corpus_root(explicit: Path | None = None) -> Path:
    return required_external_root("PSEM_CORPUS_ROOT", explicit)


def reference_root(explicit: Path | None = None) -> Path:
    return required_external_root("PSEM_REFERENCE_ROOT", explicit)


def research_root(explicit: Path | None = None) -> Path:
    return required_external_root("SRSCD_CACHE_ROOT", explicit)


def lseend_root(explicit: Path | None = None) -> Path:
    return required_external_root("PSEM_LSEEND_ROOT", explicit)


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getframerate() != 16000
            or reader.getnchannels() != 1
            or reader.getsampwidth() != 2
        ):
            raise ExperimentError(f"expected mono 16 kHz PCM16 WAV: {path}")
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").astype(
            np.float32
        )
    return samples / 32768.0


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, q))
