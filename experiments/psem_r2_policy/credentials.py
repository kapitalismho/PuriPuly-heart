from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ORIGINAL_ENV_LOCAL = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls/.env.local"
)
_SECRET_NAMES = ("DEEPGRAM_API_KEY", "OPENROUTER_API_KEY")


def _dotenv_values(path: Path) -> dict[str, str | None]:
    try:
        from dotenv import dotenv_values
    except ImportError:
        values: dict[str, str | None] = {}
        if not path.is_file():
            return values
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            values[key] = value
        return values
    return dict(dotenv_values(path))


def load_runtime_secrets() -> dict[str, str]:
    loaded = _dotenv_values(ORIGINAL_ENV_LOCAL) if ORIGINAL_ENV_LOCAL.is_file() else {}
    resolved: dict[str, str] = {}
    for name in _SECRET_NAMES:
        env_value = os.environ.get(name)
        file_value = loaded.get(name)
        chosen = env_value if env_value not in (None, "") else file_value
        resolved[name] = "" if chosen is None else str(chosen).strip()
    return resolved


def credential_presence(secrets: dict[str, str] | None = None) -> dict[str, bool]:
    payload = secrets if secrets is not None else load_runtime_secrets()
    return {name: bool(payload.get(name)) for name in _SECRET_NAMES}


def presence_from_mapping(values: dict[str, Any]) -> dict[str, bool]:
    return {name: bool(str(values.get(name) or "").strip()) for name in _SECRET_NAMES}
