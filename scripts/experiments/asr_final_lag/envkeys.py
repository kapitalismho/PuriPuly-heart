from __future__ import annotations

import os
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name] = value
    return values


def resolve(
    candidates: dict[str, tuple[str, ...]], env_files: list[Path], overrides: dict[str, str]
) -> dict[str, tuple[str, str]]:
    merged: dict[str, str] = {}
    for path in env_files:
        merged.update(read_env_file(path))
    merged.update({k: v for k, v in os.environ.items() if k not in merged})
    resolved: dict[str, tuple[str, str]] = {}
    for name, variables in candidates.items():
        for variable in (overrides[name],) if name in overrides else variables:
            if merged.get(variable):
                resolved[name] = (variable, merged[variable])
                break
    return resolved
