from __future__ import annotations

import dataclasses
import enum
import importlib
import json
from pathlib import Path
from uuid import UUID


def encode(value):
    if dataclasses.is_dataclass(value):
        return {
            "$type": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.init and not callable(getattr(value, f.name))
            },
        }
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return {"$tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if type(value).__module__ == "numpy" and type(value).__name__ == "ndarray":
        return {"samples": int(value.size)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$uuid" in value:
        return UUID(value["$uuid"])
    if "$tuple" in value:
        return tuple(decode(item) for item in value["$tuple"])
    if "$type" in value:
        module, name = value["$type"].split(":")
        if not module.startswith("puripuly_heart."):
            raise ValueError(f"Unexpected record type: {value['$type']}")
        cls = getattr(importlib.import_module(module), name)
        fields = {f.name for f in dataclasses.fields(cls) if f.init}
        return cls(**{key: decode(item) for key, item in value["fields"].items() if key in fields})
    return {key: decode(item) for key, item in value.items()}


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$type" in value:
        return {key: plain(item) for key, item in value["fields"].items()}
    if "$tuple" in value:
        return [plain(item) for item in value["$tuple"]]
    if "$uuid" in value:
        return value["$uuid"]
    return {key: plain(item) for key, item in value.items()}


def load(path):
    return plain(json.loads(Path(path).read_text(encoding="utf-8")))


def timing(values):
    if not values:
        return {"n": 0, "median_s": None, "p95_s": None}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"n": len(values), "median_s": percentile(0.5), "p95_s": percentile(0.95)}
