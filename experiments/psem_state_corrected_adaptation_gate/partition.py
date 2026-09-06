from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


CALIB_SALT = "issue-121-train-calib-v1"
CALIB_MIN_FRAC = 0.10
CALIB_MAX_FRAC = 0.15


class PartitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SourceExposure:
    source_id: str
    corpus: str
    exposure: float
    positive_frames: int = 0
    negative_frames: int = 0


def _component_key(source_id: str, components: Mapping[str, str] | None) -> str:
    if components is not None and source_id in components:
        return components[source_id]
    return source_id


def _hash_rank(salt: str, corpus: str, component: str) -> str:
    return hashlib.sha256(f"{salt}|{corpus}|{component}".encode("utf-8")).hexdigest()


def assign_train_calib(
    sources: Sequence[SourceExposure],
    components: Mapping[str, str] | None = None,
    salt: str = CALIB_SALT,
    target_frac: float = 0.12,
) -> dict[str, Any]:
    from typing import Any as _Any

    listed = list(sources)
    if not listed:
        raise PartitionError("no sources to partition")
    if not (CALIB_MIN_FRAC <= target_frac <= CALIB_MAX_FRAC):
        raise PartitionError("target fraction outside 10-15%")
    corpora = sorted({s.corpus for s in listed})
    if len(corpora) < 2:
        raise PartitionError("AMI and AliMeeting must both be represented")
    comp_exposure: dict[str, dict[str, _Any]] = {}
    for source in listed:
        component = _component_key(source.source_id, components)
        key = f"{source.corpus}|{component}"
        entry = comp_exposure.setdefault(
            key, {"corpus": source.corpus, "component": component, "exposure": 0.0, "sources": []}
        )
        entry["exposure"] += source.exposure
        entry["sources"].append(source.source_id)
    calib_components: set[str] = set()
    for corpus in corpora:
        grouped = sorted(
            (entry for entry in comp_exposure.values() if entry["corpus"] == corpus),
            key=lambda e: _hash_rank(salt, corpus, str(e["component"])),
        )
        total = sum(e["exposure"] for e in grouped)
        if total <= 0:
            raise PartitionError(f"non-positive exposure for corpus {corpus}")
        accumulated = 0.0
        taken: list[dict[str, object]] = []
        for entry in grouped:
            if accumulated / total >= target_frac:
                break
            if (accumulated + float(entry["exposure"])) / total <= CALIB_MAX_FRAC:
                taken.append(entry)
                accumulated += float(entry["exposure"])
        for entry in taken:
            calib_components.add(f"{corpus}|{entry['component']}")
        frac = accumulated / total
        if not (CALIB_MIN_FRAC <= frac <= CALIB_MAX_FRAC):
            raise PartitionError(f"TRAIN-CALIB fraction {frac:.4f} outside 10-15% for {corpus}")
    fit: list[str] = []
    calib: list[str] = []
    for source in listed:
        key = f"{source.corpus}|{_component_key(source.source_id, components)}"
        (calib if key in calib_components else fit).append(source.source_id)
    return {"fit": sorted(fit), "calib": sorted(calib), "salt": salt, "target_frac": target_frac}


def validate_partition_support(
    assignment: Mapping[str, Sequence[str]],
    by_source: Mapping[str, SourceExposure],
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for split in ("fit", "calib"):
        members = [by_source[s] for s in assignment[split]]
        result[f"{split}_positive_support"] = any(m.positive_frames > 0 for m in members)
        result[f"{split}_negative_support"] = any(m.negative_frames > 0 for m in members)
    if set(assignment["fit"]) & set(assignment["calib"]):
        raise PartitionError("TRAIN-FIT and TRAIN-CALIB must be disjoint")
    missing = [k for k, v in result.items() if not v]
    if missing:
        raise PartitionError(f"partition lacks support: {sorted(missing)}")
    return result
