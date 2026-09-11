from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.psem_r2_policy.metrics import (
    DEV_CLUSTERS,
    HOLDOUT_CLUSTERS,
    aggregate_cluster_parents,
    confirmatory_decision,
    latency_by_operation,
)

EXP = Path(__file__).resolve().parent
PROTOCOL_PATH = EXP / "PROTOCOL.json"
GATE_PATH = EXP / "HOLD_OUT_GATE.json"
PIN_PATH = EXP / "PIN_MANIFEST.json"
ARTIFACTS = EXP / "artifacts"
PIN_TARGETS = (
    "PROTOCOL.json",
    "metrics.py",
    "live_runner.py",
    "arms.py",
    "budget.py",
    "rates.json",
    "BILLING_BOUNDS.json",
    "HOLD_OUT_GATE.json",
)


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_holdout_gate() -> dict[str, Any]:
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def declared_meetings(phase: str) -> tuple[str, ...]:
    protocol = load_protocol()
    if phase == "dev":
        return tuple(protocol["development"]["meetings"])
    if phase == "holdout":
        return tuple(protocol["holdout"]["meetings"])
    raise ValueError(f"unknown phase: {phase}")


def resolve_meetings(phase: str, meeting: str | None) -> tuple[str, ...]:
    declared = declared_meetings(phase)
    if meeting in {None, "", "all"}:
        return declared
    if meeting not in declared:
        raise ValueError(f"meeting {meeting} is not declared for {phase}")
    return (meeting,)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_holdout_inputs() -> tuple[dict[str, str | None], dict[str, str | None], list[str]]:
    from experiments.psem_r2_policy.live_runner import ami_wav_path
    from experiments.psem_r2_policy.metrics import AMI_WORDS, ROLES

    protocol = load_protocol()
    audio: dict[str, str | None] = {}
    annotations: dict[str, str | None] = {}
    missing: list[str] = []
    for meeting in protocol["holdout"]["meetings"]:
        try:
            wav = ami_wav_path(meeting)
        except FileNotFoundError:
            audio[meeting] = None
            missing.append(f"audio:{meeting}")
        else:
            audio[meeting] = _sha256_file(wav)
        for role in ROLES:
            path = AMI_WORDS / f"{meeting}.{role}.words.xml"
            key = f"{meeting}.{role}"
            if path.is_file():
                annotations[key] = _sha256_file(path)
            else:
                annotations[key] = None
                missing.append(f"words:{key}")
    return audio, annotations, missing


def build_pin_manifest() -> dict[str, Any]:
    files = {name: _sha256_file(EXP / name) for name in PIN_TARGETS if (EXP / name).is_file()}
    audio, annotations, missing = _hash_holdout_inputs()
    return {
        "files": files,
        "audio": audio,
        "annotations": annotations,
        "missing_inputs": missing,
        "speaker_clusters": {"dev": DEV_CLUSTERS, "holdout": HOLDOUT_CLUSTERS},
        "metric_revision": "R2-POLICY-DIRECTOR-2",
    }


def write_pin_manifest(path: Path | None = None) -> dict[str, Any]:
    payload = build_pin_manifest()
    target = path or PIN_PATH
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def holdout_unlock_error() -> str | None:
    gate = load_holdout_gate()
    if not gate.get("frozen"):
        return "holdout is locked until Director freeze"
    if not PIN_PATH.is_file():
        return "holdout pin manifest is missing"
    pinned = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    current = build_pin_manifest()
    if current.get("missing_inputs"):
        return "holdout pin is missing audio or annotation files"
    if pinned.get("files") != current["files"]:
        return "holdout pin hashes do not match current code/config/metrics"
    if pinned.get("speaker_clusters") != current["speaker_clusters"]:
        return "holdout pin speaker clusters do not match"
    if pinned.get("audio") != current["audio"]:
        return "holdout pin audio hashes do not match"
    if pinned.get("annotations") != current["annotations"]:
        return "holdout pin annotation hashes do not match"
    return None


def case_output_path(phase: str, meeting: str, *, stamp: str | None = None) -> Path:
    token = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ARTIFACTS / phase / meeting
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{token}.json"
    if path.exists():
        raise FileExistsError(path)
    return path


def write_case_output(phase: str, meeting: str, payload: Mapping[str, Any]) -> dict[str, str]:
    path = case_output_path(phase, meeting)
    encoded = json.dumps(payload, indent=1, ensure_ascii=False)
    path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {"path": str(path), "sha256": digest}


def aggregate_phase(
    parents: Sequence[Mapping[str, Any]],
    *,
    marks: Sequence[Mapping[str, float | None]] | None = None,
) -> dict[str, Any]:
    clustered = aggregate_cluster_parents(parents)
    decision = confirmatory_decision(
        cluster_rows=clustered["cluster_rows"],
        coverage=clustered.get("coverage"),
    )
    return {
        "cluster_aggregate": clustered,
        "confirmatory": decision,
        "latency_by_operation": latency_by_operation(marks or ()),
        "n_parents": len(parents),
        "incomplete_preserved": True,
    }
