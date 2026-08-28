from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.experiment_support import (
    AnchorEpisode,
    GTSessionResult,
    ReplacementEvent,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
ACTION_REFERENCE_LEDGER_PATH = PACKAGE_ROOT / "action_reference_ledger.jsonl"
ACTION_REFERENCE_COVERAGE_PATH = PACKAGE_ROOT / "action_reference_coverage.json"
MAPPING_LEDGER_PATH = PACKAGE_ROOT / "oracle_mapping_ledger.jsonl"
MAPPING_COVERAGE_PATH = PACKAGE_ROOT / "oracle_mapping_coverage.json"
SPLIT_PATH = PACKAGE_ROOT / "split_manifest.json"
FROZEN_INPUT_ROOT = PACKAGE_ROOT / "frozen_inputs"
SOURCE_MANIFEST_PATH = FROZEN_INPUT_ROOT / "source_manifest.jsonl"
POSTERIOR_SNAPSHOT_PATH = FROZEN_INPUT_ROOT / "posterior_sessions.npz"
SOURCE_PROVENANCE_PATH = FROZEN_INPUT_ROOT / "source_evidence_provenance.json"


@dataclass(slots=True)
class SessionExamples:
    role: str
    source_id: str
    source_family: str
    confirmation_ms: int
    manifest: dict[str, Any]
    reference: GTSessionResult
    mapping_records: tuple[dict[str, Any], ...]
    episode_ids: np.ndarray
    episode_speakers: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    posterior_centers: np.ndarray
    frontiers: np.ndarray
    probabilities: np.ndarray
    alive: np.ndarray
    reset: np.ndarray
    valid: np.ndarray
    masked: np.ndarray
    speech_present: np.ndarray
    anchor_present: np.ndarray
    overlap: np.ndarray

    @property
    def target(self) -> np.ndarray:
        return np.logical_and(self.speech_present, np.logical_not(self.anchor_present))

    @property
    def weights(self) -> np.ndarray:
        return (self.ends - self.starts).astype(np.float32)

    @property
    def evidence_delay_ms(self) -> np.ndarray:
        return (self.frontiers - self.ends) * 1000.0 / 16000.0


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def _reference_event(value: dict[str, Any]) -> ReplacementEvent | None:
    event = value["reference_event"]
    return None if event is None else ReplacementEvent(**event)


def _references() -> dict[tuple[str, str], GTSessionResult]:
    cfg = config()
    rows = load_jsonl(ACTION_REFERENCE_LEDGER_PATH)
    coverage = load_json(ACTION_REFERENCE_COVERAGE_PATH)
    if coverage["ledger_sha256"] != sha256_file(ACTION_REFERENCE_LEDGER_PATH):
        raise ValueError("action reference ledger coverage seal mismatch")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for value in rows:
        grouped.setdefault((str(value["old_v2_role"]), str(value["source_id"])), []).append(
            value
        )
    result = {}
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda item: int(item["anchor_emit_sample"]))
        events = tuple(
            event for item in ordered if (event := _reference_event(item)) is not None
        )
        event_by_episode = {event.anchor_episode_id: event for event in events}
        episodes = tuple(
            AnchorEpisode(
                episode_id=str(item["anchor_episode_id"]),
                source_id=str(item["source_id"]),
                anchor_speaker=str(item["anchor_speaker"]),
                opportunity_start_sample=int(item["anchor_emit_sample"]),
                anchor_emit_sample=int(item["anchor_emit_sample"]),
                end_emit_sample=int(item["episode_end_sample"]),
                replacement_boundary_sample=(
                    event_by_episode[str(item["anchor_episode_id"])].boundary_source_sample
                    if str(item["anchor_episode_id"]) in event_by_episode
                    else None
                ),
            )
            for item in ordered
        )
        result[key] = GTSessionResult(
            source_id=key[1],
            confirmation_samples=int(cfg["action_reference_confirmation_ms"]) * 16,
            enrollment_samples=int(cfg["enrollment_ms"]) * 16,
            silence_reset_samples=int(cfg["silence_reset_ms"]) * 16,
            events=events,
            episodes=episodes,
        )
    if len(rows) != int(coverage["episode_count"]):
        raise ValueError("action reference episode coverage mismatch")
    return result


def _mappings() -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    rows = load_jsonl(MAPPING_LEDGER_PATH)
    coverage = load_json(MAPPING_COVERAGE_PATH)
    if coverage["ledger_sha256"] != sha256_file(MAPPING_LEDGER_PATH):
        raise ValueError("oracle mapping ledger coverage seal mismatch")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for value in rows:
        grouped.setdefault((str(value["old_v2_role"]), str(value["source_id"])), []).append(
            value
        )
    if len(rows) != int(coverage["episode_count"]):
        raise ValueError("oracle mapping episode coverage mismatch")
    return {key: tuple(values) for key, values in grouped.items()}


def load_sessions(
    confirmation_ms: tuple[int, ...] = (500,),
    *,
    validate_mapping_ledger: bool = True,
) -> list[SessionExamples]:
    allowed = {int(value) for value in config()["oracle_mapping_confirmation_ms"]}
    if set(map(int, confirmation_ms)) != allowed:
        raise ValueError("confirmation is outside the frozen session snapshot")
    rows = load_jsonl(SOURCE_MANIFEST_PATH)
    references = _references()
    mappings = _mappings()
    result = []
    fields = (
        "episode_ids",
        "episode_speakers",
        "starts",
        "ends",
        "posterior_centers",
        "frontiers",
        "probabilities",
        "alive",
        "reset",
        "valid",
        "masked",
        "speech_present",
        "anchor_present",
        "overlap",
    )
    with np.load(POSTERIOR_SNAPSHOT_PATH, allow_pickle=False) as frozen:
        for row in rows:
            prefix = f"s{int(row['source_index']):03d}"
            arrays = {field: frozen[f"{prefix}_{field}"].copy() for field in fields}
            key = (str(row["old_v2_role"]), str(row["source_id"]))
            manifest = {**row, "role": str(row["old_v2_role"])}
            mapping_records = mappings[key]
            if validate_mapping_ledger and len(mapping_records) != len(references[key].episodes):
                raise ValueError(f"oracle mapping source coverage mismatch: {key}")
            result.append(
                SessionExamples(
                    role=str(row["role"]),
                    source_id=str(row["source_id"]),
                    source_family=str(row["source_family"]),
                    confirmation_ms=500,
                    manifest=manifest,
                    reference=references[key],
                    mapping_records=mapping_records,
                    **arrays,
                )
            )
    return result


def freeze_action_reference_ledger() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(ACTION_REFERENCE_LEDGER_PATH)
    _references()
    return rows, load_json(ACTION_REFERENCE_COVERAGE_PATH)


def freeze_oracle_mapping_ledger() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(MAPPING_LEDGER_PATH)
    _mappings()
    return rows, load_json(MAPPING_COVERAGE_PATH)


def inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = config()
    split = load_json(SPLIT_PATH)
    rows = load_jsonl(SOURCE_MANIFEST_PATH)
    if {str(value["source_id"]) for value in rows} != {
        str(value["source_id"]) for value in split["sources"]
    }:
        raise ValueError("frozen source snapshot differs from split manifest")
    provenance = load_json(SOURCE_PROVENANCE_PATH)
    artifact_paths = {
        "source_manifest": SOURCE_MANIFEST_PATH,
        "posterior_sessions": POSTERIOR_SNAPSHOT_PATH,
        "source_evidence_provenance": SOURCE_PROVENANCE_PATH,
        "dev_sortformer_receipt": FROZEN_INPUT_ROOT / "dev_sortformer_model_receipt.json",
        "eval_sortformer_receipt": FROZEN_INPUT_ROOT / "eval_sortformer_model_receipt.json",
        "dev_vad_gate": FROZEN_INPUT_ROOT / "dev_production_vad_speech_gate.jsonl",
        "dev_vad_receipt": FROZEN_INPUT_ROOT / "dev_production_vad_replay_receipt.json",
        "eval_vad_gate": FROZEN_INPUT_ROOT / "eval_production_vad_speech_gate.jsonl",
        "eval_vad_receipt": FROZEN_INPUT_ROOT / "eval_production_vad_replay_receipt.json",
        "issue98_vad_reference": FROZEN_INPUT_ROOT / "issue98_vad_reference.json",
        "action_reference_ledger": ACTION_REFERENCE_LEDGER_PATH,
        "action_reference_coverage": ACTION_REFERENCE_COVERAGE_PATH,
        "oracle_mapping_ledger": MAPPING_LEDGER_PATH,
        "oracle_mapping_coverage": MAPPING_COVERAGE_PATH,
        "split_manifest": SPLIT_PATH,
        "feature_contract": PACKAGE_ROOT / "posterior_features.py",
        "experiment_support": PACKAGE_ROOT / "experiment_support.py",
        "experiment_config": CONFIG_PATH,
    }
    receipt = {
        "schema_version": "psem.frozen_ceiling.evidence_reuse_receipt.v2",
        "authority": cfg["authority"],
        "repository_baseline_sha": cfg["repository_baseline"],
        "dirty_state_at_issue99_start": [],
        "artifacts": {
            key: {
                "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for key, path in artifact_paths.items()
        },
        "upstream_evidence": {
            "repository_baseline_sha": provenance["repository_baseline_sha"],
            "artifact_digests": {
                key: value["sha256"] for key, value in provenance["artifacts"].items()
            },
        },
        "sortformer": provenance["sortformer"],
        "new_sortformer_inference_required": False,
        "new_sortformer_inference_performed": False,
    }
    write_json(PACKAGE_ROOT / "evidence_reuse_receipt.json", receipt)
    return split, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--freeze-reference", action="store_true")
    parser.add_argument("--freeze-mappings", action="store_true")
    args = parser.parse_args()
    if args.freeze_reference or args.freeze_mappings:
        reference_records, reference_coverage = freeze_action_reference_ledger()
        print(
            json.dumps(
                {
                    "action_reference_records": len(reference_records),
                    "action_reference_events": reference_coverage["reference_event_count"],
                }
            )
        )
    if args.freeze_mappings:
        records, coverage = freeze_oracle_mapping_ledger()
        print(
            json.dumps(
                {
                    "mapping_records": len(records),
                    "mapped_episode_fraction": coverage["mapped_episode_fraction"],
                }
            )
        )
    split, receipt = inventory()
    print(json.dumps({"split": split["schema_version"], "receipt": receipt["schema_version"]}))
    if not args.inventory_only:
        sessions = load_sessions((500,))
        print(
            json.dumps(
                {
                    "session_count": len(sessions),
                    "frame_count": sum(len(value.starts) for value in sessions),
                }
            )
        )


if __name__ == "__main__":
    main()
