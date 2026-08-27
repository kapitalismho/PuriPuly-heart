from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_ontology_simplification_gate.evaluate_simplified_ontologies import (
    _load_role_inputs,
    _posterior_inputs,
    _trace_by_source,
)
from experiments.psem_relative_occupancy_gate.decoder import ReplacementEvent
from experiments.psem_relative_occupancy_gate.io_utils import (
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.psem_relative_occupancy_gate.model_decode import oracle_anchor_mapping
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    gt_reference_session,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
PREDECESSOR_ROOT = REPOSITORY_ROOT / "experiments" / "psem_relative_occupancy_gate"
SIMPLIFICATION_ROOT = REPOSITORY_ROOT / "experiments" / "psem_ontology_simplification_gate"
CONFIG_PATH = PACKAGE_ROOT / "config.json"
ACTION_REFERENCE_LEDGER_PATH = PACKAGE_ROOT / "action_reference_ledger.jsonl"
ACTION_REFERENCE_COVERAGE_PATH = PACKAGE_ROOT / "action_reference_coverage.json"
MAPPING_LEDGER_PATH = PACKAGE_ROOT / "oracle_mapping_ledger.jsonl"
MAPPING_COVERAGE_PATH = PACKAGE_ROOT / "oracle_mapping_coverage.json"


@dataclass(slots=True)
class SessionExamples:
    role: str
    source_id: str
    source_family: str
    confirmation_ms: int
    manifest: dict[str, Any]
    reference: Any
    mapping_records: tuple[dict[str, Any], ...]
    episode_ids: np.ndarray
    episode_speakers: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
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


def source_family(row: dict[str, Any]) -> str:
    audio_ref = str(row["audio_ref"]).replace("\\", "/").lower()
    if audio_ref.startswith("alimeeting/far_ch0/"):
        return "alimeeting_far_ch0"
    if audio_ref.startswith("ami/audio/") and audio_ref.endswith(".mix-headset.wav"):
        return "ami_mix_headset"
    raise ValueError(f"unrecognized corpus-device family: {audio_ref}")


def _mapping_key(value: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(value["old_v2_role"]),
        str(value["source_id"]),
        int(value["confirmation_ms"]),
        str(value["anchor_episode_id"]),
    )


def _mapping_index() -> dict[tuple[str, str, int, str], dict[str, Any]]:
    rows = load_jsonl(MAPPING_LEDGER_PATH)
    result = {_mapping_key(value): value for value in rows}
    if len(result) != len(rows):
        raise ValueError("oracle mapping ledger has duplicate identities")
    return result


def _reference_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value["old_v2_role"]),
        str(value["source_id"]),
        str(value["anchor_episode_id"]),
    )


def _reference_index() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = load_jsonl(ACTION_REFERENCE_LEDGER_PATH)
    coverage = load_json(ACTION_REFERENCE_COVERAGE_PATH)
    digest = sha256_file(ACTION_REFERENCE_LEDGER_PATH)
    if coverage["ledger_sha256"] != digest or int(coverage["episode_count"]) != len(rows):
        raise ValueError("action reference ledger coverage seal mismatch")
    if int(coverage["reference_event_count"]) != sum(
        value["reference_event"] is not None for value in rows
    ):
        raise ValueError("action reference event count mismatch")
    receipt_path = PACKAGE_ROOT / "evidence_reuse_receipt.json"
    if receipt_path.exists():
        artifacts = load_json(receipt_path).get("artifacts", {})
        sealed = artifacts.get("action_reference_ledger")
        if sealed is not None and sealed["sha256"] != digest:
            raise ValueError("action reference evidence receipt mismatch")
    result = {_reference_key(value): value for value in rows}
    if len(result) != len(rows):
        raise ValueError("action reference ledger has duplicate identities")
    return result


def _reference_event(value: dict[str, Any]) -> ReplacementEvent | None:
    event = value["reference_event"]
    return None if event is None else ReplacementEvent(**event)


def load_sessions(
    confirmation_ms: tuple[int, ...] = (500,),
    *,
    validate_mapping_ledger: bool = True,
) -> list[SessionExamples]:
    cfg = config()
    enrollment = int(cfg["enrollment_ms"] * 16)
    silence_reset = int(cfg["silence_reset_ms"] * 16)
    allowed = set(map(int, cfg["oracle_mapping_confirmation_ms"]))
    if not set(confirmation_ms) <= allowed:
        raise ValueError("confirmation is outside the frozen oracle mapping ledger")
    ledger = _mapping_index() if validate_mapping_ledger else {}
    reference_ledger = _reference_index()
    consumed_reference_keys: set[tuple[str, str, str]] = set()
    sessions = []
    for role in ("dev", "eval"):
        manifest, receipts, _, _ = _load_role_inputs(role)
        source_receipts = _trace_by_source(receipts["streaming_sortformer"])
        for row in manifest:
            source_id = str(row["source_id"])
            cells, _, slot_ids = _posterior_inputs(row, source_receipts[source_id])
            for persistence in confirmation_ms:
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=int(persistence * 16),
                    enrollment_samples=enrollment,
                    silence_reset_samples=silence_reset,
                )
                reference_events = []
                for episode in reference.episodes:
                    expected = {
                        "schema_version": "psem.frozen_ceiling.action_reference.v1",
                        "old_v2_role": row["role"],
                        "source_id": source_id,
                        "confirmation_ms": persistence,
                        "anchor_episode_id": episode.episode_id,
                        "anchor_speaker": episode.anchor_speaker,
                        "anchor_emit_sample": episode.anchor_emit_sample,
                        "episode_end_sample": episode.end_emit_sample,
                    }
                    key = _reference_key(expected)
                    record = reference_ledger.get(key)
                    if record is None or {
                        name: record[name] for name in expected
                    } != expected:
                        raise ValueError(f"action reference ledger mismatch: {key}")
                    consumed_reference_keys.add(key)
                    event = _reference_event(record)
                    if event is not None:
                        reference_events.append(event)
                reference = replace(reference, events=tuple(reference_events))
                episode_ids: list[str] = []
                speakers: list[str] = []
                starts: list[int] = []
                ends: list[int] = []
                frontiers: list[int] = []
                probabilities: list[list[float]] = []
                alive: list[list[bool]] = []
                reset: list[bool] = []
                valid: list[bool] = []
                masked: list[bool] = []
                speech: list[bool] = []
                anchor_present: list[bool] = []
                overlap: list[bool] = []
                mapping_records = []
                for episode in reference.episodes:
                    record = {
                        "schema_version": "psem.frozen_ceiling.oracle_mapping.v1",
                        "old_v2_role": row["role"],
                        "source_id": source_id,
                        "confirmation_ms": persistence,
                        "anchor_episode_id": episode.episode_id,
                        "anchor_speaker": episode.anchor_speaker,
                        "anchor_emit_sample": episode.anchor_emit_sample,
                        "episode_end_sample": episode.end_emit_sample,
                    }
                    try:
                        mapping = oracle_anchor_mapping(episode, cells, slot_ids)
                    except ValueError as error:
                        record.update({"status": "unmapped", "reason": str(error)})
                        mapping_records.append(record)
                        if validate_mapping_ledger and ledger.get(_mapping_key(record)) != record:
                            raise ValueError(
                                f"oracle mapping ledger mismatch: {_mapping_key(record)}"
                            )
                        continue
                    record.update({"status": "mapped", **mapping.to_dict()})
                    mapping_records.append(record)
                    if validate_mapping_ledger and ledger.get(_mapping_key(record)) != record:
                        raise ValueError(f"oracle mapping ledger mismatch: {_mapping_key(record)}")
                    first = bisect_left(
                        cells,
                        episode.anchor_emit_sample,
                        key=lambda value: value.cell.center_sample,
                    )
                    last = bisect_left(
                        cells,
                        episode.end_emit_sample,
                        key=lambda value: value.cell.center_sample,
                    )
                    continuity_valid = True
                    for posterior in cells[first:last]:
                        cell = posterior.cell
                        start = max(cell.start_sample, episode.anchor_emit_sample)
                        end = min(cell.end_sample, episode.end_emit_sample)
                        if end <= start:
                            continue
                        if posterior.state_reset and start > episode.anchor_emit_sample:
                            continuity_valid = False
                        continuity_valid = bool(
                            continuity_valid
                            and posterior.trace_valid
                            and posterior.slot_alive[mapping.slot_index]
                        )
                        order = [mapping.slot_index]
                        order.extend(
                            index for index in range(len(slot_ids)) if index != mapping.slot_index
                        )
                        active = cell.active_speakers
                        episode_ids.append(episode.episode_id)
                        speakers.append(episode.anchor_speaker)
                        starts.append(start)
                        ends.append(end)
                        frontiers.append(max(posterior.evidence_frontier_sample, end))
                        probabilities.append(
                            [float(posterior.probabilities[index]) for index in order]
                        )
                        alive.append([bool(posterior.slot_alive[index]) for index in order])
                        reset.append(bool(posterior.state_reset))
                        valid.append(continuity_valid)
                        masked.append(bool(cell.masked))
                        speech.append(bool(active))
                        anchor_present.append(episode.anchor_speaker in active)
                        overlap.append(
                            episode.anchor_speaker in active
                            and any(value != episode.anchor_speaker for value in active)
                        )
                sessions.append(
                    SessionExamples(
                        role=role,
                        source_id=source_id,
                        source_family=source_family(row),
                        confirmation_ms=int(persistence),
                        manifest=row,
                        reference=reference,
                        mapping_records=tuple(mapping_records),
                        episode_ids=np.asarray(episode_ids),
                        episode_speakers=np.asarray(speakers),
                        starts=np.asarray(starts, dtype=np.int64),
                        ends=np.asarray(ends, dtype=np.int64),
                        frontiers=np.asarray(frontiers, dtype=np.int64),
                        probabilities=np.asarray(probabilities, dtype=np.float32),
                        alive=np.asarray(alive, dtype=np.bool_),
                        reset=np.asarray(reset, dtype=np.bool_),
                        valid=np.asarray(valid, dtype=np.bool_),
                        masked=np.asarray(masked, dtype=np.bool_),
                        speech_present=np.asarray(speech, dtype=np.bool_),
                        anchor_present=np.asarray(anchor_present, dtype=np.bool_),
                        overlap=np.asarray(overlap, dtype=np.bool_),
                    )
                )
    if consumed_reference_keys != set(reference_ledger):
        raise ValueError("action reference ledger coverage mismatch")
    return sessions


def _event_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(value["source_id"]),
        str(value["anchor_episode_id"]),
        str(value["anchor_id"]),
        int(value["boundary_source_sample"]),
        int(value["model_evidence_frontier_sample"]),
        int(value["decoder_emit_sample"]),
        value["compute_lag_ms"],
        int(value["confirmation_samples"]),
    )


def freeze_action_reference_ledger() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config()
    persistence = int(cfg["action_reference_confirmation_ms"])
    enrollment = int(cfg["enrollment_ms"] * 16)
    silence_reset = int(cfg["silence_reset_ms"] * 16)
    records = []
    per_role: dict[str, dict[str, int]] = {}
    dev_events: list[dict[str, Any]] = []
    for role in ("dev", "eval"):
        manifest, _, _, _ = _load_role_inputs(role)
        episode_count = 0
        event_count = 0
        for row in manifest:
            reference = gt_reference_session(
                row,
                replacement_confirmation_samples=persistence * 16,
                enrollment_samples=enrollment,
                silence_reset_samples=silence_reset,
            )
            event_by_episode = {value.anchor_episode_id: value for value in reference.events}
            for episode in reference.episodes:
                event = event_by_episode.get(episode.episode_id)
                event_value = event.to_dict() if event is not None else None
                records.append(
                    {
                        "schema_version": "psem.frozen_ceiling.action_reference.v1",
                        "old_v2_role": row["role"],
                        "source_id": row["source_id"],
                        "confirmation_ms": persistence,
                        "anchor_episode_id": episode.episode_id,
                        "anchor_speaker": episode.anchor_speaker,
                        "anchor_emit_sample": episode.anchor_emit_sample,
                        "episode_end_sample": episode.end_emit_sample,
                        "reference_event": event_value,
                    }
                )
                if role == "dev" and event_value is not None:
                    dev_events.append(event_value)
            episode_count += len(reference.episodes)
            event_count += len(reference.events)
        per_role[role] = {
            "source_count": len(manifest),
            "episode_count": episode_count,
            "reference_event_count": event_count,
        }
    predecessor_events_path = PREDECESSOR_ROOT / "results" / "dev" / "gate0_oracle_events.jsonl"
    predecessor_events = [
        value
        for value in load_jsonl(predecessor_events_path)
        if int(value["confirmation_ms"]) == persistence
    ]
    if sorted(map(_event_signature, dev_events)) != sorted(
        map(_event_signature, predecessor_events)
    ):
        raise ValueError("fixed DEV action reference differs from sealed Gate-0 events")
    issue98_counts = {}
    for role in ("dev", "eval"):
        frontier_path = SIMPLIFICATION_ROOT / "results" / role / "product_frontiers.json"
        frontier = load_json(frontier_path)
        row = next(
            value
            for value in frontier["rows"]
            if value["family"] == "streaming_sortformer"
            and value["candidate"] == "simple_anchor"
            and value["arm"] == "s1_oracle_anchor"
            and float(value["anchor_threshold"]) == float(cfg["current_anchor_threshold"])
            and int(value["replacement_confirm_ms"]) == persistence
        )
        issue98_counts[role] = int(row["reference_replacement_count"])
        if issue98_counts[role] != per_role[role]["reference_event_count"]:
            raise ValueError(f"fixed {role} action reference count differs from issue98")
    write_jsonl(ACTION_REFERENCE_LEDGER_PATH, records)
    coverage = {
        "schema_version": "psem.frozen_ceiling.action_reference_coverage.v1",
        "confirmation_ms": persistence,
        "episode_count": len(records),
        "reference_event_count": sum(
            value["reference_event"] is not None for value in records
        ),
        "ledger_sha256": sha256_file(ACTION_REFERENCE_LEDGER_PATH),
        "dev_gate0_exact_event_match": True,
        "dev_gate0_events_sha256": sha256_file(predecessor_events_path),
        "issue98_reference_event_counts": issue98_counts,
        "per_role": per_role,
    }
    write_json(ACTION_REFERENCE_COVERAGE_PATH, coverage)
    return records, coverage


def freeze_oracle_mapping_ledger() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config()
    records = []
    for persistence in map(int, cfg["oracle_mapping_confirmation_ms"]):
        sessions = load_sessions((persistence,), validate_mapping_ledger=False)
        records.extend(record for session in sessions for record in session.mapping_records)
    write_jsonl(MAPPING_LEDGER_PATH, records)
    mapped = [value for value in records if value["status"] == "mapped"]
    unmapped = [value for value in records if value["status"] == "unmapped"]
    coverage = {
        "schema_version": "psem.frozen_ceiling.oracle_mapping_coverage.v1",
        "episode_count": len(records),
        "mapped_episode_count": len(mapped),
        "unmapped_episode_count": len(unmapped),
        "mapped_episode_fraction": len(mapped) / len(records),
        "unmapped_support_seconds": sum(
            value["episode_end_sample"] - value["anchor_emit_sample"] for value in unmapped
        )
        / 16000.0,
        "ledger_sha256": sha256_file(MAPPING_LEDGER_PATH),
        "per_confirmation": {
            str(persistence): {
                "episode_count": sum(value["confirmation_ms"] == persistence for value in records),
                "unmapped_episode_count": sum(
                    value["confirmation_ms"] == persistence and value["status"] == "unmapped"
                    for value in records
                ),
            }
            for persistence in map(int, cfg["oracle_mapping_confirmation_ms"])
        },
    }
    write_json(MAPPING_COVERAGE_PATH, coverage)
    return records, coverage


def inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = config()
    rows = []
    for role in ("dev", "eval"):
        manifest, _, _, _ = _load_role_inputs(role)
        for row in manifest:
            rows.append(
                {
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "corpus": row["corpus"],
                    "audio_ref": row["audio_ref"],
                    "source_family": source_family(row),
                    "old_v2_role": row["role"],
                    "issue99_use": (
                        "architecture_freeze_only" if role == "dev" else "source_held_out_scoring"
                    ),
                    "row_sha256": row["row_sha256"],
                    "waveform_sha256": row["waveform_sha256"],
                    "intervals_sha256": row["intervals_sha256"],
                }
            )
    split = {
        "schema_version": "psem.frozen_ceiling.split.v1",
        "strategy": cfg["split"]["strategy"],
        "fresh_holdout": False,
        "limitation": "all V2 sources are development-known path-selection evidence",
        "architecture_freeze_role": cfg["split"]["architecture_freeze_role"],
        "scoring_role": cfg["split"]["scoring_role"],
        "folds": [
            {
                "held_out_family": family,
                "train_families": [value for value in cfg["split"]["families"] if value != family],
                "training_sources": [
                    value["source_id"]
                    for value in rows
                    if value["old_v2_role"] == cfg["split"]["architecture_freeze_role"]
                    and value["source_family"] != family
                ],
                "evaluation_sources": [
                    value["source_id"]
                    for value in rows
                    if value["old_v2_role"] == cfg["split"]["scoring_role"]
                    and value["source_family"] == family
                ],
            }
            for family in cfg["split"]["families"]
        ],
        "sources": rows,
    }
    receipt_paths = {
        "issue97_config": PREDECESSOR_ROOT / "config.json",
        "issue98_config": SIMPLIFICATION_ROOT / "config.json",
        "issue98_evaluator": SIMPLIFICATION_ROOT / "evaluate_simplified_ontologies.py",
        "issue98_dev_ontology_sufficiency": SIMPLIFICATION_ROOT
        / "results"
        / "dev"
        / "ontology_sufficiency.json",
        "issue98_dev_product_frontiers": SIMPLIFICATION_ROOT
        / "results"
        / "dev"
        / "product_frontiers.json",
        "issue98_eval_product_frontiers": SIMPLIFICATION_ROOT
        / "results"
        / "eval"
        / "product_frontiers.json",
        "trace_reuse_receipt": SIMPLIFICATION_ROOT / "trace_reuse_receipt.json",
        "dev_manifest": PREDECESSOR_ROOT / "results" / "dev" / "relative_occupancy_manifest.jsonl",
        "eval_manifest": PREDECESSOR_ROOT
        / "results"
        / "eval"
        / "relative_occupancy_manifest.jsonl",
        "dev_sortformer_receipt": PREDECESSOR_ROOT
        / "results"
        / "dev"
        / "sortformer_model_receipt.json",
        "eval_sortformer_receipt": PREDECESSOR_ROOT
        / "results"
        / "eval"
        / "sortformer_model_receipt.json",
        "dev_vad_receipt": SIMPLIFICATION_ROOT
        / "results"
        / "dev"
        / "production_vad_replay_receipt.json",
        "dev_vad_gate": SIMPLIFICATION_ROOT
        / "results"
        / "dev"
        / "production_vad_speech_gate.jsonl",
        "eval_vad_receipt": SIMPLIFICATION_ROOT
        / "results"
        / "eval"
        / "production_vad_replay_receipt.json",
        "eval_vad_gate": SIMPLIFICATION_ROOT
        / "results"
        / "eval"
        / "production_vad_speech_gate.jsonl",
        "dev_gt_action_events": PREDECESSOR_ROOT / "results" / "dev" / "gate0_oracle_events.jsonl",
        "dev_gt_action_metrics": PREDECESSOR_ROOT / "results" / "dev" / "gate0_oracle_metrics.json",
        "product_evaluator": PREDECESSOR_ROOT / "model_evaluate.py",
        "gt_action_oracle": PREDECESSOR_ROOT / "decoder.py",
        "oracle_mapping_code": PREDECESSOR_ROOT / "model_decode.py",
        "feature_contract": PACKAGE_ROOT / "posterior_features.py",
        "experiment_config": CONFIG_PATH,
    }
    if MAPPING_LEDGER_PATH.exists() and MAPPING_COVERAGE_PATH.exists():
        receipt_paths.update(
            {
                "oracle_mapping_ledger": MAPPING_LEDGER_PATH,
                "oracle_mapping_coverage": MAPPING_COVERAGE_PATH,
            }
        )
    if ACTION_REFERENCE_LEDGER_PATH.exists() and ACTION_REFERENCE_COVERAGE_PATH.exists():
        receipt_paths.update(
            {
                "action_reference_ledger": ACTION_REFERENCE_LEDGER_PATH,
                "action_reference_coverage": ACTION_REFERENCE_COVERAGE_PATH,
            }
        )
    sortformer_receipt = load_json(receipt_paths["eval_sortformer_receipt"])
    receipt = {
        "schema_version": "psem.frozen_ceiling.evidence_reuse_receipt.v1",
        "authority": cfg["authority"],
        "repository_baseline_sha": cfg["repository_baseline"],
        "dirty_state_at_issue99_start": [],
        "artifacts": {
            key: {
                "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for key, path in receipt_paths.items()
        },
        "sortformer": {
            key: sortformer_receipt[key]
            for key in (
                "model_repository",
                "model_revision",
                "model_sha256",
                "source_repository",
                "source_commit",
                "bench_sha256",
                "backend",
                "native_frame_ms",
                "recorded_algorithmic_lookahead_ms",
                "preset",
                "threads",
            )
        },
        "new_sortformer_inference_required": False,
        "new_sortformer_inference_performed": False,
    }
    write_json(PACKAGE_ROOT / "split_manifest.json", split)
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
                    "frame_count": sum(len(v.starts) for v in sessions),
                }
            )
        )


if __name__ == "__main__":
    main()
