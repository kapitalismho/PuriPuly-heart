from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.psem_ontology_simplification_gate.derive_anchor_overlap import (
    AnchorOverlapState,
    derive_gt_anchor_overlap_state,
    derive_model_anchor_overlap_state,
)
from experiments.psem_ontology_simplification_gate.derive_simple_anchor import (
    SimpleAnchorState,
    derive_gt_simple_anchor_state,
    derive_model_simple_anchor_state,
)
from experiments.psem_relative_occupancy_gate.contracts import AnchorLifecycle
from experiments.psem_relative_occupancy_gate.decoder import ReplacementEvent
from experiments.psem_relative_occupancy_gate.evaluate import (
    monotonic_boundary_matches,
    weighted_average_precision,
    weighted_binary_confusion,
    weighted_binary_pr_curve,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    load_json,
    load_jsonl,
    percentile,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.model_decode import (
    ModelObservation,
    OracleAnchorMapping,
    PosteriorCell,
    model_observations,
    oracle_anchor_mapping,
    posterior_cells,
    relative_probabilities,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    gate1_fail_closed_exposure,
    gt_reference_session,
    intervals_from_manifest,
)
from experiments.psem_relative_occupancy_gate.trace_io import (
    TRACE_ARCHIVE_NAMES,
    validate_trace_receipt,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
AUTHORITY_SNAPSHOT_PATH = REPOSITORY_ROOT / ".agents" / "goals" / "goal-issue-98" / "authority.snapshot.md"
PREDECESSOR_ROOT = REPOSITORY_ROOT / "experiments" / "psem_relative_occupancy_gate"
RESULTS_ROOT = PACKAGE_ROOT / "results"
FAMILY_KEYS = ("streaming_sortformer", "ls_eend")
RECEIPT_NAMES = {
    "streaming_sortformer": "sortformer_model_receipt.json",
    "ls_eend": "lseend_model_receipt.json",
}
OUTPUT_NAMES = {
    "streaming_sortformer": "sortformer",
    "ls_eend": "lseend",
}


class SimplificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    source_id: str
    episode_id: str
    start_sample: int
    end_sample: int
    anchor_label: bool
    overlap_label: bool
    active_speech: bool
    anchor_score: float
    nonanchor_score: float

    @property
    def weight_samples(self) -> int:
        return self.end_sample - self.start_sample


class SimplifiedReplacementDecoder:
    def __init__(self, source_id: str, confirmation_samples: int) -> None:
        if confirmation_samples <= 0:
            raise ValueError("confirmation_samples must be positive")
        self.source_id = source_id
        self.confirmation_samples = confirmation_samples
        self.pending_boundary_sample: int | None = None
        self.pending_evidence_samples = 0

    def clear(self) -> None:
        self.pending_boundary_sample = None
        self.pending_evidence_samples = 0

    def advance(
        self,
        *,
        start_sample: int,
        end_sample: int,
        evidence_frontier_sample: int,
        replacement_evidence: bool,
        pause: bool,
        lifecycle: AnchorLifecycle,
        anchor_id: str | None,
        anchor_episode_id: str | None,
    ) -> ReplacementEvent | None:
        if lifecycle is not AnchorLifecycle.ANCHORED:
            self.clear()
            return None
        if anchor_id is None or anchor_episode_id is None:
            raise ValueError("anchored decoding requires anchor identity")
        if pause:
            return None
        if not replacement_evidence:
            self.clear()
            return None
        if self.pending_boundary_sample is None:
            self.pending_boundary_sample = start_sample
        duration = end_sample - start_sample
        needed = self.confirmation_samples - self.pending_evidence_samples
        if duration < needed:
            self.pending_evidence_samples += duration
            return None
        qualifying_sample = start_sample + needed
        event = ReplacementEvent(
            source_id=self.source_id,
            anchor_episode_id=anchor_episode_id,
            anchor_id=anchor_id,
            boundary_source_sample=self.pending_boundary_sample,
            model_evidence_frontier_sample=evidence_frontier_sample,
            decoder_emit_sample=max(qualifying_sample, evidence_frontier_sample),
            compute_lag_ms=None,
            confirmation_samples=self.confirmation_samples,
        )
        self.clear()
        return event


def _config() -> dict[str, Any]:
    value = load_json(CONFIG_PATH)
    if not isinstance(value, dict):
        raise SimplificationError("config must be an object")
    predecessor = REPOSITORY_ROOT / str(value["predecessor"]["config_path"])
    if sha256_file(predecessor) != value["predecessor"]["config_sha256"]:
        raise SimplificationError("predecessor config binding mismatch")
    authority_text = AUTHORITY_SNAPSHOT_PATH.read_text(encoding="utf-8")
    authority = value["authority"]
    if (
        f"authority_ref: {authority['ref']}" not in authority_text
        or f"authority_pin: {authority['sha256']}" not in authority_text
        or f"document_sha256: {authority['sha256']}" not in authority_text
    ):
        raise SimplificationError("authority snapshot binding mismatch")
    return value


def _role_name(role: str) -> str:
    return "PSEM-STRATEGY-DEV" if role == "dev" else "PSEM-STRATEGY-EVAL"


def _result_dir(role: str) -> Path:
    return RESULTS_ROOT / role


def _predecessor_result(role: str, name: str) -> Path:
    return PREDECESSOR_ROOT / "results" / role / name


def _read_jsonl_stream(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SimplificationError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _trace_receipts(role: str) -> dict[str, dict[str, Any]]:
    return {
        family: load_json(_predecessor_result(role, name)) for family, name in RECEIPT_NAMES.items()
    }


def _unique_index(
    rows: Sequence[dict[str, Any]], key: Any, label: str
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in result:
            raise SimplificationError(f"duplicate {label}: {row_key}")
        result[row_key] = row
    return result


def _trace_by_source(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _unique_index(
        receipt["source_receipts"], lambda value: str(value["source_id"]), "trace source"
    )


def _validate_model_receipt(
    role: str,
    family: str,
    receipt: dict[str, Any],
    expected_sources: set[str],
    predecessor_cfg: dict[str, Any],
) -> None:
    if receipt.get("schema_version") != "psem.relative_occupancy.model_receipt.v1":
        raise SimplificationError(f"model receipt schema mismatch: {role} {family}")
    if receipt.get("family") != family:
        raise SimplificationError(f"model receipt family mismatch: {role} {family}")
    source_receipts = _trace_by_source(receipt)
    if set(source_receipts) != expected_sources:
        raise SimplificationError(f"model receipt source coverage mismatch: {role} {family}")
    if set(map(str, receipt.get("source_ids", []))) != expected_sources:
        raise SimplificationError(f"model receipt source_ids mismatch: {role} {family}")
    if int(receipt.get("source_count", -1)) != len(expected_sources):
        raise SimplificationError(f"model receipt source count mismatch: {role} {family}")
    config_key = "sortformer" if family == "streaming_sortformer" else "lseend"
    expected_pin = predecessor_cfg[config_key]
    pin_fields = (
        "backend",
        "family",
        "model_filename",
        "model_repository",
        "model_revision",
        "model_sha256",
        "repository",
        "revision",
        "sidecar_sha256",
        "variant",
    )
    for field in pin_fields:
        if field in expected_pin and receipt.get(field) != expected_pin[field]:
            raise SimplificationError(f"model pin mismatch: {role} {family} {field}")


def _inventory_family(role: str, family: str, receipt: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for source_receipt in receipt["source_receipts"]:
        trace_receipt = dict(source_receipt["trace"])
        trace_path = Path(str(trace_receipt["trace_path"]))
        trace = validate_trace_receipt(trace_path, trace_receipt)
        sources.append(
            {
                "source_id": trace.source_id,
                "trace_path": str(trace_path),
                "trace_sha256": trace_receipt["trace_sha256"],
                "trace_size_bytes": trace_receipt["trace_size_bytes"],
                "trace_schema_version": trace_receipt["trace_schema_version"],
                "frame_count": trace_receipt["frame_count"],
                "slot_ids": list(trace.slot_ids),
                "metadata_sha256": canonical_sha256(trace.metadata),
            }
        )
    model_pin = {
        key: receipt.get(key)
        for key in (
            "backend",
            "family",
            "model_filename",
            "model_repository",
            "model_revision",
            "model_sha256",
            "repository",
            "revision",
            "sidecar_sha256",
            "variant",
        )
        if receipt.get(key) is not None
    }
    field_coverage = {
        "native_frame_start_source_samples": all(int(value["frame_count"]) > 0 for value in sources),
        "native_frame_end_source_samples": all(int(value["frame_count"]) > 0 for value in sources),
        "model_evidence_frontier_source_samples": all(int(value["frame_count"]) > 0 for value in sources),
        "speaker_slot_or_attractor_ids": all(bool(value["slot_ids"]) for value in sources),
        "speaker_activity_probabilities": all(int(value["frame_count"]) > 0 for value in sources),
        "slot_validity_metadata": "slot_alive per frame; no native semantic validity flag",
        "reset_metadata": all(value["trace_schema_version"] == "psem.relative_occupancy.trace.v1" for value in sources),
    }
    return {
        "role": _role_name(role),
        "family": family,
        "model_receipt_path": str(
            _predecessor_result(role, RECEIPT_NAMES[family]).relative_to(REPOSITORY_ROOT)
        ),
        "model_receipt_sha256": sha256_file(_predecessor_result(role, RECEIPT_NAMES[family])),
        "model_pin": model_pin,
        "source_count": len(sources),
        "sources": sources,
        "field_coverage": field_coverage,
        "archive_members": list(TRACE_ARCHIVE_NAMES),
    }


def write_inventory_and_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _config()
    roles = {}
    missing_fields: list[str] = []
    for role in ("dev", "eval"):
        manifest_path = _predecessor_result(role, "relative_occupancy_manifest.jsonl")
        manifest, receipts, gate1_rows, gate2_rows = _load_role_inputs(role)
        manifest_index = _unique_index(
            manifest, lambda value: str(value["source_id"]), f"{role} manifest source"
        )
        family_inventory = {
            family: _inventory_family(role, family, receipts[family]) for family in FAMILY_KEYS
        }
        for family, inventory in family_inventory.items():
            for field, present in inventory["field_coverage"].items():
                if present is False:
                    missing_fields.append(f"{role}.{family}.{field}")
        derived_field_coverage = {
            "oracle_anchor_mapping": all(
                all(
                    "anchor_episode_id" in mapping
                    and "slot_index" in mapping
                    and "slot_id" in mapping
                    for mapping in row["oracle_mappings"]
                )
                for row in gate1_rows.values()
            ),
            "causal_anchor_lifecycle": all(
                all("lifecycle" in span for span in row["timeline"])
                for row in gate2_rows.values()
            ),
            "causal_selected_slot": all(
                all(
                    "anchor_slot_index" in annotation and "anchor_slot_id" in annotation
                    for annotation in row["annotated_episodes"]
                )
                for row in gate2_rows.values()
            ),
        }
        for field, present in derived_field_coverage.items():
            if not present:
                missing_fields.append(f"{role}.{field}")
        roles[role] = {
            "predecessor_artifacts": {
                "manifest": {
                    "path": str(manifest_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": sha256_file(manifest_path),
                    "schema_version": "psem.relative_occupancy.manifest.v1",
                    "row_count": len(manifest_index),
                },
                "gate1_event_ledger": {
                    "path": str(_predecessor_result(role, "gate1_event_ledger.jsonl").relative_to(REPOSITORY_ROOT)),
                    "sha256": sha256_file(_predecessor_result(role, "gate1_event_ledger.jsonl")),
                    "schema_version": "psem.relative_occupancy.gate_event_session.v1",
                    "row_count": len(gate1_rows),
                },
                "gate2_event_ledger": {
                    "path": str(_predecessor_result(role, "gate2_event_ledger.jsonl").relative_to(REPOSITORY_ROOT)),
                    "sha256": sha256_file(_predecessor_result(role, "gate2_event_ledger.jsonl")),
                    "schema_version": "psem.relative_occupancy.gate_event_session.v1",
                    "row_count": len(gate2_rows),
                },
            },
            "derived_field_coverage": derived_field_coverage,
            "families": family_inventory,
        }
    receipt = {
        "schema_version": "psem.ontology_simplification.trace_reuse_receipt.v1",
        "authority": cfg["authority"],
        "predecessor": cfg["predecessor"],
        "roles": roles,
        "missing_required_neutral_fields": missing_fields,
        "new_model_inference_required": False,
        "new_model_inference_performed": False,
        "reuse_decision": "derive every challenger view offline from exact issue-97 cached traces",
    }
    audited_sources = [PREDECESSOR_ROOT / "model_decode.py", PREDECESSOR_ROOT / "decoder.py"]
    audit = {
        "schema_version": "psem.ontology_simplification.causal_dependency_audit.v1",
        "authority": cfg["authority"],
        "audited_sources": [
            {
                "path": str(source_path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(source_path),
            }
            for source_path in audited_sources
        ],
        "dependencies": [
            {
                "surface": "causal enrollment",
                "material_dependency": True,
                "mechanism": "a singleton candidate is rejected when any other alive slot exceeds other_low_threshold",
            },
            {
                "surface": "selected-slot episode termination",
                "material_dependency": True,
                "mechanism": "ReplacementDecoder ends the episode after the old OTHER_ONLY state persists",
            },
            {
                "surface": "reset and uncertainty",
                "material_dependency": False,
                "mechanism": "trace reset, slot validity, and silence reset are independent of other_present",
            },
        ],
        "conclusion": "material_dependency_present",
        "s2_label": "fixed-issue-97-lifecycle-counterfactual-ablation",
        "native_simplified_ontology_runtime_claim_allowed": False,
    }
    write_json(PACKAGE_ROOT / "trace_reuse_receipt.json", receipt)
    write_json(PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json", audit)
    return receipt, audit


def _load_role_inputs(
    role: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    cfg = _config()
    predecessor_cfg = load_json(REPOSITORY_ROOT / str(cfg["predecessor"]["config_path"]))
    manifest = load_jsonl(_predecessor_result(role, "relative_occupancy_manifest.jsonl"))
    manifest_index = _unique_index(
        manifest, lambda value: str(value["source_id"]), f"{role} manifest source"
    )
    expected_sources = set(manifest_index)
    receipts = _trace_receipts(role)
    for family in FAMILY_KEYS:
        _validate_model_receipt(role, family, receipts[family], expected_sources, predecessor_cfg)
    gate1_rows = _read_jsonl_stream(_predecessor_result(role, "gate1_event_ledger.jsonl"))
    gate2_rows = _read_jsonl_stream(_predecessor_result(role, "gate2_event_ledger.jsonl"))
    def key(row: dict[str, Any]) -> tuple[str, str, int]:
        return (
            str(row["source_id"]),
            str(row["family"]),
            int(row["replacement_confirm_ms"]),
        )
    gate1 = _unique_index(gate1_rows, key, f"{role} Gate 1 ledger key")
    gate2 = _unique_index(gate2_rows, key, f"{role} Gate 2 ledger key")
    expected_keys = {
        (source_id, family, int(persistence))
        for source_id in expected_sources
        for family in FAMILY_KEYS
        for persistence in cfg["replacement_confirm_ms"]
    }
    for name, rows, gate_name in (
        ("Gate 1", gate1, "gate1_oracle_anchor"),
        ("Gate 2", gate2, "gate2_causal_anchor"),
    ):
        if set(rows) != expected_keys:
            raise SimplificationError(f"{role} {name} ledger coverage mismatch")
        if any(
            row.get("schema_version") != "psem.relative_occupancy.gate_event_session.v1"
            or row.get("gate") != gate_name
            for row in rows.values()
        ):
            raise SimplificationError(f"{role} {name} ledger schema mismatch")
    return manifest, receipts, gate1, gate2


def _posterior_inputs(
    row: dict[str, Any], source_receipt: dict[str, Any]
) -> tuple[tuple[PosteriorCell, ...], tuple[ModelObservation, ...], tuple[str, ...]]:
    trace_receipt = dict(source_receipt["trace"])
    trace = validate_trace_receipt(Path(str(trace_receipt["trace_path"])), trace_receipt)
    intervals = intervals_from_manifest(row)
    cells = posterior_cells(
        trace,
        intervals,
        int(row["scored_start_sample"]),
        int(row["scored_end_sample"]),
    )
    return cells, model_observations(cells, intervals), trace.slot_ids


def _episode_anchor_records(
    *,
    source_id: str,
    episode_id: str,
    anchor_speaker: str,
    anchor_slot_index: int,
    episode_start: int,
    episode_end: int,
    cells: Sequence[PosteriorCell],
) -> tuple[list[AnchorRecord], int, int]:
    records: list[AnchorRecord] = []
    invalidated_samples = 0
    invalidated_unmasked_active_samples = 0
    continuity_valid = True
    start_index = bisect_left(
        cells, episode_start, key=lambda value: value.cell.center_sample
    )
    end_index = bisect_left(
        cells, episode_end, key=lambda value: value.cell.center_sample
    )
    for posterior in cells[start_index:end_index]:
        if (
            not posterior.trace_valid
            or posterior.state_reset
            or not posterior.slot_alive[anchor_slot_index]
        ):
            continuity_valid = False
        if not continuity_valid:
            invalidated_samples += posterior.cell.duration_samples
            if not posterior.cell.masked and posterior.cell.active_speakers:
                invalidated_unmasked_active_samples += posterior.cell.duration_samples
            continue
        if posterior.cell.masked:
            continue
        probabilities = relative_probabilities(posterior, anchor_slot_index)
        if probabilities is None:
            raise SimplificationError("valid diagnostic continuity has no anchor probabilities")
        p_anchor, p_other = probabilities
        active = posterior.cell.active_speakers
        records.append(
            AnchorRecord(
                source_id=source_id,
                episode_id=episode_id,
                start_sample=posterior.cell.start_sample,
                end_sample=posterior.cell.end_sample,
                anchor_label=anchor_speaker in active,
                overlap_label=(
                    anchor_speaker in active
                    and any(value != anchor_speaker for value in active)
                ),
                active_speech=bool(active),
                anchor_score=p_anchor,
                nonanchor_score=p_other,
            )
        )
    return records, invalidated_samples, invalidated_unmasked_active_samples


def _oracle_anchor_records(
    row: dict[str, Any],
    cells: Sequence[PosteriorCell],
    slot_ids: Sequence[str],
    reference: Any,
) -> tuple[list[AnchorRecord], dict[str, OracleAnchorMapping], dict[str, Any]]:
    records: list[AnchorRecord] = []
    mappings: dict[str, OracleAnchorMapping] = {}
    unmapped: list[Any] = []
    continuity_invalid_episode_count = 0
    continuity_invalid_samples = 0
    continuity_invalid_unmasked_active_samples = 0
    for episode in reference.episodes:
        try:
            mapping = oracle_anchor_mapping(episode, cells, slot_ids)
        except ValueError:
            unmapped.append(episode)
            continue
        mappings[episode.episode_id] = mapping
        episode_records, invalid_samples, invalid_active_samples = _episode_anchor_records(
            source_id=str(row["source_id"]),
            episode_id=episode.episode_id,
            anchor_speaker=episode.anchor_speaker,
            anchor_slot_index=mapping.slot_index,
            episode_start=episode.anchor_emit_sample,
            episode_end=episode.end_emit_sample,
            cells=cells,
        )
        records.extend(episode_records)
        if invalid_samples:
            continuity_invalid_episode_count += 1
            continuity_invalid_samples += invalid_samples
            continuity_invalid_unmasked_active_samples += invalid_active_samples
    intervals = intervals_from_manifest(row)
    total_samples = sum(
        int(episode.end_emit_sample) - int(episode.anchor_emit_sample)
        for episode in reference.episodes
    )
    unmapped_samples = sum(
        int(episode.end_emit_sample) - int(episode.anchor_emit_sample) for episode in unmapped
    )
    unmapped_active_samples = sum(
        max(
            0,
            min(interval.end_sample, episode.end_emit_sample)
            - max(interval.start_sample, episode.anchor_emit_sample),
        )
        for episode in unmapped
        for interval in intervals
        if not interval.masked and interval.active_speakers
    )
    coverage = {
        "source_id": str(row["source_id"]),
        "episode_count": len(reference.episodes),
        "mapped_episode_count": len(mappings),
        "unmapped_episode_count": len(unmapped),
        "mapped_episode_fraction": (
            len(mappings) / len(reference.episodes) if reference.episodes else None
        ),
        "episode_support_seconds": total_samples / 16000.0,
        "unmapped_episode_support_seconds": unmapped_samples / 16000.0,
        "unmapped_unmasked_active_speech_seconds": unmapped_active_samples / 16000.0,
        "continuity_invalid_episode_count": continuity_invalid_episode_count,
        "continuity_invalid_support_seconds": continuity_invalid_samples / 16000.0,
        "continuity_invalid_unmasked_active_speech_seconds": (
            continuity_invalid_unmasked_active_samples / 16000.0
        ),
    }
    return records, mappings, coverage


def _aggregate_mapping_coverage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(value["episode_count"]) for value in rows)
    mapped = sum(int(value["mapped_episode_count"]) for value in rows)
    return {
        "source_count": len(rows),
        "episode_count": episodes,
        "mapped_episode_count": mapped,
        "unmapped_episode_count": sum(int(value["unmapped_episode_count"]) for value in rows),
        "mapped_episode_fraction": mapped / episodes if episodes else None,
        "episode_support_seconds": sum(float(value["episode_support_seconds"]) for value in rows),
        "unmapped_episode_support_seconds": sum(
            float(value["unmapped_episode_support_seconds"]) for value in rows
        ),
        "unmapped_unmasked_active_speech_seconds": sum(
            float(value["unmapped_unmasked_active_speech_seconds"]) for value in rows
        ),
        "continuity_invalid_episode_count": sum(
            int(value["continuity_invalid_episode_count"]) for value in rows
        ),
        "continuity_invalid_support_seconds": sum(
            float(value["continuity_invalid_support_seconds"]) for value in rows
        ),
        "continuity_invalid_unmasked_active_speech_seconds": sum(
            float(value["continuity_invalid_unmasked_active_speech_seconds"])
            for value in rows
        ),
        "per_source": list(rows),
    }


def _causal_anchor_records(
    row: dict[str, Any], cells: Sequence[PosteriorCell], gate2_row: dict[str, Any]
) -> list[AnchorRecord]:
    records: list[AnchorRecord] = []
    for annotation in gate2_row["annotated_episodes"]:
        anchor_speaker = annotation.get("expected_anchor_speaker")
        slot_index = annotation.get("anchor_slot_index")
        if anchor_speaker is None or slot_index is None:
            continue
        start = int(annotation["anchor_emit_sample"])
        end = int(annotation["end_emit_sample"])
        episode_records, _, _ = _episode_anchor_records(
            source_id=str(row["source_id"]),
            episode_id=str(annotation["episode_id"]),
            anchor_speaker=str(anchor_speaker),
            anchor_slot_index=int(slot_index),
            episode_start=start,
            episode_end=end,
            cells=cells,
        )
        records.extend(episode_records)
    return records


def _anchor_metrics(
    records: Sequence[AnchorRecord], anchor_threshold: float, thresholds: Sequence[float]
) -> dict[str, Any]:
    labels = [value.anchor_label for value in records]
    scores = [value.anchor_score for value in records]
    weights = [float(value.weight_samples) for value in records]
    if not labels:
        raise SimplificationError("anchor metrics require records")
    selected = weighted_binary_confusion(labels, scores, weights, anchor_threshold)
    contexts = {}
    selectors = {
        "gt_anchor_only": lambda value: value.anchor_label and not value.overlap_label,
        "gt_anchor_overlap": lambda value: value.anchor_label and value.overlap_label,
        "gt_anchor_absent_active_speech": lambda value: (
            value.active_speech and not value.anchor_label
        ),
    }
    for name, selector in selectors.items():
        chosen = [value for value in records if selector(value)]
        total = sum(value.weight_samples for value in chosen)
        positive = sum(
            value.weight_samples for value in chosen if value.anchor_score >= anchor_threshold
        )
        contexts[name] = {
            "support_seconds": total / 16000.0,
            "predicted_anchor_seconds": positive / 16000.0,
            "predicted_anchor_fraction": positive / total if total else None,
            "anchor_absence_recall": (
                1.0 - positive / total
                if name == "gt_anchor_absent_active_speech" and total
                else None
            ),
            "anchor_false_positive_duration_seconds": (
                positive / 16000.0 if name == "gt_anchor_absent_active_speech" else None
            ),
        }
    return {
        "record_count": len(records),
        "anchor_threshold": anchor_threshold,
        "anchor_auprc": weighted_average_precision(labels, scores, weights),
        "anchor_pr_curve": weighted_binary_pr_curve(labels, scores, weights, thresholds),
        "selected_threshold_confusion": selected,
        "contexts": contexts,
    }


def _sustained_dropout(records: Sequence[AnchorRecord], anchor_threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    contexts = {
        "gt_anchor_only": lambda value: value.anchor_label and not value.overlap_label,
        "gt_anchor_overlap": lambda value: value.anchor_label and value.overlap_label,
    }
    for name, selector in contexts.items():
        chosen = sorted(
            (value for value in records if selector(value)),
            key=lambda value: (value.source_id, value.episode_id, value.start_sample),
        )
        total_support = sum(value.weight_samples for value in chosen)
        episode_ids = {value.episode_id for value in chosen}
        runs: list[tuple[str, int]] = []
        active_episode: str | None = None
        active_source: str | None = None
        active_end: int | None = None
        active_duration = 0
        for value in chosen:
            below = value.anchor_score < anchor_threshold
            contiguous = (
                below
                and active_episode == value.episode_id
                and active_source == value.source_id
                and active_end == value.start_sample
            )
            if not below:
                if active_duration:
                    runs.append((str(active_episode), active_duration))
                active_episode = None
                active_source = None
                active_end = None
                active_duration = 0
                continue
            if not contiguous:
                if active_duration:
                    runs.append((str(active_episode), active_duration))
                active_episode = value.episode_id
                active_source = value.source_id
                active_duration = 0
            active_duration += value.weight_samples
            active_end = value.end_sample
        if active_duration:
            runs.append((str(active_episode), active_duration))
        horizons = {}
        for horizon_ms in (100, 300, 500):
            horizon = horizon_ms * 16
            qualifying = [value for value in runs if value[1] >= horizon]
            affected_samples = sum(value[1] for value in qualifying)
            affected_episodes = {value[0] for value in qualifying}
            horizons[str(horizon_ms)] = {
                "qualifying_run_count": len(qualifying),
                "affected_support_seconds": affected_samples / 16000.0,
                "duration_weighted_probability": (
                    affected_samples / total_support if total_support else None
                ),
                "affected_episode_count": len(affected_episodes),
                "affected_episode_fraction": (
                    len(affected_episodes) / len(episode_ids) if episode_ids else None
                ),
            }
        result[name] = {
            "support_seconds": total_support / 16000.0,
            "episode_count": len(episode_ids),
            "below_threshold_run_count": len(runs),
            "horizons_ms": horizons,
        }
    return result


def _state_decision(
    *,
    candidate: str,
    observation: ModelObservation,
    anchor_slot_index: int,
    anchor_threshold: float,
    overlap_threshold: float | None,
    strict_inconsistent: bool,
) -> tuple[bool, bool, bool]:
    if (
        not observation.trace_valid
        or not observation.slot_alive[anchor_slot_index]
        or observation.state_reset
    ):
        return False, False, False
    if observation.masked:
        return False, True, True
    p_anchor = float(observation.probabilities[anchor_slot_index])
    p_other = max(
        (
            float(value)
            for index, (value, alive) in enumerate(
                zip(observation.probabilities, observation.slot_alive, strict=True)
            )
            if index != anchor_slot_index and alive
        ),
        default=0.0,
    )
    if candidate == "simple_anchor":
        state = derive_model_simple_anchor_state(
            speech_present=observation.speech_present,
            p_anchor=p_anchor,
            anchor_threshold=anchor_threshold,
        )
        return state is SimpleAnchorState.NON_ANCHOR_SPEECH, False, True
    if overlap_threshold is None:
        raise ValueError("anchor-overlap decoding requires an overlap threshold")
    state = derive_model_anchor_overlap_state(
        speech_present=observation.speech_present,
        p_anchor=p_anchor,
        p_nonanchor_max=p_other,
        anchor_threshold=anchor_threshold,
        anchor_overlap_threshold=overlap_threshold,
        strict_inconsistent=strict_inconsistent,
    )
    return (
        state is AnchorOverlapState.NON_ANCHOR_SPEECH,
        state is AnchorOverlapState.ANCHOR_UNCERTAIN,
        True,
    )


def _episode_decisions(
    *,
    observations: Sequence[ModelObservation],
    episode_start: int,
    episode_end: int,
    candidate: str,
    anchor_slot_index: int,
    anchor_threshold: float,
    overlap_threshold: float | None,
    strict_inconsistent: bool,
) -> list[tuple[ModelObservation, int, int, bool, bool, bool]]:
    decisions = []
    continuity_valid = True
    start_index = bisect_right(observations, episode_start, key=lambda value: value.end_sample)
    end_index = bisect_left(observations, episode_end, key=lambda value: value.start_sample)
    for observation in observations[start_index:end_index]:
        start = max(observation.start_sample, episode_start)
        end = min(observation.end_sample, episode_end)
        if end <= start:
            continue
        replacement, pause, valid = _state_decision(
            candidate=candidate,
            observation=observation,
            anchor_slot_index=anchor_slot_index,
            anchor_threshold=anchor_threshold,
            overlap_threshold=overlap_threshold,
            strict_inconsistent=strict_inconsistent,
        )
        if not valid:
            continuity_valid = False
        if not continuity_valid:
            replacement, pause, valid = False, False, False
        decisions.append((observation, start, end, replacement, pause, valid))
    return decisions


def _episode_exposure_ranges(
    *,
    observations: Sequence[ModelObservation],
    episode_start: int,
    episode_end: int,
    candidate: str,
    anchor_slot_index: int,
    anchor_threshold: float,
    overlap_threshold: float | None,
    strict_inconsistent: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    anchored = []
    uncertain = []
    for observation, start, end, _, pause, valid in _episode_decisions(
        observations=observations,
        episode_start=episode_start,
        episode_end=episode_end,
        candidate=candidate,
        anchor_slot_index=anchor_slot_index,
        anchor_threshold=anchor_threshold,
        overlap_threshold=overlap_threshold,
        strict_inconsistent=strict_inconsistent,
    ):
        if not valid or (pause and not observation.masked):
            uncertain.append((start, end))
        else:
            anchored.append((start, end))
    return anchored, uncertain


def _decode_episode(
    *,
    source_id: str,
    episode_id: str,
    anchor_id: str,
    anchor_slot_index: int,
    episode_start: int,
    episode_end: int,
    observations: Sequence[ModelObservation],
    candidate: str,
    anchor_threshold: float,
    overlap_threshold: float | None,
    strict_inconsistent: bool,
    confirmation_samples: int,
) -> ReplacementEvent | None:
    decoder = SimplifiedReplacementDecoder(source_id, confirmation_samples)
    for observation, start, end, replacement, pause, valid in _episode_decisions(
        observations=observations,
        episode_start=episode_start,
        episode_end=episode_end,
        candidate=candidate,
        anchor_slot_index=anchor_slot_index,
        anchor_threshold=anchor_threshold,
        overlap_threshold=overlap_threshold,
        strict_inconsistent=strict_inconsistent,
    ):
        event = decoder.advance(
            start_sample=start,
            end_sample=end,
            evidence_frontier_sample=max(observation.evidence_frontier_sample, end),
            replacement_evidence=replacement,
            pause=pause,
            lifecycle=(AnchorLifecycle.ANCHORED if valid else AnchorLifecycle.ANCHOR_UNCERTAIN),
            anchor_id=anchor_id if valid else None,
            anchor_episode_id=episode_id if valid else None,
        )
        if event is not None:
            return event
    return None


def _contamination_samples(
    intervals: Sequence[Any],
    interval_ends: Sequence[int],
    *,
    anchor_speaker: str,
    start_sample: int,
    end_sample: int,
) -> int:
    total = 0
    index = bisect_right(interval_ends, start_sample)
    while index < len(intervals):
        interval = intervals[index]
        if interval.start_sample >= end_sample:
            break
        start = max(start_sample, interval.start_sample)
        end = min(end_sample, interval.end_sample)
        if (
            end > start
            and not interval.masked
            and anchor_speaker not in interval.active_speakers
            and bool(interval.active_speakers)
        ):
            total += end - start
        index += 1
    return total


def _product_event_metrics(
    *,
    predicted_events: Sequence[ReplacementEvent],
    reference: Any,
    intervals: Sequence[Any],
    contamination_episodes: Sequence[tuple[str, int, int]],
    tolerance_samples: int,
) -> dict[str, Any]:
    predicted = sorted(predicted_events, key=lambda value: value.boundary_source_sample)
    references = sorted(reference.events, key=lambda value: value.boundary_source_sample)
    matches = monotonic_boundary_matches(
        [value.boundary_source_sample for value in predicted],
        [value.boundary_source_sample for value in references],
        tolerance_samples,
    )
    matched_predicted = {left for left, _ in matches}
    matched_references = {right for _, right in matches}
    emit_delays = [
        (predicted[left].decoder_emit_sample - references[right].boundary_source_sample)
        * 1000.0
        / 16000.0
        for left, right in matches
    ]
    evidence_delays = [
        (predicted[left].model_evidence_frontier_sample - references[right].boundary_source_sample)
        * 1000.0
        / 16000.0
        for left, right in matches
    ]
    boundary_errors = [
        (predicted[left].boundary_source_sample - references[right].boundary_source_sample)
        * 1000.0
        / 16000.0
        for left, right in matches
    ]
    predicted_by_reference = {right: predicted[left] for left, right in matches}
    interval_ends = [value.end_sample for value in intervals]
    contamination_per_replacement = []
    scored_end_sample = intervals[-1].end_sample
    for index, reference_event in enumerate(references):
        next_boundary = (
            references[index + 1].boundary_source_sample
            if index + 1 < len(references)
            else scored_end_sample
        )
        predicted_event = predicted_by_reference.get(index)
        stop = (
            min(predicted_event.decoder_emit_sample, next_boundary)
            if predicted_event is not None
            else next_boundary
        )
        start = reference_event.boundary_source_sample
        contamination_per_replacement.append(
            _contamination_samples(
                intervals,
                interval_ends,
                anchor_speaker=reference_event.anchor_id,
                start_sample=start,
                end_sample=max(start, stop),
            )
            / 16000.0
        )
    logical_episode_contamination = sum(
        _contamination_samples(
            intervals,
            interval_ends,
            anchor_speaker=anchor,
            start_sample=start,
            end_sample=end,
        )
        for anchor, start, end in contamination_episodes
    )
    active_samples = sum(
        value.end_sample - value.start_sample for value in intervals if value.active_speakers
    )
    active_hours = active_samples / 16000.0 / 3600.0
    contamination_seconds = sum(contamination_per_replacement)
    return {
        "predicted_cut_count": len(predicted),
        "reference_replacement_count": len(references),
        "matched_replacement_count": len(matches),
        "false_cut_count": len(predicted) - len(matched_predicted),
        "missed_replacement_count": len(references) - len(matched_references),
        "speaker_induced_cut_count_per_active_speech_hour": (
            len(predicted) / active_hours if active_hours else None
        ),
        "active_speech_seconds": active_samples / 16000.0,
        "exclusive_other_contamination_seconds": contamination_seconds,
        "exclusive_other_contamination_seconds_per_active_speech_hour": (
            contamination_seconds / active_hours if active_hours else None
        ),
        "logical_episode_exclusive_other_contamination_seconds": (
            logical_episode_contamination / 16000.0
        ),
        "contamination_seconds_per_true_replacement": {
            "p50": percentile(contamination_per_replacement, 50),
            "p90": percentile(contamination_per_replacement, 90),
        },
        "replacement_emit_delay_ms": {
            "p50": percentile(emit_delays, 50),
            "p90": percentile(emit_delays, 90),
        },
        "model_evidence_delay_ms": {
            "p50": percentile(evidence_delays, 50),
            "p90": percentile(evidence_delays, 90),
        },
        "backdated_boundary_error_ms": {
            "p50": percentile(boundary_errors, 50),
            "p90": percentile(boundary_errors, 90),
        },
        "replacement_emit_delay_values_ms": emit_delays,
        "model_evidence_delay_values_ms": evidence_delays,
        "backdated_boundary_error_values_ms": boundary_errors,
        "contamination_values_seconds_per_true_replacement": contamination_per_replacement,
        "matches": [
            {
                "predicted_index": left,
                "reference_index": right,
                "predicted_boundary_sample": predicted[left].boundary_source_sample,
                "reference_boundary_sample": references[right].boundary_source_sample,
            }
            for left, right in matches
        ],
    }


def _linear_fail_closed_exposure(
    *,
    intervals: Sequence[Any],
    anchored_ranges: Sequence[tuple[int, int]],
    uncertain_ranges: Sequence[tuple[int, int]],
    exact_contamination_seconds: float,
) -> dict[str, float]:
    anchored = _merge_ranges(anchored_ranges)
    uncertain = _merge_ranges(uncertain_ranges)
    anchored_index = 0
    uncertain_index = 0
    masked_samples = 0
    masked_active_samples = 0
    unanchored_active_samples = 0
    uncertain_active_samples = 0
    for interval in intervals:
        start = int(interval.start_sample)
        end = int(interval.end_sample)
        duration = end - start
        if interval.masked:
            masked_samples += duration
            if interval.active_speakers:
                masked_active_samples += duration
            continue
        if not interval.active_speakers:
            continue
        while anchored_index < len(anchored) and anchored[anchored_index][1] <= start:
            anchored_index += 1
        anchored_samples = 0
        local_index = anchored_index
        while local_index < len(anchored) and anchored[local_index][0] < end:
            left, right = anchored[local_index]
            anchored_samples += max(0, min(end, right) - max(start, left))
            local_index += 1
        while uncertain_index < len(uncertain) and uncertain[uncertain_index][1] <= start:
            uncertain_index += 1
        uncertain_samples = 0
        local_index = uncertain_index
        while local_index < len(uncertain) and uncertain[local_index][0] < end:
            left, right = uncertain[local_index]
            uncertain_samples += max(0, min(end, right) - max(start, left))
            local_index += 1
        uncertain_active_samples += uncertain_samples
        unanchored_active_samples += max(
            0, duration - anchored_samples - uncertain_samples
        )
    unknown = masked_active_samples + unanchored_active_samples + uncertain_active_samples
    return {
        "masked_seconds": masked_samples / 16000.0,
        "masked_active_speech_seconds": masked_active_samples / 16000.0,
        "unanchored_active_speech_seconds": unanchored_active_samples / 16000.0,
        "anchor_uncertain_active_speech_seconds": uncertain_active_samples / 16000.0,
        "fail_closed_unknown_active_speech_seconds": unknown / 16000.0,
        "exclusive_other_contamination_upper_bound_seconds": (
            exact_contamination_seconds
            + unanchored_active_samples / 16000.0
            + uncertain_active_samples / 16000.0
        ),
    }


def _oracle_product_session(
    *,
    row: dict[str, Any],
    reference: Any,
    observations: Sequence[ModelObservation],
    mappings: dict[str, OracleAnchorMapping],
    candidate: str,
    anchor_threshold: float,
    overlap_threshold: float | None,
    strict_inconsistent: bool,
    confirmation_samples: int,
    tolerance_samples: int,
) -> tuple[dict[str, Any], list[ReplacementEvent]]:
    events: list[ReplacementEvent] = []
    contamination_episodes: list[tuple[str, int, int]] = []
    anchored_ranges: list[tuple[int, int]] = []
    uncertain_ranges: list[tuple[int, int]] = []
    for episode in reference.episodes:
        mapping = mappings.get(episode.episode_id)
        if mapping is None:
            uncertain_ranges.append((episode.anchor_emit_sample, episode.end_emit_sample))
            continue
        event = _decode_episode(
            source_id=str(row["source_id"]),
            episode_id=episode.episode_id,
            anchor_id=episode.anchor_speaker,
            anchor_slot_index=mapping.slot_index,
            episode_start=episode.anchor_emit_sample,
            episode_end=episode.end_emit_sample,
            observations=observations,
            candidate=candidate,
            anchor_threshold=anchor_threshold,
            overlap_threshold=overlap_threshold,
            strict_inconsistent=strict_inconsistent,
            confirmation_samples=confirmation_samples,
        )
        if event is not None:
            events.append(event)
        end = min(
            episode.end_emit_sample,
            event.decoder_emit_sample if event is not None else episode.end_emit_sample,
        )
        contamination_episodes.append((episode.anchor_speaker, episode.anchor_emit_sample, end))
        episode_anchored, episode_uncertain = _episode_exposure_ranges(
            observations=observations,
            episode_start=episode.anchor_emit_sample,
            episode_end=end,
            candidate=candidate,
            anchor_slot_index=mapping.slot_index,
            anchor_threshold=anchor_threshold,
            overlap_threshold=overlap_threshold,
            strict_inconsistent=strict_inconsistent,
        )
        anchored_ranges.extend(episode_anchored)
        uncertain_ranges.extend(episode_uncertain)
    intervals = intervals_from_manifest(row)
    metrics = _product_event_metrics(
        predicted_events=events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=contamination_episodes,
        tolerance_samples=tolerance_samples,
    )
    metrics.update(
        _linear_fail_closed_exposure(
            intervals=intervals,
            anchored_ranges=anchored_ranges,
            uncertain_ranges=uncertain_ranges,
            exact_contamination_seconds=metrics["exclusive_other_contamination_seconds"],
        )
    )
    unmasked_active = metrics["active_speech_seconds"] - metrics["masked_active_speech_seconds"]
    metrics["speaker_protection_enabled_fraction"] = (
        max(
            0.0,
            unmasked_active
            - metrics["unanchored_active_speech_seconds"]
            - metrics["anchor_uncertain_active_speech_seconds"],
        )
        / unmasked_active
        if unmasked_active
        else None
    )
    return metrics, events


def _event_from_dict(value: dict[str, Any]) -> ReplacementEvent:
    return ReplacementEvent(
        source_id=str(value["source_id"]),
        anchor_episode_id=str(value["anchor_episode_id"]),
        anchor_id=str(value["anchor_id"]),
        boundary_source_sample=int(value["boundary_source_sample"]),
        model_evidence_frontier_sample=int(value["model_evidence_frontier_sample"]),
        decoder_emit_sample=int(value["decoder_emit_sample"]),
        compute_lag_ms=(
            None if value.get("compute_lag_ms") is None else float(value["compute_lag_ms"])
        ),
        confirmation_samples=int(value["confirmation_samples"]),
    )


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _causal_exposure(
    row: dict[str, Any], candidate_uncertain_ranges: Sequence[tuple[int, int]] = ()
) -> dict[str, float]:
    masked = 0
    masked_active = 0
    unanchored_active = 0
    uncertain_active = 0
    anchored_active = 0
    uncertain_ranges = _merge_ranges(candidate_uncertain_ranges)
    uncertain_index = 0
    for span in row["timeline"]:
        span_start = int(span["start_sample"])
        span_end = int(span["end_sample"])
        duration = span_end - span_start
        if bool(span["masked"]):
            masked += duration
            if bool(span["speech_present"]):
                masked_active += duration
            continue
        if not bool(span["speech_present"]):
            continue
        lifecycle = str(span["lifecycle"])
        if lifecycle == AnchorLifecycle.ANCHORED.value:
            while (
                uncertain_index < len(uncertain_ranges)
                and uncertain_ranges[uncertain_index][1] <= span_start
            ):
                uncertain_index += 1
            candidate_uncertain = 0
            local_index = uncertain_index
            while (
                local_index < len(uncertain_ranges)
                and uncertain_ranges[local_index][0] < span_end
            ):
                start, end = uncertain_ranges[local_index]
                candidate_uncertain += max(0, min(span_end, end) - max(span_start, start))
                local_index += 1
            anchored_active += duration - candidate_uncertain
            uncertain_active += candidate_uncertain
        elif lifecycle == AnchorLifecycle.ANCHOR_UNCERTAIN.value:
            uncertain_active += duration
        else:
            unanchored_active += duration
    active = anchored_active + uncertain_active + unanchored_active
    return {
        "masked_seconds": masked / 16000.0,
        "masked_active_speech_seconds": masked_active / 16000.0,
        "unanchored_active_speech_seconds": unanchored_active / 16000.0,
        "anchor_uncertain_active_speech_seconds": uncertain_active / 16000.0,
        "fail_closed_unknown_active_speech_seconds": (
            masked_active + unanchored_active + uncertain_active
        )
        / 16000.0,
        "speaker_protection_enabled_fraction": (anchored_active / active if active else None),
    }


def _causal_safety(
    annotations: Sequence[dict[str, Any]],
    events: Sequence[ReplacementEvent],
    matches: Sequence[dict[str, Any]],
    expected_opportunity_count: int,
) -> dict[str, Any]:
    predicted = sorted(events, key=lambda value: value.boundary_source_sample)
    matched_episode_ids = {
        predicted[int(value["predicted_index"])].anchor_episode_id for value in matches
    }
    events_by_episode = {value.anchor_episode_id: value for value in events}
    wrong_false_cuts = 0
    cascades: list[int] = []
    active_cascade = 0
    for annotation in sorted(annotations, key=lambda value: int(value["anchor_emit_sample"])):
        event = events_by_episode.get(str(annotation["episode_id"]))
        if (
            event is not None
            and not bool(annotation["correct_anchor"])
            and event.anchor_episode_id not in matched_episode_ids
        ):
            wrong_false_cuts += 1
            active_cascade += 1
        elif active_cascade:
            cascades.append(active_cascade)
            active_cascade = 0
    if active_cascade:
        cascades.append(active_cascade)
    total = len(annotations)
    wrong = sum(not bool(value["correct_anchor"]) for value in annotations)
    matched_enrollments = sum(
        value.get("opportunity_start_sample") is not None for value in annotations
    )
    failures = max(expected_opportunity_count - matched_enrollments, 0)
    return {
        "total_enrollment_count": total,
        "expected_opportunity_count": expected_opportunity_count,
        "wrong_anchor_count": wrong,
        "wrong_anchor_rate": wrong / total if total else None,
        "enrollment_failure_count": failures,
        "enrollment_failure_rate": (
            failures / expected_opportunity_count if expected_opportunity_count else None
        ),
        "false_cuts_after_wrong_anchor": wrong_false_cuts,
        "anchor_error_cascade_length": {
            "maximum": max(cascades, default=0),
            "p50": percentile(cascades, 50),
            "p90": percentile(cascades, 90),
            "distribution": dict(sorted(Counter(cascades).items())),
        },
    }


def _causal_product_session(
    *,
    row: dict[str, Any],
    reference: Any,
    observations: Sequence[ModelObservation],
    gate2_row: dict[str, Any],
    candidate: str,
    anchor_threshold: float,
    overlap_threshold: float | None,
    strict_inconsistent: bool,
    confirmation_samples: int,
    tolerance_samples: int,
) -> tuple[dict[str, Any], list[ReplacementEvent]]:
    events: list[ReplacementEvent] = []
    contamination_episodes: list[tuple[str, int, int]] = []
    candidate_uncertain_ranges: list[tuple[int, int]] = []
    for annotation in gate2_row["annotated_episodes"]:
        anchor_speaker = annotation.get("expected_anchor_speaker")
        if anchor_speaker is None:
            continue
        start = int(annotation["anchor_emit_sample"])
        old_end = int(annotation["end_emit_sample"])
        event = _decode_episode(
            source_id=str(row["source_id"]),
            episode_id=str(annotation["episode_id"]),
            anchor_id=str(annotation["anchor_slot_id"]),
            anchor_slot_index=int(annotation["anchor_slot_index"]),
            episode_start=start,
            episode_end=old_end,
            observations=observations,
            candidate=candidate,
            anchor_threshold=anchor_threshold,
            overlap_threshold=overlap_threshold,
            strict_inconsistent=strict_inconsistent,
            confirmation_samples=confirmation_samples,
        )
        if event is not None:
            events.append(event)
        end = min(old_end, event.decoder_emit_sample if event is not None else old_end)
        contamination_episodes.append((str(anchor_speaker), start, end))
        _, episode_uncertain = _episode_exposure_ranges(
            observations=observations,
            episode_start=start,
            episode_end=end,
            candidate=candidate,
            anchor_slot_index=int(annotation["anchor_slot_index"]),
            anchor_threshold=anchor_threshold,
            overlap_threshold=overlap_threshold,
            strict_inconsistent=strict_inconsistent,
        )
        candidate_uncertain_ranges.extend(episode_uncertain)
    intervals = intervals_from_manifest(row)
    metrics = _product_event_metrics(
        predicted_events=events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=contamination_episodes,
        tolerance_samples=tolerance_samples,
    )
    exposure = _causal_exposure(gate2_row, candidate_uncertain_ranges)
    metrics.update(exposure)
    metrics["exclusive_other_contamination_upper_bound_seconds"] = (
        metrics["exclusive_other_contamination_seconds"]
        + exposure["unanchored_active_speech_seconds"]
        + exposure["anchor_uncertain_active_speech_seconds"]
    )
    metrics.update(
        _causal_safety(
            gate2_row["annotated_episodes"],
            events,
            metrics["matches"],
            int(gate2_row["expected_opportunity_count"]),
        )
    )
    metrics["causal_interpretation"] = "fixed-issue-97-lifecycle-counterfactual-ablation"
    return metrics, events


def _r0_oracle_session(
    *, row: dict[str, Any], reference: Any, gate1_row: dict[str, Any], tolerance_samples: int
) -> tuple[dict[str, Any], list[ReplacementEvent]]:
    events = [_event_from_dict(value) for value in gate1_row["events"]]
    events_by_episode = {value.anchor_episode_id: value for value in events}
    mapped = {str(value["anchor_episode_id"]) for value in gate1_row["oracle_mappings"]}
    contamination = []
    anchored = []
    uncertain = []
    for episode in reference.episodes:
        if episode.episode_id not in mapped:
            uncertain.append((episode.anchor_emit_sample, episode.end_emit_sample))
            continue
        event = events_by_episode.get(episode.episode_id)
        end = min(
            episode.end_emit_sample,
            event.decoder_emit_sample if event is not None else episode.end_emit_sample,
        )
        contamination.append((episode.anchor_speaker, episode.anchor_emit_sample, end))
        anchored.append((episode.anchor_emit_sample, end))
    intervals = intervals_from_manifest(row)
    metrics = _product_event_metrics(
        predicted_events=events,
        reference=reference,
        intervals=intervals,
        contamination_episodes=contamination,
        tolerance_samples=tolerance_samples,
    )
    metrics.update(
        gate1_fail_closed_exposure(
            intervals=intervals,
            anchored_ranges=anchored,
            uncertain_ranges=uncertain,
            exact_contamination_seconds=metrics["exclusive_other_contamination_seconds"],
        )
    )
    unmasked_active = metrics["active_speech_seconds"] - metrics["masked_active_speech_seconds"]
    metrics["speaker_protection_enabled_fraction"] = (
        max(
            0.0,
            unmasked_active
            - metrics["unanchored_active_speech_seconds"]
            - metrics["anchor_uncertain_active_speech_seconds"],
        )
        / unmasked_active
        if unmasked_active
        else None
    )
    return metrics, events


def _r0_causal_session(
    *, row: dict[str, Any], reference: Any, gate2_row: dict[str, Any], tolerance_samples: int
) -> tuple[dict[str, Any], list[ReplacementEvent]]:
    events = [_event_from_dict(value) for value in gate2_row["events"]]
    events_by_episode = {value.anchor_episode_id: value for value in events}
    contamination = []
    for annotation in gate2_row["annotated_episodes"]:
        anchor = annotation.get("expected_anchor_speaker")
        if anchor is None:
            continue
        event = events_by_episode.get(str(annotation["episode_id"]))
        start = int(annotation["anchor_emit_sample"])
        end = min(
            int(annotation["end_emit_sample"]),
            event.decoder_emit_sample if event is not None else int(annotation["end_emit_sample"]),
        )
        contamination.append((str(anchor), start, end))
    metrics = _product_event_metrics(
        predicted_events=events,
        reference=reference,
        intervals=intervals_from_manifest(row),
        contamination_episodes=contamination,
        tolerance_samples=tolerance_samples,
    )
    exposure = _causal_exposure(gate2_row)
    metrics.update(exposure)
    metrics["exclusive_other_contamination_upper_bound_seconds"] = (
        metrics["exclusive_other_contamination_seconds"]
        + exposure["unanchored_active_speech_seconds"]
        + exposure["anchor_uncertain_active_speech_seconds"]
    )
    metrics.update(
        _causal_safety(
            gate2_row["annotated_episodes"],
            events,
            metrics["matches"],
            int(gate2_row["expected_opportunity_count"]),
        )
    )
    return metrics, events


def _global_overlap_records(
    row: dict[str, Any], cells: Sequence[PosteriorCell]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    total_unmasked = 0
    invalid = 0
    total_support = 0
    invalid_support = 0
    invalid_active = 0
    for posterior in cells:
        if posterior.cell.masked:
            continue
        duration = posterior.cell.end_sample - posterior.cell.start_sample
        total_unmasked += 1
        total_support += duration
        if not posterior.trace_valid:
            invalid += 1
            invalid_support += duration
            if posterior.cell.active_speakers:
                invalid_active += duration
            continue
        alive_scores = sorted(
            (
                float(score)
                for score, alive in zip(posterior.probabilities, posterior.slot_alive, strict=True)
                if alive
            ),
            reverse=True,
        )
        score = alive_scores[1] if len(alive_scores) >= 2 else 0.0
        records.append(
            {
                "source_id": str(row["source_id"]),
                "start_sample": posterior.cell.start_sample,
                "end_sample": posterior.cell.end_sample,
                "label": len(posterior.cell.active_speakers) >= 2,
                "score": score,
            }
        )
    return records, {
        "source_id": str(row["source_id"]),
        "total_unmasked_cell_count": total_unmasked,
        "scored_cell_count": len(records),
        "invalid_cell_count": invalid,
        "total_unmasked_support_seconds": total_support / 16000.0,
        "invalid_support_seconds": invalid_support / 16000.0,
        "invalid_active_speech_seconds": invalid_active / 16000.0,
    }


def _overlap_runs(row: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = row["intervals"]
    runs: list[dict[str, Any]] = []
    active_start: int | None = None
    active_end: int | None = None
    start_index: int | None = None
    for index, interval in enumerate(intervals):
        overlap = not bool(interval["masked"]) and len(interval["active_speakers"]) >= 2
        if overlap and active_start is None:
            active_start = int(interval["start_sample"])
            active_end = int(interval["end_sample"])
            start_index = index
        elif overlap and active_end == int(interval["start_sample"]):
            active_end = int(interval["end_sample"])
        elif active_start is not None:
            runs.append(
                {
                    "start_sample": active_start,
                    "end_sample": int(active_end),
                    "start_index": int(start_index),
                    "end_index": index - 1,
                }
            )
            active_start = int(interval["start_sample"]) if overlap else None
            active_end = int(interval["end_sample"]) if overlap else None
            start_index = index if overlap else None
    if active_start is not None:
        runs.append(
            {
                "start_sample": active_start,
                "end_sample": int(active_end),
                "start_index": int(start_index),
                "end_index": len(intervals) - 1,
            }
        )
    for run in runs:
        before = intervals[run["start_index"] - 1] if run["start_index"] > 0 else None
        after = intervals[run["end_index"] + 1] if run["end_index"] + 1 < len(intervals) else None
        run["short_backchannel"] = run["end_sample"] - run["start_sample"] <= 16000 and any(
            value is not None and not bool(value["masked"]) and len(value["active_speakers"]) == 1
            for value in (before, after)
        )
    return runs


def _global_overlap_metrics(
    manifest: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    coverage_rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    primary_threshold: float,
) -> dict[str, Any]:
    labels = [bool(value["label"]) for value in records]
    scores = [float(value["score"]) for value in records]
    weights = [float(int(value["end_sample"]) - int(value["start_sample"])) for value in records]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for value in records:
        by_source.setdefault(str(value["source_id"]), []).append(value)
    operating_points = []
    for threshold in thresholds:
        confusion = weighted_binary_confusion(labels, scores, weights, threshold)
        false_samples = sum(
            int(value["end_sample"]) - int(value["start_sample"])
            for value in records
            if not bool(value["label"]) and float(value["score"]) >= threshold
        )
        buckets = {
            "lt_500ms": [0, 8000],
            "500_to_1500ms": [8000, 24000],
            "gte_1500ms": [24000, None],
        }
        bucket_counts = {key: {"event_count": 0, "recalled_event_count": 0} for key in buckets}
        short_count = 0
        short_recalled = 0
        for row in manifest:
            source_records = by_source.get(str(row["source_id"]), [])
            for run in _overlap_runs(row):
                duration = int(run["end_sample"]) - int(run["start_sample"])
                recalled = any(
                    int(value["end_sample"]) > int(run["start_sample"])
                    and int(value["start_sample"]) < int(run["end_sample"])
                    and float(value["score"]) >= threshold
                    for value in source_records
                )
                for key, (lower, upper) in buckets.items():
                    if duration >= lower and (upper is None or duration < upper):
                        bucket_counts[key]["event_count"] += 1
                        bucket_counts[key]["recalled_event_count"] += int(recalled)
                        break
                if bool(run["short_backchannel"]):
                    short_count += 1
                    short_recalled += int(recalled)
        for value in bucket_counts.values():
            value["event_recall"] = (
                value["recalled_event_count"] / value["event_count"]
                if value["event_count"]
                else None
            )
        operating_points.append(
            {
                "threshold": threshold,
                "duration_weighted_confusion": confusion,
                "false_overlap_duration_seconds": false_samples / 16000.0,
                "overlap_event_recall_by_duration_bucket": bucket_counts,
                "short_backchannel_overlap": {
                    "event_count": short_count,
                    "recalled_event_count": short_recalled,
                    "event_recall": (short_recalled / short_count if short_count else None),
                },
            }
        )
    total_cells = sum(int(value["total_unmasked_cell_count"]) for value in coverage_rows)
    scored_cells = sum(int(value["scored_cell_count"]) for value in coverage_rows)
    total_support = sum(float(value["total_unmasked_support_seconds"]) for value in coverage_rows)
    invalid_support = sum(float(value["invalid_support_seconds"]) for value in coverage_rows)
    return {
        "score": "second_highest_alive_speaker_activity_probability",
        "record_count": len(records),
        "global_overlap_auprc": weighted_average_precision(labels, scores, weights),
        "global_overlap_pr_curve": weighted_binary_pr_curve(labels, scores, weights, thresholds),
        "primary_threshold": primary_threshold,
        "operating_points": operating_points,
        "coverage": {
            "source_count": len(coverage_rows),
            "total_unmasked_cell_count": total_cells,
            "scored_cell_count": scored_cells,
            "invalid_cell_count": sum(int(value["invalid_cell_count"]) for value in coverage_rows),
            "total_unmasked_support_seconds": total_support,
            "invalid_support_seconds": invalid_support,
            "invalid_active_speech_seconds": sum(
                float(value["invalid_active_speech_seconds"]) for value in coverage_rows
            ),
            "scored_support_fraction": (
                (total_support - invalid_support) / total_support if total_support else None
            ),
            "per_source": list(coverage_rows),
        },
    }


def _topology_window(
    row: dict[str, Any],
    episode: dict[str, Any],
    transitions: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, int] | None:
    transitions = transitions or {value["transition_id"]: value for value in row["transitions"]}
    episode_transitions = [
        transitions[value] for value in episode["transition_ids"] if value in transitions
    ]
    indices = [
        int(index)
        for transition in episode_transitions
        for index in (
            transition.get("from_interval_index"),
            transition.get("to_interval_index"),
        )
        if index is not None
    ]
    if not indices:
        return None
    return (
        int(row["intervals"][min(indices)]["start_sample"]),
        int(row["intervals"][max(indices)]["end_sample"]),
    )


def _session_topology(
    row: dict[str, Any],
    predicted: Sequence[ReplacementEvent],
    reference: Any,
    tolerance_samples: int,
) -> dict[str, Any]:
    result: dict[str, dict[str, int]] = {}
    predicted_ordered = sorted(predicted, key=lambda value: value.boundary_source_sample)
    reference_ordered = sorted(reference.events, key=lambda value: value.boundary_source_sample)
    predicted_boundaries = [value.boundary_source_sample for value in predicted_ordered]
    reference_boundaries = [value.boundary_source_sample for value in reference_ordered]
    transitions = {value["transition_id"]: value for value in row["transitions"]}
    for episode in row["topology_episodes"]:
        if not bool(episode.get("coverage_gate_eligible", False)):
            continue
        window = _topology_window(row, episode, transitions)
        if window is None:
            continue
        start, end = window
        topology = str(episode["primary_topology"])
        values = result.setdefault(
            topology,
            {
                "eligible_episode_count": 0,
                "episodes_with_predicted_cut": 0,
                "episodes_with_reference_replacement": 0,
                "episodes_with_aligned_cut": 0,
            },
        )
        values["eligible_episode_count"] += 1
        predicted_window = predicted_ordered[
            bisect_left(predicted_boundaries, start) : bisect_left(predicted_boundaries, end)
        ]
        reference_window = reference_ordered[
            bisect_left(reference_boundaries, start) : bisect_left(reference_boundaries, end)
        ]
        values["episodes_with_predicted_cut"] += int(bool(predicted_window))
        values["episodes_with_reference_replacement"] += int(bool(reference_window))
        values["episodes_with_aligned_cut"] += int(
            any(
                0 <= left.boundary_source_sample - right.boundary_source_sample <= tolerance_samples
                for left in predicted_window
                for right in reference_window
            )
        )
    return result


def _aggregate_topology(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, dict[str, int]] = {}
    for row in rows:
        for topology, values in row["topology"].items():
            target = counters.setdefault(
                topology,
                {
                    "eligible_episode_count": 0,
                    "episodes_with_predicted_cut": 0,
                    "episodes_with_reference_replacement": 0,
                    "episodes_with_aligned_cut": 0,
                },
            )
            for key in target:
                target[key] += int(values[key])
    result = {}
    for topology, values in sorted(counters.items()):
        count = values["eligible_episode_count"]
        result[topology] = {
            **values,
            "overlap_return_preservation_rate": (
                1.0 - values["episodes_with_predicted_cut"] / count
                if topology == "overlap_return" and count
                else None
            ),
            "overlap_takeover_success_rate": (
                values["episodes_with_aligned_cut"] / count
                if topology == "overlap_takeover" and count
                else None
            ),
        }
    return result


def _aggregate_product(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = [value["metrics"] for value in rows]
    active_seconds = sum(float(value["active_speech_seconds"]) for value in metrics)
    active_hours = active_seconds / 3600.0
    contamination = sum(float(value["exclusive_other_contamination_seconds"]) for value in metrics)
    contamination_upper = sum(
        float(value["exclusive_other_contamination_upper_bound_seconds"]) for value in metrics
    )
    emit = [float(item) for value in metrics for item in value["replacement_emit_delay_values_ms"]]
    boundary = [
        float(item) for value in metrics for item in value["backdated_boundary_error_values_ms"]
    ]
    result = {
        "source_count": len(rows),
        "active_speech_hours": active_hours,
        "predicted_cut_count": sum(int(value["predicted_cut_count"]) for value in metrics),
        "reference_replacement_count": sum(
            int(value["reference_replacement_count"]) for value in metrics
        ),
        "matched_replacement_count": sum(
            int(value["matched_replacement_count"]) for value in metrics
        ),
        "false_cut_count": sum(int(value["false_cut_count"]) for value in metrics),
        "missed_replacement_count": sum(
            int(value["missed_replacement_count"]) for value in metrics
        ),
        "speaker_induced_cut_count_per_active_speech_hour": (
            sum(int(value["predicted_cut_count"]) for value in metrics) / active_hours
            if active_hours
            else None
        ),
        "exclusive_other_contamination_seconds": contamination,
        "exclusive_other_contamination_upper_bound_seconds": contamination_upper,
        "exclusive_other_contamination_seconds_per_active_speech_hour": (
            contamination / active_hours if active_hours else None
        ),
        "masked_seconds": sum(float(value["masked_seconds"]) for value in metrics),
        "masked_active_speech_seconds": sum(
            float(value["masked_active_speech_seconds"]) for value in metrics
        ),
        "unanchored_active_speech_seconds": sum(
            float(value["unanchored_active_speech_seconds"]) for value in metrics
        ),
        "anchor_uncertain_active_speech_seconds": sum(
            float(value["anchor_uncertain_active_speech_seconds"]) for value in metrics
        ),
        "replacement_emit_delay_ms": {
            "p50": percentile(emit, 50),
            "p90": percentile(emit, 90),
        },
        "backdated_boundary_error_ms": {
            "p50": percentile(boundary, 50),
            "p90": percentile(boundary, 90),
        },
        "topology": _aggregate_topology(rows),
    }
    active_unmasked = active_seconds - result["masked_active_speech_seconds"]
    result["speaker_protection_enabled_fraction"] = (
        max(
            0.0,
            active_unmasked
            - result["unanchored_active_speech_seconds"]
            - result["anchor_uncertain_active_speech_seconds"],
        )
        / active_unmasked
        if active_unmasked
        else None
    )
    if all("wrong_anchor_count" in value for value in metrics):
        total_enrollments = sum(int(value["total_enrollment_count"]) for value in metrics)
        expected_opportunities = sum(int(value["expected_opportunity_count"]) for value in metrics)
        wrong = sum(int(value["wrong_anchor_count"]) for value in metrics)
        failures = sum(int(value["enrollment_failure_count"]) for value in metrics)
        result["total_enrollment_count"] = total_enrollments
        result["expected_opportunity_count"] = expected_opportunities
        result["wrong_anchor_count"] = wrong
        result["wrong_anchor_rate"] = wrong / total_enrollments if total_enrollments else None
        result["enrollment_failure_count"] = failures
        result["enrollment_failure_rate"] = (
            failures / expected_opportunities if expected_opportunities else None
        )
        result["false_cuts_after_wrong_anchor"] = sum(
            int(value["false_cuts_after_wrong_anchor"]) for value in metrics
        )
        cascade_lengths = [
            int(length)
            for value in metrics
            for length, count in value["anchor_error_cascade_length"]["distribution"].items()
            for _ in range(int(count))
        ]
        result["anchor_error_cascade_length"] = {
            "maximum": max(cascade_lengths, default=0),
            "p50": percentile(cascade_lengths, 50),
            "p90": percentile(cascade_lengths, 90),
            "distribution": dict(sorted(Counter(cascade_lengths).items())),
        }
    return result


def _primary_candidate(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if row["candidate"] == "r0_relative_occupancy":
        return True
    if row["candidate"] == "simple_anchor":
        return row["variant"] == "primary"
    primary = cfg["candidate_b"]["threshold_grids"][row["family"]]["primary"]
    return (
        row["variant"] == "primary"
        and row["anchor_threshold"] == float(primary[0])
        and row["overlap_threshold"] == float(primary[1])
    )


def _gt_challenger_event(
    *, row: dict[str, Any], episode: Any, candidate: str, confirmation_samples: int
) -> ReplacementEvent | None:
    decoder = SimplifiedReplacementDecoder(str(row["source_id"]), confirmation_samples)
    for interval in intervals_from_manifest(row):
        start = max(interval.start_sample, episode.anchor_emit_sample)
        end = min(interval.end_sample, episode.end_emit_sample)
        if end <= start:
            continue
        speech_present = bool(interval.active_speakers)
        anchor_present = episode.anchor_speaker in interval.active_speakers
        if candidate == "simple_anchor":
            state = derive_gt_simple_anchor_state(
                speech_present=speech_present, anchor_present=anchor_present
            )
            replacement = state is SimpleAnchorState.NON_ANCHOR_SPEECH
        elif candidate == "anchor_overlap":
            state = derive_gt_anchor_overlap_state(
                speech_present=speech_present,
                anchor_present=anchor_present,
                anchor_overlap_present=(
                    anchor_present
                    and any(value != episode.anchor_speaker for value in interval.active_speakers)
                ),
            )
            replacement = state is AnchorOverlapState.NON_ANCHOR_SPEECH
        else:
            raise ValueError(f"unknown challenger: {candidate}")
        event = decoder.advance(
            start_sample=start,
            end_sample=end,
            evidence_frontier_sample=end,
            replacement_evidence=replacement,
            pause=interval.masked,
            lifecycle=AnchorLifecycle.ANCHORED,
            anchor_id=episode.anchor_speaker,
            anchor_episode_id=episode.episode_id,
        )
        if event is not None:
            return event
    return None


def _write_s0(manifest: Sequence[dict[str, Any]], cfg: dict[str, Any], result_dir: Path) -> None:
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    cells = []
    candidate_a_all_equal = True
    candidate_b_all_equal = True
    for persistence in cfg["replacement_confirm_ms"]:
        confirmation_samples = int(persistence) * 16
        source_count = 0
        episode_count = 0
        reference_event_count = 0
        candidate_a_event_count = 0
        candidate_b_event_count = 0
        candidate_a_mismatch_count = 0
        candidate_b_mismatch_count = 0
        mismatches = []
        for row in manifest:
            source_count += 1
            reference = gt_reference_session(
                row,
                replacement_confirmation_samples=confirmation_samples,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            reference_by_episode = {value.anchor_episode_id: value for value in reference.events}
            episode_count += len(reference.episodes)
            reference_event_count += len(reference.events)
            for episode in reference.episodes:
                candidate_a = _gt_challenger_event(
                    row=row,
                    episode=episode,
                    candidate="simple_anchor",
                    confirmation_samples=confirmation_samples,
                )
                candidate_b = _gt_challenger_event(
                    row=row,
                    episode=episode,
                    candidate="anchor_overlap",
                    confirmation_samples=confirmation_samples,
                )
                expected = reference_by_episode.get(episode.episode_id)
                candidate_a_event_count += int(candidate_a is not None)
                candidate_b_event_count += int(candidate_b is not None)
                signatures = [
                    None
                    if value is None
                    else (
                        value.boundary_source_sample,
                        value.decoder_emit_sample,
                        value.anchor_episode_id,
                    )
                    for value in (expected, candidate_a, candidate_b)
                ]
                candidate_a_mismatch_count += int(signatures[0] != signatures[1])
                candidate_b_mismatch_count += int(signatures[0] != signatures[2])
                if signatures[0] != signatures[1] or signatures[0] != signatures[2]:
                    mismatches.append(
                        {
                            "source_id": row["source_id"],
                            "anchor_episode_id": episode.episode_id,
                            "reference": signatures[0],
                            "candidate_a": signatures[1],
                            "candidate_b": signatures[2],
                        }
                    )
        candidate_a_all_equal = candidate_a_all_equal and candidate_a_mismatch_count == 0
        candidate_b_all_equal = candidate_b_all_equal and candidate_b_mismatch_count == 0
        cells.append(
            {
                "replacement_confirm_ms": int(persistence),
                "source_count": source_count,
                "anchor_episode_count": episode_count,
                "reference_event_count": reference_event_count,
                "candidate_a_event_count": candidate_a_event_count,
                "candidate_b_event_count": candidate_b_event_count,
                "candidate_a_mismatch_count": candidate_a_mismatch_count,
                "candidate_b_mismatch_count": candidate_b_mismatch_count,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
            }
        )
    write_json(
        result_dir / "ontology_sufficiency.json",
        {
            "schema_version": "psem.ontology_simplification.s0.v1",
            "gate": "S0_perfect_state_ontology_sufficiency",
            "shared_action_authority": "issue-97 Gate 0 product-action oracle",
            "provenance": {
                "authority_snapshot_sha256": sha256_file(AUTHORITY_SNAPSHOT_PATH),
                "config_sha256": sha256_file(CONFIG_PATH),
                "predecessor_manifest_sha256": sha256_file(
                    _predecessor_result("dev", "relative_occupancy_manifest.jsonl")
                ),
                "predecessor_gate0_oracle_events_sha256": sha256_file(
                    _predecessor_result("dev", "gate0_oracle_events.jsonl")
                ),
                "predecessor_gate0_oracle_metrics_sha256": sha256_file(
                    _predecessor_result("dev", "gate0_oracle_metrics.json")
                ),
                "trace_reuse_receipt_sha256": sha256_file(
                    PACKAGE_ROOT / "trace_reuse_receipt.json"
                ),
                "causal_dependency_audit_sha256": sha256_file(
                    PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json"
                ),
                "evaluator_source_sha256": sha256_file(Path(__file__)),
                "simple_anchor_source_sha256": sha256_file(
                    PACKAGE_ROOT / "derive_simple_anchor.py"
                ),
                "anchor_overlap_source_sha256": sha256_file(
                    PACKAGE_ROOT / "derive_anchor_overlap.py"
                ),
            },
            "candidate_a_exact_action_equivalence": candidate_a_all_equal,
            "candidate_b_exact_action_equivalence": candidate_b_all_equal,
            "candidate_a_information_sufficient": candidate_a_all_equal,
            "candidate_b_ontology_verdict": "GREEN" if candidate_b_all_equal else "RED",
            "cells": cells,
        },
    )


def _topology_value(row: dict[str, Any], topology: str, key: str) -> int:
    return int(row["topology"].get(topology, {}).get(key, 0))


def _paired_outputs(
    session_rows: Sequence[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = [value for value in session_rows if _primary_candidate(value, cfg)]
    index = {
        (
            value["family"],
            value["arm"],
            value["candidate"],
            value["replacement_confirm_ms"],
            value["source_id"],
        ): value
        for value in primary_rows
    }
    comparisons = (
        ("simple_anchor_minus_r0", "simple_anchor", "r0_relative_occupancy"),
        ("anchor_overlap_minus_simple_anchor", "anchor_overlap", "simple_anchor"),
        ("anchor_overlap_minus_r0", "anchor_overlap", "r0_relative_occupancy"),
    )
    deltas: list[dict[str, Any]] = []
    grouped: dict[
        tuple[str, str, int, str, str], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = {}
    for family in FAMILY_KEYS:
        for arm in ("s1_oracle_anchor", "s2_fixed_issue97_lifecycle"):
            for persistence in cfg["replacement_confirm_ms"]:
                source_ids = sorted(
                    {
                        str(value["source_id"])
                        for value in primary_rows
                        if value["family"] == family
                        and value["arm"] == arm
                        and value["replacement_confirm_ms"] == persistence
                    }
                )
                for comparison, left_name, right_name in comparisons:
                    pairs = []
                    for source_id in source_ids:
                        left = index[(family, arm, left_name, persistence, source_id)]
                        right = index[(family, arm, right_name, persistence, source_id)]
                        pairs.append((left, right))
                        left_metrics = left["metrics"]
                        right_metrics = right["metrics"]
                        active_hours = float(left_metrics["active_speech_seconds"]) / 3600.0
                        takeover_left_n = _topology_value(
                            left, "overlap_takeover", "episodes_with_aligned_cut"
                        )
                        takeover_right_n = _topology_value(
                            right, "overlap_takeover", "episodes_with_aligned_cut"
                        )
                        takeover_d = _topology_value(
                            left, "overlap_takeover", "eligible_episode_count"
                        )
                        deltas.append(
                            {
                                "family": family,
                                "arm": arm,
                                "replacement_confirm_ms": persistence,
                                "comparison": comparison,
                                "candidate": left_name,
                                "source_id": source_id,
                                "contamination_seconds_per_active_speech_hour_delta": (
                                    (
                                        float(left_metrics["exclusive_other_contamination_seconds"])
                                        - float(
                                            right_metrics["exclusive_other_contamination_seconds"]
                                        )
                                    )
                                    / active_hours
                                    if active_hours
                                    else None
                                ),
                                "false_cut_count_delta": int(left_metrics["false_cut_count"])
                                - int(right_metrics["false_cut_count"]),
                                "missed_replacement_count_delta": int(
                                    left_metrics["missed_replacement_count"]
                                )
                                - int(right_metrics["missed_replacement_count"]),
                                "overlap_takeover_success_rate_delta": (
                                    (takeover_left_n - takeover_right_n) / takeover_d
                                    if takeover_d
                                    else None
                                ),
                            }
                        )
                    grouped[(family, arm, int(persistence), comparison, left_name)] = pairs
    for arm in ("s1_oracle_anchor", "s2_fixed_issue97_lifecycle"):
        for persistence in cfg["replacement_confirm_ms"]:
            for candidate in ("r0_relative_occupancy", "simple_anchor", "anchor_overlap"):
                source_ids = sorted(
                    {
                        str(value["source_id"])
                        for value in primary_rows
                        if value["arm"] == arm
                        and value["candidate"] == candidate
                        and value["replacement_confirm_ms"] == persistence
                    }
                )
                pairs = []
                for source_id in source_ids:
                    left = index[("ls_eend", arm, candidate, persistence, source_id)]
                    right = index[
                        ("streaming_sortformer", arm, candidate, persistence, source_id)
                    ]
                    pairs.append((left, right))
                    left_metrics = left["metrics"]
                    right_metrics = right["metrics"]
                    active_hours = float(left_metrics["active_speech_seconds"]) / 3600.0
                    takeover_left_n = _topology_value(
                        left, "overlap_takeover", "episodes_with_aligned_cut"
                    )
                    takeover_right_n = _topology_value(
                        right, "overlap_takeover", "episodes_with_aligned_cut"
                    )
                    takeover_d = _topology_value(
                        left, "overlap_takeover", "eligible_episode_count"
                    )
                    deltas.append(
                        {
                            "family": "cross_family",
                            "arm": arm,
                            "replacement_confirm_ms": persistence,
                            "comparison": "lseend_minus_streaming_sortformer",
                            "candidate": candidate,
                            "source_id": source_id,
                            "contamination_seconds_per_active_speech_hour_delta": (
                                (
                                    float(
                                        left_metrics[
                                            "exclusive_other_contamination_seconds"
                                        ]
                                    )
                                    - float(
                                        right_metrics[
                                            "exclusive_other_contamination_seconds"
                                        ]
                                    )
                                )
                                / active_hours
                                if active_hours
                                else None
                            ),
                            "false_cut_count_delta": int(left_metrics["false_cut_count"])
                            - int(right_metrics["false_cut_count"]),
                            "missed_replacement_count_delta": int(
                                left_metrics["missed_replacement_count"]
                            )
                            - int(right_metrics["missed_replacement_count"]),
                            "overlap_takeover_success_rate_delta": (
                                (takeover_left_n - takeover_right_n) / takeover_d
                                if takeover_d
                                else None
                            ),
                        }
                    )
                grouped[
                    (
                        "cross_family",
                        arm,
                        int(persistence),
                        "lseend_minus_streaming_sortformer",
                        candidate,
                    )
                ] = pairs
    rng = np.random.default_rng(int(cfg["bootstrap"]["seed"]))
    resamples = int(cfg["bootstrap"]["resamples"])
    confidence = float(cfg["bootstrap"]["confidence"])
    lower_q = (1.0 - confidence) * 50.0
    upper_q = 100.0 - lower_q
    intervals = []
    for (family, arm, persistence, comparison, candidate), pairs in sorted(grouped.items()):
        n = len(pairs)
        selection = rng.integers(0, n, size=(resamples, n))
        active_hours = np.asarray(
            [float(left["metrics"]["active_speech_seconds"]) / 3600.0 for left, _ in pairs]
        )
        contamination = np.asarray(
            [
                float(left["metrics"]["exclusive_other_contamination_seconds"])
                - float(right["metrics"]["exclusive_other_contamination_seconds"])
                for left, right in pairs
            ]
        )
        false_cuts = np.asarray(
            [
                int(left["metrics"]["false_cut_count"]) - int(right["metrics"]["false_cut_count"])
                for left, right in pairs
            ]
        )
        missed = np.asarray(
            [
                int(left["metrics"]["missed_replacement_count"])
                - int(right["metrics"]["missed_replacement_count"])
                for left, right in pairs
            ]
        )
        takeover_delta = np.asarray(
            [
                _topology_value(left, "overlap_takeover", "episodes_with_aligned_cut")
                - _topology_value(right, "overlap_takeover", "episodes_with_aligned_cut")
                for left, right in pairs
            ]
        )
        takeover_count = np.asarray(
            [
                _topology_value(left, "overlap_takeover", "eligible_episode_count")
                for left, _ in pairs
            ]
        )
        sampled_active = active_hours[selection].sum(axis=1)
        sampled_takeover_count = takeover_count[selection].sum(axis=1)
        values = {
            "contamination_seconds_per_active_speech_hour_delta": np.divide(
                contamination[selection].sum(axis=1),
                sampled_active,
                out=np.zeros(resamples, dtype=np.float64),
                where=sampled_active != 0,
            ),
            "false_cut_count_per_session_delta": false_cuts[selection].sum(axis=1) / n,
            "missed_replacement_count_per_session_delta": missed[selection].sum(axis=1) / n,
            "overlap_takeover_success_rate_delta": np.divide(
                takeover_delta[selection].sum(axis=1),
                sampled_takeover_count,
                out=np.zeros(resamples, dtype=np.float64),
                where=sampled_takeover_count != 0,
            ),
        }
        intervals.append(
            {
                "family": family,
                "arm": arm,
                "replacement_confirm_ms": persistence,
                "comparison": comparison,
                "candidate": candidate,
                "source_count": n,
                "resamples": resamples,
                "confidence": confidence,
                "intervals": {
                    key: {
                        "lower": percentile(value, lower_q),
                        "median": percentile(value, 50),
                        "upper": percentile(value, upper_q),
                    }
                    for key, value in values.items()
                },
            }
        )
    return deltas, intervals


def _session_row(
    *,
    family: str,
    arm: str,
    candidate: str,
    variant: str,
    anchor_threshold: float,
    overlap_threshold: float | None,
    persistence: int,
    source_id: str,
    metrics: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "arm": arm,
        "candidate": candidate,
        "variant": variant,
        "anchor_threshold": anchor_threshold,
        "overlap_threshold": overlap_threshold,
        "replacement_confirm_ms": persistence,
        "source_id": source_id,
        "metrics": metrics,
        "topology": topology,
    }


def _apply_speech_gate(
    observations: Sequence[ModelObservation], spans: Sequence[dict[str, Any]]
) -> tuple[ModelObservation, ...]:
    normalized = [(int(value["start_sample"]), int(value["end_sample"])) for value in spans]
    result: list[ModelObservation] = []
    span_index = 0
    for observation in observations:
        while (
            span_index < len(normalized) and normalized[span_index][1] <= observation.start_sample
        ):
            span_index += 1
        breakpoints = {observation.start_sample, observation.end_sample}
        local_index = span_index
        while local_index < len(normalized) and normalized[local_index][0] < observation.end_sample:
            left, right = normalized[local_index]
            if observation.start_sample < left < observation.end_sample:
                breakpoints.add(left)
            if observation.start_sample < right < observation.end_sample:
                breakpoints.add(right)
            local_index += 1
        ordered = sorted(breakpoints)
        for piece_index, (start, end) in enumerate(zip(ordered, ordered[1:])):
            speech = any(
                left < end and right > start
                for left, right in normalized[max(0, span_index - 1) : local_index + 1]
            )
            result.append(
                replace(
                    observation,
                    start_sample=start,
                    end_sample=end,
                    speech_present=speech,
                    state_reset=observation.state_reset and piece_index == 0,
                )
            )
    return tuple(result)


def run_production_vad_sensitivity(role: str) -> None:
    cfg = _config()
    result_dir = _result_dir(role)
    gate_path = result_dir / "production_vad_speech_gate.jsonl"
    receipt_path = result_dir / "production_vad_replay_receipt.json"
    if not gate_path.is_file() or not receipt_path.is_file():
        raise SimplificationError("production VAD replay must be generated first")
    receipt = load_json(receipt_path)
    if receipt["speech_gate_sha256"] != sha256_file(gate_path):
        raise SimplificationError("production VAD speech-gate digest mismatch")
    manifest, receipts, _, gate2_rows = _load_role_inputs(role)
    vad_rows = {str(value["source_id"]): value for value in load_jsonl(gate_path)}
    if set(vad_rows) != {str(value["source_id"]) for value in manifest}:
        raise SimplificationError("production VAD source coverage mismatch")
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    session_rows: list[dict[str, Any]] = []
    for family in FAMILY_KEYS:
        receipt_by_source = _trace_by_source(receipts[family])
        b_primary = cfg["candidate_b"]["threshold_grids"][family]["primary"]
        candidate_cells = (
            (
                "simple_anchor",
                float(cfg["candidate_a"]["anchor_thresholds"][family][0]),
                None,
            ),
            ("anchor_overlap", float(b_primary[0]), float(b_primary[1])),
        )
        for row in manifest:
            source_id = str(row["source_id"])
            cells, gt_observations, slot_ids = _posterior_inputs(row, receipt_by_source[source_id])
            observations = _apply_speech_gate(gt_observations, vad_rows[source_id]["speech_spans"])
            for persistence in cfg["replacement_confirm_ms"]:
                persistence = int(persistence)
                confirmation_samples = persistence * 16
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=confirmation_samples,
                    enrollment_samples=enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                _, mappings, _ = _oracle_anchor_records(row, cells, slot_ids, reference)
                gate2_row = gate2_rows[(source_id, family, persistence)]
                for candidate, anchor_threshold, overlap_threshold in candidate_cells:
                    oracle_metrics, oracle_events = _oracle_product_session(
                        row=row,
                        reference=reference,
                        observations=observations,
                        mappings=mappings,
                        candidate=candidate,
                        anchor_threshold=anchor_threshold,
                        overlap_threshold=overlap_threshold,
                        strict_inconsistent=False,
                        confirmation_samples=confirmation_samples,
                        tolerance_samples=tolerance_samples,
                    )
                    session_rows.append(
                        _session_row(
                            family=family,
                            arm="s1_oracle_anchor",
                            candidate=candidate,
                            variant="production_vad",
                            anchor_threshold=anchor_threshold,
                            overlap_threshold=overlap_threshold,
                            persistence=persistence,
                            source_id=source_id,
                            metrics=oracle_metrics,
                            topology=_session_topology(
                                row, oracle_events, reference, tolerance_samples
                            ),
                        )
                    )
                    causal_metrics, causal_events = _causal_product_session(
                        row=row,
                        reference=reference,
                        observations=observations,
                        gate2_row=gate2_row,
                        candidate=candidate,
                        anchor_threshold=anchor_threshold,
                        overlap_threshold=overlap_threshold,
                        strict_inconsistent=False,
                        confirmation_samples=confirmation_samples,
                        tolerance_samples=tolerance_samples,
                    )
                    session_rows.append(
                        _session_row(
                            family=family,
                            arm="s2_fixed_issue97_lifecycle",
                            candidate=candidate,
                            variant="production_vad",
                            anchor_threshold=anchor_threshold,
                            overlap_threshold=overlap_threshold,
                            persistence=persistence,
                            source_id=source_id,
                            metrics=causal_metrics,
                            topology=_session_topology(
                                row, causal_events, reference, tolerance_samples
                            ),
                        )
                    )
    rows = []
    gt_frontier = load_json(result_dir / "product_frontiers.json")["rows"]
    for key in sorted(
        {
            (
                value["family"],
                value["arm"],
                value["candidate"],
                value["replacement_confirm_ms"],
            )
            for value in session_rows
        }
    ):
        group = [
            value
            for value in session_rows
            if (
                value["family"],
                value["arm"],
                value["candidate"],
                value["replacement_confirm_ms"],
            )
            == key
        ]
        aggregate = _aggregate_product(group)
        gt = next(
            value
            for value in gt_frontier
            if value["family"] == key[0]
            and value["arm"] == key[1]
            and value["candidate"] == key[2]
            and value["variant"] == "primary"
            and value["replacement_confirm_ms"] == key[3]
            and (
                key[2] == "simple_anchor"
                or (
                    value["anchor_threshold"]
                    == float(cfg["candidate_b"]["threshold_grids"][key[0]]["primary"][0])
                    and value["overlap_threshold"]
                    == float(cfg["candidate_b"]["threshold_grids"][key[0]]["primary"][1])
                )
            )
        )
        rows.append(
            {
                "family": key[0],
                "arm": key[1],
                "candidate": key[2],
                "replacement_confirm_ms": key[3],
                "production_vad": aggregate,
                "gt_speech_gate": {
                    name: gt[name]
                    for name in (
                        "exclusive_other_contamination_seconds_per_active_speech_hour",
                        "speaker_induced_cut_count_per_active_speech_hour",
                        "false_cut_count",
                        "missed_replacement_count",
                        "speaker_protection_enabled_fraction",
                    )
                },
                "production_vad_minus_gt_speech_gate": {
                    name: aggregate[name] - gt[name]
                    for name in (
                        "exclusive_other_contamination_seconds_per_active_speech_hour",
                        "speaker_induced_cut_count_per_active_speech_hour",
                        "false_cut_count",
                        "missed_replacement_count",
                        "speaker_protection_enabled_fraction",
                    )
                },
            }
        )
    write_json(
        result_dir / "production_vad_sensitivity.json",
        {
            "schema_version": "psem.ontology_simplification.production_vad_sensitivity.v1",
            "role": "PSEM-STRATEGY-DEV" if role == "dev" else "PSEM-STRATEGY-EVAL",
            "config_sha256": sha256_file(CONFIG_PATH),
            "production_vad_receipt_sha256": sha256_file(receipt_path),
            "production_vad_speech_gate_sha256": sha256_file(gate_path),
            "interpretation": "frozen-model product-VAD sensitivity on development-known V2",
            "production_readiness_claim": False,
            "rows": rows,
        },
    )


def run_role(role: str) -> None:
    cfg = _config()
    manifest, receipts, gate1_rows, gate2_rows = _load_role_inputs(role)
    source_rows = {str(value["source_id"]): value for value in manifest}
    if role == "dev":
        _write_s0(manifest, cfg, _result_dir(role))
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * 16
    silence_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * 16
    tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
    diagnostic_thresholds = [value / 100.0 for value in range(5, 100, 5)]
    family_payloads: dict[str, Any] = {}
    dropout_payload: dict[str, Any] = {}
    global_payload: dict[str, Any] = {}
    session_rows: list[dict[str, Any]] = []
    for family in FAMILY_KEYS:
        receipt_by_source = _trace_by_source(receipts[family])
        prepared: dict[str, dict[str, Any]] = {}
        global_records: list[dict[str, Any]] = []
        global_coverage: list[dict[str, Any]] = []
        oracle_records: list[AnchorRecord] = []
        mapping_coverage: list[dict[str, Any]] = []
        causal_records: dict[int, list[AnchorRecord]] = {
            int(value): [] for value in cfg["replacement_confirm_ms"]
        }
        for source_id in sorted(source_rows):
            row = source_rows[source_id]
            cells, observations, slot_ids = _posterior_inputs(row, receipt_by_source[source_id])
            mapping_reference = gt_reference_session(
                row,
                replacement_confirmation_samples=200 * 16,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_samples,
            )
            records, _, source_mapping_coverage = _oracle_anchor_records(
                row, cells, slot_ids, mapping_reference
            )
            oracle_records.extend(records)
            mapping_coverage.append(source_mapping_coverage)
            source_global_records, source_global_coverage = _global_overlap_records(row, cells)
            global_records.extend(source_global_records)
            global_coverage.append(source_global_coverage)
            for persistence in cfg["replacement_confirm_ms"]:
                causal_records[int(persistence)].extend(
                    _causal_anchor_records(
                        row,
                        cells,
                        gate2_rows[(source_id, family, int(persistence))],
                    )
                )
            prepared[source_id] = {
                "row": row,
                "cells": cells,
                "observations": observations,
                "slot_ids": slot_ids,
            }
        anchor_threshold = float(cfg["candidate_a"]["anchor_thresholds"][family][0])
        oracle_anchor_metrics = _anchor_metrics(
            oracle_records, anchor_threshold, diagnostic_thresholds
        )
        oracle_anchor_metrics["oracle_anchor_mapping_coverage"] = _aggregate_mapping_coverage(
            mapping_coverage
        )
        causal_anchor_metrics = {
            str(persistence): _anchor_metrics(records, anchor_threshold, diagnostic_thresholds)
            for persistence, records in causal_records.items()
        }
        dropout_payload[family] = {
            "anchor_threshold": anchor_threshold,
            "oracle_anchor_mapping_coverage": _aggregate_mapping_coverage(mapping_coverage),
            "s1_oracle_anchor": _sustained_dropout(oracle_records, anchor_threshold),
            "s2_fixed_issue97_lifecycle": {
                str(persistence): _sustained_dropout(records, anchor_threshold)
                for persistence, records in causal_records.items()
            },
        }
        global_payload[family] = _global_overlap_metrics(
            manifest,
            global_records,
            global_coverage,
            [float(value) for value in cfg["global_overlap_diagnostic"]["thresholds"]],
            float(cfg["global_overlap_diagnostic"]["primary_threshold"]),
        )
        b_grid = cfg["candidate_b"]["threshold_grids"][family]
        candidate_cells: list[tuple[str, str, float, float | None, bool]] = [
            ("simple_anchor", "primary", anchor_threshold, None, False)
        ]
        candidate_cells.extend(
            (
                "anchor_overlap",
                "primary",
                float(candidate_anchor),
                float(candidate_overlap),
                False,
            )
            for candidate_anchor in b_grid["anchor"]
            for candidate_overlap in b_grid["anchor_overlap"]
        )
        candidate_cells.append(
            (
                "anchor_overlap",
                "strict_non_anchor",
                float(b_grid["primary"][0]),
                float(b_grid["primary"][1]),
                True,
            )
        )
        for persistence in cfg["replacement_confirm_ms"]:
            persistence = int(persistence)
            confirmation_samples = persistence * 16
            for source_id in sorted(prepared):
                data = prepared[source_id]
                row = data["row"]
                reference = gt_reference_session(
                    row,
                    replacement_confirmation_samples=confirmation_samples,
                    enrollment_samples=enrollment_samples,
                    silence_reset_samples=silence_samples,
                )
                _, mappings, _ = _oracle_anchor_records(
                    row, data["cells"], data["slot_ids"], reference
                )
                gate1_row = gate1_rows[(source_id, family, persistence)]
                gate2_row = gate2_rows[(source_id, family, persistence)]
                r0_oracle_metrics, r0_oracle_events = _r0_oracle_session(
                    row=row,
                    reference=reference,
                    gate1_row=gate1_row,
                    tolerance_samples=tolerance_samples,
                )
                session_rows.append(
                    _session_row(
                        family=family,
                        arm="s1_oracle_anchor",
                        candidate="r0_relative_occupancy",
                        variant="primary",
                        anchor_threshold=float(gate1_row["anchor_threshold"]),
                        overlap_threshold=float(gate1_row["other_threshold"]),
                        persistence=persistence,
                        source_id=source_id,
                        metrics=r0_oracle_metrics,
                        topology=_session_topology(
                            row, r0_oracle_events, reference, tolerance_samples
                        ),
                    )
                )
                r0_causal_metrics, r0_causal_events = _r0_causal_session(
                    row=row,
                    reference=reference,
                    gate2_row=gate2_row,
                    tolerance_samples=tolerance_samples,
                )
                session_rows.append(
                    _session_row(
                        family=family,
                        arm="s2_fixed_issue97_lifecycle",
                        candidate="r0_relative_occupancy",
                        variant="primary",
                        anchor_threshold=float(gate2_row["anchor_threshold"]),
                        overlap_threshold=float(gate2_row["other_threshold"]),
                        persistence=persistence,
                        source_id=source_id,
                        metrics=r0_causal_metrics,
                        topology=_session_topology(
                            row, r0_causal_events, reference, tolerance_samples
                        ),
                    )
                )
                for (
                    candidate,
                    variant,
                    candidate_anchor,
                    candidate_overlap,
                    strict,
                ) in candidate_cells:
                    oracle_metrics, oracle_events = _oracle_product_session(
                        row=row,
                        reference=reference,
                        observations=data["observations"],
                        mappings=mappings,
                        candidate=candidate,
                        anchor_threshold=candidate_anchor,
                        overlap_threshold=candidate_overlap,
                        strict_inconsistent=strict,
                        confirmation_samples=confirmation_samples,
                        tolerance_samples=tolerance_samples,
                    )
                    session_rows.append(
                        _session_row(
                            family=family,
                            arm="s1_oracle_anchor",
                            candidate=candidate,
                            variant=variant,
                            anchor_threshold=candidate_anchor,
                            overlap_threshold=candidate_overlap,
                            persistence=persistence,
                            source_id=source_id,
                            metrics=oracle_metrics,
                            topology=_session_topology(
                                row, oracle_events, reference, tolerance_samples
                            ),
                        )
                    )
                    causal_metrics, causal_events = _causal_product_session(
                        row=row,
                        reference=reference,
                        observations=data["observations"],
                        gate2_row=gate2_row,
                        candidate=candidate,
                        anchor_threshold=candidate_anchor,
                        overlap_threshold=candidate_overlap,
                        strict_inconsistent=strict,
                        confirmation_samples=confirmation_samples,
                        tolerance_samples=tolerance_samples,
                    )
                    session_rows.append(
                        _session_row(
                            family=family,
                            arm="s2_fixed_issue97_lifecycle",
                            candidate=candidate,
                            variant=variant,
                            anchor_threshold=candidate_anchor,
                            overlap_threshold=candidate_overlap,
                            persistence=persistence,
                            source_id=source_id,
                            metrics=causal_metrics,
                            topology=_session_topology(
                                row, causal_events, reference, tolerance_samples
                            ),
                        )
                    )
        family_payloads[family] = {
            "s1_oracle_anchor": oracle_anchor_metrics,
            "s2_fixed_issue97_lifecycle": causal_anchor_metrics,
        }
    frontier = []
    group_keys = sorted(
        {
            (
                value["family"],
                value["arm"],
                value["candidate"],
                value["variant"],
                value["anchor_threshold"],
                value["overlap_threshold"],
                value["replacement_confirm_ms"],
            )
            for value in session_rows
        },
        key=str,
    )
    for key in group_keys:
        rows = [
            value
            for value in session_rows
            if (
                value["family"],
                value["arm"],
                value["candidate"],
                value["variant"],
                value["anchor_threshold"],
                value["overlap_threshold"],
                value["replacement_confirm_ms"],
            )
            == key
        ]
        aggregate = _aggregate_product(rows)
        frontier.append(
            {
                "family": key[0],
                "arm": key[1],
                "candidate": key[2],
                "variant": key[3],
                "anchor_threshold": key[4],
                "overlap_threshold": key[5],
                "replacement_confirm_ms": key[6],
                **aggregate,
            }
        )
    deltas, intervals = _paired_outputs(session_rows, cfg)
    result_dir = _result_dir(role)
    provenance = {
        "schema_version": "psem.ontology_simplification.result_provenance.v1",
        "role": _role_name(role),
        "authority_snapshot_sha256": sha256_file(AUTHORITY_SNAPSHOT_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "trace_reuse_receipt_sha256": sha256_file(PACKAGE_ROOT / "trace_reuse_receipt.json"),
        "causal_dependency_audit_sha256": sha256_file(
            PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json"
        ),
        "predecessor_manifest_sha256": sha256_file(
            _predecessor_result(role, "relative_occupancy_manifest.jsonl")
        ),
        "predecessor_gate1_ledger_sha256": sha256_file(
            _predecessor_result(role, "gate1_event_ledger.jsonl")
        ),
        "predecessor_gate2_ledger_sha256": sha256_file(
            _predecessor_result(role, "gate2_event_ledger.jsonl")
        ),
        "predecessor_product_frontiers_sha256": sha256_file(
            _predecessor_result(role, "product_frontiers.json")
        ),
        "predecessor_model_receipt_sha256": {
            family: sha256_file(_predecessor_result(role, RECEIPT_NAMES[family]))
            for family in FAMILY_KEYS
        },
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "simple_anchor_source_sha256": sha256_file(PACKAGE_ROOT / "derive_simple_anchor.py"),
        "anchor_overlap_source_sha256": sha256_file(
            PACKAGE_ROOT / "derive_anchor_overlap.py"
        ),
        "new_model_inference_performed": False,
        "causal_interpretation": cfg["causal_arm"]["interpretation"],
    }
    write_json(
        result_dir / "anchor_dropout_slices.json", {**provenance, "families": dropout_payload}
    )
    write_json(
        result_dir / "global_overlap_diagnostic.json", {**provenance, "families": global_payload}
    )
    write_json(result_dir / "product_frontiers.json", {**provenance, "rows": frontier})
    write_json(result_dir / "paired_session_deltas.json", {**provenance, "rows": deltas})
    write_json(result_dir / "bootstrap_intervals.json", {**provenance, "rows": intervals})
    for family in FAMILY_KEYS:
        output_name = OUTPUT_NAMES[family]
        for candidate in ("simple_anchor", "anchor_overlap"):
            write_json(
                result_dir / f"{output_name}_{candidate}_metrics.json",
                {
                    **provenance,
                    "family": family,
                    "candidate": candidate,
                    "anchor_diagnostics": family_payloads[family],
                    "product_frontier": [
                        value
                        for value in frontier
                        if value["family"] == family
                        and value["candidate"] in (candidate, "r0_relative_occupancy")
                    ],
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--role", choices=("dev", "eval"))
    parser.add_argument("--production-vad-sensitivity", action="store_true")
    args = parser.parse_args()
    write_inventory_and_audit()
    if args.inventory_only:
        return
    if args.role is None:
        parser.error("--role is required unless --inventory-only is used")
    if args.production_vad_sensitivity:
        run_production_vad_sensitivity(args.role)
    else:
        run_role(args.role)


if __name__ == "__main__":
    main()
