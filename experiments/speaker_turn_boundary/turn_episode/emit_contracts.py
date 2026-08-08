"""Emit the frozen proposal_contract.json and fusion_contract.json artifacts.

Phase 0 deliverable. Every artifact contains a canonical content hash
(``content_sha256``) computed over the canonical JSON of the artifact excluding the hash
field itself (PRD Section 27.3). The signal-extractor registry is completed at Phase 4
(PRD Section 18.3); here the registry and its schema version are declared with empty
extractor lists so the contract identity is stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.speaker_turn_boundary.turn_episode.schemas import (
    ACTIONIZATION_SCHEMA_VERSION,
    PROPOSAL_GENERATION_SCHEMA_VERSION,
    SCHEMA_VERSION,
)

PLAN_BLOB = "24340f488f1bb46c666a5fc15eef2fc87ef1f826"


def canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def with_content_hash(data: dict[str, object]) -> dict[str, object]:
    base = {key: value for key, value in data.items() if key != "content_sha256"}
    digest = hashlib.sha256(canonical_json(base).encode("utf-8")).hexdigest()
    out = dict(base)
    out["content_sha256"] = digest
    return out


def proposal_contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_generation_schema": PROPOSAL_GENERATION_SCHEMA_VERSION,
        "plan_blob": PLAN_BLOB,
        "proposal_kinds": [
            "new_track_onset",
            "dominant_replacement",
            "overlap_onset",
            "track_instability",
            "speaker_change_unknown",
        ],
        "invariants": [
            "observed_source_sample_at_emit >= boundary_source_sample",
            "audio_epoch and source_session_id mandatory",
            "confidence interpreted only under confidence_semantics_id",
            "events deterministic for identical audio/model/frontend/profile",
            "event cannot read samples beyond its observation frontier",
            "proposal generation and actionization use separate schema versions",
        ],
        "signal_extractors": [],
        "signal_extractor_schema_version": "turn_episode_v1.signal_extractor",
        "signal_extractor_declaration_fields": [
            "signal_extractor_id",
            "sign",
            "causal_horizon_ms",
            "valid_window_rule",
            "missing_observation_rule",
        ],
    }


def fusion_contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "actionization_schema": ACTIONIZATION_SCHEMA_VERSION,
        "plan_blob": PLAN_BLOB,
        "action_kinds": [
            "retain_vad",
            "accelerate_or_replace_vad",
            "add_hard_boundary",
            "emit_soft_marker",
            "suppress_detector_duplicate",
            "suppress_vad_duplicate",
            "structural_max_duration",
            "unscored_action",
        ],
        "hard_boundary_action_kinds": [
            "retain_vad",
            "accelerate_or_replace_vad",
            "add_hard_boundary",
        ],
        "cluster_kind_priority": [
            "overlap_onset",
            "dominant_replacement",
            "new_track_onset",
            "track_instability",
        ],
        "localization_tolerance_ms": {"primary": 500, "view": 250},
        "availability_deadlines_ms": [250, 500, 1000, 1500, 2000],
        "turn_owner_threshold_ms": {"primary": 100, "sensitivity_views": [50, 200]},
        "harmful_active_split_guard_ms": {"primary": 200, "sensitivity_views": [100, 300]},
        "cluster_grid": {
            "cluster_debounce_ms": [0, 100, 250],
            "cluster_boundary_radius_ms": [250, 500],
            "refractory_ms": [0, 250, 500],
        },
        "vad_association": {
            "detector_vad_radius_ms": [250, 500],
            "same_silence_interval_association": [False, True],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit turn_episode_v1 contract artifacts")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: experiments/speaker_turn_boundary/results/turn_episode_v1)",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("proposal_contract.json", proposal_contract()),
        ("fusion_contract.json", fusion_contract()),
    ):
        path = args.out / name
        path.write_text(canonical_json(with_content_hash(data)) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
