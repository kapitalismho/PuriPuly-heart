from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.psem_training_strategy_gate.data.identity_components import (
    build_identity_graph,
)
from experiments.psem_training_strategy_gate.data.pretraining_overlap import (
    build_pretraining_overlap_audit,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)

AUTHORITY_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/77"
AUTHORITY_PIN = "5778025c8aca1ea1cb7cd8fc41645b520ca1f9f749155b5f5daada32e940b559"
SAMPLE_RATE_HZ = 16000
ROLE_REQUIREMENTS = {
    "PSEM-STRATEGY-TRAIN": {"scored_hours": 20, "independent_meetings": 12},
    "PSEM-STRATEGY-DEV": {"scored_hours": 5, "independent_meetings": 4},
    "PSEM-STRATEGY-EVAL": {"scored_hours": 8, "independent_meetings": 6},
}
TOPOLOGY_REQUIREMENTS = {
    "clean_direct_different_speaker_handoff": {"train_dev": 100, "eval": 20},
    "silence_gap_different_speaker_handoff": {"train_dev": 200, "eval": 40},
    "same_speaker_silence_gap_resume": {"train_dev": 200, "eval": 40},
    "overlap_return": {"train_dev": 100, "eval": 20},
    "overlap_takeover": {"train_dev": 100, "eval": 20},
    "short_backchannel_return": {"train_dev": 80, "eval": 20},
}
NEGATIVE_EXPOSURE_REQUIREMENTS = {
    "stable_singleton_samples": {
        "train_dev": 8 * 3600 * SAMPLE_RATE_HZ,
        "eval": 2 * 3600 * SAMPLE_RATE_HZ,
    },
    "ongoing_overlap_samples": {
        "train_dev": 60 * 60 * SAMPLE_RATE_HZ,
        "eval": 15 * 60 * SAMPLE_RATE_HZ,
    },
}
NO_MODEL_FIELDS = (
    "model_predictions_consulted",
    "model_scores_consulted",
    "official_model_results_inspected",
    "official_model_training_performed",
)
NATURAL_CORPORA = frozenset({"AMI", "AliMeeting"})


class SplitFeasibilityError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitFeasibilityError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise SplitFeasibilityError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitFeasibilityError(f"invalid JSONL artifact: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise SplitFeasibilityError(f"JSONL artifact must contain objects: {path}")
    return rows


def _hours(samples: int) -> float:
    return round(samples / SAMPLE_RATE_HZ / 3600, 6)


def _deficit(observed: int, required: int) -> int:
    return max(0, required - observed)


def _combined_requirements() -> dict[str, Any]:
    return {
        "scored_samples": sum(row["scored_hours"] for row in ROLE_REQUIREMENTS.values())
        * 3600
        * SAMPLE_RATE_HZ,
        "independent_meetings": sum(
            row["independent_meetings"] for row in ROLE_REQUIREMENTS.values()
        ),
        "eval_eligible_components": ROLE_REQUIREMENTS["PSEM-STRATEGY-EVAL"]["independent_meetings"],
        "primary_topology_counts": {
            topology: minima["train_dev"] + minima["eval"]
            for topology, minima in TOPOLOGY_REQUIREMENTS.items()
        },
        "stable_singleton_samples": sum(
            NEGATIVE_EXPOSURE_REQUIREMENTS["stable_singleton_samples"].values()
        ),
        "ongoing_overlap_samples": sum(
            NEGATIVE_EXPOSURE_REQUIREMENTS["ongoing_overlap_samples"].values()
        ),
    }


def _validate_natural_sources(source_rows: list[dict[str, Any]]) -> None:
    if any(
        row.get("corpus") not in NATURAL_CORPORA
        or not all(
            isinstance(row.get(field), str) and row[field]
            for field in (
                "source_id",
                "session_id",
                "meeting_type",
                "audio_source_url",
                "license_id",
                "use_authorization",
            )
        )
        or row.get("synthetic_parent_id") not in (None, "", [])
        or row.get("synthetic_transformation_seed") not in (None, "", [])
        for row in source_rows
    ):
        raise SplitFeasibilityError("source inventory is not proven natural-only")


def _eval_eligible_component_ids(
    graph: dict[str, Any], overlap_rows: list[dict[str, Any]]
) -> set[str]:
    overlap_by_source = {row.get("source_id"): row for row in overlap_rows}
    components = graph.get("components")
    if (
        len(overlap_by_source) != len(overlap_rows)
        or not isinstance(components, list)
        or any(not isinstance(component, dict) for component in components)
    ):
        raise SplitFeasibilityError("component EVAL eligibility inventory is invalid")
    component_source_ids = {
        source_id for component in components for source_id in component.get("source_ids", [])
    }
    if set(overlap_by_source) != component_source_ids:
        raise SplitFeasibilityError("component EVAL eligibility coverage mismatch")
    eligible = set()
    for component in components:
        component_id = component.get("component_id")
        source_ids = component.get("source_ids")
        if not isinstance(component_id, str) or not isinstance(source_ids, list):
            raise SplitFeasibilityError("component EVAL eligibility structure is invalid")
        member_rows = [overlap_by_source[source_id] for source_id in source_ids]
        if any(row.get("identity_component_id") != component_id for row in member_rows):
            raise SplitFeasibilityError("overlap audit component binding mismatch")
        if (
            component.get("split_assignment_eligible") is True
            and component.get("eval_forbidden") is False
            and all(
                row.get("eval_forbidden_by_pretraining_overlap") is False for row in member_rows
            )
        ):
            eligible.add(component_id)
    return eligible


def build_split_feasibility(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
) -> dict[str, Any]:
    census = _load_json(data_dir / "topology_census.json")
    source_rows = _load_jsonl(data_dir / "source_manifest.jsonl")
    _validate_natural_sources(source_rows)
    checked_graph = _load_json(data_dir / "identity_components.json")
    rebuilt_graph = build_identity_graph(data_dir)
    if checked_graph != rebuilt_graph:
        raise SplitFeasibilityError("checked-in identity graph is not current")
    checked_overlap = _load_json(data_dir / "wavlm_pretraining_overlap.json")
    rebuilt_overlap = build_pretraining_overlap_audit(data_dir, registry_path, source_registry_path)
    if checked_overlap != rebuilt_overlap:
        raise SplitFeasibilityError("checked-in pretraining overlap audit is not current")
    overall = census.get("overall")
    graph_summary = checked_graph.get("summary")
    overlap_rows = checked_overlap.get("sources")
    if (
        not isinstance(overall, dict)
        or not isinstance(graph_summary, dict)
        or not isinstance(overlap_rows, list)
    ):
        raise SplitFeasibilityError("accepted input summaries are invalid")
    requirements = _combined_requirements()
    observed_counts = overall.get("primary_topology_counts")
    if not isinstance(observed_counts, dict):
        raise SplitFeasibilityError("topology count inventory is invalid")
    eval_eligible_components = _eval_eligible_component_ids(checked_graph, overlap_rows)
    deficits = {
        "scored_samples": _deficit(overall["scored_samples"], requirements["scored_samples"]),
        "independent_meetings": _deficit(
            graph_summary["source_count"], requirements["independent_meetings"]
        ),
        "eval_eligible_components": _deficit(
            len(eval_eligible_components), requirements["eval_eligible_components"]
        ),
        "primary_topology_counts": {
            topology: _deficit(observed_counts[topology], required)
            for topology, required in requirements["primary_topology_counts"].items()
        },
        "stable_singleton_samples": _deficit(
            overall["stable_singleton_samples"],
            requirements["stable_singleton_samples"],
        ),
        "ongoing_overlap_samples": _deficit(
            overall["ongoing_overlap_samples"],
            requirements["ongoing_overlap_samples"],
        ),
    }
    blocking_lower_bounds = [
        key
        for key in (
            "scored_samples",
            "independent_meetings",
            "eval_eligible_components",
            "stable_singleton_samples",
            "ongoing_overlap_samples",
        )
        if deficits[key]
    ]
    blocking_lower_bounds.extend(
        f"primary_topology_counts.{topology}"
        for topology, deficit in deficits["primary_topology_counts"].items()
        if deficit
    )
    scored_deficit = deficits["scored_samples"]
    search_status = (
        "skipped_proven_infeasible_by_total_lower_bounds"
        if blocking_lower_bounds
        else "component_assignment_search_required"
    )
    return {
        "schema_version": 1,
        "artifact_role": "psem_split_feasibility",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "search_status": search_status,
        "valid_assignment_exists": False if blocking_lower_bounds else None,
        "assignment_manifest_emitted": False,
        "assignments": {},
        "input_artifacts": {
            "topology_census_sha256": sha256_file(data_dir / "topology_census.json"),
            "identity_components_sha256": sha256_file(data_dir / "identity_components.json"),
            "wavlm_pretraining_overlap_sha256": sha256_file(
                data_dir / "wavlm_pretraining_overlap.json"
            ),
            "source_manifest_sha256": sha256_file(data_dir / "source_manifest.jsonl"),
            "source_ids_sha256": checked_graph["input_artifacts"]["source_ids_sha256"],
        },
        "requirements": {
            "roles": ROLE_REQUIREMENTS,
            "topologies": TOPOLOGY_REQUIREMENTS,
            "negative_exposure": NEGATIVE_EXPOSURE_REQUIREMENTS,
            "combined_lower_bounds": requirements,
        },
        "observed": {
            "scored_samples": overall["scored_samples"],
            "scored_hours": overall["scored_hours"],
            "independent_meetings": graph_summary["source_count"],
            "identity_components": graph_summary["component_count"],
            "eval_eligible_components": len(eval_eligible_components),
            "primary_topology_counts": observed_counts,
            "stable_singleton_samples": overall["stable_singleton_samples"],
            "stable_singleton_hours": overall["stable_singleton_hours"],
            "ongoing_overlap_samples": overall["ongoing_overlap_samples"],
            "ongoing_overlap_hours": overall["ongoing_overlap_hours"],
        },
        "deficits": {
            **deficits,
            "scored_hours": _hours(scored_deficit),
        },
        "blocking_lower_bounds": blocking_lower_bounds,
        "acquisition_handoff": {
            "minimum_additional_natural_scored_samples": scored_deficit,
            "minimum_additional_natural_scored_hours": _hours(scored_deficit),
            "minimum_additional_independent_meetings": deficits["independent_meetings"],
            "minimum_additional_eval_eligible_components": deficits["eval_eligible_components"],
            "minimum_additional_primary_topology_counts": deficits["primary_topology_counts"],
            "minimum_additional_stable_singleton_hours": _hours(
                deficits["stable_singleton_samples"]
            ),
            "minimum_additional_ongoing_overlap_hours": _hours(deficits["ongoing_overlap_samples"]),
            "planning_buffer_additional_hours_to_40": round(
                max(0.0, 40 - overall["scored_hours"]), 6
            ),
            "planning_buffer_additional_hours_to_45": round(
                max(0.0, 45 - overall["scored_hours"]), 6
            ),
            "natural_data_only": True,
            "recensus_and_identity_rebuild_required": True,
            "component_assignment_search_required_after_lower_bounds_pass": True,
        },
        "decision_basis_sha256": canonical_sha256(
            {
                "requirements": requirements,
                "observed": overall,
                "graph_summary": graph_summary,
                "overlap_summary": checked_overlap["summary"],
            }
        ),
        "model_policy": {field: False for field in NO_MODEL_FIELDS},
    }


def write_split_feasibility(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    output_path: Path,
) -> None:
    result = build_split_feasibility(data_dir, registry_path, source_registry_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_split_feasibility(
        args.data_dir.resolve(),
        args.registry.resolve(),
        args.source_registry.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
