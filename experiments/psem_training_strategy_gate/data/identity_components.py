from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from experiments.psem_training_strategy_gate.data.label_contract import load_contract
from experiments.psem_training_strategy_gate.data.provenance import (
    ProvenanceError,
    canonical_sha256,
    collect_prior_exposure,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.topology_census import (
    OFFICIAL_PRIMARY_TOPOLOGIES,
    _aggregate,
    _lower_bound_audit,
)

AUTHORITY_REF = "https://github.com/kapitalismho/PuriPuly-heart/issues/77"
AUTHORITY_PIN = "5778025c8aca1ea1cb7cd8fc41645b520ca1f9f749155b5f5daada32e940b559"
OFFICIAL_SPLIT_ROLES = (
    "PSEM-STRATEGY-TRAIN",
    "PSEM-STRATEGY-DEV",
    "PSEM-STRATEGY-EVAL",
)
NO_MODEL_FIELDS = (
    "model_predictions_consulted",
    "model_scores_consulted",
    "official_model_results_inspected",
    "official_model_training_performed",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_SPEAKER_IDENTITY_STATUSES = frozenset(
    {"known", "known_corpus_speaker_ids", "partially_or_fully_unknown"}
)


class IdentityGraphError(RuntimeError):
    pass


class _Components:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first


def _load_jsonl_objects(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityGraphError(f"invalid JSONL artifact: {path}") from exc
    if (not rows and not allow_empty) or any(
        not line or not isinstance(row, dict) for line, row in zip(lines, rows)
    ):
        raise IdentityGraphError(f"JSONL artifact must contain objects: {path}")
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityGraphError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise IdentityGraphError(f"JSON artifact must be an object: {path}")
    return value


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_values(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if value is None:
        return []
    if _nonempty_string(value):
        return [value]
    if (
        isinstance(value, list)
        and all(_nonempty_string(item) for item in value)
        and len(set(value)) == len(value)
    ):
        return sorted(value)
    raise IdentityGraphError(f"invalid optional identity field {field}")


def _speaker_identity_scope(row: dict[str, Any]) -> str:
    explicit_scope = row.get("speaker_identity_scope")
    if explicit_scope is not None:
        if explicit_scope not in {"corpus_global", "session_local"}:
            raise IdentityGraphError("source manifest speaker identity scope is invalid")
        return explicit_scope
    if row.get("corpus") == "AMI":
        return "corpus_global"
    if row.get("corpus") == "AliMeeting":
        return "session_local"
    if row.get("speaker_identity_status") == "known":
        return "corpus_global"
    return "session_local"


def _validate_source_row(row: dict[str, Any], contract_version: str, contract_sha: str) -> None:
    string_fields = (
        "source_id",
        "corpus",
        "session_id",
        "waveform_sha256",
        "annotation_sha256",
        "audio_ref",
        "speaker_identity_status",
        "eval_eligibility_reason",
    )
    if any(not _nonempty_string(row.get(field)) for field in string_fields):
        raise IdentityGraphError("source manifest identity fields are incomplete")
    if any(
        not _sha256_string(row.get(field))
        for field in ("waveform_sha256", "annotation_sha256")
    ):
        raise IdentityGraphError("source manifest content identities are invalid")
    if (
        row.get("contract_version") != contract_version
        or row.get("contract_document_sha256") != contract_sha
    ):
        raise IdentityGraphError("source manifest contract binding mismatch")
    speaker_ids = row.get("speaker_ids")
    unknown_agents = row.get("unknown_speaker_agents")
    unknown_count = row.get("unknown_speaker_count")
    speaker_identity_status = row.get("speaker_identity_status")
    if (
        speaker_identity_status not in SUPPORTED_SPEAKER_IDENTITY_STATUSES
        or not isinstance(speaker_ids, list)
        or any(not _nonempty_string(value) for value in speaker_ids)
        or len(set(speaker_ids)) != len(speaker_ids)
        or not isinstance(unknown_agents, list)
        or any(not _nonempty_string(value) for value in unknown_agents)
        or len(set(unknown_agents)) != len(unknown_agents)
        or not isinstance(unknown_count, int)
        or isinstance(unknown_count, bool)
        or unknown_count < 0
        or unknown_count != len(unknown_agents)
        or (not speaker_ids and not unknown_agents)
    ):
        raise IdentityGraphError("source manifest speaker identity fields are invalid")
    if (unknown_count > 0) != (
        speaker_identity_status == "partially_or_fully_unknown"
    ):
        raise IdentityGraphError("source manifest unknown identity state is inconsistent")
    _speaker_identity_scope(row)
    meeting_series = row.get("meeting_series")
    if meeting_series is not None and not _nonempty_string(meeting_series):
        raise IdentityGraphError("source manifest meeting series is invalid")
    exposed = row.get("selection_exposed")
    if not isinstance(exposed, bool):
        raise IdentityGraphError("source manifest exposure state is invalid")
    if exposed:
        if (
            row.get("eval_eligible") is not False
            or row.get("eval_eligibility_reason")
            != "forbidden_prior_selection_exposure"
        ):
            raise IdentityGraphError("prior exposure is not fail-closed for EVAL")
    elif (
        row.get("eval_eligible") is not None
        or row.get("eval_eligibility_reason")
        != "pending_identity_component_and_pretraining_overlap_audit"
    ):
        raise IdentityGraphError("unexposed EVAL status bypasses pending audits")
    for field in (
        "recurring_participant_ids",
        "source_recording_parent",
        "source_utterance_parent",
        "synthetic_parent_id",
        "synthetic_transformation_seed",
    ):
        _identity_values(row, field)


def _validate_calibration(
    data_dir: Path,
    calibration: dict[str, Any],
    source_ids: list[str],
    contract_version: str,
    contract_sha: str,
    contract_status: str,
) -> None:
    input_policy = calibration.get("input_policy")
    expected_inputs = {
        "source_manifest_sha256": sha256_file(data_dir / "source_manifest.jsonl"),
        "annotation_manifest_sha256": sha256_file(
            data_dir / "annotation_manifest.jsonl"
        ),
        "normalization_manifest_sha256": sha256_file(
            data_dir / "normalization_manifest.jsonl"
        ),
        "source_ids_sha256": canonical_sha256(sorted(source_ids)),
    }
    if (
        calibration.get("artifact_role") != "annotation_only_calibration"
        or calibration.get("authority_ref") != AUTHORITY_REF
        or calibration.get("authority_pin") != AUTHORITY_PIN
        or calibration.get("contract_version") != contract_version
        or calibration.get("contract_document_sha256") != contract_sha
        or calibration.get("contract_status") != contract_status
        or not isinstance(input_policy, dict)
        or any(input_policy.get(key) != value for key, value in expected_inputs.items())
        or any(input_policy.get(field) is not False for field in NO_MODEL_FIELDS)
        or input_policy.get("source")
        != "accepted natural source annotations only"
        or calibration.get("overall", {}).get("session_count") != len(source_ids)
    ):
        raise IdentityGraphError("annotation calibration binding mismatch")


def _validate_topology_census(
    data_dir: Path,
    source_rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    census: dict[str, Any],
    contract_version: str,
    contract_sha: str,
    contract_status: str,
) -> None:
    source_ids = [row["source_id"] for row in source_rows]
    source_by_id = {row["source_id"]: row for row in source_rows}
    topology_by_id = {row.get("source_id"): row for row in topology_rows}
    topology_identity_fields = {
        "artifact_role": "natural_topology_census_row",
        "contract_version": contract_version,
        "contract_document_sha256": contract_sha,
        "split_role": "UNASSIGNED_CANDIDATE",
        "component_id": None,
    }
    if (
        len(topology_by_id) != len(topology_rows)
        or set(topology_by_id) != set(source_by_id)
        or any(
            any(row.get(field) != value for field, value in topology_identity_fields.items())
            or row.get("corpus") != source_by_id[source_id]["corpus"]
            or row.get("session_id") != source_by_id[source_id]["session_id"]
            or row.get("source_waveform_sha256")
            != source_by_id[source_id]["waveform_sha256"]
            or row.get("annotation_sha256")
            != source_by_id[source_id]["annotation_sha256"]
            for source_id, row in topology_by_id.items()
        )
    ):
        raise IdentityGraphError("topology census inventory identity mismatch")
    input_manifests = census.get("input_manifests")
    counting_policy = census.get("counting_policy")
    census_model_policy = census.get("model_policy")
    expected_manifest_hashes = {
        "source_manifest_sha256": sha256_file(data_dir / "source_manifest.jsonl"),
        "annotation_manifest_sha256": sha256_file(
            data_dir / "annotation_manifest.jsonl"
        ),
        "normalization_manifest_sha256": sha256_file(
            data_dir / "normalization_manifest.jsonl"
        ),
        "annotation_calibration_sha256": sha256_file(
            data_dir / "annotation_calibration.json"
        ),
        "source_ids_sha256": canonical_sha256(sorted(source_ids)),
    }
    expected_counting_policy = {
        "official_primary_topology_precedence": list(
            OFFICIAL_PRIMARY_TOPOLOGIES
        ),
        "exclusive_primary_counting": True,
        "short_backchannel_member_handoffs_counted_separately": False,
        "old_r7_or_r7b_event_counts_used": False,
    }
    if (
        census.get("artifact_role") != "natural_topology_census"
        or census.get("authority_ref") != AUTHORITY_REF
        or census.get("authority_pin") != AUTHORITY_PIN
        or census.get("contract_version") != contract_version
        or census.get("contract_document_sha256") != contract_sha
        or census.get("contract_status") != contract_status
        or census.get("split_status") != "UNASSIGNED_PRE_IDENTITY_GRAPH"
        or census.get("component_status") != "PENDING_IDENTITY_GRAPH"
        or input_manifests != expected_manifest_hashes
        or census.get("topology_manifest_sha256")
        != sha256_file(data_dir / "topology_manifest.jsonl")
        or counting_policy != expected_counting_policy
        or not isinstance(census_model_policy, dict)
        or any(census_model_policy.get(field) is not False for field in NO_MODEL_FIELDS)
    ):
        raise IdentityGraphError("accepted topology census binding mismatch")
    sample_rate_hz = load_contract().sample_rate_hz
    expected_overall = _aggregate(topology_rows, sample_rate_hz)
    expected_by_corpus = {
        corpus: _aggregate(
            [row for row in topology_rows if row["corpus"] == corpus],
            sample_rate_hz,
        )
        for corpus in sorted({row["corpus"] for row in topology_rows})
    }
    if (
        census.get("overall") != expected_overall
        or census.get("by_corpus") != expected_by_corpus
        or census.get("by_split") != {"UNASSIGNED_CANDIDATE": expected_overall}
        or census.get("candidate_pool_lower_bound_audit")
        != _lower_bound_audit(expected_overall)
    ):
        raise IdentityGraphError("topology census aggregate mismatch")


def _validate_inputs(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_path = data_dir / "source_manifest.jsonl"
    prior_path = data_dir / "prior_exposure_manifest.jsonl"
    topology_path = data_dir / "topology_manifest.jsonl"
    census_path = data_dir / "topology_census.json"
    source_rows = _load_jsonl_objects(source_path)
    prior_rows = _load_jsonl_objects(prior_path, allow_empty=True)
    topology_rows = _load_jsonl_objects(topology_path)
    census = _load_json_object(census_path)
    calibration = _load_json_object(data_dir / "annotation_calibration.json")
    contract = load_contract()
    for row in source_rows:
        _validate_source_row(
            row,
            contract.contract_version,
            contract.document_sha256,
        )
    source_ids = [row["source_id"] for row in source_rows]
    if len(set(source_ids)) != len(source_ids):
        raise IdentityGraphError("source manifest identities must be unique")
    source_by_id = {row["source_id"]: row for row in source_rows}
    _validate_calibration(
        data_dir,
        calibration,
        source_ids,
        contract.contract_version,
        contract.document_sha256,
        contract.status,
    )
    _validate_topology_census(
        data_dir,
        source_rows,
        topology_rows,
        census,
        contract.contract_version,
        contract.document_sha256,
        contract.status,
    )
    prior_by_id = {row.get("source_id"): row for row in prior_rows}
    exposed_ids = {
        row["source_id"] for row in source_rows if row["selection_exposed"]
    }
    if len(prior_by_id) != len(prior_rows) or set(prior_by_id) != exposed_ids:
        raise IdentityGraphError("prior exposure inventory mismatch")
    try:
        reconstructed_prior = collect_prior_exposure(REPO_ROOT)
    except (OSError, json.JSONDecodeError, ProvenanceError) as exc:
        raise IdentityGraphError("historical prior exposure cannot be reconstructed") from exc
    reconstructed_source_ids = set(reconstructed_prior).intersection(source_by_id)
    if not reconstructed_source_ids.issubset(exposed_ids):
        raise IdentityGraphError("historical prior exposure is missing from inventory")
    compared_fields = (
        "corpus",
        "session_id",
        "meeting_series",
        "speaker_ids",
        "waveform_sha256",
        "annotation_sha256",
        "contract_version",
        "contract_document_sha256",
    )
    for source_id, prior in prior_by_id.items():
        source = source_by_id[source_id]
        prior_uses = source.get("prior_uses")
        evidence = prior.get("evidence")
        if (
            any(prior.get(field) != source.get(field) for field in compared_fields)
            or prior.get("selection_exposed") is not True
            or prior.get("eval_eligible") is not False
            or prior.get("reason") != "prior experimental selection exposure"
            or not isinstance(prior_uses, list)
            or not prior_uses
            or any(not _nonempty_string(value) for value in prior_uses)
            or len(set(prior_uses)) != len(prior_uses)
            or prior.get("prior_uses") != prior_uses
            or not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(item, dict)
                or not _nonempty_string(item.get("prior_use"))
                or not _nonempty_string(item.get("ref"))
                or not _sha256_string(item.get("sha256"))
                for item in evidence
            )
            or sorted(
                item["prior_use"] for item in evidence if isinstance(item, dict)
            )
            != sorted(prior_uses)
        ):
            raise IdentityGraphError("prior exposure identity binding mismatch")
        reconstructed = reconstructed_prior.get(source_id)
        if reconstructed is not None and (
            prior_uses != reconstructed["prior_uses"]
            or evidence != reconstructed["evidence"]
        ):
            raise IdentityGraphError("historical prior exposure evidence mismatch")
    if any(
        row.get("prior_uses") != []
        for row in source_rows
        if not row["selection_exposed"]
    ):
        raise IdentityGraphError("unexposed source contains prior-use evidence")
    return sorted(source_rows, key=lambda row: row["source_id"]), prior_rows, census


def _known_identities(row: dict[str, Any]) -> list[dict[str, str]]:
    corpus = row["corpus"]
    identities = [
        {
            "axis": "meeting_session",
            "value": f"{corpus}:{row['session_id']}",
        },
        {
            "axis": "waveform_identity",
            "value": row["waveform_sha256"],
        },
        {
            "axis": "annotation_identity",
            "value": row["annotation_sha256"],
        },
        {
            "axis": "source_recording_reference",
            "value": f"{corpus}:{row['audio_ref']}",
        },
    ]
    speaker_scope = _speaker_identity_scope(row)
    speaker_axis = (
        "known_speaker_identity"
        if speaker_scope == "corpus_global"
        else "session_local_speaker_label"
    )
    identities.extend(
        {
            "axis": speaker_axis,
            "value": (
                f"{corpus}:{speaker_id}"
                if speaker_scope == "corpus_global"
                else f"{corpus}:{row['session_id']}:{speaker_id}"
            ),
        }
        for speaker_id in sorted(row["speaker_ids"])
    )
    if row.get("meeting_series") is not None:
        identities.append(
            {
                "axis": "meeting_series",
                "value": f"{corpus}:{row['meeting_series']}",
            }
        )
    optional_axes = {
        "recurring_participant_ids": "recurring_participant",
        "source_recording_parent": "source_recording_parent",
        "source_utterance_parent": "source_utterance_parent",
        "synthetic_parent_id": "synthetic_parent",
        "synthetic_transformation_seed": "synthetic_transformation_seed",
    }
    for field, axis in optional_axes.items():
        identities.extend(
            {"axis": axis, "value": value}
            for value in _identity_values(row, field)
        )
    return sorted(identities, key=lambda item: (item["axis"], item["value"]))


def build_identity_graph(data_dir: Path) -> dict[str, Any]:
    source_rows, prior_rows, census = _validate_inputs(data_dir)
    source_ids = [row["source_id"] for row in source_rows]
    components = _Components(source_ids)
    identities_by_source = {
        row["source_id"]: _known_identities(row) for row in source_rows
    }
    sources_by_identity: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source_id, identities in identities_by_source.items():
        for identity in identities:
            sources_by_identity[(identity["axis"], identity["value"])].append(
                source_id
            )
    edge_reasons: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for (axis, value), connected_sources in sorted(sources_by_identity.items()):
        ordered_sources = sorted(set(connected_sources))
        for left, right in combinations(ordered_sources, 2):
            components.union(left, right)
            edge_reasons[(left, right)].append({"axis": axis, "value": value})
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        members_by_root[components.find(source_id)].append(source_id)
    component_id_by_source: dict[str, str] = {}
    component_rows = []
    source_by_id = {row["source_id"]: row for row in source_rows}
    for members in sorted(
        (sorted(values) for values in members_by_root.values()),
        key=lambda values: values[0],
    ):
        component_id = f"component-{canonical_sha256(members)}"
        for source_id in members:
            component_id_by_source[source_id] = component_id
        exposed = [
            source_id
            for source_id in members
            if source_by_id[source_id]["selection_exposed"]
        ]
        unresolved = [
            source_id
            for source_id in members
            if source_by_id[source_id]["unknown_speaker_count"] > 0
        ]
        shared_reasons = sorted(
            {
                (reason["axis"], reason["value"])
                for pair, reasons in edge_reasons.items()
                if pair[0] in members and pair[1] in members
                for reason in reasons
            }
        )
        component_rows.append(
            {
                "component_id": component_id,
                "source_ids": members,
                "source_count": len(members),
                "shared_identity_reasons": [
                    {"axis": axis, "value": value}
                    for axis, value in shared_reasons
                ],
                "unresolved_unknown_identity_source_ids": unresolved,
                "split_assignment_eligible": not unresolved,
                "selection_exposed_source_ids": exposed,
                "eval_forbidden": bool(exposed),
                "eval_forbidden_reason": (
                    "component_contains_prior_selection_exposure"
                    if exposed
                    else None
                ),
            }
        )
    node_rows = []
    for row in source_rows:
        source_id = row["source_id"]
        node_rows.append(
            {
                "source_id": source_id,
                "corpus": row["corpus"],
                "session_id": row["session_id"],
                "component_id": component_id_by_source[source_id],
                "speaker_identity_scope": _speaker_identity_scope(row),
                "known_identities": identities_by_source[source_id],
                "unknown_speaker_count": row["unknown_speaker_count"],
                "unknown_speaker_agents_local_only": sorted(
                    row["unknown_speaker_agents"]
                ),
                "unknown_identity_disjointness_claimed": False,
                "split_assignment_eligible": row["unknown_speaker_count"] == 0,
                "selection_exposed": row["selection_exposed"],
            }
        )
    edges = [
        {
            "left_source_id": left,
            "right_source_id": right,
            "reasons": sorted(
                reasons,
                key=lambda reason: (reason["axis"], reason["value"]),
            ),
        }
        for (left, right), reasons in sorted(edge_reasons.items())
    ]
    unknown_sources = [
        row["source_id"] for row in source_rows if row["unknown_speaker_count"]
    ]
    contract = load_contract()
    return {
        "schema_version": 1,
        "artifact_role": "identity_component_graph",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "input_artifacts": {
            "source_manifest_sha256": sha256_file(
                data_dir / "source_manifest.jsonl"
            ),
            "prior_exposure_manifest_sha256": sha256_file(
                data_dir / "prior_exposure_manifest.jsonl"
            ),
            "topology_manifest_sha256": sha256_file(
                data_dir / "topology_manifest.jsonl"
            ),
            "topology_census_sha256": sha256_file(
                data_dir / "topology_census.json"
            ),
            "source_ids_sha256": canonical_sha256(sorted(source_ids)),
            "census_annotation_calibration_sha256": census["input_manifests"][
                "annotation_calibration_sha256"
            ],
        },
        "identity_axis_policy": {
            "meeting_session": "corpus_scoped_exact_identity",
            "waveform_identity": "global_sha256_exact_identity",
            "annotation_identity": "global_sha256_exact_identity",
            "known_speaker_identity": "corpus_scoped_only_when_cross_session_stability_is_known",
            "session_local_speaker_label": "session_scoped_and_never_cross_session_linking",
            "recurring_participant": "global_exact_explicit_participant_identity",
            "meeting_series": "corpus_scoped_when_known",
            "source_recording_parent": "global_exact_explicit_parent_when_present",
            "source_utterance_parent": "global_exact_explicit_parent_when_present",
            "synthetic_parent_and_seed": "global_exact_when_materialized",
        },
        "identity_axis_coverage": {
            "meeting_session_known_source_count": len(source_rows),
            "waveform_identity_known_source_count": len(source_rows),
            "annotation_identity_known_source_count": len(source_rows),
            "globally_linkable_speaker_identity_source_count": sum(
                bool(row["speaker_ids"])
                and _speaker_identity_scope(row) == "corpus_global"
                for row in source_rows
            ),
            "session_local_speaker_label_source_count": sum(
                bool(row["speaker_ids"])
                and _speaker_identity_scope(row) == "session_local"
                for row in source_rows
            ),
            "recurring_participant_evidence_source_count": sum(
                (
                    bool(row["speaker_ids"])
                    and _speaker_identity_scope(row) == "corpus_global"
                )
                or bool(_identity_values(row, "recurring_participant_ids"))
                for row in source_rows
            ),
            "meeting_series_known_source_count": sum(
                row.get("meeting_series") is not None for row in source_rows
            ),
            "meeting_series_unknown_source_ids": [
                row["source_id"]
                for row in source_rows
                if row.get("meeting_series") is None
            ],
            "source_recording_reference_known_source_count": len(source_rows),
            "explicit_source_recording_parent_source_count": sum(
                bool(_identity_values(row, "source_recording_parent"))
                for row in source_rows
            ),
            "source_utterance_parent_source_count": sum(
                bool(_identity_values(row, "source_utterance_parent"))
                for row in source_rows
            ),
            "synthetic_parent_source_count": sum(
                bool(_identity_values(row, "synthetic_parent_id"))
                for row in source_rows
            ),
            "synthetic_transformation_seed_source_count": sum(
                bool(_identity_values(row, "synthetic_transformation_seed"))
                for row in source_rows
            ),
        },
        "unknown_identity_policy": {
            "local_unknown_labels_are_global_identities": False,
            "unknown_identities_are_claimed_disjoint": False,
            "official_split_assignment_requires_resolution": True,
        },
        "split_guard": {
            "official_roles": list(OFFICIAL_SPLIT_ROLES),
            "component_may_span_roles": False,
            "prior_selection_exposed_component_may_enter_eval": False,
        },
        "model_policy": {
            "model_predictions_consulted": False,
            "model_scores_consulted": False,
            "official_model_results_inspected": False,
            "official_model_training_performed": False,
        },
        "summary": {
            "source_count": len(source_rows),
            "component_count": len(component_rows),
            "singleton_component_count": sum(
                component["source_count"] == 1 for component in component_rows
            ),
            "multi_source_component_count": sum(
                component["source_count"] > 1 for component in component_rows
            ),
            "edge_count": len(edges),
            "globally_linkable_speaker_identity_count": len(
                {
                    identity["value"]
                    for identities in identities_by_source.values()
                    for identity in identities
                    if identity["axis"] == "known_speaker_identity"
                }
            ),
            "session_local_speaker_label_count": len(
                {
                    identity["value"]
                    for identities in identities_by_source.values()
                    for identity in identities
                    if identity["axis"] == "session_local_speaker_label"
                }
            ),
            "unknown_identity_source_count": len(unknown_sources),
            "unknown_identity_source_ids": unknown_sources,
            "prior_exposed_source_count": len(prior_rows),
            "eval_forbidden_component_count": sum(
                component["eval_forbidden"] for component in component_rows
            ),
        },
        "nodes": node_rows,
        "edges": edges,
        "components": component_rows,
    }


def validate_split_assignment(
    graph: dict[str, Any], assignments: dict[str, str], data_dir: Path
) -> None:
    if graph != build_identity_graph(data_dir):
        raise IdentityGraphError("identity graph does not match current artifacts")
    nodes = graph.get("nodes")
    components = graph.get("components")
    if (
        not isinstance(nodes, list)
        or not isinstance(components, list)
        or any(
            not isinstance(node, dict)
            or not _nonempty_string(node.get("source_id"))
            or not _nonempty_string(node.get("component_id"))
            or not isinstance(node.get("unknown_speaker_count"), int)
            or isinstance(node.get("unknown_speaker_count"), bool)
            or node["unknown_speaker_count"] < 0
            or not isinstance(node.get("selection_exposed"), bool)
            or node.get("split_assignment_eligible")
            != (node["unknown_speaker_count"] == 0)
            or node.get("unknown_identity_disjointness_claimed") is not False
            for node in nodes
        )
        or any(
            not isinstance(component, dict)
            or not isinstance(component.get("source_ids"), list)
            or not component["source_ids"]
            or not isinstance(component.get("split_assignment_eligible"), bool)
            or not isinstance(component.get("eval_forbidden"), bool)
            for component in components
        )
    ):
        raise IdentityGraphError("identity graph structure is invalid")
    node_by_id = {node.get("source_id"): node for node in nodes}
    if len(node_by_id) != len(nodes) or set(assignments) != set(node_by_id):
        raise IdentityGraphError("split assignment must cover the exact identity graph")
    if any(role not in OFFICIAL_SPLIT_ROLES for role in assignments.values()):
        raise IdentityGraphError("split assignment contains an unsupported role")
    component_by_id = {
        component.get("component_id"): component for component in components
    }
    component_sources = [
        source_id
        for component in components
        for source_id in component.get("source_ids", [])
    ]
    if (
        len(component_by_id) != len(components)
        or len(component_sources) != len(set(component_sources))
        or set(component_sources) != set(node_by_id)
        or any(
            node.get("component_id") not in component_by_id
            or node["source_id"]
            not in component_by_id[node["component_id"]].get("source_ids", [])
            for node in nodes
        )
    ):
        raise IdentityGraphError("identity graph component binding is invalid")
    for component_id, component in component_by_id.items():
        source_ids = component["source_ids"]
        unresolved_source_ids = sorted(
            source_id
            for source_id in source_ids
            if node_by_id[source_id]["unknown_speaker_count"] > 0
        )
        exposed_source_ids = sorted(
            source_id
            for source_id in source_ids
            if node_by_id[source_id]["selection_exposed"]
        )
        if (
            component.get("unresolved_unknown_identity_source_ids")
            != unresolved_source_ids
            or component["split_assignment_eligible"] != (
                not unresolved_source_ids
            )
            or component.get("selection_exposed_source_ids")
            != exposed_source_ids
            or component["eval_forbidden"] != bool(exposed_source_ids)
        ):
            raise IdentityGraphError("identity component guard state is inconsistent")
        roles = {assignments[source_id] for source_id in source_ids}
        if len(roles) != 1:
            raise IdentityGraphError(
                f"identity component spans official roles: {component_id}"
            )
        if unresolved_source_ids:
            raise IdentityGraphError(
                f"identity component contains unresolved unknown identity: {component_id}"
            )
        if (
            exposed_source_ids
            and "PSEM-STRATEGY-EVAL" in roles
        ):
            raise IdentityGraphError(
                f"prior-exposed identity component assigned to EVAL: {component_id}"
            )


def write_identity_graph(data_dir: Path, output_path: Path) -> None:
    graph = build_identity_graph(data_dir)
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_identity_graph(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
