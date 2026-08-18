from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.identity_components import (
    AUTHORITY_PIN,
    AUTHORITY_REF,
    IdentityGraphError,
    build_identity_graph,
    validate_split_assignment,
    write_identity_graph,
)
from experiments.psem_training_strategy_gate.data.label_contract import load_contract
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
    write_jsonl,
)
from experiments.psem_training_strategy_gate.data.topology_census import (
    OFFICIAL_PRIMARY_TOPOLOGIES,
    _aggregate,
    _lower_bound_audit,
)

DATA_DIR = Path(__file__).resolve().parents[1]


def _source_row(
    source_id: str,
    *,
    corpus: str = "TestCorpus",
    speaker_ids: list[str] | None = None,
    waveform_sha256: str | None = None,
    meeting_series: str | None = None,
    selection_exposed: bool = False,
    unknown_speaker_agents: list[str] | None = None,
    **optional_identities: object,
) -> dict[str, object]:
    contract = load_contract()
    session_id = source_id.removeprefix("test_")
    unknown_agents = unknown_speaker_agents or []
    return {
        "schema_version": 1,
        "source_id": source_id,
        "corpus": corpus,
        "session_id": session_id,
        "meeting_series": meeting_series,
        "speaker_ids": (
            speaker_ids if speaker_ids is not None else [f"speaker_{session_id}"]
        ),
        "unknown_speaker_agents": unknown_agents,
        "unknown_speaker_count": len(unknown_agents),
        "speaker_identity_status": (
            "partially_or_fully_unknown"
            if unknown_agents
            else "known_corpus_speaker_ids"
            if corpus == "AliMeeting"
            else "known"
        ),
        "audio_ref": f"audio/{session_id}.wav",
        "waveform_sha256": waveform_sha256
        or canonical_sha256({"source_id": source_id}),
        "annotation_sha256": canonical_sha256({"annotation": source_id}),
        "selection_exposed": selection_exposed,
        "prior_uses": ["issue_72"] if selection_exposed else [],
        "eval_eligible": False if selection_exposed else None,
        "eval_eligibility_reason": (
            "forbidden_prior_selection_exposure"
            if selection_exposed
            else "pending_identity_component_and_pretraining_overlap_audit"
        ),
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        **optional_identities,
    }


def _topology_row(source: dict[str, object]) -> dict[str, object]:
    contract = load_contract()
    return {
        "schema_version": 1,
        "artifact_role": "natural_topology_census_row",
        "source_id": source["source_id"],
        "corpus": source["corpus"],
        "session_id": source["session_id"],
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "source_waveform_sha256": source["waveform_sha256"],
        "annotation_sha256": source["annotation_sha256"],
        "scored_samples": 16000,
        "ambiguous_samples": 0,
        "unknown_identity_samples": 0,
        "masked_or_ambiguous_samples": 0,
        "stable_singleton_samples": 16000,
        "ongoing_overlap_samples": 0,
        "exclusive_primary_episode_count": 0,
        "primary_topology_counts": {
            topology: 0 for topology in OFFICIAL_PRIMARY_TOPOLOGIES
        },
        "micro_diagnostics": {
            "micro_gap_interval_count": 0,
            "micro_gap_samples": 0,
            "micro_overlap_interval_count": 0,
            "micro_overlap_samples": 0,
        },
        "mask_diagnostics": {
            "actual_transition_count": 0,
            "masked_transition_count": 0,
            "masked_transition_fraction": 0.0,
            "masked_transition_reasons": {},
            "diagnostic_masked_region_counts": {},
        },
        "split_role": "UNASSIGNED_CANDIDATE",
        "component_id": None,
    }


def _write_bundle(data_dir: Path, rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: str(row["source_id"]))
    write_jsonl(data_dir / "source_manifest.jsonl", ordered)
    write_jsonl(
        data_dir / "annotation_manifest.jsonl",
        ({"source_id": row["source_id"]} for row in ordered),
    )
    write_jsonl(
        data_dir / "normalization_manifest.jsonl",
        ({"source_id": row["source_id"]} for row in ordered),
    )
    write_jsonl(
        data_dir / "prior_exposure_manifest.jsonl",
        (
            {
                "schema_version": 1,
                "source_id": row["source_id"],
                "corpus": row["corpus"],
                "session_id": row["session_id"],
                "meeting_series": row["meeting_series"],
                "speaker_ids": row["speaker_ids"],
                "waveform_sha256": row["waveform_sha256"],
                "annotation_sha256": row["annotation_sha256"],
                "selection_exposed": True,
                "eval_eligible": False,
                "prior_uses": row["prior_uses"],
                "reason": "prior experimental selection exposure",
                "evidence": [
                    {
                        "prior_use": prior_use,
                        "ref": f"fixture/{prior_use}.json",
                        "sha256": canonical_sha256({"prior_use": prior_use}),
                    }
                    for prior_use in row["prior_uses"]
                ],
                "contract_version": row["contract_version"],
                "contract_document_sha256": row["contract_document_sha256"],
            }
            for row in ordered
            if row["selection_exposed"]
        ),
    )
    topology_rows = [_topology_row(row) for row in ordered]
    write_jsonl(data_dir / "topology_manifest.jsonl", topology_rows)
    contract = load_contract()
    calibration_path = data_dir / "annotation_calibration.json"
    source_ids = sorted(str(row["source_id"]) for row in ordered)
    calibration = {
        "artifact_role": "annotation_only_calibration",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "contract_status": contract.status,
        "input_policy": {
            "source_manifest_sha256": sha256_file(
                data_dir / "source_manifest.jsonl"
            ),
            "annotation_manifest_sha256": sha256_file(
                data_dir / "annotation_manifest.jsonl"
            ),
            "normalization_manifest_sha256": sha256_file(
                data_dir / "normalization_manifest.jsonl"
            ),
            "source_ids_sha256": canonical_sha256(source_ids),
            "source": "accepted natural source annotations only",
            "model_predictions_consulted": False,
            "model_scores_consulted": False,
            "official_model_results_inspected": False,
            "official_model_training_performed": False,
        },
        "overall": {"session_count": len(ordered)},
    }
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    overall = _aggregate(topology_rows, contract.sample_rate_hz)
    by_corpus = {
        corpus: _aggregate(
            [row for row in topology_rows if row["corpus"] == corpus],
            contract.sample_rate_hz,
        )
        for corpus in sorted({str(row["corpus"]) for row in topology_rows})
    }
    census = {
        "schema_version": 1,
        "artifact_role": "natural_topology_census",
        "authority_ref": AUTHORITY_REF,
        "authority_pin": AUTHORITY_PIN,
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "contract_status": contract.status,
        "input_manifests": {
            "source_manifest_sha256": sha256_file(
                data_dir / "source_manifest.jsonl"
            ),
            "annotation_manifest_sha256": sha256_file(
                data_dir / "annotation_manifest.jsonl"
            ),
            "normalization_manifest_sha256": sha256_file(
                data_dir / "normalization_manifest.jsonl"
            ),
            "source_ids_sha256": canonical_sha256(source_ids),
            "annotation_calibration_sha256": sha256_file(calibration_path),
        },
        "topology_manifest_sha256": sha256_file(
            data_dir / "topology_manifest.jsonl"
        ),
        "model_policy": {
            "model_predictions_consulted": False,
            "model_scores_consulted": False,
            "official_model_results_inspected": False,
            "official_model_training_performed": False,
        },
        "counting_policy": {
            "official_primary_topology_precedence": list(
                OFFICIAL_PRIMARY_TOPOLOGIES
            ),
            "exclusive_primary_counting": True,
            "short_backchannel_member_handoffs_counted_separately": False,
            "old_r7_or_r7b_event_counts_used": False,
        },
        "split_status": "UNASSIGNED_PRE_IDENTITY_GRAPH",
        "component_status": "PENDING_IDENTITY_GRAPH",
        "overall": overall,
        "by_corpus": by_corpus,
        "by_split": {"UNASSIGNED_CANDIDATE": overall},
        "candidate_pool_lower_bound_audit": _lower_bound_audit(overall),
    }
    (data_dir / "topology_census.json").write_text(
        json.dumps(census, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_checked_in_identity_graph_covers_the_accepted_census() -> None:
    graph = json.loads(
        (DATA_DIR / "identity_components.json").read_text(encoding="utf-8")
    )
    source_rows = [
        json.loads(line)
        for line in (DATA_DIR / "source_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    source_ids = sorted(row["source_id"] for row in source_rows)
    assert graph["artifact_role"] == "identity_component_graph"
    assert graph["input_artifacts"]["source_manifest_sha256"] == sha256_file(
        DATA_DIR / "source_manifest.jsonl"
    )
    assert graph["input_artifacts"]["topology_manifest_sha256"] == sha256_file(
        DATA_DIR / "topology_manifest.jsonl"
    )
    assert graph["input_artifacts"]["topology_census_sha256"] == sha256_file(
        DATA_DIR / "topology_census.json"
    )
    assert graph["input_artifacts"]["source_ids_sha256"] == canonical_sha256(
        source_ids
    )
    assert sorted(node["source_id"] for node in graph["nodes"]) == source_ids
    assert graph["summary"] == {
        "component_count": 28,
        "edge_count": 0,
        "eval_forbidden_component_count": 10,
        "globally_linkable_speaker_identity_count": 80,
        "multi_source_component_count": 0,
        "prior_exposed_source_count": 10,
        "session_local_speaker_label_count": 25,
        "singleton_component_count": 28,
        "source_count": 28,
        "unknown_identity_source_count": 0,
        "unknown_identity_source_ids": [],
    }
    assert all(
        component["component_id"]
        == f"component-{canonical_sha256(component['source_ids'])}"
        for component in graph["components"]
    )
    assert all(node["split_assignment_eligible"] for node in graph["nodes"])
    assert graph["identity_axis_coverage"] == {
        "annotation_identity_known_source_count": 28,
        "explicit_source_recording_parent_source_count": 0,
        "globally_linkable_speaker_identity_source_count": 20,
        "meeting_series_known_source_count": 20,
        "meeting_series_unknown_source_ids": [
            "alimeeting_R8001_M8004",
            "alimeeting_R8003_M8001",
            "alimeeting_R8007_M8010",
            "alimeeting_R8007_M8011",
            "alimeeting_R8008_M8013",
            "alimeeting_R8009_M8018",
            "alimeeting_R8009_M8019",
            "alimeeting_R8009_M8020",
        ],
        "meeting_session_known_source_count": 28,
        "recurring_participant_evidence_source_count": 20,
        "session_local_speaker_label_source_count": 8,
        "source_recording_reference_known_source_count": 28,
        "source_utterance_parent_source_count": 0,
        "synthetic_parent_source_count": 0,
        "synthetic_transformation_seed_source_count": 0,
        "waveform_identity_known_source_count": 28,
    }
    assert all(
        graph["model_policy"][field] is False
        for field in (
            "model_predictions_consulted",
            "model_scores_consulted",
            "official_model_results_inspected",
            "official_model_training_performed",
        )
    )


def test_shared_speaker_connects_sources_and_propagates_eval_forbidden(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(
            "test_a",
            speaker_ids=["shared"],
            selection_exposed=True,
        ),
        _source_row("test_b", speaker_ids=["shared"]),
    ]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    assert graph["summary"]["component_count"] == 1
    assert graph["edges"] == [
        {
            "left_source_id": "test_a",
            "right_source_id": "test_b",
            "reasons": [
                {
                    "axis": "known_speaker_identity",
                    "value": "TestCorpus:shared",
                }
            ],
        }
    ]
    component = graph["components"][0]
    assert component["source_ids"] == ["test_a", "test_b"]
    assert component["eval_forbidden"] is True
    assert component["selection_exposed_source_ids"] == ["test_a"]


def test_alimeeting_session_local_speaker_labels_never_connect_sessions(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(
            "test_a",
            corpus="AliMeeting",
            speaker_ids=["SPK1"],
        ),
        _source_row(
            "test_b",
            corpus="AliMeeting",
            speaker_ids=["SPK1"],
        ),
    ]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    assert graph["summary"]["component_count"] == 2
    assert graph["summary"]["globally_linkable_speaker_identity_count"] == 0
    assert graph["summary"]["session_local_speaker_label_count"] == 2
    assert graph["edges"] == []
    assert {
        identity["value"]
        for node in graph["nodes"]
        for identity in node["known_identities"]
        if identity["axis"] == "session_local_speaker_label"
    } == {"AliMeeting:a:SPK1", "AliMeeting:b:SPK1"}


@pytest.mark.parametrize(
    ("changes", "axis"),
    [
        ({"waveform_sha256": "f" * 64}, "waveform_identity"),
        ({"annotation_sha256": "e" * 64}, "annotation_identity"),
        (
            {"audio_ref": "audio/shared.wav"},
            "source_recording_reference",
        ),
        ({"meeting_series": "series"}, "meeting_series"),
        ({"source_recording_parent": "recording"}, "source_recording_parent"),
        ({"source_utterance_parent": "utterance"}, "source_utterance_parent"),
        ({"synthetic_parent_id": "parent"}, "synthetic_parent"),
        (
            {"synthetic_transformation_seed": "seed"},
            "synthetic_transformation_seed",
        ),
    ],
)
def test_each_shared_parent_axis_connects_sources(
    tmp_path: Path,
    changes: dict[str, object],
    axis: str,
) -> None:
    rows = [
        _source_row("test_a", **changes),
        _source_row("test_b", **changes),
    ]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    assert graph["summary"]["component_count"] == 1
    assert axis in {
        reason["axis"] for reason in graph["edges"][0]["reasons"]
    }


@pytest.mark.parametrize(
    "field",
    [
        "recurring_participant_ids",
        "source_recording_parent",
        "source_utterance_parent",
        "synthetic_parent_id",
        "synthetic_transformation_seed",
    ],
)
def test_explicit_global_identity_axes_connect_across_corpora(
    tmp_path: Path, field: str
) -> None:
    rows = [
        _source_row("test_a", corpus="CorpusA", **{field: "global-id"}),
        _source_row("test_b", corpus="CorpusB", **{field: "global-id"}),
    ]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    assert graph["summary"]["component_count"] == 1


def test_graph_rejects_unsupported_speaker_identity_status(tmp_path: Path) -> None:
    row = _source_row("test_a")
    row["speaker_identity_status"] = "unknown"
    _write_bundle(tmp_path, [row])
    with pytest.raises(IdentityGraphError, match="speaker identity fields"):
        build_identity_graph(tmp_path)


def test_graph_reconstructs_historical_exposure_independently(tmp_path: Path) -> None:
    row = _source_row("ami_IS1009a")
    _write_bundle(tmp_path, [row])
    with pytest.raises(IdentityGraphError, match="historical prior exposure is missing"):
        build_identity_graph(tmp_path)


def test_graph_rejects_changed_historical_exposure_evidence(tmp_path: Path) -> None:
    row = _source_row("ami_IS1009a", selection_exposed=True)
    _write_bundle(tmp_path, [row])
    with pytest.raises(IdentityGraphError, match="historical prior exposure evidence"):
        build_identity_graph(tmp_path)


def test_unknown_local_speaker_labels_never_prove_split_disjointness(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(
            "test_a",
            speaker_ids=[],
            unknown_speaker_agents=["local_a"],
        ),
        _source_row(
            "test_b",
            speaker_ids=[],
            unknown_speaker_agents=["local_b"],
        ),
    ]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    assert graph["summary"]["unknown_identity_source_ids"] == [
        "test_a",
        "test_b",
    ]
    assert all(
        node["unknown_identity_disjointness_claimed"] is False
        and node["split_assignment_eligible"] is False
        for node in graph["nodes"]
    )
    with pytest.raises(IdentityGraphError, match="unresolved unknown identity"):
        validate_split_assignment(
            graph,
            {
                "test_a": "PSEM-STRATEGY-TRAIN",
                "test_b": "PSEM-STRATEGY-TRAIN",
            },
            tmp_path,
        )


def test_split_assignment_rejects_a_component_spanning_roles(tmp_path: Path) -> None:
    rows = [
        _source_row("test_a", speaker_ids=["shared"]),
        _source_row("test_b", speaker_ids=["shared"]),
    ]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    with pytest.raises(IdentityGraphError, match="spans official roles"):
        validate_split_assignment(
            graph,
            {
                "test_a": "PSEM-STRATEGY-TRAIN",
                "test_b": "PSEM-STRATEGY-DEV",
            },
            tmp_path,
        )


def test_split_assignment_rejects_prior_exposure_in_eval(tmp_path: Path) -> None:
    rows = [_source_row("test_a", selection_exposed=True)]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    with pytest.raises(IdentityGraphError, match="assigned to EVAL"):
        validate_split_assignment(
            graph,
            {"test_a": "PSEM-STRATEGY-EVAL"},
            tmp_path,
        )


def test_graph_rejects_a_topology_identity_mismatch(tmp_path: Path) -> None:
    rows = [_source_row("test_a")]
    _write_bundle(tmp_path, rows)
    topology_rows = [
        json.loads(line)
        for line in (tmp_path / "topology_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    topology_rows[0]["annotation_sha256"] = "0" * 64
    write_jsonl(tmp_path / "topology_manifest.jsonl", topology_rows)
    census_path = tmp_path / "topology_census.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    census["topology_manifest_sha256"] = sha256_file(
        tmp_path / "topology_manifest.jsonl"
    )
    _write_json(census_path, census)
    with pytest.raises(IdentityGraphError, match="topology census inventory"):
        build_identity_graph(tmp_path)


def test_graph_rejects_a_stale_census_input_hash(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_source_row("test_a")])
    census_path = tmp_path / "topology_census.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    census["input_manifests"]["normalization_manifest_sha256"] = "0" * 64
    _write_json(census_path, census)
    with pytest.raises(IdentityGraphError, match="topology census binding"):
        build_identity_graph(tmp_path)


def test_graph_rejects_changed_census_metrics(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_source_row("test_a")])
    census_path = tmp_path / "topology_census.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    census["overall"]["scored_samples"] += 1
    _write_json(census_path, census)
    with pytest.raises(IdentityGraphError, match="census aggregate"):
        build_identity_graph(tmp_path)


def test_graph_rejects_model_contaminated_calibration(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_source_row("test_a")])
    calibration_path = tmp_path / "annotation_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["input_policy"]["model_predictions_consulted"] = True
    _write_json(calibration_path, calibration)
    census_path = tmp_path / "topology_census.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    census["input_manifests"]["annotation_calibration_sha256"] = sha256_file(
        calibration_path
    )
    _write_json(census_path, census)
    with pytest.raises(IdentityGraphError, match="calibration binding"):
        build_identity_graph(tmp_path)


def test_split_assignment_rejects_tampered_component_guards(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_source_row("test_a", selection_exposed=True)])
    graph = build_identity_graph(tmp_path)
    graph["components"][0]["eval_forbidden"] = False
    with pytest.raises(IdentityGraphError, match="current artifacts"):
        validate_split_assignment(
            graph,
            {"test_a": "PSEM-STRATEGY-EVAL"},
            tmp_path,
        )


def test_split_assignment_rejects_tampered_unknown_guards(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _source_row(
                "test_a",
                speaker_ids=[],
                unknown_speaker_agents=["local_a"],
            )
        ],
    )
    graph = build_identity_graph(tmp_path)
    graph["nodes"][0]["unknown_speaker_count"] = 0
    graph["nodes"][0]["split_assignment_eligible"] = True
    graph["components"][0]["unresolved_unknown_identity_source_ids"] = []
    graph["components"][0]["split_assignment_eligible"] = True
    with pytest.raises(IdentityGraphError, match="current artifacts"):
        validate_split_assignment(
            graph,
            {"test_a": "PSEM-STRATEGY-TRAIN"},
            tmp_path,
        )


def test_split_assignment_rejects_a_stale_graph(tmp_path: Path) -> None:
    rows = [_source_row("test_a")]
    _write_bundle(tmp_path, rows)
    graph = build_identity_graph(tmp_path)
    changed_rows = [_source_row("test_a", speaker_ids=["changed"])]
    write_jsonl(tmp_path / "source_manifest.jsonl", changed_rows)
    with pytest.raises(IdentityGraphError):
        validate_split_assignment(
            graph,
            {"test_a": "PSEM-STRATEGY-TRAIN"},
            tmp_path,
        )


def test_split_assignment_accepts_current_single_role_components(
    tmp_path: Path,
) -> None:
    _write_bundle(
        tmp_path,
        [_source_row("test_a"), _source_row("test_b")],
    )
    graph = build_identity_graph(tmp_path)
    validate_split_assignment(
        graph,
        {
            "test_a": "PSEM-STRATEGY-TRAIN",
            "test_b": "PSEM-STRATEGY-TRAIN",
        },
        tmp_path,
    )


def test_identity_graph_output_is_deterministic(tmp_path: Path) -> None:
    rows = [
        _source_row("test_a", speaker_ids=["shared"]),
        _source_row("test_b", speaker_ids=["shared"]),
    ]
    _write_bundle(tmp_path, rows)
    output = tmp_path / "identity_components.json"
    write_identity_graph(tmp_path, output)
    first = output.read_bytes()
    write_identity_graph(tmp_path, output)
    assert output.read_bytes() == first
