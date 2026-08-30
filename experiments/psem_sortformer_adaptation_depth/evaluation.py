from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    SessionExamples,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
    aggregate_rows,
    session_row,
)
from experiments.psem_sortformer_adaptation_depth.authority_registry import (
    register_execution,
    require_registered_execution,
)
from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
    align_native_predictions,
    mapping_from_action_probabilities,
    native_episode_timeline,
)
from experiments.psem_sortformer_adaptation_depth.preflight import (
    SOURCE_MANIFEST_PATH,
    _safe_external_output_root,
    canonical_sha256,
)
from experiments.psem_sortformer_adaptation_depth.protocol import (
    _eval_registry_marker,
    bind_payload,
    validate_eval_authorization,
)
from experiments.psem_sortformer_adaptation_depth.receipts import (
    build_data_split_receipt,
    evaluator_reconstruction_contract,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE, EVAL_ROLE

THRESHOLDS = (0.5,)
CONFIRMATION_MS = (500,)
PRIMARY_THRESHOLD = 0.5
PRIMARY_CONFIRMATION_MS = 500
FRAME_SAMPLES = 1280
EVIDENCE_DELAY_SAMPLES = 16640


class EvaluationError(RuntimeError):
    pass


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _corpus(session: SessionExamples) -> str:
    if session.source_family == "ami_mix_headset":
        return "AMI"
    if session.source_family == "alimeeting_far_ch0":
        return "AliMeeting"
    raise EvaluationError(f"unknown frozen source family: {session.source_family}")


def validate_prediction_set(
    value: Mapping[str, Any],
    eval_authorization: Mapping[str, Any] | None = None,
    *,
    _artifact_cache: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    require_registered_execution("prediction-set", value)
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    if (
        value.get("artifact_role") != "psem_sortformer_prediction_set"
        or value.get("payload_sha256") != canonical_sha256(payload)
        or value.get("split_role") not in {DEV_ROLE, EVAL_ROLE}
        or value.get("arm") not in {"F0-FROZEN-FLOAT", "H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}
        or (value.get("arm") == "F0-FROZEN-FLOAT" and value.get("seed") is not None)
        or (value.get("arm") != "F0-FROZEN-FLOAT" and value.get("seed") != 7301)
        or value.get("algorithmic_evidence_delay_samples") != EVIDENCE_DELAY_SAMPLES
        or value.get("native_frame_samples") != FRAME_SAMPLES
        or type(value.get("total_parameters")) is not int
        or value["total_parameters"] <= 0
        or type(value.get("trainable_parameters")) is not int
        or value["trainable_parameters"] < 0
        or value["trainable_parameters"] > value["total_parameters"]
        or (value.get("arm") == "F0-FROZEN-FLOAT" and value["trainable_parameters"] != 0)
        or (value.get("arm") != "F0-FROZEN-FLOAT" and value["trainable_parameters"] <= 0)
        or value.get("checkpoint_sha256")
        != "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
        or not isinstance(value.get("runtime_identity"), Mapping)
        or value.get("runtime_identity_sha256") != canonical_sha256(value["runtime_identity"])
        or not isinstance(value.get("parameter_policy"), Mapping)
        or value.get("parameter_policy_sha256") != canonical_sha256(value["parameter_policy"])
        or not isinstance(value.get("candidate_git_head"), str)
        or len(value["candidate_git_head"]) != 40
        or not isinstance(value.get("candidate_code_identity_sha256"), str)
        or len(value["candidate_code_identity_sha256"]) != 64
        or not isinstance(value.get("candidate_artifact_sha256s"), Mapping)
    ):
        raise EvaluationError("prediction-set identity or timing contract is invalid")
    trained_checkpoint = value.get("trained_checkpoint_sha256")
    trained_receipt = value.get("trained_checkpoint_receipt_sha256")
    if value["arm"] == "F0-FROZEN-FLOAT":
        if trained_checkpoint is not None or trained_receipt is not None:
            raise EvaluationError("frozen-float prediction set carries a trained checkpoint")
    elif any(
        not isinstance(item, str) or len(item) != 64
        for item in (trained_checkpoint, trained_receipt)
    ):
        raise EvaluationError("trained prediction set lacks checkpoint provenance")
    from experiments.psem_sortformer_adaptation_depth.execution import (
        validate_current_candidate_identity,
    )

    validate_current_candidate_identity(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_candidate_code_identity",
            "git_head": value["candidate_git_head"],
            "worktree_clean": True,
            "artifact_sha256s": dict(value["candidate_artifact_sha256s"]),
            "payload_sha256": value["candidate_code_identity_sha256"],
        }
    )
    split = build_data_split_receipt()
    output_root_value = value.get("experiment_output_root")
    registry_root_value = value.get("protocol_registry_root")
    output_root = Path(output_root_value).resolve() if isinstance(output_root_value, str) else None
    registry_root = (
        Path(registry_root_value).resolve() if isinstance(registry_root_value, str) else None
    )
    if (
        output_root is None
        or registry_root is None
        or not _safe_external_output_root(output_root)
        or not _safe_external_output_root(registry_root)
        or output_root == registry_root
    ):
        raise EvaluationError("prediction-set output or protocol registry root is invalid")
    if value["split_role"] == DEV_ROLE:
        if eval_authorization is not None or value.get("eval_authorization_sha256") is not None:
            raise EvaluationError("DEV prediction sets cannot carry EVAL authorization")
    else:
        if eval_authorization is None:
            raise EvaluationError("EVAL evaluation requires the single-open authorization")
        validate_eval_authorization(eval_authorization)
        authorization = {
            key: item for key, item in eval_authorization.items() if key != "payload_sha256"
        }
        candidates = [
            row
            for row in authorization.get("candidate_set", [])
            if row.get("arm") == value.get("arm") and row.get("seed") == value.get("seed")
        ]
        if (
            authorization.get("evaluation_roles") != [EVAL_ROLE]
            or authorization.get("eval_open_count") != 1
            or authorization.get("eval_used_for_development") is not False
            or len(candidates) != 1
            or value.get("eval_authorization_sha256") != eval_authorization.get("payload_sha256")
            or value.get("candidate_git_head") != authorization.get("candidate_git_head")
            or value.get("candidate_code_identity_sha256")
            != authorization.get("candidate_code_identity_sha256")
            or value.get("trained_checkpoint_sha256") != candidates[0].get("checkpoint_sha256")
            or value.get("trained_checkpoint_receipt_sha256")
            != candidates[0].get("checkpoint_receipt_sha256")
            or value.get("experiment_output_root") != authorization.get("experiment_output_root")
            or value.get("protocol_registry_root") != authorization.get("protocol_registry_root")
        ):
            raise EvaluationError("EVAL prediction set differs from the frozen authorization")
    expected_sources = split["source_ids_by_role"][value["split_role"]]
    descriptors = value.get("sources")
    source_ids = (
        [row.get("source_id") for row in descriptors] if isinstance(descriptors, list) else []
    )
    if source_ids != sorted(expected_sources) or len(set(source_ids)) != len(source_ids):
        raise EvaluationError("prediction-set source coverage differs from the frozen split")
    for descriptor in descriptors:
        path_value = descriptor.get("path")
        path = Path(path_value) if isinstance(path_value, str) else None
        expected_path = (
            output_root
            / "predictions"
            / value["split_role"]
            / value["arm"]
            / str(value.get("seed") or "f0")
            / f"{descriptor.get('source_id')}.jsonl"
        ).resolve()
        raw = path.read_bytes() if path is not None and path.is_file() else None
        if (
            path is None
            or not path.is_absolute()
            or not path.is_file()
            or path.resolve() != expected_path
            or raw is None
            or descriptor.get("sha256") != hashlib.sha256(raw).hexdigest()
            or descriptor.get("size_bytes") != len(raw)
            or not isinstance(descriptor.get("row_count"), int)
            or descriptor["row_count"] <= 0
        ):
            raise EvaluationError(f"prediction artifact is absent or changed: {path_value}")
        if _artifact_cache is not None:
            _artifact_cache[str(path.resolve())] = raw
    mappings = value.get("mapping_receipts")
    source_waveform_sha256s = {
        row["source_id"]: row["waveform_sha256"]
        for row in (
            json.loads(line)
            for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        )
    }
    if (
        not isinstance(mappings, list)
        or [row.get("source_id") for row in mappings] != sorted(expected_sources)
        or any(
            not isinstance(row.get("mappings"), list)
            or row.get("source_waveform_sha256")
            != source_waveform_sha256s.get(row.get("source_id"))
            or row.get("episode_count") != len(row["mappings"])
            or row.get("mapped_episode_count")
            != sum(item.get("status") == "mapped" for item in row["mappings"])
            for row in mappings
        )
    ):
        raise EvaluationError("prediction-set oracle mapping receipts are incomplete")
    return payload


def _load_prediction_rows(
    descriptor: Mapping[str, Any],
    prediction_set: Mapping[str, Any],
    raw: bytes | None = None,
) -> list[dict[str, Any]]:
    path = Path(str(descriptor["path"]))
    rows = []
    try:
        lines = (path.read_bytes() if raw is None else raw).decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationError("prediction artifact bytes are unreadable") from exc
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError("prediction artifact contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise EvaluationError("prediction rows must be JSON objects")
        rows.append(value)
    source_id = descriptor["source_id"]
    if len(rows) != descriptor["row_count"]:
        raise EvaluationError(f"prediction row count differs: {source_id}")
    for index, row in enumerate(rows):
        start = index * FRAME_SAMPLES
        numeric = [
            *row.get("raw_sortformer_activity_logits", []),
            row.get("raw_anchor_present_logit"),
            row.get("raw_replacement_evidence_logit"),
        ]
        expected_provenance = {
            "split_role": prediction_set["split_role"],
            "arm": prediction_set["arm"],
            "seed": prediction_set.get("seed"),
            "source_waveform_sha256": next(
                item["source_waveform_sha256"]
                for item in prediction_set["mapping_receipts"]
                if item["source_id"] == source_id
            ),
            "base_checkpoint_sha256": prediction_set["checkpoint_sha256"],
            "trained_checkpoint_sha256": prediction_set.get("trained_checkpoint_sha256"),
            "trained_checkpoint_receipt_sha256": prediction_set.get(
                "trained_checkpoint_receipt_sha256"
            ),
            "runtime_identity_sha256": prediction_set["runtime_identity_sha256"],
            "parameter_policy_sha256": prediction_set["parameter_policy_sha256"],
            "candidate_git_head": prediction_set["candidate_git_head"],
            "candidate_code_identity_sha256": prediction_set["candidate_code_identity_sha256"],
            "experiment_output_root": prediction_set["experiment_output_root"],
        }
        if (
            row.get("schema_version") != 1
            or row.get("artifact_role") != "psem_sortformer_frame_prediction"
            or row.get("source_id") != source_id
            or row.get("source_frame_start_sample") != start
            or row.get("source_frame_end_sample") != start + FRAME_SAMPLES
            or row.get("model_evidence_frontier_source_sample") != start + EVIDENCE_DELAY_SAMPLES
            or row.get("oracle_anchor_slot") not in {0, 1, 2, 3}
            or row.get("slot_alive") != [True, True, True, True]
            or row.get("state_reset") is not (index == 0)
            or len(row.get("raw_sortformer_activity_logits", [])) != 4
            or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(item)
                for item in numeric
            )
            or any(row.get(field) != expected for field, expected in expected_provenance.items())
        ):
            raise EvaluationError(f"prediction row violates native timing semantics: {source_id}")
    return rows


def _adapt_session(
    session: SessionExamples,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[SessionExamples, np.ndarray, np.ndarray, dict[str, Any]]:
    expected_native_episodes = native_episode_timeline(session.reference, len(rows))
    reported_native_episodes = tuple(row.get("anchor_episode_id") for row in rows)
    if reported_native_episodes != expected_native_episodes:
        raise EvaluationError(
            f"prediction native anchor episodes differ from #99: {session.source_id}"
        )
    aligned = align_native_predictions(session, [dict(row) for row in rows])
    probabilities = aligned["probabilities"]
    alive = aligned["alive"]
    reset = aligned["reset"]
    frontiers = aligned["frontiers"]
    slots_by_episode, mapping_rows = mapping_from_action_probabilities(
        session, probabilities, alive
    )
    expected_native_slots = np.asarray(
        [slots_by_episode.get(episode_id, 0) for episode_id in expected_native_episodes],
        dtype=np.int64,
    )
    reported_native_slots = np.asarray([row["oracle_anchor_slot"] for row in rows], dtype=np.int64)
    if not np.array_equal(reported_native_slots, expected_native_slots):
        raise EvaluationError(
            f"prediction carries a stale oracle slot mapping: {session.source_id}"
        )
    action_episode_ids = [
        None if str(value) in {"", "None"} else str(value) for value in session.episode_ids
    ]
    action_slots = np.asarray(
        [slots_by_episode.get(episode_id, 0) for episode_id in action_episode_ids],
        dtype=np.int64,
    )
    selected_anchor = probabilities[np.arange(len(probabilities)), action_slots]
    anchor_logits = aligned["anchor_logits"]
    replacement_logits = aligned["replacement_logits"]
    mapped = sum(row["status"] == "mapped" for row in mapping_rows)
    adapted = replace(
        session,
        probabilities=probabilities,
        alive=alive,
        reset=reset,
        frontiers=frontiers,
        mapping_records=tuple(mapping_rows),
    )
    diagnostics = {
        "episode_count": len(session.reference.episodes),
        "mapped_episode_count": mapped,
        "mapping_coverage": (
            mapped / len(session.reference.episodes) if session.reference.episodes else 1.0
        ),
        "unmapped_episode_count": len(session.reference.episodes) - mapped,
        "slot_instability_count": 0,
        "reset_exposure_count": int(aligned["native_reset"].sum()),
        "unexpected_reset_count": int(aligned["native_reset"][1:].sum()),
        "mapping_rows": mapping_rows,
    }
    return (
        adapted,
        selected_anchor,
        np.column_stack((anchor_logits, replacement_logits)),
        diagnostics,
    )


def _dropout_runs(mask: np.ndarray, frame_ms: int = 80) -> dict[str, Any]:
    runs = []
    start = None
    for index, active in enumerate(np.append(mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((index - start) * frame_ms)
            start = None
    return {
        f"sustained_{duration}_ms_count": sum(value >= duration for value in runs)
        for duration in (100, 300, 500)
    } | {"dropout_run_durations_ms": runs}


def _frame_diagnostics(
    session: SessionExamples,
    anchor_logits: np.ndarray,
) -> dict[str, Any]:
    usable = np.logical_and(session.valid, np.logical_not(session.masked))
    predicted_anchor = _sigmoid(anchor_logits) >= 0.5
    selectors = {
        "anchor_only": np.logical_and(session.anchor_present, np.logical_not(session.overlap)),
        "anchor_with_overlap": np.logical_and(session.anchor_present, session.overlap),
        "active_anchor_absent": session.target,
    }
    result = {}
    for name, selector in selectors.items():
        selected = np.logical_and(usable, selector)
        support = int(selected.sum())
        if name == "active_anchor_absent":
            successes = int(np.logical_and(selected, np.logical_not(predicted_anchor)).sum())
        else:
            successes = int(np.logical_and(selected, predicted_anchor).sum())
        result[name] = {
            "support_frames": support,
            "success_frames": successes,
            "recall": successes / support if support else None,
        }
    overlap_dropout = np.logical_and.reduce(
        (usable, session.anchor_present, session.overlap, np.logical_not(predicted_anchor))
    )
    result["gt_overlap_anchor_dropout"] = _dropout_runs(overlap_dropout)
    return result


def _decision_metrics(aggregate: Mapping[str, Any]) -> dict[str, float]:
    hours = float(aggregate["active_speech_hours"])
    if hours <= 0:
        raise EvaluationError("decision view has no active-speech exposure")
    return {
        "contamination": float(
            aggregate["exclusive_other_contamination_seconds_per_active_speech_hour"]
        ),
        "false_cuts": float(aggregate["false_cut_count"]) / hours,
        "missed_replacements": float(aggregate["missed_replacement_count"]) / hours,
    }


def _view(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_rows(rows)
    return {"metrics": _decision_metrics(aggregate), "full_metrics": aggregate}


def _equal_corpus_view(corpus_views: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    full = {corpus: value["full_metrics"] for corpus, value in corpus_views.items()}

    def mean_field(field: str) -> float | None:
        values = [value.get(field) for value in full.values()]
        return (
            sum(float(value) for value in values) / len(values)
            if values and all(_finite(value) for value in values)
            else None
        )

    def mean_quantiles(field: str) -> dict[str, float | None]:
        return {quantile: mean_field(f"{field}.{quantile}") for quantile in ("p50", "p90")}

    full_with_flattened = {
        corpus: {
            **value,
            **{
                f"{field}.{quantile}": value[field].get(quantile)
                for field in ("replacement_emit_delay_ms", "backdated_boundary_error_ms")
                for quantile in ("p50", "p90")
            },
        }
        for corpus, value in full.items()
    }
    full = full_with_flattened
    topology = {
        topology_name: {
            field: (
                sum(float(value) for value in values) / len(values)
                if values and all(_finite(value) for value in values)
                else None
            )
            for field in next(iter(full.values()))["topology"][topology_name]
            for values in [
                [
                    corpus_value["topology"][topology_name].get(field)
                    for corpus_value in full.values()
                ]
            ]
        }
        for topology_name in next(iter(full.values()))["topology"]
    }
    return {
        "metrics": {
            metric: sum(float(corpus_views[corpus]["metrics"][metric]) for corpus in corpus_views)
            / len(corpus_views)
            for metric in ("contamination", "false_cuts", "missed_replacements")
        },
        "full_metrics": {
            "speaker_induced_cut_count_per_active_speech_hour": mean_field(
                "speaker_induced_cut_count_per_active_speech_hour"
            ),
            "exclusive_other_contamination_seconds_per_active_speech_hour": mean_field(
                "exclusive_other_contamination_seconds_per_active_speech_hour"
            ),
            "false_cut_count_per_active_speech_hour": sum(
                float(corpus_views[corpus]["metrics"]["false_cuts"]) for corpus in corpus_views
            )
            / len(corpus_views),
            "missed_replacement_count_per_active_speech_hour": sum(
                float(corpus_views[corpus]["metrics"]["missed_replacements"])
                for corpus in corpus_views
            )
            / len(corpus_views),
            "replacement_emit_delay_ms": mean_quantiles("replacement_emit_delay_ms"),
            "backdated_boundary_error_ms": mean_quantiles("backdated_boundary_error_ms"),
            "overlap_return_preservation_rate": mean_field("overlap_return_preservation_rate"),
            "overlap_takeover_success_rate": mean_field("overlap_takeover_success_rate"),
            "topology": topology,
            "corpus_active_speech_hours": {
                corpus: value["active_speech_hours"] for corpus, value in full.items()
            },
        },
        "aggregation": "unweighted_mean_of_AMI_and_AliMeeting_rate_metrics",
    }


def _aggregate_frame_diagnostics(
    rows: Mapping[str, Mapping[str, Any]],
    corpora: Mapping[str, str],
) -> dict[str, Any]:
    def aggregate(source_ids: Sequence[str]) -> dict[str, Any]:
        result = {}
        for slice_name in ("anchor_only", "anchor_with_overlap", "active_anchor_absent"):
            support = sum(
                int(rows[source_id][slice_name]["support_frames"]) for source_id in source_ids
            )
            successes = sum(
                int(rows[source_id][slice_name]["success_frames"]) for source_id in source_ids
            )
            result[slice_name] = {
                "support_frames": support,
                "success_frames": successes,
                "recall": successes / support if support else None,
            }
        result["gt_overlap_anchor_dropout"] = {
            f"sustained_{duration}_ms_count": sum(
                int(rows[source_id]["gt_overlap_anchor_dropout"][f"sustained_{duration}_ms_count"])
                for source_id in source_ids
            )
            for duration in (100, 300, 500)
        }
        return result

    corpus_specific = {
        corpus: aggregate([source_id for source_id in sorted(rows) if corpora[source_id] == corpus])
        for corpus in ("AMI", "AliMeeting")
    }
    equal_corpus = {}
    for slice_name in ("anchor_only", "anchor_with_overlap", "active_anchor_absent"):
        values = [corpus_specific[corpus][slice_name]["recall"] for corpus in corpus_specific]
        equal_corpus[slice_name] = {
            "recall": (
                sum(float(value) for value in values) / len(values)
                if all(_finite(value) for value in values)
                else None
            )
        }
    equal_corpus["gt_overlap_anchor_dropout"] = {
        f"mean_sustained_{duration}_ms_count": sum(
            corpus_specific[corpus]["gt_overlap_anchor_dropout"][f"sustained_{duration}_ms_count"]
            for corpus in corpus_specific
        )
        / len(corpus_specific)
        for duration in (100, 300, 500)
    }
    return {
        "pooled": aggregate(sorted(rows)),
        "equal_corpus": equal_corpus,
        "corpus_specific": corpus_specific,
    }


def _aggregate_mapping_diagnostics(
    rows: Mapping[str, Mapping[str, Any]],
    corpora: Mapping[str, str],
) -> dict[str, Any]:
    def aggregate(source_ids: Sequence[str]) -> dict[str, Any]:
        episodes = sum(int(rows[source_id]["episode_count"]) for source_id in source_ids)
        mapped = sum(int(rows[source_id]["mapped_episode_count"]) for source_id in source_ids)
        return {
            "episode_count": episodes,
            "mapped_episode_count": mapped,
            "mapping_coverage": mapped / episodes if episodes else 1.0,
            "slot_instability_count": sum(
                int(rows[source_id]["slot_instability_count"]) for source_id in source_ids
            ),
            "reset_exposure_count": sum(
                int(rows[source_id]["reset_exposure_count"]) for source_id in source_ids
            ),
            "unexpected_reset_count": sum(
                int(rows[source_id]["unexpected_reset_count"]) for source_id in source_ids
            ),
        }

    corpus_specific = {
        corpus: aggregate([source_id for source_id in sorted(rows) if corpora[source_id] == corpus])
        for corpus in ("AMI", "AliMeeting")
    }
    return {
        "pooled": aggregate(sorted(rows)),
        "equal_corpus": {
            field: sum(float(corpus_specific[corpus][field]) for corpus in corpus_specific)
            / len(corpus_specific)
            for field in (
                "mapping_coverage",
                "slot_instability_count",
                "reset_exposure_count",
                "unexpected_reset_count",
            )
        },
        "corpus_specific": corpus_specific,
    }


def _mapping_receipt_matches(
    reported: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]
) -> bool:
    rows = reported.get("mappings")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    for left, right in zip(rows, expected, strict=True):
        if any(
            left.get(field) != right.get(field)
            for field in (
                "anchor_episode_id",
                "status",
                "slot_index",
                "support_frame_count",
            )
        ):
            return False
        left_scores = left.get("support_scores")
        right_scores = right.get("support_scores")
        if (
            not isinstance(left_scores, list)
            or len(left_scores) != 4
            or any(
                not _finite(a) or not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
                for a, b in zip(left_scores, right_scores, strict=True)
            )
        ):
            return False
    return True


def evaluate_prediction_set(
    value: Mapping[str, Any],
    eval_authorization: Mapping[str, Any] | None = None,
    *,
    historical_replay: bool = False,
) -> dict[str, Any]:
    evaluator_reconstruction_contract()
    artifact_cache: dict[str, bytes] = {}
    prediction_set = validate_prediction_set(
        value, eval_authorization, _artifact_cache=artifact_cache
    )
    split_role = prediction_set["split_role"]
    if split_role == DEV_ROLE and _eval_registry_marker().exists() and not historical_replay:
        raise EvaluationError("DEV evaluation is sealed after the canonical EVAL open")
    arm = prediction_set["arm"]
    seed = prediction_set.get("seed")
    sessions = {
        session.source_id: session
        for session in load_sessions(validate_mapping_ledger=False)
        if session.manifest["old_v2_role"] == split_role
    }
    descriptors = {row["source_id"]: row for row in prediction_set["sources"]}
    mapping_receipts = {row["source_id"]: row for row in prediction_set["mapping_receipts"]}
    if set(sessions) != set(descriptors):
        raise EvaluationError("prediction set and #99 source snapshot coverage differ")
    source_material = {}
    mapping_diagnostics = {}
    frame_diagnostics = {}
    for source_id in sorted(sessions):
        descriptor = descriptors[source_id]
        rows = _load_prediction_rows(
            descriptor,
            prediction_set,
            artifact_cache[str(Path(str(descriptor["path"])).resolve())],
        )
        adapted, selected_anchor, psem_logits, mapping = _adapt_session(sessions[source_id], rows)
        reported_mapping = mapping_receipts[source_id]
        expected_mappings = mapping.pop("mapping_rows")
        if (
            not _mapping_receipt_matches(reported_mapping, expected_mappings)
            or reported_mapping.get("episode_count") != mapping["episode_count"]
            or reported_mapping.get("mapped_episode_count") != mapping["mapped_episode_count"]
        ):
            raise EvaluationError(
                "prediction mapping receipt differs from current checkpoint outputs"
            )
        scores = 1.0 - selected_anchor if arm == "F0-FROZEN-FLOAT" else _sigmoid(psem_logits[:, 1])
        source_material[source_id] = (adapted, scores)
        mapping_diagnostics[source_id] = mapping
        frame_diagnostics[source_id] = _frame_diagnostics(adapted, psem_logits[:, 0])
    per_source_rows = []
    frontier = []
    for threshold in THRESHOLDS:
        for confirmation in CONFIRMATION_MS:
            cell_rows = []
            for source_id, (session, scores) in source_material.items():
                row = session_row(
                    session,
                    scores,
                    condition=arm,
                    probe_class=(
                        "float_simple_anchor_projection"
                        if arm == "F0-FROZEN-FLOAT"
                        else "causal_psem_head"
                    ),
                    threshold=threshold,
                    confirmation_ms=confirmation,
                    time_condition="causal",
                )
                row["arm"] = arm
                row["seed"] = seed
                row["split_role"] = split_role
                row["corpus"] = _corpus(session)
                cell_rows.append(row)
                per_source_rows.append(row)
            corpus_views = {
                corpus: _view([row for row in cell_rows if row["corpus"] == corpus])
                for corpus in ("AMI", "AliMeeting")
            }
            frontier.append(
                {
                    "threshold": threshold,
                    "confirmation_ms": confirmation,
                    "views": {
                        "pooled": _view(cell_rows),
                        **corpus_views,
                    },
                }
            )
    primary_rows = [
        row
        for row in per_source_rows
        if row["threshold"] == PRIMARY_THRESHOLD
        and row["confirmation_ms"] == PRIMARY_CONFIRMATION_MS
    ]
    per_source_primary = []
    for row in sorted(primary_rows, key=lambda item: item["source_id"]):
        metrics = row["metrics"]
        hours = float(metrics["active_speech_seconds"]) / 3600.0
        full_metrics = aggregate_rows([row])
        per_source_primary.append(
            {
                "source_id": row["source_id"],
                "corpus": row["corpus"],
                "metrics": {
                    "contamination": float(metrics["exclusive_other_contamination_seconds"])
                    / hours,
                    "false_cuts": float(metrics["false_cut_count"]) / hours,
                    "missed_replacements": float(metrics["missed_replacement_count"]) / hours,
                },
                "full_metrics": full_metrics,
                "frame_diagnostics": frame_diagnostics[row["source_id"]],
                "mapping_diagnostics": mapping_diagnostics[row["source_id"]],
            }
        )
    corpora_by_source = {source_id: _corpus(session) for source_id, session in sessions.items()}
    frame_diagnostic_views = _aggregate_frame_diagnostics(frame_diagnostics, corpora_by_source)
    mapping_diagnostic_views = _aggregate_mapping_diagnostics(
        mapping_diagnostics, corpora_by_source
    )
    slot_mapping_coverage_passed = all(
        row["mapping_coverage"] == 1.0 and row["slot_instability_count"] == 0
        for row in mapping_diagnostics.values()
    )
    timing_gate_passed = all(
        row["unexpected_reset_count"] == 0 for row in mapping_diagnostics.values()
    )
    evidence = {"frontier": frontier, "per_source_primary": per_source_primary}
    artifact_role = (
        "psem_sortformer_dev_result" if split_role == DEV_ROLE else "psem_sortformer_eval_result"
    )
    payload = {
        "schema_version": 1,
        "artifact_role": artifact_role,
        "arm": arm,
        "seed": seed,
        "split_role": split_role,
        "evaluation_roles": [split_role],
        "eval_open_count": 0 if split_role == DEV_ROLE else 1,
        "passed": slot_mapping_coverage_passed and timing_gate_passed,
        "projection": (
            "one_minus_oracle_anchor_slot_float_posterior"
            if arm == "F0-FROZEN-FLOAT"
            else "sigmoid_replacement_evidence_logit"
        ),
        "frontier": frontier,
        "per_source_primary": per_source_primary,
        "per_source_rows": per_source_rows,
        "mapping_diagnostics": mapping_diagnostics,
        "frame_diagnostics": frame_diagnostics,
        "frame_diagnostic_views": frame_diagnostic_views,
        "mapping_diagnostic_views": mapping_diagnostic_views,
        "slot_mapping_coverage_passed": slot_mapping_coverage_passed,
        "timing_gate_passed": timing_gate_passed,
        "dev_evidence_sha256"
        if split_role == DEV_ROLE
        else "eval_evidence_sha256": canonical_sha256(evidence),
        "prediction_set_sha256": value["payload_sha256"],
        "prediction_set": dict(value),
        "eval_authorization_sha256": (
            eval_authorization.get("payload_sha256") if eval_authorization is not None else None
        ),
    }
    result = bind_payload(payload)
    if not historical_replay:
        register_execution("evaluation-result", result)
    return result


def source_family_frontiers(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = result.get("per_source_rows")
    if not isinstance(rows, list):
        raise EvaluationError("result lacks per-source rows")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["corpus"], row["threshold"], row["confirmation_ms"])].append(row)
    return {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_source_family_frontiers",
        "arm": result["arm"],
        "seed": result.get("seed"),
        "rows": [
            {
                "corpus": key[0],
                "threshold": key[1],
                "confirmation_ms": key[2],
                **_view(chosen),
            }
            for key, chosen in sorted(grouped.items())
        ],
    }
