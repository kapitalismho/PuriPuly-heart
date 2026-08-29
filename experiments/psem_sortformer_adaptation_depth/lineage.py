from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
from experiments.psem_sortformer_adaptation_depth.authority_registry import register_execution
from experiments.psem_sortformer_adaptation_depth.execution import (
    _mapping_slots,
    _source_rows,
    assert_clean_candidate,
    candidate_code_identity,
    load_source_waveform,
    write_jsonl,
)
from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
    NEMO_REVISION,
    load_pinned_sortformer,
)
from experiments.psem_sortformer_adaptation_depth.predictions import prediction_rows
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256
from experiments.psem_sortformer_adaptation_depth.protocol import (
    _eval_registry_marker,
    bind_payload,
    require_bound,
)
from experiments.psem_sortformer_adaptation_depth.receipts import (
    CHECKPOINT_IDENTITY,
    Q8_IDENTITY,
    build_data_split_receipt,
    evaluator_reconstruction_contract,
    recompute_lineage_numeric_evidence,
    validate_trainable_checkpoint_lineage,
)
from experiments.psem_sortformer_adaptation_depth.runtime_audit import LOW_LATENCY_STREAMING
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    normalize_reference_session,
    open_reference_checkout,
)
from experiments.psem_training_strategy_gate.preflight import canonical_sha256 as label_sha256
from experiments.psem_training_strategy_gate.sampling import (
    DEV_ROLE,
    EVAL_ROLE,
    TOPOLOGY_MANIFEST_PATH,
    RuntimeSession,
    load_runtime_sessions,
    role_by_source,
)


class LineageError(RuntimeError):
    pass


def lineage_authorization() -> dict[str, Any]:
    split = build_data_split_receipt()
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_lineage_preflight_authorization",
            "purpose": "required_float_vs_q8_lineage_only",
            "source_ids_by_role": {
                DEV_ROLE: split["source_ids_by_role"][DEV_ROLE],
                EVAL_ROLE: split["source_ids_by_role"][EVAL_ROLE],
            },
            "adapted_checkpoint_access": False,
            "fitting_allowed": False,
            "checkpoint_selection_allowed": False,
            "threshold_selection_allowed": False,
            "eval_open_count": 0,
            "eval_used_for_development": False,
        }
    )


def _lineage_eval_sessions(
    corpus_root: Path,
    reference_root: Path,
    authorization: Mapping[str, Any],
) -> dict[str, RuntimeSession]:
    auth = require_bound(authorization, "psem_sortformer_lineage_preflight_authorization")
    expected_auth = lineage_authorization()
    if authorization != expected_auth or any(
        auth.get(field) is not False
        for field in (
            "adapted_checkpoint_access",
            "fitting_allowed",
            "checkpoint_selection_allowed",
            "threshold_selection_allowed",
            "eval_used_for_development",
        )
    ):
        raise LineageError("lineage EVAL access exceeds its required preflight-only scope")
    source_rows = _source_rows()
    topology_rows = {
        row["source_id"]: row
        for row in (
            json.loads(line)
            for line in TOPOLOGY_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        )
    }
    assignments = role_by_source()
    checkout = open_reference_checkout(reference_root)
    sessions = {}
    for source_id in auth["source_ids_by_role"][EVAL_ROLE]:
        if assignments[source_id] != EVAL_ROLE:
            raise LineageError("lineage source role differs from the frozen split")
        row = source_rows[source_id]
        normalized = normalize_reference_session(row, corpus_root, checkout)
        if topology_rows[source_id]["label_result_sha256"] != label_sha256(
            normalized.labels.to_dict()
        ):
            raise LineageError("lineage labels differ from the frozen topology artifact")
        sessions[source_id] = RuntimeSession(
            source_id=source_id,
            role=EVAL_ROLE,
            audio_ref=str(row["audio_ref"]),
            waveform_sha256=str(row["waveform_sha256"]),
            labels=normalized.labels,
        )
    return sessions


@torch.no_grad()
def build_lineage_receipt(
    *,
    checkpoint_path: Path,
    nemo_checkout: Path,
    dependency_lock: Path,
    corpus_root: Path,
    reference_root: Path,
    output_root: Path,
    device: str,
    authorization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _eval_registry_marker().exists():
        raise LineageError("lineage preflight is sealed after official EVAL opened")
    assert_clean_candidate()
    code_identity = candidate_code_identity()
    if authorization != lineage_authorization():
        raise LineageError("lineage run lacks the exact committed preflight authorization")
    dev_sessions = load_runtime_sessions(corpus_root, reference_root, roles=(DEV_ROLE,))
    eval_sessions = _lineage_eval_sessions(corpus_root, reference_root, authorization)
    sessions = {**dev_sessions, **eval_sessions}
    model, runtime_identity = load_pinned_sortformer(
        checkpoint_path,
        nemo_checkout,
        dependency_lock,
        device,
    )
    model.eval()
    snapshots = {
        session.source_id: session for session in load_sessions(validate_mapping_ledger=False)
    }
    split = build_data_split_receipt()
    role_by_id = {
        source_id: role
        for role, source_ids in split["source_ids_by_role"].items()
        for source_id in source_ids
    }
    expected_sources = sorted(
        split["source_ids_by_role"][DEV_ROLE] + split["source_ids_by_role"][EVAL_ROLE]
    )
    if sorted(sessions) != expected_sources or set(snapshots) != set(expected_sources):
        raise LineageError("lineage sources differ from exact #99 DEV/EVAL identities")
    graph = runtime_identity["model_graph"]
    source_manifest = _source_rows()
    source_receipts = []
    for source_id in expected_sources:
        waveform, duration, tail = load_source_waveform(sessions[source_id], corpus_root)
        waveform = waveform.to(model.sortformer.device)
        frame_count = waveform.shape[1] // 1280
        lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=waveform.device)
        reset = torch.zeros((1, frame_count, 1), dtype=torch.bool, device=waveform.device)
        reset[:, 0, 0] = True
        evidence = model.sortformer_evidence(waveform, lengths, state_reset=reset)
        snapshot = snapshots[source_id]
        episode_ids, slots, mapping_rows = _mapping_slots(evidence.activity_logits[0], snapshot)
        anchor = torch.zeros_like(evidence.probabilities)
        anchor[
            0,
            torch.arange(frame_count, device=anchor.device),
            torch.tensor(slots, device=anchor.device),
        ] = 1
        selected = (evidence.probabilities * anchor).sum(dim=-1).clamp(1e-7, 1 - 1e-7)
        outputs = {
            "anchor_present": torch.logit(selected),
            "replacement_evidence": torch.logit(1 - selected),
        }
        rows = prediction_rows(
            source_id=source_id,
            source_start_sample=0,
            evidence=evidence,
            psem_outputs=outputs,
            anchor_episode_ids=episode_ids,
            oracle_anchor_slots=slots,
            provenance={
                "artifact_context": "trainable_checkpoint_lineage",
                "split_role": role_by_id[source_id],
                "arm": "F0-FROZEN-FLOAT",
                "seed": None,
                "source_waveform_sha256": source_manifest[source_id]["waveform_sha256"],
                "base_checkpoint_sha256": CHECKPOINT_IDENTITY["sha256"],
                "trained_checkpoint_sha256": None,
                "trained_checkpoint_receipt_sha256": None,
                "runtime_identity_sha256": canonical_sha256(runtime_identity),
                "candidate_git_head": code_identity["git_head"],
                "candidate_code_identity_sha256": code_identity["payload_sha256"],
                "experiment_output_root": str(output_root.resolve()),
            },
        )
        descriptor = write_jsonl(
            output_root.resolve() / "lineage_predictions" / f"{source_id}.jsonl",
            rows,
        )
        source_receipts.append(
            {
                "source_id": source_id,
                "split_role": role_by_id[source_id],
                "source_waveform_sha256": source_manifest[source_id]["waveform_sha256"],
                "source_duration_samples": duration,
                "source_tail_samples_excluded": tail,
                "frame_count": frame_count,
                "first_frame_start_sample": 0,
                "last_frame_end_sample": frame_count * 1280,
                "first_evidence_frontier_sample": 16640,
                "last_evidence_frontier_sample": (frame_count - 1) * 1280 + 16640,
                "hidden_tensor_identity": "sortformer.transformer_encoder.output",
                "hidden_dimension": 192,
                "activity_logit_identity": "sortformer.sortformer_modules.single_hidden_to_spks.output_pre_sigmoid",
                "slot_count": 4,
                "slot_alive_policy": "issue_99_all_four_stable_columns_alive",
                "executable_graph_sha256": graph["executable_graph_sha256"],
                "dependency_lock_sha256": runtime_identity["dependency_lock_sha256"],
                "runtime_identity_sha256": canonical_sha256(runtime_identity),
                "candidate_git_head": code_identity["git_head"],
                "candidate_code_identity_sha256": code_identity["payload_sha256"],
                "experiment_output_root": str(output_root.resolve()),
                "prediction_artifact": descriptor,
                "mapping_rows": mapping_rows,
            }
        )
        del waveform, evidence, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    evaluator = evaluator_reconstruction_contract()
    numeric = recompute_lineage_numeric_evidence(source_receipts, evaluator)
    payload = {
        "schema_version": 1,
        "artifact_role": "trainable_checkpoint_lineage",
        "checkpoint": CHECKPOINT_IDENTITY,
        "nemo_revision": NEMO_REVISION,
        "q8_baseline": Q8_IDENTITY,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "candidate_git_head": code_identity["git_head"],
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "candidate_artifact_sha256s": code_identity["artifact_sha256s"],
        "experiment_output_root": str(output_root.resolve()),
        "evaluator_contract_sha256": canonical_sha256(evaluator),
        "dependency_lock_sha256": runtime_identity["dependency_lock_sha256"],
        "executable_graph_sha256": graph["executable_graph_sha256"],
        "streaming": {
            **LOW_LATENCY_STREAMING,
            "sample_rate_hz": 16000,
            "native_frame_samples": 1280,
            "slot_count": 4,
            "algorithmic_evidence_delay_samples": 16640,
            "reset_policy": "declared_source_or_reset_boundary_only",
        },
        "sources": source_receipts,
        "float_vs_q8_posterior_deltas": numeric["float_vs_q8_posterior_deltas"],
        "float_vs_q8_product_deltas": numeric["float_vs_q8_product_deltas"],
        "study_label": numeric["study_label"],
        "direct_q8_fine_tuning_claim": False,
        "lineage_authorization_sha256": authorization["payload_sha256"],
    }
    validated = validate_trainable_checkpoint_lineage(
        payload,
        runtime_identity=runtime_identity,
        evaluator_contract=evaluator,
    )
    register_execution("lineage", validated)
    return validated, runtime_identity
