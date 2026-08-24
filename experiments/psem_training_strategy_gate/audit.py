from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.psem_training_strategy_gate.augmentation import (
    AUGMENTATION_FAMILIES,
    AUGMENTATION_RECIPE_VERSION,
    apply_augmentation,
    augmentation_decision,
    validate_augmentation_decision,
)
from experiments.psem_training_strategy_gate.evaluator import (
    CandidateEvent,
    PredictionScore,
    ReferenceEvent,
    eventize,
    full_threshold_curve,
    shared_score_thresholds,
    sub_resolution_transitions,
)
from experiments.psem_training_strategy_gate.losses import (
    LossWeights,
    collate_targets,
    compute_losses,
)
from experiments.psem_training_strategy_gate.models import (
    ARMS,
    COMMON_HEAD_LR,
    FINETUNED_WAVLM_LR,
    SCRATCH_DILATIONS,
    SCRATCH_ENCODER_LR,
    SCRATCH_EXPANSION,
    SCRATCH_WIDTH,
    WAVLM_MODEL_ID,
    WAVLM_REVISION,
    ScratchCellEncoder,
    WavLMCellEncoder,
    build_model,
    build_optimizer,
    parameter_inventory,
    tensor_sha256,
    wavlm_parameter_allowed,
)
from experiments.psem_training_strategy_gate.preflight import canonical_sha256, sha256_file
from experiments.psem_training_strategy_gate.receipts import (
    check,
    current_binding,
    runtime_receipt,
    write_runtime_receipt,
)
from experiments.psem_training_strategy_gate.runtime_contract import (
    RUNTIME_CHECK_IDS,
    RuntimeEvidenceError,
    runtime_artifact_checks,
)
from experiments.psem_training_strategy_gate.sampling import (
    HARD_NEGATIVE_FAMILIES,
    HARD_NEGATIVE_TOPOLOGY_FAMILY,
    MAXIMUM_EPOCHS,
    POSITIVE_FAMILIES,
    POSITIVE_TOPOLOGY_FAMILY,
    SAMPLING_COUNTS,
    TRAIN_ROLE,
    WINDOWS_PER_EPOCH,
    RuntimeSession,
    iter_rows,
    load_runtime_sessions,
    load_waveform_window,
    materialize_sampling_manifest,
    target_for_row,
)


class AuditContractError(RuntimeError):
    pass


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_sha256(value),
        "size_bytes": path.stat().st_size,
    }


def _receipt_path(output_root: Path, name: str) -> Path:
    return output_root / "preflight" / f"{name}.json"


def _write_named_receipt(
    output_root: Path,
    name: str,
    artifact_role: str,
    checks: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any],
    *,
    validation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if tuple(row["id"] for row in checks) != RUNTIME_CHECK_IDS[name]:
        raise AuditContractError("runtime receipt check inventory differs from its contract")
    receipt = runtime_receipt(
        name,
        artifact_role,
        details=details,
        validation_context=validation_context,
    )
    write_runtime_receipt(
        _receipt_path(output_root, name),
        receipt,
        validation_context=validation_context,
    )
    return receipt


def _loss_weights(summary: Mapping[str, Any]) -> LossWeights:
    values = summary["loss_weights"]
    return LossWeights(
        handoff_positive=float(values["handoff_positive"]),
        state_classes=tuple(float(value) for value in values["state_classes"]),
        relation_classes=tuple(float(value) for value in values["relation_classes"]),
    )


def prepare_runtime_manifests(
    *,
    corpus_root: Path,
    reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    current_binding()
    sessions = load_runtime_sessions(
        corpus_root,
        reference_root,
        roles=(TRAIN_ROLE,),
    )
    sampling_path = output_root / "manifests" / "sampling_manifest.jsonl"
    summary = materialize_sampling_manifest(sessions, sampling_path)
    manifest_artifact = {
        "path": str(sampling_path.resolve()),
        "sha256": summary["manifest_sha256"],
        "size_bytes": sampling_path.stat().st_size,
    }
    expected_role_counts = {role: count * MAXIMUM_EPOCHS for role, count in SAMPLING_COUNTS.items()}
    sampling_checks = [
        check(
            "sampling.manifest_identity",
            summary["manifest_sha256"] == sha256_file(sampling_path)
            and summary["row_count"] == WINDOWS_PER_EPOCH * MAXIMUM_EPOCHS,
            expected={
                "row_count": WINDOWS_PER_EPOCH * MAXIMUM_EPOCHS,
                "sha256_matches": True,
            },
            observed={
                "row_count": summary["row_count"],
                "sha256_matches": summary["manifest_sha256"] == sha256_file(sampling_path),
            },
        ),
        check(
            "sampling.train_only",
            summary["eval_source_count"] == 0
            and summary["source_count"] == len(sessions)
            and all(session.role == TRAIN_ROLE for session in sessions.values()),
            expected={"eval_source_count": 0, "roles": [TRAIN_ROLE]},
            observed={
                "eval_source_count": summary["eval_source_count"],
                "source_count": summary["source_count"],
                "roles": sorted({session.role for session in sessions.values()}),
            },
        ),
        check(
            "sampling.mixture_exact",
            summary["sampling_role_counts"] == expected_role_counts
            and set(summary["topology_family_counts"])
            == {*POSITIVE_FAMILIES, *HARD_NEGATIVE_FAMILIES, "source_time_uniform"}
            and summary["topology_family_mapping"]
            == {
                "handoff_positive": dict(sorted(POSITIVE_TOPOLOGY_FAMILY.items())),
                "topology_hard_negative": dict(sorted(HARD_NEGATIVE_TOPOLOGY_FAMILY.items())),
            },
            expected={
                "role_counts": expected_role_counts,
                "positive_families": list(POSITIVE_FAMILIES),
                "hard_negative_families": list(HARD_NEGATIVE_FAMILIES),
                "topology_family_mapping": summary["topology_family_mapping"],
            },
            observed={
                "role_counts": summary["sampling_role_counts"],
                "families": sorted(summary["topology_family_counts"]),
            },
        ),
        check(
            "sampling.shared_across_arms_and_seeds",
            summary["shared_center_and_augmentation_manifest"] is True
            and summary["arms"] == list(ARMS)
            and summary["seeds"] == [7301, 7302],
            expected={
                "shared": True,
                "arms": list(ARMS),
                "seeds": [7301, 7302],
            },
            observed={
                "shared": summary["shared_center_and_augmentation_manifest"],
                "arms": summary["arms"],
                "seeds": summary["seeds"],
            },
        ),
        check(
            "sampling.loss_weights_complete",
            all(
                math.isfinite(value) and value > 0.0
                for value in (
                    summary["loss_weights"]["handoff_positive"],
                    *summary["loss_weights"]["state_classes"],
                    *summary["loss_weights"]["relation_classes"],
                )
            )
            and all(
                int(count) > 0
                for counts in summary["target_class_counts"].values()
                for count in counts.values()
            ),
            expected="finite positive weights and nonzero support for every enabled class",
            observed={
                "loss_weights": summary["loss_weights"],
                "target_class_counts": summary["target_class_counts"],
            },
        ),
    ]
    summary_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sampling_summary",
        **summary,
        "checks": sampling_checks,
    }
    summary_artifact = _write_json(
        output_root / "manifests" / "sampling_summary.json",
        summary_payload,
    )
    sampling_receipt = _write_named_receipt(
        output_root,
        "sampling_manifest",
        "psem_sampling_manifest",
        sampling_checks,
        {"artifacts": [manifest_artifact, summary_artifact]},
    )
    enabled_counts = Counter({family: 0 for family in AUGMENTATION_FAMILIES})
    row_count = 0
    for row in iter_rows(sampling_path):
        decision = row["augmentation"]
        validate_augmentation_decision(decision)
        if decision != augmentation_decision(str(row["row_id"])):
            raise AuditContractError("sampling row augmentation is not label-independent")
        for family in AUGMENTATION_FAMILIES:
            enabled_counts[family] += int(decision[family]["enabled"])
        row_count += 1
    augmentation_manifest = {
        "schema_version": 1,
        "artifact_role": "psem_augmentation_manifest",
        "recipe_version": AUGMENTATION_RECIPE_VERSION,
        "families": list(AUGMENTATION_FAMILIES),
        "decision_count": row_count,
        "enabled_counts": dict(enabled_counts),
        "decision_source": "sampling_manifest.row_id_only",
        "label_fields_consulted": [],
        "whole_window_consistency": True,
        "sampling_manifest_sha256": summary["manifest_sha256"],
        "synthetic_manifest": None,
        "synthetic_optimizer_batch_fraction": 0.0,
        "natural_training_coverage_satisfied": True,
    }
    augmentation_checks = [
        check(
            "augmentation.recipe_exact",
            augmentation_manifest["recipe_version"] == AUGMENTATION_RECIPE_VERSION
            and augmentation_manifest["families"] == list(AUGMENTATION_FAMILIES),
            expected={
                "recipe_version": AUGMENTATION_RECIPE_VERSION,
                "families": list(AUGMENTATION_FAMILIES),
            },
            observed={
                "recipe_version": augmentation_manifest["recipe_version"],
                "families": augmentation_manifest["families"],
            },
        ),
        check(
            "augmentation.label_independent_whole_window",
            augmentation_manifest["label_fields_consulted"] == []
            and augmentation_manifest["whole_window_consistency"] is True
            and augmentation_manifest["decision_source"] == "sampling_manifest.row_id_only",
            expected={
                "label_fields_consulted": [],
                "whole_window_consistency": True,
                "decision_source": "sampling_manifest.row_id_only",
            },
            observed={
                key: augmentation_manifest[key]
                for key in (
                    "label_fields_consulted",
                    "whole_window_consistency",
                    "decision_source",
                )
            },
        ),
        check(
            "augmentation.family_coverage",
            all(enabled_counts[family] > 0 for family in AUGMENTATION_FAMILIES)
            and row_count == summary["row_count"],
            expected={"all_families_enabled_at_least_once": True, "row_count": row_count},
            observed={"enabled_counts": dict(enabled_counts), "row_count": row_count},
        ),
        check(
            "augmentation.synthetic_policy",
            augmentation_manifest["synthetic_manifest"] is None
            and augmentation_manifest["synthetic_optimizer_batch_fraction"] == 0.0
            and augmentation_manifest["natural_training_coverage_satisfied"] is True,
            expected={
                "synthetic_manifest": None,
                "synthetic_optimizer_batch_fraction": 0.0,
                "natural_training_coverage_satisfied": True,
            },
            observed={
                key: augmentation_manifest[key]
                for key in (
                    "synthetic_manifest",
                    "synthetic_optimizer_batch_fraction",
                    "natural_training_coverage_satisfied",
                )
            },
        ),
        check(
            "augmentation.manifest_binding",
            augmentation_manifest["sampling_manifest_sha256"] == summary["manifest_sha256"],
            expected={"sampling_manifest_bound": True},
            observed={
                "sampling_manifest_bound": (
                    augmentation_manifest["sampling_manifest_sha256"] == summary["manifest_sha256"]
                ),
            },
        ),
    ]
    augmentation_payload = {**augmentation_manifest, "checks": augmentation_checks}
    augmentation_artifact = _write_json(
        output_root / "manifests" / "augmentation_manifest.json",
        augmentation_payload,
    )
    augmentation_receipt = _write_named_receipt(
        output_root,
        "augmentation_manifest",
        "psem_augmentation_manifest",
        augmentation_checks,
        {"artifacts": [augmentation_artifact]},
    )
    return {
        "sampling_receipt": sampling_receipt,
        "augmentation_receipt": augmentation_receipt,
    }


def _module_sha256(model: torch.nn.Module, prefix: str) -> str:
    digest = hashlib.sha256()
    selected = [
        (name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)
    ]
    if not selected:
        raise AuditContractError(f"module prefix has no parameters: {prefix}")
    for name, parameter in selected:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_stats(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    named = list(named_parameters)
    rows = []
    total_squared = 0.0
    none_count = 0
    nonfinite_count = 0
    nonzero_tensor_count = 0
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None:
            none_count += 1
            rows.append(
                {
                    "name": name,
                    "present": False,
                    "finite": None,
                    "norm": None,
                    "sha256": None,
                }
            )
            continue
        finite = bool(torch.isfinite(gradient).all())
        gradient_sha256 = tensor_sha256(gradient)
        if not finite:
            nonfinite_count += 1
            rows.append(
                {
                    "name": name,
                    "present": True,
                    "finite": False,
                    "norm": None,
                    "sha256": gradient_sha256,
                }
            )
            continue
        norm = float(gradient.norm())
        total_squared += norm * norm
        nonzero_tensor_count += int(norm > 0.0)
        rows.append(
            {
                "name": name,
                "present": True,
                "finite": True,
                "norm": norm,
                "sha256": gradient_sha256,
            }
        )
    return {
        "parameter_tensor_count": len(rows),
        "none_count": none_count,
        "nonfinite_count": nonfinite_count,
        "nonzero_tensor_count": nonzero_tensor_count,
        "aggregate_norm": math.sqrt(total_squared),
        "parameters": rows,
    }


def _active_gradient(stats: Mapping[str, Any]) -> bool:
    return (
        stats["parameter_tensor_count"] > 0
        and stats["none_count"] == 0
        and stats["nonfinite_count"] == 0
        and stats["aggregate_norm"] > 0.0
        and stats["nonzero_tensor_count"] == stats["parameter_tensor_count"]
    )


def _parameter_owner(name: str) -> str:
    if name.startswith("encoder.wavlm."):
        return "wavlm_allowed" if wavlm_parameter_allowed(name) else "wavlm_frozen"
    if name.startswith("encoder."):
        return "scratch_encoder"
    if name.startswith("projection."):
        return "projection"
    if name.startswith("head."):
        return "common_head"
    raise AuditContractError(f"parameter owner is outside the fixed model graph: {name}")


def _optimizer_audit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        names = [name_by_parameter.get(id(parameter)) for parameter in group["params"]]
        if any(name is None for name in names):
            raise AuditContractError("optimizer contains a parameter outside the model")
        groups.append(
            {
                "name": group.get("group_name"),
                "learning_rate": group["lr"],
                "weight_decay": group["weight_decay"],
                "parameter_names": names,
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            }
        )
    return {
        "type": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "defaults": {
            "learning_rate": optimizer.defaults["lr"],
            "weight_decay": optimizer.defaults["weight_decay"],
        },
        "groups": groups,
    }


def _tensor_evidence(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _target_batch_evidence(target_batch: Any) -> dict[str, Any]:
    return {
        name: _tensor_evidence(getattr(target_batch, name))
        for name in target_batch.__dataclass_fields__
    }


def comparability_provenance(
    row: Mapping[str, Any],
    session: RuntimeSession,
    raw_waveform: torch.Tensor,
    target: Any,
) -> dict[str, Any]:
    augmented_waveform = apply_augmentation(raw_waveform.detach().clone(), row["augmentation"])
    return {
        "row_id": row["row_id"],
        "center_id": f"{row['source_id']}:{row['boundary_sample']}",
        "source_id": row["source_id"],
        "source_waveform_manifest_sha256": session.waveform_sha256,
        "augmentation_decision_sha256": canonical_sha256(row["augmentation"]),
        "sampling_role": row["sampling_role"],
        "raw_waveform_tensor": _tensor_evidence(raw_waveform),
        "augmented_waveform_tensor": _tensor_evidence(augmented_waveform),
        "target_sha256": canonical_sha256(target.to_dict()),
        "target_batch_tensors": _target_batch_evidence(collate_targets((target,))),
        "observed_frontier_sample": target.observed_frontier_sample,
        "unsnapped_handoff_event_samples": list(target.handoff_event_samples),
    }


def _metric_contract_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    one_prediction = [CandidateEvent("s", 1600, 17600, 0.9)]
    two_references = [
        ReferenceEvent("s", 0, "first"),
        ReferenceEvent("s", 3200, "second"),
    ]
    one_to_two = full_threshold_curve(
        one_prediction,
        two_references,
        scored_source_samples=57_600_000,
        score_thresholds=(0.1, 0.9),
    )
    two_predictions = [
        CandidateEvent("s", 0, 16000, 0.9),
        CandidateEvent("s", 3200, 19200, 0.8),
    ]
    one_reference = [ReferenceEvent("s", 1600, "only")]
    two_to_one = full_threshold_curve(
        two_predictions,
        one_reference,
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((two_predictions,)),
    )
    nearest_candidates = [
        CandidateEvent("s", 1600, 17600, 0.9),
        CandidateEvent("s", 3200, 19200, 0.8),
    ]
    nearest = full_threshold_curve(
        nearest_candidates,
        [ReferenceEvent("s", 0, "earlier"), ReferenceEvent("s", 1600, "later")],
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((nearest_candidates,)),
    )
    collar_results = {}
    for collar_ms in (100, 250, 500):
        radius = collar_ms * 16
        candidates = [
            CandidateEvent("s", 16000, 32000, 0.9),
            CandidateEvent("s", 32000, 48000, 0.1),
        ]
        exact = full_threshold_curve(
            candidates,
            [ReferenceEvent("s", 16000 - radius, "exact")],
            scored_source_samples=57_600_000,
            score_thresholds=(0.1, 0.9),
        )
        outside = full_threshold_curve(
            candidates,
            [ReferenceEvent("s", 16000 - radius - 1, "outside")],
            scored_source_samples=57_600_000,
            score_thresholds=(0.1, 0.9),
        )
        collar_results[str(collar_ms)] = {
            "exact": exact["rows"][-1]["metrics"][str(collar_ms)]["true_positive_count"],
            "outside_by_one_sample": outside["rows"][-1]["metrics"][str(collar_ms)][
                "true_positive_count"
            ],
            "outside_false_events_per_hour": outside["rows"][-1]["metrics"][str(collar_ms)][
                "false_events_per_hour"
            ],
        }
    peaks = eventize(
        (
            PredictionScore("s", 0, 16000, 0.9),
            PredictionScore("s", 1600, 17600, 0.1),
            PredictionScore("s", 3200, 19200, 0.8),
            PredictionScore("s", 4800, 20800, 0.1),
            PredictionScore("s", 6400, 22400, 0.7),
        )
    )
    frontier_candidates = [
        CandidateEvent("s", 0, 16000, 0.1),
        CandidateEvent("s", 6400, 22400, 0.5),
        CandidateEvent("s", 12800, 28800, 0.9),
    ]
    frontier = full_threshold_curve(
        frontier_candidates,
        [ReferenceEvent("s", 35137, "unsnapped")],
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((frontier_candidates,)),
    )
    unsnapped_candidates = [
        CandidateEvent("s", 35200, 51200, 0.9),
        CandidateEvent("s", 40000, 56000, 0.1),
    ]
    unsnapped = full_threshold_curve(
        unsnapped_candidates,
        [ReferenceEvent("s", 35137, "unsnapped")],
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((unsnapped_candidates,)),
    )
    first_output = [
        CandidateEvent("first", 0, 16000, 0.1),
        CandidateEvent("first", 1600, 17600, 0.9),
    ]
    second_output = [
        CandidateEvent("second", 0, 16000, 0.2),
        CandidateEvent("second", 1600, 17600, 0.8),
    ]
    shared_thresholds = shared_score_thresholds((first_output, second_output))
    shared_curves = [
        full_threshold_curve(
            candidates,
            (),
            scored_source_samples=57_600_000,
            score_thresholds=shared_thresholds,
        )
        for candidates in (first_output, second_output)
    ]
    equal_score_candidates = [
        CandidateEvent("s", 0, 16000, 0.9),
        CandidateEvent("s", 3200, 19200, 0.9),
    ]
    equal_score = full_threshold_curve(
        equal_score_candidates,
        (),
        scored_source_samples=57_600_000,
        score_thresholds=(0.1, 0.9),
    )
    diagnostic_candidates = [
        CandidateEvent("s", 0, 16000, 0.9),
        CandidateEvent("s", 6400, 22400, 0.1),
    ]
    diagnostic_curve = full_threshold_curve(
        diagnostic_candidates,
        two_references,
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((diagnostic_candidates,)),
    )
    evaluator_contract_path = (
        Path(__file__).resolve().parent / "data" / "v2" / "evaluator_contract.json"
    )
    evaluator_contract = json.loads(evaluator_contract_path.read_text(encoding="utf-8"))
    checks = [
        check(
            "metric.one_to_one",
            one_to_two["rows"][-1]["metrics"]["100"]["true_positive_count"] == 1
            and two_to_one["rows"][0]["metrics"]["100"]["true_positive_count"] == 1
            and nearest["rows"][0]["matches"]["100"]
            == [
                {
                    "prediction_source_id": "s",
                    "prediction_source_sample": 1600,
                    "reference_source_id": "s",
                    "reference_source_sample": 0,
                    "absolute_distance_samples": 1600,
                },
                {
                    "prediction_source_id": "s",
                    "prediction_source_sample": 3200,
                    "reference_source_id": "s",
                    "reference_source_sample": 1600,
                    "absolute_distance_samples": 1600,
                },
            ],
            expected={
                "one_prediction_two_references": 1,
                "two_predictions_one_reference": 1,
                "nearest_remap_assignments": [[1600, 0], [3200, 1600]],
            },
            observed={
                "one_prediction_two_references": one_to_two["rows"][-1]["metrics"]["100"][
                    "true_positive_count"
                ],
                "two_predictions_one_reference": two_to_one["rows"][0]["metrics"]["100"][
                    "true_positive_count"
                ],
                "nearest_remap_assignments": [
                    [row["prediction_source_sample"], row["reference_source_sample"]]
                    for row in nearest["rows"][0]["matches"]["100"]
                ],
            },
        ),
        check(
            "metric.collar_boundaries",
            all(
                values["exact"] == 1 and values["outside_by_one_sample"] == 0
                for values in collar_results.values()
            ),
            expected={
                str(collar): {
                    "exact": 1,
                    "outside_by_one_sample": 0,
                    "outside_false_events_per_hour": 1.0,
                }
                for collar in (100, 250, 500)
            },
            observed=collar_results,
        ),
        check(
            "metric.duplicate_suppression",
            [row.boundary_sample for row in peaks] == [0, 6400]
            and diagnostic_curve["sub_resolution_transitions"]
            == list(sub_resolution_transitions(two_references))
            and diagnostic_curve["sub_resolution_transitions"][0]["artifact_role"]
            == "sub_resolution_transition",
            expected={
                "retained_samples": [0, 6400],
                "sub_resolution_count": 1,
                "artifact_roles": ["sub_resolution_transition"],
            },
            observed={
                "retained_samples": [row.boundary_sample for row in peaks],
                "sub_resolution_count": len(diagnostic_curve["sub_resolution_transitions"]),
                "artifact_roles": [
                    row["artifact_role"] for row in diagnostic_curve["sub_resolution_transitions"]
                ],
            },
        ),
        check(
            "metric.source_hour_denominator",
            all(
                values["outside_false_events_per_hour"] == 1.0 for values in collar_results.values()
            ),
            expected={str(collar): 1.0 for collar in (100, 250, 500)},
            observed={
                collar: values["outside_false_events_per_hour"]
                for collar, values in collar_results.items()
            },
        ),
        check(
            "metric.unsnapped_references",
            unsnapped["rows"][-1]["metrics"]["100"]["true_positive_count"] == 1
            and one_reference[0].source_sample == 1600
            and ReferenceEvent("s", 35137, "unsnapped").source_sample == 35137,
            expected={"unsnapped_source_sample": 35137, "matched": 1},
            observed={
                "unsnapped_source_sample": ReferenceEvent("s", 35137, "unsnapped").source_sample,
                "matched": unsnapped["rows"][-1]["metrics"]["100"]["true_positive_count"],
            },
        ),
        check(
            "metric.full_unique_score_range",
            frontier["score_thresholds"] == [0.1, 0.5, 0.9]
            and len(frontier["rows"]) == 3
            and frontier["false_events_per_hour_ceiling"] is None
            and equal_score["rows"][-1]["prediction_count"] == 2,
            expected={
                "thresholds": [0.1, 0.5, 0.9],
                "ceiling": None,
                "equal_score_prediction_count": 2,
            },
            observed={
                "thresholds": frontier["score_thresholds"],
                "ceiling": frontier["false_events_per_hour_ceiling"],
                "equal_score_prediction_count": equal_score["rows"][-1]["prediction_count"],
            },
        ),
        check(
            "metric.shared_evaluator_contract",
            sha256_file(evaluator_contract_path)
            == "ebaa6a81650c23dd66ca0e07e2522418dd44f81dc24892f1095ec8dae8f76d11"
            and evaluator_contract["threshold_policy"][
                "same_threshold_vector_required_for_every_output"
            ]
            is True
            and evaluator_contract["threshold_policy"]["per_corpus_thresholds_allowed"] is False
            and shared_thresholds == (0.1, 0.2, 0.8, 0.9)
            and all(
                curve["score_thresholds"] == list(shared_thresholds) for curve in shared_curves
            ),
            expected={
                "sha256": "ebaa6a81650c23dd66ca0e07e2522418dd44f81dc24892f1095ec8dae8f76d11",
                "same_threshold_vector": True,
                "per_corpus_thresholds_allowed": False,
                "shared_thresholds": [0.1, 0.2, 0.8, 0.9],
            },
            observed={
                "sha256": sha256_file(evaluator_contract_path),
                "same_threshold_vector": evaluator_contract["threshold_policy"][
                    "same_threshold_vector_required_for_every_output"
                ],
                "per_corpus_thresholds_allowed": evaluator_contract["threshold_policy"][
                    "per_corpus_thresholds_allowed"
                ],
                "shared_thresholds": list(shared_thresholds),
                "output_thresholds": [curve["score_thresholds"] for curve in shared_curves],
            },
        ),
    ]
    return checks, {
        "eventizer": "local_maxima_then_score_ordered_200ms_duplicate_suppression",
        "matching": "maximum_cardinality_one_to_one_nearest_then_earlier_source",
        "matching_collars_ms": [100, 250, 500],
        "thresholds": "complete_shared_union_of_unique_scores",
        "false_events_per_hour_ceiling": None,
        "synthetic_frontier": frontier,
        "eventizer_retained_samples": [row.boundary_sample for row in peaks],
        "unsnapped_curve": unsnapped,
        "collar_canaries": collar_results,
        "nearest_remap": nearest,
        "shared_thresholds": list(shared_thresholds),
        "shared_output_curves": shared_curves,
        "equal_score_curve": equal_score,
        "sub_resolution_transitions": diagnostic_curve["sub_resolution_transitions"],
    }


def _canary_row(
    sampling_path: Path,
    sessions: Mapping[str, RuntimeSession],
) -> Mapping[str, Any]:
    for row in iter_rows(sampling_path):
        if row.get("topology_family") != "overlap_takeover":
            continue
        session = sessions[str(row["source_id"])]
        target = target_for_row(row, session)
        if (
            target.handoff_target == 1
            and target.handoff_mask
            and any(pair.target == 0 for pair in target.relation_pairs)
            and any(pair.target == 1 for pair in target.relation_pairs)
        ):
            validate_augmentation_decision(row["augmentation"])
            return row
    raise AuditContractError("sampling manifest has no complete real canary window")


def run_runtime_audits(
    *,
    cache_root: Path,
    corpus_root: Path,
    reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    current_binding()
    sampling_path = output_root / "manifests" / "sampling_manifest.jsonl"
    summary_path = output_root / "manifests" / "sampling_summary.json"
    augmentation_path = output_root / "manifests" / "augmentation_manifest.json"
    if not sampling_path.is_file() or not summary_path.is_file() or not augmentation_path.is_file():
        raise AuditContractError("prepared sampling and augmentation manifests are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["manifest_sha256"] != sha256_file(sampling_path):
        raise AuditContractError("prepared sampling manifest is stale")
    try:
        runtime_artifact_checks("sampling_manifest", summary)
    except RuntimeEvidenceError as error:
        raise AuditContractError("prepared sampling summary is not authoritative") from error
    sessions = load_runtime_sessions(
        corpus_root,
        reference_root,
        roles=(TRAIN_ROLE,),
    )
    augmentation_payload = json.loads(augmentation_path.read_text(encoding="utf-8"))
    augmentation_manifest = {
        key: value for key, value in augmentation_payload.items() if key != "checks"
    }
    augmentation_enabled = Counter({family: 0 for family in AUGMENTATION_FAMILIES})
    sampling_roles: Counter[str] = Counter()
    topology_families: Counter[str] = Counter()
    sampled_sources: set[str] = set()
    validated_row_count = 0
    for prepared_row in iter_rows(sampling_path):
        source_id = prepared_row.get("source_id")
        if source_id not in sessions:
            raise AuditContractError("prepared sampling manifest contains a non-TRAIN source")
        target_for_row(prepared_row, sessions[str(source_id)])
        sampling_roles[str(prepared_row.get("sampling_role"))] += 1
        topology_families[str(prepared_row.get("topology_family"))] += 1
        sampled_sources.add(str(source_id))
        decision = prepared_row.get("augmentation")
        if not isinstance(decision, Mapping):
            raise AuditContractError("prepared augmentation decision is missing")
        validate_augmentation_decision(decision)
        if decision != augmentation_decision(str(prepared_row.get("row_id", ""))):
            raise AuditContractError("prepared augmentation decision is not row-bound")
        for family in AUGMENTATION_FAMILIES:
            augmentation_enabled[family] += int(decision[family]["enabled"])
        validated_row_count += 1
    expected_augmentation = {
        "schema_version": 1,
        "artifact_role": "psem_augmentation_manifest",
        "recipe_version": AUGMENTATION_RECIPE_VERSION,
        "families": list(AUGMENTATION_FAMILIES),
        "decision_count": validated_row_count,
        "enabled_counts": dict(augmentation_enabled),
        "decision_source": "sampling_manifest.row_id_only",
        "label_fields_consulted": [],
        "whole_window_consistency": True,
        "sampling_manifest_sha256": summary["manifest_sha256"],
        "synthetic_manifest": None,
        "synthetic_optimizer_batch_fraction": 0.0,
        "natural_training_coverage_satisfied": True,
    }
    augmentation_checks = augmentation_payload.get("checks")
    if (
        validated_row_count != summary.get("row_count")
        or dict(sorted(sampling_roles.items())) != summary.get("sampling_role_counts")
        or dict(sorted(topology_families.items())) != summary.get("topology_family_counts")
        or len(sampled_sources) != summary.get("source_count")
        or augmentation_manifest != expected_augmentation
        or not isinstance(augmentation_checks, list)
        or tuple(check.get("id") for check in augmentation_checks if isinstance(check, Mapping))
        != RUNTIME_CHECK_IDS["augmentation_manifest"]
        or not all(check.get("passed") is True for check in augmentation_checks)
    ):
        raise AuditContractError("prepared augmentation manifest differs from recomputed decisions")
    try:
        runtime_artifact_checks("augmentation_manifest", augmentation_payload)
    except RuntimeEvidenceError as error:
        raise AuditContractError("prepared augmentation manifest is not authoritative") from error
    row = _canary_row(sampling_path, sessions)
    session = sessions[str(row["source_id"])]
    raw_waveform = load_waveform_window(row, session, corpus_root)
    target = target_for_row(row, session)
    expected_comparability = comparability_provenance(row, session, raw_waveform, target)
    weights = _loss_weights(summary)
    target_sha256 = canonical_sha256(target.to_dict())
    models_code_sha256 = sha256_file(Path(__file__).with_name("models.py"))
    losses_code_sha256 = sha256_file(Path(__file__).with_name("losses.py"))
    loss_contract = {
        "coefficients": {"handoff": 1.0, "state": 0.5, "relation": 0.5},
        "class_weights": {
            "handoff_positive": weights.handoff_positive,
            "state_classes": list(weights.state_classes),
            "relation_classes": list(weights.relation_classes),
        },
        "implementation_sha256": losses_code_sha256,
    }
    graph_rows: dict[str, Any] = {}
    inventories: dict[str, Any] = {}
    gradient_rows: dict[str, Any] = {}
    update_rows: dict[str, Any] = {}
    comparability_rows: dict[str, Any] = {}
    common_head_sha256: dict[str, str] = {}
    projection_sha256: dict[str, str] = {}
    for arm in ARMS:
        torch.manual_seed(9001)
        model = build_model(arm, cache_root=cache_root, seed=7301)
        model.train()
        optimizer = build_optimizer(model)
        optimizer_evidence = _optimizer_audit(model, optimizer)
        inventory = parameter_inventory(model)
        inventory["optimizer"] = optimizer_evidence
        inventories[arm] = inventory
        common_head_sha256[arm] = _module_sha256(model, "head.")
        projection_sha256[arm] = _module_sha256(model, "projection.")
        parameters = dict(model.named_parameters())
        before = {name: tensor_sha256(parameter) for name, parameter in parameters.items()}
        initial_wavlm_sha256 = (
            _module_sha256(model, "encoder.wavlm.") if arm != "SCRATCH-PSEM" else None
        )
        arm_raw_waveform = raw_waveform.detach().clone()
        arm_augmented_waveform = apply_augmentation(arm_raw_waveform, row["augmentation"])
        target_batch = collate_targets((target,))
        torch.manual_seed(9001)
        outputs = model(arm_augmented_waveform.unsqueeze(0))
        losses = compute_losses(model, outputs, target_batch, weights)
        total = losses["total"]
        if not isinstance(total, torch.Tensor):
            raise AuditContractError("canary total loss is not a tensor")
        total.backward()
        named = list(model.named_parameters())
        wavlm_all = [row for row in named if row[0].startswith("encoder.wavlm.")]
        wavlm_allowed = [row for row in wavlm_all if wavlm_parameter_allowed(row[0])]
        wavlm_frozen = [row for row in wavlm_all if not wavlm_parameter_allowed(row[0])]
        gradient_rows[arm] = {
            "losses": {
                key: float(value.detach()) if isinstance(value, torch.Tensor) else value
                for key, value in losses.items()
            },
            "projection": _gradient_stats(row for row in named if row[0].startswith("projection.")),
            "temporal": _gradient_stats(
                row for row in named if row[0].startswith("head.temporal.")
            ),
            "handoff_head": _gradient_stats(
                row for row in named if row[0].startswith("head.handoff_head.")
            ),
            "state_head": _gradient_stats(
                row for row in named if row[0].startswith("head.state_head.")
            ),
            "relation_head": _gradient_stats(
                row for row in named if row[0].startswith("head.relation_head.")
            ),
            "wavlm_all": _gradient_stats(wavlm_all),
            "wavlm_allowed": _gradient_stats(wavlm_allowed),
            "wavlm_frozen": _gradient_stats(wavlm_frozen),
            "scratch_encoder": _gradient_stats(
                row
                for row in named
                if row[0].startswith("encoder.") and not row[0].startswith("encoder.wavlm.")
            ),
            "finetuned_blocks": {
                str(index): _gradient_stats(
                    row
                    for row in named
                    if row[0].startswith(f"encoder.wavlm.encoder.layers.{index}.")
                )
                for index in range(8, 12)
            },
            "finetuned_final_normalization": _gradient_stats(
                row for row in named if row[0].startswith("encoder.wavlm.encoder.layer_norm.")
            ),
        }
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            5.0,
        )
        optimizer.step()
        after = {name: tensor_sha256(parameter) for name, parameter in parameters.items()}
        optimizer_group_by_name = {
            name: group["name"]
            for group in optimizer_evidence["groups"]
            for name in group["parameter_names"]
        }
        update_rows[arm] = {
            "parameters": [
                {
                    "name": name,
                    "owner": _parameter_owner(name),
                    "trainable": parameter.requires_grad,
                    "optimizer_group": optimizer_group_by_name.get(name),
                    "before_sha256": before[name],
                    "after_sha256": after[name],
                    "changed": before[name] != after[name],
                }
                for name, parameter in parameters.items()
            ]
        }
        temporal = model.head.temporal
        common_head_contract = {
            "temporal": {
                "type": type(temporal).__name__,
                "input_size": temporal.input_size,
                "hidden_size": temporal.hidden_size,
                "layers": temporal.num_layers,
                "bidirectional": temporal.bidirectional,
                "dropout": temporal.dropout,
                "batch_first": temporal.batch_first,
            },
            "handoff_head_linear_shapes": [
                [module.in_features, module.out_features]
                for module in model.head.handoff_head
                if isinstance(module, torch.nn.Linear)
            ],
            "state_head_shape": [
                model.head.state_head.in_features,
                model.head.state_head.out_features,
            ],
            "relation_head_linear_shapes": [
                [module.in_features, module.out_features]
                for module in model.head.relation_head
                if isinstance(module, torch.nn.Linear)
            ],
            "implementation_sha256": models_code_sha256,
        }
        if isinstance(model.encoder, WavLMCellEncoder):
            config = model.encoder.wavlm.config
            source_path = model.encoder.model_root
            encoder_contract = {
                "type": type(model.encoder).__name__,
                "model_id": source_path.parent.name,
                "revision": source_path.name,
                "local_model_root": str(source_path),
                "config": {
                    "model_type": config.model_type,
                    "hidden_size": config.hidden_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "conv_kernel": list(config.conv_kernel),
                    "conv_stride": list(config.conv_stride),
                    "do_stable_layer_norm": config.do_stable_layer_norm,
                    "output_frames": int(
                        model.encoder.wavlm._get_feat_extract_output_lengths(
                            torch.tensor([arm_augmented_waveform.numel()])
                        )[0]
                    ),
                },
                "trainable_parameter_names": [
                    name
                    for name, parameter in model.named_parameters()
                    if name.startswith("encoder.wavlm.") and parameter.requires_grad
                ],
                "initial_parameter_sha256": initial_wavlm_sha256,
            }
        elif isinstance(model.encoder, ScratchCellEncoder):
            frontend = model.encoder.frontend
            encoder_contract = {
                "type": type(model.encoder).__name__,
                "frontend": {
                    "sample_rate_hz": frontend.sample_rate,
                    "n_fft": frontend.n_fft,
                    "win_length": frontend.win_length,
                    "hop_length": frontend.hop_length,
                    "center": frontend.spectrogram.center,
                    "power": frontend.power,
                    "mel_bins": frontend.n_mels,
                    "mel_norm": frontend.mel_scale.norm,
                    "mel_scale": frontend.mel_scale.mel_scale,
                },
                "stem": {
                    "input_channels": model.encoder.stem.in_channels,
                    "output_channels": model.encoder.stem.out_channels,
                    "kernel": model.encoder.stem.kernel_size[0],
                },
                "blocks": [
                    {
                        "width": block.normalization.num_channels,
                        "expansion": block.depthwise.in_channels
                        // block.normalization.num_channels,
                        "kernel": block.depthwise.kernel_size[0],
                        "dilation": block.depthwise.dilation[0],
                        "groups": block.depthwise.groups,
                    }
                    for block in model.encoder.blocks
                ],
                "final_normalization_channels": model.encoder.final_normalization.num_channels,
                "pretrained_artifacts": [],
                "implementation_sha256": models_code_sha256,
            }
        else:
            raise AuditContractError("official model uses an unknown encoder implementation")
        graph_rows[arm] = {
            "input": {
                "kind": "raw_waveform",
                "sample_rate_hz": session.labels.sample_rate_hz,
                "samples": int(arm_augmented_waveform.numel()),
                "upstream_transform": AUGMENTATION_RECIPE_VERSION,
                "cached_feature_inputs": [],
            },
            "encoder": encoder_contract,
            "projection": {
                "input_dimension": model.projection[0].in_features,
                "output_dimension": model.projection[0].out_features,
                "normalization_shape": list(model.projection[2].normalized_shape),
            },
            "cell_output_shape": list(outputs["cells"].shape),
            "handoff_output_shape": list(outputs["handoff_logits"].shape),
            "state_output_shape": list(outputs["state_logits"].shape),
            "common_head": common_head_contract,
            "losses": loss_contract,
            "optimizer": optimizer_evidence,
        }
        comparability_rows[arm] = {
            "row_id": row["row_id"],
            "center_id": f"{row['source_id']}:{row['boundary_sample']}",
            "source_id": row["source_id"],
            "source_waveform_manifest_sha256": row["source_waveform_sha256"],
            "raw_waveform_tensor": _tensor_evidence(arm_raw_waveform),
            "augmented_waveform_tensor": _tensor_evidence(arm_augmented_waveform),
            "augmentation_decision_sha256": canonical_sha256(row["augmentation"]),
            "target_sha256": target_sha256,
            "target_batch_tensors": _target_batch_evidence(target_batch),
            "sampling_role": row["sampling_role"],
            "observed_frontier_sample": target.observed_frontier_sample,
            "unsnapped_handoff_event_samples": list(target.handoff_event_samples),
            "common_head_sha256": common_head_sha256[arm],
            "common_head_contract": graph_rows[arm]["common_head"],
            "loss_contract": graph_rows[arm]["losses"],
            "initial_wavlm_sha256": initial_wavlm_sha256,
            "encoder_strategy": {
                "arm": arm,
                "encoder_type": encoder_contract["type"],
                "trainable_encoder_parameter_names": [
                    name
                    for name, parameter in model.named_parameters()
                    if name.startswith("encoder.") and parameter.requires_grad
                ],
            },
        }
        del outputs, losses, optimizer, model, target_batch
        gc.collect()
    graph_payload = {
        "schema_version": 1,
        "artifact_role": "psem_model_graphs",
        "canary_source_id": row["source_id"],
        "canary_boundary_sample": row["boundary_sample"],
        "arms": graph_rows,
    }
    parameter_payload = {
        "schema_version": 1,
        "artifact_role": "psem_parameter_inventory",
        "arms": inventories,
    }
    gradient_payload = {
        "schema_version": 1,
        "artifact_role": "psem_gradient_canary",
        "real_batch": {
            "row_id": row["row_id"],
            "source_id": row["source_id"],
            "boundary_sample": row["boundary_sample"],
            "raw_waveform_tensor": comparability_rows[ARMS[0]]["raw_waveform_tensor"],
            "augmented_waveform_tensor": comparability_rows[ARMS[0]]["augmented_waveform_tensor"],
            "target_sha256": target_sha256,
            "target_batch_tensors": comparability_rows[ARMS[0]]["target_batch_tensors"],
        },
        "arms": gradient_rows,
    }
    update_payload = {
        "schema_version": 1,
        "artifact_role": "psem_weight_update_canary",
        "arms": update_rows,
    }
    comparability_payload = {
        "schema_version": 1,
        "artifact_role": "psem_arm_comparability",
        "row_id": row["row_id"],
        "source_id": row["source_id"],
        "boundary_sample": row["boundary_sample"],
        "arms": comparability_rows,
    }
    graph_checks = [
        check(
            "model_graph.raw_waveform_paths_exact",
            all(
                values["input"]
                == {
                    "kind": "raw_waveform",
                    "sample_rate_hz": 16000,
                    "samples": 48000,
                    "upstream_transform": AUGMENTATION_RECIPE_VERSION,
                    "cached_feature_inputs": [],
                }
                for values in graph_rows.values()
            ),
            expected={
                "kind": "raw_waveform",
                "sample_rate_hz": 16000,
                "samples": 48000,
                "upstream_transform": AUGMENTATION_RECIPE_VERSION,
                "cached_feature_inputs": [],
            },
            observed={arm: values["input"] for arm, values in graph_rows.items()},
        ),
        check(
            "model_graph.wavlm_identity_shared",
            graph_rows["FROZEN-WAVLM"]["encoder"]["model_id"] == WAVLM_MODEL_ID
            and graph_rows["FINETUNE-WAVLM"]["encoder"]["model_id"] == WAVLM_MODEL_ID
            and graph_rows["FROZEN-WAVLM"]["encoder"]["revision"] == WAVLM_REVISION
            and graph_rows["FINETUNE-WAVLM"]["encoder"]["revision"] == WAVLM_REVISION
            and graph_rows["FROZEN-WAVLM"]["encoder"]["config"]
            == graph_rows["FINETUNE-WAVLM"]["encoder"]["config"]
            and graph_rows["FROZEN-WAVLM"]["encoder"]["initial_parameter_sha256"]
            == graph_rows["FINETUNE-WAVLM"]["encoder"]["initial_parameter_sha256"]
            and projection_sha256["FROZEN-WAVLM"] == projection_sha256["FINETUNE-WAVLM"],
            expected={
                "model_id": WAVLM_MODEL_ID,
                "revision": WAVLM_REVISION,
                "config": {
                    "model_type": "wavlm",
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
                    "do_stable_layer_norm": False,
                    "output_frames": 149,
                },
                "initial_wavlm_parameters_shared": True,
                "pretrained_projection_initialization_shared": True,
            },
            observed={
                "frozen": graph_rows["FROZEN-WAVLM"]["encoder"],
                "finetune": graph_rows["FINETUNE-WAVLM"]["encoder"],
                "initial_wavlm_parameters_shared": (
                    graph_rows["FROZEN-WAVLM"]["encoder"]["initial_parameter_sha256"]
                    == graph_rows["FINETUNE-WAVLM"]["encoder"]["initial_parameter_sha256"]
                ),
                "pretrained_projection_initialization_shared": (
                    projection_sha256["FROZEN-WAVLM"] == projection_sha256["FINETUNE-WAVLM"]
                ),
            },
        ),
        check(
            "model_graph.finetune_whitelist_exact",
            all(
                row["trainable"] == wavlm_parameter_allowed(row["name"])
                for row in inventories["FINETUNE-WAVLM"]["parameters"]
                if row["name"].startswith("encoder.wavlm.")
            )
            and graph_rows["FROZEN-WAVLM"]["encoder"]["trainable_parameter_names"] == []
            and graph_rows["FINETUNE-WAVLM"]["encoder"]["trainable_parameter_names"]
            == [
                row["name"]
                for row in inventories["FINETUNE-WAVLM"]["parameters"]
                if row["trainable"] and row["name"].startswith("encoder.wavlm.")
            ],
            expected="only WavLM blocks 8-11 and final encoder normalization trainable",
            observed={
                "trainable_names": [
                    row["name"]
                    for row in inventories["FINETUNE-WAVLM"]["parameters"]
                    if row["trainable"] and row["name"].startswith("encoder.wavlm.")
                ],
                "graph_trainable_names": graph_rows["FINETUNE-WAVLM"]["encoder"][
                    "trainable_parameter_names"
                ],
            },
        ),
        check(
            "model_graph.scratch_architecture_exact",
            graph_rows["SCRATCH-PSEM"]["encoder"]["type"] == "ScratchCellEncoder"
            and graph_rows["SCRATCH-PSEM"]["encoder"]["frontend"]
            == {
                "sample_rate_hz": 16000,
                "n_fft": 400,
                "win_length": 400,
                "hop_length": 160,
                "center": False,
                "power": 2.0,
                "mel_bins": 64,
                "mel_norm": "slaney",
                "mel_scale": "slaney",
            }
            and graph_rows["SCRATCH-PSEM"]["encoder"]["stem"]
            == {"input_channels": 64, "output_channels": 320, "kernel": 5}
            and [block["dilation"] for block in graph_rows["SCRATCH-PSEM"]["encoder"]["blocks"]]
            == list(SCRATCH_DILATIONS)
            and all(
                block["width"] == SCRATCH_WIDTH
                and block["expansion"] == SCRATCH_EXPANSION
                and block["kernel"] == 5
                and block["groups"] == SCRATCH_WIDTH * SCRATCH_EXPANSION
                for block in graph_rows["SCRATCH-PSEM"]["encoder"]["blocks"]
            )
            and graph_rows["SCRATCH-PSEM"]["encoder"]["final_normalization_channels"]
            == SCRATCH_WIDTH
            and graph_rows["SCRATCH-PSEM"]["encoder"]["pretrained_artifacts"] == [],
            expected={
                "frontend": "16 kHz, 400-sample FFT/window, 160-sample hop, 64 Slaney mels",
                "width": 320,
                "expansion": 2,
                "kernel": 5,
                "dilations": [1, 2, 4, 8, 16, 1, 2, 4],
                "pretrained_artifacts": [],
            },
            observed=graph_rows["SCRATCH-PSEM"]["encoder"],
        ),
        check(
            "model_graph.common_output_and_head_exact",
            all(values["cell_output_shape"] == [1, 30, 256] for values in graph_rows.values())
            and all(values["handoff_output_shape"] == [1] for values in graph_rows.values())
            and all(values["state_output_shape"] == [1, 30, 3] for values in graph_rows.values())
            and len(set(common_head_sha256.values())) == 1
            and len({canonical_sha256(values["common_head"]) for values in graph_rows.values()})
            == 1
            and len({canonical_sha256(values["losses"]) for values in graph_rows.values()}) == 1,
            expected={
                "cell_output_shape": [1, 30, 256],
                "handoff_output_shape": [1],
                "state_output_shape": [1, 30, 3],
                "common_head_initialization_shared": True,
                "common_head_graph_shared": True,
                "losses_shared": True,
            },
            observed={
                "cell_output_shapes": {
                    arm: values["cell_output_shape"] for arm, values in graph_rows.items()
                },
                "handoff_output_shapes": {
                    arm: values["handoff_output_shape"] for arm, values in graph_rows.items()
                },
                "state_output_shapes": {
                    arm: values["state_output_shape"] for arm, values in graph_rows.items()
                },
                "common_head_sha256": common_head_sha256,
                "common_heads": {arm: values["common_head"] for arm, values in graph_rows.items()},
                "losses": {arm: values["losses"] for arm, values in graph_rows.items()},
            },
        ),
    ]
    parameter_checks = [
        check(
            "parameters.inventory_complete",
            all(
                sum(row["numel"] for row in values["parameters"]) == values["total_parameters"]
                and len({row["name"] for row in values["parameters"]}) == len(values["parameters"])
                for values in inventories.values()
            ),
            expected="every named parameter appears once and sums to the model total",
            observed={
                arm: {
                    "parameter_rows": len(values["parameters"]),
                    "total_parameters": values["total_parameters"],
                }
                for arm, values in inventories.items()
            },
        ),
        check(
            "parameters.frozen_wavlm_zero_trainable",
            inventories["FROZEN-WAVLM"]["trainable_wavlm_parameters"] == 0,
            expected=0,
            observed=inventories["FROZEN-WAVLM"]["trainable_wavlm_parameters"],
        ),
        check(
            "parameters.finetune_wavlm_nonzero_and_whitelisted",
            inventories["FINETUNE-WAVLM"]["trainable_wavlm_parameters"] > 0
            and all(
                wavlm_parameter_allowed(row["name"])
                for row in inventories["FINETUNE-WAVLM"]["parameters"]
                if row["trainable"] and row["name"].startswith("encoder.wavlm.")
            ),
            expected="nonzero WavLM parameters limited to blocks 8-11 and final normalization",
            observed={
                "trainable_wavlm_parameters": inventories["FINETUNE-WAVLM"][
                    "trainable_wavlm_parameters"
                ],
                "outside_whitelist": [
                    row["name"]
                    for row in inventories["FINETUNE-WAVLM"]["parameters"]
                    if row["trainable"]
                    and row["name"].startswith("encoder.wavlm.")
                    and not wavlm_parameter_allowed(row["name"])
                ],
            },
        ),
        check(
            "parameters.scratch_no_pretrained_and_size",
            5_000_000 <= inventories["SCRATCH-PSEM"]["total_parameters"] <= 10_000_000
            and not any(
                "wavlm" in row["name"].lower() for row in inventories["SCRATCH-PSEM"]["parameters"]
            ),
            expected={"minimum": 5_000_000, "maximum": 10_000_000, "pretrained": False},
            observed={
                "total_parameters": inventories["SCRATCH-PSEM"]["total_parameters"],
                "pretrained_parameter_names": [
                    row["name"]
                    for row in inventories["SCRATCH-PSEM"]["parameters"]
                    if "wavlm" in row["name"].lower()
                ],
            },
        ),
        check(
            "parameters.optimizer_coverage_and_learning_rates",
            all(
                values["optimizer"]["type"] == "torch.optim.adamw.AdamW"
                and values["optimizer"]["defaults"]["weight_decay"] == 1e-4
                and all(group["weight_decay"] == 1e-4 for group in values["optimizer"]["groups"])
                and sorted(
                    name
                    for group in values["optimizer"]["groups"]
                    for name in group["parameter_names"]
                )
                == sorted(row["name"] for row in values["parameters"] if row["trainable"])
                and len(
                    [
                        name
                        for group in values["optimizer"]["groups"]
                        for name in group["parameter_names"]
                    ]
                )
                == len(
                    {
                        name
                        for group in values["optimizer"]["groups"]
                        for name in group["parameter_names"]
                    }
                )
                for values in inventories.values()
            )
            and {
                group["name"]: group["learning_rate"]
                for group in inventories["FROZEN-WAVLM"]["optimizer"]["groups"]
            }
            == {"common_head_and_projection": COMMON_HEAD_LR}
            and {
                group["name"]: group["learning_rate"]
                for group in inventories["FINETUNE-WAVLM"]["optimizer"]["groups"]
            }
            == {
                "common_head_and_projection": COMMON_HEAD_LR,
                "finetuned_wavlm": FINETUNED_WAVLM_LR,
            }
            and {
                group["name"]: group["learning_rate"]
                for group in inventories["SCRATCH-PSEM"]["optimizer"]["groups"]
            }
            == {
                "common_head_and_projection": COMMON_HEAD_LR,
                "scratch_encoder": SCRATCH_ENCODER_LR,
            },
            expected={
                "optimizer_type": "torch.optim.adamw.AdamW",
                "weight_decay": 1e-4,
                "frozen_groups": {"common_head_and_projection": 1e-3},
                "common_head_and_projection": 1e-3,
                "finetuned_wavlm": 1e-5,
                "scratch_encoder": 3e-4,
                "coverage": "exactly once",
            },
            observed={arm: values["optimizer"] for arm, values in inventories.items()},
        ),
    ]
    frozen_gradient = gradient_rows["FROZEN-WAVLM"]
    fine_gradient = gradient_rows["FINETUNE-WAVLM"]
    scratch_gradient = gradient_rows["SCRATCH-PSEM"]
    common_gradient_groups = (
        "projection",
        "temporal",
        "handoff_head",
        "state_head",
        "relation_head",
    )
    gradient_checks = [
        check(
            "gradient.frozen_wavlm",
            frozen_gradient["wavlm_all"]["none_count"]
            == frozen_gradient["wavlm_all"]["parameter_tensor_count"]
            and frozen_gradient["wavlm_all"]["parameter_tensor_count"] > 0
            and frozen_gradient["wavlm_all"]["nonfinite_count"] == 0
            and frozen_gradient["wavlm_all"]["nonzero_tensor_count"] == 0
            and frozen_gradient["wavlm_all"]["aggregate_norm"] == 0.0,
            expected="every frozen WavLM gradient is None",
            observed=frozen_gradient["wavlm_all"],
        ),
        check(
            "gradient.finetune_wavlm",
            all(_active_gradient(value) for value in fine_gradient["finetuned_blocks"].values())
            and _active_gradient(fine_gradient["finetuned_final_normalization"])
            and fine_gradient["wavlm_frozen"]["none_count"]
            == fine_gradient["wavlm_frozen"]["parameter_tensor_count"]
            and fine_gradient["wavlm_frozen"]["parameter_tensor_count"] > 0
            and fine_gradient["wavlm_frozen"]["nonfinite_count"] == 0
            and fine_gradient["wavlm_frozen"]["nonzero_tensor_count"] == 0,
            expected="blocks 8-11 and final normalization active; all other WavLM gradients None",
            observed={
                "blocks": fine_gradient["finetuned_blocks"],
                "final_normalization": fine_gradient["finetuned_final_normalization"],
                "frozen": fine_gradient["wavlm_frozen"],
            },
        ),
        check(
            "gradient.scratch_psem",
            _active_gradient(scratch_gradient["scratch_encoder"]),
            expected="finite nonzero scratch encoder gradient",
            observed=scratch_gradient["scratch_encoder"],
        ),
        check(
            "gradient.common_losses",
            all(
                all(_active_gradient(values[group]) for group in common_gradient_groups)
                and all(
                    values["losses"][count] > 0
                    for count in (
                        "handoff_valid_count",
                        "state_valid_count",
                        "relation_valid_count",
                    )
                )
                and all(
                    math.isfinite(values["losses"][loss]) and values["losses"][loss] > 0.0
                    for loss in ("total", "handoff", "state", "relation")
                )
                for values in gradient_rows.values()
            ),
            expected="all three loss paths have valid examples and every common module has finite nonzero gradient",
            observed=gradient_rows,
        ),
    ]

    def update_subset(arm: str, predicate) -> list[Mapping[str, Any]]:
        return [row for row in update_rows[arm]["parameters"] if predicate(row)]

    frozen_wavlm_updates = update_subset(
        "FROZEN-WAVLM", lambda value: value["name"].startswith("encoder.wavlm.")
    )
    fine_allowed_updates = update_subset(
        "FINETUNE-WAVLM", lambda value: wavlm_parameter_allowed(value["name"])
    )
    fine_frozen_updates = update_subset(
        "FINETUNE-WAVLM",
        lambda value: value["name"].startswith("encoder.wavlm.")
        and not wavlm_parameter_allowed(value["name"]),
    )
    scratch_encoder_updates = update_subset(
        "SCRATCH-PSEM", lambda value: value["name"].startswith("encoder.")
    )
    common_update_violations = {
        arm: [
            value["name"]
            for value in update_subset(
                arm,
                lambda row: row["owner"] in {"projection", "common_head"} and row["trainable"],
            )
            if not value["changed"]
        ]
        for arm in ARMS
    }
    update_checks = [
        check(
            "weight_update.frozen_wavlm_unchanged",
            bool(frozen_wavlm_updates)
            and all(not value["changed"] for value in frozen_wavlm_updates),
            expected={"all_wavlm_unchanged": True},
            observed={
                "parameter_count": len(frozen_wavlm_updates),
                "changed_names": [
                    value["name"] for value in frozen_wavlm_updates if value["changed"]
                ],
            },
        ),
        check(
            "weight_update.finetune_wavlm_allowed_only",
            bool(fine_allowed_updates)
            and all(value["changed"] for value in fine_allowed_updates)
            and bool(fine_frozen_updates)
            and all(not value["changed"] for value in fine_frozen_updates),
            expected={"every_allowed_changed": True, "every_frozen_unchanged": True},
            observed={
                "allowed_parameter_count": len(fine_allowed_updates),
                "allowed_unchanged_names": [
                    value["name"] for value in fine_allowed_updates if not value["changed"]
                ],
                "frozen_parameter_count": len(fine_frozen_updates),
                "frozen_changed_names": [
                    value["name"] for value in fine_frozen_updates if value["changed"]
                ],
            },
        ),
        check(
            "weight_update.scratch_encoder_changed",
            bool(scratch_encoder_updates)
            and all(value["changed"] for value in scratch_encoder_updates),
            expected={"every_scratch_encoder_parameter_changed": True},
            observed={
                "parameter_count": len(scratch_encoder_updates),
                "unchanged_names": [
                    value["name"] for value in scratch_encoder_updates if not value["changed"]
                ],
            },
        ),
        check(
            "weight_update.required_common_head_changed",
            all(not names for names in common_update_violations.values()),
            expected={arm: [] for arm in ARMS},
            observed=common_update_violations,
        ),
    ]
    expected_raw_waveform = expected_comparability["raw_waveform_tensor"]
    expected_augmented_waveform = expected_comparability["augmented_waveform_tensor"]
    expected_target_batch = expected_comparability["target_batch_tensors"]
    shared_identity_fields = (
        "row_id",
        "center_id",
        "source_id",
        "source_waveform_manifest_sha256",
        "augmentation_decision_sha256",
        "sampling_role",
    )
    identity_observed = {
        field: {arm: values[field] for arm, values in comparability_rows.items()}
        for field in shared_identity_fields
    }
    comparability_checks = [
        check(
            "comparability.raw_waveform_identical",
            all(
                values["raw_waveform_tensor"] == expected_raw_waveform
                and values["augmented_waveform_tensor"] == expected_augmented_waveform
                for values in comparability_rows.values()
            ),
            expected={
                "raw_waveform_tensor": expected_raw_waveform,
                "augmented_waveform_tensor": expected_augmented_waveform,
            },
            observed={
                arm: {
                    "raw_waveform_tensor": values["raw_waveform_tensor"],
                    "augmented_waveform_tensor": values["augmented_waveform_tensor"],
                }
                for arm, values in comparability_rows.items()
            },
        ),
        check(
            "comparability.targets_identical",
            all(
                values["target_sha256"] == expected_comparability["target_sha256"]
                and values["target_batch_tensors"] == expected_target_batch
                for values in comparability_rows.values()
            ),
            expected={
                "target_sha256": expected_comparability["target_sha256"],
                "target_batch_tensors": expected_target_batch,
            },
            observed={
                arm: {
                    "target_sha256": values["target_sha256"],
                    "target_batch_tensors": values["target_batch_tensors"],
                }
                for arm, values in comparability_rows.items()
            },
        ),
        check(
            "comparability.sampling_role_identical",
            all(len(set(values.values())) == 1 for values in identity_observed.values())
            and comparability_rows[ARMS[0]]["row_id"] == row["row_id"]
            and comparability_rows[ARMS[0]]["center_id"]
            == f"{row['source_id']}:{row['boundary_sample']}"
            and comparability_rows[ARMS[0]]["source_waveform_manifest_sha256"]
            == session.waveform_sha256,
            expected={
                "row_id": row["row_id"],
                "center_id": f"{row['source_id']}:{row['boundary_sample']}",
                "source_id": row["source_id"],
                "source_waveform_manifest_sha256": session.waveform_sha256,
                "augmentation_decision_sha256": canonical_sha256(row["augmentation"]),
                "sampling_role": row["sampling_role"],
            },
            observed=identity_observed,
        ),
        check(
            "comparability.observed_frontier_identical",
            len({values["observed_frontier_sample"] for values in comparability_rows.values()})
            == 1,
            expected=target.observed_frontier_sample,
            observed={
                arm: values["observed_frontier_sample"]
                for arm, values in comparability_rows.items()
            },
        ),
        check(
            "comparability.evaluation_reference_identical",
            len(
                {
                    tuple(values["unsnapped_handoff_event_samples"])
                    for values in comparability_rows.values()
                }
            )
            == 1
            and comparability_rows[ARMS[0]]["unsnapped_handoff_event_samples"]
            == list(target.handoff_event_samples),
            expected=list(target.handoff_event_samples),
            observed={
                arm: values["unsnapped_handoff_event_samples"]
                for arm, values in comparability_rows.items()
            },
        ),
        check(
            "comparability.only_encoder_strategy_differs",
            len({values["common_head_sha256"] for values in comparability_rows.values()}) == 1
            and len(
                {
                    canonical_sha256(values["common_head_contract"])
                    for values in comparability_rows.values()
                }
            )
            == 1
            and len(
                {
                    canonical_sha256(values["loss_contract"])
                    for values in comparability_rows.values()
                }
            )
            == 1
            and comparability_rows["FROZEN-WAVLM"]["initial_wavlm_sha256"]
            == comparability_rows["FINETUNE-WAVLM"]["initial_wavlm_sha256"]
            and comparability_rows["FROZEN-WAVLM"]["initial_wavlm_sha256"] is not None
            and comparability_rows["SCRATCH-PSEM"]["initial_wavlm_sha256"] is None
            and {values["encoder_strategy"]["arm"] for values in comparability_rows.values()}
            == set(ARMS),
            expected={
                "common_head_shared": True,
                "losses_and_class_weights_shared": True,
                "pretrained_initial_wavlm_shared": True,
                "scratch_has_no_wavlm": True,
                "encoder_strategy_arms": list(ARMS),
            },
            observed={
                "common_head_sha256": common_head_sha256,
                "common_head_contracts": {
                    arm: values["common_head_contract"]
                    for arm, values in comparability_rows.items()
                },
                "loss_contracts": {
                    arm: values["loss_contract"] for arm, values in comparability_rows.items()
                },
                "initial_wavlm_sha256": {
                    arm: values["initial_wavlm_sha256"]
                    for arm, values in comparability_rows.items()
                },
                "encoder_strategies": {
                    arm: values["encoder_strategy"] for arm, values in comparability_rows.items()
                },
            },
        ),
    ]
    graph_payload["checks"] = graph_checks
    parameter_payload["checks"] = parameter_checks
    gradient_payload["checks"] = gradient_checks
    update_payload["checks"] = update_checks
    comparability_payload["checks"] = comparability_checks
    graph_artifact = _write_json(output_root / "audits" / "model_graphs.json", graph_payload)
    parameter_artifact = _write_json(
        output_root / "audits" / "parameter_inventory.json", parameter_payload
    )
    gradient_artifact = _write_json(
        output_root / "audits" / "gradient_canary.json", gradient_payload
    )
    update_artifact = _write_json(
        output_root / "audits" / "weight_update_canary.json", update_payload
    )
    comparability_artifact = _write_json(
        output_root / "audits" / "arm_comparability.json", comparability_payload
    )
    metric_checks, metric_details = _metric_contract_audit()
    metric_artifact = _write_json(
        output_root / "audits" / "metric_contract.json",
        {
            "schema_version": 1,
            "artifact_role": "psem_metric_contract",
            "checks": metric_checks,
            "details": metric_details,
        },
    )
    receipts = {
        "model_graphs": _write_named_receipt(
            output_root,
            "model_graphs",
            "psem_model_graph_receipt",
            graph_checks,
            {"artifacts": [graph_artifact]},
        ),
        "parameter_inventory": _write_named_receipt(
            output_root,
            "parameter_inventory",
            "psem_parameter_inventory",
            parameter_checks,
            {"artifacts": [parameter_artifact]},
        ),
        "gradient_canary": _write_named_receipt(
            output_root,
            "gradient_canary",
            "psem_gradient_canary",
            gradient_checks,
            {"artifacts": [gradient_artifact]},
            validation_context={"parameter_inventory": parameter_payload},
        ),
        "weight_update_canary": _write_named_receipt(
            output_root,
            "weight_update_canary",
            "psem_weight_update_canary",
            update_checks,
            {"artifacts": [update_artifact]},
            validation_context={"parameter_inventory": parameter_payload},
        ),
        "arm_comparability": _write_named_receipt(
            output_root,
            "arm_comparability",
            "psem_arm_comparability",
            comparability_checks,
            {"artifacts": [comparability_artifact]},
            validation_context={"comparability": expected_comparability},
        ),
        "metric_contract": _write_named_receipt(
            output_root,
            "metric_contract",
            "psem_metric_contract",
            metric_checks,
            {"artifacts": [metric_artifact]},
        ),
    }
    return receipts
