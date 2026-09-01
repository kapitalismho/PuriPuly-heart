from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.psem_sortformer_adaptation_depth.authority_registry import (
    require_registered_execution,
)
from experiments.psem_sortformer_adaptation_depth.preflight import (
    PreflightPaths,
    _safe_external_output_root,
    build_preflight,
    canonical_sha256,
    sha256_file,
)
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    LOW_LATENCY_STREAMING,
    canary_bundle_runtime_passed,
    model_graph_runtime_passed,
    parameter_inventory_runtime_passed,
)
from experiments.psem_training_strategy_gate.sampling import (
    DEV_ROLE,
    EVAL_ROLE,
    TRAIN_ROLE,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
DATA_ROOT = REPOSITORY_ROOT / "experiments" / "psem_training_strategy_gate" / "data" / "v2"
SOURCE_MANIFEST = DATA_ROOT / "source_manifest.jsonl"
TOPOLOGY_MANIFEST = DATA_ROOT / "topology_manifest.jsonl"
SPLIT_MANIFEST = DATA_ROOT / "split_manifest.json"
EXPECTED_DATA_HASHES = {
    "source_manifest": "76d5a6640ffabbc3cf91c25f5a94284f9869ad266e621ee06f48a987d5d7c6de",
    "topology_manifest": "728c33d17d239dedf08eed9e014cd7e42f4b980c9bcb5b7826c67449f897d7cd",
    "split_manifest": "dce084ca8394f70e4f7fe4c72687bbfd95998d26e9ce43e600ef2eb8a65490b4",
}
EXPECTED_COUNTS = {
    TRAIN_ROLE: {"AMI": 50, "AliMeeting": 14},
    DEV_ROLE: {"AMI": 7, "AliMeeting": 3},
    EVAL_ROLE: {"AMI": 11, "AliMeeting": 8},
}
EXPECTED_HOURS = {
    TRAIN_ROLE: {"AMI": 24.996625, "AliMeeting": 7.461752},
    DEV_ROLE: {"AMI": 4.568083, "AliMeeting": 1.537742},
    EVAL_ROLE: {"AMI": 6.033410, "AliMeeting": 3.974088},
}
CHECKPOINT_IDENTITY = {
    "repository": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "revision": "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d",
    "filename": "diar_streaming_sortformer_4spk-v2.1.nemo",
    "sha256": "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8",
}
NEMO_REVISION = "1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6"
Q8_IDENTITY = {
    "filename": "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
    "sha256": "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9",
    "runtime_source_revision": "d42c3bbdfa2f63c37e5891e27de47a612d62f221",
}
Q8_POSTERIOR_SESSIONS = {
    "path": "experiments/psem_frozen_ceiling_gate/frozen_inputs/posterior_sessions.npz",
    "sha256": "27b7eaaa5c2ee332c3b81c048f8e2666499da9a3a4d7e46e8c994876c6ddcee8",
}
EXPECTED_EVALUATOR_PROVENANCE_SHA256 = (
    "2e94f1e801a4d84a33f6c40221018a3b6937cddaf7a759f87d85b905086c6cca"
)


def _candidate_head_is_resume_compatible(observed_head: object, current_head: str) -> bool:
    from experiments.psem_sortformer_adaptation_depth.execution import (
        candidate_git_head_is_resume_compatible,
    )

    return candidate_git_head_is_resume_compatible(observed_head, current_head)


EXPECTED_EVALUATOR_ARTIFACTS = {
    "gt_action_oracle": {
        "path": "experiments/psem_relative_occupancy_gate/decoder.py",
        "sha256": "b836be0cab5f401c38e294f4e7c48af90e431668c3ac8c5c6460bff371d6c5f8",
    },
    "issue98_evaluator": {
        "path": "experiments/psem_ontology_simplification_gate/evaluate_simplified_ontologies.py",
        "sha256": "015b2c1a7ef600c99885af5a4c1e86f386fec357e905ca9b1f572019682b41b4",
    },
    "oracle_mapping_code": {
        "path": "experiments/psem_relative_occupancy_gate/model_decode.py",
        "sha256": "83469df925883cd5212a03bef6b9fd666b5f03f18ba8b65739b2a99ea0322fe4",
    },
    "oracle_mapping_coverage": {
        "path": "experiments/psem_frozen_ceiling_gate/oracle_mapping_coverage.json",
        "sha256": "6723cc7150b0f6a17e720e4950b59d30554320af821632e41976297671447b35",
    },
    "oracle_mapping_ledger": {
        "path": "experiments/psem_frozen_ceiling_gate/oracle_mapping_ledger.jsonl",
        "sha256": "43064f16b2fc2419b681a99c81a31cd8e2cb95362624f35d99360665902c5525",
    },
    "product_evaluator": {
        "path": "experiments/psem_relative_occupancy_gate/model_evaluate.py",
        "sha256": "15cc61daab0faaa25ab7487798cfc45082802cf9f0ceca1f1345ed6bb81f0bcc",
    },
}
EXECUTED_ISSUE99_EVALUATOR_ARTIFACTS = {
    "session_builder": {
        "path": "experiments/psem_frozen_ceiling_gate/build_ceiling_examples.py",
        "sha256": "bfaaa5d87ffd0761c3895787c1066417109b802f42b4939a10fb6073f2313a7b",
    },
    "source_time_evaluator": {
        "path": "experiments/psem_frozen_ceiling_gate/evaluate_ceiling.py",
        "sha256": "dee7a20ea69181a0854347f4102cd0bac326c86306394500bf3fe5bc79eeb1d6",
    },
    "action_reference_builder": {
        "path": "experiments/psem_frozen_ceiling_gate/experiment_support.py",
        "sha256": "7ddc8b5b6ba80c9053b34dd7cc7d2dbbd75599325792b31bb3ece555e0f5c5c3",
    },
    "action_reference_ledger": {
        "path": "experiments/psem_frozen_ceiling_gate/action_reference_ledger.jsonl",
        "sha256": "0294d0fe35964e138fd9f4a295adbb913056bccc0a0ab2ffa9c6b015b772ae19",
    },
    "action_reference_coverage": {
        "path": "experiments/psem_frozen_ceiling_gate/action_reference_coverage.json",
        "sha256": "eb893222c2ca6b709263d65701c06bf73560bacbc4fc0e78dc611b90e6ee1959",
    },
    "issue99_g_frontier": {
        "path": "experiments/psem_frozen_ceiling_gate/results/frozen_ceiling_1/gt_causal_action_frontier.json",
        "sha256": "6b93f4e7e0dd584fdc9fdaace72ea61d7df36ebad1919c5400c4d245462b039d",
    },
    "issue99_q8_s_current": {
        "path": "experiments/psem_frozen_ceiling_gate/results/frozen_ceiling_1/scalar_current_metrics.json",
        "sha256": "e7dce5b500c27e105fa313053d85529e570a12a0ca72c970148562d22821b0f7",
    },
}


class ReceiptContractError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ReceiptContractError(f"invalid JSONL artifact: {path}")
    return values


def build_data_split_receipt() -> dict[str, Any]:
    observed_hashes = {
        "source_manifest": sha256_file(SOURCE_MANIFEST),
        "topology_manifest": sha256_file(TOPOLOGY_MANIFEST),
        "split_manifest": sha256_file(SPLIT_MANIFEST),
    }
    if observed_hashes != EXPECTED_DATA_HASHES:
        raise ReceiptContractError("V2 data artifacts differ from the issue-107 contract")
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assignments: dict[str, str] = {}
    component_by_source: dict[str, str] = {}
    for component in split["assignments"]["components"]:
        for source_id in component["source_ids"]:
            if source_id in assignments:
                raise ReceiptContractError("a source is assigned more than once")
            assignments[source_id] = component["role"]
            component_by_source[source_id] = component["component_id"]
    source_rows = {row["source_id"]: row for row in _jsonl(SOURCE_MANIFEST)}
    topology_rows = {row["source_id"]: row for row in _jsonl(TOPOLOGY_MANIFEST)}
    if set(assignments) != set(source_rows) or set(source_rows) != set(topology_rows):
        raise ReceiptContractError("V2 source, topology, and split identities differ")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, Counter[str]] = defaultdict(Counter)
    role_sources: dict[str, list[str]] = defaultdict(list)
    components_by_role: dict[str, set[str]] = defaultdict(set)
    for source_id, row in source_rows.items():
        role = assignments[source_id]
        corpus = row["corpus"]
        counts[role][corpus] += 1
        samples[role][corpus] += int(topology_rows[source_id]["scored_samples"])
        role_sources[role].append(source_id)
        components_by_role[role].add(component_by_source[source_id])
    observed_counts = {role: dict(counts[role]) for role in EXPECTED_COUNTS}
    if observed_counts != EXPECTED_COUNTS:
        raise ReceiptContractError(f"V2 role/corpus counts differ: {observed_counts}")
    observed_hours = {
        role: {
            corpus: round(samples[role][corpus] / 16000 / 3600, 6)
            for corpus in EXPECTED_HOURS[role]
        }
        for role in EXPECTED_HOURS
    }
    if observed_hours != EXPECTED_HOURS:
        raise ReceiptContractError(f"V2 scored hours differ: {observed_hours}")
    role_pairs = [(TRAIN_ROLE, DEV_ROLE), (TRAIN_ROLE, EVAL_ROLE), (DEV_ROLE, EVAL_ROLE)]
    if any(components_by_role[left] & components_by_role[right] for left, right in role_pairs):
        raise ReceiptContractError("a connected component crosses split roles")
    return {
        "schema_version": 1,
        "artifact_role": "data_split_receipt",
        "passed": True,
        "dataset_id": "PSEM-STRATEGY-DATA-v2",
        "artifact_hashes": observed_hashes,
        "counts": observed_counts,
        "scored_hours": observed_hours,
        "source_ids_by_role": {
            role: sorted(role_sources[role]) for role in (TRAIN_ROLE, DEV_ROLE, EVAL_ROLE)
        },
        "component_disjoint": True,
        "fit_roles": [TRAIN_ROLE],
        "checkpoint_selection_roles": [DEV_ROLE],
        "eval_open_policy": "once_after_candidate_checkpoint_threshold_report_and_rules_freeze",
        "eval_absent_from_sampling_and_overfit": True,
    }


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _contains_direct_q8_claim(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.lower().replace("-", " ").replace("_", " ")
        return "direct q8" in normalized or "q8 fine tuning" in normalized
    if isinstance(value, Mapping):
        return any(_contains_direct_q8_claim(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_direct_q8_claim(item) for item in value)
    return False


def paired_source_bootstrap_v1(
    deltas: Mapping[str, float], *, seed: int, resamples: int
) -> dict[str, Any]:
    if (
        not deltas
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 2000
        or not all(_finite_number(value) for value in deltas.values())
    ):
        raise ReceiptContractError("paired source bootstrap inputs are invalid")
    source_ids = list(deltas)
    randomizer = random.Random(seed)
    estimates = [
        sum(float(deltas[source_ids[randomizer.randrange(len(source_ids))]]) for _ in source_ids)
        / len(source_ids)
        for _ in range(resamples)
    ]
    ordered = sorted(estimates)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return ordered[lower_index]
        weight = position - lower_index
        return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight

    return {
        "lower": quantile(0.025),
        "upper": quantile(0.975),
        "replicate_estimates_sha256": canonical_sha256(estimates),
    }


def _validate_prediction_artifact(row: Mapping[str, Any]) -> bytes:
    descriptor = row.get("prediction_artifact")
    if not isinstance(descriptor, Mapping):
        raise ReceiptContractError("lineage prediction artifact descriptor is absent")
    raw_path = descriptor.get("path")
    unresolved_path = Path(raw_path) if isinstance(raw_path, str) else None
    path = unresolved_path.resolve() if unresolved_path is not None else None
    output_value = row.get("experiment_output_root")
    output_root = Path(output_value).resolve() if isinstance(output_value, str) else None
    raw = path.read_bytes() if path is not None and path.is_file() else None
    if (
        path is None
        or unresolved_path is None
        or not unresolved_path.is_absolute()
        or not path.is_file()
        or path.is_relative_to(REPOSITORY_ROOT.resolve())
        or output_root is None
        or path != (output_root / "lineage_predictions" / f"{row.get('source_id')}.jsonl")
        or raw is None
        or descriptor.get("sha256") != hashlib.sha256(raw).hexdigest()
        or descriptor.get("size_bytes") != len(raw)
    ):
        raise ReceiptContractError(
            "lineage prediction artifact is absent, mutable, or in-repository"
        )
    source_id = row.get("source_id")
    frame_count = row.get("frame_count")
    count = 0
    first_start = None
    last_end = None
    first_frontier = None
    last_frontier = None
    previous_alive = [False, False, False, False]
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReceiptContractError("lineage prediction artifact is not UTF-8") from exc
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReceiptContractError("lineage prediction artifact contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise ReceiptContractError(f"lineage prediction row is invalid: {source_id}")
        start = value.get("source_frame_start_sample")
        end = value.get("source_frame_end_sample")
        frontier = value.get("model_evidence_frontier_source_sample")
        activity_logits = value.get("raw_sortformer_activity_logits")
        slot_alive = value.get("slot_alive")
        state_reset = value.get("state_reset")
        if (
            value.get("artifact_role") != "psem_sortformer_frame_prediction"
            or value.get("source_id") != source_id
            or not isinstance(start, int)
            or isinstance(start, bool)
            or end != start + 1280
            or frontier != start + 16640
            or (count and start != last_end)
            or not isinstance(activity_logits, list)
            or len(activity_logits) != 4
            or not all(_finite_number(item) for item in activity_logits)
            or not _finite_number(value.get("raw_anchor_present_logit"))
            or not _finite_number(value.get("raw_replacement_evidence_logit"))
            or not isinstance(slot_alive, list)
            or len(slot_alive) != 4
            or any(type(item) is not bool for item in slot_alive)
            or slot_alive != [True, True, True, True]
            or any(was_alive and not alive for was_alive, alive in zip(previous_alive, slot_alive))
            or type(state_reset) is not bool
            or state_reset is not (count == 0)
            or value.get("oracle_anchor_slot") not in {0, 1, 2, 3}
            or isinstance(value.get("oracle_anchor_slot"), bool)
            or (
                value.get("anchor_episode_id") is not None
                and not isinstance(value.get("anchor_episode_id"), str)
            )
            or value.get("artifact_context") != "trainable_checkpoint_lineage"
            or value.get("split_role") != row.get("split_role")
            or value.get("arm") != "F0-FROZEN-FLOAT"
            or value.get("seed") is not None
            or value.get("source_waveform_sha256") != row.get("source_waveform_sha256")
            or value.get("base_checkpoint_sha256") != CHECKPOINT_IDENTITY["sha256"]
            or value.get("trained_checkpoint_sha256") is not None
            or value.get("trained_checkpoint_receipt_sha256") is not None
            or value.get("runtime_identity_sha256") != row.get("runtime_identity_sha256")
            or value.get("candidate_git_head") != row.get("candidate_git_head")
            or value.get("candidate_code_identity_sha256")
            != row.get("candidate_code_identity_sha256")
            or value.get("experiment_output_root") != row.get("experiment_output_root")
        ):
            raise ReceiptContractError(f"lineage prediction row is invalid: {source_id}")
        first_start = start if first_start is None else first_start
        first_frontier = frontier if first_frontier is None else first_frontier
        last_end = end
        last_frontier = frontier
        previous_alive = slot_alive
        count += 1
    if (
        count != frame_count
        or descriptor.get("row_count") != frame_count
        or first_start != row.get("first_frame_start_sample")
        or last_end != row.get("last_frame_end_sample")
        or first_frontier != row.get("first_evidence_frontier_sample")
        or last_frontier != row.get("last_evidence_frontier_sample")
    ):
        raise ReceiptContractError(f"lineage prediction artifact coverage differs: {source_id}")
    return raw


def recompute_lineage_numeric_evidence(
    sources: Sequence[Mapping[str, Any]],
    evaluator_contract: Mapping[str, Any],
    *,
    artifact_bytes: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import load_sessions
    from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import session_row
    from experiments.psem_sortformer_adaptation_depth.evaluation import _adapt_session
    from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
        align_native_predictions,
    )

    snapshots = {
        session.source_id: session for session in load_sessions(validate_mapping_ledger=False)
    }
    posterior_rows = []
    product_deltas = {
        metric: {} for metric in ("contamination", "false_cuts", "missed_replacements")
    }
    for source in sources:
        source_id = source["source_id"]
        snapshot = snapshots.get(source_id)
        descriptor = source.get("prediction_artifact")
        raw_path = descriptor.get("path") if isinstance(descriptor, Mapping) else None
        path = Path(raw_path) if isinstance(raw_path, str) else None
        if snapshot is None or path is None:
            raise ReceiptContractError("lineage numeric source evidence is absent")
        try:
            raw = path.read_bytes() if artifact_bytes is None else artifact_bytes[source_id]
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptContractError("lineage numeric artifact is unreadable") from exc
        float_probabilities = align_native_predictions(snapshot, rows)["probabilities"].astype(
            np.float64
        )
        q8_probabilities = snapshot.probabilities.astype(np.float64)
        if float_probabilities.shape != q8_probabilities.shape:
            raise ReceiptContractError("lineage float/Q8 posterior geometry differs")
        absolute = np.abs(float_probabilities - q8_probabilities)
        posterior_rows.append(
            {
                "source_id": source_id,
                "metrics": {
                    "mean_absolute_posterior_delta": float(absolute.mean()),
                    "maximum_absolute_posterior_delta": float(absolute.max()),
                },
            }
        )
        float_session, selected_float, _, _ = _adapt_session(snapshot, rows)
        float_row = session_row(
            float_session,
            (selected_float < 0.5).astype(np.float32),
            condition="F0-FROZEN-FLOAT",
            probe_class="current",
            threshold=0.5,
            confirmation_ms=500,
            time_condition="causal",
        )
        q8_selected = q8_probabilities[:, 0]
        q8_row = session_row(
            snapshot,
            (q8_selected < 0.5).astype(np.float32),
            condition="Q8-S-current",
            probe_class="current",
            threshold=0.5,
            confirmation_ms=500,
            time_condition="causal",
        )
        float_metrics = float_row["metrics"]
        q8_metrics = q8_row["metrics"]
        metric_fields = {
            "contamination": "exclusive_other_contamination_seconds",
            "false_cuts": "false_cut_count",
            "missed_replacements": "missed_replacement_count",
        }
        for metric, field in metric_fields.items():
            product_deltas[metric][source_id] = float(float_metrics[field]) - float(
                q8_metrics[field]
            )
    descriptors = [
        {
            "source_id": row["source_id"],
            "sha256": row["prediction_artifact"]["sha256"],
            "size_bytes": row["prediction_artifact"]["size_bytes"],
            "row_count": row["prediction_artifact"]["row_count"],
        }
        for row in sources
    ]
    float_set_sha = canonical_sha256(descriptors)
    q8_set_sha = evaluator_contract["q8_posterior_sessions"]["sha256"]
    product_metrics = {}
    materially_different = 0
    for index, metric in enumerate(("contamination", "false_cuts", "missed_replacements")):
        deltas = product_deltas[metric]
        interval = paired_source_bootstrap_v1(deltas, seed=10700 + index, resamples=2000)
        excludes = interval["upper"] < 0 or interval["lower"] > 0
        materially_different += int(excludes)
        product_metrics[metric] = {
            "paired_source_deltas": deltas,
            "point_estimate": sum(deltas.values()) / len(deltas),
            "bootstrap_95": {
                **interval,
                "seed": 10700 + index,
                "resamples": 2000,
                "unit": "source_or_meeting",
                "algorithm": "paired_source_bootstrap_v1",
            },
            "paired_source_bootstrap_interval_excludes_zero": excludes,
        }
    return {
        "float_vs_q8_posterior_deltas": {
            "per_source": posterior_rows,
            "paired_source_count": len(sources),
            "float_prediction_set_sha256": float_set_sha,
            "q8_prediction_set_sha256": q8_set_sha,
        },
        "float_vs_q8_product_deltas": {
            "metrics": product_metrics,
            "float_prediction_set_sha256": float_set_sha,
            "q8_prediction_set_sha256": q8_set_sha,
        },
        "study_label": (
            "float-checkpoint adaptation study"
            if materially_different >= 2
            else "Q8-linked float adaptation study"
        ),
        "float_checkpoint_materially_differs_on_two_or_more_metrics": materially_different >= 2,
    }


def validate_trainable_checkpoint_lineage(
    receipt: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    evaluator_contract: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
        validate_dependency_lock,
    )

    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_role") != "trainable_checkpoint_lineage"
        or receipt.get("checkpoint") != CHECKPOINT_IDENTITY
    ):
        raise ReceiptContractError("lineage checkpoint identity differs from the frozen artifact")
    if receipt.get("nemo_revision") != NEMO_REVISION:
        raise ReceiptContractError("lineage NeMo revision differs from the frozen commit")
    if receipt.get("q8_baseline") != Q8_IDENTITY:
        raise ReceiptContractError("lineage Q8 baseline differs from #99")
    output_value = receipt.get("experiment_output_root")
    output_root = Path(output_value).resolve() if isinstance(output_value, str) else None
    if output_root is None or not _safe_external_output_root(output_root):
        raise ReceiptContractError("lineage output root is not a safe external cache")
    from experiments.psem_sortformer_adaptation_depth.execution import (
        validate_current_candidate_identity,
    )

    validate_current_candidate_identity(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_candidate_code_identity",
            "git_head": receipt.get("candidate_git_head"),
            "worktree_clean": True,
            "artifact_sha256s": receipt.get("candidate_artifact_sha256s"),
            "payload_sha256": receipt.get("candidate_code_identity_sha256"),
        }
    )
    expected_evaluator = evaluator_reconstruction_contract()
    graph = runtime_identity.get("model_graph")
    dependency_lock = runtime_identity.get("dependency_lock")
    dependency_lock_path = (
        Path(dependency_lock.get("path")) if isinstance(dependency_lock, Mapping) else None
    )
    try:
        observed_dependency_lock = (
            validate_dependency_lock(dependency_lock_path)
            if dependency_lock_path is not None and dependency_lock_path.is_absolute()
            else None
        )
    except Exception as exc:
        raise ReceiptContractError("lineage dependency lock is not reproducible") from exc
    if (
        runtime_identity.get("checkpoint_sha256") != CHECKPOINT_IDENTITY["sha256"]
        or runtime_identity.get("nemo_revision") != NEMO_REVISION
        or not isinstance(dependency_lock, Mapping)
        or observed_dependency_lock != dependency_lock
        or runtime_identity.get("dependency_lock_sha256")
        != runtime_identity["dependency_lock"].get("sha256")
        or not isinstance(graph, Mapping)
        or not model_graph_runtime_passed(graph)
        or evaluator_contract != expected_evaluator
        or receipt.get("runtime_identity_sha256") != canonical_sha256(runtime_identity)
        or receipt.get("evaluator_contract_sha256") != canonical_sha256(expected_evaluator)
        or receipt.get("dependency_lock_sha256") != runtime_identity.get("dependency_lock_sha256")
        or receipt.get("executable_graph_sha256") != graph.get("executable_graph_sha256")
    ):
        raise ReceiptContractError("lineage is not bound to the executable runtime and evaluator")
    streaming = receipt.get("streaming")
    expected_streaming = {
        **LOW_LATENCY_STREAMING,
        "sample_rate_hz": 16000,
        "native_frame_samples": 1280,
        "slot_count": 4,
        "algorithmic_evidence_delay_samples": 16640,
        "reset_policy": "declared_source_or_reset_boundary_only",
    }
    if streaming != expected_streaming:
        raise ReceiptContractError("lineage streaming geometry differs from #99 low latency")
    split_receipt = build_data_split_receipt()
    expected_sources = sorted(
        split_receipt["source_ids_by_role"][DEV_ROLE]
        + split_receipt["source_ids_by_role"][EVAL_ROLE]
    )
    sources = receipt.get("sources")
    source_ids = [row.get("source_id") for row in sources] if isinstance(sources, list) else []
    if source_ids != expected_sources or len(set(source_ids)) != len(source_ids):
        raise ReceiptContractError(
            "lineage does not cover the exact #99 DEV/EVAL source identities"
        )
    source_rows = {row["source_id"]: row for row in _jsonl(SOURCE_MANIFEST)}
    role_by_source = {
        source_id: role
        for role, values in split_receipt["source_ids_by_role"].items()
        for source_id in values
    }
    artifact_bytes: dict[str, bytes] = {}
    for row in sources:
        frame_count = row.get("frame_count")
        first_start = row.get("first_frame_start_sample")
        last_end = row.get("last_frame_end_sample")
        first_frontier = row.get("first_evidence_frontier_sample")
        last_frontier = row.get("last_evidence_frontier_sample")
        source_manifest_row = source_rows.get(row.get("source_id"), {})
        duration_samples = source_manifest_row.get("duration_samples")
        expected_last_end = (
            duration_samples - duration_samples % 1280
            if isinstance(duration_samples, int) and not isinstance(duration_samples, bool)
            else None
        )
        if (
            not isinstance(frame_count, int)
            or isinstance(frame_count, bool)
            or frame_count <= 0
            or not all(
                isinstance(value, int)
                for value in (first_start, last_end, first_frontier, last_frontier)
            )
            or first_start != 0
            or last_end != expected_last_end
            or frame_count != expected_last_end // 1280
            or row.get("source_duration_samples") != duration_samples
            or row.get("source_tail_samples_excluded") != duration_samples % 1280
            or first_start % 1280
            or last_end - first_start != frame_count * 1280
            or first_frontier - first_start != 16640
            or last_frontier - (last_end - 1280) != 16640
            or row.get("hidden_tensor_identity") != "sortformer.transformer_encoder.output"
            or row.get("hidden_dimension") != 192
            or row.get("slot_count") != 4
            or row.get("slot_alive_policy") != "issue_99_all_four_stable_columns_alive"
            or row.get("split_role") != role_by_source[row.get("source_id")]
            or row.get("source_waveform_sha256")
            != source_rows[row.get("source_id")]["waveform_sha256"]
            or row.get("activity_logit_identity")
            != "sortformer.sortformer_modules.single_hidden_to_spks.output_pre_sigmoid"
            or row.get("executable_graph_sha256") != graph.get("executable_graph_sha256")
            or row.get("dependency_lock_sha256") != runtime_identity.get("dependency_lock_sha256")
            or row.get("runtime_identity_sha256") != receipt.get("runtime_identity_sha256")
            or row.get("candidate_git_head") != receipt.get("candidate_git_head")
            or row.get("candidate_code_identity_sha256")
            != receipt.get("candidate_code_identity_sha256")
            or row.get("experiment_output_root") != str(output_root)
        ):
            raise ReceiptContractError(
                f"lineage frame/tap invariant failed: {row.get('source_id')}"
            )
        artifact_bytes[str(row["source_id"])] = _validate_prediction_artifact(row)
    recomputed_numeric = recompute_lineage_numeric_evidence(
        sources,
        expected_evaluator,
        artifact_bytes=artifact_bytes,
    )
    from experiments.psem_sortformer_adaptation_depth.lineage import lineage_authorization

    if (
        receipt.get("float_vs_q8_posterior_deltas")
        != recomputed_numeric["float_vs_q8_posterior_deltas"]
        or receipt.get("float_vs_q8_product_deltas")
        != recomputed_numeric["float_vs_q8_product_deltas"]
        or receipt.get("lineage_authorization_sha256") != lineage_authorization()["payload_sha256"]
    ):
        raise ReceiptContractError("lineage numeric evidence is not reproducible from artifacts")
    float_prediction_set_sha256 = canonical_sha256(
        [
            {
                "source_id": row["source_id"],
                "sha256": row["prediction_artifact"]["sha256"],
                "size_bytes": row["prediction_artifact"]["size_bytes"],
                "row_count": row["prediction_artifact"]["row_count"],
            }
            for row in sources
        ]
    )
    q8_prediction_set_sha256 = expected_evaluator["q8_posterior_sessions"]["sha256"]
    posterior = receipt.get("float_vs_q8_posterior_deltas")
    product = receipt.get("float_vs_q8_product_deltas")
    posterior_rows = posterior.get("per_source") if isinstance(posterior, dict) else None
    if (
        not isinstance(posterior_rows, list)
        or [row.get("source_id") for row in posterior_rows] != expected_sources
        or any(
            set(row.get("metrics", {}))
            != {"mean_absolute_posterior_delta", "maximum_absolute_posterior_delta"}
            or not all(_finite_number(value) and value >= 0 for value in row["metrics"].values())
            for row in posterior_rows
        )
        or posterior.get("paired_source_count") != len(expected_sources)
        or posterior.get("float_prediction_set_sha256") != float_prediction_set_sha256
        or posterior.get("q8_prediction_set_sha256") != q8_prediction_set_sha256
    ):
        raise ReceiptContractError("paired float/Q8 posterior deltas are incomplete")
    required_metrics = {"contamination", "false_cuts", "missed_replacements"}
    metrics = product.get("metrics") if isinstance(product, dict) else None
    if (
        not isinstance(metrics, dict)
        or set(metrics) != required_metrics
        or product.get("float_prediction_set_sha256") != float_prediction_set_sha256
        or product.get("q8_prediction_set_sha256") != q8_prediction_set_sha256
    ):
        raise ReceiptContractError("paired float/Q8 product deltas are incomplete")
    materially_different = 0
    for metric, value in metrics.items():
        deltas = value.get("paired_source_deltas") if isinstance(value, dict) else None
        interval = value.get("bootstrap_95") if isinstance(value, dict) else None
        lower = interval.get("lower") if isinstance(interval, dict) else None
        upper = interval.get("upper") if isinstance(interval, dict) else None
        excludes_zero = _finite_number(lower) and _finite_number(upper) and (upper < 0 or lower > 0)
        recomputed_point = (
            sum(float(delta) for delta in deltas.values()) / len(deltas)
            if isinstance(deltas, dict) and deltas
            else None
        )
        expected_bootstrap = (
            paired_source_bootstrap_v1(
                deltas,
                seed=interval.get("seed"),
                resamples=interval.get("resamples"),
            )
            if isinstance(deltas, dict)
            and deltas
            and isinstance(interval, dict)
            and isinstance(interval.get("seed"), int)
            and not isinstance(interval.get("seed"), bool)
            and isinstance(interval.get("resamples"), int)
            and not isinstance(interval.get("resamples"), bool)
            and interval["resamples"] >= 2000
            else None
        )
        if (
            not isinstance(deltas, dict)
            or list(deltas) != expected_sources
            or not all(_finite_number(delta) for delta in deltas.values())
            or not _finite_number(value.get("point_estimate"))
            or not _finite_number(recomputed_point)
            or not math.isclose(
                float(value["point_estimate"]),
                float(recomputed_point),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not isinstance(interval, dict)
            or interval.get("unit") != "source_or_meeting"
            or not isinstance(interval.get("resamples"), int)
            or isinstance(interval.get("resamples"), bool)
            or interval["resamples"] < 2000
            or interval.get("algorithm") != "paired_source_bootstrap_v1"
            or not isinstance(interval.get("seed"), int)
            or isinstance(interval.get("seed"), bool)
            or expected_bootstrap is None
            or interval.get("replicate_estimates_sha256")
            != expected_bootstrap["replicate_estimates_sha256"]
            or not _finite_number(lower)
            or not _finite_number(upper)
            or lower > upper
            or not math.isclose(
                float(lower), float(expected_bootstrap["lower"]), rel_tol=0, abs_tol=1e-12
            )
            or not math.isclose(
                float(upper), float(expected_bootstrap["upper"]), rel_tol=0, abs_tol=1e-12
            )
            or value.get("paired_source_bootstrap_interval_excludes_zero") is not excludes_zero
        ):
            raise ReceiptContractError(f"paired float/Q8 product delta is invalid: {metric}")
        materially_different += int(excludes_zero)
    study_label = (
        "float-checkpoint adaptation study"
        if materially_different >= 2
        else "Q8-linked float adaptation study"
    )
    if (
        receipt.get("study_label") != study_label
        or receipt.get("study_label") != recomputed_numeric["study_label"]
        or receipt.get("direct_q8_fine_tuning_claim") is not False
        or _contains_direct_q8_claim(receipt)
    ):
        raise ReceiptContractError("lineage study claim overstates the paired numeric evidence")
    payload = {
        **{key: value for key, value in receipt.items() if key != "payload_sha256"},
        "passed": True,
        "float_checkpoint_materially_differs_on_two_or_more_metrics": materially_different >= 2,
        "study_label": study_label,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def evaluator_reconstruction_contract() -> dict[str, Any]:
    provenance_path = (
        REPOSITORY_ROOT
        / "experiments"
        / "psem_frozen_ceiling_gate"
        / "frozen_inputs"
        / "source_evidence_provenance.json"
    )
    resolved_provenance = provenance_path.resolve()
    if (
        not resolved_provenance.is_relative_to(REPOSITORY_ROOT.resolve())
        or sha256_file(resolved_provenance) != EXPECTED_EVALUATOR_PROVENANCE_SHA256
    ):
        raise ReceiptContractError("#99 evaluator provenance identity drifted")
    provenance = json.loads(resolved_provenance.read_text(encoding="utf-8"))
    artifacts = provenance["artifacts"]
    required = set(EXPECTED_EVALUATOR_ARTIFACTS)
    if any(artifacts.get(key) != EXPECTED_EVALUATOR_ARTIFACTS[key] for key in required):
        raise ReceiptContractError("#99 evaluator provenance descriptors drifted")
    bound = {}
    for key in sorted(required):
        descriptor = EXPECTED_EVALUATOR_ARTIFACTS[key]
        path = (REPOSITORY_ROOT / descriptor["path"]).resolve()
        if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
            raise ReceiptContractError(f"#99 evaluator artifact escapes the repository: {key}")
        digest = sha256_file(path)
        if digest != descriptor["sha256"]:
            raise ReceiptContractError(f"#99 evaluator artifact drifted: {key}")
        bound[key] = {"path": str(path), "sha256": digest}
    executed = {}
    for key, descriptor in sorted(EXECUTED_ISSUE99_EVALUATOR_ARTIFACTS.items()):
        path = (REPOSITORY_ROOT / descriptor["path"]).resolve()
        if (
            not path.is_relative_to(REPOSITORY_ROOT.resolve())
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ReceiptContractError(f"executed #99 evaluator artifact drifted: {key}")
        executed[key] = {"path": str(path), "sha256": descriptor["sha256"]}
    q8_posterior_path = (REPOSITORY_ROOT / Q8_POSTERIOR_SESSIONS["path"]).resolve()
    if (
        not q8_posterior_path.is_relative_to(REPOSITORY_ROOT.resolve())
        or sha256_file(q8_posterior_path) != Q8_POSTERIOR_SESSIONS["sha256"]
    ):
        raise ReceiptContractError("#99 Q8 posterior sessions drifted")
    return {
        "schema_version": 1,
        "artifact_role": "evaluator_reconstruction_contract",
        "passed": True,
        "provenance_sha256": EXPECTED_EVALUATOR_PROVENANCE_SHA256,
        "artifacts": bound,
        "executed_evaluator_artifacts": executed,
        "q8_posterior_sessions": {
            **Q8_POSTERIOR_SESSIONS,
            "path": str(q8_posterior_path),
        },
        "required_reconstructions": [
            "issue-99-G-causal-frontier",
            "issue-99-fixed-simple-anchor-evaluator",
        ],
        "threshold_grid": [0.5],
        "confirmation_ms_grid": [500],
        "primary_cell": {"threshold": 0.5, "confirmation_ms": 500},
        "eval_threshold_selection_allowed": False,
    }


def _legacy_validate_overfit_canary(
    receipt: Mapping[str, Any],
    *,
    sampling_rows: list[Mapping[str, Any]],
    sampling_manifest_path: Path,
    corpus_by_source: Mapping[str, str],
    canary_receipts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.sampling import (
        load_sampling_rows,
        select_overfit_rows,
    )

    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    if receipt.get("payload_sha256") != canonical_sha256(payload):
        raise ReceiptContractError("overfit canary payload is not content-bound")
    if list(sampling_rows) != load_sampling_rows(sampling_manifest_path):
        raise ReceiptContractError("overfit rows differ from the persisted sampling manifest")
    if receipt.get("split_roles") != [TRAIN_ROLE] or receipt.get("eval_source_count") != 0:
        raise ReceiptContractError("overfit canary is not TRAIN-only")
    if receipt.get("duration_minutes") != 30 or receipt.get("maximum_optimizer_steps") != 500:
        raise ReceiptContractError("overfit canary budget differs from the frozen recipe")
    sources = receipt.get("sources")
    if not isinstance(sources, list) or len({row.get("source_id") for row in sources}) != 4:
        raise ReceiptContractError("overfit canary source inventory is absent")
    corpus_counts = Counter(row.get("corpus") for row in sources)
    if corpus_counts != Counter({"AMI": 2, "AliMeeting": 2}):
        raise ReceiptContractError("overfit canary must contain two sources per corpus")
    expected_rows = select_overfit_rows(sampling_rows, corpus_by_source)
    expected_row_ids = [row.get("row_id") for row in expected_rows]
    if (
        receipt.get("selection_rule") != "issue-107-overfit-source-v1+issue-107-overfit-window-v1"
        or receipt.get("sampling_manifest_sha256") != sha256_file(sampling_manifest_path)
        or receipt.get("selected_row_ids") != expected_row_ids
        or len(set(expected_row_ids)) != 60
        or receipt.get("selected_rows_sha256") != canonical_sha256(list(expected_rows))
    ):
        raise ReceiptContractError("overfit canary is not bound to the canonical 60 windows")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or not {"H-HEAD", "T2-TOP"} <= set(arms):
        raise ReceiptContractError("mandatory trainable arms are absent from the overfit canary")
    for arm, values in arms.items():
        if arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}:
            raise ReceiptContractError(f"unauthorized overfit arm: {arm}")
        initial = values.get("initial_replacement_loss")
        final = values.get("final_replacement_loss")
        native = values.get("final_native_sortformer_loss")
        average_precision = values.get("duration_weighted_replacement_average_precision")
        bound = canary_receipts.get(arm, {})
        gradient = bound.get("gradient_canary_receipt")
        update = bound.get("update_canary_receipt")
        timing = bound.get("timing_receipt")
        inventory = bound.get("parameter_inventory")
        graph = bound.get("model_graph_receipt")
        conditional_authorization = values.get("conditional_arm_audit_authorization")
        raw_overfit = {
            key: item
            for key, item in values.items()
            if key
            not in {
                "gradient_canary_sha256",
                "update_canary_sha256",
                "timing_receipt_sha256",
                "parameter_inventory_sha256",
                "model_graph_receipt_sha256",
            }
        }
        try:
            require_registered_execution("overfit-arm", raw_overfit)
            for kind, runtime_receipt in (
                ("gradient-canary", gradient),
                ("update-canary", update),
                ("timing-receipt", timing),
                ("parameter-inventory", inventory),
                ("model-graph", graph),
            ):
                require_registered_execution(kind, runtime_receipt)
        except Exception as exc:
            raise ReceiptContractError(
                f"overfit execution evidence is not registered: {arm}"
            ) from exc
        if (
            values.get("arm") != arm
            or values.get("sampling_manifest_sha256") != sha256_file(sampling_manifest_path)
            or values.get("overfit_input_identity_sha256") != canonical_sha256(list(expected_rows))
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in (initial, final, native)
            )
            or initial <= 0
            or final < 0
            or (initial - final) / initial < 0.3
            or not _finite_number(average_precision)
            or not 0.85 <= average_precision <= 1
            or values.get("optimizer_steps") != 500
            or not isinstance(gradient, Mapping)
            or not isinstance(update, Mapping)
            or not isinstance(timing, Mapping)
            or not isinstance(inventory, Mapping)
            or not isinstance(graph, Mapping)
            or not canary_bundle_runtime_passed(
                gradient,
                update,
                timing,
                arm,
                parameter_inventory_receipt=inventory,
                model_graph_receipt=graph,
            )
            or values.get("gradient_canary_sha256") != canonical_sha256(gradient)
            or values.get("update_canary_sha256") != canonical_sha256(update)
            or values.get("timing_receipt_sha256") != canonical_sha256(timing)
            or values.get("parameter_inventory_sha256") != canonical_sha256(inventory)
            or values.get("model_graph_receipt_sha256") != canonical_sha256(graph)
            or timing.get("algorithmic_evidence_delay_samples") != 16640
            or timing.get("native_frame_samples") != 1280
            or conditional_authorization != bound.get("conditional_arm_audit_authorization")
            or (
                arm == "TA-ALL-TEMPORAL"
                and (
                    not isinstance(conditional_authorization, Mapping)
                    or conditional_authorization.get("artifact_role")
                    != "conditional_arm_audit_authorization"
                    or conditional_authorization.get("arm") != arm
                    or conditional_authorization.get("payload_sha256")
                    != canonical_sha256(
                        {
                            key: value
                            for key, value in conditional_authorization.items()
                            if key != "payload_sha256"
                        }
                    )
                )
            )
            or (arm != "TA-ALL-TEMPORAL" and conditional_authorization is not None)
        ):
            raise ReceiptContractError(f"overfit canary failed for {arm}")
    validated = {**payload, "passed": True}
    return {**validated, "payload_sha256": canonical_sha256(validated)}


def _bound_payload(receipt: Mapping[str, Any], role: str) -> dict[str, Any]:
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    if receipt.get("artifact_role") != role or receipt.get("payload_sha256") != canonical_sha256(
        payload
    ):
        raise ReceiptContractError(f"receipt is absent or not content-bound: {role}")
    return payload


def _validate_runtime_preflight(receipt: Mapping[str, Any]) -> None:
    check_rows = receipt.get("checks")
    checks_by_id = (
        {
            row.get("id"): row
            for row in check_rows
            if isinstance(row, Mapping) and isinstance(row.get("id"), str)
        }
        if isinstance(check_rows, list)
        else {}
    )
    required_path_checks = {
        "checkpoint": "runtime.checkpoint_path",
        "corpus_root": "runtime.corpus_root",
        "reference_root": "runtime.reference_root",
        "output_root": "runtime.output_root",
        "protocol_registry_root": "runtime.protocol_registry_root",
    }
    if len(checks_by_id) != len(check_rows or []) or any(
        not isinstance(checks_by_id.get(check_id, {}).get("observed"), str)
        for check_id in required_path_checks.values()
    ):
        raise ReceiptContractError("runtime preflight path evidence is incomplete")
    preflight_paths = PreflightPaths(
        **{
            field: Path(checks_by_id[check_id]["observed"]).resolve()
            for field, check_id in required_path_checks.items()
        }
    )
    if receipt != build_preflight(preflight_paths, static_only=False):
        raise ReceiptContractError("runtime preflight does not match a current exact rerun")


def _legacy_validate_staged_authorization(
    receipt: Mapping[str, Any],
    arm: str,
    seed: int,
    dev_results: Sequence[Mapping[str, Any]],
) -> None:
    from experiments.psem_sortformer_adaptation_depth.protocol import (
        validate_staged_execution_state,
    )

    try:
        validate_staged_execution_state(receipt, dev_results)
    except Exception as exc:
        raise ReceiptContractError("staged authorization is not reproducible") from exc
    payload = _bound_payload(receipt, "staged_execution_state")
    from experiments.psem_sortformer_adaptation_depth.execution import candidate_code_identity

    current_identity = candidate_code_identity()
    if any(
        not isinstance(result.get("prediction_set"), Mapping)
        or result["prediction_set"].get("candidate_git_head") != current_identity["git_head"]
        or result["prediction_set"].get("candidate_code_identity_sha256")
        != current_identity["payload_sha256"]
        for result in dev_results
    ):
        raise ReceiptContractError("staged DEV evidence belongs to another candidate identity")
    if payload.get("eval_open_count") != 0 or payload.get("eval_used_for_development") is not False:
        raise ReceiptContractError("EVAL entered staged training authorization")
    completed = payload.get("completed_runs")
    if not isinstance(completed, list):
        raise ReceiptContractError("staged run history is absent")
    completed_ids = [
        (row.get("arm"), row.get("seed")) for row in completed if isinstance(row, Mapping)
    ]
    allowed_ids = {("F0-FROZEN-FLOAT", None)} | {
        (candidate_arm, candidate_seed)
        for candidate_arm in ("H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL")
        for candidate_seed in (7301, 7302)
    }
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("arm"), str)
        or (
            row.get("seed") is not None
            and (not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool))
        )
        or row.get("passed") is not True
        or row.get("evaluation_roles") != [DEV_ROLE]
        or row.get("dev_evidence_sha256") in {None, ""}
        for row in completed
    ) or any(value not in allowed_ids for value in completed_ids):
        raise ReceiptContractError("staged history is not bound to DEV-only evidence")
    if (
        not completed_ids
        or completed_ids[0] != ("F0-FROZEN-FLOAT", None)
        or len(set(completed_ids)) != len(completed_ids)
        or (arm, seed) in completed_ids
    ):
        raise ReceiptContractError("F0-FROZEN-FLOAT must complete before material training")
    positions = {value: index for index, value in enumerate(completed_ids)}
    for index, (completed_arm, completed_seed) in enumerate(completed_ids):
        if completed_arm in {"T2-TOP", "TA-ALL-TEMPORAL"} and (
            positions.get(("H-HEAD", 7301), len(completed_ids)) >= index
        ):
            raise ReceiptContractError("H-HEAD seed 7301 must precede deeper training")
        if completed_arm == "TA-ALL-TEMPORAL" and (
            positions.get(("T2-TOP", 7301), len(completed_ids)) >= index
        ):
            raise ReceiptContractError("T2-TOP seed 7301 must precede all-temporal training")
        if completed_seed == 7302 and (
            positions.get((completed_arm, 7301), len(completed_ids)) >= index
            or positions.get(("T2-TOP", 7301), len(completed_ids)) >= index
        ):
            raise ReceiptContractError("confirmation seeds precede the completed primary stage")
    completed_id_set = set(completed_ids)
    if arm in {"T2-TOP", "TA-ALL-TEMPORAL"} and ("H-HEAD", 7301) not in completed_id_set:
        raise ReceiptContractError("H-HEAD seed 7301 must complete before deeper training")

    escalation = payload.get("ta_escalation")

    def ta_opened() -> bool:
        if not isinstance(escalation, Mapping) or escalation.get("decision") != "opened":
            return False
        favorable = escalation.get("favorable_intervals")
        harmful = escalation.get("wholly_harmful_metric_counts_by_corpus")
        return bool(
            escalation.get("equal_corpus_not_pareto_dominated") is True
            and isinstance(favorable, Mapping)
            and any(
                metric in {"contamination", "missed_replacements"}
                and _finite_number(interval.get("lower"))
                and _finite_number(interval.get("upper"))
                and interval["lower"] <= interval["upper"]
                and interval["upper"] < 0
                and interval.get("unit") == "source_or_meeting"
                and isinstance(interval.get("resamples"), int)
                and not isinstance(interval.get("resamples"), bool)
                and interval["resamples"] >= 2000
                and interval.get("algorithm") == "paired_source_bootstrap_v1"
                for metric, interval in favorable.items()
                if isinstance(interval, Mapping)
            )
            and isinstance(harmful, Mapping)
            and harmful == {"AMI": harmful.get("AMI"), "AliMeeting": harmful.get("AliMeeting")}
            and all(
                isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2
                for value in harmful.values()
            )
            and escalation.get("slot_mapping_coverage_passed") is True
            and escalation.get("timing_gate_passed") is True
            and escalation.get("dev_evidence_sha256") not in {None, ""}
        )

    if arm == "TA-ALL-TEMPORAL" or any(
        completed_arm == "TA-ALL-TEMPORAL" for completed_arm, _ in completed_ids
    ):
        if ("T2-TOP", 7301) not in completed_id_set or not ta_opened():
            raise ReceiptContractError("TA-ALL-TEMPORAL was not opened by the frozen DEV rule")

    def confirmation_allowed(candidate_arm: str) -> bool:
        confirmation = payload.get("confirmation_seed_authorization")
        authorized = confirmation.get("arms") if isinstance(confirmation, Mapping) else None
        evidence_by_arm = (
            confirmation.get("evidence_by_arm") if isinstance(confirmation, Mapping) else None
        )
        arm_evidence = (
            evidence_by_arm.get(candidate_arm) if isinstance(evidence_by_arm, Mapping) else None
        )
        uncertainty = (
            arm_evidence.get("leader_difference_bootstrap_95")
            if isinstance(arm_evidence, Mapping)
            else None
        )
        uncertainty_allows = (
            isinstance(uncertainty, Mapping)
            and _finite_number(uncertainty.get("lower"))
            and _finite_number(uncertainty.get("upper"))
            and uncertainty["lower"] <= 0 <= uncertainty["upper"]
            and uncertainty.get("unit") == "source_or_meeting"
            and isinstance(uncertainty.get("resamples"), int)
            and not isinstance(uncertainty.get("resamples"), bool)
            and uncertainty["resamples"] >= 2000
            and uncertainty.get("algorithm") == "paired_source_bootstrap_v1"
        )
        return bool(
            (candidate_arm, 7301) in completed_id_set
            and ("T2-TOP", 7301) in completed_id_set
            and isinstance(escalation, Mapping)
            and escalation.get("decision") in {"opened", "closed"}
            and escalation.get("dev_evidence_sha256") not in {None, ""}
            and (
                escalation.get("decision") != "opened"
                or ("TA-ALL-TEMPORAL", 7301) in completed_id_set
            )
            and isinstance(authorized, list)
            and all(isinstance(value, str) for value in authorized)
            and len(set(authorized)) == len(authorized)
            and set(authorized) <= {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"}
            and isinstance(evidence_by_arm, Mapping)
            and set(evidence_by_arm) == set(authorized)
            and candidate_arm in authorized
            and confirmation.get("rule")
            == "dev_non_dominated_or_difference_within_paired_bootstrap_uncertainty"
            and confirmation.get("dev_evidence_sha256") not in {None, ""}
            and isinstance(arm_evidence, Mapping)
            and (arm_evidence.get("non_dominated") is True or uncertainty_allows)
        )

    completed_confirmation_arms = [
        completed_arm for completed_arm, completed_seed in completed_ids if completed_seed == 7302
    ]
    if seed == 7302 or completed_confirmation_arms:
        if not all(
            confirmation_allowed(candidate_arm)
            for candidate_arm in [*completed_confirmation_arms, *([arm] if seed == 7302 else [])]
        ):
            raise ReceiptContractError("confirmation seed is not authorized by DEV evidence")


def _legacy_validate_material_training_gate(
    *,
    arm: str,
    seed: int,
    preflight_receipt: Mapping[str, Any],
    sampling_validation: Mapping[str, Any],
    sampling_manifest_path: Path,
    sampling_rows: Sequence[Mapping[str, Any]],
    training_sessions: Mapping[str, Any],
    class_weight_receipt: Mapping[str, Any],
    lineage_receipt: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    evaluator_contract: Mapping[str, Any],
    parameter_inventory: Mapping[str, Any],
    gradient_receipt: Mapping[str, Any],
    update_receipt: Mapping[str, Any],
    timing_receipt: Mapping[str, Any],
    overfit_receipt: Mapping[str, Any],
    overfit_canary_receipts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    staged_execution_receipt: Mapping[str, Any],
    staged_dev_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"} or seed not in {7301, 7302}:
        raise ReceiptContractError("material training arm or seed is unauthorized")
    from experiments.psem_sortformer_adaptation_depth.protocol import (
        _legacy_authorize_conditional_arm_audit,
    )

    expected_conditional_authorization = _legacy_authorize_conditional_arm_audit(
        arm,
        staged_execution_receipt if arm == "TA-ALL-TEMPORAL" else None,
        staged_dev_results if arm == "TA-ALL-TEMPORAL" else (),
    )
    bound_overfit_arm = overfit_receipt.get("arms", {}).get(arm)
    bound_canary_arm = overfit_canary_receipts.get(arm)
    if (
        not isinstance(bound_overfit_arm, Mapping)
        or not isinstance(bound_canary_arm, Mapping)
        or bound_overfit_arm.get("conditional_arm_audit_authorization")
        != expected_conditional_authorization
        or bound_canary_arm.get("conditional_arm_audit_authorization")
        != expected_conditional_authorization
    ):
        raise ReceiptContractError("arm canaries are not bound to the current staged authorization")
    _validate_runtime_preflight(preflight_receipt)
    preflight_payload = {
        key: value for key, value in preflight_receipt.items() if key != "payload_sha256"
    }
    preflight_binding = preflight_payload.get("binding")
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if (
        preflight_receipt.get("mode") != "runtime"
        or preflight_receipt.get("ready_for_runtime_audit") is not True
        or preflight_receipt.get("payload_sha256") != canonical_sha256(preflight_payload)
        or not isinstance(preflight_binding, Mapping)
        or not _candidate_head_is_resume_compatible(
            preflight_binding.get("git_head"), current_head
        )
        or current_dirty
    ):
        raise ReceiptContractError("runtime preflight has not passed")
    from experiments.psem_sortformer_adaptation_depth.execution import candidate_code_identity

    current_identity = candidate_code_identity()
    split = build_data_split_receipt()
    manifest_sha = sampling_validation.get("manifest_sha256")
    if (
        sampling_validation.get("passed") is not True
        or sampling_validation.get("row_count") != 8 * 4096
        or manifest_sha != sha256_file(sampling_manifest_path)
        or sampling_validation.get("eval_source_count") != 0
        or sampling_validation.get("data_split_receipt_sha256") != canonical_sha256(split)
        or sampling_validation.get("split_manifest_sha256")
        != split["artifact_hashes"]["split_manifest"]
        or sampling_validation.get("source_manifest_sha256")
        != split["artifact_hashes"]["source_manifest"]
        or len(sampling_rows) != 8 * 4096
    ):
        raise ReceiptContractError(
            "shared sampling manifest is not bound to the frozen TRAIN split"
        )
    from experiments.psem_sortformer_adaptation_depth.training import (
        build_manifest_class_weight_receipt,
    )

    expected_class_weights = build_manifest_class_weight_receipt(
        sampling_rows, training_sessions, sampling_manifest_path
    )
    weight_payload = _bound_payload(class_weight_receipt, "train_class_weight_receipt")
    if (
        class_weight_receipt != expected_class_weights
        or weight_payload.get("sampling_manifest_sha256") != manifest_sha
        or weight_payload.get("split_roles") != [TRAIN_ROLE]
        or weight_payload.get("eval_source_count") != 0
        or weight_payload.get("row_count") != len(sampling_rows)
    ):
        raise ReceiptContractError("TRAIN class weights are not bound to the shared manifest")
    validated_lineage = validate_trainable_checkpoint_lineage(
        lineage_receipt,
        runtime_identity=runtime_identity,
        evaluator_contract=evaluator_contract,
    )
    lineage_payload = _bound_payload(validated_lineage, "trainable_checkpoint_lineage")
    if lineage_receipt != validated_lineage or lineage_payload.get("passed") is not True:
        raise ReceiptContractError("Q8-to-float lineage has not passed")
    require_registered_execution("lineage", lineage_receipt)
    expected_evaluator = evaluator_reconstruction_contract()
    graph = runtime_identity.get("model_graph")
    for kind, runtime_receipt in (
        ("gradient-canary", gradient_receipt),
        ("update-canary", update_receipt),
        ("timing-receipt", timing_receipt),
        ("parameter-inventory", parameter_inventory),
        ("model-graph", graph),
    ):
        if not isinstance(runtime_receipt, Mapping):
            raise ReceiptContractError("registered runtime evidence is absent")
        require_registered_execution(kind, runtime_receipt)
    if (
        evaluator_contract != expected_evaluator
        or lineage_payload.get("runtime_identity_sha256") != canonical_sha256(runtime_identity)
        or lineage_payload.get("evaluator_contract_sha256") != canonical_sha256(expected_evaluator)
        or not isinstance(graph, Mapping)
        or graph.get("passed") is not True
        or graph.get("runtime_canary_tap_paths")
        != {
            "final_temporal_hidden": "runtime_taps.final_temporal_hidden",
            "speaker_activity_logits": "runtime_taps.speaker_activity_logits",
            "psem_outputs": "psem_head",
        }
        or parameter_inventory.get("artifact_role") != "parameter_inventory"
        or parameter_inventory.get("arm") != arm
        or not parameter_inventory_runtime_passed(
            parameter_inventory,
            arm,
            model_graph_receipt=graph,
        )
        or not canary_bundle_runtime_passed(
            gradient_receipt,
            update_receipt,
            timing_receipt,
            arm,
            parameter_inventory_receipt=parameter_inventory,
            model_graph_receipt=graph,
        )
    ):
        raise ReceiptContractError(
            "model graph, evaluator, gradient, or update gate has not passed"
        )
    corpus_by_source: dict[str, str] = {}
    for row in sampling_rows:
        source_id = row.get("source_id")
        corpus = row.get("corpus")
        if (
            not isinstance(source_id, str)
            or corpus not in {"AMI", "AliMeeting"}
            or (source_id in corpus_by_source and corpus_by_source[source_id] != corpus)
        ):
            raise ReceiptContractError("sampling corpus identity is inconsistent")
        corpus_by_source[source_id] = corpus
    current_canaries = overfit_canary_receipts.get(arm)
    if (
        not isinstance(current_canaries, Mapping)
        or current_canaries.get("gradient_canary_receipt") != gradient_receipt
        or current_canaries.get("update_canary_receipt") != update_receipt
        or current_canaries.get("timing_receipt") != timing_receipt
        or current_canaries.get("parameter_inventory") != parameter_inventory
        or current_canaries.get("model_graph_receipt") != graph
    ):
        raise ReceiptContractError("material canaries differ from the overfit canaries")
    validated_overfit = _legacy_validate_overfit_canary(
        overfit_receipt,
        sampling_rows=list(sampling_rows),
        sampling_manifest_path=sampling_manifest_path,
        corpus_by_source=corpus_by_source,
        canary_receipts=overfit_canary_receipts,
    )
    overfit_payload = _bound_payload(validated_overfit, "overfit_canary")
    if (
        overfit_receipt != validated_overfit
        or overfit_payload.get("passed") is not True
        or arm not in overfit_payload.get("arms", {})
    ):
        raise ReceiptContractError("the arm has not passed the TRAIN-only overfit canary")
    _legacy_validate_staged_authorization(staged_execution_receipt, arm, seed, staged_dev_results)
    shared_input_identity = canonical_sha256(
        [
            {
                key: row.get(key)
                for key in (
                    "row_id",
                    "source_id",
                    "corpus",
                    "window_start_sample",
                    "window_end_sample",
                    "target_identity_sha256",
                    "augmentation_identity_sha256",
                    "state_reset_at_window_start",
                )
            }
            for row in sampling_rows
        ]
    )
    if any(
        row.get("split_role") != TRAIN_ROLE
        or not isinstance(row.get("row_id"), str)
        or not isinstance(row.get("source_id"), str)
        or row.get("corpus") not in {"AMI", "AliMeeting"}
        or not isinstance(row.get("window_start_sample"), int)
        or row.get("window_end_sample") != row.get("window_start_sample") + 480000
        or not isinstance(row.get("target_identity_sha256"), str)
        or not isinstance(row.get("augmentation_identity_sha256"), str)
        or row.get("state_reset_at_window_start") is not True
        for row in sampling_rows
    ):
        raise ReceiptContractError("shared training inputs contain invalid provenance")
    payload = {
        "schema_version": 1,
        "artifact_role": "material_training_authorization",
        "passed": True,
        "arm": arm,
        "seed": seed,
        "git_head": current_head,
        "candidate_code_identity_sha256": current_identity["payload_sha256"],
        "preflight_receipt_sha256": preflight_receipt["payload_sha256"],
        "sampling_manifest_sha256": manifest_sha,
        "shared_input_identity_sha256": shared_input_identity,
        "class_weight_receipt_sha256": class_weight_receipt["payload_sha256"],
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "model_graph_receipt_sha256": canonical_sha256(graph),
        "lineage_receipt_sha256": lineage_receipt["payload_sha256"],
        "evaluator_contract_sha256": canonical_sha256(expected_evaluator),
        "parameter_inventory_sha256": canonical_sha256(parameter_inventory),
        "gradient_receipt_sha256": canonical_sha256(gradient_receipt),
        "update_receipt_sha256": canonical_sha256(update_receipt),
        "timing_receipt_sha256": canonical_sha256(timing_receipt),
        "overfit_receipt_sha256": overfit_receipt["payload_sha256"],
        "overfit_canary_receipts_sha256": canonical_sha256(overfit_canary_receipts),
        "staged_execution_receipt_sha256": staged_execution_receipt["payload_sha256"],
        "dev_source_ids_sha256": canonical_sha256(split["source_ids_by_role"][DEV_ROLE]),
        "authorized_checkpoint_path": next(
            row["observed"]
            for row in preflight_receipt["checks"]
            if row["id"] == "runtime.checkpoint_path"
        ),
        "authorized_corpus_root": next(
            row["observed"]
            for row in preflight_receipt["checks"]
            if row["id"] == "runtime.corpus_root"
        ),
        "authorized_reference_root": next(
            row["observed"]
            for row in preflight_receipt["checks"]
            if row["id"] == "runtime.reference_root"
        ),
        "authorized_output_root": next(
            row["observed"]
            for row in preflight_receipt["checks"]
            if row["id"] == "runtime.output_root"
        ),
        "authorized_protocol_registry_root": next(
            row["observed"]
            for row in preflight_receipt["checks"]
            if row["id"] == "runtime.protocol_registry_root"
        ),
        "eval_source_count": 0,
        "validation_bundle": {
            "preflight_receipt": dict(preflight_receipt),
            "sampling_validation": dict(sampling_validation),
            "sampling_manifest_path": str(sampling_manifest_path.resolve()),
            "class_weight_receipt": dict(class_weight_receipt),
            "lineage_receipt": dict(lineage_receipt),
            "runtime_identity": dict(runtime_identity),
            "evaluator_contract": dict(evaluator_contract),
            "parameter_inventory": dict(parameter_inventory),
            "gradient_receipt": dict(gradient_receipt),
            "update_receipt": dict(update_receipt),
            "timing_receipt": dict(timing_receipt),
            "overfit_receipt": dict(overfit_receipt),
            "overfit_canary_receipts": {
                key: dict(value) for key, value in overfit_canary_receipts.items()
            },
            "staged_execution_receipt": dict(staged_execution_receipt),
            "staged_dev_results": [dict(value) for value in staged_dev_results],
        },
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def _legacy_revalidate_material_training_gate(
    gate: Mapping[str, Any],
    *,
    sampling_manifest_path: Path,
    sampling_rows: Sequence[Mapping[str, Any]],
    training_sessions: Mapping[str, Any],
    class_weight_receipt: Mapping[str, Any],
    checkpoint_path: Path,
    corpus_root: Path,
    reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    bundle = gate.get("validation_bundle")
    if not isinstance(bundle, Mapping):
        raise ReceiptContractError("material gate lacks its reproducible validation bundle")
    if (
        bundle.get("sampling_manifest_path") != str(sampling_manifest_path.resolve())
        or gate.get("authorized_checkpoint_path") != str(checkpoint_path.resolve())
        or gate.get("authorized_corpus_root") != str(corpus_root.resolve())
        or gate.get("authorized_reference_root") != str(reference_root.resolve())
        or gate.get("authorized_output_root") != str(output_root.resolve())
        or bundle.get("class_weight_receipt") != class_weight_receipt
    ):
        raise ReceiptContractError(
            "material execution paths or class weights differ from preflight"
        )
    recomputed = validate_material_training_gate(
        arm=str(gate.get("arm")),
        seed=int(gate.get("seed")),
        preflight_receipt=bundle["preflight_receipt"],
        sampling_validation=bundle["sampling_validation"],
        sampling_manifest_path=sampling_manifest_path,
        sampling_rows=sampling_rows,
        training_sessions=training_sessions,
        class_weight_receipt=class_weight_receipt,
        lineage_receipt=bundle["lineage_receipt"],
        runtime_identity=bundle["runtime_identity"],
        evaluator_contract=bundle["evaluator_contract"],
        parameter_inventory=bundle["parameter_inventory"],
        gradient_receipt=bundle["gradient_receipt"],
        update_receipt=bundle["update_receipt"],
        timing_receipt=bundle["timing_receipt"],
        overfit_receipt=bundle["overfit_receipt"],
        overfit_canary_receipts=bundle["overfit_canary_receipts"],
        staged_execution_receipt=bundle["staged_execution_receipt"],
        staged_dev_results=bundle["staged_dev_results"],
    )
    if recomputed != dict(gate):
        raise ReceiptContractError(
            "material training gate differs from a current exact revalidation"
        )
    return dict(gate)


def _legacy_revalidate_material_training_gate_from_bundle(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = gate.get("validation_bundle")
    if not isinstance(bundle, Mapping):
        raise ReceiptContractError("material gate lacks its reproducible validation bundle")
    required_paths = {
        "sampling_manifest_path": bundle.get("sampling_manifest_path"),
        "checkpoint_path": gate.get("authorized_checkpoint_path"),
        "corpus_root": gate.get("authorized_corpus_root"),
        "reference_root": gate.get("authorized_reference_root"),
        "output_root": gate.get("authorized_output_root"),
    }
    if any(not isinstance(value, str) or not value for value in required_paths.values()):
        raise ReceiptContractError("material gate execution paths are incomplete")
    class_weights = bundle.get("class_weight_receipt")
    if not isinstance(class_weights, Mapping):
        raise ReceiptContractError("material gate class weights are absent")
    from experiments.psem_sortformer_adaptation_depth.sampling import (
        load_sampling_rows,
        load_training_sessions,
    )

    sampling_manifest_path = Path(required_paths["sampling_manifest_path"])
    corpus_root = Path(required_paths["corpus_root"])
    reference_root = Path(required_paths["reference_root"])
    return revalidate_material_training_gate(
        gate,
        sampling_manifest_path=sampling_manifest_path,
        sampling_rows=load_sampling_rows(sampling_manifest_path),
        training_sessions=load_training_sessions(corpus_root, reference_root),
        class_weight_receipt=class_weights,
        checkpoint_path=Path(required_paths["checkpoint_path"]),
        corpus_root=corpus_root,
        reference_root=reference_root,
        output_root=Path(required_paths["output_root"]),
    )


def validate_cost_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _bound_payload(value, "psem_sortformer_cost_receipt")
    required = (
        "hourly_price_usd",
        "hourly_price_source",
        "actual_gpu_seconds",
        "projected_remaining_gpu_seconds",
        "actual_cost_usd",
        "projected_remaining_cost_usd",
        "projected_total_gpu_seconds",
        "projected_total_cost_usd",
        "target_total_usd",
        "hard_stop_usd",
        "target_is_informational",
        "command",
    )
    if any(key not in payload for key in required):
        raise ReceiptContractError("cost receipt fields are incomplete")
    numeric = (
        "hourly_price_usd",
        "actual_gpu_seconds",
        "projected_remaining_gpu_seconds",
        "actual_cost_usd",
        "projected_remaining_cost_usd",
        "projected_total_gpu_seconds",
        "projected_total_cost_usd",
        "target_total_usd",
        "hard_stop_usd",
    )
    if (
        not isinstance(payload["hourly_price_source"], str)
        or not payload["hourly_price_source"].strip()
        or not isinstance(payload["command"], str)
        or not payload["command"].strip()
        or any(not _finite_number(payload[key]) or float(payload[key]) < 0 for key in numeric)
        or float(payload["hourly_price_usd"]) <= 0
        or float(payload["target_total_usd"]) != 15.0
        or float(payload["hard_stop_usd"]) != 30.0
        or payload["target_is_informational"] is not True
    ):
        raise ReceiptContractError("cost receipt values are invalid")
    price = float(payload["hourly_price_usd"])
    actual_seconds = float(payload["actual_gpu_seconds"])
    remaining_seconds = float(payload["projected_remaining_gpu_seconds"])
    expected_actual = actual_seconds * price / 3600.0
    expected_remaining = remaining_seconds * price / 3600.0
    expected_total = (actual_seconds + remaining_seconds) * price / 3600.0
    if (
        not math.isclose(
            float(payload["actual_cost_usd"]), expected_actual, rel_tol=0, abs_tol=1e-9
        )
        or not math.isclose(
            float(payload["projected_remaining_cost_usd"]),
            expected_remaining,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(payload["projected_total_gpu_seconds"]),
            actual_seconds + remaining_seconds,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(payload["projected_total_cost_usd"]), expected_total, rel_tol=0, abs_tol=1e-9
        )
        or float(payload["projected_total_cost_usd"]) > 30.0
    ):
        raise ReceiptContractError("cost receipt formula or USD-30 hard stop failed")
    return dict(value)


def _require_bound_mapping(value: Mapping[str, Any], role: str) -> dict[str, Any]:
    return _bound_payload(value, role)


def _validate_short_smoke_provenance(
    short_smoke_receipt: Mapping[str, Any],
    *,
    arm: str,
    manifest_sha256: str,
    class_weight_receipt: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    ta_open_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = _bound_payload(short_smoke_receipt, "short_smoke_metrics")
    if (
        payload.get("arm") != arm
        or payload.get("seed") != 7301
        or payload.get("sampling_manifest_sha256") != manifest_sha256
        or payload.get("class_weight_receipt_sha256") != class_weight_receipt.get("payload_sha256")
        or payload.get("runtime_identity") != runtime_identity
        or payload.get("runtime_identity_sha256") != canonical_sha256(runtime_identity)
        or payload.get("base_checkpoint_sha256") != runtime_identity.get("checkpoint_sha256")
        or payload.get("dependency_lock_sha256") != runtime_identity.get("dependency_lock_sha256")
        or payload.get("optimizer_steps") != 32
        or payload.get("consumed_row_count") != 512
        or payload.get("finite_forward_backward_update") is not True
        or payload.get("frozen_parameters_unchanged") is not True
        or payload.get("weights_discarded") is not True
        or payload.get("ta_open_authorization_sha256")
        != (
            ta_open_authorization.get("payload_sha256")
            if ta_open_authorization is not None
            else None
        )
        or not _finite_number(payload.get("first_eight_mean_total_loss"))
        or not _finite_number(payload.get("last_eight_mean_total_loss"))
    ):
        raise ReceiptContractError("short smoke metrics do not pass the lean gate")
    return payload


def _validate_ta_authorization_provenance(
    ta_open_authorization: Mapping[str, Any],
    *,
    cost_receipt: Mapping[str, Any],
    staged_execution_receipt: Mapping[str, Any],
    staged_dev_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.protocol import (
        validate_dev_result,
        validate_operator_dev_decision,
        validate_staged_execution_state,
    )

    validate_staged_execution_state(staged_execution_receipt, staged_dev_results)
    staged_payload = _bound_payload(staged_execution_receipt, "staged_execution_state")
    validated_results = [validate_dev_result(value) for value in staged_dev_results]
    result_sha256s = dict(
        sorted(
            (
                f"{result['arm']}:{result.get('seed')}",
                result["payload_sha256"],
            )
            for result in validated_results
        )
    )
    expected_runs = [
        ("F0-FROZEN-FLOAT", None),
        ("H-HEAD", 7301),
        ("T2-TOP", 7301),
    ]
    payload = _bound_payload(ta_open_authorization, "psem_sortformer_open_ta_authorization")
    decision = payload.get("operator_decision")
    validated_decision = (
        validate_operator_dev_decision(decision) if isinstance(decision, Mapping) else None
    )
    decision_payload = (
        _bound_payload(validated_decision, "psem_sortformer_operator_dev_decision")
        if isinstance(validated_decision, Mapping)
        else None
    )
    if (
        decision_payload is None
        or [(row.get("arm"), row.get("seed")) for row in staged_payload.get("completed_runs", [])]
        != expected_runs
        or [(result.get("arm"), result.get("seed")) for result in validated_results]
        != expected_runs
        or staged_payload.get("eval_open_count") != 0
        or staged_payload.get("eval_used_for_development") is not False
        or payload.get("arm") != "TA-ALL-TEMPORAL"
        or payload.get("seed") != 7301
        or payload.get("cost_receipt_sha256") != cost_receipt.get("payload_sha256")
        or payload.get("operator_decision") != validated_decision
        or payload.get("operator_decision_sha256") != validated_decision.get("payload_sha256")
        or payload.get("staged_execution_receipt_sha256")
        != staged_execution_receipt.get("payload_sha256")
        or payload.get("staged_dev_result_sha256s") != result_sha256s
        or decision_payload.get("decision") != "open_ta"
        or decision_payload.get("selected_arm") != "TA-ALL-TEMPORAL"
        or decision_payload.get("available_dev_result_sha256s") != result_sha256s
    ):
        raise ReceiptContractError(
            "TA authorization differs from the current cost or staged DEV evidence"
        )
    return payload


def validate_material_training_gate(
    *,
    arm: str,
    seed: int,
    preflight_receipt: Mapping[str, Any],
    sampling_validation: Mapping[str, Any],
    sampling_manifest_path: Path,
    sampling_rows: Sequence[Mapping[str, Any]],
    training_sessions: Mapping[str, Any],
    class_weight_receipt: Mapping[str, Any],
    lineage_receipt: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    evaluator_contract: Mapping[str, Any],
    staged_execution_receipt: Mapping[str, Any],
    staged_dev_results: Sequence[Mapping[str, Any]],
    short_smoke_receipt: Mapping[str, Any],
    cost_receipt: Mapping[str, Any],
    ta_open_authorization: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if arm not in {"H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL"} or seed != 7301:
        raise ReceiptContractError("material training arm or seed is unauthorized")
    preflight_payload = {
        key: value for key, value in preflight_receipt.items() if key != "payload_sha256"
    }
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if (
        preflight_receipt.get("mode") != "runtime"
        or preflight_receipt.get("ready_for_runtime_audit") is not True
        or preflight_receipt.get("payload_sha256") != canonical_sha256(preflight_payload)
        or not _candidate_head_is_resume_compatible(
            preflight_payload.get("binding", {}).get("git_head"), current_head
        )
        or current_dirty
    ):
        raise ReceiptContractError("runtime preflight is not bound to a clean current candidate")
    from experiments.psem_sortformer_adaptation_depth.execution import (
        candidate_code_identity,
    )

    current_identity = candidate_code_identity()
    split = build_data_split_receipt()
    manifest_sha = sampling_validation.get("manifest_sha256")
    if (
        sampling_validation.get("passed") is not True
        or sampling_validation.get("row_count") != 4096
        or len(sampling_rows) != 4096
        or manifest_sha != sha256_file(sampling_manifest_path)
        or sampling_validation.get("eval_source_count") != 0
        or any(
            row.get("split_role") != TRAIN_ROLE
            or row.get("epoch") != 1
            or row.get("epoch_index") != index
            or row.get("row_id") != f"epoch-01-window-{index:04d}"
            for index, row in enumerate(sampling_rows)
        )
    ):
        raise ReceiptContractError(
            "shared sampling manifest is not the exact epoch-1 TRAIN manifest"
        )
    weight_payload = _bound_payload(class_weight_receipt, "train_class_weight_receipt")
    if (
        weight_payload.get("sampling_manifest_sha256") != manifest_sha
        or weight_payload.get("split_roles") != [TRAIN_ROLE]
        or weight_payload.get("eval_source_count") != 0
        or weight_payload.get("row_count") != 4096
    ):
        raise ReceiptContractError("TRAIN class weights are not bound to the one-epoch manifest")
    lineage_payload = _bound_payload(lineage_receipt, "trainable_checkpoint_lineage")
    expected_evaluator = evaluator_contract
    graph = runtime_identity.get("model_graph")
    if (
        lineage_payload.get("passed") is not True
        or not isinstance(graph, Mapping)
    ):
        raise ReceiptContractError("lineage or runtime identity is absent")
    _validate_short_smoke_provenance(
        short_smoke_receipt,
        arm=arm,
        manifest_sha256=manifest_sha,
        class_weight_receipt=class_weight_receipt,
        runtime_identity=runtime_identity,
        ta_open_authorization=ta_open_authorization,
    )
    validate_cost_receipt(cost_receipt)
    from experiments.psem_sortformer_adaptation_depth.protocol import (
        validate_dev_result,
        validate_staged_execution_state,
    )

    validate_staged_execution_state(staged_execution_receipt, staged_dev_results)
    staged_payload = _bound_payload(staged_execution_receipt, "staged_execution_state")
    validated_staged_results = [validate_dev_result(value) for value in staged_dev_results]
    completed_ids = [
        (row.get("arm"), row.get("seed")) for row in staged_payload.get("completed_runs", [])
    ]
    expected_completed = {
        "H-HEAD": [("F0-FROZEN-FLOAT", None)],
        "T2-TOP": [("F0-FROZEN-FLOAT", None), ("H-HEAD", 7301)],
        "TA-ALL-TEMPORAL": [
            ("F0-FROZEN-FLOAT", None),
            ("H-HEAD", 7301),
            ("T2-TOP", 7301),
        ],
    }[arm]
    if (
        completed_ids != expected_completed
        or [(value.get("arm"), value.get("seed")) for value in validated_staged_results]
        != expected_completed
        or staged_payload.get("eval_open_count") != 0
        or staged_payload.get("eval_used_for_development") is not False
    ):
        raise ReceiptContractError("staged DEV state does not authorize the next lean arm")
    if arm == "TA-ALL-TEMPORAL":
        if not isinstance(ta_open_authorization, Mapping):
            raise ReceiptContractError("TA training requires an explicit open_ta authorization")
        _validate_ta_authorization_provenance(
            ta_open_authorization,
            cost_receipt=cost_receipt,
            staged_execution_receipt=staged_execution_receipt,
            staged_dev_results=staged_dev_results,
        )
    elif ta_open_authorization is not None:
        raise ReceiptContractError("non-TA training cannot carry a TA authorization")
    observed_paths = {
        row.get("id"): row.get("observed")
        for row in preflight_receipt.get("checks", [])
        if isinstance(row, Mapping)
    }
    required_paths = {
        "authorized_checkpoint_path": observed_paths.get("runtime.checkpoint_path"),
        "authorized_corpus_root": observed_paths.get("runtime.corpus_root"),
        "authorized_reference_root": observed_paths.get("runtime.reference_root"),
        "authorized_output_root": observed_paths.get("runtime.output_root"),
        "authorized_protocol_registry_root": observed_paths.get("runtime.protocol_registry_root"),
    }
    if (
        any(not isinstance(value, str) or not value for value in required_paths.values())
        or required_paths["authorized_output_root"] != staged_payload.get("experiment_output_root")
        or required_paths["authorized_protocol_registry_root"]
        != staged_payload.get("protocol_registry_root")
    ):
        raise ReceiptContractError("runtime preflight paths differ from staged DEV state")
    shared_identity = canonical_sha256(
        [
            {
                key: row.get(key)
                for key in (
                    "row_id",
                    "source_id",
                    "corpus",
                    "window_start_sample",
                    "window_end_sample",
                    "target_identity_sha256",
                    "augmentation_identity_sha256",
                    "state_reset_at_window_start",
                )
            }
            for row in sampling_rows
        ]
    )
    payload = {
        "schema_version": 1,
        "artifact_role": "material_training_authorization",
        "passed": True,
        "arm": arm,
        "seed": seed,
        "git_head": current_head,
        "candidate_code_identity_sha256": current_identity["payload_sha256"],
        "preflight_receipt_sha256": preflight_receipt["payload_sha256"],
        "sampling_manifest_sha256": manifest_sha,
        "class_weight_receipt_sha256": class_weight_receipt["payload_sha256"],
        "short_smoke_receipt_sha256": short_smoke_receipt["payload_sha256"],
        "cost_receipt_sha256": cost_receipt["payload_sha256"],
        "shared_input_identity_sha256": shared_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "model_graph_receipt_sha256": canonical_sha256(graph),
        "lineage_receipt_sha256": lineage_receipt["payload_sha256"],
        "evaluator_contract_sha256": canonical_sha256(expected_evaluator),
        "staged_execution_receipt_sha256": staged_execution_receipt["payload_sha256"],
        "ta_open_authorization_sha256": (
            ta_open_authorization.get("payload_sha256")
            if ta_open_authorization is not None
            else None
        ),
        "dev_source_ids_sha256": canonical_sha256(split["source_ids_by_role"][DEV_ROLE]),
        **required_paths,
        "eval_source_count": 0,
        "validation_bundle": {
            "sampling_manifest_path": str(sampling_manifest_path.resolve()),
            "class_weight_receipt": dict(class_weight_receipt),
            "preflight_receipt": dict(preflight_receipt),
            "sampling_validation": dict(sampling_validation),
            "lineage_receipt": dict(lineage_receipt),
            "runtime_identity": dict(runtime_identity),
            "evaluator_contract": dict(evaluator_contract),
            "short_smoke_receipt": dict(short_smoke_receipt),
            "cost_receipt": dict(cost_receipt),
            "staged_execution_receipt": dict(staged_execution_receipt),
            "staged_dev_results": [dict(value) for value in staged_dev_results],
            "ta_open_authorization": (
                dict(ta_open_authorization) if ta_open_authorization is not None else None
            ),
        },
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def revalidate_material_training_gate(
    gate: Mapping[str, Any],
    *,
    sampling_manifest_path: Path,
    sampling_rows: Sequence[Mapping[str, Any]],
    training_sessions: Mapping[str, Any],
    class_weight_receipt: Mapping[str, Any],
    checkpoint_path: Path,
    corpus_root: Path,
    reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    bundle = gate.get("validation_bundle")
    if not isinstance(bundle, Mapping):
        raise ReceiptContractError("material gate lacks its lean validation bundle")
    if (
        bundle.get("sampling_manifest_path") != str(sampling_manifest_path.resolve())
        or bundle.get("class_weight_receipt") != class_weight_receipt
        or gate.get("authorized_checkpoint_path") != str(checkpoint_path.resolve())
        or gate.get("authorized_corpus_root") != str(corpus_root.resolve())
        or gate.get("authorized_reference_root") != str(reference_root.resolve())
        or gate.get("authorized_output_root") != str(output_root.resolve())
    ):
        raise ReceiptContractError(
            "material execution paths or class weights differ from preflight"
        )
    recomputed = validate_material_training_gate(
        arm=str(gate.get("arm")),
        seed=int(gate.get("seed")),
        preflight_receipt=bundle["preflight_receipt"],
        sampling_validation=bundle["sampling_validation"],
        sampling_manifest_path=sampling_manifest_path,
        sampling_rows=sampling_rows,
        training_sessions=training_sessions,
        class_weight_receipt=class_weight_receipt,
        lineage_receipt=bundle["lineage_receipt"],
        runtime_identity=bundle["runtime_identity"],
        evaluator_contract=bundle["evaluator_contract"],
        parameter_inventory=bundle["parameter_inventory"],
        gradient_receipt=bundle["gradient_receipt"],
        update_receipt=bundle["update_receipt"],
        timing_receipt=bundle["timing_receipt"],
        staged_execution_receipt=bundle["staged_execution_receipt"],
        staged_dev_results=bundle.get("staged_dev_results", []),
        short_smoke_receipt=bundle["short_smoke_receipt"],
        cost_receipt=bundle["cost_receipt"],
        ta_open_authorization=bundle.get("ta_open_authorization"),
    )
    if recomputed != dict(gate):
        raise ReceiptContractError(
            "material training gate differs from a current exact revalidation"
        )
    return dict(gate)


def revalidate_material_training_gate_from_bundle(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = gate.get("validation_bundle")
    if not isinstance(bundle, Mapping):
        raise ReceiptContractError("material gate lacks its lean validation bundle")
    from experiments.psem_sortformer_adaptation_depth.sampling import (
        load_sampling_rows,
        load_training_sessions,
    )

    manifest = Path(str(bundle["sampling_manifest_path"]))
    required_paths = {
        "checkpoint_path": gate.get("authorized_checkpoint_path"),
        "corpus_root": gate.get("authorized_corpus_root"),
        "reference_root": gate.get("authorized_reference_root"),
        "output_root": gate.get("authorized_output_root"),
    }
    if any(not isinstance(value, str) or not value for value in required_paths.values()):
        raise ReceiptContractError("material gate execution paths are incomplete")
    corpus_root = Path(required_paths["corpus_root"])
    reference_root = Path(required_paths["reference_root"])
    return revalidate_material_training_gate(
        gate,
        sampling_manifest_path=manifest,
        sampling_rows=load_sampling_rows(manifest),
        training_sessions=load_training_sessions(corpus_root, reference_root),
        class_weight_receipt=bundle["class_weight_receipt"],
        checkpoint_path=Path(required_paths["checkpoint_path"]),
        corpus_root=corpus_root,
        reference_root=reference_root,
        output_root=Path(required_paths["output_root"]),
    )
