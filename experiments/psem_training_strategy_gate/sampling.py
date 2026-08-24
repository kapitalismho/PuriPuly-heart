from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torchaudio

from experiments.psem_training_strategy_gate.augmentation import augmentation_decision
from experiments.psem_training_strategy_gate.data.label_contract import LabelResult
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    PinnedReferenceCheckout,
    normalize_reference_session,
    open_reference_checkout,
)
from experiments.psem_training_strategy_gate.losses import LossWeights
from experiments.psem_training_strategy_gate.preflight import canonical_sha256
from experiments.psem_training_strategy_gate.targets import (
    WINDOW_SAMPLES,
    WindowTargets,
    _masked_transition_centers,
    build_window_targets,
    nearest_grid_sample,
    valid_center_samples,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_ROOT / "data" / "v2"
SOURCE_MANIFEST_PATH = DATA_DIR / "source_manifest.jsonl"
SPLIT_MANIFEST_PATH = DATA_DIR / "split_manifest.json"
TOPOLOGY_MANIFEST_PATH = DATA_DIR / "topology_manifest.jsonl"
TRAIN_ROLE = "PSEM-STRATEGY-TRAIN"
DEV_ROLE = "PSEM-STRATEGY-DEV"
EVAL_ROLE = "PSEM-STRATEGY-EVAL"
MAXIMUM_EPOCHS = 20
WINDOWS_PER_EPOCH = 4096
OFFICIAL_EFFECTIVE_BATCH_SIZE = 4
POSITIVE_FAMILIES = (
    "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff",
    "overlap_takeover",
)
HARD_NEGATIVE_FAMILIES = (
    "stable_singleton_continuation",
    "same_speaker_silence_gap_resume",
    "overlap_return",
    "overlap_continuation",
    "silence_continuation",
)
POSITIVE_TOPOLOGY_FAMILY = {
    "clean_direct_different_speaker_handoff": "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff": "silence_gap_different_speaker_handoff",
    "micro_gap_different_speaker_handoff": "silence_gap_different_speaker_handoff",
    "overlap_takeover": "overlap_takeover",
    "micro_overlap_takeover": "overlap_takeover",
    "overlap_gap_takeover": "overlap_takeover",
}
HARD_NEGATIVE_TOPOLOGY_FAMILY = {
    "same_speaker_silence_gap_resume": "same_speaker_silence_gap_resume",
    "micro_gap_same_speaker_resume": "same_speaker_silence_gap_resume",
    "overlap_return": "overlap_return",
    "micro_overlap_return": "overlap_return",
    "overlap_gap_return": "overlap_return",
}
SAMPLING_COUNTS = {
    "handoff_positive": WINDOWS_PER_EPOCH // 4,
    "topology_hard_negative": WINDOWS_PER_EPOCH // 4,
    "source_time_uniform": WINDOWS_PER_EPOCH // 2,
}


class SamplingContractError(RuntimeError):
    pass


@dataclass(slots=True)
class BatchValidityAccumulator:
    batch_size: int
    rows_in_batch: int = 0
    current: dict[str, int] = field(
        default_factory=lambda: {"handoff": 0, "state": 0, "relation": 0}
    )
    minimum: dict[str, int] = field(default_factory=dict)

    def add(self, target: WindowTargets) -> None:
        self.current["handoff"] += int(target.handoff_mask)
        self.current["state"] += sum(int(value) for value in target.state_mask)
        self.current["relation"] += len(target.relation_pairs)
        self.rows_in_batch += 1
        if self.rows_in_batch == self.batch_size:
            for key, value in self.current.items():
                self.minimum[key] = min(self.minimum.get(key, value), value)
            self.rows_in_batch = 0
            self.current = {"handoff": 0, "state": 0, "relation": 0}

    def finish(self) -> dict[str, int]:
        if self.batch_size <= 0 or self.rows_in_batch or set(self.minimum) != set(self.current):
            raise SamplingContractError("sampling rows do not form complete official batches")
        return dict(sorted(self.minimum.items()))


@dataclass(frozen=True, slots=True)
class CandidateCenter:
    source_id: str
    boundary_sample: int
    family: str

    @property
    def center_id(self) -> str:
        return f"{self.source_id}:{self.boundary_sample}"


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    source_id: str
    role: str
    audio_ref: str
    waveform_sha256: str
    labels: LabelResult


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise SamplingContractError(f"JSONL artifact is invalid: {path}")
    return values


def role_by_source() -> dict[str, str]:
    split = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    components = split["assignments"]["components"]
    result: dict[str, str] = {}
    for component in components:
        for source_id in component["source_ids"]:
            if source_id in result:
                raise SamplingContractError("split manifest assigns a source more than once")
            result[source_id] = component["role"]
    source_ids = {str(row["source_id"]) for row in _jsonl(SOURCE_MANIFEST_PATH)}
    if set(result) != source_ids:
        raise SamplingContractError("split manifest does not assign every frozen source")
    return result


def load_runtime_sessions(
    corpus_root: Path,
    reference_root: Path,
    *,
    roles: Sequence[str] = (TRAIN_ROLE,),
) -> dict[str, RuntimeSession]:
    if EVAL_ROLE in roles:
        raise SamplingContractError("EVAL must remain closed during sampling and audit preparation")
    source_rows = {str(row["source_id"]): row for row in _jsonl(SOURCE_MANIFEST_PATH)}
    topology_rows = {str(row["source_id"]): row for row in _jsonl(TOPOLOGY_MANIFEST_PATH)}
    assignments = role_by_source()
    checkout: PinnedReferenceCheckout = open_reference_checkout(reference_root)
    sessions: dict[str, RuntimeSession] = {}
    for source_id in sorted(source_rows):
        role = assignments[source_id]
        if role not in roles:
            continue
        source_row = source_rows[source_id]
        normalized = normalize_reference_session(source_row, corpus_root, checkout)
        labels = normalized.labels
        expected = topology_rows[source_id]
        if expected["label_result_sha256"] != canonical_sha256(labels.to_dict()):
            raise SamplingContractError("runtime labels differ from the frozen topology artifact")
        sessions[source_id] = RuntimeSession(
            source_id=source_id,
            role=role,
            audio_ref=str(source_row["audio_ref"]),
            waveform_sha256=str(source_row["waveform_sha256"]),
            labels=labels,
        )
    if not sessions or any(session.role == EVAL_ROLE for session in sessions.values()):
        raise SamplingContractError("runtime session selection violates frozen data roles")
    return sessions


def _transition_center(labels: LabelResult, transition: Mapping[str, Any]) -> int | None:
    sample = transition.get("handoff_source_sample")
    if isinstance(sample, int):
        return nearest_grid_sample(sample)
    current_index = transition.get("to_interval_index")
    if isinstance(current_index, int):
        return nearest_grid_sample(labels.intervals[current_index].start_sample)
    return None


def candidate_pools(
    sessions: Mapping[str, RuntimeSession],
) -> dict[str, tuple[CandidateCenter, ...]]:
    pools: dict[str, dict[str, CandidateCenter]] = {
        family: {} for family in (*POSITIVE_FAMILIES, *HARD_NEGATIVE_FAMILIES)
    }
    uniform: dict[str, CandidateCenter] = {}
    excluded_centers: dict[str, set[int]] = {}
    for source_id, session in sorted(sessions.items()):
        if session.role != TRAIN_ROLE:
            continue
        labels = session.labels
        centers = valid_center_samples(
            labels.intervals[0].start_sample,
            labels.intervals[-1].end_sample,
        )
        excluded = _masked_transition_centers(labels)
        for transition in labels.transitions:
            center = _transition_center(labels, transition)
            if center is None or center not in centers:
                continue
            if transition.get("mask_state") == "masked":
                excluded.add(center)
            topology = transition.get("primary_topology")
            valid = transition.get("mask_state") == "valid"
            handoff_target = transition.get("handoff_confirmed")
            if valid and handoff_target == 1:
                excluded.add(center)
                family = POSITIVE_TOPOLOGY_FAMILY.get(str(topology))
                if family is not None:
                    candidate = CandidateCenter(source_id, center, family)
                    pools[family][candidate.center_id] = candidate
            if valid and handoff_target == 0:
                family = HARD_NEGATIVE_TOPOLOGY_FAMILY.get(str(topology))
                if family is not None:
                    candidate = CandidateCenter(source_id, center, family)
                    pools[family][candidate.center_id] = candidate
                    excluded.add(center)
        excluded_centers[source_id] = excluded
        interval_index = 0
        for center in centers:
            while center >= labels.intervals[interval_index].end_sample:
                interval_index += 1
            candidate = CandidateCenter(source_id, center, "source_time_uniform")
            uniform[candidate.center_id] = candidate
            if center in excluded:
                continue
            activity = labels.activity_labels[interval_index]
            interval = labels.intervals[interval_index]
            if (
                activity["mask_state"] != "valid"
                or interval.ambiguous
                or not interval.speaker_identity_known
                or len(interval.active_speakers) >= 3
                or interval.handoff_relation_mask_classes
            ):
                continue
            state = activity["state"]
            family = {
                "singleton": "stable_singleton_continuation",
                "overlap": "overlap_continuation",
                "silence": "silence_continuation",
            }.get(state)
            if family is not None:
                selected = CandidateCenter(source_id, center, family)
                pools[family][selected.center_id] = selected
    result = {
        family: tuple(values[key] for key in sorted(values)) for family, values in pools.items()
    }
    result["source_time_uniform"] = tuple(uniform[key] for key in sorted(uniform))
    if any(not result[family] for family in (*POSITIVE_FAMILIES, *HARD_NEGATIVE_FAMILIES)):
        raise SamplingContractError("one or more mandatory sampling pools are empty")
    return result


def _stable_order(
    values: Sequence[CandidateCenter],
    *,
    family: str,
) -> tuple[CandidateCenter, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"psem-sampling-v1\0{family}\0{value.center_id}".encode()
            ).digest(),
        )
    )


def _quota(total: int, families: Sequence[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(families))
    return {family: base + int(index < remainder) for index, family in enumerate(families)}


def _take(
    values: Sequence[CandidateCenter],
    *,
    count: int,
    offset: int,
) -> list[CandidateCenter]:
    if not values:
        raise SamplingContractError("cannot sample from an empty pool")
    return [values[(offset + index) % len(values)] for index in range(count)]


def epoch_plan(
    pools: Mapping[str, Sequence[CandidateCenter]],
    epoch: int,
) -> tuple[tuple[str, CandidateCenter], ...]:
    if epoch < 1 or epoch > MAXIMUM_EPOCHS:
        raise SamplingContractError("epoch lies outside the frozen optimization contract")
    selected: list[tuple[str, CandidateCenter]] = []
    positive_quota = _quota(SAMPLING_COUNTS["handoff_positive"], POSITIVE_FAMILIES)
    hard_quota = _quota(SAMPLING_COUNTS["topology_hard_negative"], HARD_NEGATIVE_FAMILIES)
    for sampling_role, families, quota in (
        ("handoff_positive", POSITIVE_FAMILIES, positive_quota),
        ("topology_hard_negative", HARD_NEGATIVE_FAMILIES, hard_quota),
    ):
        for family in families:
            values = _stable_order(pools[family], family=family)
            count = quota[family]
            selected.extend(
                (sampling_role, value)
                for value in _take(values, count=count, offset=(epoch - 1) * count)
            )
    uniform = _stable_order(pools["source_time_uniform"], family="source_time_uniform")
    uniform_count = SAMPLING_COUNTS["source_time_uniform"]
    selected.extend(
        ("source_time_uniform", value)
        for value in _take(
            uniform,
            count=uniform_count,
            offset=(epoch - 1) * uniform_count,
        )
    )
    selected.sort(
        key=lambda item: hashlib.sha256(
            f"psem-epoch-order-v1\0{epoch}\0{item[0]}\0{item[1].center_id}".encode()
        ).digest()
    )
    if len(selected) != WINDOWS_PER_EPOCH:
        raise SamplingContractError("epoch sample count differs from the frozen mixture")
    return tuple(selected)


def _class_weights(counts: Counter[int], classes: int) -> tuple[float, ...]:
    if any(counts[index] <= 0 for index in range(classes)):
        raise SamplingContractError("loss-class weighting requires every class")
    total = sum(counts.values())
    values = [total / (classes * counts[index]) for index in range(classes)]
    mean = sum(values) / classes
    return tuple(value / mean for value in values)


def materialize_sampling_manifest(
    sessions: Mapping[str, RuntimeSession],
    output_path: Path,
) -> dict[str, Any]:
    pools = candidate_pools(sessions)
    waveform_by_source = {
        source_id: session.waveform_sha256 for source_id, session in sessions.items()
    }
    target_cache: dict[str, WindowTargets] = {}
    handoff_counts: Counter[int] = Counter()
    state_counts: Counter[int] = Counter()
    relation_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    batch_validity = BatchValidityAccumulator(OFFICIAL_EFFECTIVE_BATCH_SIZE)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(1, MAXIMUM_EPOCHS + 1):
            for epoch_index, (sampling_role, candidate) in enumerate(epoch_plan(pools, epoch)):
                target = target_cache.get(candidate.center_id)
                if target is None:
                    target = build_window_targets(
                        candidate.source_id,
                        sessions[candidate.source_id].labels,
                        candidate.boundary_sample,
                    )
                    target_cache[candidate.center_id] = target
                if sampling_role == "handoff_positive" and not (
                    target.handoff_mask and target.handoff_target == 1
                ):
                    raise SamplingContractError("positive pool contains a non-positive target")
                if sampling_role == "topology_hard_negative" and not (
                    target.handoff_mask and target.handoff_target == 0
                ):
                    raise SamplingContractError("hard-negative pool contains an invalid target")
                row_id = f"epoch-{epoch:02d}-window-{epoch_index:04d}"
                row = {
                    "schema_version": 1,
                    "artifact_role": "psem_training_window",
                    "row_id": row_id,
                    "epoch": epoch,
                    "epoch_index": epoch_index,
                    "source_id": candidate.source_id,
                    "source_waveform_sha256": waveform_by_source[candidate.source_id],
                    "boundary_sample": candidate.boundary_sample,
                    "window_start_sample": target.window_start_sample,
                    "window_end_sample": target.window_end_sample,
                    "observed_frontier_sample": target.observed_frontier_sample,
                    "sampling_role": sampling_role,
                    "topology_family": candidate.family,
                    "target_sha256": canonical_sha256(target.to_dict()),
                    "unsnapped_handoff_event_samples": list(target.handoff_event_samples),
                    "augmentation": augmentation_decision(row_id),
                }
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                role_counts[sampling_role] += 1
                family_counts[candidate.family] += 1
                if target.handoff_mask:
                    handoff_counts[target.handoff_target] += 1
                state_counts.update(
                    state
                    for state, mask in zip(target.state_targets, target.state_mask, strict=True)
                    if mask
                )
                relation_counts.update(pair.target for pair in target.relation_pairs)
                batch_validity.add(target)
    temporary.replace(output_path)
    if role_counts != Counter(
        {role: count * MAXIMUM_EPOCHS for role, count in SAMPLING_COUNTS.items()}
    ):
        raise SamplingContractError("materialized sampling mixture differs from the contract")
    loss_weights = LossWeights(
        handoff_positive=handoff_counts[0] / handoff_counts[1],
        state_classes=_class_weights(state_counts, 3),
        relation_classes=_class_weights(relation_counts, 2),
    )
    minimum_valid_counts_per_batch = batch_validity.finish()
    return {
        "manifest_path": str(output_path.resolve()),
        "manifest_sha256": _sha256_file(output_path),
        "row_count": sum(role_counts.values()),
        "epoch_count": MAXIMUM_EPOCHS,
        "windows_per_epoch": WINDOWS_PER_EPOCH,
        "effective_batch_size": OFFICIAL_EFFECTIVE_BATCH_SIZE,
        "minimum_valid_counts_per_batch": minimum_valid_counts_per_batch,
        "sampling_role_counts": dict(sorted(role_counts.items())),
        "topology_family_counts": dict(sorted(family_counts.items())),
        "pool_counts": {family: len(values) for family, values in sorted(pools.items())},
        "source_count": len({row.source_id for values in pools.values() for row in values}),
        "arms": ["FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM"],
        "seeds": [7301, 7302],
        "shared_center_and_augmentation_manifest": True,
        "topology_family_mapping": {
            "handoff_positive": dict(sorted(POSITIVE_TOPOLOGY_FAMILY.items())),
            "topology_hard_negative": dict(sorted(HARD_NEGATIVE_TOPOLOGY_FAMILY.items())),
        },
        "eval_source_count": 0,
        "loss_weights": {
            "handoff_positive": loss_weights.handoff_positive,
            "state_classes": list(loss_weights.state_classes),
            "relation_classes": list(loss_weights.relation_classes),
        },
        "target_class_counts": {
            "handoff": dict(sorted(handoff_counts.items())),
            "state": dict(sorted(state_counts.items())),
            "relation": dict(sorted(relation_counts.items())),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sampling_rows(path: Path) -> list[Mapping[str, Any]]:
    rows = _jsonl(path)
    for row in rows:
        if row.get("artifact_role") != "psem_training_window" or row.get(
            "augmentation"
        ) != augmentation_decision(str(row.get("row_id", ""))):
            raise SamplingContractError("sampling manifest row is not canonical")
    return rows


def validate_sampling_manifest(
    path: Path,
    sessions: Mapping[str, RuntimeSession],
) -> dict[str, Any]:
    rows = load_sampling_rows(path)
    if len(rows) != MAXIMUM_EPOCHS * WINDOWS_PER_EPOCH:
        raise SamplingContractError("sampling manifest row count differs from the frozen budget")
    pools = candidate_pools(sessions)
    epoch_plans = {epoch: epoch_plan(pools, epoch) for epoch in range(1, MAXIMUM_EPOCHS + 1)}
    role_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    handoff_counts: Counter[int] = Counter()
    state_counts: Counter[int] = Counter()
    relation_counts: Counter[int] = Counter()
    sources = set()
    batch_validity = BatchValidityAccumulator(OFFICIAL_EFFECTIVE_BATCH_SIZE)
    expected_keys = {
        "schema_version",
        "artifact_role",
        "row_id",
        "epoch",
        "epoch_index",
        "source_id",
        "source_waveform_sha256",
        "boundary_sample",
        "window_start_sample",
        "window_end_sample",
        "observed_frontier_sample",
        "sampling_role",
        "topology_family",
        "target_sha256",
        "unsnapped_handoff_event_samples",
        "augmentation",
    }
    for absolute_index, row in enumerate(rows):
        epoch = absolute_index // WINDOWS_PER_EPOCH + 1
        epoch_index = absolute_index % WINDOWS_PER_EPOCH
        row_id = f"epoch-{epoch:02d}-window-{epoch_index:04d}"
        expected_role, expected_candidate = epoch_plans[epoch][epoch_index]
        source_id = row.get("source_id")
        session = sessions.get(str(source_id))
        if (
            set(row) != expected_keys
            or row.get("schema_version") != 1
            or row.get("artifact_role") != "psem_training_window"
            or row.get("row_id") != row_id
            or row.get("epoch") != epoch
            or row.get("epoch_index") != epoch_index
            or row.get("sampling_role") != expected_role
            or row.get("topology_family") != expected_candidate.family
            or row.get("source_id") != expected_candidate.source_id
            or row.get("boundary_sample") != expected_candidate.boundary_sample
            or session is None
            or session.role != TRAIN_ROLE
            or row.get("source_waveform_sha256") != session.waveform_sha256
        ):
            raise SamplingContractError("sampling manifest row identity is invalid")
        target = target_for_row(row, session)
        role = row.get("sampling_role")
        family = row.get("topology_family")
        if (
            (role == "handoff_positive" and (not target.handoff_mask or target.handoff_target != 1))
            or (
                role == "topology_hard_negative"
                and (not target.handoff_mask or target.handoff_target != 0)
            )
            or (role == "source_time_uniform" and family != "source_time_uniform")
            or (role == "handoff_positive" and family not in POSITIVE_FAMILIES)
            or (role == "topology_hard_negative" and family not in HARD_NEGATIVE_FAMILIES)
            or role not in {"handoff_positive", "topology_hard_negative", "source_time_uniform"}
        ):
            raise SamplingContractError("sampling manifest row role or target is invalid")
        batch_validity.add(target)
        role_counts[str(role)] += 1
        family_counts[str(family)] += 1
        sources.add(str(source_id))
        if target.handoff_mask:
            handoff_counts[target.handoff_target] += 1
        state_counts.update(
            state
            for state, mask in zip(target.state_targets, target.state_mask, strict=True)
            if mask
        )
        relation_counts.update(pair.target for pair in target.relation_pairs)
    weights = LossWeights(
        handoff_positive=handoff_counts[0] / handoff_counts[1],
        state_classes=_class_weights(state_counts, 3),
        relation_classes=_class_weights(relation_counts, 2),
    )
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256_file(path),
        "row_count": len(rows),
        "epoch_count": MAXIMUM_EPOCHS,
        "windows_per_epoch": WINDOWS_PER_EPOCH,
        "effective_batch_size": OFFICIAL_EFFECTIVE_BATCH_SIZE,
        "minimum_valid_counts_per_batch": batch_validity.finish(),
        "sampling_role_counts": dict(sorted(role_counts.items())),
        "topology_family_counts": dict(sorted(family_counts.items())),
        "pool_counts": {family: len(values) for family, values in sorted(pools.items())},
        "source_count": len(sources),
        "arms": ["FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM"],
        "seeds": [7301, 7302],
        "shared_center_and_augmentation_manifest": True,
        "topology_family_mapping": {
            "handoff_positive": dict(sorted(POSITIVE_TOPOLOGY_FAMILY.items())),
            "topology_hard_negative": dict(sorted(HARD_NEGATIVE_TOPOLOGY_FAMILY.items())),
        },
        "eval_source_count": 0,
        "loss_weights": {
            "handoff_positive": weights.handoff_positive,
            "state_classes": list(weights.state_classes),
            "relation_classes": list(weights.relation_classes),
        },
        "target_class_counts": {
            "handoff": {str(key): value for key, value in sorted(handoff_counts.items())},
            "state": {str(key): value for key, value in sorted(state_counts.items())},
            "relation": {str(key): value for key, value in sorted(relation_counts.items())},
        },
    }


def load_waveform_window(
    row: Mapping[str, Any],
    session: RuntimeSession,
    corpus_root: Path,
) -> torch.Tensor:
    target = target_for_row(row, session)
    path = (corpus_root.resolve() / session.audio_ref).resolve()
    if not path.is_relative_to(corpus_root.resolve()):
        raise SamplingContractError("waveform path escapes the bound corpus root")
    waveform, sample_rate = torchaudio.load(
        path,
        frame_offset=target.window_start_sample,
        num_frames=WINDOW_SAMPLES,
    )
    if sample_rate != 16000 or waveform.shape != (1, WINDOW_SAMPLES):
        raise SamplingContractError("waveform window differs from the raw-audio contract")
    return waveform[0]


def target_for_row(
    row: Mapping[str, Any],
    session: RuntimeSession,
) -> WindowTargets:
    boundary_sample = row.get("boundary_sample")
    if (
        row.get("source_id") != session.source_id
        or row.get("source_waveform_sha256") != session.waveform_sha256
        or not isinstance(boundary_sample, int)
        or isinstance(boundary_sample, bool)
    ):
        raise SamplingContractError("sampling row and runtime session differ")
    target = build_window_targets(
        session.source_id,
        session.labels,
        boundary_sample,
    )
    expected = {
        "window_start_sample": target.window_start_sample,
        "window_end_sample": target.window_end_sample,
        "observed_frontier_sample": target.observed_frontier_sample,
        "target_sha256": canonical_sha256(target.to_dict()),
        "unsnapped_handoff_event_samples": list(target.handoff_event_samples),
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise SamplingContractError("sampling row target binding is stale")
    return target


def iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SamplingContractError("sampling manifest row must be an object")
            yield value
