from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    SessionExamples,
    config,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
    RESULTS_ROOT,
    aggregate_conditions,
    session_row,
)
from experiments.psem_frozen_ceiling_gate.experiment_support import (
    load_json,
    percentile,
    sha256_file,
    weighted_average_precision,
    write_json,
)
from experiments.psem_frozen_ceiling_gate.posterior_features import (
    TemporalContract,
    fullslot_base,
    scalar_base,
    temporal_features,
)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


@dataclass(slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-5] = 1.0
        return cls(mean, scale)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)


@dataclass(slots=True)
class LinearProbe:
    standardizer: Standardizer
    weights: np.ndarray
    bias: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return sigmoid(self.standardizer.apply(values) @ self.weights + self.bias)


@dataclass(slots=True)
class TinyMLPProbe:
    standardizer: Standardizer
    first: np.ndarray
    first_bias: np.ndarray
    second: np.ndarray
    second_bias: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        hidden = np.maximum(self.standardizer.apply(values) @ self.first + self.first_bias, 0.0)
        return sigmoid(hidden @ self.second + self.second_bias)


def training_subset(
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) != len(targets) or len(values) != len(weights) or not len(values):
        raise ValueError("training arrays must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(targets)
    negative = np.flatnonzero(np.logical_not(targets))
    if not len(positive) or not len(negative):
        raise ValueError("training requires both target classes")
    per_class = cap // 2
    positive = rng.choice(positive, min(len(positive), per_class), replace=False)
    negative = rng.choice(negative, min(len(negative), cap - len(positive)), replace=False)
    selected = np.concatenate((positive, negative))
    rng.shuffle(selected)
    return (
        values[selected],
        targets[selected].astype(np.float32),
        weights[selected].astype(np.float32),
    )


def balanced_weights(targets: np.ndarray, duration_weights: np.ndarray) -> np.ndarray:
    positive = duration_weights[targets > 0.5].sum()
    negative = duration_weights[targets <= 0.5].sum()
    values = duration_weights.copy()
    values[targets > 0.5] *= 0.5 / positive
    values[targets <= 0.5] *= 0.5 / negative
    return values * len(values)


def fit_linear(
    values: np.ndarray,
    targets: np.ndarray,
    duration_weights: np.ndarray,
    *,
    cfg: dict[str, Any],
    seed: int,
) -> LinearProbe:
    standardizer = Standardizer.fit(values)
    features = standardizer.apply(values)
    weights = np.zeros(features.shape[1], dtype=np.float32)
    bias = 0.0
    rng = np.random.default_rng(seed)
    sample_weights = balanced_weights(targets, duration_weights)
    batch = int(cfg["training_batch_size"])
    step = 0
    for _ in range(int(cfg["training_epochs"])):
        order = rng.permutation(len(features))
        for first in range(0, len(order), batch):
            chosen = order[first : first + batch]
            x = features[chosen]
            y = targets[chosen]
            w = sample_weights[chosen]
            error = (sigmoid(x @ weights + bias) - y) * w
            learning_rate = 0.03 / (1.0 + step / 300.0)
            weights -= learning_rate * (x.T @ error / len(chosen) + 1e-4 * weights)
            bias -= learning_rate * float(error.mean())
            step += 1
    return LinearProbe(standardizer, weights, bias)


def fit_mlp(
    values: np.ndarray,
    targets: np.ndarray,
    duration_weights: np.ndarray,
    *,
    cfg: dict[str, Any],
    seed: int,
) -> TinyMLPProbe:
    standardizer = Standardizer.fit(values)
    features = standardizer.apply(values)
    rng = np.random.default_rng(seed)
    hidden_units = int(cfg["tiny_mlp_hidden_units"])
    first = rng.normal(0.0, 0.08, (features.shape[1], hidden_units)).astype(np.float32)
    first_bias = np.zeros(hidden_units, dtype=np.float32)
    second = rng.normal(0.0, 0.08, hidden_units).astype(np.float32)
    second_bias = 0.0
    sample_weights = balanced_weights(targets, duration_weights)
    batch = int(cfg["training_batch_size"])
    step = 0
    for _ in range(int(cfg["training_epochs"])):
        order = rng.permutation(len(features))
        for start in range(0, len(order), batch):
            chosen = order[start : start + batch]
            x = features[chosen]
            y = targets[chosen]
            w = sample_weights[chosen]
            hidden_pre = x @ first + first_bias
            hidden = np.maximum(hidden_pre, 0.0)
            error = (sigmoid(hidden @ second + second_bias) - y) * w
            grad_second = hidden.T @ error / len(chosen) + 1e-4 * second
            grad_second_bias = float(error.mean())
            hidden_error = np.outer(error, second) * (hidden_pre > 0.0)
            grad_first = x.T @ hidden_error / len(chosen) + 1e-4 * first
            grad_first_bias = hidden_error.mean(axis=0)
            learning_rate = 0.02 / (1.0 + step / 300.0)
            first -= learning_rate * grad_first
            first_bias -= learning_rate * grad_first_bias
            second -= learning_rate * grad_second
            second_bias -= learning_rate * grad_second_bias
            step += 1
    return TinyMLPProbe(standardizer, first, first_bias, second, second_bias)


def feature_matrix(
    session: SessionExamples,
    *,
    evidence: str,
    noncausal: bool,
    contract: TemporalContract,
) -> np.ndarray:
    if evidence == "scalar":
        base = scalar_base(
            session.probabilities,
            session.evidence_delay_ms,
        )
    else:
        base = fullslot_base(
            session.probabilities,
            session.alive,
            session.evidence_delay_ms,
            session.reset,
        )
    return temporal_features(base, session.episode_ids, contract, noncausal=noncausal)


def training_data(
    sessions: list[SessionExamples],
    matrices: dict[str, np.ndarray],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    weights = []
    for session in sessions:
        usable = np.logical_and(session.valid, np.logical_not(session.masked))
        features.append(matrices[session.source_id][usable])
        targets.append(session.target[usable])
        weights.append(session.weights[usable])
    return training_subset(
        np.concatenate(features),
        np.concatenate(targets),
        np.concatenate(weights),
        cap=int(cfg["training_sample_cap"]),
        seed=seed,
    )


def fit_sanity(
    probe: LinearProbe | TinyMLPProbe,
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    scores = probe.predict(values)
    predictions = scores >= 0.5
    return {
        "sample_count": len(values),
        "positive_fraction": float(np.average(targets, weights=weights)),
        "duration_weighted_accuracy": float(np.average(predictions == targets, weights=weights)),
        "duration_weighted_average_precision": weighted_average_precision(
            targets.astype(bool).tolist(), scores.tolist(), weights.tolist()
        ),
    }


def run() -> dict[str, Any]:
    cfg = config()
    package_root = RESULTS_ROOT.parents[1]
    split_path = package_root / "split_manifest.json"
    evidence_path = package_root / "evidence_reuse_receipt.json"
    mapping_path = package_root / "oracle_mapping_ledger.jsonl"
    action_reference_path = package_root / "action_reference_ledger.jsonl"
    split = load_json(split_path)
    evidence_receipt = load_json(evidence_path)
    if split["strategy"] != cfg["split"]["strategy"]:
        raise ValueError("frozen split strategy mismatch")
    frozen_sources = {value["source_id"]: value for value in split["sources"]}
    folds = {value["held_out_family"]: value for value in split["folds"]}
    provenance = {
        "config_sha256": sha256_file(package_root / "config.json"),
        "split_manifest_sha256": sha256_file(split_path),
        "evidence_reuse_receipt_sha256": sha256_file(evidence_path),
        "oracle_mapping_ledger_sha256": sha256_file(mapping_path),
        "action_reference_ledger_sha256": sha256_file(action_reference_path),
        "repository_baseline_sha": evidence_receipt["repository_baseline_sha"],
    }
    training_sessions = load_sessions((500,))
    for session in training_sessions:
        frozen = frozen_sources.get(session.source_id)
        if (
            frozen is None
            or frozen["source_family"] != session.source_family
            or frozen["row_sha256"] != session.manifest["row_sha256"]
        ):
            raise ValueError(f"session differs from frozen split: {session.source_id}")
    contract = TemporalContract(
        tuple(map(int, cfg["causal_lag_frames"])),
        tuple(map(int, cfg["noncausal_future_frames"])),
    )
    source_rows: list[dict[str, Any]] = []
    training_receipts = []
    models: dict[tuple[str, str, str], LinearProbe | TinyMLPProbe] = {}
    conditions = (
        ("S-probe", "scalar", False),
        ("P-C", "fullslot", False),
        ("P-NC", "fullslot", True),
    )
    for condition_index, (condition, evidence_kind, noncausal) in enumerate(conditions):
        matrices = {
            session.source_id: feature_matrix(
                session,
                evidence=evidence_kind,
                noncausal=noncausal,
                contract=contract,
            )
            for session in training_sessions
        }
        for fold_index, held_family in enumerate(cfg["split"]["families"]):
            fold = folds[held_family]
            train_ids = set(map(str, fold["training_sources"]))
            eval_ids = set(map(str, fold["evaluation_sources"]))
            if train_ids & eval_ids:
                raise ValueError(f"frozen split leaks source IDs: {held_family}")
            train = [value for value in training_sessions if value.source_id in train_ids]
            if {value.source_id for value in train} != train_ids:
                raise ValueError(f"frozen training source coverage mismatch: {held_family}")
            seed = int(cfg["training_seed"]) + condition_index * 10 + fold_index
            train_x, train_y, train_w = training_data(train, matrices, cfg, seed)
            for probe_name in cfg["probe_classes"]:
                probe = (
                    fit_linear(train_x, train_y, train_w, cfg=cfg, seed=seed)
                    if probe_name == "linear"
                    else fit_mlp(train_x, train_y, train_w, cfg=cfg, seed=seed)
                )
                models[(condition, probe_name, held_family)] = probe
                training_receipts.append(
                    {
                        "condition": condition,
                        "probe_class": probe_name,
                        "held_out_family": held_family,
                        "train_source_ids": sorted(train_ids),
                        "eval_source_ids": sorted(eval_ids),
                        "future_context_ms": cfg["noncausal_horizon_ms"] if noncausal else 0,
                        "train_fit_sanity": fit_sanity(probe, train_x, train_y, train_w),
                    }
                )
    scoring_sessions = [value for value in training_sessions if value.role == "eval"]
    for persistence in map(int, cfg["current_confirmation_ms"]):
        for session in scoring_sessions:
            scores = (session.probabilities[:, 0] < float(cfg["current_anchor_threshold"])).astype(
                np.float32
            )
            source_rows.append(
                session_row(
                    session,
                    scores,
                    condition="S-current",
                    probe_class="current",
                    threshold=0.5,
                    confirmation_ms=persistence,
                    time_condition="causal",
                )
            )
    for persistence in map(int, cfg["probe_confirmation_ms"]):
        for condition, evidence_kind, noncausal in conditions:
            for session in scoring_sessions:
                if session.source_id not in set(folds[session.source_family]["evaluation_sources"]):
                    raise ValueError(
                        f"source is outside its frozen evaluation fold: {session.source_id}"
                    )
                matrix = feature_matrix(
                    session,
                    evidence=evidence_kind,
                    noncausal=noncausal,
                    contract=contract,
                )
                for probe_name in cfg["probe_classes"]:
                    scores = models[(condition, probe_name, session.source_family)].predict(matrix)
                    for threshold in cfg["probe_thresholds"]:
                        source_rows.append(
                            session_row(
                                session,
                                scores,
                                condition=condition,
                                probe_class=probe_name,
                                threshold=float(threshold),
                                confirmation_ms=persistence,
                                time_condition=("bounded_noncausal" if noncausal else "causal"),
                                future_context_frames=(
                                    max(contract.future_lags) if noncausal else 0
                                ),
                            )
                        )
    artifacts = {
        "S-current": "scalar_current_metrics.json",
        "S-probe": "scalar_probe_metrics.json",
        "P-C": "fullslot_causal_metrics.json",
        "P-NC": "fullslot_noncausal_metrics.json",
    }
    for condition, name in artifacts.items():
        chosen = [value for value in source_rows if value["condition"] == condition]
        write_json(
            RESULTS_ROOT / name,
            {
                "schema_version": "psem.frozen_ceiling.probe_metrics.v1",
                "condition": condition,
                "provenance": provenance,
                "rows": aggregate_conditions(chosen),
                "training_receipts": [
                    value for value in training_receipts if value["condition"] == condition
                ],
            },
        )
    compact_rows = []
    for row in source_rows:
        metrics = row["metrics"]
        compact_rows.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in ("metrics", "diagnostics")
                },
                "metrics": {
                    key: metrics[key]
                    for key in (
                        "active_speech_seconds",
                        "predicted_cut_count",
                        "reference_replacement_count",
                        "matched_replacement_count",
                        "false_cut_count",
                        "missed_replacement_count",
                        "exclusive_other_contamination_seconds",
                    )
                },
                "diagnostics": row["diagnostics"],
            }
        )
    write_json(
        RESULTS_ROOT / "source_family_results.json",
        {
            "schema_version": "psem.frozen_ceiling.source_results.v1",
            "provenance": provenance,
            "rows": compact_rows,
        },
    )
    bootstrap = paired_bootstrap(
        source_rows, int(cfg["bootstrap_resamples"]), int(cfg["training_seed"])
    )
    write_json(RESULTS_ROOT / "paired_deltas_or_bootstrap.json", bootstrap)
    return {
        "source_row_count": len(source_rows),
        "training_receipt_count": len(training_receipts),
    }


def paired_bootstrap(rows: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    def select(condition: str) -> dict[str, dict[str, Any]]:
        return {
            value["source_id"]: value
            for value in rows
            if value["condition"] == condition
            and value["probe_class"] == "tiny_mlp"
            and value["threshold"] == 0.5
            and value["confirmation_ms"] == 300
        }

    scalar = select("S-probe")
    full = select("P-C")
    source_ids = sorted(set(scalar) & set(full))
    deltas = np.asarray(
        [
            [
                full[source]["metrics"]["exclusive_other_contamination_seconds"]
                - scalar[source]["metrics"]["exclusive_other_contamination_seconds"],
                full[source]["metrics"]["false_cut_count"]
                - scalar[source]["metrics"]["false_cut_count"],
                full[source]["metrics"]["missed_replacement_count"]
                - scalar[source]["metrics"]["missed_replacement_count"],
            ]
            for source in source_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = deltas[rng.integers(0, len(deltas), (resamples, len(deltas)))].mean(axis=1)
    names = ("contamination_seconds", "false_cut_count", "missed_replacement_count")
    return {
        "schema_version": "psem.frozen_ceiling.paired_bootstrap.v1",
        "comparison": "P-C minus S-probe at predeclared tiny_mlp threshold=0.5 confirmation=300ms",
        "unit": "source_session",
        "source_count": len(source_ids),
        "resamples": resamples,
        "deltas": {
            name: {
                "point_mean": float(deltas[:, index].mean()),
                "lower_95": percentile(sampled[:, index], 2.5),
                "upper_95": percentile(sampled[:, index], 97.5),
            }
            for index, name in enumerate(names)
        },
    }


if __name__ == "__main__":
    print(run())
