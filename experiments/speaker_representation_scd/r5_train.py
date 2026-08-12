from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from experiments.speaker_representation_scd.provenance import load_json, with_self_sha256
from experiments.speaker_representation_scd.r3_probe import eer, roc_auc
from experiments.speaker_representation_scd.r4_continuous import (
    _anchor_coordinates,
    _load_panel_sources,
)
from experiments.speaker_representation_scd.r5_data import (
    R4_POOLED_DIR,
    linear_change_descriptors,
    load_config,
    read_jsonl,
)
from experiments.speaker_representation_scd.r5_models import CausalTCN, LinearProbe
from experiments.speaker_representation_scd.r5_scoring import (
    causal_match_events,
    detect_probability_events,
    event_metrics,
    select_operating_point,
)

OUTPUT_DIR = Path("manifests/r5")
CHECKPOINT_DIR = Path("data/r5/checkpoints")
SEQUENCE_DIR = Path("data/r5/legacy_common_gt/sequences")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    thresholds = np.linspace(0.05, 0.95, 19)
    points = []
    for threshold in thresholds:
        prediction = probabilities >= threshold
        true_positive = int(np.sum(prediction & (labels == 1)))
        false_positive = int(np.sum(prediction & (labels == 0)))
        false_negative = int(np.sum(~prediction & (labels == 1)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        points.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1})
    best = min(points, key=lambda row: (-row["f1"], row["threshold"]))
    return {
        "roc_auc": roc_auc(positive.tolist(), negative.tolist()),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "eer": eer(positive.tolist(), negative.tolist()),
        "best_threshold": best,
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
    }


def run_linear(cache_root: Path, model_id: str) -> Path:
    config = load_config()
    features, labels, metadata = linear_change_descriptors(cache_root, model_id)
    train_mask = np.asarray([row["split"] == "train" for row in metadata])
    dev_mask = ~train_mask
    rows = []
    checkpoint_root = cache_root / CHECKPOINT_DIR / "linear" / model_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    settings = config["linear"]
    for seed in config["seeds"]:
        _seed(int(seed))
        model = LinearProbe(features.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
        x_train = torch.from_numpy(features[train_mask])
        y_train = torch.from_numpy(labels[train_mask])
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        generator = torch.Generator().manual_seed(int(seed))
        loader = DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        positives = float(y_train.sum())
        pos_weight = torch.tensor([(len(y_train) - positives) / positives])
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        x_dev = torch.from_numpy(features[dev_mask])
        y_dev = torch.from_numpy(labels[dev_mask])
        for epoch in range(int(settings["max_epochs"])):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                dev_loss = float(loss_function(model(x_dev), y_dev))
            if dev_loss < best_loss - 1e-6:
                best_loss = dev_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= int(settings["patience"]):
                break
        if best_state is None:
            raise RuntimeError("linear probe did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x_dev)).numpy()
        metrics = _binary_metrics(labels[dev_mask], probabilities)
        checkpoint = checkpoint_root / f"seed_{seed}.pt"
        torch.save(
            {
                "model_state": best_state,
                "input_dimension": int(features.shape[1]),
                "seed": int(seed),
            },
            checkpoint,
        )
        rows.append({"seed": int(seed), "best_dev_loss": best_loss, "epochs": epoch + 1, "metrics": metrics})
    result = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r5_a_linear_probe_result",
            "model_id": model_id,
            "layer_id": config["models"][model_id],
            "train_count": int(train_mask.sum()),
            "dev_count": int(dev_mask.sum()),
            "seeds": rows,
            "median_dev_roc_auc": float(np.median([row["metrics"]["roc_auc"] for row in rows])),
            "median_dev_eer": float(np.median([row["metrics"]["eer"] for row in rows])),
        }
    )
    path = cache_root / OUTPUT_DIR / "r5_a" / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class SequenceDataset(Dataset):
    def __init__(self, vectors: np.ndarray, rows: list[dict[str, Any]]) -> None:
        self.vectors = vectors
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        vector_rows = np.asarray(row["vector_rows"], dtype=np.int64)
        values = torch.from_numpy(np.asarray(self.vectors[vector_rows], dtype=np.float32))
        labels = torch.tensor(row["labels"], dtype=torch.float32)
        return values, labels, row


def _collate(batch: Sequence[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    values, labels, rows = zip(*batch, strict=True)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    padded_values = pad_sequence(values, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)
    mask = torch.arange(padded_values.shape[1])[None, :] < lengths[:, None]
    return padded_values, padded_labels, mask, list(rows)


def _masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    return losses[mask].mean()


def _dev_frame_metrics(
    model: CausalTCN,
    vectors: np.ndarray,
    rows: list[dict[str, Any]],
    batch_size: int,
    pos_weight: torch.Tensor,
) -> tuple[float, float]:
    losses = []
    labels = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch_values, batch_labels, mask, _ in DataLoader(
            SequenceDataset(vectors, rows),
            batch_size=batch_size,
            collate_fn=_collate,
        ):
            logits = model(batch_values)
            losses.append(float(_masked_loss(logits, batch_labels, mask, pos_weight)))
            labels.append(batch_labels[mask].numpy())
            probabilities.append(torch.sigmoid(logits)[mask].numpy())
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    return float(np.mean(losses)), float(average_precision_score(y_true, y_score))


def _predict(model: CausalTCN, values: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    receptive_history = 14
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            history_start = max(0, start - receptive_history)
            chunk = torch.from_numpy(np.asarray(values[history_start : start + chunk_size], dtype=np.float32))[None]
            probabilities = torch.sigmoid(model(chunk))[0].numpy()
            outputs.append(probabilities[start - history_start :])
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def _dev_operating_points(model: CausalTCN, vectors: np.ndarray, rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = [(row, _predict(model, vectors[np.asarray(row["vector_rows"], dtype=np.int64)])) for row in rows]
    grid = []
    for threshold in config["sequence"]["thresholds"]:
        for confirmation in config["sequence"]["confirmation_hops"]:
            total_events = 0
            matched: list[dict[str, Any]] = []
            ground_truth_count = 0
            source_seconds = 0.0
            for row, probabilities in predictions:
                frontiers = row["frontier_samples"]
                events = detect_probability_events(probabilities, frontiers, float(threshold), int(confirmation))
                ground_truth = [int(row["coordinate"])] if row["class"] == "positive" else []
                matched.extend(causal_match_events(ground_truth, events))
                total_events += len(events)
                ground_truth_count += len(ground_truth)
                source_seconds += (frontiers[-1] - frontiers[0] + int(config["hop_samples"])) / 16000
            metrics = event_metrics(matched, total_events, ground_truth_count, source_seconds / 3600)
            grid.append(
                {
                    "config_id": f"threshold={float(threshold):.2f}|confirmation={int(confirmation)}",
                    "threshold": float(threshold),
                    "confirmation_hops": int(confirmation),
                    "metrics": metrics,
                }
            )
    return grid, select_operating_point(grid, float(config["development_false_event_budget_per_hour"]))


def _r4_metrics(cache_root: Path, model_id: str, model: CausalTCN, operating_point: dict[str, Any]) -> dict[str, Any]:
    vector_path = cache_root / R4_POOLED_DIR / model_id / "vectors_300.npy"
    index_path = cache_root / R4_POOLED_DIR / model_id / "index_300.jsonl"
    vectors = np.load(vector_path, mmap_mode="r")
    index = read_jsonl(index_path)
    ground_truth = _anchor_coordinates(cache_root)
    sources, _ = _load_panel_sources(cache_root)
    source_map = {str(row["session_id"]): row for row in sources}
    matched: list[dict[str, Any]] = []
    total_events = 0
    ground_truth_count = 0
    source_hours = 0.0
    for row in index:
        start = int(row["row_start"])
        count = int(row["row_count"])
        probabilities = _predict(model, vectors[start : start + count])
        events = detect_probability_events(
            probabilities,
            row["frontier_samples"],
            float(operating_point["threshold"]),
            int(operating_point["confirmation_hops"]),
        )
        session_id = str(row["session_id"])
        gt = [int(value["coordinate"]) for value in ground_truth.get(session_id, [])]
        matched.extend(causal_match_events(gt, events))
        total_events += len(events)
        ground_truth_count += len(gt)
        source = source_map[session_id]
        source_hours += (int(source["eligible_end_sample"]) - int(source["eligible_start_sample"])) / 16000 / 3600
    return event_metrics(matched, total_events, ground_truth_count, source_hours)


def run_b0(cache_root: Path, model_id: str, seeds: Sequence[int] | None = None) -> Path:
    config = load_config()
    root = cache_root / SEQUENCE_DIR / model_id
    vectors = np.load(root / "vectors.npy", mmap_mode="r")
    rows = read_jsonl(root / "index.jsonl")
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    settings = config["tcn"]
    positive = sum(sum(row["labels"]) for row in train_rows)
    total = sum(len(row["labels"]) for row in train_rows)
    pos_weight = torch.tensor([(total - positive) / positive], dtype=torch.float32)
    result_rows = []
    checkpoint_root = cache_root / CHECKPOINT_DIR / "b0" / model_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    selected_seeds = [int(value) for value in (seeds or config["seeds"])]
    for seed in selected_seeds:
        _seed(int(seed))
        model = CausalTCN(
            int(vectors.shape[1]),
            int(settings["hidden_dimension"]),
            int(settings["kernel_size"]),
            [int(value) for value in settings["dilations"]],
            float(settings["dropout"]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
        generator = torch.Generator().manual_seed(int(seed))
        loader = DataLoader(
            SequenceDataset(vectors, train_rows),
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            collate_fn=_collate,
            generator=generator,
        )
        best_loss = float("inf")
        best_pr_auc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(int(settings["max_epochs"])):
            model.train()
            for batch_values, batch_labels, mask, _ in loader:
                optimizer.zero_grad()
                loss = _masked_loss(model(batch_values), batch_labels, mask, pos_weight)
                loss.backward()
                optimizer.step()
            dev_loss, dev_pr_auc = _dev_frame_metrics(
                model,
                vectors,
                dev_rows,
                int(settings["batch_size"]),
                pos_weight,
            )
            if dev_pr_auc > best_pr_auc + 1e-6:
                best_loss = dev_loss
                best_pr_auc = dev_pr_auc
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= int(settings["patience"]):
                break
        if best_state is None:
            raise RuntimeError("B0 did not produce a checkpoint")
        model.load_state_dict(best_state)
        grid, selected = _dev_operating_points(model, vectors, dev_rows, config)
        r4_metrics = _r4_metrics(cache_root, model_id, model, selected)
        checkpoint = checkpoint_root / f"seed_{seed}.pt"
        torch.save(
            {
                "model_state": best_state,
                "input_dimension": int(vectors.shape[1]),
                "seed": int(seed),
                "operating_point": selected,
            },
            checkpoint,
        )
        result_rows.append(
            {
                "seed": int(seed),
                "epochs": epoch + 1,
                "best_dev_loss": best_loss,
                "best_dev_pr_auc": best_pr_auc,
                "dev_operating_point": selected,
                "dev_grid": grid,
                "r4_fixed_operating_point_metrics": r4_metrics,
            }
        )
    path = cache_root / OUTPUT_DIR / "r5_b0" / f"{model_id}.json"
    prior_rows = []
    if path.is_file():
        prior_rows = [row for row in load_json(path).get("seeds", []) if int(row["seed"]) not in selected_seeds]
    combined_rows = sorted([*prior_rows, *result_rows], key=lambda row: int(row["seed"]))
    result = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r5_b0_frozen_causal_tcn_result",
            "model_id": model_id,
            "layer_id": config["models"][model_id],
            "train_sequence_count": len(train_rows),
            "dev_sequence_count": len(dev_rows),
            "input_dimension": int(vectors.shape[1]),
            "seeds": combined_rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("linear", "b0"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seeds")
    args = parser.parse_args()
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else None
    path = (
        run_linear(cache_root, args.model_id)
        if args.stage == "linear"
        else run_b0(cache_root, args.model_id, seeds)
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
