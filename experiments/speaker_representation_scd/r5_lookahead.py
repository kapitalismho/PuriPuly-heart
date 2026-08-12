from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from experiments.speaker_representation_scd.provenance import with_self_sha256
from experiments.speaker_representation_scd.r4_continuous import _anchor_coordinates
from experiments.speaker_representation_scd.r5_capability import PROFILE_RULES, _profile
from experiments.speaker_representation_scd.r5_data import (
    R4_POOLED_DIR,
    load_config,
    read_jsonl,
)
from experiments.speaker_representation_scd.r5_models import CausalTCN
from experiments.speaker_representation_scd.r5_scoring import (
    causal_match_events,
    detect_probability_events,
    event_metrics,
)
from experiments.speaker_representation_scd.r5_train import (
    _collate,
    _masked_loss,
    _predict,
    _seed,
)

SEQUENCE_DIR = Path("data/r5/legacy_common_gt/sequences")
CHECKPOINT_DIR = Path("data/r5/checkpoints/lookahead")
OUTPUT_DIR = Path("manifests/r5/r5_b0_lookahead")


class LookaheadDataset(Dataset):
    def __init__(self, vectors: np.ndarray, rows: list[dict[str, Any]], lookahead_hops: int) -> None:
        self.vectors = vectors
        self.rows = rows
        self.lookahead_hops = lookahead_hops

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        shift = self.lookahead_hops
        vector_rows = np.asarray(row["vector_rows"], dtype=np.int64)
        selected = vector_rows[shift:] if shift else vector_rows
        labels = row["labels"][:-shift] if shift else row["labels"]
        values = torch.from_numpy(np.asarray(self.vectors[selected], dtype=np.float32))
        return values, torch.tensor(labels, dtype=torch.float32), row


def _records_from_rows(
    model: CausalTCN,
    vectors: np.ndarray,
    rows: Sequence[dict[str, Any]],
    lookahead_hops: int,
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        vector_rows = np.asarray(row["vector_rows"], dtype=np.int64)
        selected = vector_rows[lookahead_hops:] if lookahead_hops else vector_rows
        logical = row["frontier_samples"][:-lookahead_hops] if lookahead_hops else row["frontier_samples"]
        availability = row["frontier_samples"][lookahead_hops:] if lookahead_hops else row["frontier_samples"]
        if len(selected) == 0:
            continue
        records.append(
            {
                "logical_frontiers": logical,
                "availability_frontiers": availability,
                "ground_truth": [int(row["coordinate"])] if row["class"] == "positive" else [],
                "probabilities": _predict(model, vectors[selected]),
                "source_seconds": (int(logical[-1]) - int(logical[0]) + 1600) / 16000,
            }
        )
    return records


def _r4_records(
    cache_root: Path,
    model_id: str,
    model: CausalTCN,
    lookahead_hops: int,
) -> list[dict[str, Any]]:
    vectors = np.load(
        cache_root / R4_POOLED_DIR / model_id / "vectors_300.npy",
        mmap_mode="r",
    )
    ground_truth = _anchor_coordinates(cache_root)
    records = []
    for row in read_jsonl(cache_root / R4_POOLED_DIR / model_id / "index_300.jsonl"):
        start = int(row["row_start"])
        count = int(row["row_count"])
        if count <= lookahead_hops:
            continue
        session_id = str(row["session_id"])
        logical = row["frontier_samples"][:-lookahead_hops]
        availability = row["frontier_samples"][lookahead_hops:]
        records.append(
            {
                "logical_frontiers": logical,
                "availability_frontiers": availability,
                "ground_truth": [
                    int(value["coordinate"]) for value in ground_truth.get(session_id, [])
                ],
                "probabilities": _predict(
                    model,
                    vectors[start + lookahead_hops : start + count],
                ),
                "source_seconds": (
                    int(logical[-1])
                    - int(logical[0])
                    + 1600
                )
                / 16000,
            }
        )
    return records


def _dev_metrics(
    model: CausalTCN,
    vectors: np.ndarray,
    rows: list[dict[str, Any]],
    lookahead_hops: int,
    batch_size: int,
    pos_weight: torch.Tensor,
) -> tuple[float, float]:
    losses = []
    labels = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch_values, batch_labels, mask, _ in DataLoader(
            LookaheadDataset(vectors, rows, lookahead_hops),
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


def _score_grid(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    config = load_config()
    rows = []
    source_hours = sum(float(record["source_seconds"]) for record in records) / 3600
    ground_truth_count = sum(len(record["ground_truth"]) for record in records)
    for threshold in config["sequence"]["thresholds"]:
        for confirmation in config["sequence"]["confirmation_hops"]:
            total_events = 0
            matched = []
            for record in records:
                events = detect_probability_events(
                    record["probabilities"],
                    record["logical_frontiers"],
                    float(threshold),
                    int(confirmation),
                )
                for event in events:
                    event["emit_sample"] = int(
                        record["availability_frontiers"][int(event["emit_hop"])]
                    )
                matched.extend(
                    causal_match_events(
                        record["ground_truth"],
                        events,
                        tolerance_ms=1500,
                    )
                )
                total_events += len(events)
            rows.append(
                {
                    "config_id": f"threshold={float(threshold):.2f}|confirmation={int(confirmation)}",
                    "threshold": float(threshold),
                    "confirmation_hops": int(confirmation),
                    "metrics": event_metrics(
                        matched,
                        total_events,
                        ground_truth_count,
                        source_hours,
                    ),
                }
            )
    return rows


def run_lookahead(cache_root: Path, model_id: str, lookahead_ms: int, seed: int) -> Path:
    if lookahead_ms not in (100, 300):
        raise ValueError("R5 bounded lookahead must be 100 or 300 ms")
    config = load_config()
    lookahead_hops = lookahead_ms // 100
    root = cache_root / SEQUENCE_DIR / model_id
    vectors = np.load(root / "vectors.npy", mmap_mode="r")
    rows = read_jsonl(root / "index.jsonl")
    train_rows = [row for row in rows if row["split"] == "train" and len(row["labels"]) > lookahead_hops]
    dev_rows = [row for row in rows if row["split"] == "dev" and len(row["labels"]) > lookahead_hops]
    settings = config["tcn"]
    positive = sum(sum(row["labels"][:-lookahead_hops]) for row in train_rows)
    total = sum(len(row["labels"]) - lookahead_hops for row in train_rows)
    pos_weight = torch.tensor([(total - positive) / positive], dtype=torch.float32)
    _seed(seed)
    model = CausalTCN(
        int(vectors.shape[1]),
        int(settings["hidden_dimension"]),
        int(settings["kernel_size"]),
        [int(value) for value in settings["dilations"]],
        float(settings["dropout"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        LookaheadDataset(vectors, train_rows, lookahead_hops),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
    )
    best_loss = float("inf")
    best_pr_auc = -1.0
    best_state = None
    stale = 0
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        for batch_values, batch_labels, mask, _ in loader:
            optimizer.zero_grad()
            loss = _masked_loss(model(batch_values), batch_labels, mask, pos_weight)
            loss.backward()
            optimizer.step()
        dev_loss, dev_pr_auc = _dev_metrics(
            model,
            vectors,
            dev_rows,
            lookahead_hops,
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
        raise RuntimeError("lookahead training did not produce a checkpoint")
    model.load_state_dict(best_state)
    dev_records = _records_from_rows(model, vectors, dev_rows, lookahead_hops)
    dev_grid = _score_grid(dev_records)
    r4_grid = _score_grid(_r4_records(cache_root, model_id, model, lookahead_hops))
    dev_profiles = {name: _profile(dev_grid, name) for name in PROFILE_RULES}
    r4_by_id = {row["config_id"]: row for row in r4_grid}
    fixed_profiles = {
        name: {
            **profile,
            "r4_fixed_metrics": r4_by_id[profile["selected"]["config_id"]]["metrics"],
        }
        for name, profile in dev_profiles.items()
    }
    checkpoint = cache_root / CHECKPOINT_DIR / model_id / f"lookahead_{lookahead_ms}ms" / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "input_dimension": int(vectors.shape[1]),
            "lookahead_ms": lookahead_ms,
            "seed": seed,
        },
        checkpoint,
    )
    document = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r5_b0_bounded_lookahead_result",
            "claim_level": "internal_exploratory_sensitivity",
            "model_id": model_id,
            "seed": seed,
            "lookahead_ms": lookahead_ms,
            "availability_rule": "logical_frontier_plus_lookahead_plus_confirmation",
            "epochs": epoch + 1,
            "best_dev_loss": best_loss,
            "best_dev_pr_auc": best_pr_auc,
            "dev_selected_r4_fixed_profiles": fixed_profiles,
            "r4_exploratory_profiles": {
                name: _profile(r4_grid, name) for name in PROFILE_RULES
            },
            "dev_grid": dev_grid,
            "r4_grid": r4_grid,
        }
    )
    path = cache_root / OUTPUT_DIR / f"{model_id}.lookahead_{lookahead_ms}ms.seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--lookahead-ms", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    print(run_lookahead(cache_root, args.model_id, args.lookahead_ms, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
