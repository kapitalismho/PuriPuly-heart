from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_representation_scd.provenance import sha256_file, with_self_sha256
from experiments.speaker_representation_scd.r3_probe import (
    BATCH_SIZE,
    ERES_MODEL_ID,
    REGISTRY_LAYER_ORDER,
    _make_extractor,
    _waveform_paths,
)
from experiments.speaker_representation_scd.r4_continuous import _mean_pool_l2_batch
from experiments.speaker_representation_scd.r5_data import (
    R3_POOLED_DIR,
    R5DataError,
    load_config,
    read_jsonl,
    sequence_rows,
)

OUTPUT_DIR = Path("data/r5/legacy_common_gt/sequences")


def _r3_vectors(cache_root: Path, model_id: str, layer_id: str) -> dict[tuple[str, int, int], np.ndarray]:
    index_path = cache_root / R3_POOLED_DIR / model_id / "index_300.jsonl"
    vector_path = cache_root / R3_POOLED_DIR / model_id / "vectors_300.npy"
    rows = read_jsonl(index_path)
    values = np.load(vector_path, mmap_mode="r")
    layer_index = REGISTRY_LAYER_ORDER[model_id].index(layer_id)
    return {
        (str(row["waveform_id"]), int(row["window_start_sample"]), int(row["window_end_sample"])): np.asarray(
            values[int(row["row_index"]), layer_index], dtype=np.float32
        )
        for row in rows
    }


def extract_sequences(cache_root: Path, model_id: str, *, threads: int) -> Path:
    config = load_config()
    if model_id not in config["models"]:
        raise R5DataError(f"unknown R5 model: {model_id}")
    layer_id = str(config["models"][model_id])
    rows = sequence_rows(cache_root)
    output = cache_root / OUTPUT_DIR / model_id
    vector_path = output / "vectors.npy"
    index_path = output / "index.jsonl"
    manifest_path = output / "manifest.json"
    if vector_path.is_file() and index_path.is_file() and manifest_path.is_file():
        return manifest_path
    output.mkdir(parents=True, exist_ok=True)
    windows: dict[tuple[str, int, int], int] = {}
    candidate_keys: dict[str, list[tuple[str, int, int]]] = {}
    for row in rows:
        keys = [
            (str(row["waveform_id"]), int(start), int(end))
            for start, end in zip(row["window_start_samples"], row["window_end_samples"], strict=True)
        ]
        candidate_keys[str(row["candidate_id"])] = keys
        for key in keys:
            windows.setdefault(key, len(windows))
    ordered = [key for key, _ in sorted(windows.items(), key=lambda item: item[1])]
    reused = _r3_vectors(cache_root, model_id, layer_id)
    dimension = next(iter(reused.values())).shape[0]
    partial_vectors = output / "vectors.partial.npy"
    pooled = np.lib.format.open_memmap(
        partial_vectors,
        mode="w+",
        dtype=np.float32,
        shape=(len(ordered), dimension),
    )
    missing: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    reused_count = 0
    for row_index, key in enumerate(ordered):
        if key in reused:
            pooled[row_index] = reused[key]
            reused_count += 1
        else:
            missing[key[0]].append((row_index, key[1], key[2]))
    if missing:
        import soundfile as sf

        extractor = _make_extractor(model_id, cache_root, threads=threads)
        waveform_paths = _waveform_paths(cache_root)
        for waveform_id in sorted(missing):
            path = waveform_paths[waveform_id]
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            if sample_rate != 16000 or audio.shape[1] != 1:
                raise R5DataError(f"waveform geometry differs: {waveform_id}")
            waveform = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
            window_rows = missing[waveform_id]
            for batch_start in range(0, len(window_rows), BATCH_SIZE):
                batch_rows = window_rows[batch_start : batch_start + BATCH_SIZE]
                batch_windows = [
                    np.ascontiguousarray(waveform[start:end], dtype=np.float32)
                    for _, start, end in batch_rows
                ]
                observed = [end for _, _, end in batch_rows]
                layer_argument = "tap_ids" if model_id == ERES_MODEL_ID else "layer_ids"
                batch = extractor.extract(
                    batch_windows,
                    observed,
                    **{layer_argument: [layer_id]},
                )
                vectors = _mean_pool_l2_batch(batch.layers[layer_id], batch.valid_lengths[layer_id])
                for position, (row_index, _, _) in enumerate(batch_rows):
                    pooled[row_index] = vectors[position]
    pooled.flush()
    del pooled
    partial_index = output / "index.partial.jsonl"
    by_candidate = {str(row["candidate_id"]): row for row in rows}
    with partial_index.open("w", encoding="utf-8") as handle:
        for candidate_id in sorted(by_candidate):
            row = by_candidate[candidate_id]
            document: dict[str, Any] = {
                **row,
                "vector_rows": [windows[key] for key in candidate_keys[candidate_id]],
            }
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
    partial_vectors.replace(vector_path)
    partial_index.replace(index_path)
    manifest = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r5_sequence_feature_manifest",
            "model_id": model_id,
            "layer_id": layer_id,
            "context_ms": int(config["context_ms"]),
            "sequence_count": len(rows),
            "unique_window_count": len(ordered),
            "reused_r3_window_count": reused_count,
            "extracted_window_count": len(ordered) - reused_count,
            "dimension": dimension,
            "vectors_sha256": sha256_file(vector_path),
            "index_sha256": sha256_file(index_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    print(extract_sequences(cache_root, args.model_id, threads=args.threads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
