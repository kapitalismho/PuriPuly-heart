from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil

from experiments.speaker_turn_boundary.adapters.eres2netv2 import EresEmbeddingRuntime
from experiments.speaker_turn_boundary.run_eres_sweep import ERES_CHECKPOINTS

from .phase4_signal import default_eres_root, load_inputs, read_wav, source_by_wav

WORKERS = 10
JOBS = 500
MARGIN = 0.75


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark(experiment_dir: Path, model_root: Path) -> dict[str, Any]:
    inputs = load_inputs(experiment_dir)
    source_lookup = source_by_wav(inputs)
    selected: list[tuple[str, int, int]] = []
    samples: dict[str, Any] = {}
    for wav in sorted(inputs.embedding_windows):
        source = source_lookup[wav]
        if source.public:
            continue
        samples[wav] = read_wav(source)
        for start, end in sorted(inputs.embedding_windows[wav])[:5]:
            selected.append((wav, start, end))
        if len(selected) >= 100:
            break
    selected = (selected * ((JOBS + len(selected) - 1) // len(selected)))[:JOBS]
    windows = [samples[wav][start:end] for wav, start, end in selected]
    selection_rows = [
        {"wav_sha256": wav, "start": start, "end": end} for wav, start, end in selected
    ]
    results: dict[str, Any] = {}
    process = psutil.Process()
    for checkpoint in ("E-standard", "E-w24s4ep4"):
        model = model_root / str(ERES_CHECKPOINTS[checkpoint]["onnx"])
        chunks = [windows[index::WORKERS] for index in range(WORKERS)]
        peak_rss = process.memory_info().rss
        stop = threading.Event()

        def sample_memory() -> None:
            nonlocal peak_rss
            while not stop.wait(0.01):
                peak_rss = max(peak_rss, process.memory_info().rss)

        def run_worker(values: list[Any]) -> dict[str, float]:
            load_start = time.perf_counter()
            runtime = EresEmbeddingRuntime(str(model))
            load_seconds = time.perf_counter() - load_start
            service_start = time.perf_counter()
            for value in values:
                runtime.embed(value)
            return {
                "load_seconds": load_seconds,
                "service_seconds": time.perf_counter() - service_start,
            }

        sampler = threading.Thread(target=sample_memory)
        sampler.start()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            worker_rows = list(executor.map(run_worker, chunks))
        wall_seconds = time.perf_counter() - started
        stop.set()
        sampler.join()
        throughput = JOBS / wall_seconds
        results[checkpoint] = {
            "jobs": JOBS,
            "workers": WORKERS,
            "wall_seconds_including_load": wall_seconds,
            "jobs_per_second_including_load": throughput,
            "conservative_jobs_per_second": throughput * MARGIN,
            "worker_load_seconds": [row["load_seconds"] for row in worker_rows],
            "worker_service_seconds": [row["service_seconds"] for row in worker_rows],
            "peak_process_rss_bytes": peak_rss,
            "model_sha256": sha256_file(model),
        }
    return {
        "schema_version": "turn_episode_phase4_parallel_benchmark.v1",
        "workers": WORKERS,
        "job_count_per_checkpoint": JOBS,
        "throughput_margin": MARGIN,
        "selection_sha256": sha256_bytes(
            b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in selection_rows)
        ),
        "selection_unique_count": len(set(selected)),
        "results": results,
        "generated_from": {"phase4_parallel_benchmark.py": sha256_file(Path(__file__).resolve())},
    }


def write_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    encoded = (canonical_json(body) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=default_eres_root())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parents[1]
    out = (
        args.out
        or experiment_dir / "results" / "turn_episode_v1" / "phase_4_parallel_benchmark.json"
    )
    written = write_payload(out, benchmark(experiment_dir, args.model_root))
    print(canonical_json({"path": str(out), "content_sha256": written["content_sha256"]}))


if __name__ == "__main__":
    main()
