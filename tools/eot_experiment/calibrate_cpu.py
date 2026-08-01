from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

DATASET_ID = "livekit/eot-bench-data"
DATASET_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
EOT_BENCH_REVISION = "0c6c86768921bb44eee64f9b9b3427689dceeeaa"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
LANGUAGES = ("ko", "ja", "en", "zh")
PROBES_MS = (224, 512)
TIMEOUT_MS = 800.0
SAMPLE_RATE_HZ = 16000
MIN_SILENCE_MS = 100.0
CV_SEEDS = (17, 29, 43, 71, 97)
N_FOLDS = 5
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_RESAMPLES = 1000
REAL_AUDIO_SEED = 20260802


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, percentile))


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "stddev_ms": None,
            "coefficient_of_variation": None,
        }
    mean = float(np.mean(array))
    stddev = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(np.max(array)),
        "stddev_ms": stddev,
        "coefficient_of_variation": stddev / mean if mean else None,
    }


def _installed_ort_packages() -> list[str]:
    names = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
    result = []
    for name in names:
        try:
            result.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


class CpuScorer:
    def __init__(self, model_path: Path, *, intra_op_threads: int = 2) -> None:
        import onnxruntime as ort

        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.intra_op_threads = int(intra_op_threads)
        if self.intra_op_threads < 1:
            raise ValueError("intra_op_threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        providers = list(self.session.get_providers())
        if not providers or providers[0] != "CPUExecutionProvider":
            raise RuntimeError(f"CPUExecutionProvider was not first: {providers}")
        self.providers = providers
        self.provider_options = self.session.get_provider_options()
        self.ort_version = ort.__version__
        self.available_providers = list(ort.get_available_providers())
        self.execution_mode = "ORT_SEQUENTIAL"
        self.graph_optimization_level = "ORT_ENABLE_ALL"

    def metadata(self) -> dict[str, Any]:
        return {
            "onnxruntime_version": self.ort_version,
            "available_providers": self.available_providers,
            "session_providers": self.providers,
            "provider_options": self.provider_options,
            "model_path": str(self.model_path),
            "model_filename": self.model_path.name,
            "model_sha256": _sha256(self.model_path),
            "execution_provider": self.providers[0],
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": 1,
            "execution_mode": self.execution_mode,
            "graph_optimization_level": self.graph_optimization_level,
        }

    def predict(self, audio: np.ndarray) -> tuple[float, float]:
        from puripuly_heart.core.vad.smart_turn_features import compute_whisper_log_mel_features

        started = time.perf_counter_ns()
        prepared = _prepare_smart_turn_audio(audio)
        features = compute_whisper_log_mel_features(prepared)
        outputs = self.session.run(None, {"input_features": np.expand_dims(features, axis=0)})
        if not outputs:
            raise RuntimeError("Smart Turn returned no outputs")
        score = float(np.asarray(outputs[0]).reshape(-1)[0])
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return score, elapsed_ms


def _prepare_smart_turn_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Smart Turn audio must be one-dimensional, got {audio.shape}")
    max_samples = SAMPLE_RATE_HZ * 8
    if audio.size > max_samples:
        return audio[-max_samples:].copy()
    if audio.size < max_samples:
        return np.pad(audio, (max_samples - audio.size, 0), mode="constant")
    return audio.copy()


def _decode_audio(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    import soundfile as sf

    if audio.get("bytes") is not None:
        array, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    elif audio.get("path"):
        array, sample_rate = sf.read(str(audio["path"]), dtype="float32", always_2d=False)
    else:
        raise ValueError("Dataset audio has neither bytes nor path")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if int(sample_rate) != SAMPLE_RATE_HZ:
        raise ValueError(f"Expected 16 kHz dataset audio, got {sample_rate}")
    return array.reshape(-1), int(sample_rate)


def _map_language(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"cmn", "zh-cn", "zh_cn", "chinese"}:
        return "zh"
    return raw


def _load_dataset_language(
    language: str,
    *,
    dataset_revision: str,
    split: str,
    dataset_cache_dir: Path,
):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if split != "validation":
        raise ValueError("The CPU calibration currently pins the validation parquet layout")
    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=f"data/{language}/validation-00000-of-00001.parquet",
        revision=dataset_revision,
        local_dir=str(dataset_cache_dir),
    )
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=8):
        yield from batch.to_pylist()


def generate_predictions(
    *,
    model_path: Path,
    output_dir: Path,
    languages: tuple[str, ...],
    dataset_revision: str,
    split: str,
    dataset_cache_dir: Path,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    scorer = CpuScorer(model_path, intra_op_threads=2)
    metadata = scorer.metadata()
    counts: dict[str, dict[str, int]] = {}
    for language in languages:
        rows: list[dict[str, Any]] = []
        total_calls = 0
        dataset = _load_dataset_language(
            language,
            dataset_revision=dataset_revision,
            split=split,
            dataset_cache_dir=dataset_cache_dir,
        )
        for row_number, row in enumerate(dataset):
            row_language = _map_language(row.get("language"))
            if row_language != language:
                continue
            audio, sample_rate = _decode_audio(row["audio"])
            spans = row.get("silence_spans") or []
            for span_index, span in enumerate(spans):
                start_s = float(span["start"])
                end_s = float(span["end"])
                duration_ms = (end_s - start_s) * 1000.0
                if duration_ms < MIN_SILENCE_MS - 1e-6:
                    continue
                row_id = str(row["id"])
                record: dict[str, Any] = {
                    "id": row_id,
                    "conversation_id": row_id,
                    "turn_id": row_id,
                    "span_id": f"{row_id}::{span_index}",
                    "language": language,
                    "span_index": int(span_index),
                    "label": "eot" if span_index == len(spans) - 1 else "hold",
                    "span_duration_ms": duration_ms,
                    "span_start_ms": start_s * 1000.0,
                    "span_end_ms": end_s * 1000.0,
                    "score_224": None,
                    "inference_latency_224_ms": None,
                    "score_512": None,
                    "inference_latency_512_ms": None,
                    "model_revision": MODEL_REVISION,
                    "model_sha256": metadata["model_sha256"],
                    "execution_provider": metadata["execution_provider"],
                    "intra_op_threads": metadata["intra_op_threads"],
                    "inter_op_threads": metadata["inter_op_threads"],
                }
                for probe_ms in PROBES_MS:
                    if duration_ms < probe_ms - 1e-6:
                        continue
                    end_sample = int(math.floor((start_s + probe_ms / 1000.0) * sample_rate + 1e-6))
                    snapshot = audio[:end_sample]
                    score, latency_ms = scorer.predict(snapshot)
                    record[f"score_{probe_ms}"] = score
                    record[f"inference_latency_{probe_ms}_ms"] = latency_ms
                    total_calls += 1
                rows.append(record)
            if row_number and row_number % 25 == 0:
                print(
                    f"{language}: rows={row_number}, spans={len(rows)}, inference_calls={total_calls}",
                    flush=True,
                )
        if not rows:
            raise RuntimeError(f"No eligible rows generated for {language}")
        output_path = output_dir / f"cpu_predictions_{language}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, output_path, compression="zstd")
        counts[language] = {
            "spans": len(rows),
            "eot": sum(row["label"] == "eot" for row in rows),
            "hold": sum(row["label"] == "hold" for row in rows),
            "score_224": sum(row["score_224"] is not None for row in rows),
            "score_512": sum(row["score_512"] is not None for row in rows),
            "inference_calls": total_calls,
        }
    result = {
        "dataset": {
            "id": DATASET_ID,
            "revision": dataset_revision,
            "split": split,
            "min_silence_ms": MIN_SILENCE_MS,
        },
        "model": metadata | {"revision": MODEL_REVISION, "filename": MODEL_FILENAME},
        "languages": counts,
    }
    _write_json(output_dir / "prediction_generation.json", result)
    return result


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    array, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if int(sample_rate) != SAMPLE_RATE_HZ:
        raise ValueError(f"Expected 16 kHz WAV: {path} has {sample_rate}")
    return array.reshape(-1), int(sample_rate)


def _rss_mb(process: Any) -> float:
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _benchmark_calls(
    scorer: CpuScorer,
    inputs: list[tuple[np.ndarray, dict[str, Any]]],
    *,
    warmups: int,
    measured: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import psutil

    process = psutil.Process(os.getpid())
    for _ in range(warmups):
        scorer.predict(inputs[_ % len(inputs)][0])
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    peak_rss = _rss_mb(process)
    peak_threads = process.num_threads()
    scores: list[float] = []
    for index in range(measured):
        audio, metadata = inputs[index % len(inputs)]
        score, latency_ms = scorer.predict(audio)
        latencies.append(latency_ms)
        scores.append(score)
        peak_rss = max(peak_rss, _rss_mb(process))
        peak_threads = max(peak_threads, process.num_threads())
        rows.append(metadata | {"call_index": index, "latency_ms": latency_ms, "score": score})
    result = _stats(latencies)
    result.update(
        {
            "score_mean": float(np.mean(scores)),
            "memory_before_mb": None,
            "peak_process_rss_mb": peak_rss,
            "peak_process_thread_count": peak_threads,
        }
    )
    return result, rows


def _synthetic_input(duration_s: float) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE_HZ))
    rng = np.random.default_rng(1000 + int(duration_s * 10))
    return rng.standard_normal(samples).astype(np.float32)


def _real_audio_inputs(
    language: str,
    path: Path,
    *,
    count: int,
    seed: int,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    audio, _ = _load_wav(path)
    rng = np.random.default_rng(seed + sum(ord(char) for char in language))
    durations = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    inputs = []
    for index in range(count):
        duration_s = float(rng.choice(durations))
        length = int(round(duration_s * SAMPLE_RATE_HZ))
        if audio.size <= length:
            start = 0
            snapshot = audio
        else:
            start = int(rng.integers(0, audio.size - length + 1))
            snapshot = audio[start : start + length]
        inputs.append(
            (
                snapshot.copy(),
                {
                    "language": language,
                    "duration_s": duration_s,
                    "input_start_sample": start,
                    "input_samples": int(snapshot.size),
                },
            )
        )
    return inputs


def _cold_worker(model_path: Path) -> None:
    started = time.perf_counter()
    load_started = time.perf_counter()
    scorer = CpuScorer(model_path, intra_op_threads=2)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    infer_started = time.perf_counter()
    score, first_inference_ms = scorer.predict(_synthetic_input(8.0))
    first_inference_wall_ms = (time.perf_counter() - infer_started) * 1000.0
    result = {
        "model_load_ms": model_load_ms,
        "first_inference_ms": first_inference_ms,
        "first_inference_wall_ms": first_inference_wall_ms,
        "total_cold_start_ms": (time.perf_counter() - started) * 1000.0,
        "score": score,
        "provider": scorer.metadata(),
    }
    print(json.dumps(result, default=_json_default), flush=True)


def _run_cold_initialization(model_path: Path, *, runs: int) -> dict[str, Any]:
    rows = []
    for _ in range(runs):
        started = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--cold-worker",
                "--model",
                str(model_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        parent_wall_ms = (time.perf_counter() - started) * 1000.0
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Cold worker returned no JSON")
        row = json.loads(lines[-1])
        row["parent_process_wall_ms"] = parent_wall_ms
        rows.append(row)
    return {
        "runs": runs,
        "model_load": _stats(row["model_load_ms"] for row in rows),
        "first_inference": _stats(row["first_inference_ms"] for row in rows),
        "total_cold_start": _stats(row["total_cold_start_ms"] for row in rows),
        "parent_process_wall": _stats(row["parent_process_wall_ms"] for row in rows),
        "samples": rows,
    }


def run_latency_benchmarks(
    *,
    model_path: Path,
    output_dir: Path,
    real_audio_root: Path,
    languages: tuple[str, ...],
    warmups: int,
    measured: int,
    real_calls_per_language: int,
    cold_runs: int,
) -> dict[str, Any]:
    import psutil

    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = CpuScorer(model_path, intra_op_threads=2)
    before_session = _rss_mb(psutil.Process(os.getpid()))
    provider = scorer.metadata()
    after_session = _rss_mb(psutil.Process(os.getpid()))
    synthetic: list[dict[str, Any]] = []
    for duration_s in (0.5, 1.0, 2.0, 4.0, 8.0):
        stats, _ = _benchmark_calls(
            scorer,
            [(_synthetic_input(duration_s), {"duration_s": duration_s, "input_kind": "synthetic"})],
            warmups=warmups,
            measured=measured,
        )
        stats["duration_s"] = duration_s
        stats["input_kind"] = "synthetic"
        synthetic.append(stats)

    real_rows: list[dict[str, Any]] = []
    real_summary: list[dict[str, Any]] = []
    for language in languages:
        inputs = _real_audio_inputs(
            language,
            real_audio_root / f"{language}.wav",
            count=real_calls_per_language,
            seed=REAL_AUDIO_SEED,
        )
        stats, rows = _benchmark_calls(
            scorer, inputs, warmups=warmups, measured=real_calls_per_language
        )
        stats.update(
            {"language": language, "input_kind": "real_audio", "calls": real_calls_per_language}
        )
        real_summary.append(stats)
        real_rows.extend(rows)
    with (output_dir / "cpu_latency_real_audio.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = sorted({key for row in real_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(real_rows)

    cold = _run_cold_initialization(model_path, runs=cold_runs)
    two_thread = {
        "configuration": provider,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "process_env": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "warmup_calls": warmups,
        "measured_calls": measured,
        "real_calls_per_language": real_calls_per_language,
        "cold_initialization": cold,
        "rss_before_session_mb": before_session,
        "rss_after_session_mb": after_session,
        "rss_increase_mb": after_session - before_session,
        "synthetic": synthetic,
        "real_audio": real_summary,
    }
    _write_json(output_dir / "cpu_latency_2thread.json", two_thread)

    control = CpuScorer(model_path, intra_op_threads=1)
    control_synthetic, _ = _benchmark_calls(
        control,
        [(_synthetic_input(8.0), {"duration_s": 8.0, "input_kind": "synthetic"})],
        warmups=warmups,
        measured=measured,
    )
    control_real = []
    for language in languages:
        inputs = _real_audio_inputs(
            language,
            real_audio_root / f"{language}.wav",
            count=real_calls_per_language,
            seed=REAL_AUDIO_SEED,
        )
        stats, _ = _benchmark_calls(
            control, inputs, warmups=warmups, measured=real_calls_per_language
        )
        stats.update(
            {"language": language, "input_kind": "real_audio", "calls": real_calls_per_language}
        )
        control_real.append(stats)
    one_thread = {
        "configuration": control.metadata(),
        "environment": two_thread["environment"],
        "warmup_calls": warmups,
        "measured_calls": measured,
        "real_calls_per_language": real_calls_per_language,
        "synthetic_8s": control_synthetic,
        "real_audio": control_real,
    }
    _write_json(output_dir / "cpu_latency_1thread_control.json", one_thread)
    return {"two_thread": two_thread, "one_thread": one_thread}


def _validate_prediction_rows(rows: list[dict[str, Any]], *, language: str) -> None:
    seen_span_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        row_language = _map_language(row.get("language"))
        if row_language != language:
            raise ValueError(
                f"prediction row {row_number}: language {row_language!r} != {language!r}"
            )
        label = str(row.get("label", "")).strip().lower()
        if label not in {"hold", "eot"}:
            raise ValueError(f"prediction row {row_number}: invalid label {label!r}")
        duration = row.get("span_duration_ms")
        if not _finite(duration) or float(duration) < 0.0:
            raise ValueError(f"prediction row {row_number}: invalid span_duration_ms")
        span_id = str(row.get("span_id", row.get("id", row_number)))
        if span_id in seen_span_ids:
            raise ValueError(f"duplicate prediction row for span {span_id!r}")
        seen_span_ids.add(span_id)
        for probe_ms in PROBES_MS:
            score_key = f"score_{probe_ms}"
            latency_key = f"inference_latency_{probe_ms}_ms"
            score_present = _finite(row.get(score_key))
            latency_present = _finite(row.get(latency_key))
            survives = float(duration) >= probe_ms - 1e-6
            if survives and (not score_present or not latency_present):
                raise ValueError(
                    f"prediction row {row_number}: {score_key} and {latency_key} are required"
                )
            if not survives and (score_present or latency_present):
                raise ValueError(
                    f"prediction row {row_number}: {score_key} is present before its probe"
                )


def _load_prediction_rows(output_dir: Path, language: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = output_dir / f"cpu_predictions_{language}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    _validate_prediction_rows(rows, language=language)
    return rows


def _array_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "duration": np.asarray([float(row["span_duration_ms"]) for row in rows], dtype=np.float64),
        "hold": np.asarray([row["label"] == "hold" for row in rows], dtype=bool),
        "eot": np.asarray([row["label"] == "eot" for row in rows], dtype=bool),
        "score224": np.asarray(
            [float(row["score_224"]) if _finite(row.get("score_224")) else np.nan for row in rows],
            dtype=np.float64,
        ),
        "score512": np.asarray(
            [float(row["score_512"]) if _finite(row.get("score_512")) else np.nan for row in rows],
            dtype=np.float64,
        ),
        "lat224": np.asarray(
            [
                (
                    float(row["inference_latency_224_ms"])
                    if _finite(row.get("inference_latency_224_ms"))
                    else np.nan
                )
                for row in rows
            ],
            dtype=np.float64,
        ),
        "lat512": np.asarray(
            [
                (
                    float(row["inference_latency_512_ms"])
                    if _finite(row.get("inference_latency_512_ms"))
                    else np.nan
                )
                for row in rows
            ],
            dtype=np.float64,
        ),
        "groups": np.asarray([str(row["turn_id"]) for row in rows], dtype=object),
    }


def _policy_trace(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float | None = None,
    threshold512: float | None = None,
    *,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = _array_data(rows) if array_data is None else array_data
    duration = data["duration"]
    score224 = data["score224"]
    score512 = data["score512"]
    lat224 = data["lat224"]
    lat512 = data["lat512"]
    has224 = np.isfinite(score224) & np.isfinite(lat224)
    has512 = np.isfinite(score512) & np.isfinite(lat512)
    arrival224 = 224.0 + np.where(np.isfinite(lat224), lat224, 0.0)
    first_valid = np.zeros(len(rows), dtype=bool)
    second_scheduled = np.zeros(len(rows), dtype=bool)
    second_valid = np.zeros(len(rows), dtype=bool)
    stale224 = np.zeros(len(rows), dtype=bool)
    stale512 = np.zeros(len(rows), dtype=bool)
    overlap = np.zeros(len(rows), dtype=bool)

    if policy == "B0":
        decision = np.full(len(rows), 512.0, dtype=np.float64)
        probe = np.full(len(rows), "512ms_fixed", dtype=object)
    elif policy == "B1":
        decision = np.full(len(rows), TIMEOUT_MS, dtype=np.float64)
        probe = np.full(len(rows), "timeout", dtype=object)
    elif policy in {"P1", "P2", "P3"}:
        if threshold224 is None:
            raise ValueError(f"{policy} requires threshold224")
        first_valid = (
            has224 & (duration > arrival224) & (arrival224 < TIMEOUT_MS) & (score224 > threshold224)
        )
        stale224 = has224 & ((duration <= arrival224) | (arrival224 >= TIMEOUT_MS))
        decision = np.full(len(rows), TIMEOUT_MS, dtype=np.float64)
        probe = np.full(len(rows), "timeout", dtype=object)
        decision[first_valid] = arrival224[first_valid]
        probe[first_valid] = "224ms"
        if policy in {"P2", "P3"}:
            if threshold512 is None:
                raise ValueError(f"{policy} requires threshold512")
            start512 = np.maximum(512.0, arrival224)
            arrival512 = start512 + np.where(np.isfinite(lat512), lat512, 0.0)
            second_scheduled = has512 & ~first_valid
            overlap = second_scheduled & (arrival224 > 512.0)
            second_valid = (
                second_scheduled
                & (duration > arrival512)
                & (arrival512 < TIMEOUT_MS)
                & (score512 > threshold512)
            )
            stale512 = second_scheduled & ((duration <= arrival512) | (arrival512 >= TIMEOUT_MS))
            decision[second_valid] = arrival512[second_valid]
            probe[second_valid] = "512ms"
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return {
        "decision_ms": decision,
        "probe": probe,
        "first_valid": first_valid,
        "second_scheduled": second_scheduled,
        "second_valid": second_valid,
        "stale224": stale224,
        "stale512": stale512,
        "overlap": overlap,
    }


def _trace_metrics(rows: list[dict[str, Any]], trace: dict[str, Any]) -> dict[str, Any]:
    data = _array_data(rows)
    decision = trace["decision_ms"]
    hold = data["hold"]
    eot = data["eot"]
    false_cut = hold & (data["duration"] > decision)
    eligible_hold = hold & (data["duration"] > 224.0)
    eot_latencies = decision[eot]
    groups = data["groups"]
    group_to_false: dict[str, int] = defaultdict(int)
    group_to_any: dict[str, bool] = defaultdict(bool)
    for group, is_false in zip(groups, false_cut, strict=True):
        group_to_any[str(group)] = True
        if is_false:
            group_to_false[str(group)] += 1
    affected = [group for group, count in group_to_false.items() if count > 0]
    all_groups = len(group_to_any)
    false_count = int(false_cut.sum())
    hold_count = int(hold.sum())
    eligible_count = int(eligible_hold.sum())
    hard_timeout = decision >= TIMEOUT_MS - 1e-6
    scheduled_first = data["duration"] >= 224.0
    inference_attempts = int(scheduled_first.sum() + trace["second_scheduled"].sum())
    return {
        "n_spans": len(rows),
        "eot_spans": int(eot.sum()),
        "hold_spans": hold_count,
        "false_cutoffs": false_count,
        "false_cutoff_rate": false_count / hold_count if hold_count else 0.0,
        "eligible_hold_spans": eligible_count,
        "eligible_false_cutoffs": int((false_cut & eligible_hold).sum()),
        "eligible_false_cutoff_rate": (
            int((false_cut & eligible_hold).sum()) / eligible_count if eligible_count else 0.0
        ),
        "mean_endpoint_latency_ms": float(np.mean(eot_latencies)) if eot_latencies.size else None,
        "p50_endpoint_latency_ms": (
            float(np.percentile(eot_latencies, 50)) if eot_latencies.size else None
        ),
        "p90_endpoint_latency_ms": (
            float(np.percentile(eot_latencies, 90)) if eot_latencies.size else None
        ),
        "p95_endpoint_latency_ms": (
            float(np.percentile(eot_latencies, 95)) if eot_latencies.size else None
        ),
        "p99_endpoint_latency_ms": (
            float(np.percentile(eot_latencies, 99)) if eot_latencies.size else None
        ),
        "acceptance_224_rate": float(trace["first_valid"].mean()) if len(rows) else 0.0,
        "acceptance_512_rate": float(trace["second_valid"].mean()) if len(rows) else 0.0,
        "hard_timeout_rate": float(hard_timeout.mean()) if len(rows) else 0.0,
        "eot_early_detection_rate": (
            float(((eot) & (decision < TIMEOUT_MS)).sum() / eot.sum()) if eot.sum() else 0.0
        ),
        "probe_overlap_rate": (
            float(trace["overlap"].sum() / trace["second_scheduled"].sum())
            if trace["second_scheduled"].sum()
            else 0.0
        ),
        "probe_overlap_count": int(trace["overlap"].sum()),
        "second_scheduled_count": int(trace["second_scheduled"].sum()),
        "stale_result_rate": (
            float((trace["stale224"].sum() + trace["stale512"].sum()) / inference_attempts)
            if inference_attempts
            else 0.0
        ),
        "stale_result_count": int(trace["stale224"].sum() + trace["stale512"].sum()),
        "turns": all_groups,
        "turns_with_false_cutoff": len(affected),
        "turn_fragmentation_rate": len(affected) / all_groups if all_groups else 0.0,
        "false_splits_per_100_turns": false_count / all_groups * 100.0 if all_groups else 0.0,
        "mean_false_cutoffs_per_affected_turn": false_count / len(affected) if affected else 0.0,
    }


def simulate_policy(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float | None = None,
    threshold512: float | None = None,
) -> dict[str, Any]:
    trace = _policy_trace(rows, policy, threshold224, threshold512)
    return _trace_metrics(rows, trace)


def _threshold_grid() -> list[float]:
    values = [i / 100.0 for i in range(91)]
    values.extend(round(0.90 + i * 0.002, 6) for i in range(1, 46))
    values.extend(round(0.99 + i * 0.0005, 6) for i in range(1, 21))
    return sorted(set(value for value in values if value <= 1.0))


def _candidate_thresholds(
    rows: list[dict[str, Any]], key: str, *, exact_limit: int = 2500
) -> tuple[list[float], str]:
    values = sorted({round(float(row[key]), 9) for row in rows if _finite(row.get(key))})
    if len(values) <= exact_limit:
        return sorted(set([0.0, 1.0, *values])), "observed_training_scores"
    return _threshold_grid(), "coarse_to_fine_grid"


def _relative_reduction(baseline: float, value: float) -> float:
    return (baseline - value) / baseline if baseline else 0.0


def _candidate_is_valid(
    metrics: dict[str, Any], baseline: dict[str, Any], target_reduction: float
) -> bool:
    return (
        _relative_reduction(
            float(baseline["false_cutoff_rate"]), float(metrics["false_cutoff_rate"])
        )
        >= target_reduction - 1e-12
        and float(metrics["false_cutoff_rate"]) <= float(baseline["false_cutoff_rate"]) + 1e-12
        and float(metrics["mean_endpoint_latency_ms"] or math.inf) <= 600.0
        and float(metrics["p50_endpoint_latency_ms"] or math.inf) <= 600.0
        and float(metrics["hard_timeout_rate"]) <= 0.25 + 1e-12
    )


def _candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["metrics"]
    return (
        float(metrics["mean_endpoint_latency_ms"] or math.inf),
        float(metrics["false_cutoff_rate"]),
        -float(metrics["eot_early_detection_rate"]),
        -float(item.get("threshold224", 0.0)),
    )


def _candidate_metrics(
    rows: list[dict[str, Any]],
    policy: str,
    threshold224: float,
    threshold512: float | None = None,
    *,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = _array_data(rows) if array_data is None else array_data
    trace = _policy_trace(rows, policy, threshold224, threshold512, array_data=data)
    eot = data["eot"]
    hold = data["hold"]
    decision = trace["decision_ms"]
    eot_latencies = decision[eot]
    false_cutoffs = int((hold & (data["duration"] > decision)).sum())
    timeout = decision >= TIMEOUT_MS - 1e-6
    return {
        "false_cutoff_rate": false_cutoffs / int(hold.sum()) if hold.sum() else 0.0,
        "mean_endpoint_latency_ms": (float(np.mean(eot_latencies)) if eot_latencies.size else None),
        "p50_endpoint_latency_ms": (
            float(np.percentile(eot_latencies, 50)) if eot_latencies.size else None
        ),
        "hard_timeout_rate": float(timeout.mean()) if len(rows) else 0.0,
        "eot_early_detection_rate": (
            float((eot & (decision < TIMEOUT_MS)).sum() / eot.sum()) if eot.sum() else 0.0
        ),
    }


def _enumerate_candidates(
    rows: list[dict[str, Any]], policy: str
) -> tuple[list[dict[str, Any]], str]:
    array_data = _array_data(rows)
    candidates224, source224 = _candidate_thresholds(rows, "score_224")
    if policy == "P1":
        return [
            {
                "threshold224": threshold,
                "threshold512": None,
                "metrics": _candidate_metrics(rows, policy, threshold, array_data=array_data),
            }
            for threshold in candidates224
        ], source224
    candidates512, source512 = _candidate_thresholds(rows, "score_512")
    if policy == "P2":
        values = sorted(set(candidates224 + candidates512))
        return [
            {
                "threshold224": threshold,
                "threshold512": threshold,
                "metrics": _candidate_metrics(
                    rows, policy, threshold, threshold, array_data=array_data
                ),
            }
            for threshold in values
        ], f"shared_{source224}_{source512}"
    if policy == "P3":
        coarse = [
            value
            for value in _threshold_grid()
            if value in set(round(candidate, 6) for candidate in _threshold_grid())
        ]
        initial: list[dict[str, Any]] = []
        for threshold224 in coarse:
            for threshold512 in coarse:
                if threshold512 + 1e-12 < threshold224:
                    continue
                initial.append(
                    {
                        "threshold224": threshold224,
                        "threshold512": threshold512,
                        "metrics": _candidate_metrics(
                            rows,
                            policy,
                            threshold224,
                            threshold512,
                            array_data=array_data,
                        ),
                    }
                )
        initial.sort(key=_candidate_sort_key)
        neighborhoods = set()
        for item in initial[:20]:
            for base224 in (item["threshold224"],):
                for base512 in (item["threshold512"],):
                    for offset224 in range(-10, 11):
                        for offset512 in range(-10, 11):
                            threshold224 = round(base224 + offset224 * 0.002, 6)
                            threshold512 = round(base512 + offset512 * 0.002, 6)
                            if 0.0 <= threshold224 <= threshold512 <= 1.0:
                                neighborhoods.add((threshold224, threshold512))
        refined = []
        for threshold224, threshold512 in neighborhoods:
            refined.append(
                {
                    "threshold224": threshold224,
                    "threshold512": threshold512,
                    "metrics": _candidate_metrics(
                        rows,
                        policy,
                        threshold224,
                        threshold512,
                        array_data=array_data,
                    ),
                }
            )
        return refined, f"{source224}_{source512}_coarse_to_fine"
    raise ValueError(policy)


def _select_candidate(
    candidates: list[dict[str, Any]], baseline: dict[str, Any], target_reduction: float
) -> dict[str, Any] | None:
    valid = [
        item
        for item in candidates
        if _candidate_is_valid(item["metrics"], baseline, target_reduction)
    ]
    if not valid:
        return None
    return min(valid, key=_candidate_sort_key)


def _group_splits(
    rows: list[dict[str, Any]], seed: int
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]]:
    groups = sorted({str(row["turn_id"]) for row in rows})
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    assignments = {group: index % N_FOLDS for index, group in enumerate(shuffled)}
    result = []
    for fold in range(N_FOLDS):
        test_groups = {group for group, assigned in assignments.items() if assigned == fold}
        train = [row for row in rows if str(row["turn_id"]) not in test_groups]
        test = [row for row in rows if str(row["turn_id"]) in test_groups]
        result.append((fold, train, test))
    return result


METRIC_FIELDS = (
    "false_cutoff_rate",
    "eligible_false_cutoff_rate",
    "mean_endpoint_latency_ms",
    "p50_endpoint_latency_ms",
    "p90_endpoint_latency_ms",
    "p95_endpoint_latency_ms",
    "p99_endpoint_latency_ms",
    "acceptance_224_rate",
    "acceptance_512_rate",
    "hard_timeout_rate",
    "eot_early_detection_rate",
    "probe_overlap_rate",
    "stale_result_rate",
    "turn_fragmentation_rate",
    "false_splits_per_100_turns",
    "mean_false_cutoffs_per_affected_turn",
)


def _metric_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    return {field: metrics.get(field) for field in METRIC_FIELDS}


def _empty_metric_columns() -> dict[str, Any]:
    return {field: None for field in METRIC_FIELDS}


def _cv_metric_row(
    *,
    language: str,
    seed: int,
    fold: int,
    target: str,
    policy: str,
    selection_kind: str,
    train_rows: int,
    test_rows: int,
    train_groups: int,
    test_groups: int,
    threshold224: float | None,
    threshold512: float | None,
    threshold_source: str | None,
    metrics: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    row = {
        "language": language,
        "seed": seed,
        "fold": fold,
        "target": target,
        "policy": policy,
        "selection_kind": selection_kind,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "threshold224": threshold224,
        "threshold512": threshold512,
        "threshold_source": threshold_source,
        "status": status,
    }
    row.update(_metric_columns(metrics) if metrics is not None else _empty_metric_columns())
    return row


def _matched_candidate(
    candidates: list[dict[str, Any]], reference_cutoff: float
) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            abs(float(item["metrics"]["false_cutoff_rate"]) - reference_cutoff),
            float(item["metrics"].get("mean_endpoint_latency_ms") or math.inf),
        ),
    )


def _increment_row(
    *,
    language: str,
    seed: int,
    fold: int,
    target: str,
    comparison: str,
    train_reference: dict[str, Any],
    train_candidate: dict[str, Any],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_policy = "P1"
    candidate_policy = "P2" if comparison == "P2_vs_P1" else "P3"
    reference_trace = _policy_trace(
        test_rows,
        reference_policy,
        train_reference.get("threshold224"),
        train_reference.get("threshold512"),
    )
    candidate_trace = _policy_trace(
        test_rows,
        candidate_policy,
        train_candidate.get("threshold224"),
        train_candidate.get("threshold512"),
    )
    reference_metrics = _trace_metrics(test_rows, reference_trace)
    candidate_metrics = _trace_metrics(test_rows, candidate_trace)
    data = _array_data(test_rows)
    recovered = data["eot"] & ~reference_trace["first_valid"] & candidate_trace["second_valid"]
    reference_false = data["hold"] & (data["duration"] > reference_trace["decision_ms"])
    candidate_false = data["hold"] & (data["duration"] > candidate_trace["decision_ms"])
    new_false = candidate_false & ~reference_false & (candidate_trace["probe"] == "512ms")
    return {
        "language": language,
        "seed": seed,
        "fold": fold,
        "target": target,
        "comparison": comparison,
        "reference_threshold224": train_reference.get("threshold224"),
        "reference_threshold512": train_reference.get("threshold512"),
        "candidate_threshold224": train_candidate.get("threshold224"),
        "candidate_threshold512": train_candidate.get("threshold512"),
        "train_reference_false_cutoff_rate": train_reference["metrics"]["false_cutoff_rate"],
        "train_candidate_false_cutoff_rate": train_candidate["metrics"]["false_cutoff_rate"],
        "heldout_reference_false_cutoff_rate": reference_metrics["false_cutoff_rate"],
        "heldout_candidate_false_cutoff_rate": candidate_metrics["false_cutoff_rate"],
        "false_cutoff_delta": candidate_metrics["false_cutoff_rate"]
        - reference_metrics["false_cutoff_rate"],
        "true_eot_recovered_at_512": int(recovered.sum()),
        "new_false_cutoffs_at_512": int(new_false.sum()),
        "mean_latency_change_ms": (candidate_metrics["mean_endpoint_latency_ms"] or 0.0)
        - (reference_metrics["mean_endpoint_latency_ms"] or 0.0),
        "timeout_rate_change_pp": (
            candidate_metrics["hard_timeout_rate"] - reference_metrics["hard_timeout_rate"]
        )
        * 100.0,
        "turn_fragmentation_change_pp": (
            candidate_metrics["turn_fragmentation_rate"]
            - reference_metrics["turn_fragmentation_rate"]
        )
        * 100.0,
        "reference_mean_latency_ms": reference_metrics["mean_endpoint_latency_ms"],
        "candidate_mean_latency_ms": candidate_metrics["mean_endpoint_latency_ms"],
        "reference_timeout_rate": reference_metrics["hard_timeout_rate"],
        "candidate_timeout_rate": candidate_metrics["hard_timeout_rate"],
    }


def _p3_vs_p2_increment(
    *,
    language: str,
    seed: int,
    fold: int,
    target: str,
    p2_candidate: dict[str, Any],
    p3_candidate: dict[str, Any],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    p2_trace = _policy_trace(
        test_rows, "P2", p2_candidate["threshold224"], p2_candidate["threshold512"]
    )
    p3_trace = _policy_trace(
        test_rows, "P3", p3_candidate["threshold224"], p3_candidate["threshold512"]
    )
    p2_metrics = _trace_metrics(test_rows, p2_trace)
    p3_metrics = _trace_metrics(test_rows, p3_trace)
    data = _array_data(test_rows)
    p2_false = data["hold"] & (data["duration"] > p2_trace["decision_ms"])
    p3_false = data["hold"] & (data["duration"] > p3_trace["decision_ms"])
    new_false = p3_false & ~p2_false & (p3_trace["probe"] == "512ms")
    return {
        "language": language,
        "seed": seed,
        "fold": fold,
        "target": target,
        "comparison": "P3_vs_P2",
        "reference_threshold224": p2_candidate["threshold224"],
        "reference_threshold512": p2_candidate["threshold512"],
        "candidate_threshold224": p3_candidate["threshold224"],
        "candidate_threshold512": p3_candidate["threshold512"],
        "train_reference_false_cutoff_rate": p2_candidate["metrics"]["false_cutoff_rate"],
        "train_candidate_false_cutoff_rate": p3_candidate["metrics"]["false_cutoff_rate"],
        "heldout_reference_false_cutoff_rate": p2_metrics["false_cutoff_rate"],
        "heldout_candidate_false_cutoff_rate": p3_metrics["false_cutoff_rate"],
        "false_cutoff_delta": p3_metrics["false_cutoff_rate"] - p2_metrics["false_cutoff_rate"],
        "true_eot_recovered_at_512": int(
            (p3_trace["second_valid"] & ~p2_trace["second_valid"]).sum()
        ),
        "new_false_cutoffs_at_512": int(new_false.sum()),
        "mean_latency_change_ms": (p3_metrics["mean_endpoint_latency_ms"] or 0.0)
        - (p2_metrics["mean_endpoint_latency_ms"] or 0.0),
        "timeout_rate_change_pp": (
            p3_metrics["hard_timeout_rate"] - p2_metrics["hard_timeout_rate"]
        )
        * 100.0,
        "turn_fragmentation_change_pp": (
            p3_metrics["turn_fragmentation_rate"] - p2_metrics["turn_fragmentation_rate"]
        )
        * 100.0,
        "reference_mean_latency_ms": p2_metrics["mean_endpoint_latency_ms"],
        "candidate_mean_latency_ms": p3_metrics["mean_endpoint_latency_ms"],
        "reference_timeout_rate": p2_metrics["hard_timeout_rate"],
        "candidate_timeout_rate": p3_metrics["hard_timeout_rate"],
    }


def cross_validate(
    rows_by_language: dict[str, list[dict[str, Any]]], *, output_dir: Path
) -> dict[str, Any]:
    cv_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    policies = ("P1", "P2", "P3")
    targets = (("low_latency", 0.20), ("stability", 0.35))
    for language, rows in rows_by_language.items():
        for seed in CV_SEEDS:
            for fold, train_rows, test_rows in _group_splits(rows, seed):
                train_groups = len({str(row["turn_id"]) for row in train_rows})
                test_groups = len({str(row["turn_id"]) for row in test_rows})
                train_b0 = simulate_policy(train_rows, "B0")
                test_b0 = simulate_policy(test_rows, "B0")
                train_b1 = simulate_policy(train_rows, "B1")
                test_b1 = simulate_policy(test_rows, "B1")
                candidate_cache: dict[str, tuple[list[dict[str, Any]], str]] = {
                    policy: _enumerate_candidates(train_rows, policy) for policy in policies
                }
                for target_name, target_reduction in targets:
                    for policy, train_metrics, test_metrics in (
                        ("B0", train_b0, test_b0),
                        ("B1", train_b1, test_b1),
                    ):
                        cv_rows.append(
                            _cv_metric_row(
                                language=language,
                                seed=seed,
                                fold=fold,
                                target=target_name,
                                policy=policy,
                                selection_kind="baseline",
                                train_rows=len(train_rows),
                                test_rows=len(test_rows),
                                train_groups=train_groups,
                                test_groups=test_groups,
                                threshold224=None,
                                threshold512=None,
                                threshold_source=None,
                                metrics=test_metrics,
                                status="available",
                            )
                        )
                    selections: dict[str, dict[str, Any] | None] = {}
                    for policy in policies:
                        candidates, source = candidate_cache[policy]
                        selected = _select_candidate(candidates, train_b0, target_reduction)
                        selections[policy] = selected
                        if selected is None:
                            cv_rows.append(
                                _cv_metric_row(
                                    language=language,
                                    seed=seed,
                                    fold=fold,
                                    target=target_name,
                                    policy=policy,
                                    selection_kind="selected",
                                    train_rows=len(train_rows),
                                    test_rows=len(test_rows),
                                    train_groups=train_groups,
                                    test_groups=test_groups,
                                    threshold224=None,
                                    threshold512=None,
                                    threshold_source=source,
                                    metrics=None,
                                    status="unavailable",
                                )
                            )
                            continue
                        selected_test = simulate_policy(
                            test_rows,
                            policy,
                            selected["threshold224"],
                            selected["threshold512"],
                        )
                        selected_record = selected | {
                            "language": language,
                            "seed": seed,
                            "fold": fold,
                            "target": target_name,
                            "policy": policy,
                            "threshold_source": source,
                            "test_metrics": selected_test,
                        }
                        all_selected.append(selected_record)
                        cv_rows.append(
                            _cv_metric_row(
                                language=language,
                                seed=seed,
                                fold=fold,
                                target=target_name,
                                policy=policy,
                                selection_kind="selected",
                                train_rows=len(train_rows),
                                test_rows=len(test_rows),
                                train_groups=train_groups,
                                test_groups=test_groups,
                                threshold224=selected["threshold224"],
                                threshold512=selected["threshold512"],
                                threshold_source=source,
                                metrics=selected_test,
                                status="available",
                            )
                        )

                    p1_selected = selections["P1"]
                    p1_reference = p1_selected or _matched_candidate(
                        candidate_cache["P1"][0], train_b0["false_cutoff_rate"]
                    )
                    if p1_reference is None:
                        continue
                    p1_cutoff = float(p1_reference["metrics"]["false_cutoff_rate"])
                    reference_selection = (
                        "target_selected" if p1_selected is not None else "fallback_b0_match"
                    )
                    if p1_selected is None:
                        cv_rows.append(
                            _cv_metric_row(
                                language=language,
                                seed=seed,
                                fold=fold,
                                target=target_name,
                                policy="P1",
                                selection_kind="fallback_reference",
                                train_rows=len(train_rows),
                                test_rows=len(test_rows),
                                train_groups=train_groups,
                                test_groups=test_groups,
                                threshold224=p1_reference["threshold224"],
                                threshold512=None,
                                threshold_source="training_b0_cutoff_match",
                                metrics=simulate_policy(
                                    test_rows, "P1", p1_reference["threshold224"]
                                ),
                                status="available",
                            )
                        )
                    matched: dict[str, dict[str, Any] | None] = {"P1": p1_reference}
                    for policy in ("P2", "P3"):
                        candidates, _source = candidate_cache[policy]
                        matched[policy] = _matched_candidate(candidates, p1_cutoff)
                        if matched[policy] is not None:
                            increment = _increment_row(
                                language=language,
                                seed=seed,
                                fold=fold,
                                target=target_name,
                                comparison=f"{policy}_vs_P1",
                                train_reference=p1_reference,
                                train_candidate=matched[policy],
                                test_rows=test_rows,
                            )
                            increment["reference_selection"] = reference_selection
                            increment_rows.append(increment)
                            cv_rows.append(
                                _cv_metric_row(
                                    language=language,
                                    seed=seed,
                                    fold=fold,
                                    target=target_name,
                                    policy=policy,
                                    selection_kind="matched_to_P1",
                                    train_rows=len(train_rows),
                                    test_rows=len(test_rows),
                                    train_groups=train_groups,
                                    test_groups=test_groups,
                                    threshold224=matched[policy]["threshold224"],
                                    threshold512=matched[policy]["threshold512"],
                                    threshold_source="training_cutoff_match",
                                    metrics=simulate_policy(
                                        test_rows,
                                        policy,
                                        matched[policy]["threshold224"],
                                        matched[policy]["threshold512"],
                                    ),
                                    status="available",
                                )
                            )
                    if matched["P2"] is not None and matched["P3"] is not None:
                        increment = _p3_vs_p2_increment(
                            language=language,
                            seed=seed,
                            fold=fold,
                            target=target_name,
                            p2_candidate=matched["P2"],
                            p3_candidate=matched["P3"],
                            test_rows=test_rows,
                        )
                        increment["reference_selection"] = reference_selection
                        increment_rows.append(increment)

    _write_csv(output_dir / "cv_results_all.csv", cv_rows)
    for language in rows_by_language:
        _write_csv(
            output_dir / f"cv_results_{language}.csv",
            [row for row in cv_rows if row["language"] == language],
        )
    _write_csv(output_dir / "probe_512_increment.csv", increment_rows)
    threshold_stability = _threshold_stability(cv_rows)
    _write_csv(output_dir / "threshold_stability.csv", threshold_stability)
    turn_fragmentation = _aggregate_turn_fragmentation(cv_rows)
    _write_csv(output_dir / "turn_fragmentation.csv", turn_fragmentation)
    return {
        "cv_rows": cv_rows,
        "increment_rows": increment_rows,
        "threshold_stability": threshold_stability,
        "turn_fragmentation": turn_fragmentation,
        "selected": all_selected,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _threshold_stability(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["status"] != "available" or row["selection_kind"] not in {
            "baseline",
            "selected",
            "fallback_reference",
            "matched_to_P1",
        }:
            continue
        grouped[(row["language"], row["policy"], row["target"], row["selection_kind"])].append(row)
    result = []
    for (language, policy, target, selection_kind), rows in sorted(grouped.items()):
        for key, label in (("threshold224", "T224"), ("threshold512", "T512")):
            values = [float(row[key]) for row in rows if _finite(row.get(key))]
            if not values:
                result.append(
                    {
                        "language": language,
                        "policy": policy,
                        "target": target,
                        "selection_kind": selection_kind,
                        "threshold": label,
                        "count": 0,
                        "median": None,
                        "q1": None,
                        "q3": None,
                        "iqr": None,
                        "minimum": None,
                        "maximum": None,
                    }
                )
                continue
            q1, median, q3 = np.percentile(np.asarray(values), [25, 50, 75])
            result.append(
                {
                    "language": language,
                    "policy": policy,
                    "target": target,
                    "selection_kind": selection_kind,
                    "threshold": label,
                    "count": len(values),
                    "median": float(median),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "minimum": float(min(values)),
                    "maximum": float(max(values)),
                }
            )
    return result


def _aggregate_turn_fragmentation(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["status"] == "available" and row["selection_kind"] in {
            "baseline",
            "selected",
            "fallback_reference",
            "matched_to_P1",
        }:
            grouped[(row["language"], row["policy"], row["target"])].append(row)
    result = []
    for (language, policy, target), rows in sorted(grouped.items()):
        result.append(
            {
                "language": language,
                "policy": policy,
                "target": target,
                "heldout_turn_fragmentation_mean": float(
                    np.mean([row["turn_fragmentation_rate"] for row in rows])
                ),
                "heldout_false_splits_per_100_turns_mean": float(
                    np.mean([row["false_splits_per_100_turns"] for row in rows])
                ),
                "heldout_false_cutoff_rate_mean": float(
                    np.mean([row["false_cutoff_rate"] for row in rows])
                ),
                "folds": len(rows),
            }
        )
    return result


def score_distributions(
    rows_by_language: dict[str, list[dict[str, Any]]], *, output_dir: Path
) -> None:
    quantiles = (10, 25, 50, 75, 90, 95, 99)
    for language, rows in rows_by_language.items():
        output = []
        for probe_ms, key in ((224, "score_224"), (512, "score_512")):
            for label in ("eot", "hold"):
                values = [
                    float(row[key])
                    for row in rows
                    if row["label"] == label and _finite(row.get(key))
                ]
                row = {
                    "language": language,
                    "probe_ms": probe_ms,
                    "label": label,
                    "count": len(values),
                }
                if values:
                    row.update(
                        {
                            f"p{quantile}": float(np.percentile(values, quantile))
                            for quantile in quantiles
                        }
                    )
                else:
                    row.update({f"p{quantile}": None for quantile in quantiles})
                output.append(row)
        _write_csv(output_dir / f"score_distribution_{language}.csv", output)


def _aggregate_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if _finite(row.get(field))]
        result[field] = float(np.mean(values)) if values else None
    return result


def _final_operating_points(
    cv_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    selected: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    fallback: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["status"] != "available":
            continue
        key = (row["language"], row["policy"], row["target"])
        if row["selection_kind"] == "selected":
            selected[key].append(row)
        elif row["selection_kind"] in {"fallback_reference", "matched_to_P1"}:
            fallback[key].append(row)
    points: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in sorted(set(selected) | set(fallback)):
        rows = selected.get(key) or fallback[key]
        selection_kind = "selected" if key in selected else "fallback"
        language, policy, target = key
        threshold224 = [
            float(row["threshold224"]) for row in rows if _finite(row.get("threshold224"))
        ]
        threshold512 = [
            float(row["threshold512"]) for row in rows if _finite(row.get("threshold512"))
        ]
        points[key] = {
            "language": language,
            "policy": policy,
            "target": target,
            "selection_kind": selection_kind,
            "folds_available": len(rows),
            "folds_expected": len(CV_SEEDS) * N_FOLDS,
            "threshold224": float(np.median(threshold224)) if threshold224 else None,
            "threshold512": float(np.median(threshold512)) if threshold512 else None,
            "threshold224_iqr": (
                float(np.percentile(threshold224, 75) - np.percentile(threshold224, 25))
                if threshold224
                else None
            ),
            "threshold512_iqr": (
                float(np.percentile(threshold512, 75) - np.percentile(threshold512, 25))
                if threshold512
                else None
            ),
            "heldout_mean": _aggregate_rows(rows, METRIC_FIELDS),
        }
    return points


def bootstrap_confidence_intervals(
    rows_by_language: dict[str, list[dict[str, Any]]],
    points: dict[tuple[str, str, str], dict[str, Any]],
    *,
    output_dir: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    output = []
    metrics = (
        "false_cutoff_rate",
        "relative_false_cutoff_reduction",
        "mean_endpoint_latency_ms",
        "turn_fragmentation_rate",
    )
    for (language, policy, target), point in sorted(points.items()):
        rows = rows_by_language[language]
        groups = sorted({str(row["turn_id"]) for row in rows})
        group_rows = {
            group: [row for row in rows if str(row["turn_id"]) == group] for group in groups
        }
        rng = random.Random(
            seed
            + sum(ord(char) for char in language)
            + sum(ord(char) for char in policy)
            + len(target)
        )
        sampled_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(resamples):
            sampled_groups = [rng.choice(groups) for _ in groups]
            sampled = [row for group in sampled_groups for row in group_rows[group]]
            policy_metrics = simulate_policy(
                sampled,
                policy,
                point["threshold224"],
                point["threshold512"],
            )
            baseline = simulate_policy(sampled, "B0")
            sampled_values["false_cutoff_rate"].append(float(policy_metrics["false_cutoff_rate"]))
            sampled_values["relative_false_cutoff_reduction"].append(
                _relative_reduction(
                    float(baseline["false_cutoff_rate"]), float(policy_metrics["false_cutoff_rate"])
                )
            )
            sampled_values["mean_endpoint_latency_ms"].append(
                float(policy_metrics["mean_endpoint_latency_ms"] or 0.0)
            )
            sampled_values["turn_fragmentation_rate"].append(
                float(policy_metrics["turn_fragmentation_rate"])
            )
        for metric, values in sampled_values.items():
            output.append(
                {
                    "language": language,
                    "policy": policy,
                    "target": target,
                    "comparison": policy,
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "resamples": resamples,
                    "seed": seed,
                }
            )

    for language in rows_by_language:
        for target in ("low_latency", "stability"):
            p1 = points.get((language, "P1", target))
            if p1 is None:
                continue
            rows = rows_by_language[language]
            groups = sorted({str(row["turn_id"]) for row in rows})
            group_rows = {
                group: [row for row in rows if str(row["turn_id"]) == group] for group in groups
            }
            for comparison_policy in ("P2", "P3"):
                candidate = points.get((language, comparison_policy, target))
                if candidate is None:
                    continue
                rng = random.Random(
                    seed
                    + 1000
                    + sum(ord(char) for char in language)
                    + sum(ord(char) for char in comparison_policy)
                )
                differences: dict[str, list[float]] = {
                    "false_cutoff_rate_delta": [],
                    "mean_endpoint_latency_delta_ms": [],
                    "turn_fragmentation_delta": [],
                }
                for _ in range(resamples):
                    sampled_groups = [rng.choice(groups) for _ in groups]
                    sampled = [row for group in sampled_groups for row in group_rows[group]]
                    p1_metrics = simulate_policy(
                        sampled, "P1", p1["threshold224"], p1["threshold512"]
                    )
                    candidate_metrics = simulate_policy(
                        sampled,
                        comparison_policy,
                        candidate["threshold224"],
                        candidate["threshold512"],
                    )
                    differences["false_cutoff_rate_delta"].append(
                        float(
                            candidate_metrics["false_cutoff_rate"] - p1_metrics["false_cutoff_rate"]
                        )
                    )
                    differences["mean_endpoint_latency_delta_ms"].append(
                        float(
                            (candidate_metrics["mean_endpoint_latency_ms"] or 0.0)
                            - (p1_metrics["mean_endpoint_latency_ms"] or 0.0)
                        )
                    )
                    differences["turn_fragmentation_delta"].append(
                        float(
                            candidate_metrics["turn_fragmentation_rate"]
                            - p1_metrics["turn_fragmentation_rate"]
                        )
                    )
                for metric, values in differences.items():
                    output.append(
                        {
                            "language": language,
                            "policy": comparison_policy,
                            "target": target,
                            "comparison": f"{comparison_policy}_vs_P1",
                            "metric": metric,
                            "estimate": float(np.mean(values)),
                            "ci_low": float(np.percentile(values, 2.5)),
                            "ci_high": float(np.percentile(values, 97.5)),
                            "resamples": resamples,
                            "seed": seed,
                        }
                    )
    _write_csv(output_dir / "bootstrap_confidence_intervals.csv", output)
    return output


def audit_providers(*, output_dir: Path, model_path: Path) -> dict[str, Any]:
    import onnxruntime as ort

    old_root = (
        ROOT
        / ".data"
        / "eot-bench"
        / "output"
        / "livekit__eot-bench-data__validation__min_silence_100ms"
    )
    manifest = old_root / "ko" / "smart_turn_audio_adapter__cb4f57229a" / "manifest.json"
    old_prediction = manifest.with_name("predictions.parquet")
    old_cpu_artifact = ROOT / ".data" / "eot-bench" / "results-v5" / "cpu_benchmark.json"
    old_manifest = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    old_cpu = (
        json.loads(old_cpu_artifact.read_text(encoding="utf-8"))
        if old_cpu_artifact.is_file()
        else {}
    )
    current_runtime = {
        "onnxruntime_version": ort.__version__,
        "available_providers": list(ort.get_available_providers()),
        "installed_ort_packages": _installed_ort_packages(),
    }
    audit = {
        "previous_policy_predictions": {
            "origin": "downloaded",
            "model_variant": "gpu_fp32",
            "execution_provider": "unknown",
            "evidence": [
                f"{manifest.relative_to(ROOT)} is tracked in the eot-bench repository",
                f"manifest model.adapter_id={old_manifest.get('model', {}).get('adapter_id')}",
                "prediction parquet contains no execution-provider metadata",
                f"prediction_sha256={_sha256(old_prediction) if old_prediction.is_file() else 'missing'}",
            ],
            "manifest": old_manifest,
        },
        "previous_local_benchmark": {
            "model_variant": "cpu_int8",
            "available_providers": [],
            "session_providers": [],
            "installed_ort_packages": current_runtime["installed_ort_packages"],
            "execution_provider": "unknown",
            "evidence": [
                f"{old_cpu_artifact.relative_to(ROOT)} records model filename {Path(old_cpu.get('model', '')).name}",
                f"model_sha256={old_cpu.get('model_sha256')}",
                "benchmark_cpu.py created InferenceSession without an explicit providers argument",
                "cpu_benchmark.json did not record get_available_providers(), get_providers(), or provider options",
            ],
        },
        "new_cpu_prediction_run": {
            "origin": "local",
            "model_variant": "cpu_int8",
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 2,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "model_revision": MODEL_REVISION,
            "model_sha256": _sha256(model_path),
            "evidence": [
                "CpuScorer passes providers=['CPUExecutionProvider']",
                "CpuScorer verifies session.get_providers()[0]",
                f"model_path={model_path}",
            ],
        },
        "current_audit_runtime": current_runtime,
    }
    _write_json(output_dir / "provider_audit.json", audit)
    lines = [
        "# Provider audit",
        "",
        "| Artifact or run | Origin | Model | Provider | Confidence | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Previous policy predictions | downloaded/committed external artifact | GPU FP32 variant | unknown | high for model, unknown for provider | manifest adapter id is `smart-turn-v3.2-gpu`; provider not stored |",
        "| Previous local benchmark | local | CPU int8 | unknown | high for model, unknown for provider | artifact identifies CPU model; session provider was not recorded and code relied on automatic selection |",
        "| New CPU prediction run | local | CPU int8 | CPUExecutionProvider | confirmed | explicit provider list and post-construction provider assertion |",
        "",
        "The local AMD GPU is not evidence that the previous policy parquet was GPU-generated; that parquet is a committed external artifact.",
        "",
        f"Current audit runtime: ONNX Runtime `{current_runtime['onnxruntime_version']}`, available providers `{current_runtime['available_providers']}`, installed packages `{current_runtime['installed_ort_packages']}`.",
    ]
    (output_dir / "provider_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def _aggregate_cv(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_rows:
        if row["status"] != "available" or row["selection_kind"] not in {
            "baseline",
            "selected",
            "fallback_reference",
            "matched_to_P1",
        }:
            continue
        grouped[(row["language"], row["target"], row["policy"], row["selection_kind"])].append(row)
    output = []
    for (language, target, policy, selection_kind), rows in sorted(grouped.items()):
        output.append(
            {
                "language": language,
                "target": target,
                "policy": policy,
                "selection_kind": selection_kind,
                "folds": len(rows),
                **_aggregate_rows(rows, METRIC_FIELDS),
                "threshold224_median": (
                    float(
                        np.median(
                            [
                                row["threshold224"]
                                for row in rows
                                if _finite(row.get("threshold224"))
                            ]
                        )
                    )
                    if any(_finite(row.get("threshold224")) for row in rows)
                    else None
                ),
                "threshold512_median": (
                    float(
                        np.median(
                            [
                                row["threshold512"]
                                for row in rows
                                if _finite(row.get("threshold512"))
                            ]
                        )
                    )
                    if any(_finite(row.get("threshold512")) for row in rows)
                    else None
                ),
            }
        )
    return output


def _aggregate_increments(increment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in increment_rows:
        grouped[(row["language"], row["target"], row["comparison"])].append(row)
    result = []
    for (language, target, comparison), rows in sorted(grouped.items()):
        result.append(
            {
                "language": language,
                "target": target,
                "comparison": comparison,
                "folds": len(rows),
                "true_eot_recovered_at_512_mean": float(
                    np.mean([row["true_eot_recovered_at_512"] for row in rows])
                ),
                "new_false_cutoffs_at_512_mean": float(
                    np.mean([row["new_false_cutoffs_at_512"] for row in rows])
                ),
                "mean_latency_change_ms": float(
                    np.mean([row["mean_latency_change_ms"] for row in rows])
                ),
                "timeout_rate_change_pp": float(
                    np.mean([row["timeout_rate_change_pp"] for row in rows])
                ),
                "turn_fragmentation_change_pp": float(
                    np.mean([row["turn_fragmentation_change_pp"] for row in rows])
                ),
                "false_cutoff_delta": float(np.mean([row["false_cutoff_delta"] for row in rows])),
            }
        )
    return result


def _increment_qualifies(row: dict[str, Any]) -> bool:
    return float(row["false_cutoff_delta"]) <= 0.005 + 1e-12 and (
        float(row["mean_latency_change_ms"]) <= -20.0 + 1e-12
        or float(row["timeout_rate_change_pp"]) <= -5.0 + 1e-12
    )


def _independent_qualifies(row: dict[str, Any]) -> bool:
    return float(row["false_cutoff_delta"]) <= 0.005 + 1e-12 and (
        float(row["mean_latency_change_ms"]) <= -15.0 + 1e-12
        or float(row["false_cutoff_delta"]) <= -0.01 + 1e-12
    )


def _choose_target_rows(
    rows: list[dict[str, Any]], language: str, comparison: str
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows if row["language"] == language and row["comparison"] == comparison
    ]
    for target in ("stability", "low_latency"):
        matches = [row for row in candidates if row["target"] == target]
        if matches:
            return matches[0]
    return None


def _cpu_gate(latency: dict[str, Any], probe_overlap_rate: float = 0.0) -> dict[str, Any]:
    synthetic = next(
        (
            row
            for row in latency["two_thread"].get("synthetic", [])
            if abs(float(row["duration_s"]) - 8.0) < 1e-9
        ),
        None,
    )
    if synthetic is None:
        return {
            "two_thread_8s_p95_ms": None,
            "two_thread_8s_p99_ms": None,
            "preferred_p95_pass": False,
            "acceptable_p95_pass": False,
            "acceptable_p99_pass": False,
            "probe_overlap_rate": probe_overlap_rate,
            "probe_overlap_pass": probe_overlap_rate < 0.01,
        }
    return {
        "two_thread_8s_p95_ms": synthetic["p95_ms"],
        "two_thread_8s_p99_ms": synthetic["p99_ms"],
        "preferred_p95_pass": float(synthetic["p95_ms"]) <= 150.0,
        "acceptable_p95_pass": float(synthetic["p95_ms"]) <= 200.0,
        "acceptable_p99_pass": float(synthetic["p99_ms"]) <= 250.0,
        "probe_overlap_rate": probe_overlap_rate,
        "probe_overlap_pass": probe_overlap_rate < 0.01,
    }


def build_summary(
    *,
    output_dir: Path,
    rows_by_language: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    cv: dict[str, Any],
    latency: dict[str, Any],
) -> dict[str, Any]:
    aggregates = _aggregate_cv(cv["cv_rows"])
    increments = _aggregate_increments(cv["increment_rows"])
    points = _final_operating_points(cv["cv_rows"])
    overlap_values = [
        float(row["probe_overlap_rate"])
        for row in cv["cv_rows"]
        if row["status"] == "available"
        and row["selection_kind"] == "selected"
        and row["policy"] in {"P2", "P3"}
        and _finite(row.get("probe_overlap_rate"))
    ]
    cpu_gate = _cpu_gate(latency, max(overlap_values, default=0.0))
    chosen_increments = []
    for language in rows_by_language:
        for comparison in ("P2_vs_P1", "P3_vs_P1", "P3_vs_P2"):
            row = _choose_target_rows(increments, language, comparison)
            if row is not None:
                chosen_increments.append(row)
    p2_qualified = [
        row
        for row in chosen_increments
        if row["comparison"] == "P2_vs_P1" and _increment_qualifies(row)
    ]
    p3_qualified = [
        row
        for row in chosen_increments
        if row["comparison"] == "P3_vs_P1" and _increment_qualifies(row)
    ]
    independent_qualified = [
        row
        for row in chosen_increments
        if row["comparison"] == "P3_vs_P2" and _independent_qualifies(row)
    ]
    available_points = [
        point
        for point in points.values()
        if point["selection_kind"] == "selected" and point["folds_available"] >= 20
    ]
    unstable = len(available_points) < len(LANGUAGES) * 3
    if (
        not cpu_gate["acceptable_p95_pass"]
        or not cpu_gate["acceptable_p99_pass"]
        or not cpu_gate["probe_overlap_pass"]
    ):
        decision = "STOP"
    elif unstable:
        decision = "PARTIAL_RECALIBRATION"
    elif len(independent_qualified) >= 3 and len(p3_qualified) >= 3:
        decision = "PROCEED_WITH_P3_SHADOW"
    elif len(p2_qualified) >= 3:
        decision = "PROCEED_WITH_P2_SHADOW"
    else:
        decision = "PROCEED_WITH_P1_SHADOW"

    span_counts = {
        language: {
            "spans": len(rows),
            "eot": sum(row["label"] == "eot" for row in rows),
            "hold": sum(row["label"] == "hold" for row in rows),
            "score_224": sum(_finite(row.get("score_224")) for row in rows),
            "score_512": sum(_finite(row.get("score_512")) for row in rows),
            "groups": len({str(row["turn_id"]) for row in rows}),
        }
        for language, rows in rows_by_language.items()
    }
    selected = {}
    for language in rows_by_language:
        selected[language] = {}
        for policy in ("P1", "P2", "P3"):
            point = points.get((language, policy, "stability")) or points.get(
                (language, policy, "low_latency")
            )
            selected[language][policy] = point
    selected_payload = {
        "selection_rule": "median threshold from held-out training-only selections; stability target preferred, low-latency fallback; B0-matched fallback points are marked separately when target constraints are unavailable",
        "points": selected,
        "cv_aggregates": aggregates,
    }
    _write_json(output_dir / "selected_operating_points.json", selected_payload)
    summary = {
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "validation",
            "eot_bench_revision": EOT_BENCH_REVISION,
        },
        "model": {
            "repository": "pipecat-ai/smart-turn-v3",
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "sha256": audit["new_cpu_prediction_run"]["model_sha256"],
            "variant": "cpu_int8",
        },
        "span_counts": span_counts,
        "cross_validation": {
            "folds": N_FOLDS,
            "seeds": list(CV_SEEDS),
            "heldout_evaluations_per_language": N_FOLDS * len(CV_SEEDS),
            "aggregates": aggregates,
            "increments": increments,
        },
        "provider_audit": audit,
        "cpu_latency": latency,
        "cpu_gate": cpu_gate,
        "decision": decision,
        "decision_checks": {
            "p2_languages_meeting_rule": len(p2_qualified),
            "p3_languages_meeting_rule": len(p3_qualified),
            "independent_threshold_languages_meeting_rule": len(independent_qualified),
            "unstable": unstable,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _format_pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _format_ms(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1f}"


def write_report(summary: dict[str, Any], *, output_dir: Path) -> None:
    counts = summary["span_counts"]
    aggregates = summary["cross_validation"]["aggregates"]
    increments = summary["cross_validation"]["increments"]
    lines = [
        "# Smart Turn CPU-int8 two-thread calibration",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Provider audit",
        "",
        "| Artifact or run | Origin | Model | Provider | Confidence | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Previous policy predictions | downloaded/committed | GPU FP32 variant | unknown | model confirmed; provider unknown | manifest adapter id and missing provider metadata |",
        "| Previous local benchmark | local | CPU int8 | unknown | model confirmed; provider unknown | prior code did not pin or record provider |",
        "| New CPU prediction run | local | CPU int8 | CPUExecutionProvider | confirmed | explicit provider and post-session assertion |",
        "",
        "## Data integrity",
        "",
        "| Language | Spans | EOT | Hold | 224 scores | 512 scores | Groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language in LANGUAGES:
        item = counts[language]
        lines.append(
            f"| {language} | {item['spans']} | {item['eot']} | {item['hold']} | {item['score_224']} | {item['score_512']} | {item['groups']} |"
        )
    lines.extend(
        [
            "",
            "All thresholds were selected within training conversation groups. Each language has 5 folds across 5 independent group-shuffle seeds; individual pause spans from a turn never cross the split boundary.",
            "Where the low-latency/stability constraints yielded no target-selected point, the table and bootstrap use a B0 false-cutoff-matched fallback for diagnostics only; fallback points cannot satisfy the final proceed gate.",
            "",
            "## Held-out policy results",
            "",
            "| Language | Target | Policy | Selection | Folds | Cutoff | Mean latency | P50 | P95 | Timeout | T224 | T512 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['language']} | {row['target']} | {row['policy']} | {row['selection_kind']} | {row['folds']} | {_format_pct(row['false_cutoff_rate'])} | {_format_ms(row['mean_endpoint_latency_ms'])} ms | {_format_ms(row['p50_endpoint_latency_ms'])} ms | {_format_ms(row['p95_endpoint_latency_ms'])} ms | {_format_pct(row['hard_timeout_rate'])} | {row['threshold224_median'] if row['threshold224_median'] is not None else '—'} | {row['threshold512_median'] if row['threshold512_median'] is not None else '—'} |"
        )
    lines.extend(
        [
            "",
            "## Incremental value of the 512 ms probe",
            "",
            "The comparison rows use thresholds selected on training groups to match the P1 training false-cutoff rate, then report the held-out result.",
            "",
            "| Language | Target | Comparison | Recovered EOT | New false cuts | Mean latency Δ | Timeout Δ | Cutoff Δ |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in increments:
        lines.append(
            f"| {row['language']} | {row['target']} | {row['comparison']} | {row['true_eot_recovered_at_512_mean']:.2f} | {row['new_false_cutoffs_at_512_mean']:.2f} | {row['mean_latency_change_ms']:.2f} ms | {row['timeout_rate_change_pp']:.2f} pp | {row['false_cutoff_delta'] * 100:.2f} pp |"
        )
    gate = summary["cpu_gate"]
    lines.extend(
        [
            "",
            "## Two-thread CPU latency",
            "",
            f"The scorer uses `{summary['cpu_latency']['two_thread']['configuration']['execution_provider']}`, 2 intra-op threads, 1 inter-op thread, sequential execution, and an explicit provider list.",
            "",
            "| Input | Calls | P50 | P95 | P99 | Max |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["cpu_latency"]["two_thread"]["synthetic"]:
        lines.append(
            f"| {row['duration_s']} s | {row['count']} | {_format_ms(row['p50_ms'])} ms | {_format_ms(row['p95_ms'])} ms | {_format_ms(row['p99_ms'])} ms | {_format_ms(row['max_ms'])} ms |"
        )
    lines.extend(
        [
            "",
            "| Gate | Result |",
            "| --- | --- |",
            f"| 8 s p95 <= 150 ms preferred | `{gate['preferred_p95_pass']}` ({gate['two_thread_8s_p95_ms']:.2f} ms) |",
            f"| 8 s p95 <= 200 ms acceptable | `{gate['acceptable_p95_pass']}` |",
            f"| 8 s p99 <= 250 ms | `{gate['acceptable_p99_pass']}` ({gate['two_thread_8s_p99_ms']:.2f} ms) |",
            f"| Selected-policy probe overlap < 1% | `{gate['probe_overlap_pass']}` ({gate['probe_overlap_rate'] * 100:.3f}%) |",
            "",
            "## Threshold stability",
            "",
            "See `threshold_stability.csv` for median, IQR, min, and max over the 25 held-out selections per language/policy/target.",
            "",
            "## Confidence intervals",
            "",
            "`bootstrap_confidence_intervals.csv` contains 1,000 conversation-level bootstrap resamples with seed `20260802`. The same audio span is never treated as an additional accuracy sample merely because it was scored once at each probe.",
            "",
            "## Decision",
            "",
            f"**{summary['decision']}**",
            "",
            "The result is based on held-out, runtime-aware CPU-int8 policy metrics. It does not authorize active endpointing or removal of speculative translation.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_languages(values: list[str]) -> tuple[str, ...]:
    languages = tuple(_map_language(value) for value in values)
    if not languages or any(language not in LANGUAGES for language in languages):
        raise SystemExit(f"languages must be selected from {LANGUAGES}")
    return languages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--real-audio-root", type=Path, default=ROOT / ".data" / "eot-bench" / "real_audio"
    )
    parser.add_argument(
        "--dataset-cache-dir", type=Path, default=ROOT / ".data" / "eot-bench" / "dataset_cache"
    )
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--measured", type=int, default=1000)
    parser.add_argument("--real-calls-per-language", type=int, default=250)
    parser.add_argument("--cold-runs", type=int, default=20)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--cold-worker", action="store_true")
    args = parser.parse_args()
    if args.cold_worker:
        _cold_worker(args.model)
        return
    if args.output_dir is None:
        parser.error("the following arguments are required: --output-dir")
    languages = _parse_languages(args.languages)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_providers(output_dir=output_dir, model_path=args.model)
    if not args.skip_predictions:
        generate_predictions(
            model_path=args.model,
            output_dir=output_dir,
            languages=languages,
            dataset_revision=args.dataset_revision,
            split=args.split,
            dataset_cache_dir=args.dataset_cache_dir,
        )
    rows_by_language = {
        language: _load_prediction_rows(output_dir, language) for language in languages
    }
    score_distributions(rows_by_language, output_dir=output_dir)
    if args.skip_latency:
        latency = {
            "two_thread": {
                "synthetic": [],
                "real_audio": [],
                "configuration": {},
            },
            "one_thread": {},
        }
    else:
        latency = run_latency_benchmarks(
            model_path=args.model,
            output_dir=output_dir,
            real_audio_root=args.real_audio_root,
            languages=languages,
            warmups=args.warmups,
            measured=args.measured,
            real_calls_per_language=args.real_calls_per_language,
            cold_runs=args.cold_runs,
        )
    cv = cross_validate(rows_by_language, output_dir=output_dir)
    points = _final_operating_points(cv["cv_rows"])
    bootstrap_confidence_intervals(rows_by_language, points, output_dir=output_dir)
    summary = build_summary(
        output_dir=output_dir,
        rows_by_language=rows_by_language,
        audit=audit,
        cv=cv,
        latency=latency,
    )
    write_report(summary, output_dir=output_dir)
    print(
        json.dumps({"output_dir": str(output_dir), "decision": summary["decision"]}, indent=2),
        flush=True,
    )


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eot_experiment import policy_analysis as _policy_analysis

_POLICY_EXPORTS = (
    "_candidate_thresholds",
    "_final_operating_points",
    "_group_splits",
    "_matched_candidate",
    "_policy_trace",
    "_threshold_grid",
    "_trace_metrics",
    "_validate_prediction_rows",
    "audit_providers",
    "bootstrap_confidence_intervals",
    "build_summary",
    "cross_validate",
    "hierarchical_bootstrap",
    "simulate_policy",
    "validate_input_artifacts",
    "write_report",
)
globals().update({name: getattr(_policy_analysis, name) for name in _POLICY_EXPORTS})
policy_main = _policy_analysis.main
globals()["main"] = policy_main


if __name__ == "__main__":
    policy_main()
