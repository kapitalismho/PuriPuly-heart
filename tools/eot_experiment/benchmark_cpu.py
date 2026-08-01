from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from puripuly_heart.core.vad.smart_turn import prepare_smart_turn_audio
from puripuly_heart.core.vad.smart_turn_features import compute_whisper_log_mel_features

SAMPLE_RATE_HZ = 16000
DEFAULT_LENGTHS_S = (0.5, 1.0, 2.0, 4.0, 8.0)
THREAD_SETTINGS = ("default", "one")
THREAD_SETTING_CHOICES = ("default", "one", "1", "2", "4", "8")


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        array, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)
        return array.reshape(-1), int(sample_rate)
    except ImportError:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if sample_width != 2:
            raise RuntimeError("soundfile is required for non-16-bit WAV samples")
        array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            array = array.reshape(-1, channels).mean(axis=1)
        return array, int(sample_rate)


def _rss_mb() -> float:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for process memory measurement") from exc
    return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)


def _process_cpu_seconds(process: Any) -> float:
    times = process.cpu_times()
    return float(times.user + times.system)


def _cpu_usage(
    *, process: Any, cpu_started: float, wall_started: float, logical_cpu_count: int
) -> dict[str, float]:
    wall_seconds = max(time.perf_counter() - wall_started, 1e-9)
    cpu_seconds = max(_process_cpu_seconds(process) - cpu_started, 0.0)
    one_core_percent = cpu_seconds / wall_seconds * 100.0
    return {
        "process_cpu_percent_of_total": one_core_percent / logical_cpu_count,
        "process_cpu_percent_one_core_equivalent": one_core_percent,
    }


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def _synthetic_audio(duration_s: float) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE_HZ))
    time_axis = np.arange(samples, dtype=np.float32) / SAMPLE_RATE_HZ
    return (0.08 * np.sin(2.0 * np.pi * 220.0 * time_axis)).astype(np.float32)


def _predict(session: Any, audio: np.ndarray, *, sample_rate_hz: int) -> float:
    prepared = prepare_smart_turn_audio(audio, sample_rate_hz=sample_rate_hz)
    features = compute_whisper_log_mel_features(prepared)
    outputs = session.run(None, {"input_features": np.expand_dims(features, axis=0)})
    return float(np.asarray(outputs[0]).reshape(-1)[0])


def _run_setting(
    *,
    model_path: Path,
    thread_setting: str,
    lengths_s: tuple[float, ...],
    real_audio: dict[str, Path],
    real_audio_ids: dict[str, str],
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    import onnxruntime as ort
    import psutil

    memory_before = _rss_mb()
    process = psutil.Process(os.getpid())
    logical_cpu_count = psutil.cpu_count(logical=True) or 1
    initialization_started = time.perf_counter()
    session_options = None
    if thread_setting != "default":
        thread_count = 1 if thread_setting == "one" else int(thread_setting)
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = thread_count
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    cold_initialization_ms = (time.perf_counter() - initialization_started) * 1000.0
    memory_after = _rss_mb()

    synthetic_rows: list[dict[str, Any]] = []
    for duration_s in lengths_s:
        audio = _synthetic_audio(duration_s)
        for _ in range(warmups):
            _predict(session, audio, sample_rate_hz=SAMPLE_RATE_HZ)
        latencies: list[float] = []
        scores: list[float] = []
        cpu_started = _process_cpu_seconds(process)
        wall_started = time.perf_counter()
        for _ in range(repeats):
            started = time.perf_counter()
            scores.append(_predict(session, audio, sample_rate_hz=SAMPLE_RATE_HZ))
            latencies.append((time.perf_counter() - started) * 1000.0)
        row = {
            "duration_s": duration_s,
            "input_kind": "synthetic",
            "samples": int(audio.size),
            "score_mean": float(np.mean(scores)),
        }
        row.update(_stats(latencies))
        row.update(
            _cpu_usage(
                process=process,
                cpu_started=cpu_started,
                wall_started=wall_started,
                logical_cpu_count=logical_cpu_count,
            )
        )
        synthetic_rows.append(row)

    real_rows: list[dict[str, Any]] = []
    for language, path in sorted(real_audio.items()):
        audio, sample_rate_hz = _load_audio(path)
        if sample_rate_hz != SAMPLE_RATE_HZ:
            raise ValueError(
                f"{language} real audio is {sample_rate_hz}Hz, expected {SAMPLE_RATE_HZ}Hz"
            )
        for _ in range(warmups):
            _predict(session, audio, sample_rate_hz=sample_rate_hz)
        latencies = []
        scores = []
        cpu_started = _process_cpu_seconds(process)
        wall_started = time.perf_counter()
        for _ in range(repeats):
            started = time.perf_counter()
            scores.append(_predict(session, audio, sample_rate_hz=sample_rate_hz))
            latencies.append((time.perf_counter() - started) * 1000.0)
        row = {
            "language": language,
            "sample_id": real_audio_ids.get(language),
            "audio": str(path),
            "input_kind": "real_audio",
            "input_duration_s": float(audio.size / sample_rate_hz),
            "effective_duration_s": float(min(audio.size / sample_rate_hz, 8.0)),
            "samples": int(audio.size),
            "score_mean": float(np.mean(scores)),
        }
        row.update(_stats(latencies))
        row.update(
            _cpu_usage(
                process=process,
                cpu_started=cpu_started,
                wall_started=wall_started,
                logical_cpu_count=logical_cpu_count,
            )
        )
        real_rows.append(row)

    return {
        "thread_setting": thread_setting,
        "cold_initialization_ms": cold_initialization_ms,
        "memory_before_mb": memory_before,
        "memory_after_session_mb": memory_after,
        "memory_increase_mb": memory_after - memory_before,
        "synthetic": synthetic_rows,
        "real_audio": real_rows,
    }


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--thread-setting", choices=THREAD_SETTING_CHOICES)
    parser.add_argument(
        "--thread-settings",
        nargs="+",
        choices=THREAD_SETTING_CHOICES,
        default=list(THREAD_SETTINGS),
    )
    parser.add_argument("--lengths", nargs="+", type=float, required=True)
    parser.add_argument("--real-audio", nargs="*", default=[])
    parser.add_argument("--real-audio-id", nargs="*", default=[])
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    return parser


def _parse_real_audio(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"real-audio must be LANGUAGE=PATH: {value}")
        language, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[language] = path
    return result


def _parse_real_audio_ids(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"real-audio-id must be LANGUAGE=ID: {value}")
        language, sample_id = value.split("=", 1)
        result[language] = sample_id
    return result


def _run_worker(args: argparse.Namespace) -> None:
    if args.thread_setting is None:
        raise SystemExit("--thread-setting is required in worker mode")
    result = _run_setting(
        model_path=args.model,
        thread_setting=args.thread_setting,
        lengths_s=tuple(args.lengths),
        real_audio=_parse_real_audio(args.real_audio),
        real_audio_ids=_parse_real_audio_ids(args.real_audio_id),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    print(json.dumps(result, allow_nan=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = _worker_parser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 10:
        raise SystemExit("warmups must be at least 10")
    if args.repeats < 100:
        raise SystemExit("repeats must be at least 100")
    if not args.worker and args.output is None:
        raise SystemExit("--output is required outside worker mode")
    if args.worker:
        _run_worker(args)
        return

    real_audio = _parse_real_audio(args.real_audio)
    real_audio_ids = _parse_real_audio_ids(args.real_audio_id)
    settings: list[dict[str, Any]] = []
    for thread_setting in args.thread_settings:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            str(args.model),
            "--thread-setting",
            thread_setting,
            "--lengths",
            *(str(value) for value in args.lengths),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--real-audio",
            *(f"{language}={path}" for language, path in sorted(real_audio.items())),
            "--real-audio-id",
            *(f"{language}={sample_id}" for language, sample_id in sorted(real_audio_ids.items())),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        settings.append(json.loads(completed.stdout))

    eight_second = {
        setting["thread_setting"]: next(
            row for row in setting["synthetic"] if abs(row["duration_s"] - 8.0) <= 1e-9
        )
        for setting in settings
    }
    one_thread_row = eight_second.get("one") or eight_second.get("1")
    result = {
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "warmup_calls": args.warmups,
        "measured_calls": args.repeats,
        "input_lengths_s": list(args.lengths),
        "real_audio_languages": sorted(real_audio),
        "real_audio_ids": real_audio_ids,
        "settings": settings,
        "cpu_gate": {
            "preferred_8s_p95_ms": 150.0,
            "acceptable_8s_p95_ms": 200.0,
            "one_thread_8s_p95_ms": one_thread_row["p95_ms"] if one_thread_row else None,
            "preferred_pass": one_thread_row is not None and one_thread_row["p95_ms"] <= 150.0,
            "acceptable_pass": one_thread_row is not None and one_thread_row["p95_ms"] <= 200.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "one_thread_8s_p95_ms": one_thread_row["p95_ms"] if one_thread_row else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
