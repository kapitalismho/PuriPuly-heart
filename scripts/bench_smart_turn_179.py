"""Measured Windows SmartTurn S12/S11/F12/F11 probe for issue #179.

Owns NO production code and NO tests. Reuses the exact production numerical
functions and snapshot contract for every arm:

- S12: production SmartTurnOnnxInference unchanged (ORT 1/2, split offloads).
- S11: same split dispatch, ORT 1/1 (probe-local session, identical predict).
- F12/F11: preparation + features + ONNX in ONE blocking offload after the
  owner snapshot, returning a scalar to the same event loop.

Subcommands:
  fetch       download lawful public speech clips + write sha256 manifest
  smoke       tiny end-to-end validation (NOT a benchmark)
  stages      synchronous stage micro-profile per arm (prepare/features/ONNX)
  matrix      rotated four-arm owner comparison, 30-50 measured calls/arm
  controller  production LISTEN owner/controller exercise, real receipt/deadline
  paced       paced replay with idle gaps + Silero VAD co-load, loop-lag/CPU
  cpu         warmed isolated four-arm process-CPU accounting
  recheck     warmed isolated selected-vs-baseline process-CPU recheck
  hann        bounded Hann allocation micro-probe
  support     ORT spinning-config support check (no benchmark)

Machine-readable JSON goes to --out (default under ignored .data/).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_COMPLETE_THRESHOLD,
    SMART_TURN_INPUT_REVISION,
    SMART_TURN_RESOURCE_SHA256,
    SMART_TURN_SAMPLE_RATE_HZ,
    SMART_TURN_WINDOW_SAMPLES,
    SmartTurnInferenceOwner,
    SmartTurnRequestIdentity,
    bundled_smart_turn_onnx_path,
    prepare_smart_turn_audio,
)
from puripuly_heart.core.audio.smart_turn_features import (
    compute_whisper_log_mel_features,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart

SCHEMA = "smartturn-cpu-179-v2"

LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
LIBRISPEECH_PARQUET = "all/test.clean/0000.parquet"
LIBRISPEECH_URL = (
    f"https://huggingface.co/datasets/{LIBRISPEECH_REPO}/blob/"
    f"{LIBRISPEECH_REVISION}/{LIBRISPEECH_PARQUET}"
)
LIBRISPEECH_PICKS = [
    (0, 0, 0, "6930-75918-0000", 6930, 0, 56080),
    (78, 0, 78, "1320-122617-0000", 1320, 0, 125360),
    (137, 1, 37, "5639-40744-0000", 5639, 62160, 128000),
    (179, 1, 79, "260-123440-0000", 260, 0, 34640),
    (261, 2, 61, "7729-102255-0000", 7729, 0, 52560),
    (308, 3, 8, "2094-142345-0000", 2094, 116560, 128000),
    (369, 3, 69, "3575-170457-0000", 3575, 1840, 128000),
    (426, 4, 26, "7127-75947-0000", 7127, 79760, 128000),
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_flac_mono(blob: bytes) -> np.ndarray:
    import io

    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    assert rate == SMART_TURN_SAMPLE_RATE_HZ, f"rate {rate}"
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def cmd_fetch(args) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, timeout=15)
        text = (
            out.stdout.decode("cp949", errors="replace")
            if isinstance(out.stdout, bytes)
            else (out.stdout or "")
        )
        pinned = text.strip().splitlines()[0] if text.strip() else "unavailable: empty-output"
    except Exception as exc:
        pinned = f"unavailable: {type(exc).__name__}"
    (audio_dir / "power_scheme_initial.txt").write_text(pinned + "\n")
    parquet_path = hf_hub_download(
        LIBRISPEECH_REPO,
        LIBRISPEECH_PARQUET,
        repo_type="dataset",
        revision=LIBRISPEECH_REVISION,
    )
    parquet_sha256 = _sha256_file(Path(parquet_path))
    parquet_file = pq.ParquetFile(parquet_path)
    entries = []
    for (
        global_row,
        row_group,
        local_row,
        utterance_id,
        speaker_id,
        offset,
        length,
    ) in LIBRISPEECH_PICKS:
        table = parquet_file.read_row_groups([row_group]).slice(local_row, 1)
        assert (
            table["id"][0].as_py() == utterance_id
        ), f"row drift: {table['id'][0].as_py()} != {utterance_id}"
        blob = table["audio"][0].as_py()["bytes"]
        clip = _decode_flac_mono(blob)
        window = clip[offset : offset + length]
        assert window.size == length, f"{utterance_id}: window {window.size} != {length}"
        dest = audio_dir / f"{utterance_id}.f32.npy"
        np.save(dest, window)
        entries.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "text": table["text"][0].as_py(),
                "global_row": global_row,
                "row_group": row_group,
                "local_row": local_row,
                "window_offset_samples": offset,
                "window_length_samples": length,
                "clip_samples": int(clip.size),
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "window_sha256": hashlib.sha256(window.tobytes()).hexdigest(),
                "file": dest.name,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "provenance": (
            "LibriSpeech ASR corpus (test.clean), derived from LibriVox public "
            "domain audiobooks; corpus grant CC-BY-4.0 per openslr.org/12 and "
            "the openslr/librispeech_asr dataset card (cardData license "
            "cc-by-4.0). Accessed as pinned HuggingFace parquet rows; decode "
            "is probe-local (pyarrow/soundfile, not production deps)."
        ),
        "source": {
            "repo": LIBRISPEECH_REPO,
            "revision": LIBRISPEECH_REVISION,
            "parquet": LIBRISPEECH_PARQUET,
            "url": LIBRISPEECH_URL,
            "parquet_sha256": parquet_sha256,
        },
        "clips": entries,
    }
    (audio_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fetched {len(entries)} windows -> {audio_dir}")
    for entry in entries:
        print(
            f"  {entry['utterance_id']} sp={entry['speaker_id']} "
            f"win={entry['window_length_samples']} sha256:{entry['window_sha256'][:16]}"
        )
    return 0


def load_fixtures(audio_dir: Path) -> list[dict]:
    """8 speech windows + 5 synthetic guards, all float32 mono 16 kHz."""
    manifest = json.loads((audio_dir / "manifest.json").read_text())
    assert manifest["source"]["revision"] == LIBRISPEECH_REVISION
    assert manifest["source"]["parquet_sha256"], "parquet hash missing"
    fixtures: list[dict] = []
    for entry in manifest["clips"]:
        window = np.load(audio_dir / entry["file"]).astype(np.float32).reshape(-1)
        assert entry["window_sha256"] == hashlib.sha256(window.tobytes()).hexdigest()
        fixtures.append(
            {
                "name": entry["utterance_id"],
                "kind": "speech",
                "audio": window,
                "sha256": entry["window_sha256"],
            }
        )
    rng = np.random.default_rng(179)
    speech_concat = np.concatenate([fixtures[0]["audio"], fixtures[4]["audio"]])
    guards = [
        ("guard_silence_8s", np.zeros(SMART_TURN_WINDOW_SAMPLES, dtype=np.float32)),
        (
            "guard_near_silence_8s",
            (rng.standard_normal(SMART_TURN_WINDOW_SAMPLES) * 1e-4).astype(np.float32),
        ),
        (
            "guard_short_1s",
            (0.25 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32),
        ),
        (
            "guard_exact_8s",
            (
                np.concatenate(
                    [
                        speech_concat,
                        np.zeros(SMART_TURN_WINDOW_SAMPLES - speech_concat.size, dtype=np.float32),
                    ]
                )
                if speech_concat.size < SMART_TURN_WINDOW_SAMPLES
                else speech_concat[:SMART_TURN_WINDOW_SAMPLES].copy()
            ),
        ),
        (
            "guard_over_8s_10s",
            np.concatenate(
                [
                    speech_concat,
                    rng.standard_normal(10 * 16000 - speech_concat.size).astype(np.float32) * 0.05,
                ]
            ),
        ),
    ]
    for name, audio in guards:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        fixtures.append(
            {
                "name": name,
                "kind": "guard",
                "audio": audio,
                "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            }
        )
    return fixtures


def runtime_record() -> dict:
    import onnxruntime as ort

    record = {
        "baseline_sha": "13274569769d3c1ec7a896a2d15b919b76136a6e",
        "probe_sha256": _sha256_file(Path(__file__)),
        "checkout_head": None,
        "background_load": "not sampled",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "cpu": platform.processor() or platform.machine(),
        "power_scheme": "",
        "model_file": "smart-turn-v3.2-cpu.onnx",
        "model_bytes": bundled_smart_turn_onnx_path().stat().st_size,
        "model_sha256": _sha256_file(bundled_smart_turn_onnx_path()),
        "expected_model_sha256": SMART_TURN_RESOURCE_SHA256,
        "model_repo_revision": "f766f81d3cfdf7737ac64aad813d91bbfd56bf93",
        "input_revision": SMART_TURN_INPUT_REVISION,
        "blas": [],
        "env_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=15
        )
        if revision.returncode == 0:
            record["checkout_head"] = revision.stdout.strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, timeout=15)
        raw = out.stdout if isinstance(out.stdout, bytes) else (out.stdout or "").encode()
        text = raw.decode("cp949", errors="replace")
        record["power_scheme"] = (
            text.strip().splitlines()[0] if text.strip() else "unavailable: empty-output"
        )
    except Exception as exc:
        record["power_scheme"] = f"unavailable: {type(exc).__name__}"
    pinned_path = ROOT / ".data" / "smartturn-179" / "audio" / "power_scheme_initial.txt"
    try:
        record["power_scheme_pinned"] = pinned_path.read_text().strip()
    except OSError:
        record["power_scheme_pinned"] = None
    record["power_scheme_drift"] = (
        record["power_scheme_pinned"] is not None
        and record["power_scheme_pinned"] != record["power_scheme"]
    )
    try:
        import threadpoolctl

        record["blas"] = threadpoolctl.threadpool_info()
    except Exception as exc:
        record["blas"] = [{"error": type(exc).__name__}]
    return record


def make_session(model_path: Path, *, inter: int, intra: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = inter
    options.intra_op_num_threads = intra
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


async def _await_owned(awaitable):
    operation = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


class ProbeSplitInference:
    def __init__(self, model_path: Path, *, inter: int, intra: int) -> None:
        self._session = make_session(model_path, inter=inter, intra=intra)
        self.last_trace: dict[str, object] = {}

    async def _offload(self, trace: dict[str, object], name: str, function, *args):
        trace[f"{name}_queue_submit_perf"] = time.perf_counter()

        def run():
            trace[f"{name}_worker_start_perf"] = time.perf_counter()
            try:
                return function(*args)
            finally:
                trace[f"{name}_worker_end_perf"] = time.perf_counter()

        result = await _await_owned(asyncio.to_thread(run))
        trace[f"{name}_loop_return_perf"] = time.perf_counter()
        return result

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
        trace: dict[str, object] = {"predict_enter_perf": time.perf_counter()}
        self.last_trace = trace
        prepared = prepare_smart_turn_audio(audio, sample_rate_hz=sample_rate_hz)
        trace["prepared_sha256"] = hashlib.sha256(prepared.tobytes()).hexdigest()
        features = await self._offload(
            trace, "features", compute_whisper_log_mel_features, prepared
        )
        trace["features_sha256"] = hashlib.sha256(features.tobytes()).hexdigest()
        session = self._session
        if session is None:
            raise RuntimeError("Smart Turn ONNX session is closed")
        model_input = np.expand_dims(features, axis=0)
        trace["model_input_sha256"] = hashlib.sha256(model_input.tobytes()).hexdigest()
        outputs = await self._offload(
            trace, "onnx", session.run, None, {"input_features": model_input}
        )
        if not outputs:
            raise RuntimeError("Smart Turn ONNX model returned no outputs")
        trace["predict_return_perf"] = time.perf_counter()
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def close(self) -> None:
        self._session = None


class ProbeFusedInference:
    def __init__(self, model_path: Path, *, inter: int, intra: int) -> None:
        self._session = make_session(model_path, inter=inter, intra=intra)
        self.last_trace: dict[str, object] = {}

    def _run_blocking(self, trace: dict[str, object], audio: np.ndarray) -> float:
        trace["worker_start_perf"] = time.perf_counter()
        try:
            prepared = prepare_smart_turn_audio(audio, sample_rate_hz=SMART_TURN_SAMPLE_RATE_HZ)
            trace["prepared_sha256"] = hashlib.sha256(prepared.tobytes()).hexdigest()
            trace["prepare_worker_end_perf"] = time.perf_counter()
            features = compute_whisper_log_mel_features(prepared)
            trace["features_sha256"] = hashlib.sha256(features.tobytes()).hexdigest()
            trace["features_worker_end_perf"] = time.perf_counter()
            if self._session is None:
                raise RuntimeError("Smart Turn ONNX session is closed")
            model_input = np.expand_dims(features, axis=0)
            trace["model_input_sha256"] = hashlib.sha256(model_input.tobytes()).hexdigest()
            outputs = self._session.run(None, {"input_features": model_input})
            trace["onnx_worker_end_perf"] = time.perf_counter()
            if not outputs:
                raise RuntimeError("Smart Turn ONNX model returned no outputs")
            return float(np.asarray(outputs[0]).reshape(-1)[0])
        finally:
            trace["worker_end_perf"] = time.perf_counter()

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
        if sample_rate_hz != SMART_TURN_SAMPLE_RATE_HZ:
            raise ValueError("Smart Turn audio must use 16 kHz sampling")
        trace: dict[str, object] = {
            "predict_enter_perf": time.perf_counter(),
            "queue_submit_perf": time.perf_counter(),
        }
        self.last_trace = trace

        def run() -> float:
            return self._run_blocking(trace, audio)

        result = await _await_owned(asyncio.to_thread(run))
        trace["loop_return_perf"] = time.perf_counter()
        trace["predict_return_perf"] = trace["loop_return_perf"]
        return result

    def close(self) -> None:
        self._session = None


ARMS = ("S12", "S11", "F12", "F11")


def make_owner(arm: str, model_path: Path) -> SmartTurnInferenceOwner:
    factories = {
        "S12": lambda path: ProbeSplitInference(path, inter=1, intra=2),
        "S11": lambda path: ProbeSplitInference(path, inter=1, intra=1),
        "F12": lambda path: ProbeFusedInference(path, inter=1, intra=2),
        "F11": lambda path: ProbeFusedInference(path, inter=1, intra=1),
    }
    try:
        factory = factories[arm]
    except KeyError as exc:
        raise ValueError(f"unknown arm {arm}") from exc
    return SmartTurnInferenceOwner(model_path=model_path, inference_factory=factory)


async def _wait_ready(owner: SmartTurnInferenceOwner, timeout_s: float = 60.0) -> None:
    owner.request_prepare()
    async with asyncio.timeout(timeout_s):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.005)
            if owner.snapshot.availability == "error":
                raise RuntimeError(f"prepare failed: {owner.snapshot.last_error}")


def _identity(pause_id: int) -> SmartTurnRequestIdentity:
    now = time.monotonic()
    return SmartTurnRequestIdentity(
        activation_generation=1,
        segment_id=uuid4(),
        pause_id=pause_id,
        context_revision=pause_id,
        input_revision=SMART_TURN_INPUT_REVISION,
        source_frontier=0,
        probe_frontier_monotonic_s=now,
        complete_deadline_monotonic_s=now + 60.0,
    )


async def _run_one(
    owner: SmartTurnInferenceOwner,
    audio: np.ndarray,
    pause_id: int,
    *,
    round_index: int | None = None,
) -> dict:
    outcome: dict = {}
    submit_start_perf = time.perf_counter()
    completed: list = []

    async def receive(result) -> None:
        completed.append((time.perf_counter(), time.monotonic(), result))

    status = owner.submit(_identity(pause_id), audio, receive)
    submit_return_perf = time.perf_counter()
    outcome["admission"] = status
    outcome["submit_call_ms"] = (submit_return_perf - submit_start_perf) * 1000.0
    if round_index is not None:
        outcome["round"] = round_index
    if status != "started":
        outcome["submit_to_return_ms"] = (submit_return_perf - submit_start_perf) * 1000.0
        return outcome
    async with asyncio.timeout(120.0):
        while not completed:
            await asyncio.sleep(0.002)
    receipt_perf, receipt_mono, result = completed[0]
    outcome["submit_to_receipt_ms"] = (receipt_perf - submit_start_perf) * 1000.0
    outcome["completed_to_receipt_ms"] = (receipt_mono - result.completed_at_monotonic_s) * 1000.0
    outcome["duration_s"] = result.duration_s
    outcome["score"] = result.score
    outcome["outcome"] = result.outcome
    inference = owner._inference
    trace = getattr(inference, "last_trace", {})
    if trace:
        outcome["trace"] = {key: value for key, value in trace.items() if key.endswith("_sha256")}
        predict_enter = trace.get("predict_enter_perf")
        predict_return = trace.get("predict_return_perf")
        if isinstance(predict_enter, float):
            outcome["task_start_ms"] = (predict_enter - submit_start_perf) * 1000.0
        if isinstance(predict_return, float):
            outcome["worker_to_loop_return_ms"] = (receipt_perf - predict_return) * 1000.0
        for name in ("features", "onnx"):
            queue_submit = trace.get(f"{name}_queue_submit_perf")
            worker_start = trace.get(f"{name}_worker_start_perf")
            worker_end = trace.get(f"{name}_worker_end_perf")
            loop_return = trace.get(f"{name}_loop_return_perf")
            if isinstance(queue_submit, float) and isinstance(worker_start, float):
                outcome[f"{name}_queue_wait_ms"] = (worker_start - queue_submit) * 1000.0
            if isinstance(worker_start, float) and isinstance(worker_end, float):
                outcome[f"{name}_worker_ms"] = (worker_end - worker_start) * 1000.0
            if isinstance(worker_end, float) and isinstance(loop_return, float):
                outcome[f"{name}_worker_to_loop_ms"] = (loop_return - worker_end) * 1000.0
        queue_submit = trace.get("queue_submit_perf")
        worker_start = trace.get("worker_start_perf")
        worker_end = trace.get("worker_end_perf")
        loop_return = trace.get("loop_return_perf")
        if isinstance(queue_submit, float) and isinstance(worker_start, float):
            outcome["inference_queue_wait_ms"] = (worker_start - queue_submit) * 1000.0
        if isinstance(worker_start, float) and isinstance(worker_end, float):
            outcome["inference_worker_ms"] = (worker_end - worker_start) * 1000.0
        if isinstance(worker_end, float) and isinstance(loop_return, float):
            outcome["inference_worker_to_loop_ms"] = (loop_return - worker_end) * 1000.0
        outcome.update({key: value for key, value in trace.items() if key.endswith("_sha256")})
    return outcome


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return {"n": 0}
    median = (
        ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    )
    p95_index = min(count - 1, int(-(-95 * count // 100)) - 1)
    return {
        "n": count,
        "median": median,
        "p95": ordered[p95_index],
        "worst": ordered[-1],
        "best": ordered[0],
    }


def describe_array(values: list[float]) -> dict:
    summary = summarize(values)
    summary["values"] = [round(value, 4) for value in values]
    return summary


async def _lag_watcher(
    stop: asyncio.Event,
    samples: list[float],
    active_flag: list[bool] | None = None,
    active_rss: list[float] | None = None,
    active_threads: list[int] | None = None,
) -> None:
    period = 0.01
    next_due = time.perf_counter()
    while not stop.is_set():
        now = time.perf_counter()
        samples.append(max(0.0, (now - next_due) * 1000.0))
        next_due = max(now, next_due + period)
        if active_flag and active_flag[0]:
            sample = _process_sample()
            if active_rss is not None:
                active_rss.append(sample["rss_mb"])
            if active_threads is not None:
                active_threads.append(sample["threads"])
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, next_due - time.perf_counter()))
        except asyncio.TimeoutError:
            pass


def _process_sample() -> dict:
    import psutil

    process = psutil.Process()
    times = process.cpu_times()
    return {
        "threads": process.num_threads(),
        "rss_mb": round(process.memory_info().rss / (1 << 20), 2),
        "cpu_user_s": times.user,
        "cpu_system_s": times.system,
        "ctx_voluntary": process.num_ctx_switches().voluntary,
        "ctx_involuntary": process.num_ctx_switches().involuntary,
    }


async def run_matrix(args) -> dict:
    import psutil

    fixtures = load_fixtures(Path(args.audio_dir))
    model_path = bundled_smart_turn_onnx_path()
    runtime = runtime_record()
    owners = {arm: make_owner(arm, model_path) for arm in ARMS}
    for arm in ARMS:
        await _wait_ready(owners[arm])
    for arm in ARMS:
        for index in range(args.warmup):
            await _run_one(
                owners[arm],
                fixtures[index % len(fixtures)]["audio"],
                -(index + 1),
            )
    stop = asyncio.Event()
    lag_samples: list[float] = []
    watcher = asyncio.create_task(_lag_watcher(stop, lag_samples))
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    execution_order: list[str] = []
    calls = args.calls
    process = psutil.Process()
    cpu_before = process.cpu_times().user + process.cpu_times().system
    wall_before = time.perf_counter()
    pause_id = 0
    for round_index in range(calls):
        fixture = fixtures[round_index % len(fixtures)]
        order = [ARMS[(round_index + offset) % len(ARMS)] for offset in range(len(ARMS))]
        for arm in order:
            execution_order.append(arm)
            pause_id += 1
            record = await _run_one(
                owners[arm], fixture["audio"], pause_id, round_index=round_index
            )
            record["fixture"] = fixture["name"]
            record["fixture_sha256"] = fixture["sha256"]
            results[arm].append(record)
    cpu_after = process.cpu_times().user + process.cpu_times().system
    wall_after = time.perf_counter()
    stop.set()
    await watcher
    for arm in ARMS:
        await owners[arm].close()
    report = {
        "schema": SCHEMA,
        "mode": "matrix",
        "runtime": runtime,
        "calls_per_arm": calls,
        "warmup_per_arm": args.warmup,
        "execution_order": execution_order,
        "fixture_sequence": [
            {
                "round": index,
                "fixture": fixtures[index % len(fixtures)]["name"],
                "fixture_sha256": fixtures[index % len(fixtures)]["sha256"],
            }
            for index in range(calls)
        ],
    }
    timing_keys = (
        "submit_to_receipt_ms",
        "task_start_ms",
        "completed_to_receipt_ms",
        "worker_to_loop_return_ms",
        "features_queue_wait_ms",
        "features_worker_ms",
        "features_worker_to_loop_ms",
        "onnx_queue_wait_ms",
        "onnx_worker_ms",
        "onnx_worker_to_loop_ms",
        "inference_queue_wait_ms",
        "inference_worker_ms",
        "inference_worker_to_loop_ms",
    )
    for arm in ARMS:
        records = results[arm]
        ok = [
            record
            for record in records
            if record["admission"] == "started" and record.get("outcome") == "complete"
        ]
        report[arm] = {
            "admissions": {
                status: sum(1 for record in records if record["admission"] == status)
                for status in ("started", "busy", "unavailable")
            },
            "outcomes": {
                status: sum(1 for record in records if record.get("outcome") == status)
                for status in ("complete", "error", "nonfinite")
            },
            "timings_ms": {
                key: describe_array([record[key] for record in ok if key in record])
                for key in timing_keys
            },
            "records": [
                {key: value for key, value in record.items() if key not in {"trace"}}
                | record.get("trace", {})
                for record in records
            ],
        }
    base = {
        (record["round"], record["fixture"]): record
        for record in results["S12"]
        if record.get("outcome") == "complete"
    }
    parity = {}
    for arm in ("S11", "F12", "F11"):
        matching = [
            (base[(record["round"], record["fixture"])], record)
            for record in results[arm]
            if (record["round"], record["fixture"]) in base and record.get("outcome") == "complete"
        ]
        parity[arm] = {
            "n": len(matching),
            "score_max_abs_diff": (
                max(abs(float(left["score"]) - float(right["score"])) for left, right in matching)
                if matching
                else None
            ),
            "prepared_equal": all(
                left.get("prepared_sha256") == right.get("prepared_sha256")
                for left, right in matching
            ),
            "features_equal": all(
                left.get("features_sha256") == right.get("features_sha256")
                for left, right in matching
            ),
            "model_input_equal": all(
                left.get("model_input_sha256") == right.get("model_input_sha256")
                for left, right in matching
            ),
        }
    report["parity"] = parity
    report["process_all_arms"] = {
        "cpu_time_s": round(cpu_after - cpu_before, 4),
        "wall_s": round(wall_after - wall_before, 4),
        **_process_sample(),
    }
    report["event_loop_lag_ms"] = describe_array(lag_samples)
    return report


def cmd_stages(args) -> int:
    fixtures = load_fixtures(Path(args.audio_dir))
    model_path = bundled_smart_turn_onnx_path()
    report = {"schema": SCHEMA, "mode": "stages", "runtime": runtime_record(), "arms": {}}
    for arm in ARMS:
        sessions = {
            "S12": (1, 2),
            "S11": (1, 1),
            "F12": (1, 2),
            "F11": (1, 1),
        }[arm]
        session = make_session(model_path, inter=sessions[0], intra=sessions[1])
        prepared_all = [
            prepare_smart_turn_audio(f["audio"], sample_rate_hz=16000) for f in fixtures
        ]
        prepare_ms, feature_ms, onnx_ms = [], [], []
        for _ in range(args.repeats):
            for prepared, fixture in zip(prepared_all, fixtures):
                start = time.perf_counter()
                prepare_smart_turn_audio(fixture["audio"], sample_rate_hz=16000)
                prepare_ms.append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                features = compute_whisper_log_mel_features(prepared)
                feature_ms.append((time.perf_counter() - start) * 1000.0)
                assert features.shape == (80, 800)
                start = time.perf_counter()
                outputs = session.run(None, {"input_features": np.expand_dims(features, axis=0)})
                onnx_ms.append((time.perf_counter() - start) * 1000.0)
                assert outputs
        report["arms"][arm] = {
            "prepare_ms": summarize(prepare_ms),
            "features_ms": summarize(feature_ms),
            "onnx_ms": summarize(onnx_ms),
        }
        print(
            f"{arm}: prepare {summarize(prepare_ms)['median']:.3f}ms "
            f"features {summarize(feature_ms)['median']:.3f}ms "
            f"onnx {summarize(onnx_ms)['median']:.3f}ms"
        )
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


class _StubVad:
    def __init__(self) -> None:
        self.segment_id = None
        self.ends: list[SpeechEnd] = []

    def open(self, segment_id) -> None:
        self.segment_id = segment_id

    def seal_active(self, *, reason: str):
        return self._seal(reason)

    def seal_active_for_rollover(self, *, reason: str):
        return self._seal(reason)

    def _seal(self, reason: str):
        if self.segment_id is None:
            return None
        event = SpeechEnd(self.segment_id, reason=reason)
        self.segment_id = None
        self.ends.append(event)
        return event


def _ledger() -> PeerAudioSegmentLedger:
    settings = AudioSegmentSettingsSnapshot(
        provider_id="probe",
        provider_signature=("probe",),
        runtime_signature=("probe",),
        source_mode="manual",
        source_language="en",
        expected_languages=(),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.5,
        vad_hangover_ms=500,
        vad_pre_roll_ms=0,
        delivery_profile_effective="on",
        delivery_availability="ready",
        delivery_threshold=SMART_TURN_COMPLETE_THRESHOLD,
    )
    return PeerAudioSegmentLedger(activation_generation=7, settings=settings)


class _ProbeOwnerProxy:
    def __init__(self, owner: SmartTurnInferenceOwner) -> None:
        self.owner = owner
        self.submit_started_mono: float | None = None
        self.submit_return_mono: float | None = None
        self.callback_entry_mono: float | None = None
        self.last_completion = None

    @property
    def snapshot(self):
        return self.owner.snapshot

    def request_prepare(self) -> None:
        self.owner.request_prepare()

    def submit(self, identity, audio, completion):
        self.submit_started_mono = time.monotonic()

        async def receive(result):
            self.callback_entry_mono = time.monotonic()
            self.last_completion = result
            return await completion(result)

        status = self.owner.submit(identity, audio, receive)
        self.submit_return_mono = time.monotonic()
        return status

    def record_late(self) -> None:
        self.owner.record_late()


class _ProbeListenController(ListenDeliveryController):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probe_submit_started_perf: float | None = None
        self.probe_submit_return_perf: float | None = None
        self.probe_status_at_submit: str = "none"
        self.seals: list[dict] = []
        self.pause_boundary_decision = None
        self.pause_boundary_pause_ms: int | None = None
        self.pause_boundary_source_time_ms: float | None = None
        self.probe_source_time_ms: float | None = None

    def _request_probe(self, activation_generation, segment_id) -> None:
        self.probe_submit_started_perf = time.perf_counter()
        super()._request_probe(activation_generation, segment_id)
        self.probe_status_at_submit = self._probe_status
        self.probe_submit_return_perf = time.perf_counter()

    async def _seal_locked(self, segment_id, *, reason: str, rollover: bool) -> bool:
        decision = self._completion_boundary_decision
        completed = self._completion
        pause_ms = self._observed_pause_ms()
        if pause_ms >= 512 and self.pause_boundary_decision is None:
            self.pause_boundary_decision = decision
            self.pause_boundary_pause_ms = pause_ms
            self.pause_boundary_source_time_ms = self.probe_source_time_ms
        sealed = await super()._seal_locked(segment_id, reason=reason, rollover=rollover)
        if sealed:
            self.seals.append(
                {
                    "decision_before_reset": decision,
                    "reason": reason,
                    "completed_score": completed.score if completed else None,
                    "pause_ms_at_seal": pause_ms,
                    "at_mono": time.monotonic(),
                }
            )
        return sealed


async def _controller_probe(owner: SmartTurnInferenceOwner, audio: np.ndarray) -> dict:
    from puripuly_heart.core.audio.format import AudioCaptureSpan

    vad = _StubVad()
    ledger = _ledger()
    proxy = _ProbeOwnerProxy(owner)
    source_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_samples = 512
    speech_frame_count = 32
    speech_prefix_samples = frame_samples * speech_frame_count
    if source_audio.size < speech_prefix_samples:
        raise ValueError("controller fixture is shorter than the speech prefix")
    speech_prefix = np.ascontiguousarray(source_audio[:speech_prefix_samples])

    async def emit(_event) -> None:
        return None

    controller = _ProbeListenController(
        vad=vad,
        ledger=ledger,
        emit=emit,
        monotonic_clock=time.monotonic,
        smart_turn_owner=proxy,
    )
    record: dict = {
        "frame_period_ms": 32,
        "timeline_ms": {
            "probe_pause": 224,
            "complete_boundary": 512,
            "fallback": 800,
            "speech_prefix": speech_frame_count * 32,
        },
        "speech_prefix_samples": speech_prefix_samples,
        "speech_prefix_sha256": hashlib.sha256(speech_prefix.tobytes()).hexdigest(),
    }
    try:
        segment_id = uuid4()
        vad.open(segment_id)
        origin = time.monotonic()
        silence_frame = np.zeros(frame_samples, dtype=np.float32)
        for frame_index in range(speech_frame_count + 26):
            source_start = origin + frame_index * 0.032
            source_end = source_start + 0.032
            await asyncio.sleep(max(0.0, source_end - time.monotonic()))
            start_sample = frame_index * frame_samples
            capture = (
                AudioCaptureSpan(
                    capture_epoch=1,
                    callback_sequence=frame_index + 1,
                    source_sample_rate_hz=16000,
                    source_start_sample=start_sample,
                    source_end_sample=start_sample + frame_samples,
                    source_start_monotonic_s=source_start,
                    source_end_monotonic_s=source_end,
                    normalized_sample_rate_hz=16000,
                    normalized_start_sample=start_sample,
                    normalized_end_sample=start_sample + frame_samples,
                ),
            )
            is_speech = frame_index < speech_frame_count
            if is_speech:
                chunk = speech_prefix[start_sample : start_sample + frame_samples]
            else:
                chunk = silence_frame
            if frame_index == 0:
                await controller.handle_vad_event(
                    SpeechStart(
                        segment_id,
                        pre_roll=np.empty(0, dtype=np.float32),
                        chunk=chunk,
                        chunk_capture=capture,
                        genuine_onset=True,
                    )
                )
            else:
                await controller.handle_vad_event(
                    SpeechChunk(segment_id, chunk, chunk_capture=capture)
                )
            controller.probe_source_time_ms = (source_end - origin) * 1000.0
            await controller.observe_acoustic_chunk(
                speech_observed=is_speech,
                capture=capture,
            )
            pause_ms = controller._observed_pause_ms()
            if not is_speech and pause_ms >= 512 and controller.pause_boundary_decision is None:
                controller.pause_boundary_decision = controller._completion_boundary_decision
                controller.pause_boundary_pause_ms = pause_ms
                controller.pause_boundary_source_time_ms = (source_end - origin) * 1000.0
            if controller.current_segment_id is None and controller.seals:
                break
        if proxy.callback_entry_mono is None:
            async with asyncio.timeout(30.0):
                while proxy.callback_entry_mono is None:
                    await asyncio.sleep(0.001)
        record["decision_at_pause_512"] = controller.pause_boundary_decision
        record["pause_at_boundary_ms"] = controller.pause_boundary_pause_ms
        record["source_time_at_pause_boundary_ms"] = controller.pause_boundary_source_time_ms
        record["probe_status"] = controller.probe_status_at_submit
        record["seals"] = [
            {key: value for key, value in seal.items() if key != "at_mono"}
            for seal in controller.seals
        ]
        record["sealed"] = controller.current_segment_id is None
        record["seal_reason"] = controller.seals[-1]["reason"] if controller.seals else None
        record["seal_pause_ms"] = (
            controller.seals[-1]["pause_ms_at_seal"] if controller.seals else None
        )
        if proxy.submit_started_mono is not None and proxy.callback_entry_mono is not None:
            record["submit_to_receipt_ms"] = (
                proxy.callback_entry_mono - proxy.submit_started_mono
            ) * 1000.0
        completion = proxy.last_completion
        if completion is not None and proxy.callback_entry_mono is not None:
            record["completed_to_receipt_ms"] = (
                proxy.callback_entry_mono - completion.completed_at_monotonic_s
            ) * 1000.0
            record["score"] = completion.score
            record["outcome"] = completion.outcome
            record["deadline_slack_ms"] = (
                completion.identity.complete_deadline_monotonic_s - proxy.callback_entry_mono
            ) * 1000.0
            record["late_by_deadline"] = (
                proxy.callback_entry_mono >= completion.identity.complete_deadline_monotonic_s
            )
        if (
            controller.probe_submit_started_perf is not None
            and controller.probe_submit_return_perf is not None
        ):
            record["request_probe_call_ms"] = (
                controller.probe_submit_return_perf - controller.probe_submit_started_perf
            ) * 1000.0
        if proxy.submit_started_mono is not None and proxy.submit_return_mono is not None:
            record["owner_submit_call_ms"] = (
                proxy.submit_return_mono - proxy.submit_started_mono
            ) * 1000.0
    finally:
        await controller.close()
    return record


async def run_controller(args) -> dict:
    fixtures = load_fixtures(Path(args.audio_dir))
    speech = [fixture for fixture in fixtures if fixture["kind"] == "speech"]
    model_path = bundled_smart_turn_onnx_path()
    report = {
        "schema": SCHEMA,
        "mode": "controller",
        "runtime": runtime_record(),
        "timeline": {"probe_pause_ms": 224, "complete_boundary_ms": 512, "fallback_ms": 800},
        "arms": {},
    }
    for arm in args.arms.split(","):
        owner = make_owner(arm, model_path)
        await _wait_ready(owner)
        probes = []
        for fixture in speech:
            probe = await _controller_probe(owner, fixture["audio"])
            probe["fixture"] = fixture["name"]
            probe["fixture_sha256"] = fixture["sha256"]
            probes.append(probe)
        await owner.close()
        receipts = [probe["submit_to_receipt_ms"] for probe in probes]
        report["arms"][arm] = {
            "n": len(probes),
            "submit_to_receipt_ms": summarize(receipts),
            "min_deadline_slack_ms": round(min(probe["deadline_slack_ms"] for probe in probes), 3),
            "decision_at_pause_512": {
                decision: sum(1 for probe in probes if probe["decision_at_pause_512"] == decision)
                for decision in ("early", "incomplete", None)
            },
            "pause_at_boundary_ms": summarize([probe["pause_at_boundary_ms"] for probe in probes]),
            "seal_pause_ms": summarize([probe["seal_pause_ms"] for probe in probes]),
            "seal_reasons": {
                reason: sum(1 for probe in probes if probe["seal_reason"] == reason)
                for reason in ("delivery_pause", "delivery_deadline", None)
            },
            "late_count": sum(1 for probe in probes if probe["late_by_deadline"]),
            "scores": [probe["score"] for probe in probes],
            "speech_prefix_samples": sorted({probe["speech_prefix_samples"] for probe in probes}),
            "speech_prefix_sha256": sorted({probe["speech_prefix_sha256"] for probe in probes}),
            "probes": probes,
        }
    decisions = {
        arm: [probe["decision_at_pause_512"] for probe in data["probes"]]
        for arm, data in report["arms"].items()
    }
    first = next(iter(decisions.values()))
    report["decision_parity"] = all(sequence == first for sequence in decisions.values())
    return report


async def run_paced(args) -> dict:
    import psutil

    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
    from puripuly_heart.core.vad.gating import create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    fixtures = load_fixtures(Path(args.audio_dir))
    speech = [fixture for fixture in fixtures if fixture["kind"] == "speech"]
    model_path = bundled_smart_turn_onnx_path()
    sequence = [
        {
            "round": round_index,
            "fixture": fixture["name"],
            "fixture_sha256": fixture["sha256"],
        }
        for round_index in range(args.rounds)
        for fixture in speech
    ]
    report = {
        "schema": SCHEMA,
        "mode": "paced",
        "runtime": runtime_record(),
        "gap_s": args.gap,
        "rounds": args.rounds,
        "coload": args.coload,
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(
            json.dumps(sequence, sort_keys=True).encode()
        ).hexdigest(),
        "arms": {},
    }
    for arm in args.arms.split(","):
        owner = make_owner(arm, model_path)
        await _wait_ready(owner)
        vad_gating = None
        if args.coload == "silero":
            vad_gating = create_peer_vad_gating(
                SileroVadOnnx(bundled_silero_vad_onnx_path()),
                sample_rate_hz=16000,
                ring_buffer_ms=500,
                hangover_ms=500,
            )
        process = psutil.Process()
        stop = asyncio.Event()
        active_flag = [False]
        lag_samples: list[float] = []
        active_rss: list[float] = []
        active_threads: list[int] = []
        watcher = asyncio.create_task(
            _lag_watcher(
                stop,
                lag_samples,
                active_flag,
                active_rss,
                active_threads,
            )
        )
        cpu_before = process.cpu_times().user + process.cpu_times().system
        wall_before = time.perf_counter()
        idle_cpu_s = 0.0
        request_window_cpu_s = 0.0
        receipts: list[float] = []
        completed = 0
        idle_rss: list[float] = []
        capture_lags: list[float] = []
        missed_slots = 0
        vad_calls = 0
        endpoint_counts = {"SpeechStart": 0, "SpeechEnd": 0}
        fixture_records = []
        pause_id = 0

        async def replay_vad(audio: np.ndarray) -> dict:
            nonlocal missed_slots, vad_calls
            if args.coload == "none":
                return {"calls": 0, "missed_slots": 0, "endpoint_counts": {}}
            vad_gating.reset()
            frame_count = audio.size // 512
            capture_start = time.perf_counter()
            local_lags: list[float] = []
            local_missed = 0
            local_endpoints = {"SpeechStart": 0, "SpeechEnd": 0}
            for frame_index in range(frame_count):
                source_end = capture_start + (frame_index + 1) * 0.032
                await asyncio.sleep(max(0.0, source_end - time.perf_counter()))
                lag_ms = max(0.0, (time.perf_counter() - source_end) * 1000.0)
                local_lags.append(lag_ms)
                if lag_ms >= 32.0:
                    local_missed += 1
                frame = audio[frame_index * 512 : (frame_index + 1) * 512]
                events = vad_gating.process_chunk(frame)
                for event in events:
                    event_name = type(event).__name__
                    if event_name in local_endpoints:
                        local_endpoints[event_name] += 1
            if vad_gating.in_speech:
                endpoint_outcome = "open_external_boundary"
            else:
                endpoint_outcome = "sealed"
            return {
                "calls": frame_count,
                "missed_slots": local_missed,
                "capture_lag_ms": describe_array(local_lags),
                "endpoint_counts": local_endpoints,
                "endpoint_outcome": endpoint_outcome,
            }

        for round_index in range(args.rounds):
            for fixture in speech:
                pause_id += 1
                active_flag[0] = True
                active_before = process.cpu_times().user + process.cpu_times().system
                inference_task = asyncio.create_task(
                    _run_one(
                        owner,
                        fixture["audio"],
                        pause_id,
                        round_index=round_index,
                    )
                )
                vad_task = asyncio.create_task(replay_vad(fixture["audio"]))
                try:
                    record, vad_record = await asyncio.gather(inference_task, vad_task)
                finally:
                    active_flag[0] = False
                active_after = process.cpu_times().user + process.cpu_times().system
                request_window_cpu_s += active_after - active_before
                if record.get("outcome") == "complete":
                    completed += 1
                    receipts.append(record["submit_to_receipt_ms"])
                vad_calls += vad_record.get("calls", 0)
                missed_slots += vad_record.get("missed_slots", 0)
                capture_lags.extend(vad_record.get("capture_lag_ms", {}).get("values", []))
                for key, value in vad_record.get("endpoint_counts", {}).items():
                    endpoint_counts[key] = endpoint_counts.get(key, 0) + value
                fixture_records.append(
                    {
                        "round": round_index,
                        "fixture": fixture["name"],
                        "fixture_sha256": fixture["sha256"],
                        "inference": record,
                        "vad": vad_record,
                    }
                )
                idle_before = process.cpu_times().user + process.cpu_times().system
                await asyncio.sleep(args.gap)
                idle_after = process.cpu_times().user + process.cpu_times().system
                idle_cpu_s += idle_after - idle_before
                idle_rss.append(_process_sample()["rss_mb"])
        wall_s = time.perf_counter() - wall_before
        cpu_s = (process.cpu_times().user + process.cpu_times().system) - cpu_before
        stop.set()
        await watcher
        await owner.close()
        report["arms"][arm] = {
            "submit_to_receipt_ms": describe_array(receipts),
            "fixed_replay_wall_s": round(wall_s, 3),
            "coload_process_cpu_s": round(cpu_s, 4),
            "coload_process_cpu_per_completed_request_s": (
                round(cpu_s / completed, 6) if completed else None
            ),
            "request_window_cpu_s": round(request_window_cpu_s, 4),
            "request_window_cpu_per_completed_request_s": (
                round(request_window_cpu_s / completed, 6) if completed else None
            ),
            "idle_gap_cpu_s": round(idle_cpu_s, 4),
            "idle_gap_cpu_per_gap_s": (
                round(idle_cpu_s / (args.rounds * len(speech)), 6)
                if args.rounds and speech
                else None
            ),
            "rss_idle_mb": describe_array(idle_rss),
            "rss_active_mb": describe_array(active_rss),
            "threads_active": describe_array([float(value) for value in active_threads]),
            "event_loop_lag_ms": describe_array(lag_samples),
            "capture_frame_lag_ms": describe_array(capture_lags),
            "capture_frame_missed_slots": missed_slots,
            "vad_calls": vad_calls,
            "endpoint_counts": endpoint_counts,
            "fixture_records": fixture_records,
        }
    return report


async def run_isolated_cpu(args) -> dict:
    import psutil

    fixtures = load_fixtures(Path(args.audio_dir))
    speech = [fixture for fixture in fixtures if fixture["kind"] == "speech"]
    model_path = bundled_smart_turn_onnx_path()
    sequence = [
        {
            "round": index // len(speech),
            "fixture": speech[index % len(speech)]["name"],
            "fixture_sha256": speech[index % len(speech)]["sha256"],
        }
        for index in range(args.calls)
    ]
    report = {
        "schema": SCHEMA,
        "mode": args.command,
        "runtime": runtime_record(),
        "calls": args.calls,
        "warmup": args.warmup,
        "coload": "none",
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(
            json.dumps(sequence, sort_keys=True).encode()
        ).hexdigest(),
        "cpu_counter_source": "psutil.Process.cpu_times user+system",
        "process_time_clock_resolution_s": time.get_clock_info("process_time").resolution,
        "arms": {},
    }
    for arm in args.arms.split(","):
        owner = make_owner(arm, model_path)
        await _wait_ready(owner)
        pause_id = 0
        for index in range(args.warmup):
            pause_id += 1
            await _run_one(
                owner,
                speech[index % len(speech)]["audio"],
                pause_id,
                round_index=index // len(speech),
            )
        measured: list[dict] = []
        cpu_deltas: list[float] = []
        process = psutil.Process()
        cpu_before = process.cpu_times().user + process.cpu_times().system
        for index in range(args.calls):
            pause_id += 1
            fixture = speech[index % len(speech)]
            before = process.cpu_times().user + process.cpu_times().system
            measured.append(
                await _run_one(
                    owner,
                    fixture["audio"],
                    pause_id,
                    round_index=index // len(speech),
                )
            )
            after = process.cpu_times().user + process.cpu_times().system
            cpu_deltas.append(after - before)
        cpu_after = process.cpu_times().user + process.cpu_times().system
        await owner.close()
        nonzero_deltas = [delta for delta in cpu_deltas if delta > 0.0]
        receipts = [
            record["submit_to_receipt_ms"]
            for record in measured
            if record.get("outcome") == "complete"
        ]
        report["arms"][arm] = {
            "admissions": {
                status: sum(1 for record in measured if record.get("admission") == status)
                for status in ("started", "busy", "unavailable")
            },
            "completed": len(receipts),
            "submit_to_receipt_ms": summarize(receipts),
            "process_cpu_s": round(cpu_after - cpu_before, 6),
            "process_cpu_per_request_s": round((cpu_after - cpu_before) / args.calls, 6),
            "process_cpu_counter_nonzero_quantum_s": (
                round(min(nonzero_deltas), 6) if nonzero_deltas else None
            ),
            "process_cpu_counter_deltas_s": [round(delta, 6) for delta in cpu_deltas],
            "records": measured,
        }
    return report


def run_hann_probe(args) -> dict:
    import puripuly_heart.core.audio.smart_turn_features as feature_module

    fixtures = load_fixtures(Path(args.audio_dir))
    fixture = next(item for item in fixtures if item["kind"] == "speech")
    prepared = prepare_smart_turn_audio(fixture["audio"], sample_rate_hz=SMART_TURN_SAMPLE_RATE_HZ)

    def power_spectrogram(audio: np.ndarray, cached: bool) -> np.ndarray:
        padded = np.pad(
            np.asarray(audio, dtype=np.float64),
            (feature_module._N_FFT // 2, feature_module._N_FFT // 2),
            mode="reflect",
        )
        windows = feature_module.sliding_window_view(padded, feature_module._N_FFT)[
            :: feature_module._HOP_LENGTH
        ]
        window = (
            feature_module._HANN_WINDOW
            if cached
            else feature_module._HANN_WINDOW.astype(np.float64)
        )
        spectrum = np.fft.rfft(windows * window, axis=-1)
        return (np.abs(spectrum) ** 2).T

    def compute(audio: np.ndarray, cached: bool) -> np.ndarray:
        value = np.asarray(audio, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError(f"audio must be one-dimensional, got {value.shape}")
        expected_samples = feature_module._SAMPLE_RATE_HZ * 8
        if value.size < expected_samples:
            value = np.pad(value, (0, expected_samples - value.size), mode="constant")
        elif value.size > expected_samples:
            value = value[:expected_samples]
        value = (value - value.mean()) / np.sqrt(
            value.var() + feature_module._NORMALIZATION_EPSILON
        )
        mel = np.maximum(
            feature_module._MEL_FLOOR,
            feature_module._MEL_FILTERS.T @ power_spectrogram(value, cached),
        )
        log_mel = np.log10(mel)[:, :-1]
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        return ((log_mel + 4.0) / 4.0).astype(np.float32)

    for cached in (False, True):
        compute(prepared, cached)
    baseline_values: list[float] = []
    cached_values: list[float] = []
    baseline_output = None
    cached_output = None
    for _ in range(args.repeats):
        started = time.perf_counter_ns()
        baseline_output = compute(prepared, False)
        baseline_values.append((time.perf_counter_ns() - started) / 1_000_000.0)
        started = time.perf_counter_ns()
        cached_output = compute(prepared, True)
        cached_values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    cast_values: list[float] = []
    for _ in range(args.repeats * 10):
        started = time.perf_counter_ns()
        feature_module._HANN_WINDOW.astype(np.float64)
        cast_values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    identity_values: list[float] = []
    for _ in range(args.repeats * 10):
        started = time.perf_counter_ns()
        _ = feature_module._HANN_WINDOW
        identity_values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    max_abs_diff = float(np.max(np.abs(baseline_output - cached_output)))
    baseline_summary = summarize(baseline_values)
    cached_summary = summarize(cached_values)
    return {
        "schema": SCHEMA,
        "mode": "hann",
        "fixture": fixture["name"],
        "fixture_sha256": fixture["sha256"],
        "repeats": args.repeats,
        "prepared_samples": int(prepared.size),
        "baseline_full_features_ms": baseline_summary,
        "cached_hann_full_features_ms": cached_summary,
        "full_feature_delta_ms": {
            "median": cached_summary["median"] - baseline_summary["median"],
            "p95": cached_summary["p95"] - baseline_summary["p95"],
        },
        "hann_cast_only_ms": summarize(cast_values),
        "hann_identity_only_ms": summarize(identity_values),
        "max_abs_output_diff": max_abs_diff,
        "disposition": "reject: cached Hann is numerically identical but the measured end-to-end saving is negligible relative to SmartTurn inference",
    }


def cmd_hann(args) -> int:
    report = run_hann_probe(args)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


def cmd_support(args) -> int:
    import onnxruntime as ort

    options = ort.SessionOptions()
    supported: dict = {}
    for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
        try:
            options.add_session_config_entry(key, "0")
            supported[key] = f"accepted, current={options.get_session_config_entry(key)!r}"
        except Exception as exc:
            supported[key] = f"rejected: {type(exc).__name__}: {exc}"
    print(json.dumps(supported, indent=2))
    return 0


async def _async_main(args) -> int:
    if args.command == "matrix":
        report = await run_matrix(args)
    elif args.command == "controller":
        report = await run_controller(args)
    elif args.command == "paced":
        report = await run_paced(args)
    elif args.command in {"cpu", "recheck"}:
        report = await run_isolated_cpu(args)
    else:
        raise ValueError(args.command)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    for arm, data in report.get("arms", {}).items():
        timing = data.get("timings_ms", data.get("submit_to_receipt_ms", {}))
        if "submit_to_receipt_ms" in timing:
            timing = timing["submit_to_receipt_ms"]
        print(
            f"  {arm}: n={timing.get('n')} median={timing.get('median')} "
            f"p95={timing.get('p95')} worst={timing.get('worst')}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default=".data/smartturn-179/audio")
    parser.add_argument("--out", default=".data/smartturn-179/result.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--calls", type=int, default=1)
    stages = sub.add_parser("stages")
    stages.add_argument("--repeats", type=int, default=3)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--calls", type=int, default=40)
    matrix.add_argument("--warmup", type=int, default=5)
    controller = sub.add_parser("controller")
    controller.add_argument("--arms", default="S12,S11,F12,F11")
    paced = sub.add_parser("paced")
    paced.add_argument("--arms", default="S12")
    paced.add_argument("--rounds", type=int, default=2)
    paced.add_argument("--gap", type=float, default=2.0)
    paced.add_argument("--coload", default="silero", choices=["none", "silero"])
    cpu = sub.add_parser("cpu")
    cpu.add_argument("--arms", default="S12,S11,F12,F11")
    cpu.add_argument("--calls", type=int, default=24)
    cpu.add_argument("--warmup", type=int, default=5)
    recheck = sub.add_parser("recheck")
    recheck.add_argument("--arms", default="S12,F12")
    recheck.add_argument("--calls", type=int, default=24)
    recheck.add_argument("--warmup", type=int, default=5)
    hann = sub.add_parser("hann")
    hann.add_argument("--repeats", type=int, default=20)
    sub.add_parser("support")
    args = parser.parse_args()
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "stages":
        return cmd_stages(args)
    if args.command == "hann":
        return cmd_hann(args)
    if args.command == "support":
        return cmd_support(args)
    if args.command == "smoke":
        args.warmup = 1
        args.command = "matrix"
        return asyncio.run(_async_main(args))
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
