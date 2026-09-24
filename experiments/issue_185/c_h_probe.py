from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import threading
import time
import tracemalloc
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes, pcm16le_bytes_to_float32
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend


class Client:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._closed = False
        self._events = asyncio.Event()
        self.path: Path | None = None
        self.wav: bytes | None = None

    @property
    def pid(self) -> int:
        return 1

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def activate(self, *, model_path: Path, device_id: str):
        from puripuly_heart.app.ports.gpu_worker import GpuWorkerActivation, GpuWorkerDevice

        return GpuWorkerActivation(
            device=GpuWorkerDevice(
                device_id=device_id,
                registry_index=0,
                name="stub",
                description="stub",
                device_type="discrete",
                memory_total_bytes=1,
                memory_free_bytes=1,
            ),
            model_load_seconds=0.0,
            warmup_seconds=0.0,
        )

    async def transcribe(
        self, *, request_id, channel, audio_path, language_hint=None, on_request_sent=None
    ):
        from puripuly_heart.app.ports.gpu_worker import GpuWorkerTranscription

        self.path = audio_path
        with wave.open(str(audio_path), "rb") as wav:
            self.wav = wav.readframes(wav.getnframes())
        if on_request_sent:
            on_request_sent()
        self.started.set()
        return GpuWorkerTranscription(
            text="valid", detected_language="en", audio_seconds=1.0, decode_seconds=0.01, rtf=0.01
        )

    async def next_event(self):
        from puripuly_heart.app.ports.gpu_worker import GpuWorkerClosedError

        await self._events.wait()
        raise GpuWorkerClosedError("closed")

    async def cancel(self, target_request_id):
        return None

    async def close(self):
        self._closed = True
        self._events.set()

    async def force_close(self):
        await self.close()


class Factory:
    def __init__(self, client):
        self.client = client

    async def start(self, *, mode):
        return self.client


async def run() -> dict:
    client = Client()
    events = []
    runtime = SharedGpuASRRuntime(process_factory=Factory(client), diagnostic_sink=events.append)
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=Path("model.gguf"),
        model_id="fixture",
        device_id="stub",
    )
    session = await backend.open_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="probe")
    )
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1, segment_order=1, segment_id=uuid4(), capture_epoch=1
        ),
        provider_epoch_id="probe",
        provider_turn_id="probe-turn",
    )
    request = STTProviderTurnRequest(
        identity=identity,
        channel="self",
        settings=AudioSegmentSettingsSnapshot(
            provider_id="local_qwen_gpu",
            provider_signature=("local_qwen_gpu",),
            runtime_signature=("local_qwen_gpu",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.4,
            vad_hangover_ms=800,
            vad_pre_roll_ms=500,
        ),
    )
    samples = np.tile(
        np.array([-1.0, -0.9, -0.5, -0.00003, 0.0, 0.00003, 0.5, 0.9, 1.0], dtype=np.float32), 1778
    )
    release = threading.Event()
    entered = threading.Event()
    original_unlink = Path.unlink

    def gated_unlink(path, *args, **kwargs):
        if path.suffix == ".wav":
            entered.set()
            if not release.wait(3):
                raise TimeoutError("probe unlink gate")
        return original_unlink(path, *args, **kwargs)

    Path.unlink = gated_unlink
    try:
        start = time.perf_counter()
        await session.begin_turn(request)
        await session.send_turn_audio(
            identity,
            float32_to_pcm16le_bytes(samples),
            payload_sequence=1,
            source_ranges=(),
            context_only=False,
        )
        await session.seal_turn(
            identity, sealed_content_ranges=(), seal_reason="end", observed_trailing_silence_ms=800
        )
        pending = asyncio.create_task(anext(session.turn_events()))
        await asyncio.wait_for(client.started.wait(), timeout=5)
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
        delete_entered = time.perf_counter()
        await asyncio.sleep(0.02)
        delivered_during_delete = pending.done()
        release.set()
        result = await asyncio.wait_for(pending, timeout=5)
        delivered = time.perf_counter()
        assert result.text == "valid"
        assert client.wav is not None
        baseline_bytes = (
            np.rint(
                np.clip(pcm16le_bytes_to_float32(float32_to_pcm16le_bytes(samples)), -1, 1) * 32767
            )
            .astype("<i2")
            .tobytes()
        )
        assert client.wav == baseline_bytes
    finally:
        release.set()
        Path.unlink = original_unlink
        await session.close()
        await backend.close()
        await runtime.close()

    counts = []
    candidate_counts = []
    for _ in range(40):
        tracemalloc.start()
        begin = time.perf_counter_ns()
        pcm = float32_to_pcm16le_bytes(samples)
        f32 = pcm16le_bytes_to_float32(pcm).copy()
        joined = np.concatenate([f32[:5000], f32[5000:]])
        result_bytes = np.rint(np.clip(joined, -1, 1) * 32767).astype("<i2").tobytes()
        elapsed = time.perf_counter_ns() - begin
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert result_bytes == baseline_bytes
        counts.append((elapsed, peak))
        tracemalloc.start()
        begin = time.perf_counter_ns()
        pcm = float32_to_pcm16le_bytes(samples)
        joined_pcm = b"".join((pcm[:10000], pcm[10000:]))
        f32 = pcm16le_bytes_to_float32(joined_pcm)
        result_bytes = np.rint(np.clip(f32, -1, 1) * 32767).astype("<i2").tobytes()
        elapsed = time.perf_counter_ns() - begin
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert result_bytes == baseline_bytes
        candidate_counts.append((elapsed, peak))
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "profile": "stub owner, 16002 mono samples, 40 fixed-input representation iterations, no Vulkan inference",
        "sample_count": len(samples),
        "delivered_during_blocked_unlink": delivered_during_delete,
        "worker_to_delete_gate_ms": round((delete_entered - start) * 1000, 3),
        "delete_gate_to_result_ms": round((delivered - delete_entered) * 1000, 3),
        "baseline_pcm_bytes": len(pcm),
        "baseline_representations": "f32->pcm16 bytes->f32 conversion->copy->concat->re-quantized pcm16 WAV",
        "representation_median_us": round(float(np.median([n / 1000 for n, _ in counts])), 3),
        "representation_peak_bytes_median": int(np.median([peak for _, peak in counts])),
        "candidate_representation_median_us": round(
            float(np.median([n / 1000 for n, _ in candidate_counts])), 3
        ),
        "candidate_representation_peak_bytes_median": int(
            np.median([peak for _, peak in candidate_counts])
        ),
        "diagnostics": [
            {"kind": e.kind, "fields": dict(e.fields)}
            for e in events
            if e.kind in ("decode_timing", "file_cleanup", "decode_attempt")
        ],
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), sort_keys=True))
