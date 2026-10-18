from __future__ import annotations

import asyncio
import json
import os
import time
import wave
from pathlib import Path

import numpy as np

from bench import observe
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.source import PhysicalCaptureProgression

SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 320
CONTROL_ENV = "DIARIZATION_BENCH_CONTROL_DIR"


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        facts = source.getframerate(), source.getnchannels(), source.getsampwidth()
        if facts != (SAMPLE_RATE_HZ, 1, 2):
            raise ValueError(f"benchmark audio must be 16 kHz mono PCM16: {path} {facts}")
        data = source.readframes(source.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class BenchLoopbackSource:
    """Plays the queued benchmark case in real time, then continues with silence like a device."""

    def __init__(self, device_name: str = "", **_: object) -> None:
        control = Path(os.environ[CONTROL_ENV])
        request = json.loads((control / "next_case.json").read_text(encoding="utf-8"))
        self.case_id = request["case_id"]
        self.record_path = Path(request["record_path"])
        self.lead_in_samples = int(round(float(request.get("lead_in_s", 1.0)) * SAMPLE_RATE_HZ))
        self.audio = read_pcm16_mono(Path(request["wav"]))
        self.requested_device_name = device_name
        self._progression = PhysicalCaptureProgression(sample_rate_hz=SAMPLE_RATE_HZ)
        self._closed = False
        self._lags: list[float] = []
        self._stalls: list[tuple[float, float]] = []
        self._origin: float | None = None
        self._emitted = 0
        self._case_start_monotonic: float | None = None
        self._case_end_monotonic: float | None = None
        self._capture_epoch: int | None = None
        observe.current_case["case_id"] = self.case_id
        observe.emit(
            "source_open",
            device_name=device_name,
            lead_in_samples=self.lead_in_samples,
            case_samples=int(self.audio.size),
        )

    resolved_device_name = "diarization-bench"
    resolved_device_index = -1
    resolved_channels = 1
    actual_sample_rate_hz = SAMPLE_RATE_HZ
    used_default_fallback = False
    callback_status_count = 0
    queue_drop_count = 0
    last_callback_status = None

    @property
    def capture_progression_snapshot(self):
        return self._progression.snapshot

    def _samples(self, start: int, count: int) -> np.ndarray:
        out = np.zeros(count, dtype=np.float32)
        audio_start = start - self.lead_in_samples
        lo = max(0, audio_start)
        hi = min(self.audio.size, audio_start + count)
        if hi > lo:
            out[lo - audio_start : hi - audio_start] = self.audio[lo:hi]
        return out

    async def frames(self):
        self._origin = time.monotonic()
        case_end = self.lead_in_samples + self.audio.size
        while not self._closed:
            due = self._origin + (self._emitted + FRAME_SAMPLES) / SAMPLE_RATE_HZ
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if self._closed:
                return
            now = time.monotonic()
            self._lags.append(now - due)
            if now - due > 0.1:
                self._stalls.append((round(self._emitted / SAMPLE_RATE_HZ, 3), round(now - due, 4)))
            if len(self._lags) > 100_000:
                self._lags = self._lags[-50_000:]
            samples = self._samples(self._emitted, FRAME_SAMPLES)
            device_clock = self._origin + (self._emitted + FRAME_SAMPLES) / SAMPLE_RATE_HZ
            capture = self._progression.observe_frame(
                sample_count=FRAME_SAMPLES, observed_at_monotonic_s=device_clock
            )
            self._capture_epoch = capture.capture_epoch
            if (
                self._case_start_monotonic is None
                and capture.source_end_sample > self.lead_in_samples
            ):
                self._case_start_monotonic = (
                    capture.source_start_monotonic_s
                    + (self.lead_in_samples - capture.source_start_sample) / SAMPLE_RATE_HZ
                )
            if self._case_end_monotonic is None and capture.source_end_sample >= case_end:
                self._case_end_monotonic = (
                    capture.source_start_monotonic_s
                    + (case_end - capture.source_start_sample) / SAMPLE_RATE_HZ
                )
            self._emitted += FRAME_SAMPLES
            self._progression.mark_admitted(capture)
            yield AudioFrameF32(
                samples=samples,
                sample_rate_hz=SAMPLE_RATE_HZ,
                channels=1,
                capture=capture,
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        lags = np.asarray(self._lags or [0.0])
        record = {
            "case_id": self.case_id,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frame_samples": FRAME_SAMPLES,
            "lead_in_samples": self.lead_in_samples,
            "case_samples": int(self.audio.size),
            "emitted_samples": self._emitted,
            "capture_epoch": self._capture_epoch,
            "playback_origin_monotonic": self._origin,
            "case_start_monotonic": self._case_start_monotonic,
            "case_end_monotonic": self._case_end_monotonic,
            "case_completed": self._case_end_monotonic is not None,
            "stalls_over_100ms": self._stalls[:200],
            "frame_lag_s": {
                "count": int(len(self._lags)),
                "p50": float(np.percentile(lags, 50)),
                "p99": float(np.percentile(lags, 99)),
                "max": float(lags.max()),
            },
        }
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        observe.emit("source_close", record=record)
