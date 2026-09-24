"""Opt-in synthetic producer through the actual SoundDeviceAudioSource callback/Janus/VAD loop."""

from __future__ import annotations

import asyncio
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import sounddevice as sd

from puripuly_heart.core.audio.source import SoundDeviceAudioSource
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.vad.gating import VadGating


def dist(values):
    seq = sorted(values)
    return (
        {
            "n": len(seq),
            "min": seq[0],
            "p50": statistics.median(seq),
            "p95": seq[int((len(seq) - 1) * 0.95)],
            "max": seq[-1],
        }
        if seq
        else {"n": 0}
    )


class FakeStream:
    def __init__(self, *, callback, samplerate, channels, blocksize, duration_s, **kwargs):
        self.callback = callback
        self.samplerate = 48000 if samplerate is None else samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.duration_s = duration_s
        self.done = threading.Event()
        self.stop_event = threading.Event()
        self.lags = []
        self.samples = np.zeros((480, channels), dtype=np.float32)
        self.thread = None

    def start(self):
        def run():
            start = time.monotonic()
            for i in range(round(self.duration_s / 0.010)):
                due = start + i * 0.010
                delay = due - time.monotonic()
                if delay > 0 and self.stop_event.wait(delay):
                    break
                if self.stop_event.is_set():
                    break
                self.lags.append((time.monotonic() - due) * 1000)
                self.callback(self.samples, 480, None, False)
            self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def close(self):
        self.stop()


class Source(SoundDeviceAudioSource):
    def __post_init__(self):
        self.trace = []
        self.highwater = 0
        super().__post_init__()

    async def frames(self):
        q = self._queue
        while not self._stream.done.is_set() or not q.async_q.empty():
            with q._sync_mutex:
                queued = tuple(q._queue)
            n = len(queued)
            self.highwater = max(self.highwater, n)
            now = time.monotonic()
            oldest = (now - queued[0].capture.source_end_monotonic_s) * 1000 if n else None
            newest = (now - queued[-1].capture.source_end_monotonic_s) * 1000 if n else None
            try:
                frame = q.async_q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
                continue
            if frame is None:
                break
            capture = frame.capture
            self.trace.append(
                {
                    "seq": capture.callback_sequence,
                    "start": capture.source_start_sample,
                    "end": capture.source_end_sample,
                    "capture_end": capture.source_end_monotonic_s,
                    "epoch": capture.capture_epoch,
                    "gap": (
                        capture.discontinuity_before.kind if capture.discontinuity_before else None
                    ),
                    "gap_samples": (
                        capture.discontinuity_before.lost_source_samples
                        if capture.discontinuity_before
                        else 0
                    ),
                    "queue_depth": n,
                    "oldest_age_ms": oldest,
                    "newest_age_ms": newest,
                    "frame_age_ms": (now - capture.source_end_monotonic_s) * 1000,
                }
            )
            yield frame


class VadEngine:
    def speech_probability(self, samples, *, sample_rate_hz):
        return 0.0

    def reset(self):
        pass


class TracedVad(VadGating):
    def __init__(self, engine, *, sample_rate_hz):
        super().__init__(engine, sample_rate_hz=sample_rate_hz)
        self.input_end_ages_ms = []

    def process_owned_chunk(self, chunk, capture):
        if capture:
            self.input_end_ages_ms.append(
                (time.monotonic() - capture[-1].source_end_monotonic_s) * 1000
            )
        return super().process_owned_chunk(chunk, capture)


class SlowSink:
    def __init__(self, delay):
        self.delay = delay
        self.chunks = 0
        self.events = []

    async def handle_vad_event(self, event):
        self.events.append(type(event).__name__)

    async def observe_source_activity(self, *, speech_observed, observed_at_monotonic_s):
        self.chunks += 1
        await asyncio.sleep(self.delay)


async def scenario(delay, duration):
    original = sd.InputStream
    stream = []

    def factory(**kwargs):
        candidate = FakeStream(duration_s=duration, **kwargs)
        stream.append(candidate)
        return candidate

    sd.InputStream = factory
    try:
        source = Source(sample_rate_hz=48000, channels=1, blocksize=0, max_queue_frames=64)
    finally:
        sd.InputStream = original
    sink = SlowSink(delay)
    vad = TracedVad(VadEngine(), sample_rate_hz=16000)
    start = time.monotonic()
    try:
        await asyncio.wait_for(
            run_audio_vad_loop(source=source, vad=vad, sink=sink, target_sample_rate_hz=16000),
            timeout=20,
        )
        elapsed = time.monotonic() - start
    finally:
        await source.close()
    trace = source.trace
    gaps = [
        {
            "seq": o["seq"],
            "epoch": o["epoch"],
            "kind": o["gap"],
            "source_start": o["start"],
            "missing_samples": o["gap_samples"],
        }
        for o in trace
        if o["gap"]
    ]
    dropped = source.queue_drop_count
    counted = sum(g["missing_samples"] or 0 for g in gaps)
    produced = source.capture_progression_snapshot.next_callback_sequence
    return {
        "delay_each_vad_chunk_ms": delay * 1000,
        "callback_spacing_ms": 10,
        "producer_duration_s": duration,
        "total_elapsed_s": elapsed,
        "produced": produced,
        "consumed": len(trace),
        "callback_dropped": dropped,
        "drop_ranges": gaps,
        "known_dropped_frames_before_resumed": counted // 480,
        "trailing_dropped_frames": produced - len(trace) - counted // 480,
        "highwater_frames": source.highwater,
        "vad_chunks": sink.chunks,
        "queue_depth_at_dequeue": dist([x["queue_depth"] for x in trace]),
        "oldest_frame_age_ms": dist(
            [x["oldest_age_ms"] for x in trace if x["oldest_age_ms"] is not None]
        ),
        "newest_frame_age_ms": dist(
            [x["newest_age_ms"] for x in trace if x["newest_age_ms"] is not None]
        ),
        "capture_to_vad_at_dequeue_lower_bound_ms": dist([x["frame_age_ms"] for x in trace]),
        "capture_to_vad_observation_ms": dist(vad.input_end_ages_ms),
        "producer_scheduling_lag_ms": dist(stream[0].lags),
        "note": "Capture-to-VAD observation uses last actual mapped source span of each 512-sample chunk. Source-frame dequeue is a distinct lower bound. No raw PCM persisted. Gap ranges derive actual PhysicalCaptureProgression, not synthetic silence.",
    }


async def main():
    result = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {x: version(x) for x in ("janus", "numpy", "sounddevice", "soxr")},
        "owner": "SoundDeviceAudioSource callback -> janus.Queue(64) -> Source.frames instrumentation -> run_audio_vad_loop -> CaptureMappedStreamingResampler -> VadGating",
        "scenarios": [await scenario(0.002, 2.4), await scenario(0.055, 2.4)],
    }
    Path(__file__).with_name("f_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "delay_ms": x["delay_each_vad_chunk_ms"],
                    "drops": x["callback_dropped"],
                    "highwater": x["highwater_frames"],
                    "gaps": len(x["drop_ranges"]),
                    "ages": x["oldest_frame_age_ms"],
                }
                for x in result["scenarios"]
            ]
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
