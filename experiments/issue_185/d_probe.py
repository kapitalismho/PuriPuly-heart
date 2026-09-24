"""Opt-in metadata-only microphone comparison; no PCM is written to disk."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import sounddevice as sd

from puripuly_heart.core.audio.source import (
    SoundDeviceAudioSource,
    determine_self_mic_capture_channels,
)
from puripuly_heart.core.audio.streaming_resampler import CaptureMappedStreamingResampler
from puripuly_heart.core.runtime.audio_vad_loop import _capture_prefix


def distribution(values):
    values = sorted(x for x in values if math.isfinite(x))
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": values[0],
        "p50": values[len(values) // 2],
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def completed_vad_frontiers(mapped, output_count, capture_buffer, available, ready_at):
    """Apply the production VAD capture-prefix split to each completed 512-sample block."""
    if output_count:
        if mapped is None:
            raise RuntimeError("Normalized output has no mapped capture")
        capture_buffer.append(mapped)
    available += output_count
    ages = []
    while available >= 512:
        block_capture = _capture_prefix(capture_buffer, 512)
        if not block_capture:
            raise RuntimeError("VAD block has no capture frontier")
        ages.append((ready_at - block_capture[-1].source_end_monotonic_s) * 1000)
        available -= 512
    return available, ages


def open_with_production_channels(device, wrapped):
    """Mirror Self's preferred-channel open and conditional mono retry on this device."""
    decision = determine_self_mic_capture_channels(device_idx=device, internal_channels=1)
    attempts = []
    original = sd.InputStream
    sd.InputStream = wrapped
    try:
        preferred = decision.preferred_capture_channels
        attempts.append({"channels": preferred})
        try:
            source = SoundDeviceAudioSource(
                device=device, channels=preferred, blocksize=0, sample_rate_hz=None
            )
        except Exception as exc:
            attempts[-1]["error"] = repr(exc)
            if preferred <= decision.internal_channels:
                raise
            fallback = decision.internal_channels
            attempts.append({"channels": fallback})
            source = SoundDeviceAudioSource(
                device=device, channels=fallback, blocksize=0, sample_rate_hz=None
            )
        return source, decision, attempts
    finally:
        sd.InputStream = original


async def measure(device, seconds, latency):
    original = sd.InputStream
    observations = []
    opened = {}

    def wrapped(**kwargs):
        callback = kwargs["callback"]
        requested = dict(kwargs)
        requested["callback"] = "SoundDeviceAudioSource callback"
        kwargs["callback"] = lambda data, frames, pa_time, status: (
            observations.append(
                {
                    "host": time.monotonic(),
                    "adc": float(pa_time.inputBufferAdcTime),
                    "current": float(pa_time.currentTime),
                    "frames": frames,
                    "status": str(status),
                }
            ),
            callback(data, frames, pa_time, status),
        )[-1]
        if latency is not None:
            kwargs["latency"] = latency
            requested["latency"] = latency
        stream = original(**kwargs)
        opened.update(
            requested=requested,
            rate=float(stream.samplerate),
            latency_s=float(stream.latency),
            blocksize=int(stream.blocksize),
        )
        return stream

    source, decision, channel_attempts = open_with_production_channels(device, wrapped)
    resampler = None
    source_format = None
    input_ages, vad_ages, queue_ages, frame_ages = [], [], [], []
    available = 0
    capture_buffer = []
    statuses = Counter()
    start = time.monotonic()
    try:
        async for frame in source.frames():
            now = time.monotonic()
            span = frame.capture
            assert span is not None
            seq = span.callback_sequence
            if seq < len(observations):
                obs = observations[seq]
                queue_ages.append((now - span.source_end_monotonic_s) * 1000)
                adc, current, frames = obs["adc"], obs["current"], obs["frames"]
                if (
                    all(map(math.isfinite, (adc, current)))
                    and adc > 0
                    and current > 0
                    and 0 <= current - adc <= 1
                ):
                    input_ages.append(
                        (current - adc - frames / source.actual_sample_rate_hz) * 1000
                    )
                if obs["status"]:
                    statuses[obs["status"]] += 1
            frame_format = (frame.sample_rate_hz, frame.channels)
            if source_format is None:
                source_format = frame_format
                resampler = CaptureMappedStreamingResampler(
                    input_sample_rate_hz=frame.sample_rate_hz,
                    output_sample_rate_hz=16000,
                    input_channels=frame.channels,
                )
            elif frame_format != source_format:
                raise ValueError(f"Capture format changed: {source_format} -> {frame_format}")
            assert resampler is not None
            frame_ages.append((now - span.source_end_monotonic_s) * 1000)
            output, mapped, discarded = resampler.process(frame.samples, span)
            if discarded or span.discontinuity_before is not None:
                available = 0
                capture_buffer.clear()
            available, completed = completed_vad_frontiers(
                mapped, int(output.size), capture_buffer, available, time.monotonic()
            )
            vad_ages.extend(completed)
            if time.monotonic() - start >= seconds:
                break
    finally:
        await source.close()
    paired = [
        o
        for o in observations
        if math.isfinite(o["adc"])
        and math.isfinite(o["current"])
        and o["adc"] > 0
        and o["current"] > 0
        and 0 <= o["current"] - o["adc"] <= 1
    ]
    offsets = [
        (o["host"] - o["current"]) * 1000
        for o in observations
        if math.isfinite(o["current"]) and o["current"] > 0
    ]
    raw_clock_examples = [{k: o[k] for k in ("adc", "current", "host")} for o in observations[:2]]
    invalid_reasons = Counter(
        (
            "nonfinite"
            if not all(map(math.isfinite, (o["adc"], o["current"])))
            else (
                "nonpositive"
                if o["adc"] <= 0 or o["current"] <= 0
                else (
                    "current_before_adc"
                    if o["current"] < o["adc"]
                    else "clock_domain_or_unit_mismatch" if o["current"] - o["adc"] > 1 else "valid"
                )
            )
        )
        for o in observations
    )
    rates = Counter(o["frames"] for o in observations)
    callbacks = [o["host"] for o in observations]
    expected_ms = (statistics.median(rates.elements()) / opened["rate"] * 1000) if rates else 0
    return {
        "opened": opened,
        "channel_decision": {
            "internal_channels": decision.internal_channels,
            "preferred_capture_channels": decision.preferred_capture_channels,
            "metadata_status": decision.metadata.metadata_status,
            "max_input_channels": decision.metadata.max_input_channels,
        },
        "channel_attempts": channel_attempts,
        "effective_channels": {
            "requested": source.requested_channels,
            "opened": source.opened_channels,
            "frame": source.frame_channels,
        },
        "callbacks": len(observations),
        "callback_sizes": dict(rates),
        "raw_clock_examples": raw_clock_examples,
        "clock_validity_counts": dict(invalid_reasons),
        "adc_minus_current_ms": distribution(
            [(o["adc"] - o["current"]) * 1000 for o in observations]
        ),
        "callback_statuses": dict(statuses),
        "source_status_count": source.callback_status_count,
        "queue_drops": source.queue_drop_count,
        "source_progression": str(source.capture_progression_snapshot),
        "clock_valid_pairs": len(paired),
        "clock_invalid_pairs": len(observations) - len(paired),
        "host_minus_pa_current_ms": distribution(offsets),
        "offset_drift_ms": offsets[-1] - offsets[0] if offsets else None,
        "pa_end_frontier_age_ms": distribution(input_ages),
        "callback_interval_ms": distribution(
            [(b - a) * 1000 for a, b in zip(callbacks, callbacks[1:])]
        ),
        "callback_interval_minus_median_block_ms": distribution(
            [(b - a) * 1000 - expected_ms for a, b in zip(callbacks, callbacks[1:])]
        ),
        "queue_age_at_dequeue_ms": distribution(queue_ages),
        "source_to_vad_512_frontier_ready_ms": distribution(vad_ages),
        "source_to_frame_consumer_ms": distribution(frame_ages),
        "notes": "Each completed 512-sample block uses the production VAD capture-prefix split; callback-end host mapping is not acoustic sample age or a Silero invocation timestamp. Reject ADC/current pairs with reversed order or >1s apparent difference as invalid for this short diagnostic; host-minus-current pairing alone does not validate ADC. A callback may lack an admitted frame, so callback sequence pairing for computed age is valid only when all callbacks admit frames. Callback PCM stays transient.",
    }


async def main(args):
    devices = sd.query_devices()
    apis = sd.query_hostapis()
    inventory = [
        {
            "index": i,
            "name": d["name"],
            "hostapi": apis[d["hostapi"]]["name"],
            "channels": d["max_input_channels"],
            "default_rate": d["default_samplerate"],
        }
        for i, d in enumerate(devices)
        if d["max_input_channels"]
    ]
    selected = int(sd.default.device[0] if args.device is None else args.device)
    if selected < 0 or selected >= len(devices) or devices[selected]["max_input_channels"] < 1:
        raise RuntimeError("No usable input device selected")
    if args.expected_name and devices[selected]["name"] != args.expected_name:
        raise RuntimeError(f"Selected device changed: {devices[selected]['name']!r}")
    selected_hostapi = apis[devices[selected]["hostapi"]]["name"]
    if args.expected_hostapi and selected_hostapi != args.expected_hostapi:
        raise RuntimeError(f"Selected host API changed: {selected_hostapi!r}")
    import psutil

    vrchat = [
        p.info["name"]
        for p in psutil.process_iter(["name"])
        if p.info["name"] and "vrchat" in p.info["name"].lower()
    ]
    result = {
        "revision": args.revision,
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {x: version(x) for x in ("numpy", "soxr", "sounddevice", "janus", "psutil")},
        "device_inventory": inventory,
        "selected_device": selected,
        "selected_hostapi": selected_hostapi,
        "vrchat_processes_at_start": vrchat,
        "seconds_each": args.seconds,
        "runs": [],
    }
    for setting in (None, "low", None, "low"):
        try:
            measurement = await measure(selected, args.seconds, setting)
            result["runs"].append(
                {
                    "latency_request": "omitted(default high)" if setting is None else "low",
                    **measurement,
                }
            )
        except Exception as exc:
            result["runs"].append({"latency_request": str(setting), "error": repr(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "runs": [
                    {
                        "request": r["latency_request"],
                        "callbacks": r.get("callbacks"),
                        "error": r.get("error"),
                        "age": r.get("source_to_vad_512_frontier_ready_ms"),
                    }
                    for r in result["runs"]
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Reviewed source revision for this run")
    parser.add_argument("--expected-name", help="Require this exact device inventory name")
    parser.add_argument("--expected-hostapi", help="Require this host API inventory name")
    parser.add_argument("--device", type=int)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("d_results.json"))
    asyncio.run(main(parser.parse_args()))
