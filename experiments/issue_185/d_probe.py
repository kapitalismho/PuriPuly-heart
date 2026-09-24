"""Opt-in metadata-only microphone comparison; no PCM is written to disk."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import sounddevice as sd

from puripuly_heart.core.audio.source import SoundDeviceAudioSource
from puripuly_heart.core.audio.streaming_resampler import CaptureMappedStreamingResampler


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

    sd.InputStream = wrapped
    try:
        source = SoundDeviceAudioSource(device=device, channels=1, blocksize=0, sample_rate_hz=None)
    finally:
        sd.InputStream = original
    resampler = CaptureMappedStreamingResampler(source.actual_sample_rate_hz)
    input_ages, vad_ages, queue_ages, frame_ages = [], [], [], []
    available = 0
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
            frame_ages.append((now - span.source_end_monotonic_s) * 1000)
            output, mapped, discarded = resampler.process(frame.samples, span)
            if discarded:
                available = 0
            available += output.size
            if available >= 512:
                # The actual VAD loop drains complete 512-sample blocks after normalization.
                # This is an end-frontier age upper-bound proxy, not a VAD invocation timestamp.
                if mapped is not None:
                    vad_ages.append((time.monotonic() - mapped.source_end_monotonic_s) * 1000)
                available %= 512
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
        "source_to_vad_ready_proxy_ms": distribution(vad_ages),
        "source_to_frame_consumer_ms": distribution(frame_ages),
        "notes": "Reject ADC/current pairs with reversed order or >1s apparent difference as invalid for this short diagnostic; host-minus-current pairing alone does not validate ADC. One callback may lack an admitted frame, so callback sequence pairing for computed age is valid only when all callbacks admit frames. VAD-ready proxy follows mapped end span, not acoustic-time truth. Callback PCM stays transient.",
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
    if selected < 0 or devices[selected]["max_input_channels"] < 1:
        raise RuntimeError("No usable input device selected")
    import psutil

    vrchat = [
        p.info["name"]
        for p in psutil.process_iter(["name"])
        if p.info["name"] and "vrchat" in p.info["name"].lower()
    ]
    result = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {x: version(x) for x in ("numpy", "soxr", "sounddevice", "janus", "psutil")},
        "device_inventory": inventory,
        "selected_device": selected,
        "selected_hostapi": apis[devices[selected]["hostapi"]]["name"],
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
                        "age": r.get("pa_end_frontier_age_ms"),
                    }
                    for r in result["runs"]
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("d_results.json"))
    asyncio.run(main(parser.parse_args()))
