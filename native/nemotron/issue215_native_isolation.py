"""CPU-only issue215 acoustic isolation; NOT causal acceptance evidence.

Delivered-packet modes check actual live PCM hashes. Full-waveform mode uses
the exact uncut returning-A/B source assembly, with no extra context or padding.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from issue215_smoke import ROOT, create_local_cpu_backend, digest, input_cases, np, serial

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes


async def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    home = Path.home()
    _, assembled = input_cases(
        SimpleNamespace(
            corpus=home / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1",
            correction_root=ROOT.parent
            / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification",
        )
    )
    live = json.loads(args.trace.read_text(encoding="utf-8"))
    flush_samples = args.right_flush_ms * 16
    flush_label = f"diagnostic_zero_right_flush_{args.right_flush_ms}ms"
    report = {
        "schema": "issue215-exact-native-isolation-1",
        "scope": "CPU-only exact-packet diagnostic; not causal or SourceChild acceptance evidence",
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "model": "parakeet-tdt-0.6b-v3-int8-sherpa",
        "source_language": "en",
        "extra_context_samples": 0,
        "order": args.order,
        "cases": [],
    }
    selected_cases = list(live["cases"])
    if args.guards:
        guards = json.loads(args.guards.read_text(encoding="utf-8"))
        if guards["asr_hashes"] != live["asr_hashes"]:
            raise ValueError("guard ASR artifacts differ")
        selected_cases += [
            case
            for case in guards["cases"]
            if case["case"] in ("six_second_rollover", "linguistic_correction")
        ]
        report["guard_trace"] = str(args.guards)
        report["guard_trace_sha256"] = digest(args.guards)
    if args.order == "right_flush":
        report["synthetic_native_right_context"] = {
            "samples": flush_samples,
            "milliseconds": args.right_flush_ms,
            "kind": "diagnostic-only zero suffix, not source ownership/context replay",
            "rationale": (
                "Upstream Parakeet v3 test_onnx.py aliases v2, whose native ONNX reference adds2s zeros before per-feature normalization and TDT decode; this compares that convention through the installed recognizer, not a parameter sweep."
                if args.right_flush_ms == 2000
                else "One empirical fixed quiet right acoustic window to test native decoder endpoint flush;160ms is not an official requirement."
            ),
            "source_timing_changed": False,
            "fed_to_nemotron": False,
        }
    report["asr_hashes"] = {
        relative: digest(home / "AppData/Local/puripuly-heart/models" / relative)
        for relative in live["asr_hashes"]
    }
    if report["asr_hashes"] != live["asr_hashes"]:
        raise ValueError("ASR artifact changed since the retained live trace")
    for case in selected_cases:
        values, _ = assembled[case["case"]]
        if hashlib.sha256(values.tobytes()).hexdigest() != case["input_pcm_f32_sha256"]:
            raise ValueError("live source PCM differs from the approved assembly")
        capture = float32_to_pcm16le_bytes(values)
        backend = create_local_cpu_backend(
            report["model"],
            model_root=home / "AppData/Local/puripuly-heart/models",
            source_language="en",
            sample_rate_hz=16000,
            stream_label="peer",
        )
        await backend._ensure_recognizer()
        result = {"case": case["case"], "packets": []}
        result["resolved_native_options"] = serial(backend)
        try:
            packets = []
            if args.order == "full_waveform":
                mapping = case["assembly"]
                if len(mapping) != 3 or mapping[0]["input"] != "A" or mapping[-1]["input"] != "A":
                    raise ValueError("full-waveform diagnosis requires the actual A-B-A assembly")
                result["full_waveform_source_maps"] = mapping[1:]
                b_start, b_end = mapping[1]["scenario_sample_range"]
                a_start, a_end = mapping[2]["scenario_sample_range"]
                full_a = capture[a_start * 2 : a_end * 2]
                full_b = capture[b_start * 2 : b_end * 2]
                support = np.flatnonzero(np.abs(values[a_start:a_end]) > 0.01)
                result["returning_A_support"] = {
                    "threshold_abs_amplitude": 0.01,
                    "scenario_sample_range": [
                        a_start + int(support[0]),
                        a_start + int(support[-1]) + 1,
                    ],
                    "kind": "measured waveform support, NOT word timestamps or alignment",
                }
                packets = [
                    ("full_returning_A_fresh_first", full_a, None),
                    ("full_B_separate_stream", full_b, None),
                    ("full_returning_A_after_B_separate_stream", full_a, None),
                    ("contiguous_full_B_plus_returning_A", full_b + full_a, None),
                ]
            else:
                submits = [event for event in case["trace"] if event["event"] == "native_submit"]
                if args.order == "right_flush":
                    selected = [event for event in submits if event["seal_reason"] != "speaker"]
                    if case["case"] == "short_A_B_A":
                        selected.insert(0, submits[1])  # Actual short-No packet guard.
                    if case["case"] == "six_second_rollover":
                        selected = [
                            event for event in selected if event["seal_reason"] == "delivery_pause"
                        ]
                    submits = selected
                if args.order == "tail_first":
                    tails = [event for event in submits if event["seal_reason"] != "speaker"]
                    prefixes = [event for event in submits if event["seal_reason"] == "speaker"]
                    submits = tails + prefixes + tails
                for event in submits:
                    packet = b"".join(
                        capture[
                            span["normalized_start_sample"] * 2 : span["normalized_end_sample"] * 2
                        ]
                        for payload in event["payloads"]
                        for span in payload["source_ranges"]
                    )
                    if hashlib.sha256(packet).hexdigest() != event["pcm16_sha256"]:
                        raise ValueError("reconstructed packet differs from actual delivered PCM")
                    packets.append(
                        (
                            "actual_delivered_native_packet",
                            packet,
                            event["identity"]["provider_turn_id"],
                        )
                    )
                    if args.order == "right_flush":
                        packets.append(
                            (
                                flush_label,
                                packet + bytes(flush_samples * 2),
                                event["identity"]["provider_turn_id"],
                            )
                        )
            for label, packet, request_id in packets:
                started = time.monotonic()
                text = await backend.decode_pcm16le(packet)
                result["packets"].append(
                    {
                        "diagnostic_label": label,
                        "base_native_request_id": request_id,
                        "samples": len(packet) // 2,
                        "pcm16_sha256": hashlib.sha256(packet).hexdigest(),
                        "synthetic_right_context_samples": (
                            flush_samples if label == flush_label else 0
                        ),
                        "base_pcm16_sha256": hashlib.sha256(
                            packet[: -flush_samples * 2] if label == flush_label else packet
                        ).hexdigest(),
                        "text": text,
                        "wall_seconds": time.monotonic() - started,
                    }
                )
        finally:
            await backend.close()
        report["cases"].append(result)
    text = json.dumps(report, indent=2)
    text = text.replace(str(home).replace("\\", "\\\\"), "<USER_HOME>").replace(
        str(home), "<USER_HOME>"
    )
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--guards", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--order",
        choices=("capture", "tail_first", "full_waveform", "right_flush"),
        default="capture",
    )
    parser.add_argument("--right-flush-ms", type=int, choices=(160, 2000), default=160)
    asyncio.run(run(parser.parse_args()))
