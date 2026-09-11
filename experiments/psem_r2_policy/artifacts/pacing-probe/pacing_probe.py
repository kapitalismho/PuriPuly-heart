"""Zero-cost pacing probe for the native poll versus paced feed coupling.

Runs the real native Sortformer producer against the real paced feed path twice:
once with the repaired non-blocking poll and once with the pre-repair blocking
recv timeout emulated at the socket, then records source position versus
monotonic for both. No providers, no network, no paid calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
TARGET = PACKAGE.parents[1]
HZ = 16000
PACED_PRODUCER_EXE = Path("C:/tmp/psem-e2o2-paced/bin/transcribe-cli.exe")
SORTFORMER_MODEL = Path(
    "C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf"
)
AMI_MEETING_WAV = Path(
    "C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/audio/ES2009a/"
    "ES2009a.Mix-Headset.wav"
)


def default_capsule() -> Path:
    manifests = sorted(
        (PACKAGE / ".capsule").glob("*/capsule_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not manifests:
        raise SystemExit("no built capsule found; run launch.py --prepare first")
    return manifests[-1].parent


def window_rates(samples: list[list[float]], *, min_span_s: float) -> list[float]:
    rates: list[float] = []
    index = 0
    while index < len(samples) - 1:
        start = samples[index]
        end_index = index + 1
        while end_index < len(samples) and samples[end_index][1] - start[1] < min_span_s:
            end_index += 1
        if end_index >= len(samples):
            break
        end = samples[end_index]
        wall = float(end[0]) - float(start[0])
        if wall > 0.0:
            rates.append((float(end[1]) - float(start[1])) / wall)
        index = end_index
    return rates


async def run_case(wav: Path, out_dir: Path, *, poll_timeout: float | None) -> dict:
    from experiments.psem_r2_policy.live_runner import (
        hello_there_script,
        install_deepgram_intercept,
        run_continuous_wav,
    )
    from experiments.psem_r2_policy.sortformer_live import NativeSortformerProducer

    scripts = (hello_there_script(),)
    original_attach = NativeSortformerProducer.attach
    if poll_timeout is not None:

        def blocking_attach(self, conn: socket.socket) -> None:
            self._conn = conn
            conn.settimeout(poll_timeout)

        NativeSortformerProducer.attach = blocking_attach
    try:
        with install_deepgram_intercept(scripts):
            return await run_continuous_wav(
                wav,
                network=False,
                secrets={},
                intercept=scripts,
                meeting="ES2009a-pacing-probe",
                sortformer=True,
                pace=True,
                artifact_dir=out_dir,
            )
    finally:
        NativeSortformerProducer.attach = original_attach


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--start-s", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=90)
    args = parser.parse_args()
    capsule = (args.capsule or default_capsule()).resolve()
    package_dir = capsule / "experiments" / "psem_r2_policy"
    if not package_dir.is_dir():
        raise SystemExit(f"capsule package dir not found: {package_dir}")
    sys.path.insert(0, str(capsule))
    import experiments.psem_r2_policy.run  # noqa: F401  (installs the ROOT/src bootstrap)
    from experiments.psem_r2_policy.live_runner import load_wav_16k, write_pcm_wav

    for module_name in (
        "experiments.psem_r2_policy.run",
        "experiments.psem_r2_policy.live_runner",
        "experiments.psem_r2_policy.sortformer_live",
    ):
        origin = Path(sys.modules[module_name].__file__).resolve()
        if not origin.is_relative_to(capsule):
            raise SystemExit(f"{module_name} resolved outside the capsule: {origin}")

    for asset in (PACED_PRODUCER_EXE, SORTFORMER_MODEL, AMI_MEETING_WAV):
        if not asset.is_file():
            raise SystemExit(f"missing asset: {asset}")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_wav_16k(AMI_MEETING_WAV)[
        args.start_s * HZ : (args.start_s + args.seconds) * HZ
    ]
    wav = write_pcm_wav(out_dir / f"es2009a_{args.seconds}s_paced.wav", samples)
    results: dict[str, dict] = {}
    for label, poll_timeout in (("fixed", None), ("old_blocking", 0.05)):
        run_started = time.monotonic()
        case = asyncio.run(run_case(wav, out_dir / label, poll_timeout=poll_timeout))
        run_wall_s = time.monotonic() - run_started
        capture = case["capture_timing"]
        trace = capture["feed_progress"]["samples"]
        rates = window_rates(trace, min_span_s=3.0)
        source_eof_s = float(capture["last_arrival_monotonic_s"])
        arrivals = [
            float(row["available_at_monotonic_s"])
            for row in case["native_chunks"]
            if row.get("available_at_monotonic_s") is not None
        ]
        results[label] = {
            "poll_timeout_s": poll_timeout,
            "audio_seconds": args.seconds,
            "run_wall_s": run_wall_s,
            "n_native_chunks": len(case["native_chunks"]),
            "native_arrivals_after_source_eof": sum(1 for item in arrivals if item > source_eof_s),
            "native_chunk_arrival_distinct_stamps": capture[
                "native_chunk_arrival_distinct_stamps"
            ],
            "feed_progress_interval_s": capture["feed_progress"]["interval_s"],
            "feed_progress": trace,
            "window_rates": rates,
            "min_window_rate": min(rates, default=None),
            "max_window_rate": max(rates, default=None),
            "source_seconds": trace[-1][1] - trace[0][1] if trace else None,
            "wall_seconds": trace[-1][0] - trace[0][0] if trace else None,
        }
    payload = {
        "meeting": "ES2009a",
        "source_wav": str(AMI_MEETING_WAV),
        "slice_seconds": [args.start_s, args.start_s + args.seconds],
        "capsule": str(capsule),
        "results": results,
    }
    (out_dir / "pacing-trace.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    summary = {
        label: {
            "min_window_rate": row["min_window_rate"],
            "max_window_rate": row["max_window_rate"],
            "source_seconds": row["source_seconds"],
            "wall_seconds": row["wall_seconds"],
            "run_wall_s": round(row["run_wall_s"], 3),
            "n_native_chunks": row["n_native_chunks"],
            "native_arrivals_after_source_eof": row["native_arrivals_after_source_eof"],
        }
        for label, row in results.items()
    }
    print(json.dumps(summary, ensure_ascii=False))
    ok = (
        results["fixed"]["min_window_rate"] is not None
        and results["fixed"]["min_window_rate"] >= 0.85
        and results["fixed"]["max_window_rate"] <= 1.4
        and results["old_blocking"]["min_window_rate"] < 0.85
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
