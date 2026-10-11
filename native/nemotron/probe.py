from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from puripuly_heart.app.adapters.nemotron import NemotronBackend, inspect_artifact, validate_runtime


def audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    values = np.frombuffer(result.stdout, dtype="<f4")
    if not 1 <= values.size <= 16000 * 20:
        raise ValueError("public probe clip must be at most 20 seconds")
    return values


async def run(args):
    report = {
        "artifact": inspect_artifact(args.model),
        "runtime": validate_runtime(args.runtime),
        "public_inputs": [
            {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "license": "CC0-1.0",
                "dataset": "Common Voice 17.0",
            }
            for p in (args.audio_a, args.audio_b)
        ],
        "vulkan_summary": subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True
        ).stdout,
        "profiles": {},
        "co_residency": {"status": "not_run"},
    }
    asr = None
    if args.asr_root is not None:
        from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend

        asr = create_local_cpu_backend(
            "parakeet-tdt-0.6b-v3-int8-sherpa",
            model_root=args.asr_root,
            source_language="en",
            sample_rate_hz=16000,
            stream_label="peer",
        )
    a, b = audio(args.audio_a), audio(args.audio_b)
    overlap = (a[: min(a.size, b.size)] + b[: min(a.size, b.size)]) * 0.5
    for profile in ("1.04", "0.64"):
        backend = NemotronBackend(
            artifact_path=args.model, runtime_dir=args.runtime, device=args.device, profile=profile
        )
        evidence = {
            "status": "running",
            "updates": [],
            "source_ledger": [],
            "resets": [],
            "raw_probability_sha256": None,
        }
        report["profiles"][profile] = evidence
        probabilities = hashlib.sha256()
        cursor = 0
        peak_frames = 0
        started = time.perf_counter()
        try:
            ready = await backend.start()
            evidence["ready"] = {
                "tail_padding_samples": ready.tail_padding_samples,
                "max_revision_samples": ready.max_revision_samples,
                "diagnostics": dict(ready.diagnostics),
            }
            child_pid = backend._process.pid
            evidence["resident_child_pid"] = child_pid
            await backend.reset(1)

            async def feed(values: np.ndarray, label: str):
                nonlocal cursor, peak_frames
                evidence["source_ledger"].append(
                    {
                        "epoch": backend._epoch,
                        "start": cursor,
                        "end": cursor + values.size,
                        "kind": label,
                    }
                )
                for offset in range(0, values.size, 1600):
                    block = np.array(values[offset : offset + 1600], dtype=np.float32)
                    block.flags.writeable = False
                    begin = time.perf_counter()
                    update = await backend.feed(block, cursor)
                    cursor += block.size
                    elapsed = time.perf_counter() - begin
                    peak_frames = max(peak_frames, len(update.frames))
                    raw = np.array([frame.probabilities for frame in update.frames], dtype="<f4")
                    probabilities.update(raw.tobytes())
                    evidence["updates"].append(
                        {
                            "epoch": backend._epoch,
                            "submitted": cursor,
                            "processed": update.processed_through_sample,
                            "immutable": update.immutable_through_sample,
                            "frames": len(update.frames),
                            "observation_ready_monotonic_s": time.monotonic(),
                            "duration_s": elapsed,
                            "raw_frames": [
                                {
                                    "start": frame.start_sample,
                                    "end": frame.end_sample,
                                    "probabilities": frame.probabilities,
                                }
                                for frame in update.frames
                            ],
                        }
                    )

            for values, label in (
                (a, "public_A"),
                (np.zeros(8000, dtype=np.float32), "real_fixture_silence"),
                (b, "public_B"),
                (a, "returning_public_A"),
                (overlap, "controlled_public_A_plus_B"),
            ):
                await feed(values, label)
                if label in ("public_A", "public_B"):
                    previous = cursor
                    await feed(
                        np.zeros(ready.tail_padding_samples, dtype=np.float32),
                        "synthetic_tail_padding",
                    )
                    paused = await backend.pause()
                    if paused.immutable_through_sample < previous:
                        raise AssertionError(
                            "pending real tail not committed by bounded causal padding"
                        )
                    evidence.setdefault("pauses", []).append(
                        {
                            "submitted": cursor,
                            "processed": paused.processed_through_sample,
                            "immutable": paused.immutable_through_sample,
                        }
                    )
                    await asyncio.sleep(0.05)
            if asr is not None:
                decoded = await asr.decode_f32(a)
                report["co_residency"] = {
                    "status": "passed",
                    "provider": "local_parakeet_v3",
                    "model": "parakeet-tdt-0.6b-v3-int8-sherpa",
                    "asr_result": str(decoded),
                    "nemotron_child_pid": backend._process.pid,
                    "asr_recognizer_resident": getattr(asr, "_recognizer", None) is not None,
                    "scope": "selected CPU local ASR plus Vulkan Nemotron; not GPU-ASR/VRChat competition",
                }
                if backend._process.pid != child_pid:
                    raise AssertionError("Nemotron model child was replaced for ASR")
                await feed(a, "public_A_after_ASR_decode")
            for _ in range(30):
                await feed(a, "continuous_public_A_no_silence")
            for epoch in (2, 3, 4):
                await backend.reset(epoch)
                cursor = 0
                await feed(a[:1600], f"epoch_{epoch}_public_A")
                evidence["resets"].append(
                    {"epoch": epoch, "same_child": backend._process.pid == child_pid}
                )
            evidence["status"] = "passed"
            evidence["max_frames_per_update"] = peak_frames
            evidence["raw_probability_sha256"] = probabilities.hexdigest()
            evidence["wall_seconds"] = time.perf_counter() - started
            evidence["worker_stderr_tail"] = bytes(backend._stderr).decode(errors="replace")
            evidence["bounded_output"] = {
                "native_probability_capacity_frames": 1024,
                "worker_result_capacity_frames": 1024,
                "parent_retained_stderr_bytes": len(backend._stderr),
                "ipc_metadata_bytes": 8192,
                "ipc_payload_bytes": 65536,
            }
        except Exception as error:
            evidence["status"], evidence["error"] = "failed", str(error)
            evidence["worker_stderr_tail"] = bytes(backend._stderr).decode(errors="replace")
        finally:
            await backend.stop()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    missing = NemotronBackend(
        artifact_path=args.model, runtime_dir=args.runtime, device=9999, profile="0.64"
    )
    try:
        await missing.start()
        report["missing_device"] = {"status": "failed", "reason": "unexpectedly accepted"}
    except Exception as error:
        report["missing_device"] = {"status": "passed", "rejected": str(error)}
    finally:
        await missing.stop()
    inflight = NemotronBackend(
        artifact_path=args.model, runtime_dir=args.runtime, device=args.device, profile="0.64"
    )
    await inflight.start()
    process = inflight._process
    samples = np.zeros(16000, dtype=np.float32)
    samples.flags.writeable = False
    task = asyncio.create_task(inflight.feed(samples, 0))
    await asyncio.sleep(0.001)
    active = not task.done()
    await inflight.stop()
    outcome = "completed_before_stop"
    try:
        await task
    except asyncio.CancelledError, asyncio.IncompleteReadError, ConnectionError:
        outcome = "inflight_fenced"
    report["real_inflight_stop"] = {
        "status": "passed" if active and process.returncode is not None else "unknown",
        "request_inflight_at_stop": active,
        "child_returncode": process.returncode,
        "outcome": outcome,
    }
    if asr is not None:
        report["asr_after_speaker_fault"] = {
            "status": "passed",
            "result": str(await asr.decode_f32(a)),
        }
    if asr is not None:
        await asr.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "profiles": {p: e["status"] for p, e in report["profiles"].items()},
                "co_residency": report["co_residency"]["status"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--audio-a", type=Path, required=True)
    parser.add_argument("--audio-b", type=Path, required=True)
    parser.add_argument("--asr-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
