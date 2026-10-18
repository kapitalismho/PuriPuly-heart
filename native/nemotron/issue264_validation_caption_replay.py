from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import psutil
from issue264_validation_runtime import ROOT, ObservedDesktop, local, save, source_manifest

SOURCE = ROOT / "build/issue264-validation-repaired-gpu-rest-01/trace.json"


async def run(args):
    if os.environ.get("ISSUE264_VALIDATION_READY_SHA") != args.candidate:
        raise RuntimeError("Use the authorized isolated launcher")
    original = SOURCE.read_bytes()
    source = json.loads(original)
    if source["candidate"] != args.candidate:
        raise ValueError("Recorded snapshot product must match the frozen candidate")
    selected = [
        case
        for case in source["cases"]
        if case["case"] == "linguistic_correction" and case["profile"] == "0.64"
    ]
    assert len(selected) == 1
    case = selected[0]
    requests = case["renderer_snapshots"]
    assert len(requests) == 2
    assert len(requests[0]["snapshot"]["blocks"]) == 1
    assert requests[1]["snapshot"]["blocks"] == []
    original_invalid = case["renderer_readbacks"]
    assert len(original_invalid) == 1 and not original_invalid[0]["readback_valid"]
    desktop = ObservedDesktop(args.output)
    report = {
        "schema": "issue264-correction-presentation-only-replay-1",
        "candidate": args.candidate,
        "source_trace": SOURCE.relative_to(ROOT).as_posix(),
        "source_trace_sha256": hashlib.sha256(original).hexdigest(),
        "source_case": "linguistic_correction",
        "source_profile": "0.64",
        "source_original_readback_revision": original_invalid[0]["revision"],
        "source_original_invalid_reason": original_invalid[0]["invalid_readback_reason"],
        "scope": "One Director-authorized no-model actual Flet replay of the already-recorded authoritative correction snapshots. Snapshot transport revisions are rebased; cards, text, palette and layout metadata are retained. Replayed snapshots are scheduled against the recorded interval; requested/observed times are separate. No ASR, speaker, translation, source admission, history or TTL inference is rerun. Does not recover original coupled latency or qualify the original invalid readback.",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(
            (ROOT / "native/nemotron/issue264_validation_launch.py").read_bytes()
        ).hexdigest(),
        "sources_sha256": source_manifest(),
        "model_calls": 0,
        "translation_service_calls": 0,
        "cloud_calls": 0,
    }
    try:
        await desktop.start()
        report["owned_renderer_pid"] = desktop.pid
        report["owned_backdrop_pid"] = desktop.backdrop_pid
        started = time.monotonic()
        for index, request in enumerate(requests):
            target = started + request["submitted_at"] - requests[0]["submitted_at"]
            await asyncio.sleep(max(0, target - time.monotonic()))
            snapshot = local.OverlayPresentationSnapshot.from_dict(request["snapshot"])
            assert snapshot.to_dict() == request["snapshot"]
            await desktop.replace_snapshot(snapshot)
            assert desktop.snapshots[-1]["snapshot"]["blocks"] == request["snapshot"]["blocks"]
            if index == 0:
                await asyncio.gather(*desktop.readback_tasks)
                report["presentation_capture"] = await desktop.picture("recorded_correction")
                assert report["presentation_capture"].get("visibility_valid") is True
                assert desktop.readbacks[-1]["readback_valid"]
                assert desktop.readbacks[-1]["capture"]["exact_color_pixel_counts"]["gold"] > 0
        children = psutil.Process().children(recursive=True)
        report["observed_owned_children"] = [
            {"pid": child.pid, "name": child.name()} for child in children
        ]
        assert not any(
            "llama" in child.name().lower() or "gpuworker" in child.name().lower()
            for child in children
        )
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == report["source_trace_sha256"]
        report["status"] = "presentation_replay_completed_requires_image_inspection"
    except Exception as error:
        report["status"] = "failed_observation"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        await desktop.close()
        report["snapshots"] = desktop.snapshots
        report["readbacks"] = desktop.readbacks
        report["renderer_errors"] = desktop.errors
        report["requested_inter_snapshot_s"] = (
            requests[1]["submitted_at"] - requests[0]["submitted_at"]
        )
        report["observed_inter_snapshot_s"] = (
            desktop.snapshots[1]["submitted_at"] - desktop.snapshots[0]["submitted_at"]
            if len(desktop.snapshots) == 2
            else None
        )
        save(args.output / "presentation_replay.json", report)
    return 0 if report["status"] != "failed_observation" else 1


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()
