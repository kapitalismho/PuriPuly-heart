from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from probe import audio

from puripuly_heart.app.adapters.nemotron import NemotronBackend
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.speaker import RuntimeLimits, SpeakerRuntime


def event(
    values: np.ndarray, start: int, sequence: int, capture_epoch: int = 7, *, speech: bool = True
):
    span = AudioCaptureSpan(
        capture_epoch,
        sequence,
        16000,
        start,
        start + values.size,
        1000.0 + start / 16000,
        1000.0 + (start + values.size) / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=start + values.size,
    )
    return CaptureStreamInput(values, (span,), speech_observed=speech)


class CompletionGate:
    def __init__(self, backend):
        self.backend = backend
        self.release = None
        self.completed = asyncio.Event()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    async def feed(self, pcm, compact_start_sample):
        update = await self.backend.feed(pcm, compact_start_sample)
        if self.release is not None:
            self.native_update_frames = len(update.frames)
            self.completed.set()
            await self.release.wait()
        return update


async def run(args):
    a, b = audio(args.audio_a), audio(args.audio_b)
    report = {
        "scope": "actual SpeakerRuntime owner through isolated patched Vulkan child on licensed fixture PCM; no ASR replay/context path invoked",
        "source_clock": "constructed normalized fixture source coordinates; observation-ready times are actual monotonic clock",
        "profiles": {},
    }
    for profile in ("1.04", "0.64"):
        backend = NemotronBackend(
            artifact_path=args.model, runtime_dir=args.runtime, device=args.device, profile=profile
        )
        limits = RuntimeLimits(max_evidence_items=128)
        result = {
            "status": "running",
            "snapshots": [],
            "evidence": [],
            "source_ledger": [],
            "checks": {},
        }
        report["profiles"][profile] = result
        source = sequence = 0
        session = "public-fixture-session"
        cursor_peak = {
            "pending_samples": 0,
            "pending_items": 0,
            "retained_map_pieces": 0,
            "evidence_items": 0,
            "owned_tasks": 0,
        }

        delivery = CompletionGate(backend)
        owner = SpeakerRuntime(delivery, limits=limits)

        def snapshot(label):
            state = owner.snapshot()
            result["snapshots"].append({"label": label, **asdict(state)})
            for key in cursor_peak:
                cursor_peak[key] = max(cursor_peak[key], getattr(state, key))
            return state

        def drain():
            values = owner.drain_evidence(128)
            result["evidence"].extend(asdict(value) for value in values)
            return values

        async def idle():
            deadline = time.monotonic() + 30
            while owner.snapshot().pending_items:
                if time.monotonic() > deadline:
                    raise TimeoutError("owner input drain timeout")
                if not owner.snapshot().ready:
                    raise RuntimeError(owner.snapshot().error)
                await asyncio.sleep(0.002)
            await asyncio.sleep(0)
            if not owner.snapshot().ready:
                raise RuntimeError(owner.snapshot().error)

        async def feed(values, label, *, speech=True, consume=True, capture_epoch=7):
            nonlocal source, sequence
            result["source_ledger"].append(
                {
                    "session": session,
                    "epoch": owner.snapshot().speaker_epoch,
                    "start": source,
                    "end": source + values.size,
                    "label": label,
                    "vad_speech": speech,
                }
            )
            records = []
            for offset in range(0, values.size, 1600):
                block = np.array(values[offset : offset + 1600], dtype=np.float32)
                block.flags.writeable = False
                owner.observe_input(
                    session, event(block, source, sequence, capture_epoch, speech=speech)
                )
                source += block.size
                sequence += 1
                snapshot("input_admitted")
                await idle()
                if consume:
                    records.extend(drain())
            return records

        try:
            await owner.start()
            initial = snapshot("ready")
            if not initial.ready:
                raise RuntimeError(initial.error)
            child_pid = backend._process.pid
            result["resident_pid"] = child_pid
            await feed(np.zeros(8000, np.float32), "onset_context", speech=False)
            records_a = await feed(a, "A_before_pause")
            first_epoch = owner.snapshot().speaker_epoch
            await feed(np.zeros(48000, np.float32), "causal_long_silence", speech=False)
            paused = snapshot("ordinary_pause")
            await asyncio.sleep(0.05)
            retained = snapshot("pause_without_new_input")
            result["checks"]["causal_tail_pause"] = {
                "status": (
                    "passed"
                    if paused.paused
                    and paused.speaker_epoch == first_epoch
                    and retained.processed_through_sample == paused.processed_through_sample
                    else "failed"
                )
            }
            records_b = await feed(b, "B_after_pause")
            await feed(np.zeros(48000, np.float32), "causal_long_silence", speech=False)
            records_return = await feed(a, "returning_A")
            continuity = snapshot("returning_A_same_epoch")

            def dominant(records):
                rows = [
                    value.frame.probabilities
                    for value in records
                    if value.frame is not None
                    and any(piece.capture is not None for piece in value.source_map)
                ]
                if not rows:
                    return None
                mean = np.mean(rows, axis=0)
                return {
                    "slot": int(np.argmax(mean)),
                    "mean_raw_probabilities": mean.tolist(),
                    "frame_count": len(rows),
                }

            first, second, returned = (
                dominant(records_a),
                dominant(records_b),
                dominant(records_return),
            )
            result["checks"]["return_key"] = {
                "status": (
                    "passed"
                    if first
                    and returned
                    and first["slot"] == returned["slot"]
                    and continuity.speaker_epoch == first_epoch
                    else "unknown"
                ),
                "first_A": first,
                "B": second,
                "return_A": returned,
                "meaning": "same-epoch raw slot with highest fixture mean, not identity/confidence policy or accuracy guarantee",
            }
            count = min(a.size, b.size)
            overlap = await feed((a[:count] + b[:count]) * 0.5, "controlled_A_plus_B")
            result["checks"]["overlap_representation"] = {
                "status": (
                    "passed"
                    if overlap
                    and all(
                        len(value.frame.probabilities) == 8
                        for value in overlap
                        if value.frame is not None
                    )
                    else "failed"
                ),
                "meaning": "eight independent raw channels preserved; no forced single-active-slot or new threshold; acoustic overlap accuracy not scored",
            }
            before = snapshot("slow_consumer_before")
            for _ in range(5):
                await feed(a, "continuous_speech_slow_consumer", consume=False)
            slow = snapshot("slow_consumer_after")
            result["checks"]["slow_consumer"] = {
                "status": (
                    "passed"
                    if slow.evidence_items <= limits.max_evidence_items
                    and slow.evidence_evictions > before.evidence_evictions
                    and slow.speaker_epoch == before.speaker_epoch
                    else "failed"
                ),
                "evictions": slow.evidence_evictions - before.evidence_evictions,
                "retained_items": slow.evidence_items,
                "duration_input_samples": a.size * 5,
            }
            drain()
            old_epoch = owner.snapshot().speaker_epoch
            owner.reset("manual_probe_reset")
            await feed(a[:1600], "after_manual_reset")
            manual = snapshot("manual_reset")
            result["checks"]["manual_reset"] = {
                "status": (
                    "passed"
                    if manual.speaker_epoch > old_epoch and backend._process.pid == child_pid
                    else "failed"
                )
            }
            old_epoch = manual.speaker_epoch
            owner.observe_input(
                session,
                CaptureStreamInput(
                    np.empty(0, np.float32), (), boundary_reason="source_discontinuity"
                ),
            )
            source = 0
            await feed(a[:1600], "after_source_discontinuity", capture_epoch=8)
            discontinuity = snapshot("real_source_discontinuity")
            result["checks"]["source_discontinuity"] = {
                "status": (
                    "passed"
                    if discontinuity.speaker_epoch > old_epoch and backend._process.pid == child_pid
                    else "failed"
                )
            }
            block = np.array(a[:1600], np.float32)
            block.flags.writeable = False
            owner.observe_input(session, event(block, source, sequence, 8))
            source += block.size
            eof_epoch = owner.snapshot().speaker_epoch
            owner.observe_input(
                session,
                CaptureStreamInput(np.empty(0, np.float32), (), boundary_reason="source_eof"),
            )
            owner.end_capture(session, "source_eof")
            await idle()
            deadline = time.monotonic() + 30
            while owner.snapshot().capture_session_id is not None and time.monotonic() < deadline:
                await asyncio.sleep(0.002)
            eof_records = drain()
            eof = snapshot("graceful_eof_after_immediate_end")
            result["checks"]["graceful_eof"] = {
                "status": (
                    "passed"
                    if eof.speaker_epoch == eof_epoch
                    and eof.capture_session_id is None
                    and any(value.reason == "source_eof" for value in eof_records)
                    else "failed"
                ),
                "same_epoch": eof.speaker_epoch == eof_epoch,
                "reasons": [value.reason for value in eof_records if value.reason],
                "coverage_pending": any(
                    value.reason == "source_eof_pending_coverage" for value in eof_records
                ),
            }
            old_session = session
            session, source = "public-successor-capture", 0
            await feed(a[:1600], "successor_physical_capture", capture_epoch=9)
            successor = snapshot("successor_new_epoch_compact_zero")
            result["checks"]["successor_reset"] = {
                "status": (
                    "passed"
                    if successor.speaker_epoch > eof_epoch
                    and successor.processed_through_sample == 0
                    and backend._process.pid == child_pid
                    else "failed"
                )
            }
            stale_before = successor.stale_events
            owner.observe_input(old_session, event(block, 3200, sequence, 8))
            stale = snapshot("late_old_capture_callback")
            result["checks"]["late_callback"] = {
                "status": (
                    "passed"
                    if stale.stale_events > stale_before
                    and stale.speaker_epoch == successor.speaker_epoch
                    else "failed"
                )
            }
            large = np.array(np.resize(a, 16000), dtype=np.float32)
            large.flags.writeable = False
            delivery.release = asyncio.Event()
            owner.observe_input(session, event(large, source, sequence, 9))
            source += large.size
            await asyncio.wait_for(delivery.completed.wait(), 30)
            in_flight = snapshot("before_inflight_reset")
            owner.reset("inflight_probe_reset")
            delivery.release.set()
            await idle()
            delivery.release = None
            fenced = snapshot("after_inflight_reset")
            rejected = drain()
            result["checks"]["stale_result"] = {
                "status": (
                    "passed"
                    if in_flight.pending_items
                    and delivery.native_update_frames > 0
                    and fenced.speaker_epoch > in_flight.speaker_epoch
                    and fenced.processed_through_sample == 0
                    and fenced.immutable_through_sample == 0
                    and not any(value.frame is not None for value in rejected)
                    else "failed"
                ),
                "old_native_update_frames": delivery.native_update_frames,
                "delivery": "deterministically held a completed real native GPU update until after owner epoch reset; rejection established from public zero frontiers and absence of emitted frames, not stale-callback counter",
            }
            owner.observe_input(session, event(large, source, sequence + 1, 9))
            await asyncio.sleep(0.001)
            stopping = snapshot("before_real_inflight_stop")
            process = backend._process
            await owner.stop()
            stopped = snapshot("stopped")
            result["checks"]["inflight_stop"] = {
                "status": (
                    "passed"
                    if stopping.pending_items
                    and stopped.state == "stopped"
                    and stopped.owned_tasks == 0
                    and process.returncode is not None
                    else "unknown"
                )
            }
            result["capacity_peaks"] = cursor_peak
            result["status"] = (
                "passed"
                if all(check["status"] == "passed" for check in result["checks"].values())
                else "incomplete"
            )
        except Exception as error:
            result["status"], result["error"] = "failed", str(error)
        finally:
            await owner.stop()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                profile: {
                    "status": value["status"],
                    "checks": value["checks"],
                    "error": value.get("error"),
                }
                for profile, value in report["profiles"].items()
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
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
