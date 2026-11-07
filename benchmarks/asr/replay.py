from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from uuid import uuid4

import numpy as np
from score import percentiles

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadGating
from puripuly_heart.core.vad.silero import SileroVadOnnx


async def replay(adapter, manifest, pcm, root, variant, output):
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    gate = VadGating(
        SileroVadOnnx(root / "src/puripuly_heart/data/vad/silero_vad.onnx"),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        speech_threshold=0.3,
        continuation_threshold=0.2,
        hangover_ms=500,
    )
    epoch = f"benchmark-{uuid4().hex}"
    session = await adapter.backend.open_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch)
    )
    coordinator = session._decode_coordinator
    segments = {}
    sequence_to_id = {}
    timeline = []
    counts = Counter(delivered=0, failed=0, expired=0, dropped=0, replaced=0, merged=0, empty=0)
    origin = time.perf_counter()
    max_pending = 0
    final_queue = asyncio.Queue()
    stopped = None
    terminal_count = 0
    completion_handler = coordinator.on_completion
    failure_handler = coordinator.on_failure
    expiry_handler = coordinator.on_expired

    def details(job, inference_ms=None, queue_wait_ms=None):
        key = sequence_to_id.get(job.sequence)
        if key is not None:
            segments[key].update(
                decode_ms=inference_ms,
                queue_wait_ms=queue_wait_ms,
                native=adapter.observed.last_native,
            )

    async def completion(item):
        details(item.job, item.inference_ms, item.queue_wait_ms)
        await completion_handler(item)

    async def failure(item):
        details(item.job, item.inference_ms, item.queue_wait_ms)
        await failure_handler(item)

    async def expired(item):
        details(item.job, None, item.queue_wait_ms)
        await expiry_handler(item)

    coordinator.on_completion = completion
    coordinator.on_failure = failure
    coordinator.on_expired = expired

    async def receive():
        async for event in session.turn_events():
            if isinstance(event, STTProviderTurnTerminal):
                final_queue.put_nowait((event, time.perf_counter() - origin))

    async def consume():
        nonlocal terminal_count
        while True:
            event, producer_at = await final_queue.get()
            now = time.perf_counter() - origin
            key = event.identity.provider_turn_id
            row = segments[key]
            row.update(
                status=event.outcome,
                text=event.text,
                failure_reason=event.failure_reason,
                producer_final_s=producer_at,
                consumer_final_s=now,
                final_consumer_queue_ms=(now - producer_at) * 1000,
            )
            counts["delivered" if event.outcome == "final" else event.outcome] += 1
            terminal_count += 1
            end = row.get("estimated_speech_end_s")
            row["speech_end_to_consumer_ms"] = (now - end) * 1000 if end is not None else None
            final_queue.task_done()

    receive_task = asyncio.create_task(receive())
    consume_task = asyncio.create_task(consume())
    active = None
    payload_sequence = 0

    async def send(samples_f32):
        nonlocal payload_sequence
        payload_sequence += 1
        payload = (np.clip(samples_f32, -1, 1) * 32767).astype("<i2").tobytes()
        await session.send_turn_audio(
            active, payload, payload_sequence=payload_sequence, source_ranges=(), context_only=False
        )

    async def process(events, source_end):
        nonlocal active, payload_sequence, max_pending
        for event in events:
            if isinstance(event, SpeechStart):
                key = str(event.utterance_id)
                active = STTProviderTurnIdentity(
                    AudioSegmentIdentity(1, len(segments) + 1, event.utterance_id, 1), epoch, key
                )
                settings = AudioSegmentSettingsSnapshot(
                    adapter.backend.provider_id,
                    (adapter.backend.provider_id,),
                    (adapter.backend.provider_id,),
                    "microphone",
                    manifest["language"],
                    (manifest["language"],),
                    16000,
                    0.3,
                    500,
                    500,
                    0.2,
                )
                start = max(0, source_end - (len(event.pre_roll) + len(event.chunk)) / 16000)
                segments[key] = {
                    "id": key,
                    "source_start_s": start,
                    "source_end_s": None,
                    "status": "open",
                    "text": "",
                }
                await session.begin_turn(STTProviderTurnRequest(active, settings, channel="self"))
                payload_sequence = 0
                if event.pre_roll.size:
                    await send(event.pre_roll)
                await send(event.chunk)
            elif isinstance(event, SpeechChunk):
                await send(event.chunk)
            elif isinstance(event, SpeechEnd):
                key = active.provider_turn_id
                row = segments[key]
                row.update(
                    source_end_s=source_end,
                    enqueue_s=time.perf_counter() - origin,
                    seal_reason=event.reason,
                )
                overlapping = [
                    t
                    for t in manifest["turns"]
                    if t["offset_s"] < source_end
                    and t["offset_s"] + t["duration_s"] > row["source_start_s"]
                ]
                ends = [
                    t["speech_end_s"]
                    for t in overlapping
                    if row["source_start_s"] <= t.get("speech_end_s", -1) <= source_end
                ]
                row["estimated_speech_end_s"] = max(ends) if ends else None
                row["source_turn_ids"] = [t["id"] for t in overlapping]
                row["endpoint_ms"] = (
                    (row["enqueue_s"] - row["estimated_speech_end_s"]) * 1000 if ends else None
                )
                await session.seal_turn(
                    active,
                    sealed_content_ranges=(),
                    seal_reason=event.reason,
                    observed_trailing_silence_ms=event.trailing_silence_ms,
                )
                sequence_to_id[coordinator._next_sequence - 1] = key
                max_pending = max(max_pending, coordinator.pending_jobs)
                active = None

    try:
        for offset in range(0, len(samples), 512):
            source_end = min(len(samples), offset + 512) / 16000
            await asyncio.sleep(max(0, origin + source_end - time.perf_counter()))
            chunk = samples[offset : offset + 512]
            if chunk.size < 512:
                chunk = np.pad(chunk, (0, 512 - chunk.size))
            await process(gate.process_chunk(chunk), source_end)
            max_pending = max(max_pending, coordinator.pending_jobs)
            if offset % (512 * 16) == 0:
                timeline.append(
                    {
                        "source_s": source_end,
                        "wall_s": time.perf_counter() - origin,
                        "pending": coordinator.pending_jobs,
                        "buffered_audio_ms": coordinator.buffered_audio_ms,
                    }
                )
            if counts["failed"] >= 3 or coordinator.pending_jobs > 12:
                stopped = "three failures" if counts["failed"] >= 3 else "pending backlog >12"
                break
        admitted_end = source_end
        final_event = gate.seal_active(reason="source_eof")
        if final_event is not None:
            await process([final_event], admitted_end)
        ingress_end = time.perf_counter() - origin
        pending_at_ingress_end = coordinator.pending_jobs
        drain_deadline = time.perf_counter() + 45
        while terminal_count < len(segments) and time.perf_counter() < drain_deadline:
            await asyncio.sleep(0.05)
        if terminal_count < len(segments):
            stopped = stopped or "45s drain bound"
            client = getattr(adapter.backend, "_recognizer", None)
            if hasattr(client, "force_close"):
                await client.force_close()
        await final_queue.join()
        pending_after_drain = coordinator.pending_jobs
        for row in segments.values():
            if row["status"] in ("open",):
                row["status"] = "dropped"
                counts["dropped"] += 1
        turns = []
        for turn in manifest["turns"]:
            end = turn.get("speech_end_s")
            candidates = [
                r
                for r in segments.values()
                if end is not None and r["source_start_s"] <= end <= r["source_end_s"]
            ]
            finals = [r for r in candidates if r["status"] == "final"]
            selected = min(finals, key=lambda r: r["consumer_final_s"]) if finals else None
            turns.append(
                {
                    "id": turn["id"],
                    "speech_end_s": end,
                    "reference_available": turn.get("reference_available", False),
                    "input_end_covered": bool(candidates),
                    "nonempty_final_covers_end": bool(finals),
                    "segment_ids": [r["id"] for r in candidates],
                    "speech_end_to_consumer_ms": (
                        (selected["consumer_final_s"] - end) * 1000 if selected else None
                    ),
                }
            )
        rows = list(segments.values())
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        return {
            "language": manifest["language"],
            "variant": variant,
            "duration_s": manifest["duration_s"],
            "admitted_source_end_s": admitted_end,
            "ingress_wall_s": ingress_end,
            "total_wall_s": time.perf_counter() - origin,
            "counts": dict(counts),
            "segments": len(rows),
            "max_pending_including_active": max_pending,
            "pending_at_ingress_end": pending_at_ingress_end,
            "pending_after_drain": pending_after_drain,
            "recovery_after_ingress_s": max(0, time.perf_counter() - origin - ingress_end),
            "stop_reason": stopped,
            "segment_speech_end_to_consumer_ms": percentiles(
                [r.get("speech_end_to_consumer_ms") for r in rows if r["status"] == "final"]
            ),
            "source_turn_speech_end_to_consumer_ms": percentiles(
                [r["speech_end_to_consumer_ms"] for r in turns]
            ),
            "endpoint_ms": percentiles([r.get("endpoint_ms") for r in rows]),
            "queue_wait_ms": percentiles([r.get("queue_wait_ms") for r in rows]),
            "decode_ms": percentiles([r.get("decode_ms") for r in rows]),
            "final_consumer_queue_ms": percentiles(
                [r.get("final_consumer_queue_ms") for r in rows]
            ),
            "endpoint_multi_source_turn_segments": sum(
                len(r.get("source_turn_ids", [])) > 1 for r in rows
            ),
            "source_turns": turns,
            "queue_timeline": timeline,
            "boundary": "Independent 1x PCM producer -> current Silero Self endpoint -> actual scoped LocalOfflineSTTSession (FIFO pending TTL12s, 30s seal watchdog and current failure preservation) -> terminal event -> experiment final consumer queue. Bypasses capture-owner pre-provider retention/ingress limits. Not GUI/translation/TTS/overlay/network latency. Source coverage does not prove lexical correctness. First partial N/A; queue replaced/merged are zero by current policy, endpoint grouping separate.",
            "speech_end": "Recovered shared clean-source energy estimate inside human annotations, biased by background/overlap; not file end/enqueue or acoustic ground truth.",
        }
    finally:
        await session.close()
        receive_task.cancel()
        consume_task.cancel()
        await asyncio.gather(receive_task, consume_task, return_exceptions=True)
