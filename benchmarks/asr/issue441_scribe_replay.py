from __future__ import annotations

import argparse
import asyncio
import json
import wave
from collections import Counter
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

if __package__:
    from . import issue441_records as records
    from .issue441_lexical import canonical_hash, checked, digest, save_new
    from .issue441_scribe_score import MANIFEST, SELECTION, candidate_import_root, make_freeze
else:
    import issue441_records as records
    from issue441_lexical import canonical_hash, checked, digest, save_new
    from issue441_scribe_score import MANIFEST, SELECTION, candidate_import_root, make_freeze


E0_DIRECTORY = Path(".tmp/issue441/selected-runs/E0")


def recording_pack(root, recording, frozen):
    relative = E0_DIRECTORY / f"{recording['id']}.json"
    path = root / relative
    expected = next(
        value
        for key, value in frozen["preserved"].items()
        if Path(key).as_posix() == relative.as_posix()
    )
    checked(digest(path) == expected, "E0 retained input changed")
    run = records.load(path)
    native = [row for row in run["events"] if row["kind"] == "native"]
    checked(
        native == sorted(native, key=lambda row: (row["arrival_s"], row["seq"])),
        "Original E0 receipt order changed",
    )
    startup = [row for row in native if row["raw"].get("message_type") == "session_started"]
    checked(len(startup) == 1 and startup[0] is native[0], "E0 session opening unavailable")
    boundaries = [row for row in run["events"] if row["kind"] == "controller_boundary"]
    by_segment = {row["raw"]["segment"]["identity"]["segment_id"]: row for row in boundaries}
    states = {
        row["raw"]["segment_index"]: row
        for row in run["events"]
        if row["kind"] == "native_commit_control_state"
    }
    commits = {
        row["raw"]["segment_index"]: row
        for row in native
        if row["raw"].get("message_type") == "committed_transcript"
        and type(row["raw"].get("segment_index")) is int
    }
    commands, previous = [], 0
    for i, row in enumerate(run["sends"]):
        if row["status"] != "completed" or row["kind"] not in {"controller_commit", "eof_commit"}:
            continue
        end = round(row["transmitted_end_s"] * 16000)
        control = row["control"]
        candidates = [
            index
            for index, state in states.items()
            if state["raw"]["last_command"] is not None
            and state["raw"]["last_command"]["completed_s"] == row["completed_s"]
        ]
        index = candidates[0] if len(candidates) == 1 else None
        receipt = commits.get(index)
        boundary = by_segment.get(control.get("segment_id"))
        if boundary is None and row["kind"] == "eof_commit":
            seen = [item for item in boundaries if item["raw"]["event"]["reason"] == "source_eof"]
            boundary = seen[0] if len(seen) == 1 else None
        commands.append(
            {
                "send_index": i,
                "send_start_s": row["start_s"],
                "send_completion_s": row["completed_s"],
                "control": control,
                "source_ranges_samples": [[previous, end]],
                "retained_controller_boundary": boundary,
                "association": (
                    "last_completed_command_at_native_receipt_not_server_ack"
                    if receipt
                    else "unavailable"
                ),
                "segment_index": index,
                "commit_event_seq": receipt["seq"] if receipt else None,
                "commit_receipt_s": receipt["arrival_s"] if receipt else None,
                "commit_state_seq": states[index]["seq"] if receipt else None,
                "kind": row["kind"],
            }
        )
        previous = end
    connected = [
        item["commit_event_seq"] for item in commands if item["commit_event_seq"] is not None
    ]
    return {
        "schema_version": 1,
        "historical_arm": "E0",
        "recording": recording,
        "retained_log": {"path": relative.as_posix(), "sha256": expected, "status": run["status"]},
        "original_receipt_clock_origin_mono_s": run["source_origin_mono_s"],
        "provider_session_id": startup[0]["raw"]["session_id"],
        "historical_journal_session_id": startup[0]["session_id"],
        "native_frames": native,
        "original_startup_config": run["metadata"]["config"],
        "original_eof": run["eof"],
        "original_drain": run["drain"],
        "original_unassociated_commit_sequences": [
            row["seq"] for row in commits.values() if row["seq"] not in connected
        ],
        "lifecycle": "Start and chunks reconstructed from continuous PCM between completed manual command frontiers. Retained production-controller boundary, capture ID, settings and silence included when available, but that separate controller did not provide recognition turn receipts. Provider association is recorded last-command evidence, not inferred by text or sequence adjacency.",
        "command_turns": commands,
    }


def prepare(root, frozen, output):
    checked(frozen == make_freeze(root), "Source/method pin mismatch")
    output.mkdir(parents=True, exist_ok=False)
    manifest = records.load(root / MANIFEST)
    packs = {row["id"]: recording_pack(root, row, frozen) for row in manifest["recordings"]}
    for ident, pack in packs.items():
        save_new(output / f"{ident}.json", pack)
    selection = records.load(root / SELECTION)["selection"]
    checked(
        [row["id"] for row in selection] == [f"mt-{i:02d}" for i in range(1, 8)],
        "Selection changed",
    )
    for case in selection:
        pack = packs[case["recording_id"]]
        commit, metadata = case["E0"]["commit_seq"], case["E0"]["diarized_event_seq"]
        events = {row["seq"]: row for row in pack["native_frames"]}
        checked(
            events[commit]["raw"]["segment_index"]
            == events[metadata]["raw"]["segment_index"]
            == case["E0"]["index"],
            "Frozen interval does not share indexed native identity",
        )
        save_new(
            output / f"{case['id']}.json",
            {
                **pack,
                "frozen_case": case,
                "source_method_sha256": canonical_hash(frozen),
                "warmup": "Replay all source/session events before this interval unchanged for coherent context. Do not make service calls outside the selected requests.",
                "matching_native_event_seq": {"commit": commit, "diarization": metadata},
            },
        )
    return packs


def selected_requests(result, case):
    index = case["E0"]["index"]
    turn_ids = {
        turn["turn_id"] for turn in result["source_turns"] if turn["segment_index"] == index
    }
    picked = []
    for row in result["requests"]:
        lineage = row["lineage"] or {}
        if lineage.get("caption_id") in turn_ids or any(
            part.get("segment_index") == index for part in lineage.get("source_parts", [])
        ):
            picked.append(row["request_index"])
    return picked


def classify_metadata(previous, current, starts, revised):
    def partition(groups):
        return [
            [(item["source_parts"], item["source"]) for item in family["groups"]]
            for family in groups
        ]

    if partition(previous) != partition(current):
        return (
            "changed_partition_with_new_request"
            if starts
            else "changed_partition_without_backend_start"
        )
    if starts:
        return "unchanged_partition_with_new_request"
    if previous != current:
        return "identity_only"
    return "evidence_without_application_change" if revised else "no_product_revision_event"


def export_selection(root, packs, result_path, output):
    selection = records.load(root / SELECTION)["selection"]
    for case in selection:
        ident = case["recording_id"]
        pack = records.load(packs / f"{case['id']}.json")
        replay_path = result_path / f"{ident}.json"
        result = records.load(replay_path)
        selected = set(selected_requests(result, case))
        start, end = case["source_interval_s"]
        source_by_index = {
            turn["segment_index"]: turn["source_ranges_samples"]
            for turn in pack["command_turns"]
            if turn["segment_index"] is not None
        }

        def export_request(row):
            lineage = row["lineage"] or {}
            ranges = {
                part["segment_index"]: source_by_index[part["segment_index"]]
                for part in lineage.get("source_parts", [])
                if part.get("segment_index") in source_by_index
            }
            return {
                **row,
                "selected_native_segment": row["request_index"] in selected,
                "retained_command_source_ranges_samples": ranges,
                "command_source_intersects_frozen_interval": any(
                    max(start, a / 16000) < min(end, b / 16000)
                    for spans in ranges.values()
                    for a, b in spans
                ),
            }

        save_new(
            output / f"{case['id']}.json",
            {
                **pack,
                "candidate_replay": {
                    "pin": result["candidate"],
                    "path": str(replay_path),
                    "sha256": digest(replay_path),
                    "provider_epoch_id": result["provider_epoch_id"],
                    "failure": result["failure"],
                    "source_method": result["source_method"],
                },
                "request_lineage": [export_request(row) for row in result["requests"]],
                "selected_request_indices": sorted(selected),
                "selected_source_turns": [
                    row
                    for row in result["source_turns"]
                    if row["segment_index"] == case["E0"]["index"]
                ],
                "metadata_action": next(
                    (
                        row
                        for row in result["metadata_actions"]
                        if row["original_input_event_seq"] == case["E0"]["diarized_event_seq"]
                    ),
                    None,
                ),
            },
        )


class ReplayWire:
    def __init__(self, startup):
        self.rows = asyncio.Queue()
        self.rows.put_nowait(startup["raw"])
        self.awaiting_ack = False
        self.commits = 0

    async def recv(self):
        if self.awaiting_ack:
            self.rows.task_done()
        item = await self.rows.get()
        self.awaiting_ack = True
        return json.dumps(item, ensure_ascii=False)

    async def send(self, payload):
        self.commits += json.loads(payload).get("commit") is True

    async def deliver(self, row):
        await self.rows.put(row["raw"])
        await asyncio.wait_for(self.rows.join(), 3.0)

    async def close(self):
        pass


def parts_for(caption):
    parts = []
    for part in caption.scribe_parts:
        if part.commit is None:
            parts.append({"join_separator": " "})
            continue
        native = part.commit.native_segment
        parts.append(
            {
                "provider_session_id": native.provider_session_id,
                "segment_index": native.segment_index,
                "commit_char_range": [part.start, part.end],
                "source": part.commit.transcript.text[part.start : part.end],
            }
        )
    return parts


def groups_for(owner):
    families = {}
    for origin, family in owner._scribe_families.items():
        key = tuple(str(caption.utterance_id) for caption in family.captions)
        if key not in families:
            families[key] = {
                "origin_turn_ids": [],
                "groups": [
                    {
                        "caption_id": str(caption.utterance_id),
                        "order": caption.source_order,
                        "revision": caption.revision,
                        "source": caption.source,
                        "source_parts": parts_for(caption),
                        "speaker_assignment": records.encode(caption.speaker_assignment),
                    }
                    for caption in family.captions
                ],
            }
        families[key]["origin_turn_ids"].append(str(origin))
    return list(families.values())


def exact_coverage(groups, committed):
    copies = Counter()
    mismatches = []
    for family in groups:
        for group in family["groups"]:
            seen = []
            for part in group["source_parts"]:
                if "join_separator" in part:
                    seen.append(part["join_separator"])
                    continue
                index = part["segment_index"]
                a, b = part["commit_char_range"]
                seen.append(part["source"])
                copies.update((index, offset) for offset in range(a, b))
            if "".join(seen) != group["source"]:
                mismatches.append(group["caption_id"])
    missing = [
        [index, offset]
        for index, text in committed.items()
        for offset in range(len(text))
        if copies[index, offset] == 0
    ]
    return {
        "missing_commit_characters": missing,
        "multiply_covered_commit_characters": [
            [index, offset, count] for (index, offset), count in copies.items() if count != 1
        ],
        "group_text_mismatches": mismatches,
    }


async def replay(root, pack, output, candidate, source_method):
    from puripuly_heart.config.overlay_calibration import OverlayCalibration
    from puripuly_heart.config.prompts import get_default_prompt
    from puripuly_heart.core.audio.format import AudioCaptureSpan, pcm16le_bytes_to_float32
    from puripuly_heart.core.audio.ownership import (
        AudioSegmentSettingsSnapshot,
        PeerAudioSegmentLedger,
    )
    from puripuly_heart.core.clock import FakeClock
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner
    from puripuly_heart.core.overlay.presenter import OverlayPresenter
    from puripuly_heart.core.stt.backend import (
        STTFinalSpeakerEvidence,
        STTProviderEpochEnded,
        STTProviderTurnIdentity,
        STTProviderTurnRequest,
        STTProviderTurnTerminal,
        STTProviderTurnUpdate,
        STTSessionProjection,
        scoped_event_for_delivery,
    )
    from puripuly_heart.core.translation_backend import TranslationBackend
    from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
    from puripuly_heart.domain.models import Translation
    from puripuly_heart.providers.stt.elevenlabs_scribe import ElevenLabsScribeSTTBackend
    from tests.core.test_peer_overlay_policy import Timeline
    from tests.helpers.fakes import RecordingOscQueue
    from tests.helpers.translation_owners import compose_translation_test_harness

    rate = 16000
    provider = "elevenlabs_scribe"
    epoch_id = str(uuid5(NAMESPACE_URL, f"issue441:{candidate}:{pack['provider_session_id']}"))
    base = 1000.0
    frames = pack["native_frames"]
    wire = ReplayWire(frames[0])
    timeline = Timeline(clock=FakeClock(base + frames[0]["arrival_s"]))
    now = timeline.clock.now
    recording = pack["recording"]
    result = {
        "candidate": candidate,
        "source_method": source_method,
        "provider_session_id": pack["provider_session_id"],
        "provider_epoch_id": epoch_id,
        "retained_log": pack["retained_log"],
        "original_receipt_clock_origin_mono_s": pack["original_receipt_clock_origin_mono_s"],
        "mode": "Original E0 receive chronology; reconstructed source turns and immediate deterministic MT; production raw adapter, caption, output and logical presenter; no service or actual Desktop",
        "source_language": recording["language"],
        "target_language": "en" if recording["language"] == "ko" else "ko",
        "requests": [],
        "prepared_groups": [],
        "scoped_events": [],
        "source_turns": [],
        "snapshots": [],
        "metadata_actions": [],
        "failure": None,
    }
    current_seq = frames[0]["seq"]
    made = {}
    committed = {}
    request_lineage = ContextVar("request_lineage", default=None)

    class Backend(TranslationBackend):
        async def translate(self, request):
            result["requests"].append(
                {
                    "request_index": len(result["requests"]),
                    "arrival_s": now() - base,
                    "original_input_event_seq": current_seq,
                    "request": records.encode(request),
                    "lineage": request_lineage.get(),
                }
            )
            return Translation(utterance_id=request.utterance_id, text=request.text)

        async def close(self):
            pass

    class Surface:
        async def replace_snapshot(self, snapshot, *, block_expirations=None):
            result["snapshots"].append(
                {
                    "arrival_s": now() - base,
                    "original_input_event_seq": current_seq,
                    "snapshot": records.encode(snapshot),
                    "block_expirations": block_expirations,
                }
            )

        async def broadcast_shutdown(self):
            pass

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=Surface(),
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=Backend(),
        osc=RecordingOscQueue(),
        clock=timeline.clock,
        peer_source_language=recording["language"],
        peer_target_language=result["target_language"],
        system_prompt=get_default_prompt(),
        peer_translation_enabled=True,
        overlay_sink=presenter,
    )
    peer = harness.peer_owner
    captions = peer.cloud_captions
    captions.speaker_diarization = lambda provider_id: provider_id == provider
    prepare = PeerCaptionOwner._prepare
    execute = PeerCaptionOwner._execute

    def observe_preparation(owner, caption, source, kind, config):
        prepared = prepare(owner, caption, source, kind, config)
        result["prepared_groups"].append(
            {
                "arrival_s": now() - base,
                "original_input_event_seq": current_seq,
                "caption_id": str(caption.utterance_id),
                "source_order": caption.source_order,
                "revision": caption.revision,
                "kind": kind,
                "source": source,
                "parts": parts_for(caption),
                "frozen_context": caption.context,
                "speaker_assignment": records.encode(caption.speaker_assignment),
            }
        )
        return prepared

    async def observe_execution(owner, caption, source, kind, revision, *args, **kwargs):
        token = request_lineage.set(
            {
                "caption_id": str(caption.utterance_id),
                "source_order": caption.source_order,
                "revision": revision,
                "kind": kind,
                "complete_source": source,
                "source_parts": parts_for(caption),
            }
        )
        try:
            return await execute(owner, caption, source, kind, revision, *args, **kwargs)
        finally:
            request_lineage.reset(token)

    async def open_socket(_url, **_kwargs):
        return wire

    backend = ElevenLabsScribeSTTBackend(
        api_key="controlled-local-only", diarize=True, websocket_factory=open_socket
    )
    signature = (provider, recording["language"])
    settings = AudioSegmentSettingsSnapshot(
        provider, signature, signature, "manual", recording["language"], (), rate, 0.4, 900, 500
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    application_done = asyncio.Event()
    application_done.set()
    settled_turns = set()

    async def consume(session):
        async for event in session.turn_events():
            application_done.clear()
            try:
                result["scoped_events"].append(
                    {
                        "original_input_event_seq": current_seq,
                        "arrival_s": now() - base,
                        "event": records.encode(event),
                    }
                )
                accepted = scoped_event_for_delivery(event)
                if isinstance(accepted, STTProviderTurnUpdate):
                    await peer.handle_provider_update(replace(accepted, observed_at=now()))
                elif isinstance(accepted, STTFinalSpeakerEvidence):
                    await peer.handle_speaker_evidence(
                        accepted.revisions,
                        is_current=accepted.is_current,
                        speaker_merge=accepted.speaker_merge,
                    )
                elif isinstance(accepted, STTProviderTurnTerminal):
                    identity = accepted.identity
                    ledger.terminalize(
                        identity.segment.segment_id,
                        outcome=accepted.outcome,
                        now_monotonic_s=now(),
                        provider_epoch_id=identity.provider_epoch_id,
                        provider_turn_id=identity.provider_turn_id,
                        text_authority=accepted.text_authority,
                        failure_reason=accepted.failure_reason,
                    )
                    receipts = ledger.take_ready_terminal_receipts()
                    checked(len(receipts) == 1, "Unexpected reconstructed terminal order")
                    await peer.handle_provider_turn_terminal(receipts[0], accepted)
                    settled_turns.add(identity.segment.segment_order)
                    if accepted.native_segment is not None and accepted.text:
                        committed[accepted.native_segment.segment_index] = accepted.text
                elif isinstance(accepted, STTProviderEpochEnded):
                    await peer.handle_provider_epoch_ended(accepted)
            finally:
                application_done.set()

    async def idle():
        await asyncio.sleep(0)
        if receive_task.done():
            receive_task.result()
        await asyncio.wait_for(application_done.wait(), 3.0)
        await asyncio.wait_for(captions.wait_for_idle(), 3.0)
        await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), 3.0)

    async def advance(target):
        while due := [
            seg.next_check_at
            for seg in captions._segments.values()
            if not seg.closed and now() <= seg.next_check_at < target
        ]:
            await timeline.advance(max(0.0, min(due) - now()))
            await captions.check_previews()
            await idle()
        await timeline.advance(max(0.0, target - now()))

    source_path = root / recording["audio_path"]
    checked(digest(source_path) == recording["sha256"], "Replay WAV hash differs")
    with wave.open(str(source_path), "rb") as source:
        pcm = source.readframes(source.getnframes())
    retained_sends = records.load(root / pack["retained_log"]["path"])["sends"]
    sessions = []
    receive_task = None
    PeerCaptionOwner._prepare = observe_preparation
    PeerCaptionOwner._execute = observe_execution
    try:
        await harness.start()
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch_id))
        sessions.append(session)
        receive_task = asyncio.create_task(consume(session))
        schedule = []
        for order, command in enumerate(pack["command_turns"], 1):
            a, _ = command["source_ranges_samples"][0]
            previous_send = pack["command_turns"][order - 2]["send_index"] if order > 1 else -1
            chunks = [
                (i, row)
                for i in range(previous_send + 1, command["send_index"])
                if (row := retained_sends[i])["kind"] == "audio" and row["status"] == "completed"
            ]
            checked(bool(chunks), "Reconstructed E0 turn has no retained source chunks")
            checked(command["segment_index"] is not None, "Unassociated original command")
            previous_ready = (
                pack["command_turns"][order - 2]["commit_receipt_s"] if order > 1 else 0.0
            )
            ready = max(a / rate, previous_ready + 0.00000001)
            for sequence, (send_index, row) in enumerate(chunks, 1):
                schedule.append(
                    (
                        max(ready, row["completed_s"]),
                        send_index,
                        "audio",
                        order,
                        (command, sequence, row),
                    )
                )
            schedule.append(
                (command["send_start_s"], command["send_index"], "seal", order, command)
            )
        schedule.extend((row["arrival_s"], row["seq"], "native", 0, row) for row in frames[1:])

        for t, seq, kind, order, item in sorted(schedule):
            await advance(base + t)
            if kind == "native":
                current_seq = seq
                message = item["raw"]["message_type"]
                metadata = message in {"diarized_transcript", "speakers_merged"}
                offset = len(result["requests"])
                previous_events = len(result["scoped_events"])
                previous = groups_for(captions) if metadata else None
                await wire.deliver(item)
                await idle()
                if metadata:
                    current = groups_for(captions)
                    starts = list(range(offset, len(result["requests"])))
                    revised = any(
                        row["event"].get("$type", "").endswith(":STTFinalSpeakerEvidence")
                        for row in result["scoped_events"][previous_events:]
                    )
                    result["metadata_actions"].append(
                        {
                            "original_input_event_seq": seq,
                            "arrival_s": t,
                            "segment_index": item["raw"].get("segment_index"),
                            "message_type": message,
                            "requests_started": starts,
                            "classification": classify_metadata(previous, current, starts, revised),
                            "history": records.encode(harness.peer_runtime.translation_history),
                            "before": previous,
                            "after": current,
                        }
                    )
                continue
            if kind == "audio":
                item, sequence, row = item
                a, b = [round(row[key] * rate) for key in ("source_start_s", "source_end_s")]
            boundary = item["retained_controller_boundary"]
            boundary_event = boundary["raw"]["event"] if boundary else {}
            if kind == "audio" and sequence == 1:
                checked(
                    order == 1 or order - 1 in settled_turns,
                    "Replay turn started before original predecessor settled",
                )
                uid = (
                    UUID(boundary_event["utterance_id"])
                    if boundary_event
                    else uuid5(NAMESPACE_URL, f"issue441:{pack['provider_session_id']}:{order}")
                )
                span = AudioCaptureSpan(0, a // 512, rate, a, b, base + a / rate, base + b / rate)
                payload = pcm[2 * a : 2 * b]
                data = pcm16le_bytes_to_float32(payload)
                owned = ledger.observe_vad_event(
                    SpeechStart(uid, data[:0], data, (), (span,)),
                    now_monotonic_s=now(),
                )
                identity = STTProviderTurnIdentity(
                    owned.segment.identity,
                    epoch_id,
                    str(uid),
                    (provider, signature, signature),
                    recognizer_id=provider,
                )
                made[order] = identity
                await session.begin_turn(STTProviderTurnRequest(identity, settings, "peer"))
                await session.send_turn_audio(
                    identity, payload, payload_sequence=1, source_ranges=(span,), context_only=False
                )
                result["source_turns"].append(
                    {
                        "order": order,
                        "turn_id": str(uid),
                        "segment_index": item["segment_index"],
                        "source_sample_range": item["source_ranges_samples"][0],
                        "reconstructed_begin_s": t,
                        "retained_boundary_seq": boundary["seq"] if boundary else None,
                        "lifecycle_known_at_receipt": "Source block is reconstructed, not historical product ownership",
                    }
                )
            elif kind == "audio":
                identity = made[order]
                span = AudioCaptureSpan(0, a // 512, rate, a, b, base + a / rate, base + b / rate)
                payload = pcm[2 * a : 2 * b]
                ledger.observe_vad_event(
                    SpeechChunk(
                        identity.segment.segment_id, pcm16le_bytes_to_float32(payload), (span,)
                    ),
                    now_monotonic_s=now(),
                )
                await session.send_turn_audio(
                    identity,
                    payload,
                    payload_sequence=sequence,
                    source_ranges=(span,),
                    context_only=False,
                )
            else:
                identity = made[order]
                reason = boundary_event.get("reason", "source_eof")
                silence = boundary_event.get("trailing_silence_ms", 0)
                owned = ledger.observe_vad_event(
                    SpeechEnd(identity.segment.segment_id, silence, reason),
                    now_monotonic_s=now(),
                )
                await session.seal_turn(
                    identity,
                    sealed_content_ranges=owned.segment.content_ranges,
                    seal_reason=reason,
                    observed_trailing_silence_ms=silence,
                )
                result["source_turns"][-1].update(
                    reconstructed_seal_s=t,
                    seal_reason=reason,
                    trailing_silence_ms=silence if boundary else None,
                    original_commit_event_seq=item["commit_event_seq"],
                )
        result["commit_commands_completed"] = wire.commits
        result["terminal_turns"] = len(settled_turns)
        checked(
            len(settled_turns) == wire.commits == len(pack["command_turns"]),
            "Replay lost a terminal turn",
        )
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "event_seq": current_seq,
        }
        raise
    finally:
        result["counts"] = dict(captions.request_counts)
        result["history"] = records.encode(harness.peer_runtime.translation_history)
        result["groups"] = groups_for(captions)
        result["source_coverage"] = exact_coverage(result["groups"], committed)
        PeerCaptionOwner._prepare = prepare
        PeerCaptionOwner._execute = execute
        save_new(output, result)
        for session in sessions:
            await session.close()
        if receive_task is not None:
            await asyncio.gather(receive_task, return_exceptions=True)
        await harness.stop()
        await presenter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packs", type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--recording")
    args = parser.parse_args()
    root = args.root.resolve()
    freeze = records.load(args.frozen)
    if args.packs is None:
        prepare(root, freeze, args.output)
        return 0
    checked(freeze == make_freeze(root), "Source/method pin mismatch")
    pin = candidate_import_root(root, args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    method = {
        "file_sha256": digest(args.frozen),
        "canonical_sha256": canonical_hash(freeze),
    }

    async def execute_suite():
        failures = []
        for rec in freeze["recordings"]:
            ident = rec["id"]
            if args.recording is not None and ident != args.recording:
                continue
            pack = records.load(args.packs / f"{ident}.json")
            checked(
                pack == recording_pack(root, pack["recording"], freeze),
                "Replay pack differs from retained source",
            )
            try:
                await replay(root, pack, args.output / f"{ident}.json", pin, method)
            except Exception as exc:
                failures.append(
                    {"recording_id": ident, "type": type(exc).__name__, "reason": str(exc)}
                )
        checked(freeze == make_freeze(root), "Original evidence drifted during structural replay")
        save_new(args.output / "suite.json", {"candidate": pin, "failures": failures})
        if args.recording is None and not failures:
            export_selection(root, args.packs, args.output, args.output / "selection")
        return int(bool(failures))

    return asyncio.run(execute_suite())


if __name__ == "__main__":
    raise SystemExit(main())
