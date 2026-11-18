from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from contextlib import AsyncExitStack
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import issue441_records as records
    from .issue441_lexical import canonical_hash, checked, digest, save_new
    from .issue441_score import source_lexical_unit
    from .issue441_scribe_replay import groups_for, parts_for
    from .issue441_scribe_score import ARM, SELECTION, candidate_import_root
else:
    import issue441_records as records
    from issue441_lexical import canonical_hash, checked, digest, save_new
    from issue441_score import source_lexical_unit
    from issue441_scribe_replay import groups_for, parts_for
    from issue441_scribe_score import ARM, SELECTION, candidate_import_root


ADMISSION = ContextVar("i441_admission", default=None)
REQUEST = ContextVar("i441_group", default=None)


def current_request_lineage():
    return REQUEST.get()


def request_key(lineage):
    if lineage is None:
        return None
    return canonical_hash(
        {
            "admission_ordinal": lineage["admission_ordinal"],
            "source_order": lineage["source_order"],
            "kind": lineage["kind"],
            "revision": lineage["revision"],
            "source_parts": lineage["source_parts"],
        }
    )


def request_body(request):
    fields = records.plain(records.encode(request))
    return {key: value for key, value in fields.items() if key != "utterance_id"}


def reserve_selected(selected, used, key, body):
    target = selected.get(key)
    if target is None:
        return False
    checked(
        key not in used and target["body"] == body,
        "Selected I441 request lineage/body not exactly matched",
    )
    used.add(key)
    return True


class Application:
    def __init__(
        self,
        recording,
        clock,
        origin,
        *,
        overlay_sink=None,
        translation_backend=None,
        selected=(),
        checkpoint=None,
        journal=None,
    ):
        self.recording = recording
        self.clock = clock
        self.origin = origin
        self.presenter = overlay_sink
        self.backend = translation_backend
        self.selected = {row["request_key"]: row for row in selected}
        checked(len(selected) == len(self.selected), "Duplicate selected request lineage")
        checked(
            not selected or translation_backend is not None,
            "Selected replay requires a supplied MT backend",
        )
        self.checkpoint = checkpoint
        self.journal = journal
        self.used = set()
        self.trace = {
            "admissions": [],
            "requests": [],
            "preparations": [],
            "snapshots": [],
            "group_checkpoints": [],
        }
        self._stack = AsyncExitStack()

    def now(self):
        return self.clock.now() - self.origin()

    async def notify(self, kind, row):
        if self.journal is not None:
            self.journal(kind, row)
        if self.checkpoint is not None:
            await self.checkpoint(kind, row)

    async def __aenter__(self):
        from puripuly_heart.config.overlay_calibration import OverlayCalibration
        from puripuly_heart.config.prompts import get_default_prompt
        from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner
        from puripuly_heart.core.overlay.presenter import OverlayPresenter
        from puripuly_heart.core.translation_backend import TranslationBackend
        from puripuly_heart.domain.models import Translation
        from tests.helpers.fakes import RecordingOscQueue
        from tests.helpers.translation_owners import compose_translation_test_harness

        app = self

        class Backend(TranslationBackend):
            async def translate(self, request):
                lineage = REQUEST.get()
                key, body = request_key(lineage), request_body(request)
                row = {
                    "request_index": len(app.trace["requests"]),
                    "t": app.now(),
                    "request_id": str(request.utterance_id),
                    "lineage": lineage,
                    "request_key": key,
                    "body": body,
                    "disposition": "started",
                }
                app.trace["requests"].append(row)
                await app.notify("request_started", row)
                try:
                    row["service"] = "not_dispatched"
                    if reserve_selected(app.selected, app.used, key, body):
                        row["service"] = "caller_backend"
                        response = await app.backend.translate(request)
                    else:
                        row["service"] = "controlled"
                        response = Translation(utterance_id=request.utterance_id, text=request.text)
                    row["disposition"] = "backend_returned"
                    return response
                except BaseException as exc:
                    row["disposition"] = type(exc).__name__
                    raise
                finally:
                    row["ended_t"] = app.now()
                    await app.notify("request_settled", row)

            async def close(self):
                pass

        class Surface:
            async def replace_snapshot(self, snapshot, *, block_expirations=None):
                row = {
                    "t": app.now(),
                    "snapshot": records.encode(snapshot),
                    "block_expirations": block_expirations,
                }
                app.trace["snapshots"].append(row)
                await app.notify("presenter_snapshot", row)

            async def broadcast_shutdown(self):
                pass

        if self.presenter is None:
            self.presenter = OverlayPresenter(
                calibration=OverlayCalibration(), clock=self.clock, bridge=Surface()
            )
            self._stack.push_async_callback(self.presenter.close)
        language = self.recording["language"]
        self.harness = compose_translation_test_harness(
            stt=None,
            llm=Backend(),
            osc=RecordingOscQueue(),
            clock=self.clock,
            peer_source_language=language,
            peer_target_language="en" if language == "ko" else "ko",
            system_prompt=get_default_prompt(),
            peer_translation_enabled=True,
            overlay_sink=self.presenter,
        )
        self.peer = self.harness.peer_owner
        captions = self.peer.cloud_captions
        captions.speaker_diarization = lambda provider: provider == "elevenlabs_scribe"
        prepare, execute = PeerCaptionOwner._prepare, PeerCaptionOwner._execute

        def describe(caption, source, kind, revision):
            return {
                "admission_ordinal": ADMISSION.get(),
                "caption_id": str(caption.utterance_id),
                "source_order": caption.source_order,
                "revision": revision,
                "kind": kind,
                "complete_source": source,
                "source_parts": parts_for(caption),
            }

        def observe_prepare(owner, caption, source, kind, config):
            value = prepare(owner, caption, source, kind, config)
            if owner is captions:
                self.trace["preparations"].append(
                    {
                        **describe(caption, source, kind, caption.revision),
                        "prepared": records.encode(value) if value is not None else None,
                        "t": self.now(),
                        "frozen_context": caption.context,
                    }
                )
            return value

        async def observe_execute(owner, caption, source, kind, revision, *args, **kwargs):
            if owner is not captions:
                return await execute(owner, caption, source, kind, revision, *args, **kwargs)
            token = REQUEST.set(describe(caption, source, kind, revision))
            try:
                return await execute(owner, caption, source, kind, revision, *args, **kwargs)
            finally:
                REQUEST.reset(token)

        self._stack.enter_context(patch.object(PeerCaptionOwner, "_prepare", observe_prepare))
        self._stack.enter_context(patch.object(PeerCaptionOwner, "_execute", observe_execute))
        self._stack.push_async_callback(self.harness.stop)
        try:
            await self.harness.start()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *_):
        await self._stack.aclose()

    def bind_live_capture(self, capture_owner, run):
        from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
            TranslationChannelOwnerCallbacks,
        )
        from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
        from puripuly_heart.core.stt.backend import scoped_event_for_delivery

        callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
        callbacks.bind_peer(self.peer)
        callbacks.bind_peer_capture(capture_owner)
        capture_owner.bind_publication_generation_observer(
            activated=lambda value: self.generation("activate", value),
            retired=lambda value: self.generation("retire", value),
        )
        current_row = ContextVar("i441_scoped_event_row", default=None)
        peer_type = type(self.peer)

        def observe(method, kind):
            async def wrapped(peer, *args, **kwargs):
                if peer is not self.peer:
                    return await method(peer, *args, **kwargs)
                if kind == "update" and scoped_event_for_delivery(args[0]) is None:
                    return await method(peer, *args, **kwargs)
                if kind == "terminal":
                    run.row("final", receipt=args[0], event=args[1])
                return await self.deliver(
                    kind,
                    *args,
                    speaker_merge=kwargs.get("speaker_merge"),
                    scoped_row_index=current_row.get(),
                    action=lambda: method(peer, *args, **kwargs),
                )

            return wrapped

        for attr, kind in (
            ("handle_provider_update", "update"),
            ("handle_provider_turn_terminal", "terminal"),
            ("handle_speaker_evidence", "speaker_evidence"),
            ("handle_provider_epoch_ended", "epoch_ended"),
        ):
            self._stack.enter_context(
                patch.object(peer_type, attr, observe(getattr(peer_type, attr), kind))
            )

        async def dispatch(event, row_index):
            token = current_row.set(row_index)
            try:
                await callbacks.peer_event_handler(event)
            finally:
                current_row.reset(token)

        return dispatch

    def generation(self, kind, value, *, row=None):
        if row is None:
            row = {
                "ordinal": len(self.trace["admissions"]),
                "kind": kind,
                "t": self.now(),
                "mono": self.clock.now(),
                "generation": value,
            }
        self.trace["admissions"].append(row)
        if self.journal is not None:
            self.journal("application_ingress", row)
        output = self.harness.output_runtime
        if kind == "activate":
            output.activate_peer_generation(value)
        else:
            output.retire_peer_generation(value)

    async def deliver(
        self,
        kind,
        *values,
        is_current=None,
        speaker_merge=None,
        scoped_row_index=None,
        row=None,
        action=None,
    ):
        if row is None:
            row = {
                "ordinal": len(self.trace["admissions"]),
                "kind": kind,
                "t": self.now(),
                "mono": self.clock.now(),
                "values": records.encode(values),
                "scoped_row_index": scoped_row_index,
            }
            if kind == "speaker_evidence":
                row["speaker_merge"] = records.encode(speaker_merge)
        self.trace["admissions"].append(row)
        token = ADMISSION.set(row["ordinal"])
        try:
            await self.notify("application_ingress", row)
            if action is not None:
                await action()
            elif kind == "update":
                await self.peer.handle_provider_update(values[0])
            elif kind == "terminal":
                await self.peer.handle_provider_turn_terminal(*values)
            elif kind == "speaker_evidence":
                await self.peer.handle_speaker_evidence(
                    values[0], is_current=is_current, speaker_merge=speaker_merge
                )
            elif kind == "epoch_ended":
                await self.peer.handle_provider_epoch_ended(values[0])
            elif kind == "input_release":
                self.peer.release_input_segment(values[0])
            else:
                raise ValueError(f"Unsupported recorded Peer admission: {kind}")
            if kind in {"terminal", "speaker_evidence", "epoch_ended"}:
                self.trace["group_checkpoints"].append(
                    {
                        "ordinal": row["ordinal"],
                        "t": self.now(),
                        "groups": groups_for(self.peer.cloud_captions),
                        "history": records.encode(self.harness.peer_runtime.translation_history),
                    }
                )
            await self.notify("application_returned", row)
        finally:
            ADMISSION.reset(token)

    async def settle(self):
        async with asyncio.timeout(35):
            await self.peer.cloud_captions.wait_for_idle()
            await self.harness.output_runtime.wait_for_peer_output_idle()
        self.trace["counts"] = dict(self.peer.cloud_captions.request_counts)
        self.trace["unreached_selected_request_keys"] = sorted(set(self.selected) - self.used)
        await self.notify(
            "application_settled",
            {
                "t": self.now(),
                "counts": self.trace["counts"],
                "unreached_selected_request_keys": self.trace["unreached_selected_request_keys"],
            },
        )


def decode_speaker_admission(row, scoped_rows, values):
    from puripuly_heart.core.stt.backend import STTFinalSpeakerEvidence

    index = row["scoped_row_index"]
    checked(type(index) is int and 0 <= index < len(scoped_rows), "Missing scoped speaker evidence")
    linked = scoped_rows[index]
    checked(linked["kind"] == "scoped_event", "Speaker evidence must link to a scoped event")
    event = records.decode(linked["event"])
    checked(isinstance(event, STTFinalSpeakerEvidence), "Unexpected scoped speaker evidence type")
    revisions = values[0]
    checked(
        all(
            revision in event.revisions
            and revision.identity.provider_epoch_id == event.provider_epoch_id
            for revision in revisions
        ),
        "Recorded speaker revisions do not match admitted scoped identity",
    )
    merge = event.speaker_merge
    if "speaker_merge" in row:
        checked(
            records.decode(row["speaker_merge"]) == merge,
            "Recorded speaker merge does not match scoped evidence",
        )
    if merge is not None:
        checked(
            all(
                revision.native_segment.provider_session_id == merge.provider_session_id
                for revision in revisions
            ),
            "Speaker merge belongs to another native session",
        )
    return event.provider_epoch_id, merge


async def replay_admitted(
    pack, *, overlay_sink, translation_backend, clock, checkpoint=None, advance_to=None
):
    from puripuly_heart.core.stt.backend import STTProviderTurnUpdate

    ingress = pack["application"]["admissions"]
    checked(bool(ingress), "No actual I441 admission to replay")
    first = min(0.0, ingress[0]["t"])
    origin = clock.now() - first
    live_epochs = set()
    async with Application(
        pack["recording"],
        clock,
        lambda: origin,
        overlay_sink=overlay_sink,
        translation_backend=translation_backend,
        selected=pack["selected_requests"],
        checkpoint=checkpoint,
    ) as app:
        for row in ingress:
            target = origin + row["t"]
            if advance_to is None:
                await asyncio.sleep(max(0.0, target - clock.now()))
            else:
                await advance_to(target)
            kind = row["kind"]
            if kind in {"activate", "retire"}:
                app.generation(kind, row["generation"], row=row)
                if checkpoint is not None:
                    await checkpoint("application_ingress", row)
                continue
            values = records.decode(row["values"])
            guard = None
            speaker_merge = None
            if kind in {"update", "terminal"}:
                event = values[-1]
                live_epochs.add(event.identity.provider_epoch_id)
                if isinstance(event, STTProviderTurnUpdate):
                    delta = origin - pack["source_origin_mono_s"]
                    values = (
                        dataclasses.replace(
                            event,
                            observed_at=(
                                None if event.observed_at is None else event.observed_at + delta
                            ),
                            first_nonempty_at=(
                                None
                                if event.first_nonempty_at is None
                                else event.first_nonempty_at + delta
                            ),
                        ),
                    )
            elif kind == "speaker_evidence":
                epoch, speaker_merge = decode_speaker_admission(row, pack["owned_rows"], values)

                def guard(epoch=epoch):
                    return epoch in live_epochs

            elif kind == "epoch_ended":
                live_epochs.discard(values[0].provider_epoch_id)
            await app.deliver(kind, *values, is_current=guard, speaker_merge=speaker_merge, row=row)
        await app.settle()
        return app.trace


def actual_source_index(capture):
    scopes = {}
    for terminal_index, row in enumerate(capture["rows"]):
        if row["kind"] != "final":
            continue
        event, receipt = row["event"], row["receipt"]
        native = event.get("native_segment")
        if native is not None:
            key = (native["provider_session_id"], native["segment_index"])
            scopes[key] = {
                "text": event["text"],
                "segment": receipt["segment"],
                "provider_epoch_id": receipt["provider_epoch_id"],
                "provider_turn_id": receipt["provider_turn_id"],
                "terminal_row_index": terminal_index,
                "mapped_words": [],
                "commit_native_event_seqs": [],
                "completed_commit_write_indices": [],
            }
    observations = {}
    for word in capture["observations"]:
        if word["kind"] == "diarized_transcript":
            observations.setdefault(word["event_seq"], []).append(word)
    for row in capture["events"]:
        if row["kind"] != "native":
            continue
        if row["raw"].get("message_type") == "committed_transcript":
            entry = scopes.get((row["session_id"], row["raw"].get("segment_index")))
            if entry is not None:
                entry["commit_native_event_seqs"].append(row["seq"])
            continue
        if row["raw"].get("message_type") != "diarized_transcript":
            continue
        raw = row["raw"]
        key = (row["session_id"], raw.get("segment_index"))
        entry = scopes.get(key)
        text = raw.get("text", "")
        words = observations.get(row["seq"], [])
        if (
            entry is None
            or text.strip() != entry["text"]
            or "".join(w["text"] for w in words) != text
        ):
            continue
        offset = 0
        left = len(text) - len(text.lstrip())
        for word in words:
            end = offset + len(word["text"])
            a, b = max(0, offset - left), min(len(entry["text"]), end - left)
            if a < b and word["mapping_status"] == "mapped" and source_lexical_unit(word):
                entry["mapped_words"].append(
                    {
                        "chars": [a, b],
                        "source_ranges": word["source_ranges"],
                        "event_seq": row["seq"],
                        "word_index": word["word_index"],
                    }
                )
            offset = end
    by_turn = {
        (session, value["provider_epoch_id"], value["provider_turn_id"]): value
        for (session, _), value in scopes.items()
    }
    for i, row in enumerate(capture["sends"]):
        if row["status"] != "completed" or row["kind"] not in {"controller_commit", "eof_commit"}:
            continue
        identity = row["identity"]
        entry = by_turn.get(
            (row["session_id"], identity["provider_epoch_id"], identity["provider_turn_id"])
        )
        if entry is not None:
            entry["completed_commit_write_indices"].append(i)
    return scopes


def match_interval(row, source_index, interval):
    lineage = row["lineage"]
    if not lineage or lineage["kind"] == "preview":
        return "not_complete_group", []
    localized, owned_overlap = [], False
    start, end = interval
    for part in lineage["source_parts"]:
        key = part.get("provider_session_id"), part.get("segment_index")
        observed = source_index.get(key)
        if observed is None:
            continue
        a, b = part["commit_char_range"]
        for word in observed["mapped_words"]:
            if max(a, word["chars"][0]) < min(b, word["chars"][1]) and any(
                max(start, span["start_s"]) < min(end, span["end_s"])
                for span in word["source_ranges"]
            ):
                localized.append({"provider_session_id": key[0], "segment_index": key[1], **word})
        owned_overlap |= any(
            max(start, span["source_start_sample"] / span["source_sample_rate_hz"])
            < min(end, span["source_end_sample"] / span["source_sample_rate_hz"])
            for span in observed["segment"]["content_ranges"]
        )
    if localized:
        return "localized_in_frozen_interval", localized
    return (
        "owned_range_intersects_without_localized_group"
        if owned_overlap
        else "no_localized_intersection"
    ), []


def make_selection(capture, case):
    index = actual_source_index(capture)
    requests = []
    for row in capture["application"]["requests"]:
        status, evidence = match_interval(row, index, case["source_interval_s"])
        requests.append({**row, "selection_status": status, "source_word_evidence": evidence})
    selected = [
        row
        for row in requests
        if row["selection_status"] == "localized_in_frozen_interval"
        and row["disposition"] == "backend_returned"
    ]
    return {
        "status": "selected" if selected else "unavailable_no_localized_completed_request",
        "request_lineage": requests,
        "selected_requests": selected,
    }


def export_fresh(root, frozen, capture_directory, output, candidate, capture_candidate):
    method = frozen["fresh_application_and_display"]
    selected = set(method["selected_cases"])
    checked(digest(root / SELECTION) == frozen["preserved"][SELECTION], "Frozen selection drift")
    output.mkdir(parents=True, exist_ok=False)
    cases = [row for row in records.load(root / SELECTION)["selection"] if row["id"] in selected]
    recording_ids = {row["recording_id"] for row in cases}
    outputs = []
    for recording in frozen["recordings"]:
        if recording["id"] not in recording_ids:
            continue
        path = capture_directory / f"{recording['id']}.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        capture = records.plain(stored)
        meta = capture["metadata"]
        checked(
            (capture["arm"], capture["recording_id"], meta["checkout"], meta["input_sha256"])
            == (ARM, recording["id"], capture_candidate, recording["sha256"]),
            "Capture identity drift",
        )
        checked(
            meta["live_provider"] and meta["source_method_sha256"] == canonical_hash(frozen),
            "Controlled or differently frozen source cannot feed natural I441 evidence",
        )
        pack = {
            "schema_version": 1,
            "recording": recording,
            "candidate": candidate,
            "source_method_sha256": canonical_hash(frozen),
            "source_method_file_sha256": meta["source_method_file_sha256"],
            "source_origin_mono_s": capture["source_origin_mono_s"],
            "capture": {
                "candidate": capture_candidate,
                "path": str(path),
                "sha256": digest(path),
                "status": capture["status"],
            },
            "boundary": "Admitted production Peer input from the captured I441 session; no raw rejoin or reconstructed source ownership in Desktop replay",
            "transport_sessions": capture["transport_sessions"],
            "session_scopes": capture["session_scopes"],
            "eof": capture["eof"],
            "drain": capture["drain"],
            "native_frames": [row for row in capture["events"] if row["kind"] == "native"],
            "owned_rows": stored["rows"],
            "native_writes": stored["sends"],
            "terminal_source_links": [
                {"provider_session_id": session, "segment_index": index, **value}
                for (session, index), value in actual_source_index(capture).items()
            ],
            "application": stored.get("application", {"admissions": [], "requests": []}),
            "selected_requests": [],
        }
        for case in cases:
            if case["recording_id"] != recording["id"]:
                continue
            capture.setdefault("application", {"admissions": [], "requests": []})
            selection_path = output / "selection" / f"{case['id']}.json"
            selection = make_selection(capture, case)
            save_new(
                selection_path,
                {
                    **pack,
                    "frozen_case_id": case["id"],
                    "source_interval_s": case["source_interval_s"],
                    **selection,
                },
            )
            outputs.append(
                {
                    "case": case["id"],
                    "path": str(selection_path),
                    "sha256": digest(selection_path),
                    "status": selection["status"],
                    "selected_requests": len(selection["selected_requests"]),
                }
            )
    checked({row["case"] for row in outputs} == selected, "Missing authorized display case")
    save_new(
        output / "selection-index.json",
        {
            "candidate": candidate,
            "capture_candidate": capture_candidate,
            "cases": sorted(outputs, key=lambda row: row["case"]),
            "superseded_display_cases": method["superseded_display_cases"],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--capture-candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = candidate_import_root(root, args.candidate)
    export_fresh(
        root,
        records.load(args.frozen),
        args.captures,
        args.output,
        candidate,
        args.capture_candidate,
    )


if __name__ == "__main__":
    main()
