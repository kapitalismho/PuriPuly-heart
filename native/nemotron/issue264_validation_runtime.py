from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import issue218_experiment as local
from issue218_policy_visibility import GuardedDesktop

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.local_translation.assets import validate_gemma_install
from puripuly_heart.core.local_translation.provisioning import GemmaProvisioningCancelled
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.overlay.manifest import (
    OVERLAY_CONTRACT_VERSION,
    OVERLAY_SPEAKER_IDENTITY_CONTRACT,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.stt import speaker_attribution
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
from puripuly_heart.core.stt.scoped_engine import STTRecognitionWatchdogs
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.core.vad.gating import SpeechChunk
from puripuly_heart.providers.stt import deepgram

prior = local.prior
CASES = (
    "clean_A_B",
    "distinct_repeated_A_B_A",
    "short_A_B_A",
    "linguistic_correction",
    "six_second_rollover",
)


def save(path, value):
    text = json.dumps(prior.serial(value), indent=2, ensure_ascii=False, default=str)
    for home in (str(Path.home()), str(Path.home()).replace("\\", "/")):
        text = text.replace(home.replace("\\", "\\\\"), "<USER_HOME>").replace(home, "<USER_HOME>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


def namespaces(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespaces(v) for k, v in value.items()})
    if isinstance(value, list):
        return [namespaces(v) for v in value]
    return value


def tapes(args, cases):
    with zipfile.ZipFile(args.native_archive) as archive:
        raw = archive.read(args.native_member)
    historical = json.loads(raw)
    selected = {
        c["case"]: c
        for c in historical["cases"]
        if c["mode"] == "baseline" and c["repetition"] == 1
    }
    result = {}
    for name in args.cases.split(","):
        case = selected[name]
        values = cases[name][0]
        if hashlib.sha256(values.tobytes()).hexdigest() != case["input_pcm_f32_sha256"]:
            raise ValueError("Replay capture hash mismatch: " + name)
        pcm = float32_to_pcm16le_bytes(values)
        payload_bytes = []
        for event in case["trace"]:
            if event["event"] != "native_submit":
                continue
            submitted = []
            for payload in event["payloads"]:
                data = b"".join(
                    pcm[s["normalized_start_sample"] * 2 : s["normalized_end_sample"] * 2]
                    for s in payload["source_ranges"]
                )
                if (
                    hashlib.sha256(data).hexdigest() != payload["pcm16_sha256"]
                    or len(data) // 2 != payload["samples"]
                ):
                    raise ValueError("Replay mapped payload hash mismatch: " + name)
                submitted.append(data)
            data = b"".join(submitted)
            if hashlib.sha256(data).hexdigest() != event["pcm16_sha256"]:
                raise ValueError("Replay request hash mismatch: " + name)
            payload_bytes.append(data)
        cursor = 0
        results = []
        controls = []
        for event in case["trace"]:
            if event["event"] == "wire_write_complete":
                cursor = event["successful_sent_samples_after"]
                if event.get("control"):
                    controls.append((event["control"], cursor))
            if event["event"] == "raw_wire_result":
                results.append(
                    {
                        "delay": event["at"] - case["playback_started_at"],
                        "cursor": cursor,
                        "value": event["value"],
                    }
                )
        data = b"".join(payload_bytes)
        if len(data) // 2 != case["actual_audio_samples_written"]:
            raise ValueError("Replay successful cursor mismatch: " + name)
        result[name] = {"case": case, "pcm": data, "results": results, "controls": controls}
    return result, {
        "archive": str(args.native_archive),
        "archive_sha256": prior.digest(args.native_archive),
        "member": args.native_member,
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "historical_candidate": historical["candidate"],
        "clock": "Only playback host origin rebased; lexical clocks, successful PCM maps, receipt delays and controls retained. Native ASR timings are historical, not fresh cloud latency.",
    }


class CaptureReceipts:
    provider_source_snapshot = PeerCaptureSessionOwner.provider_source_snapshot
    admit_provider_terminal = PeerCaptureSessionOwner.admit_provider_terminal
    note_input_terminal = PeerCaptureSessionOwner.note_input_terminal
    owns_provider_turn = PeerCaptureSessionOwner.owns_provider_turn

    def __init__(self, ledger, clock, engine):
        self._segment_ledgers = [ledger]
        self._publication_generations = {ledger.activation_generation}
        self._provider_terminal_events = {}
        self.clock = clock
        self.engine = engine
        self.ledger = ledger

    def is_current_recognition_stream(self, stream):
        return (
            stream.activation_generation == self.ledger.activation_generation
            and stream.capture_epoch == 7
            and self.engine.is_current_recognition_stream(stream)
        )


class EngineScope:
    def __init__(self, engine, ledger, evidence_view):
        self.engine = engine
        self.ledger = ledger
        self.speaker_evidence = evidence_view

    def is_current_recognition_stream(self, channel, stream):
        return (
            channel == "peer"
            and stream.activation_generation == self.ledger.activation_generation
            and stream.capture_epoch == 7
            and self.engine.is_current_recognition_stream(stream)
        )


class ObservedBatch(CausalBatchSession):
    def __init__(self, inner, trace, evidence_view):
        super().__init__(inner, evidence_view=evidence_view)
        self.observation_trace = trace

    def _snapshot_native_terminal(self, event):
        super()._snapshot_native_terminal(event)
        try:
            self._observe_native_terminal(event)
        except Exception as error:
            self.observation_trace.append(
                {
                    "event": "observation_exception",
                    "at": time.monotonic(),
                    "stage": "native_final_observer",
                    "exception_class": type(error).__name__,
                    "message": str(error),
                }
            )

    def _observe_native_terminal(self, event):
        child = self._slices.get(event.identity)
        if child is not None:
            self.observation_trace.append(
                {
                    "event": "own_native_final_attribution",
                    "at": time.monotonic(),
                    "identity": prior.serial(event.identity),
                    "source_ranges": prior.serial(child.audio.source_ranges),
                    "context_ranges": prior.serial(child.audio.context_ranges),
                    "proof_id": child.audio.speaker_proof_id,
                    "delivered_frame_count": len(self.evidence.frames),
                    "immutable": self.evidence.immutable,
                    "processed": self.evidence.processed,
                    "accepted_revision": self.evidence.revision,
                    "generation": self.evidence.generation,
                    "result": prior.serial(child.speaker),
                }
            )


def pipeline_status(record):
    if any(event["event"] == "observation_exception" for event in record["trace"]):
        return "invalid_observation"
    if any(event["event"] == "unhandled_task_exception" for event in record["trace"]):
        return "pipeline_task_failure"
    if any(
        event["event"] == "gpu_asr_diagnostic"
        and event["value"]["kind"] == "decode_attempt"
        and event["value"]["fields"].get("result") == "decode_failure"
        for event in record["trace"]
    ):
        return "pipeline_completed_with_native_failures"
    if record["errors"]:
        return "pipeline_or_cleanup_failure"
    if not record["sources"]:
        return "failed_content_empty"
    return "pipeline_completed"


def execution_inventory(args):
    names = sorted(args.cases.split(","), key=lambda value: value != "distinct_repeated_A_B_A")
    authorized = [
        {"route": args.route, "case": name, "profile": profile, "repetition": 1}
        for profile in ("1.04", "0.64")
        for name in names
    ]
    end = None if args.case_count is None else args.case_offset + args.case_count
    return authorized, authorized[args.case_offset : end]


class ChatboxGuard(local.ChatboxGuard):
    bind_delivery_observer = None


class ObservedDesktop(GuardedDesktop):
    assignment_observer = None

    async def replace_snapshot(self, snapshot, *, block_expirations=None):
        if self.assignment_observer is not None:
            self.assignment_observer("logical_snapshot_application")
        return await super().replace_snapshot(snapshot, block_expirations=block_expirations)


class RecordedSession(deepgram._DeepgramSDKSession):
    def __init__(self, projection, tape, record, playback_ready):
        super().__init__(
            api_key="",
            model="nova-3",
            language="en",
            sample_rate_hz=16000,
            connect_timeout_s=1,
            keyterms=[],
            stream_label="peer",
            drain_timeout_s=1.5,
            projection=projection,
        )
        self.tape = tape
        self.record = record
        self.playback_ready = playback_ready
        self.written_samples = 0
        self.control_index = 0
        self.invalid = None

    async def start(self):
        self._connected.set()
        self._startup_done.set()
        self.record["native_transport"] = (
            "Fixed original submitted input/control schedule; no provider network or fresh VAD qualification"
        )

    async def _write_payload(self, payload):
        if isinstance(payload, bytes):
            end = self.written_samples + len(payload) // 2
            expected = self.tape["pcm"][self.written_samples * 2 : end * 2]
            if payload != expected:
                self.invalid = "Recorded transport submitted PCM mismatch"
                raise ValueError(self.invalid)
            self.written_samples = end
            self.record["trace"].append(
                {
                    "event": "recorded_transport_write",
                    "at": time.monotonic(),
                    "cursor": end,
                    "samples": len(payload) // 2,
                    "pcm16_sha256": hashlib.sha256(payload).hexdigest(),
                    "matches_canonical_tape": True,
                }
            )
            return
        control = (
            "Finalize"
            if payload is deepgram._FINALIZE
            else "CloseStream" if payload is deepgram._CLOSE_STREAM else "other"
        )
        if control == "CloseStream" and self.control_index == len(self.tape["controls"]):
            return
        actual = (control, self.written_samples)
        expected = (
            self.tape["controls"][self.control_index]
            if self.control_index < len(self.tape["controls"])
            else None
        )
        if actual != expected:
            self.invalid = "Recorded transport control/cursor mismatch: " + repr((actual, expected))
            raise ValueError(self.invalid)
        self.control_index += 1
        self.record["trace"].append(
            {
                "event": "recorded_transport_control",
                "at": time.monotonic(),
                "control": control,
                "cursor": self.written_samples,
            }
        )


class RecordedBackend:
    provider_id = "deepgram"
    model_id = "nova-3"

    def __init__(self, tape, record, playback_ready):
        self.tape = tape
        self.record = record
        self.playback_ready = playback_ready
        self.sessions = []

    async def open_session(self, *, projection):
        if self.sessions:
            raise ValueError("Canonical recorded tape requires one native epoch")
        session = RecordedSession(projection, self.tape, self.record, self.playback_ready)
        self.sessions.append(session)
        await session.start()
        return session

    async def close(self):
        for session in self.sessions:
            await session.close()


class RecordedInputCoordinator:
    def __init__(self, backend, factory, ledger, peer, event_sink, values, record):
        self.backend = backend
        self.factory = factory
        self.ledger = ledger
        self.peer = peer
        self.event_sink = event_sink
        self.values = values
        self.record = record
        self.session = None
        self.consumer = None
        self.requests = {}
        self.normalizers = {}
        self.terminals = 0
        self.feed_done = False
        self.closed = False
        self.epoch = "recorded-" + os.urandom(16).hex()
        self.settings_scope = (
            ledger.settings.provider_id,
            ledger.settings.provider_signature,
            ledger.settings.runtime_signature,
        )
        self.payloads = iter(
            [
                (e["identity"], p)
                for e in backend.tape["case"]["trace"]
                if e["event"] == "native_submit"
                for p in e["payloads"]
            ]
        )
        self.raw_results = iter(backend.tape["results"])
        self.actions = [
            e
            for e in backend.tape["case"]["trace"]
            if e["event"] in ("vad", "native_begin", "native_submit", "raw_wire_result")
            or (e["event"] == "wire_write_begin" and e["samples"])
        ]
        self.action_index = 0

    @property
    def is_at_turn_boundary(self):
        return self.feed_done and self.terminals == len(self.requests)

    def is_current_recognition_stream(self, stream):
        return (
            not self.closed
            and stream.channel == "peer"
            and stream.activation_generation == self.ledger.activation_generation
            and stream.capture_epoch == 7
            and stream.provider_epoch_id == self.epoch
            and stream.settings_scope == self.settings_scope
        )

    async def start(self):
        self.session = await self.factory(self.ledger.settings, self.epoch)
        self.consumer = asyncio.create_task(self.consume())

    async def consume(self):
        async for event in self.session.turn_events():
            if isinstance(event, STTProviderTurnUpdate):
                event = self.normalizers[event.identity].apply_update(event)
            elif isinstance(event, STTProviderTurnTerminal):
                event = self.normalizers[event.identity].apply_terminal(event)
            if event is None:
                continue
            await self.event_sink(event)
            if isinstance(event, STTProviderTurnTerminal):
                self.terminals += 1

    async def observe_speaker_evidence(self, items):
        if self.session is not None and callable(
            getattr(self.session, "observe_speaker_evidence", None)
        ):
            await self.session.observe_speaker_evidence(items)

    async def wait_for_event_ingress_drain(self):
        while not self.is_at_turn_boundary:
            if self.consumer.done():
                await self.consumer
                raise RuntimeError("Recorded current-source consumer ended before all terminals")
            await asyncio.sleep(0.01)

    async def close_backend(self):
        self.closed = True
        if self.session:
            await self.session.close()
        await self.backend.close()
        if self.consumer:
            await self.consumer

    def spans(self, values):
        delta = (
            self.record["playback_started_at"] - self.backend.tape["case"]["playback_started_at"]
        )
        return tuple(
            prior.AudioCaptureSpan(
                **{
                    **s,
                    "source_start_monotonic_s": s["source_start_monotonic_s"] + delta,
                    "source_end_monotonic_s": s["source_end_monotonic_s"] + delta,
                }
            )
            for s in values
        )

    def samples(self, spans):
        return (
            local.np.concatenate(
                [self.values[s.normalized_start_sample : s.normalized_end_sample] for s in spans]
            )
            if spans
            else local.np.zeros(0, local.np.float32)
        )

    async def apply(self, event):
        old_origin = self.backend.tape["case"]["playback_started_at"]
        kind = event["event"]
        due = self.record["playback_started_at"] + event["at"] - old_origin
        await asyncio.sleep(max(0, due - time.monotonic()))
        if kind == "vad":
            value = event["value"]["event"]
            identity = UUID(value["utterance_id"])
            if "pre_roll" in value:
                pre = self.spans(value["pre_roll_capture"])
                chunk = self.spans(value["chunk_capture"])
                vad = prior.SpeechStart(
                    identity,
                    self.samples(pre),
                    self.samples(chunk),
                    pre,
                    chunk,
                    value["genuine_onset"],
                )
            else:
                vad = prior.SpeechEnd(identity, value["trailing_silence_ms"], value["reason"])
            owned = self.ledger.observe_vad_event(vad, now_monotonic_s=due)
            self.record["trace"].append(
                {
                    "event": "vad",
                    "at": time.monotonic(),
                    "historical_delay_s": event["at"] - old_origin,
                    "value": prior.serial(owned),
                    "fixed_recorded_input": True,
                }
            )
            self.peer._record_peer_owned_vad_event(owned)
        elif kind == "native_begin":
            value = event["request"]
            segment = value["identity"]["segment"]
            segment = AudioSegmentIdentity(
                segment["activation_generation"],
                segment["segment_order"],
                UUID(segment["segment_id"]),
                segment["capture_epoch"],
            )
            identity = STTProviderTurnIdentity(
                segment, self.epoch, value["identity"]["provider_turn_id"], self.settings_scope
            )
            self.requests[value["identity"]["provider_turn_id"]] = identity
            self.normalizers[identity] = STTScopedTurnNormalizer(identity)
            cutoff = value.get("optional_cutoff_at_monotonic_s")
            request = STTProviderTurnRequest(
                identity,
                self.ledger.settings,
                "peer",
                UUID(value["display_group_id"]) if value["display_group_id"] else None,
                value["isolate_failure"],
                value["native_options"],
                (
                    None
                    if cutoff is None
                    else cutoff + self.record["playback_started_at"] - old_origin
                ),
            )
            await self.session.begin_turn(request)
        elif kind == "wire_write_begin":
            original_identity, payload = next(self.payloads)
            spans = self.spans(payload["source_ranges"])
            required = max(s.normalized_end_sample for s in spans)
            if required > self.record["capture_input_samples"]:
                raise ValueError(
                    f'Original submission requires unavailable future capture: {required}>{self.record["capture_input_samples"]}'
                )
            if max(s.source_end_monotonic_s for s in spans) > due + 0.000001:
                raise ValueError(
                    "Original native submission predates its mapped capture availability"
                )
            identity = self.requests[original_identity["provider_turn_id"]]
            if not payload["context_only"] and payload["payload_sequence"] != 2:
                self.ledger.observe_vad_event(
                    SpeechChunk(identity.segment.segment_id, self.samples(spans), spans),
                    now_monotonic_s=due,
                )
            data = float32_to_pcm16le_bytes(self.samples(spans))
            if (
                len(data) // 2 != event["samples"]
                or hashlib.sha256(data).hexdigest() != payload["pcm16_sha256"]
            ):
                raise ValueError("Original successful write schedule/payload mismatch")
            self.record["trace"].append(
                {
                    "event": "original_capture_availability",
                    "at": time.monotonic(),
                    "submitted_end_sample": required,
                    "available_capture_samples": self.record["capture_input_samples"],
                    "historical_delay_s": event["at"] - old_origin,
                }
            )
            await self.session.send_turn_audio(
                identity,
                data,
                payload_sequence=payload["payload_sequence"],
                source_ranges=spans,
                context_only=payload["context_only"],
            )
        elif kind == "native_submit":
            identity = self.requests[event["identity"]["provider_turn_id"]]
            await self.session.seal_turn(
                identity,
                sealed_content_ranges=self.spans(event["sealed_content_ranges"]),
                seal_reason=event["seal_reason"],
                observed_trailing_silence_ms=event["observed_trailing_silence_ms"],
            )
        else:
            item = next(self.raw_results)
            native = self.backend.sessions[0]
            if native._sent_samples != item["cursor"]:
                raise ValueError(
                    f'Original receipt cursor mismatch: {native._sent_samples}!={item["cursor"]}'
                )
            self.record["trace"].append(
                {
                    "event": "recorded_raw_result",
                    "at": time.monotonic(),
                    "historical_delay_s": item["delay"],
                    "historical_cursor": item["cursor"],
                    "actual_cursor": native._sent_samples,
                    "value": item["value"],
                    "fixed_recorded_input": True,
                }
            )
            native._build_transcript_event(namespaces(item["value"]))

    async def advance(self, until):
        origin = self.backend.tape["case"]["playback_started_at"]
        while self.action_index < len(self.actions):
            event = self.actions[self.action_index]
            due = self.record["playback_started_at"] + event["at"] - origin
            if due > until:
                return
            await self.apply(event)
            self.action_index += 1
        self.feed_done = True


async def case_run(
    args, name, values, mapping, translation, desktop, mode, profile, repetition, tape=None
):
    record = {
        "case": name,
        "mode": mode,
        "profile": profile if mode == "new" else None,
        "route": args.route,
        "repetition": repetition,
        "status": "running",
        "input_samples": values.size,
        "input_pcm_f32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "assembly_scoring_only": mapping,
        "sources": [],
        "trace": [],
        "evidence": [],
        "resources": [],
        "errors": [],
        "frame_delivery_lags": [],
        "evidence_poll_seconds": 0.05,
    }
    trace = record["trace"]
    playback_ready = asyncio.Event()
    gpu = None
    if args.route == "gpu":
        gpu = local.SharedGpuASRRuntime(
            process_factory=local.DefaultGpuWorkerProcessFactory(executable_path=args.gpu_worker),
            diagnostic_sink=lambda d: trace.append(
                {"event": "gpu_asr_diagnostic", "at": time.monotonic(), "value": prior.serial(d)}
            ),
        )
        devices = await gpu.discover_devices()
        record["gpu_devices"] = prior.serial(devices)
        device = next(d for d in devices if d.registry_index == 0)
        backend = local.LocalGpuSTTBackend(
            runtime=gpu,
            channel="peer",
            model_path=args.gpu_model,
            model_id="qwen3-asr-1.7b-q6-k-transcribe-vulkan",
            device_id=device.device_id,
            language_hint="en",
        )
        await backend.prepare()
        record["gpu_activation"] = prior.serial(gpu._activation)
    else:
        backend = RecordedBackend(tape, record, playback_ready)
    provider_id = "local_gpu" if args.route == "gpu" else backend.provider_id
    settings = prior.AudioSegmentSettingsSnapshot(
        provider_id,
        (provider_id, backend.model_id),
        ("issue264_validation",),
        "manual",
        "en",
        ("en",),
        16000,
        0.3,
        600,
        500,
    )
    ledger = prior.PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    config = prior.TranslationRuntimeConfigSnapshot(
        0,
        prior.TranslationRuntimeConfig(
            source_language="en",
            target_language="ko",
            peer_source_language="en",
            peer_target_language="ko",
            peer_translation_enabled=True,
            concurrency_limit=3,
            integrated_context_enabled=True,
            system_prompt=local.resolve_system_prompt(""),
        ),
    )
    clock = prior.SystemClock()
    evidence_view = SpeakerEvidenceView()
    presenter = local.OverlayPresenter(
        calibration=local.OverlayCalibration(), bridge=desktop, translation_enabled=True
    )
    output = local.OutputRuntime(
        chatbox=ChatboxGuard(),
        clock=clock,
        overlay_sink=presenter,
        routing_observer=lambda d: trace.append(
            {"event": "routing_decision", "at": time.monotonic(), "value": prior.serial(d)}
        ),
        activation_notice_enabled=False,
    )
    output.activate_peer_generation(1)
    diagnostics = local.TranslationLatencyDiagnosticsOwner(
        clock=clock, config_snapshot=lambda: config
    )
    projection = local.TranslationOutputProjectionOwner(
        output_runtime=output,
        ui_messages=local.TranslationUiMessageQueue(asyncio.Queue(), output),
        diagnostics=diagnostics,
        clock=clock,
    )
    peer_runtime = prior.ChannelRuntime("peer")
    requests = prior.TranslationRequestOwner(
        config_snapshot=lambda: config,
        self_runtime=prior.ChannelRuntime("self"),
        peer_runtime=peer_runtime,
        context_resolver=prior.ContextResolver(clock=clock, config_snapshot=lambda: config),
        provider_runtime=local.ProviderRuntimeHandle(name="llm", provider=translation),
        diagnostics=diagnostics,
        presentation=projection,
        clock=clock,
    )

    async def call(method, *a):
        return await getattr(peer, method)(*a)

    turns = prior.TranslationTurnLifecycleOwner(
        on_child_created=lambda *a: call("on_child_created", *a),
        on_child_started=lambda *a: call("on_child_started", *a),
        process_child=lambda *a: call("process_child", *a),
        on_child_terminal=lambda *a: call("on_child_terminal", *a),
        on_parent_closed=lambda *a: call("on_parent_closed", *a),
        on_parent_rejected=lambda *a: call("on_parent_rejected", *a),
        on_parent_admitted=lambda *a: call("on_parent_admitted", *a),
        output=SimpleNamespace(
            submit_translation_output=lambda *a: call("submit_translation_output", *a)
        ),
        config_snapshot=lambda: config,
        predecessor_wait_observer=diagnostics.record_translation_wait,
        turn_generation_observer=projection.retire_turn_generation,
    )
    peer = local.ObservedPeer(
        runtime=peer_runtime,
        config_snapshot=lambda: config,
        translation_turns=turns,
        local_asr_runtime=SimpleNamespace(speaker_evidence=evidence_view),
        translation_requests=requests,
        output_projection=projection,
        diagnostics=diagnostics,
        clock=clock,
    )
    peer.record = record

    def observe_assignments(reason):
        identities = peer._speaker_identities
        trace.append(
            {
                "event": "presentation_assignments",
                "at": time.monotonic(),
                "reason": reason,
                "records": {
                    item["source"]["source_id"]: prior.serial(snapshot)
                    for item in record["sources"]
                    if (snapshot := identities.source_snapshot(UUID(item["source"]["source_id"])))
                    is not None
                },
                "evidence_revision": evidence_view.revision,
                "evidence_generation": evidence_view.generation,
                "retained_frames": len(evidence_view.frames),
                "retained_proofs": len(evidence_view._proofs),
            }
        )

    desktop.assignment_observer = observe_assignments
    callbacks = TranslationChannelOwnerCallbacks(
        SimpleNamespace(
            record=lambda event: trace.append(
                {"event": "session_state", "at": time.monotonic(), "value": prior.serial(event)}
            )
        )
    )
    callbacks.bind_peer(peer)

    async def event_sink(event):
        trace.append(
            {"event": "provider_result", "at": time.monotonic(), "value": prior.serial(event)}
        )
        await callbacks.peer_event_handler(event)

    sessions = []

    async def factory(_settings, epoch):
        inner = prior.SessionTrace(
            await backend.open_session(projection=prior.STTSessionProjection("scoped", epoch)),
            trace,
            values,
        )
        session = (
            ObservedBatch(inner, trace, evidence_view)
            if mode == "new" and args.route == "gpu"
            else (
                StreamingSpeakerSession(inner, evidence_view=evidence_view)
                if mode == "new"
                else inner
            )
        )
        sessions.append(session)
        return session

    engine = (
        prior.ScopedRecognitionEngine(
            factory,
            channel="peer",
            event_sink=event_sink,
            backend_close=backend.close,
            diagnostic_sink=lambda e: trace.append(
                {"event": "engine_diagnostic", "at": time.monotonic(), "value": prior.serial(e)}
            ),
        )
        if args.route == "gpu"
        else RecordedInputCoordinator(backend, factory, ledger, peer, event_sink, values, record)
    )
    peer.local_asr_runtime = EngineScope(engine, ledger, evidence_view)
    callbacks.bind_peer_capture(CaptureReceipts(ledger, clock, engine))
    speaker = nemotron = None
    finished = asyncio.Event()
    pump = sampler = None
    received = hashlib.sha256()
    record["capture_input_samples"] = 0
    handler = prior.TraceHandler(trace)
    logger = logging.getLogger("puripuly_heart.core.stt.batch_speaker")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    async def evidence_pump():
        while not finished.is_set():
            items = speaker.drain_evidence(2048)
            trace.append({"event": "evidence_poll", "at": time.monotonic(), "count": len(items)})
            if items:
                record["evidence"].extend(prior.serial(item) for item in items)
                evidence_view.observe(items)
                await engine.observe_speaker_evidence(items)
                observe_assignments("accepted_evidence")
            await asyncio.sleep(0.05)

    class Sink:
        async def handle_stream_input(self, event):
            record["capture_input_samples"] += event.chunk.size
            trace.append(
                {
                    "event": "capture_input_observed",
                    "at": time.monotonic(),
                    "capture": prior.serial(event.capture),
                    "speech_observed": event.speech_observed,
                    "boundary_reason": event.boundary_reason,
                }
            )
            received.update(event.chunk.tobytes())
            if speaker:
                speaker.observe_input("issue264_validation:" + name, event)

        async def handle_owned_vad_event(self, owned):
            if args.route == "gpu":
                if isinstance(owned.event, (prior.SpeechStart, prior.SpeechEnd)):
                    trace.append(
                        {"event": "vad", "at": time.monotonic(), "value": prior.serial(owned)}
                    )
                    peer._record_peer_owned_vad_event(owned)
                await engine.handle_owned_vad_event(owned)
            elif isinstance(owned.event, (prior.SpeechStart, prior.SpeechEnd)):
                trace.append(
                    {
                        "event": "current_capture_vad",
                        "at": time.monotonic(),
                        "value": prior.serial(owned),
                    }
                )

        async def observe_source_activity(self, **kwargs):
            if args.route == "gpu":
                await engine.observe_source_activity(**kwargs)

    class Source:
        terminal_reason = "closed"

        async def frames(self):
            for sequence, offset in enumerate(range(0, values.size, 512)):
                end = min(values.size, offset + 512)
                due = started + end / 16000
                if args.route != "gpu":
                    await engine.advance(due)
                await asyncio.sleep(max(0, due - time.monotonic()))
                record["frame_delivery_lags"].append(time.monotonic() - due)
                yield prior.AudioFrameF32(
                    values[offset:end],
                    16000,
                    capture=prior.AudioCaptureSpan(
                        7, sequence, 16000, offset, end, started + offset / 16000, due
                    ),
                )
            if args.route != "gpu":
                await engine.advance(float("inf"))

    starts = (len(desktop.events), len(desktop.snapshots), len(desktop.readbacks))
    try:
        if mode == "new":
            nemotron = prior.NemotronBackend(
                artifact_path=args.model, runtime_dir=args.runtime, device=0, profile=profile
            )
            speaker = prior.SpeakerRuntime(nemotron)
            await speaker.start()
            if not speaker.snapshot().ready:
                raise RuntimeError(speaker.snapshot().error)
            record["speaker_initial"] = prior.serial(speaker.snapshot())
            pump = asyncio.create_task(evidence_pump())
        if args.route == "gpu":
            await engine._ensure_session(
                settings, STTRecognitionWatchdogs(), engine._authority_generation
            )
        else:
            await engine.start()
        record["translation_config"] = prior.serial(config)
        record["asr"] = {
            k: prior.serial(getattr(backend, k, None))
            for k in (
                "model_id",
                "provider_id",
                "language_hint",
                "hotwords",
                "source_mode",
                "active_decode_timeout_s",
            )
        }
        vad = prior.create_peer_vad_gating(
            engine=prior.SileroVadOnnx(prior.bundled_silero_vad_onnx_path()),
            sample_rate_hz=16000,
            ring_buffer_ms=500,
            speech_threshold=0.3,
            hangover_ms=600,
        )
        record["vad"] = {
            "sha256": prior.digest(prior.bundled_silero_vad_onnx_path()),
            "speech_threshold": 0.3,
            "hangover_ms": 600,
            "onset_pre_roll_ms": 500,
            "hard_rollover_pre_roll_ms": 300,
            "chunk_samples": vad.chunk_samples,
            "trimming": False,
            "synthetic_native_tail": False,
        }
        started = time.monotonic()
        record["playback_started_at"] = started
        playback_ready.set()
        sampler = asyncio.create_task(local.resources(record, finished, speaker, gpu, turns))
        if args.route == "gpu":
            await prior.run_audio_vad_loop(
                source=Source(),
                vad=vad,
                sink=Sink(),
                target_sample_rate_hz=16000,
                channel_label="peer",
                segment_ledger=ledger,
            )
        else:
            capture_ledger = prior.PeerAudioSegmentLedger(
                activation_generation=1, settings=settings
            )
            await prior.run_audio_vad_loop(
                source=Source(),
                vad=vad,
                sink=Sink(),
                target_sample_rate_hz=16000,
                channel_label="peer",
                segment_ledger=capture_ledger,
            )
        deadline = time.monotonic() + 60
        while not engine.is_at_turn_boundary or (speaker and speaker.snapshot().pending_items):
            if isinstance(backend, RecordedBackend) and any(s.invalid for s in backend.sessions):
                raise ValueError(next(s.invalid for s in backend.sessions if s.invalid))
            if isinstance(engine, RecordedInputCoordinator) and engine.consumer.done():
                await engine.consumer
                raise RuntimeError("Recorded source consumer ended before terminal drain")
            if time.monotonic() > deadline:
                raise TimeoutError("Actual ASR/speaker drain timeout")
            await asyncio.sleep(0.01)
        await engine.wait_for_event_ingress_drain()
        await asyncio.wait_for(turns.wait_for_idle(), 90)
        if isinstance(backend, RecordedBackend):
            for session in backend.sessions:
                if session.invalid:
                    raise ValueError(session.invalid)
                if session.written_samples != len(tape["pcm"]) // 2 or session.control_index != len(
                    tape["controls"]
                ):
                    raise ValueError("Recorded transport incomplete input/control coverage")
            record["recorded_native"] = {
                "canonical_case": name,
                "historical_native_results": len(tape["results"]),
                "submitted_samples": len(tape["pcm"]) // 2,
                "submitted_pcm16_sha256": hashlib.sha256(tape["pcm"]).hexdigest(),
                "controls": tape["controls"],
                "verified": True,
            }
        await asyncio.sleep(0.1)
        assignment_deadline = peer._speaker_identities.next_deadline()
        if assignment_deadline is not None:
            trace.append(
                {
                    "event": "post_completion_observation_only",
                    "at": time.monotonic(),
                    "until": assignment_deadline,
                    "limit": "No source/translation/display wait or extra input; observe pending existing one-shot presentation deadline",
                }
            )
            await asyncio.sleep(max(0.0, assignment_deadline + 0.05 - time.monotonic()))
        observe_assignments("completed_output_observation")
        record["status"] = "pipeline_completed"
        record["wall_seconds"] = time.monotonic() - started
        record["snapshot_capture"] = await desktop.picture(
            f"{args.route}_{mode}_{profile}_{repetition}_{name}"
        )
    except Exception as error:
        record["status"] = (
            "invalid_replay" if isinstance(backend, RecordedBackend) else "failed_execution"
        )
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
    finally:
        finished.set()
        if pump:
            results = await asyncio.gather(pump, return_exceptions=True)
            record["errors"].extend(str(e) for e in results if isinstance(e, BaseException))
        if sampler:
            await sampler
        record["speaker_final"] = prior.serial(speaker.snapshot()) if speaker else None
        observe_assignments("before_resource_retirement")
        record["presentation_resolution_counts"] = {
            "readable_unresolved": peer._speaker_identities.readable_unresolved,
            "within_500ms": peer._speaker_identities.resolved_within_500ms,
            "within_1s": peer._speaker_identities.resolved_within_1s,
            "resolution_delays": list(peer._speaker_identities.resolution_delays),
        }
        record["segments"] = prior.serial(ledger.snapshots)
        record["terminal_receipts"] = prior.serial(ledger.terminal_receipts)
        record["capture_received_once"] = (
            record["capture_input_samples"] == values.size
            and received.hexdigest() == record["input_pcm_f32_sha256"]
        )
        for label, operation in [
            ("engine", engine.close_backend),
            ("turns", turns.close),
            ("output", output.close),
            ("presenter", presenter.close),
        ]:
            try:
                await operation()
            except Exception as error:
                record["errors"].append({"cleanup": label, "error": str(error)})
        if speaker:
            await speaker.stop()
            record["nemotron_stderr"] = bytes(nemotron._stderr).decode(errors="replace")
        if gpu:
            await gpu.close()
        logger.removeHandler(handler)
        if desktop.readback_tasks:
            await asyncio.gather(*desktop.readback_tasks)
        record["renderer_events"] = desktop.events[starts[0] :]
        record["renderer_snapshots"] = desktop.snapshots[starts[1] :]
        record["renderer_readbacks"] = desktop.readbacks[starts[2] :]
        await desktop.replace_snapshot(local.OverlayPresentationSnapshot(revision=0, blocks=[]))
        desktop.assignment_observer = None
    record["observation_errors"] = [
        event for event in trace if event["event"] == "observation_exception"
    ]
    if record["status"] == "pipeline_completed":
        record["status"] = pipeline_status(record)
    record["scenario_quality_disposition"] = (
        "failed_native_execution"
        if record["status"] == "pipeline_completed_with_native_failures"
        else "not_automatically_assessed"
    )
    return record


def source_manifest():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "native/nemotron/issue218_policy_summary.py",
        ROOT / "native/nemotron/issue218_experiment.py",
        ROOT / "native/nemotron/issue215_smoke.py",
        ROOT / "native/nemotron/issue216_presentation.py",
        ROOT / "native/nemotron/probe.py",
        ROOT / "native/nemotron/issue218_counters.ps1",
    ]
    paths += [ROOT / "native/nemotron/issue218_policy_visibility.py"]
    paths += [ROOT / "native/nemotron/issue264_validation_launch.py"]
    paths += [
        ROOT / "native/nemotron/issue264_validation_caption.py",
        ROOT / "native/nemotron/issue264_validation_summary.py",
        ROOT / "tests/core/test_issue264_shared_speaker.py",
        ROOT / "tests/core/test_causal_batch_speaker.py",
        ROOT / "tests/helpers/overlay_source_cards.py",
    ]
    return {p.relative_to(ROOT).as_posix(): prior.digest(p) for p in paths}


async def gemma_start(args, report):
    captures = []
    messages = []
    validations = []

    async def existing_install(
        *, install_dir, spec, downloader=None, cancel_event=None, on_status=None, locale=None
    ):
        if downloader is not None:
            raise RuntimeError("Validation permits only the existing pinned Gemma install")

        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise GemmaProvisioningCancelled("Read-only Gemma validation cancelled")

        manifest = await asyncio.to_thread(
            validate_gemma_install,
            install_dir,
            spec=spec,
            verify_checksums=True,
            check_cancelled=check_cancelled,
        )
        validations.append(
            {
                "at": time.monotonic(),
                "manifest": prior.serial(manifest),
                "pinned_spec": prior.serial(spec),
                "checksums_verified": True,
            }
        )
        return manifest

    settings = local._settings("gpu")
    runtime = local.ManagedGemmaRuntimeOwner(
        install_dir=args.gemma_model,
        runtime_paths=local.default_gemma_runtime_paths(args.gemma_runtime),
        provisioner=existing_install,
        process_factory=local._captured_process_factory(captures),
        transport_factory=local.HttpxManagedGemmaTransport,
        log_sink=lambda message, level: messages.append(
            {
                "at": time.monotonic(),
                "level": level,
                "message": message,
            }
        ),
    )
    owner = local.ManagedGemmaTranslationOwner(runtime=runtime)
    try:
        activation = await owner.prepare(local.managed_gemma_selection(settings))
        backend = local.create_translation_backend(
            translation_model="managed_gemma",
            secrets=local.InMemorySecretStore(),
            http_extensions=local.HttpExtensionRegistry(args.output / "unused-http-extensions"),
            runtime_input=local.runtime_resolution_input_from_vnext(settings),
            extras=local.llm_factory_extras_from_vnext(settings),
            managed_gemma_runtime=activation.runtime,
            managed_gemma_release=activation.release,
        )
        if not local._is_managed_only_backend(backend):
            await backend.close()
            raise RuntimeError("Nonlocal translation backend is prohibited")
    except BaseException:
        await owner.close()
        for capture in captures:
            await capture.scope.close()
        raise
    report["gemma"] = {
        "requested_backend": "gpu",
        "readiness": prior.serial(activation.readiness),
        "selection": prior.serial(local.managed_gemma_selection(settings)),
        "messages": messages,
        "runtime_commands": [list(c.command) for c in captures],
        "existing_install_validations": validations,
        "controlled_boundary": "Authorized read-only existing-install checksum validator replaces acquisition/provisioning only; no shared lease, download, install or persistence qualification",
        "prefix_disk_cache": "disabled",
        "warmup": "actual managed owner prepare/prefix prefill; no additional request warmup",
        "prompt_cache": "production managed runtime kept resident throughout both full-profile smoke arms",
    }
    return backend, owner, captures


def preflight(args):
    if OVERLAY_CONTRACT_VERSION != 16 or OVERLAY_SPEAKER_IDENTITY_CONTRACT != {
        "version": 3,
        "policy": "bounded_late_speaker_assignment",
    }:
        raise RuntimeError("Pinned v3 bounded-late-assignment/wire16 renderer contract required")
    if args.evidence_poll_ms != 50 or args.repetitions != 1:
        raise ValueError("Only approved 50ms and bounded repetitions are supported")
    expected = CASES if args.route == "gpu" else ("distinct_repeated_A_B_A", "short_A_B_A")
    if tuple(args.cases.split(",")) != expected:
        raise ValueError("Only approved reused case matrix is supported")
    if args.case_offset < 0 or (args.case_count is not None and args.case_count < 1):
        raise ValueError("Case slicing must retain at least one approved case")
    authorized, requested = execution_inventory(args)
    if not requested:
        raise ValueError("Requested inventory is empty")
    provenance, cases = prior.input_cases(args)
    selected_tapes, replay_info = tapes(args, cases) if args.route != "gpu" else ({}, None)
    for path in (
        args.model,
        args.runtime / "runtime-manifest.json",
        args.gpu_worker,
        args.gpu_model,
        args.gemma_runtime / "vulkan/llama-server.exe",
        args.gemma_model / "model.gguf",
    ):
        if not path.is_file():
            if path == args.gemma_model / "model.gguf" and any(args.gemma_model.glob("*.gguf")):
                continue
            raise FileNotFoundError(path)
    report = {
        "schema": "issue264-changed-path-validation-1",
        "requested_observations": requested,
        "authorized_full_smoke_inventory": authorized,
        "candidate": args.candidate,
        "baseline": "cc75792e9704e8609f38a3aa0794c03a1c7e3416",
        "route": args.route,
        "command": sys.argv,
        "driver_sha256": prior.digest(Path(__file__)),
        "sources_sha256": source_manifest(),
        "speaker_identity_contract": OVERLAY_SPEAKER_IDENTITY_CONTRACT,
        "overlay_wire_version": OVERLAY_CONTRACT_VERSION,
        "inputs": provenance,
        "input_hashes": {
            name: hashlib.sha256(cases[name][0].tobytes()).hexdigest() for name in expected
        },
        "nemotron_artifact": prior.inspect_artifact(args.model),
        "nemotron_runtime": prior.validate_runtime(args.runtime),
        "runtime_manifest_sha256": prior.digest(args.runtime / "runtime-manifest.json"),
        "gpu_model_sha256": prior.digest(args.gpu_model),
        "gpu_worker_sha256": prior.digest(args.gpu_worker),
        "gemma_models_sha256": {p.name: prior.digest(p) for p in args.gemma_model.glob("*.gguf")},
        "gemma_server_sha256": prior.digest(args.gemma_runtime / "vulkan/llama-server.exe"),
        "dependencies": {
            k: importlib.metadata.version(k)
            for k in (
                "numpy",
                "onnxruntime",
                "sherpa-onnx",
                "sherpa-onnx-core",
                "httpx",
                "flet",
                "psutil",
                "deepgram-sdk",
            )
        },
        "policy_constants": {
            k: v
            for k, v in vars(speaker_attribution).items()
            if k.isupper() and isinstance(v, (int, float, str, bool))
        },
        "batch_bounds": {
            k: v
            for k, v in vars(CausalBatchSession).items()
            if k.isupper() and isinstance(v, (int, float, str, bool))
        },
        "evidence_poll_ms": 50,
        "replay": replay_info,
        "privacy": {
            "cloud_calls": 0,
            "ambient_capture": False,
            "settings_read_write": False,
            "VRChat_SteamVR": False,
            "credentials_read": False,
        },
        "cases": [],
        "status": "preflight_passed",
    }
    return report, cases, selected_tapes


async def run(args):
    if os.environ.get("ISSUE264_VALIDATION_READY_SHA") != args.candidate:
        raise RuntimeError("Use the isolated launcher after Director stable-candidate readiness")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GGML_VK_DISABLE_GRAPH_OPTIMIZE"] = "1"
    os.environ["NEMO_SPEECH_SCHEDULE_CAPTURE"] = "1"
    for name in tuple(os.environ):
        if re.search(
            r"(?:API_KEY|TOKEN|SECRET|CREDENTIAL|AUTHORIZATION)", name, re.I
        ) or name.startswith(("GOOGLE_", "GEMINI_")):
            os.environ.pop(name, None)
    report, cases, selected_tapes = preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output / "execution_sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in report["sources_sha256"]:
            archive.write(ROOT / relative, relative)
    save(args.output / "preflight.json", report)
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "route": args.route,
                    "driver_sha256": report["driver_sha256"],
                    "input_hashes": report["input_hashes"],
                    "replay": report["replay"],
                }
            ),
            flush=True,
        )
        return 0
    if not re.fullmatch("[0-9a-f]{40}", args.candidate):
        raise ValueError("Acoustic execution requires Director-pinned full product SHA")
    desktop = ObservedDesktop(args.output)
    translation = owner = None
    captures = []
    counter = await asyncio.create_subprocess_exec(
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "native/nemotron/issue218_counters.ps1"),
        "-Samples",
        "3600",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    counter_output = asyncio.create_task(counter.communicate())
    report["counter_clock_anchor"] = {"monotonic": time.monotonic(), "unix_seconds": time.time()}
    try:
        translation, owner, captures = await gemma_start(args, report)
        if report["gemma"]["readiness"].get("effective_backend") != "vulkan":
            raise RuntimeError("Managed Gemma effective Vulkan required; CPU fallback prohibited")
        await desktop.start()
        report["desktop"] = {
            "renderer": type(desktop.renderer).__name__,
            "window": type(desktop.renderer.window).__name__,
            "pid": desktop.pid,
            "boundary": "Actual Flet dispatch ACK and owned timestamped PNG readback; no pixel onset, photon or VR claim",
        }
        arms = [("new", "1.04"), ("new", "0.64")]
        for repetition in range(args.repetitions):
            offset = repetition % 3
            for mode, profile in arms[offset:] + arms[:offset]:
                for name in sorted(
                    args.cases.split(","), key=lambda value: value != "distinct_repeated_A_B_A"
                ):
                    case_index = arms.index((mode, profile)) * len(args.cases.split(",")) + sorted(
                        args.cases.split(","), key=lambda value: value != "distinct_repeated_A_B_A"
                    ).index(name)
                    if case_index < args.case_offset or (
                        args.case_count is not None
                        and case_index >= args.case_offset + args.case_count
                    ):
                        continue
                    values, mapping = cases[name]
                    record = await case_run(
                        args,
                        name,
                        values,
                        mapping,
                        translation,
                        desktop,
                        mode,
                        profile,
                        repetition + 1,
                        selected_tapes.get(name),
                    )
                    report["cases"].append(record)
                    save(args.output / "trace.json", report)
                    print(
                        json.dumps(
                            {
                                "route": args.route,
                                "arm": mode + ":" + profile,
                                "repetition": repetition + 1,
                                "case": name,
                                "status": record["status"],
                                "sources": [s["source"]["raw_text"] for s in record["sources"]],
                                "translation_calls": sum(
                                    e["event"] == "translation_start" for e in record["trace"]
                                ),
                                "error": record.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if record["status"] != "pipeline_completed":
                        raise RuntimeError(
                            "Stopped on explicit pipeline/native/observation failure; original records retained"
                        )
        report["status"] = "requested_pipeline_completed"
    except Exception as error:
        report["status"] = "failed_execution"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        await desktop.close()
        if translation:
            await translation.close()
        if owner:
            await owner.close()
        for capture in captures:
            await capture.scope.close()
        report["gemma_runtime_output"] = [c.output for c in captures]
        report["desktop_errors"] = desktop.errors
        if counter.returncode is None:
            counter.terminate()
        stdout, stderr = await counter_output
        (args.output / "gpu_counters.jsonl").write_bytes(stdout)
        report["gpu_counter_observation"] = {
            "file": "gpu_counters.jsonl",
            "sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr": stderr.decode(errors="replace"),
            "returncode": counter.returncode,
            "scope": "Windows sampled adapter/process counters, not inference timestamp queries or isolated whole-host VRAM",
        }
        save(args.output / "trace.json", report)
        authorized, requested = execution_inventory(args)
        executed = [
            {key: case[key] for key in ("route", "case", "profile", "repetition")}
            for case in report["cases"]
        ]
        report["execution_inventory"] = {
            "authorized": authorized,
            "requested": requested,
            "executed": executed,
            "requested_inventory_complete": executed == requested,
            "complete_authorized_smoke_executed": executed == authorized,
        }
        report["scenario_quality_disposition"] = "not_inferred_from_pipeline_completion"
        save(args.output / "trace.json", report)
    return 0 if report["status"] == "requested_pipeline_completed" else 1


def arguments():
    parser = argparse.ArgumentParser()
    home = Path.home()
    parser.add_argument("--route", choices=("gpu", "deepgram-native-replay"), required=True)
    parser.add_argument("--candidate", default="preflight-unpinned")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--evidence-poll-ms", type=int, default=50)
    parser.add_argument(
        "--model",
        type=Path,
        default=home / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf",
    )
    parser.add_argument("--runtime", type=Path, default=ROOT / "build/nemotron")
    parser.add_argument(
        "--gpu-model",
        type=Path,
        default=ROOT.parent
        / "issue/.agents/goals/goal-issue-20-local-asr-backends-001/evidence/gpu-production/model-root/qwen3-asr-1.7b-q6-k-transcribe-vulkan/Qwen3-ASR-1.7B-Q6_K.gguf",
    )
    parser.add_argument(
        "--gpu-worker",
        type=Path,
        default=home / "Documents/dev/puripuly_heart/build/gpu_worker/PuriPulyHeartGpuWorker.exe",
    )
    parser.add_argument(
        "--gemma-model",
        type=Path,
        default=home / "AppData/Local/puripuly-heart/models/gemma-4-e4b-it-qat-ud-q4-k-xl",
    )
    parser.add_argument(
        "--gemma-runtime",
        type=Path,
        default=home / "Documents/dev/puripuly_heart/build/llama.cpp/llama.cpp-b11537",
    )
    parser.add_argument(
        "--corpus", type=Path, default=home / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1"
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=ROOT.parent
        / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification",
    )
    parser.add_argument(
        "--native-archive",
        type=Path,
        default=ROOT / "native/nemotron/issue218_cloud_deepgram_raw.zip",
    )
    parser.add_argument("--native-member", default="valid/trace.json")
    parser.add_argument("--cases")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.cases = args.cases or ",".join(
        CASES if args.route == "gpu" else ("distinct_repeated_A_B_A", "short_A_B_A")
    )
    args.repetitions = args.repetitions or 1
    args.output = args.output.resolve()
    if args.output.parent != ROOT / "build" or not args.output.name.startswith(
        "issue264-validation-"
    ):
        raise ValueError("Output must remain assigned build/issue264-validation-* resources")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(arguments())))
