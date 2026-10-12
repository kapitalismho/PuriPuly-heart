"""Authorized issue216 production-adapter smoke; no ambient capture or settings writes."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import dataclasses
from contextlib import contextmanager
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from issue215_smoke import (
    AudioCaptureSpan, AudioFrameF32, AudioSegmentSettingsSnapshot,
    ChannelRuntime, ContextResolver, NemotronBackend, PeerAudioSegmentLedger,
    PeerTranslationChannelOwner, SessionTrace, SpeakerRuntime, SpeechEnd, SpeechStart,
    STTContributionConsumptionLedger, STTProviderTurnTerminal, STTProviderTurnUpdate,
    STTSessionProjection, ScopedRecognitionEngine, SileroVadOnnx, SystemClock,
    TranslationRequestOwner, TranslationRuntimeConfig, TranslationRuntimeConfigSnapshot,
    TranslationTurnLifecycleOwner, TranslationTurnRequest, bundled_silero_vad_onnx_path,
    contribution_transcript, create_peer_vad_gating, digest, input_cases,
    inspect_artifact, np, run_audio_vad_loop, serial, validate_runtime,
)
from puripuly_heart.core.stt.backend import PermanentSTTScopedSessionError, STTProviderAudioBoundary
from puripuly_heart.core.stt.scoped_engine import STTRecognitionWatchdogs
from puripuly_heart.providers.stt import deepgram as adapter
from puripuly_heart.core import network_clients

SPEC = importlib.util.spec_from_file_location("issue212_common", ROOT.parent / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification/common.py")
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)


def clean(value):
    if dataclasses.is_dataclass(value):
        return clean(serial(value))
    if isinstance(value, dict):
        sensitive = {"api_key", "token", "account_id", "project_id", "organization_id", "user_id", "email"}
        return {k: "[redacted]" if k.lower() in sensitive or (k.lower() == "authorization" and not isinstance(v, dict)) else clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [clean(v) for v in value]
    return serial(value)


def save(path, value):
    text = json.dumps(clean(value), indent=2, ensure_ascii=False)
    for key in COMMON._SECRETS:
        text = text.replace(key, "[redacted]")
    home = str(Path.home())
    text = text.replace(home.replace("\\", "\\\\"), "<USER_HOME>").replace(home.replace("\\", "/"), "<USER_HOME>")
    path.write_text(text + "\n", encoding="utf-8")


def add_long_transfer(cases):
    original, mapping = cases["clean_A_B"]
    utterances = {item["input"]: (original[item["scenario_sample_range"][0]:item["scenario_sample_range"][1]], item["input_sample_range"]) for item in mapping}
    blocks, ranges, cursor = [np.zeros(8000, np.float32)], [], 8000
    for name in ("A", "A", "B", "B"):
        values, input_range = utterances[name]
        blocks.append(values)
        ranges.append({"input": name, "input_sample_range": input_range, "scenario_sample_range": [cursor, cursor + values.size]})
        cursor += values.size
    blocks.append(np.zeros(20000, np.float32))
    cases["long_A_long_B"] = (np.concatenate(blocks), ranges)


class Budget:
    def __init__(self, path):
        self.path = path
        self.value = json.loads(path.read_text(encoding="utf-8"))

    def reserve(self, kind, amount, details):
        field = "connections_reserved" if kind == "connection" else "audio_samples_reserved"
        limit = 12 if kind == "connection" else 180 * 16000
        if self.value[field] + amount > limit:
            raise PermanentSTTScopedSessionError("authorized cloud budget exhausted")
        self.value[field] += amount
        self.value["events"].append({"kind": kind, "amount": amount, "at": time.monotonic(), **details})
        save(self.path, self.value)


@contextmanager
def instrument(budget, result):
    cls = adapter._DeepgramSDKSession
    original_result, original_write = cls._build_transcript_event, cls._write_payload
    original_connect = network_clients.external_websocket_connect

    def connect(url, **kwargs):
        budget.reserve("connection", 1, {"case": result["case"], "mode": result["mode"], "provider": "deepgram", "endpoint": url})
        result["trace"].append({"event": "connection_attempt", "at": time.monotonic(), "endpoint": url})
        return original_connect(url, **kwargs)

    def raw_event(self, value):
        data = value.model_dump(mode="json") if hasattr(value, "model_dump") else {"type": type(value).__name__}
        result["trace"].append({"event": "raw_wire_result", "at": time.monotonic(), "value": clean(data)})
        return original_result(self, value)

    async def write(self, payload):
        if isinstance(payload, bytes):
            budget.reserve("audio", len(payload) // 2, {"case": result["case"], "mode": result["mode"], "provider": "deepgram"})
        else:
            name = "Finalize" if payload is adapter._FINALIZE else "CloseStream" if payload is adapter._CLOSE_STREAM else "other"
            result["trace"].append({"event": "control_requested", "at": time.monotonic(), "control": name})
        await original_write(self, payload)
        if isinstance(payload, bytes):
            result["actual_audio_samples_written"] += len(payload) // 2
        else:
            result["trace"].append({"event": "control_written", "at": time.monotonic(), "control": name, "sent_until": result["actual_audio_samples_written"]})

    cls._build_transcript_event, cls._write_payload = raw_event, write
    network_clients.external_websocket_connect = connect
    try:
        yield
    finally:
        cls._build_transcript_event, cls._write_payload = original_result, original_write
        network_clients.external_websocket_connect = original_connect


async def case_run(args, budget, name, values, mapping):
    result = {"case": name, "mode": args.mode, "status": "running", "assembly": mapping, "input_samples": int(values.size), "input_pcm_f32_sha256": __import__("hashlib").sha256(values.tobytes()).hexdigest(), "sources": [], "evidence": [], "trace": [], "errors": [], "retention": [], "actual_audio_samples_written": 0, "wrapper_state": []}
    trace = result["trace"]
    key = COMMON.secret("deepgram_api_key", "DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("Deepgram credential missing; no cloud call")
    backend = adapter.DeepgramRealtimeSTTBackend(api_key=key, language="en", model="nova-3", sample_rate_hz=16000, keyterms=(), stream_label="peer")
    settings = AudioSegmentSettingsSnapshot("deepgram", ("deepgram", "nova-3"), ("issue216",), "manual", "en", ("en",), 16000, 0.3, 600, 500)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    consumption = STTContributionConsumptionLedger()
    config = TranslationRuntimeConfigSnapshot(0, TranslationRuntimeConfig(source_language="en", target_language="ko", peer_source_language="en", peer_target_language="ko", peer_translation_enabled=True))
    peer = ChannelRuntime("peer")
    request_owner = TranslationRequestOwner(config_snapshot=lambda: config, self_runtime=ChannelRuntime("self"), peer_runtime=peer, context_resolver=ContextResolver(), provider_runtime=SimpleNamespace(provider=object(), current_provider_generation=lambda: (None, 0)), diagnostics=None, presentation=None, clock=SystemClock())

    async def nothing(*_args):
        return None

    async def source_only(*_args):
        return "source_only"

    async def admitted(children):
        requests = tuple(PeerTranslationChannelOwner._translation_process_request(child, child.target_language) for child in children)
        prepared = request_owner.admit_peer(requests)
        for child in children:
            item = prepared.get(child.utterance_id)
            result["sources"].append({"admitted_at": time.monotonic(), "source": serial(child.source_child), "prior_context_texts": list(item.context_texts) if item else None, "history_after": serial(peer.translation_history), "effective_equals_raw": child.source_child.effective_text == child.source_child.raw_text})

    async def rejected(parent_id):
        result["errors"].append(f"Source parent rejected: {parent_id}")

    async def terminal(child, outcome):
        if outcome in ("failed", "cancelled"):
            result["errors"].append(f"Source child terminal: {outcome}")

    turns = TranslationTurnLifecycleOwner(on_child_created=nothing, on_child_started=nothing, process_child=source_only, on_child_terminal=terminal, on_parent_closed=nothing, on_parent_rejected=rejected, on_parent_admitted=admitted)
    publication_order = 0

    async def event_sink(event):
        nonlocal publication_order
        trace.append({"event": "normalized_provider_result", "at": time.monotonic(), "value": serial(event)})
        if isinstance(event, STTProviderAudioBoundary):
            consumption.record_boundary(event.identity, event.boundary)
            trace.append({"event": "common_ledger_audio_boundary", "at": time.monotonic(), "value": serial(event)})
            return
        if not isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal)):
            return
        if args.mode == "baseline" and isinstance(event, STTProviderTurnUpdate):
            trace.append({"event": "baseline_early_admission_suppressed", "at": time.monotonic(), "identity": serial(event.identity), "source_release": serial(event.source_release)})
            return
        if isinstance(event, STTProviderTurnUpdate) and event.source_release is None:
            return
        snapshot = next((s for s in ledger.snapshots if s.identity == event.identity.segment), None)
        if snapshot is None:
            raise RuntimeError("result lacks original capture ledger scope")
        if isinstance(event, STTProviderTurnTerminal):
            if snapshot.sealed_at_monotonic_s is not None:
                consumption.seal_audio(event.identity)
            if event.outcome not in {"final", "degraded"} or not event.text:
                consumption.retire(event.identity)
                if event.outcome not in {"empty", "final"}:
                    result["errors"].append(f"Provider terminal {event.outcome}: {event.failure_reason}")
                return
        pieces = consumption.consume_parts(event)
        if pieces:
            publication_order += 1
            transcript = contribution_transcript(snapshot, event, pieces, created_at=time.monotonic(), publication_order=publication_order)
            await turns.submit(TranslationTurnRequest(transcript, "Peer", "peer", ("ko",), config))

    sessions = []
    async def factory(_settings, epoch):
        try:
            inner = SessionTrace(await backend.open_session(projection=STTSessionProjection("scoped", epoch)), trace, values)
        except Exception as error:
            raise PermanentSTTScopedSessionError(COMMON.safe_error(error)) from error
        if args.mode == "new":
            from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
            inner = StreamingSpeakerSession(inner)
        sessions.append(inner)
        return inner

    engine = ScopedRecognitionEngine(factory, channel="peer", event_sink=event_sink, diagnostic_sink=lambda event: trace.append({"event": "engine_diagnostic", "at": time.monotonic(), "value": serial(event)}))
    speaker = nemotron = None
    started = 0.0
    finished = False
    capture_hash = __import__("hashlib").sha256()
    capture_samples = 0
    result["capture_delivery_max_lag_seconds"] = 0.0

    def state():
        result["retention"].append({"at": time.monotonic(), "value": serial(engine.retention_snapshot)})
        for session in sessions:
            if hasattr(session, "audio_boundaries"):
                result["wrapper_state"].append({"at": time.monotonic(), "sent_until": session.sent_until, "audio_boundaries": serial(tuple(session.audio_boundaries))})

    async def evidence_pump():
        while not finished:
            items = speaker.drain_evidence(2048)
            if items:
                result["evidence"].extend(serial(item) for item in items)
                trace.append({"event": "speaker_evidence_observed", "at": time.monotonic(), "value": serial(items)})
                await engine.observe_speaker_evidence(items)
                state()
            await asyncio.sleep(0.005)

    class Sink:
        async def handle_stream_input(self, event):
            nonlocal capture_samples
            capture_samples += event.chunk.size
            capture_hash.update(event.chunk.tobytes())
            if speaker is not None:
                speaker.observe_input(f"issue216:{name}", event)
            state()

        async def handle_owned_vad_event(self, owned):
            if isinstance(owned.event, (SpeechStart, SpeechEnd)):
                trace.append({"event": "vad", "at": time.monotonic(), "value": serial(owned)})
            await engine.handle_owned_vad_event(owned)

        async def observe_source_activity(self, **kwargs):
            await engine.observe_source_activity(**kwargs)

    class Source:
        terminal_reason = "closed"
        async def frames(self):
            for sequence, offset in enumerate(range(0, values.size, 512)):
                end = min(values.size, offset + 512)
                await asyncio.sleep(max(0, started + end / 16000 - time.monotonic()))
                result["capture_delivery_max_lag_seconds"] = max(result["capture_delivery_max_lag_seconds"], time.monotonic() - (started + end / 16000))
                yield AudioFrameF32(values[offset:end], 16000, capture=AudioCaptureSpan(7, sequence, 16000, offset, end, started + offset / 16000, started + end / 16000))

    pump = None
    try:
        with instrument(budget, result):
            if args.mode == "new":
                nemotron = NemotronBackend(artifact_path=args.model, runtime_dir=args.runtime, device=0, profile=args.profile)
                speaker = SpeakerRuntime(nemotron)
                await speaker.start()
                if not speaker.snapshot().ready:
                    raise RuntimeError(speaker.snapshot().error)
                result["speaker_initial"] = serial(speaker.snapshot())
                pump = asyncio.create_task(evidence_pump())
            vad = create_peer_vad_gating(engine=SileroVadOnnx(bundled_silero_vad_onnx_path()), sample_rate_hz=16000, ring_buffer_ms=500, speech_threshold=0.3, hangover_ms=600)
            await engine._ensure_session(settings, STTRecognitionWatchdogs(), engine._authority_generation)
            result["provider_session_ready_at"] = time.monotonic()
            result["readiness_scope"] = "Actual provider and VAD/model ready before fixed realtime playback; first-connect startup buffering not qualified by this smoke."
            started = time.monotonic()
            result["playback_started_at"] = started
            result["vad_settings"] = {"model_sha256": digest(bundled_silero_vad_onnx_path()), "sample_rate_hz": 16000, "chunk_samples": 512, "speech_threshold": 0.3, "hangover_ms": 600, "onset_pre_roll_ms": 500, "hard_rollover_pre_roll_ms": 300}
            await run_audio_vad_loop(source=Source(), vad=vad, sink=Sink(), target_sample_rate_hz=16000, channel_label="peer", segment_ledger=ledger)
            result["input_finished_at"] = time.monotonic()
            deadline = time.monotonic() + 25
            while not engine.is_at_turn_boundary or (speaker is not None and speaker.snapshot().pending_items):
                if time.monotonic() > deadline:
                    raise TimeoutError("actual model/provider owned-input drain timeout")
                await asyncio.sleep(0.01)
            await engine.wait_for_event_ingress_drain()
            await turns.wait_for_idle()
            state()
            result["segments"] = serial(ledger.snapshots)
            if speaker is not None:
                result["speaker_final"] = serial(speaker.snapshot())
            result["original_capture_received_once"] = capture_samples == values.size and capture_hash.hexdigest() == result["input_pcm_f32_sha256"]
            result["status"] = "passed" if result["sources"] and not result["errors"] else "failed"
            result["wall_seconds"] = time.monotonic() - started
            finished = True
            if pump is not None:
                await pump
            await engine.close_backend()
    except Exception as error:
        result["status"], result["error"] = "failed", COMMON.safe_error(error)
    finally:
        finished = True
        if pump is not None:
            pump_results = await asyncio.gather(pump, return_exceptions=True)
            result["errors"].extend(COMMON.safe_error(error) for error in pump_results if isinstance(error, Exception))
        for label, cleanup in (("engine", engine.close_backend), ("turns", turns.close)):
            try:
                await cleanup()
            except Exception as error:
                result["errors"].append(f"{label} cleanup: {COMMON.safe_error(error)}")
        if speaker is not None:
            try:
                await speaker.stop()
            except Exception as error:
                result["errors"].append(f"speaker cleanup: {COMMON.safe_error(error)}")
            result["worker_stderr"] = bytes(nemotron._stderr).decode(errors="replace")
        if result["errors"]:
            result["status"] = "failed"
        if result["status"] != "passed":
            save(args.output.with_name(f"{args.output.stem}_{name}_partial.json"), result)
    return result


async def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GGML_VK_DISABLE_GRAPH_OPTIMIZE"] = "1"
    os.environ["NEMO_SPEECH_SCHEDULE_CAPTURE"] = "1"
    inputs, cases = input_cases(args)
    add_long_transfer(cases)
    selected = args.cases.split(",")
    for name in selected:
        if name not in cases:
            raise ValueError(f"unknown authorized case: {name}")
    budget = Budget(args.budget)
    report = {"schema": "issue216-real-stream-smoke-1", "command": sys.argv, "python": sys.version, "mode": args.mode, "profile": args.profile if args.mode == "new" else None, "trimming": "OFF", "scope": "Actual production Silero/delivery -> Deepgram adapter -> optional real SpeakerRuntime+StreamingSpeakerSession -> production ScopedRecognitionEngine normalizer -> contribution consumption -> SourceChild lifecycle -> TranslationRequestOwner context admission. Translation execution and presentation intentionally not claimed.", "privacy": {"public_or_previously_authorized_synthetic_only": True, "ambient_capture": False, "private_uploads": False, "automatic_retries": 0}, "inputs": inputs, "nemotron_artifact": inspect_artifact(args.model), "runtime": validate_runtime(args.runtime), "runtime_manifest_sha256": digest(args.runtime / "runtime-manifest.json"), "cases": [], "limits": {"translation_execution": "not_run", "caption_surface": "not_run", "HMD": "not_run", "profile_selection": "not_run"}}
    report["baseline_contrast"] = "Same integrated Deepgram adapter/settings/input and application physical cut schedule. Baseline has no StreamingSpeakerSession/Nemotron and suppresses all update-driven source admission, preserving complete adapter text for ordinary terminal admission. New permits production complete-final source release plus real mapped Nemotron attribution. Disabling evidence alone would still early-admit gray and is not this baseline."
    report["dependencies"] = {name: version(name) for name in ("deepgram-sdk", "websockets", "onnxruntime", "numpy")}
    paths = [Path(__file__), ROOT / "src/puripuly_heart/providers/stt/deepgram.py", ROOT / "src/puripuly_heart/core/stt/backend.py", ROOT / "src/puripuly_heart/core/stt/scoped_normalizer.py", ROOT / "src/puripuly_heart/core/stt/scoped_engine.py", ROOT / "src/puripuly_heart/core/stt/streaming_speaker.py", ROOT / "src/puripuly_heart/core/speaker/runtime.py", ROOT / "src/puripuly_heart/core/orchestrator/translation_turn.py", ROOT / "src/puripuly_heart/core/orchestrator/translation_request.py"]
    paths.extend(ROOT / relative for relative in (
        "native/nemotron/issue215_smoke.py",
        "src/puripuly_heart/providers/stt/sdk_network.py",
        "src/puripuly_heart/core/stt/recognition_units.py",
        "src/puripuly_heart/core/audio/ownership.py",
        "src/puripuly_heart/core/audio/listen_delivery.py",
        "src/puripuly_heart/core/runtime/audio_vad_loop.py",
        "src/puripuly_heart/core/vad/gating.py",
        "src/puripuly_heart/app/adapters/nemotron/backend.py",
    ))
    report["execution_source_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in paths if path.exists()}
    for name in selected:
        values, mapping = cases[name]
        report["cases"].append(await case_run(args, budget, name, values, mapping))
        current_hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths if path.exists()}
        report["cases"][-1]["candidate_code_unchanged_during_execution"] = current_hashes == report["execution_source_sha256"]
        if current_hashes != report["execution_source_sha256"]:
            report["cases"][-1]["status"] = "failed"
            report["cases"][-1]["errors"].append("candidate code changed during paired evidence execution")
        report["budget_after"] = {k: v for k, v in budget.value.items() if k != "events"}
        save(args.output, report)
        if report["cases"][-1]["status"] != "passed":
            break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    parser.add_argument("--mode", choices=("baseline", "new"), required=True)
    parser.add_argument("--model", type=Path, default=home / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf")
    parser.add_argument("--runtime", type=Path, default=ROOT / "build/nemotron")
    parser.add_argument("--profile", choices=("0.64", "1.04"), default="0.64")
    parser.add_argument("--corpus", type=Path, default=home / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1")
    parser.add_argument("--correction-root", type=Path, default=ROOT.parent / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification")
    parser.add_argument("--cases", default="clean_A_B")
    parser.add_argument("--budget", type=Path, default=ROOT / "native/nemotron/issue216_budget.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.name.startswith("issue216"):
        raise ValueError("evidence output must remain an issue216 task-local artifact")
    asyncio.run(run(args))
