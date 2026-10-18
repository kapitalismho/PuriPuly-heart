from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import issue215_smoke as prior
import numpy as np
import psutil
from issue216_presentation import OwnedClientCapture

from puripuly_heart.app.adapters.gpu_worker_process import DefaultGpuWorkerProcessFactory
from puripuly_heart.app.services.managed_gemma_translation import ManagedGemmaTranslationOwner
from puripuly_heart.app.wiring.wiring_llm_factory import (
    llm_factory_extras_from_vnext,
    runtime_resolution_input_from_vnext,
)
from puripuly_heart.app.wiring.wiring_managed_gemma import managed_gemma_selection
from puripuly_heart.app.wiring.wiring_translation_backend import create_translation_backend
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.prompts import resolve_system_prompt
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.local_translation.runtime_profile import default_gemma_runtime_paths
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOutputProjectionOwner,
    TranslationUiMessageQueue,
)
from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.providers.llm.managed_gemma import HttpxManagedGemmaTransport
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend
from puripuly_heart.release_evidence.managed_gemma_production import (
    _captured_process_factory,
    _is_managed_only_backend,
    _settings,
)
from puripuly_heart.ui.desktop_overlay import DesktopOverlayRenderer, DiagnosticLocalRendererPort
from puripuly_heart.ui.desktop_overlay_repro import (
    LocalAuthenticatedRawIngress,
    _NeverExitParentMonitor,
)

serial = prior.serial
digest = prior.digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(serial(value), indent=2, ensure_ascii=False, default=str)
    for home in (str(Path.home()), str(Path.home()).replace("\\", "/")):
        text = text.replace(home.replace("\\", "\\\\"), "<USER_HOME>").replace(home, "<USER_HOME>")
    path.write_text(text + "\n", encoding="utf-8")


class ChatboxGuard:
    def __getattr__(self, name):
        if name in ("clear_typing_reasons", "drop_pending", "process_due", "clear_pending"):
            return lambda *args, **kwargs: None
        raise RuntimeError("Task must not send chatbox/OSC: " + name)


class Desktop:
    def __init__(self, output):
        self.output = output
        self.events = []
        self.snapshots = []
        self.readbacks = []
        self.readback_tasks = []
        self.errors = []
        self.committed = {}
        self.revision = 0
        self.ingress = LocalAuthenticatedRawIngress(
            secrets.token_urlsafe(32),
            OverlayPresentationSnapshot(revision=0, blocks=[]),
            "issue218",
        )
        self.port = DiagnosticLocalRendererPort(acknowledgement_timeout_s=10)
        self.ready = asyncio.Event()
        self.renderer = None
        self.task = None
        self.consumer = None
        self.backdrop = None
        self.capture = OwnedClientCapture()
        self.title = "issue218-owned-" + secrets.token_hex(8)

    async def lifecycle(self, event):
        if event.get("type") == "overlay_ready":
            self.ready.set()
        if event.get("type") in ("startup_error", "runtime_error"):
            self.errors.append(event)

    async def consume(self):
        while True:
            envelope = await self.port.next_event()
            record = dict(envelope.record)
            self.events.append({"observed_at": time.monotonic(), "record": record})
            if envelope.acknowledgement is not None:
                self.port.acknowledge_render_commit(record["renderer_revision"])
            if record["event_type"] == "render_commit_acknowledgement":
                self.committed.setdefault(record["renderer_revision"], asyncio.Event()).set()

    async def start(self):
        self.backdrop = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(ROOT / "native/nemotron/issue216_presentation.py"),
                "--backdrop",
                self.title,
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        header = await asyncio.wait_for(asyncio.to_thread(self.backdrop.stdout.readline), 10)
        self.backdrop_pid = json.loads(header)["pid"]
        url = await self.ingress.start()
        manifest = OverlayLaunchManifest(
            contract_version=OVERLAY_CONTRACT_VERSION,
            app_version="issue218",
            overlay_instance_id="issue218",
            bridge_url=url,
            session_token=self.ingress.token,
            parent_pid=os.getpid(),
            startup_deadline_ms=10000,
            log_dir="diagnostic",
            log_level="WARNING",
            locale="en",
        )
        self.renderer = DesktopOverlayRenderer(
            manifest,
            lifecycle_sink=SimpleNamespace(emit=self.lifecycle),
            parent_monitor=_NeverExitParentMonitor(),
            diagnostic_port=self.port,
        )
        self.consumer = asyncio.create_task(self.consume())
        self.task = asyncio.create_task(self.renderer.run())
        await self.ingress.wait_connected()
        await asyncio.wait_for(self.ready.wait(), 20)
        await self.ingress.send_runtime_control(
            {"command": "apply_window_bounds", "x": 32, "y": 32, "width": 1344, "height": 336}
        )
        await self.ingress.send_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        self.pid = (
            self.renderer.window._owner_process_pid() or self.renderer.window._flet_process_pid
        )
        if not self.pid or self.pid == os.getpid():
            raise RuntimeError("Actual independent Flet renderer PID unresolved")

    async def replace_snapshot(self, snapshot, *, block_expirations=None):
        self.revision += 1
        snapshot = dataclasses.replace(snapshot, revision=self.revision)
        started = time.monotonic()
        record = {
            "submitted_at": started,
            "snapshot": snapshot.to_dict(),
            "block_expirations": serial(block_expirations),
        }
        self.snapshots.append(record)
        await self.ingress.write_batch([snapshot])
        await asyncio.wait_for(self.committed.setdefault(self.revision, asyncio.Event()).wait(), 15)
        record["software_commit_observed_at"] = time.monotonic()
        if snapshot.blocks:
            self.readback_tasks.append(
                asyncio.create_task(self.readback(snapshot, record["software_commit_observed_at"]))
            )

    async def broadcast_shutdown(self):
        await self.ingress.close()

    async def readback(self, snapshot, ack_at):
        await asyncio.sleep(0.06)
        started = time.monotonic()
        value = {
            "revision": snapshot.revision,
            "dispatch_ack_at": ack_at,
            "readback_started_at": started,
            "requested_snapshot": snapshot.to_dict(),
            "current_revision_at_readback_start": self.revision,
        }
        try:
            value["capture"] = await asyncio.to_thread(
                self.capture.capture,
                self.pid,
                self.backdrop_pid,
                self.title,
                self.output / f"commit_{snapshot.revision}.png",
            )
        except Exception as error:
            value["capture_error"] = str(error)
        value["readback_completed_at"] = time.monotonic()
        value["current_revision_at_readback_end"] = self.revision
        value["correlation_limit"] = (
            "Expected live snapshot/logical_turn_key recorded; exact visible text requires image observation. Readback is an upper observation bound, not pixel onset or photon latency."
        )
        self.readbacks.append(value)

    async def picture(self, label):
        await asyncio.sleep(0.15)
        try:
            return self.capture.capture(
                self.pid, self.backdrop_pid, self.title, self.output / (label + ".png")
            )
        except Exception as error:
            return {"status": "not_captured", "reason": str(error)}

    async def close(self):
        if self.readback_tasks:
            await asyncio.gather(*self.readback_tasks)
        await self.ingress.close()
        if self.renderer:
            await self.renderer.shutdown()
        if self.task:
            await asyncio.wait_for(self.task, 10)
        if self.consumer:
            self.consumer.cancel()
            await asyncio.gather(self.consumer, return_exceptions=True)
        if self.backdrop:
            with contextlib.suppress(OSError):
                self.backdrop.stdin.write("close\n")
                self.backdrop.stdin.flush()
            try:
                await asyncio.to_thread(self.backdrop.wait, 5)
            except subprocess.TimeoutExpired:
                self.backdrop.kill()
                await asyncio.to_thread(self.backdrop.wait, 5)


class ObservedPeer(prior.PeerTranslationChannelOwner):
    async def on_parent_admitted(self, children):
        await super().on_parent_admitted(children)
        for child in children:
            prepared = self._prepared_requests.get(child.utterance_id)
            self.record["sources"].append(
                {
                    "admitted_at": time.monotonic(),
                    "source": serial(child.source_child),
                    "prior_context_texts": list(prepared.context_texts) if prepared else None,
                    "prepared": serial(prepared),
                    "history_after": serial(self.runtime.translation_history),
                }
            )

    async def process_child(self, child, cancellation_requested):
        started = time.monotonic()
        self.record["trace"].append(
            {
                "event": "translation_start",
                "at": started,
                "source_id": str(child.source_child.source_id),
            }
        )
        result = await super().process_child(child, cancellation_requested)
        self.record["trace"].append(
            {
                "event": "translation_end",
                "at": time.monotonic(),
                "source_id": str(child.source_child.source_id),
                "result": serial(result),
            }
        )
        return result

    async def submit_translation_output(self, submission):
        self.record["trace"].append(
            {
                "event": "ordered_publication_wait",
                "at": time.monotonic(),
                "submission": serial(submission),
            }
        )
        receipt = await super().submit_translation_output(submission)
        self.record["trace"].append(
            {
                "event": "ordered_publication_complete",
                "at": time.monotonic(),
                "source_id": str(submission.source_child.source_id),
                "receipt": serial(receipt),
            }
        )
        return receipt

    async def on_child_terminal(self, child, outcome):
        self.record["trace"].append(
            {
                "event": "child_terminal",
                "at": time.monotonic(),
                "source_id": str(child.source_child.source_id),
                "outcome": outcome,
            }
        )
        await super().on_child_terminal(child, outcome)


async def gemma_start(args, report):
    captures = []
    messages = []
    settings = _settings("gpu")
    runtime = ManagedGemmaRuntimeOwner(
        install_dir=args.gemma_model,
        runtime_paths=default_gemma_runtime_paths(args.gemma_runtime),
        process_factory=_captured_process_factory(captures),
        transport_factory=HttpxManagedGemmaTransport,
        log_sink=lambda message, level: messages.append(
            {"at": time.monotonic(), "level": level, "message": message}
        ),
    )
    owner = ManagedGemmaTranslationOwner(runtime=runtime)
    selection = managed_gemma_selection(settings)
    activation = await owner.prepare(selection)
    backend = create_translation_backend(
        translation_model="managed_gemma",
        secrets=InMemorySecretStore(),
        http_extensions=HttpExtensionRegistry(args.output / "unused-http-extensions"),
        runtime_input=runtime_resolution_input_from_vnext(settings),
        extras=llm_factory_extras_from_vnext(settings),
        managed_gemma_runtime=activation.runtime,
        managed_gemma_release=activation.release,
    )
    if not _is_managed_only_backend(backend):
        raise RuntimeError("Nonlocal translation provider is prohibited")
    report["gemma"] = {
        "requested_backend": "gpu",
        "readiness": serial(activation.readiness),
        "selection": serial(selection),
        "messages": messages,
        "runtime_commands": [list(c.command) for c in captures],
        "prefix_disk_cache": "disabled",
        "warmup": "production owner prepare/prefix prefill; no additional request warmup",
        "prompt_cache": "llama cache-prompt enabled; runtime kept resident throughout rotated arms",
    }
    return backend, owner, captures


async def resources(record, finished, speaker, gpu, turns):
    proc = psutil.Process()
    while not finished.is_set():
        sample = {"at": time.monotonic(), "processes": []}
        for child in [proc, *proc.children(recursive=True)]:
            try:
                mem = child.memory_info()
                sample["processes"].append(
                    {
                        "pid": child.pid,
                        "name": child.name(),
                        "rss": mem.rss,
                        "private": getattr(mem, "private", None),
                        "cpu_times": list(child.cpu_times()),
                    }
                )
            except psutil.NoSuchProcess, psutil.AccessDenied:
                pass
        sample["speaker"] = serial(speaker.snapshot()) if speaker else None
        sample["gpu_asr_pending"] = gpu.pending_count if gpu else None
        sample["translation_active_tasks"] = len(turns._active_tasks)
        sample["translation_parents"] = len(turns._parents)
        record["resources"].append(sample)
        try:
            await asyncio.wait_for(finished.wait(), 0.25)
        except TimeoutError:
            pass


async def case_run(args, name, values, mapping, translation, desktop, mode, profile, repetition):
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
    }
    trace = record["trace"]
    lang = "ko" if name.startswith("korean") else "ja" if name.startswith("japanese") else "en"
    target = "en" if lang == "ko" else "ko"
    gpu = None
    if args.route == "gpu":
        gpu = SharedGpuASRRuntime(
            process_factory=DefaultGpuWorkerProcessFactory(executable_path=args.gpu_worker),
            diagnostic_sink=lambda d: trace.append(
                {"event": "gpu_asr_diagnostic", "at": time.monotonic(), "value": serial(d)}
            ),
        )
        devices = await gpu.discover_devices()
        record["gpu_devices"] = serial(devices)
        try:
            device = next(d for d in devices if d.registry_index == 0)
            backend = LocalGpuSTTBackend(
                runtime=gpu,
                channel="peer",
                model_path=args.gpu_model,
                model_id="qwen3-asr-1.7b-q6-k-transcribe-vulkan",
                device_id=device.device_id,
                language_hint=lang,
            )
            await backend.prepare()
            record["gpu_activation"] = serial(gpu._activation)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            save(args.output / "gpu_setup_failure.json", record)
            await gpu.close()
            raise
    else:
        backend = prior.create_local_cpu_backend(
            args.asr_model,
            model_root=args.asr_root,
            source_language=lang,
            sample_rate_hz=16000,
            stream_label="peer",
        )
        await backend._ensure_recognizer()
    provider_id = getattr(backend, "provider_id", "local_gpu")
    settings = prior.AudioSegmentSettingsSnapshot(
        provider_id,
        (provider_id, args.route, backend.model_id),
        ("issue218",),
        "manual",
        lang,
        (lang,),
        16000,
        0.3,
        600,
        500,
    )
    ledger = prior.PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    consumption = prior.STTContributionConsumptionLedger()
    config = prior.TranslationRuntimeConfigSnapshot(
        0,
        prior.TranslationRuntimeConfig(
            source_language=lang,
            target_language=target,
            peer_source_language=lang,
            peer_target_language=target,
            peer_translation_enabled=True,
            concurrency_limit=3,
            integrated_context_enabled=True,
            system_prompt=resolve_system_prompt(""),
        ),
    )
    clock = prior.SystemClock()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), bridge=desktop, translation_enabled=True
    )
    output = OutputRuntime(
        chatbox=ChatboxGuard(),
        clock=clock,
        overlay_sink=presenter,
        routing_observer=lambda decision: trace.append(
            {"event": "routing_decision", "at": time.monotonic(), "value": serial(decision)}
        ),
        activation_notice_enabled=False,
    )
    output.activate_peer_generation(1)
    ui_events = asyncio.Queue()
    diagnostics = TranslationLatencyDiagnosticsOwner(clock=clock, config_snapshot=lambda: config)
    projection = TranslationOutputProjectionOwner(
        output_runtime=output,
        ui_messages=TranslationUiMessageQueue(ui_events, output),
        diagnostics=diagnostics,
        clock=clock,
    )
    peer_runtime = prior.ChannelRuntime("peer")
    requests = prior.TranslationRequestOwner(
        config_snapshot=lambda: config,
        self_runtime=prior.ChannelRuntime("self"),
        peer_runtime=peer_runtime,
        context_resolver=prior.ContextResolver(clock=clock, config_snapshot=lambda: config),
        provider_runtime=ProviderRuntimeHandle(name="llm", provider=translation),
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
    peer = ObservedPeer(
        runtime=peer_runtime,
        config_snapshot=lambda: config,
        translation_turns=turns,
        local_asr_runtime=None,
        translation_requests=requests,
        output_projection=projection,
        diagnostics=diagnostics,
        clock=clock,
    )
    peer.record = record
    order = 0

    async def event_sink(event):
        nonlocal order
        trace.append({"event": "provider_result", "at": time.monotonic(), "value": serial(event)})
        if not isinstance(event, (prior.STTProviderTurnUpdate, prior.STTProviderTurnTerminal)):
            return
        if isinstance(event, prior.STTProviderTurnUpdate) and event.source_release is None:
            return
        snapshot = next(s for s in ledger.snapshots if s.identity == event.identity.segment)
        if isinstance(event, prior.STTProviderTurnTerminal):
            consumption.seal_audio(event.identity)
        pieces = consumption.consume_parts(event)
        if pieces:
            order += 1
            transcript = prior.contribution_transcript(
                snapshot, event, pieces, created_at=time.monotonic(), publication_order=order
            )
            await peer.handle_stt_event(STTFinalEvent(transcript.utterance_id, transcript))

    sessions = []

    async def factory(_settings, epoch):
        session = prior.SessionTrace(
            await backend.open_session(projection=prior.STTSessionProjection("scoped", epoch)),
            trace,
            values,
        )
        if mode == "new":
            from puripuly_heart.core.stt.batch_speaker import CausalBatchSession

            record["boundary_policy"] = {
                k: v
                for k, v in vars(CausalBatchSession).items()
                if k.isupper() and isinstance(v, (int, float, str, bool))
            }
            session = CausalBatchSession(session)
        sessions.append(session)
        return session

    engine = prior.ScopedRecognitionEngine(
        factory, channel="peer", event_sink=event_sink, backend_close=backend.close
    )
    speaker = nemotron = None
    finished = asyncio.Event()
    pump = sampler = None
    handler = prior.TraceHandler(trace)
    logger = logging.getLogger("puripuly_heart.core.stt.batch_speaker")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    async def evidence_pump():
        while not finished.is_set():
            items = speaker.drain_evidence(2048)
            if items:
                record["evidence"].extend(serial(item) for item in items)
                trace.append(
                    {"event": "evidence_delivery", "at": time.monotonic(), "count": len(items)}
                )
                await engine.observe_speaker_evidence(items)
            await asyncio.sleep(0.005)

    received = hashlib.sha256()
    record["capture_input_samples"] = 0

    class Sink:
        async def handle_stream_input(self, event):
            record["capture_input_samples"] += event.chunk.size
            received.update(event.chunk.tobytes())
            if speaker:
                speaker.observe_input("issue218:" + name, event)

        async def handle_owned_vad_event(self, owned):
            if isinstance(owned.event, (prior.SpeechStart, prior.SpeechEnd)):
                trace.append({"event": "vad", "at": time.monotonic(), "value": serial(owned)})
                peer._record_peer_owned_vad_event(owned)
            await engine.handle_owned_vad_event(owned)

        async def observe_source_activity(self, **kwargs):
            await engine.observe_source_activity(**kwargs)

    class Source:
        terminal_reason = "closed"

        async def frames(self):
            for sequence, offset in enumerate(range(0, values.size, 512)):
                end = min(values.size, offset + 512)
                due = started + end / 16000
                await asyncio.sleep(max(0, due - time.monotonic()))
                record["frame_delivery_lags"].append(time.monotonic() - due)
                yield prior.AudioFrameF32(
                    values[offset:end],
                    16000,
                    capture=prior.AudioCaptureSpan(
                        7, sequence, 16000, offset, end, started + offset / 16000, due
                    ),
                )

    event_start, snapshot_start, readback_start = (
        len(desktop.events),
        len(desktop.snapshots),
        len(desktop.readbacks),
    )
    try:
        if mode == "new":
            nemotron = prior.NemotronBackend(
                artifact_path=args.model, runtime_dir=args.runtime, device=0, profile=profile
            )
            speaker = prior.SpeakerRuntime(nemotron)
            await speaker.start()
            if not speaker.snapshot().ready:
                raise RuntimeError(speaker.snapshot().error)
            record["speaker_initial"] = serial(speaker.snapshot())
            pump = asyncio.create_task(evidence_pump())
        started = time.monotonic()
        record["playback_started_at"] = started
        record["translation_config"] = serial(config)
        record["asr"] = {
            k: serial(getattr(backend, k, None))
            for k in (
                "model_id",
                "provider",
                "num_threads",
                "language_hint",
                "hotwords",
                "source_mode",
                "active_decode_timeout_s",
            )
        }
        sampler = asyncio.create_task(resources(record, finished, speaker, gpu, turns))
        vad = prior.create_peer_vad_gating(
            engine=prior.SileroVadOnnx(prior.bundled_silero_vad_onnx_path()),
            sample_rate_hz=16000,
            ring_buffer_ms=500,
            speech_threshold=0.3,
            hangover_ms=600,
        )
        record["vad"] = {
            "sha256": digest(prior.bundled_silero_vad_onnx_path()),
            "speech_threshold": 0.3,
            "hangover_ms": 600,
            "onset_pre_roll_ms": 500,
            "hard_rollover_pre_roll_ms": 300,
            "chunk_samples": vad.chunk_samples,
            "continuation_threshold": vad.continuation_threshold,
            "start_debounce_chunks": vad.start_debounce_chunks,
            "start_commit_chunks": vad.start_commit_chunks,
        }
        await prior.run_audio_vad_loop(
            source=Source(),
            vad=vad,
            sink=Sink(),
            target_sample_rate_hz=16000,
            channel_label="peer",
            segment_ledger=ledger,
        )
        deadline = time.monotonic() + 60
        while not engine.is_at_turn_boundary or (speaker and speaker.snapshot().pending_items):
            if time.monotonic() > deadline:
                raise TimeoutError("Actual ASR/speaker drain timeout")
            await asyncio.sleep(0.01)
        await engine.wait_for_event_ingress_drain()
        await asyncio.wait_for(turns.wait_for_idle(), 90)
        await asyncio.sleep(0.1)
        record["speaker_final"] = serial(speaker.snapshot()) if speaker else None
        record["segments"] = serial(ledger.snapshots)
        record["capture_received_once"] = (
            record["capture_input_samples"] == values.size
            and received.hexdigest() == record["input_pcm_f32_sha256"]
        )
        record["status"] = "executed" if record["sources"] else "failed"
        record["wall_seconds"] = time.monotonic() - started
        record["snapshot_capture"] = await desktop.picture(
            f"{args.route}_{mode}_{profile}_{repetition}_{name}"
        )
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
    finally:
        finished.set()
        if pump:
            try:
                await pump
            except Exception as error:
                record["errors"].append("evidence_pump: " + str(error))
        if sampler:
            await sampler
        await engine.close_backend()
        await turns.close()
        await output.close()
        await presenter.close()
        if speaker:
            await speaker.stop()
            record["nemotron_stderr"] = bytes(nemotron._stderr).decode(errors="replace")
        if gpu:
            await gpu.close()
        logger.removeHandler(handler)
        record["renderer_events"] = desktop.events[event_start:]
        record["renderer_snapshots"] = desktop.snapshots[snapshot_start:]
        record["renderer_readbacks"] = desktop.readbacks[readback_start:]
        await desktop.replace_snapshot(OverlayPresentationSnapshot(revision=0, blocks=[]))
    return record


def inputs(args):
    provenance, cases = prior.input_cases(args)
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    for label, language in [("korean_read", "ko"), ("japanese_read", "ja")]:
        source = next(
            t["source"]
            for t in manifest["traces"]
            if t["source"]["locale"].split("_")[0] == language
        )
        source = {k: v for k, v in source.items() if k != "speaker_id"}
        path = args.corpus / source["relative_audio_path"]
        if digest(path) != source["source_audio_sha256"]:
            raise RuntimeError("Public input hash mismatch")
        values = prior.audio(path)
        provenance[label] = source
        cases[label] = (
            np.concatenate([np.zeros(8000, np.float32), values, np.zeros(20000, np.float32)]),
            [
                {
                    "input": label,
                    "input_sample_range": [0, values.size],
                    "scenario_sample_range": [8000, 8000 + values.size],
                }
            ],
        )
    a, b = cases["clean_A_B"][0], cases["ordinary_onset"][0]
    overlap = a.copy()
    start = min(24000, overlap.size - 1)
    size = min(b.size, overlap.size - start)
    overlap[start : start + size] = np.clip(
        overlap[start : start + size] * 0.7 + b[:size] * 0.7, -1, 1
    )
    cases["synthetic_overlap"] = (
        overlap,
        [
            {
                "kind": "scoring_only_synthetic_mix",
                "base": "clean_A_B",
                "mixed_input": "ordinary_onset",
                "offset_samples": start,
                "samples": size,
                "gains": [0.7, 0.7],
            }
        ],
    )
    return provenance, cases


async def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GGML_VK_DISABLE_GRAPH_OPTIMIZE"] = "1"
    os.environ["NEMO_SPEECH_SCHEDULE_CAPTURE"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    provenance, cases = inputs(args)
    report = {
        "schema": "issue218-causal-real-local-e2e-1",
        "candidate": "ea488774c3d1a89b92130b862d2d4102c91c1349",
        "command": sys.argv,
        "driver_sha256": digest(Path(__file__)),
        "official_model_card_revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        "nemotron_artifact": prior.inspect_artifact(args.model),
        "nemotron_runtime": prior.validate_runtime(args.runtime),
        "runtime_manifest_sha256": digest(args.runtime / "runtime-manifest.json"),
        "inputs": provenance,
        "corpus_manifest_sha256": digest(args.corpus / "manifest.json"),
        "dependencies": {
            k: importlib.metadata.version(k)
            for k in ("numpy", "onnxruntime", "sherpa-onnx", "httpx", "flet", "psutil")
        },
        "privacy": {
            "cloud_calls": 0,
            "ambient_capture": False,
            "user_settings_read_write": False,
            "VRChat_or_SteamVR_launched": False,
            "private_audio": False,
            "network": "only task-owned loopback HTTP/websocket/native IPC",
        },
        "policy": {
            "trimming": "OFF",
            "padding_workaround": False,
            "threshold_tuning": False,
            "soniox_endpointing_inherited": "ON, not executed",
            "arms": "natural/six-second baseline without early release; complete1.04; complete0.64; same current UI",
            "ui": "production OverlayPresenter default calibration/two cards/TTL; no fixed1s wait",
            "resource_limits": "quarter-second own descendant process memory/CPU and speaker buffers; GPU timestamps unknown; physical capture loss not observed by generated replay",
        },
        "route": args.route,
        "cases": [],
    }
    report["asr_model_sha256"] = (
        digest(args.gpu_model)
        if args.route == "gpu"
        else {p.name: digest(p) for p in (args.asr_root / args.asr_model).rglob("*") if p.is_file()}
    )
    if args.route == "gpu":
        report["gpu_worker_sha256"] = digest(args.gpu_worker)
    desktop = Desktop(args.output)
    translation = owner = None
    captures = []
    counter_process = await asyncio.create_subprocess_exec(
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
    counter_output = asyncio.create_task(counter_process.communicate())
    report["counter_clock_anchor"] = {"monotonic": time.monotonic(), "unix_seconds": time.time()}
    try:
        translation, owner, captures = await gemma_start(args, report)
        await desktop.start()
        report["desktop"] = {
            "renderer": type(desktop.renderer).__name__,
            "window": type(desktop.renderer.window).__name__,
            "pid": desktop.pid,
            "real_render_boundary": "Real Flet dispatch completion and diagnostic render_commit ACK; separately timestamped owned screen readback upper bounds; no pixel onset/HMD photons/wearer readability",
        }
        rotations = [("baseline", "1.04"), ("new", "1.04"), ("new", "0.64")]
        for repetition in range(args.repetitions):
            offset = (repetition + args.rotation_offset) % 3
            rotated = rotations[offset:] + rotations[:offset]
            for mode, profile in rotated:
                for name in args.cases.split(","):
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
                            }
                        ),
                        flush=True,
                    )
        report["status"] = "executed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        await desktop.close()
        if translation:
            await translation.close()
        if owner:
            await owner.close()
        for c in captures:
            await c.scope.close()
        report["gemma_runtime_output"] = [c.output for c in captures]
        report["desktop_errors"] = desktop.errors
        if counter_process.returncode is None:
            counter_process.terminate()
        counter_stdout, counter_stderr = await counter_output
        (args.output / "gpu_counters.jsonl").write_bytes(counter_stdout)
        report["gpu_counter_observation"] = {
            "file": "gpu_counters.jsonl",
            "sha256": hashlib.sha256(counter_stdout).hexdigest(),
            "stderr": counter_stderr.decode(errors="replace"),
            "returncode": counter_process.returncode,
            "scope": "adapter memory/global engine utilization and PID-labelled memory; surrounding applications included; no GPU timestamp-query attribution",
        }
        save(args.output / "trace.json", report)
    return 0 if report["status"] == "executed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    home = Path.home()
    parser.add_argument("--route", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=home / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf",
    )
    parser.add_argument("--runtime", type=Path, default=ROOT / "build/nemotron")
    parser.add_argument(
        "--asr-root", type=Path, default=home / "AppData/Local/puripuly-heart/models"
    )
    parser.add_argument("--asr-model", default="parakeet-tdt-0.6b-v3-int8-sherpa")
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
        default=home / "Documents/dev/puripuly_heart/build/llama.cpp/llama.cpp-b10423",
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
        "--cases",
        default="clean_A_B,distinct_repeated_A_B_A,six_second_rollover,linguistic_correction,short_A_B_A,korean_read,japanese_read,synthetic_overlap",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--rotation-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not (
        args.output.parent in (ROOT / "native/nemotron", ROOT / "build")
        and args.output.name.startswith("issue218_")
    ):
        raise ValueError(
            "Output must remain task-owned native/nemotron/issue218_* or build/issue218_*"
        )
    raise SystemExit(asyncio.run(run(args)))
