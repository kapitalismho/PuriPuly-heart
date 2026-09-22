from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import hashlib
import json
import os
import signal
import sys
import time
import traceback
import wave
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from puripuly_heart.app.ports.local_asr_production_evidence import (
    LocalASRProductionEvidenceFactoryPort,
)
from puripuly_heart.app.wiring_translation_runtime_configuration import (
    replace_translation_runtime_enabled,
)
from puripuly_heart.composition.local_asr_production_evidence import (
    compose_local_asr_production_evidence,
)
from puripuly_heart.config.paths import default_settings_path
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.runtime_layout import current_runtime_layout


def _read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        facts = source.getframerate(), source.getnchannels(), source.getsampwidth()
        if facts != (16_000, 1, 2):
            raise RuntimeError(f"unexpected evidence WAV format: {facts}")
        pcm = source.readframes(source.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _console_safe(text: str, *, encoding: str | None) -> str:
    selected = encoding or "utf-8"
    return text.encode(selected, errors="backslashreplace").decode(selected)


def _event_fact(event: object) -> dict[str, object]:
    if isinstance(event, STTProviderTurnTerminal):
        return {
            "type": type(event).__name__,
            "text": event.text,
            "is_final": event.outcome == "final",
            "channel": None,
            "outcome": event.outcome,
            "text_authority": event.text_authority,
            "failure_reason": event.failure_reason,
            "final_language_runs": [
                dataclasses.asdict(item) if dataclasses.is_dataclass(item) else repr(item)
                for item in event.final_language_runs
            ],
        }
    transcript = getattr(event, "transcript", event)
    return {
        "type": type(event).__name__,
        "text": getattr(transcript, "text", None),
        "is_final": getattr(transcript, "is_final", None),
        "channel": getattr(transcript, "channel", None),
        "final_language_runs": [
            dataclasses.asdict(item) if dataclasses.is_dataclass(item) else repr(item)
            for item in getattr(transcript, "final_language_runs", ())
        ],
    }


def _is_final(event: object) -> bool:
    if isinstance(event, STTProviderTurnTerminal):
        return event.outcome == "final"
    return bool(getattr(getattr(event, "transcript", event), "is_final", False))


def _require_final(event: object, *, channel: str, stage: str) -> dict[str, object]:
    fact = _event_fact(event)
    if not fact["is_final"]:
        raise RuntimeError(f"{stage} did not return a final transcript")
    observed_channel = fact["channel"]
    if observed_channel is not None and observed_channel != channel:
        raise RuntimeError(f"{stage} returned channel {observed_channel!r}, expected {channel!r}")
    if not str(fact["text"] or "").strip():
        raise RuntimeError(f"{stage} returned empty text")
    fact["channel"] = channel
    return fact


def _snapshot_fact(owner: LocalASRProviderRuntimeOwner) -> dict[str, object]:
    snapshot = owner.snapshot
    return {
        "revision": snapshot.revision,
        "closed": snapshot.closed,
        "gpu": dataclasses.asdict(snapshot.gpu),
        "channels": [dataclasses.asdict(item) for item in snapshot.channels],
    }


def _process_present(pid: int) -> bool:
    from ctypes import wintypes

    synchronize = 0x00100000
    wait_object_0 = 0x00000000
    wait_timeout = 0x00000102
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == error_invalid_parameter:
            return False
        raise ctypes.WinError(error)
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result == wait_timeout:
            return True
        if wait_result == wait_object_0:
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


async def _wait_until(predicate, *, timeout: float) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


def _task_wait_fact(task: asyncio.Task[object]) -> dict[str, object]:
    return {
        "name": task.get_name(),
        "done": task.done(),
        "cancelled": task.cancelled(),
        "stack": [
            f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
            for frame in task.get_stack()
        ],
    }


async def _run_stage(
    report: dict[str, object],
    stage: str,
    awaitable,
    *,
    timeout: float = 60.0,
) -> object:
    report["active_stage"] = stage
    print(f"[NativeEvidence] stage={stage} state=started", flush=True)
    task = asyncio.ensure_future(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=timeout)
    if task not in done:
        wait_fact = _task_wait_fact(task)
        report["stage_timeout"] = wait_fact
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise RuntimeError(
            f"production composition stage {stage!r} exceeded {timeout:.1f}s; " f"wait={wait_fact}"
        )
    result = task.result()
    report["active_stage"] = None
    report["last_completed_stage"] = stage
    print(f"[NativeEvidence] stage={stage} state=completed", flush=True)
    return result


async def _wait_final(events: list[object], start: int) -> object:
    await _wait_until(
        lambda: any(_is_final(event) for event in events[start:]),
        timeout=240.0,
    )
    return next(event for event in events[start:] if _is_final(event))


def _channel_generation(owner: LocalASRProviderRuntimeOwner, channel: str) -> int:
    return next(item.generation for item in owner.snapshot.channels if item.channel == channel)


def _channel_pending_handoff(owner: LocalASRProviderRuntimeOwner, channel: str) -> bool:
    return next(item.pending_handoff for item in owner.snapshot.channels if item.channel == channel)


def _channel_model_id(owner: LocalASRProviderRuntimeOwner, channel: str) -> str | None:
    return next(item.model_id for item in owner.snapshot.channels if item.channel == channel)


def _channel_request(
    owner: LocalASRProviderRuntimeOwner,
    channel: str,
) -> ProviderRuntimeBuildRequest:
    request = owner._last_requests.get(channel)
    if request is None:
        raise RuntimeError(f"production {channel} request is unavailable")
    return request


def _segment_settings(request: ProviderRuntimeBuildRequest) -> AudioSegmentSettingsSnapshot:
    if request.provider_signature is None or request.runtime_signature is None:
        raise RuntimeError("production request is missing its scoped settings identity")
    config = request.config
    options = request.session_options
    source_language = options.source_language if options is not None else config.source_language
    source_mode = options.source_mode if options is not None else config.source_mode
    return AudioSegmentSettingsSnapshot(
        provider_id=request.provider_id,
        provider_signature=request.provider_signature,
        runtime_signature=request.runtime_signature,
        source_mode=source_mode,
        source_language=source_language,
        expected_languages=(source_language,),
        target_sample_rate_hz=config.sample_rate_hz,
        vad_speech_threshold=config.vad_speech_threshold,
        vad_hangover_ms=config.vad_hangover_ms,
        vad_pre_roll_ms=config.vad_pre_roll_ms,
    )


def _require_gpu_session(
    owner: LocalASRProviderRuntimeOwner,
    *,
    channel: str,
    expected_device_id: str,
    expected_pid: int | None = None,
    stage: str,
) -> int:
    gpu = owner.snapshot.gpu
    pid = gpu.worker_pid
    if pid is None:
        raise RuntimeError(f"{stage} did not start the GPU worker")
    if expected_pid is not None and pid != expected_pid:
        raise RuntimeError("production Self and Peer did not share one worker")
    if channel not in gpu.active_channels:
        raise RuntimeError(f"{stage} did not activate GPU channel {channel!r}")
    if expected_pid is not None and gpu.active_channels != frozenset({"self", "peer"}):
        raise RuntimeError("production Self and Peer residency was not shared")
    if gpu.configured_device_id != expected_device_id:
        raise RuntimeError(f"{stage} GPU device identity mismatch")
    if not gpu.model_resident:
        raise RuntimeError(f"{stage} GPU model was not resident")
    if expected_pid is not None:
        self_model = _channel_model_id(owner, "self")
        peer_model = _channel_model_id(owner, "peer")
        if self_model is None or self_model != peer_model:
            raise RuntimeError("production Self and Peer did not share one model")
    return int(pid)


def _require_activation_device(
    owner: LocalASRProviderRuntimeOwner,
    *,
    diagnostics_start: int,
    expected_device_id: str,
    stage: str,
) -> dict[str, object]:
    matches = [
        diagnostic
        for diagnostic in owner.diagnostics[diagnostics_start:]
        if diagnostic.event == "activation_ready"
    ]
    if not matches or matches[-1].device_id != expected_device_id:
        raise RuntimeError(f"{stage} resolved GPU device identity mismatch")
    return dataclasses.asdict(matches[-1])


def _owned_speech_pair(
    *,
    samples: np.ndarray,
    request: ProviderRuntimeBuildRequest,
    activation_generation: int,
) -> tuple[object, object]:
    utterance_id = uuid4()
    captured_at = time.monotonic()
    duration_s = samples.size / 16_000.0
    capture = AudioCaptureSpan(
        capture_epoch=activation_generation,
        callback_sequence=0,
        source_sample_rate_hz=16000,
        source_start_sample=0,
        source_end_sample=int(samples.size),
        source_start_monotonic_s=captured_at - duration_s,
        source_end_monotonic_s=captured_at,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=0,
        normalized_end_sample=int(samples.size),
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=activation_generation,
        settings=_segment_settings(request),
    )
    owned_start = ledger.observe_vad_event(
        SpeechStart(
            utterance_id=utterance_id,
            pre_roll=np.empty(0, np.float32),
            chunk=samples,
            chunk_capture=(capture,),
        ),
        now_monotonic_s=captured_at,
    )
    owned_end = ledger.observe_vad_event(
        SpeechEnd(utterance_id=utterance_id),
        now_monotonic_s=time.monotonic(),
    )
    return owned_start, owned_end


async def _emit_owned_vad(*, application, channel: str, owned: object) -> None:
    if channel == "self":
        await application.self_vad.handle_vad_event(owned)
        return
    await application.peer_vad.handle_peer_owned_vad_event(owned)


async def _dispatch_owned_utterance(
    *,
    application,
    channel: str,
    samples: np.ndarray,
    request: ProviderRuntimeBuildRequest,
    activation_generation: int,
) -> None:
    owned_start, owned_end = _owned_speech_pair(
        samples=samples,
        request=request,
        activation_generation=activation_generation,
    )
    await _emit_owned_vad(application=application, channel=channel, owned=owned_start)
    await _emit_owned_vad(application=application, channel=channel, owned=owned_end)


async def _wait_final_any(*groups: tuple[list[object], int]) -> object:
    def found() -> bool:
        return any(any(_is_final(event) for event in events[start:]) for events, start in groups)

    await _wait_until(found, timeout=240.0)
    for events, start in groups:
        for event in events[start:]:
            if _is_final(event):
                return event
    raise RuntimeError("scoped terminal was not observed")


async def _send_utterance(
    *,
    application,
    channel: str,
    samples: np.ndarray,
    events: list[object],
    request: ProviderRuntimeBuildRequest,
    activation_generation: int,
) -> object:
    start = len(events)
    await _dispatch_owned_utterance(
        application=application,
        channel=channel,
        samples=samples,
        request=request,
        activation_generation=activation_generation,
    )
    return await _wait_final(events, start)


async def _send_utterance_staged(
    *,
    report: dict[str, object],
    stage: str,
    application,
    channel: str,
    samples: np.ndarray,
    events: list[object],
    request: ProviderRuntimeBuildRequest,
    activation_generation: int,
) -> object:
    start = len(events)
    owned_start, owned_end = _owned_speech_pair(
        samples=samples,
        request=request,
        activation_generation=activation_generation,
    )
    await _run_stage(
        report,
        f"{stage}_speech_start",
        _emit_owned_vad(application=application, channel=channel, owned=owned_start),
    )
    await _run_stage(
        report,
        f"{stage}_speech_end",
        _emit_owned_vad(application=application, channel=channel, owned=owned_end),
    )
    return await _run_stage(
        report,
        f"{stage}_terminal",
        _wait_final(events, start),
    )


def _attach_event_evidence(
    owner: LocalASRProviderRuntimeOwner,
    self_events: list[object],
    peer_events: list[object],
    retired_events: list[object],
) -> None:
    for channel, events in (("self", self_events), ("peer", peer_events)):
        handle = owner._handles[channel]
        original = handle._event_handler

        async def event_handler(event: object, *, sink=events, delegate=original) -> None:
            sink.append(event)
            if delegate is not None:
                await delegate(event)

        handle._event_handler = event_handler
        retired = handle._retired_event_handler

        async def retired_handler(event: object, *, delegate=retired) -> None:
            retired_events.append(event)
            if delegate is not None:
                await delegate(event)

        handle._retired_event_handler = retired_handler


async def _execute(
    *,
    audio_path: Path,
    candidate: str,
    expected_gpu_name: str,
    composition_factory: LocalASRProductionEvidenceFactoryPort = (
        compose_local_asr_production_evidence
    ),
) -> dict[str, object]:
    runtime_layout = current_runtime_layout()
    if os.name != "nt" or runtime_layout.host_kind == "source":
        raise RuntimeError("production composition evidence requires the packaged Windows app")
    model_path = local_gpu_model_path()
    if not model_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError({"model": str(model_path), "audio": str(audio_path)})
    samples = _read_audio(audio_path)
    application = composition_factory(
        config_path=default_settings_path(),
    )
    settings = application.load_compatibility_settings()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(settings.intent.stt, provider="local_qwen_gpu", gpu_device_id="auto"),
            peer_stt=replace(settings.intent.peer_stt, provider="local_qwen_gpu"),
            translation=replace(settings.intent.translation, model="local_llm"),
            secrets=replace(
                settings.intent.secrets,
                backend="encrypted_file",
                encrypted_file_path="release-evidence-secrets.json",
            ),
            osc=replace(settings.intent.osc, chatbox_send=False),
        ),
    )
    os.environ["PURIPULY_HEART_SECRETS_PASSPHRASE"] = uuid4().hex
    self_events: list[object] = []
    peer_events: list[object] = []
    retired_events: list[object] = []
    report: dict[str, object] = {
        "status": "running",
        "candidate": candidate,
        "packaged": True,
        "host_kind": runtime_layout.host_kind,
        "executable": str(runtime_layout.host_executable),
        "config_path": str(application.config_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "audio": str(audio_path),
        "audio_samples": int(samples.size),
        "audio_seconds": samples.size / 16_000.0,
        "composition": {},
    }
    owner: LocalASRProviderRuntimeOwner | None = None
    try:
        await application.initialize(settings)
        owner = application.owner
        await application.llm_runtime.replace_provider(None, start=False)
        config_owner = application.translation_runtime_configuration
        replace_translation_runtime_enabled(config_owner, False)
        if application.llm_runtime.provider is not None:
            raise RuntimeError("production evidence did not disable the external LLM provider")
        report["composition"] = {
            **application.composition_facts(),
            "external_llm_disabled": True,
            "secrets_backend": settings.intent.secrets.backend,
        }
        _attach_event_evidence(owner, self_events, peer_events, retired_events)
        await application.start_runtime()
        discovery = await owner.discover_gpu(force=True)
        report["discovery"] = [dataclasses.asdict(item) for item in discovery.gpu.devices]
        physical = next(
            item
            for item in discovery.gpu.devices
            if expected_gpu_name.casefold() in f"{item.name} {item.description}".casefold()
        )
        settings = replace(
            settings,
            intent=replace(
                settings.intent,
                stt=replace(settings.intent.stt, gpu_device_id=physical.device_id),
            ),
        )
        report["selected_device"] = dataclasses.asdict(physical)

        self_request = application.build_self_provider_request(settings, warmup=True)
        await application.channel_reset.reset_provider_channel("self")
        self_result = await owner.replace_provider(self_request, start=True)
        if self_result.status != "applied":
            raise RuntimeError("production Self GPU activation failed")

        peer_request = application.build_peer_provider_request(settings, warmup=True)
        await application.channel_reset.reset_provider_channel("peer")
        peer_result = await owner.replace_provider(
            peer_request,
            start=True,
            on_terminal_failure=None,
        )
        if peer_result.status != "applied":
            raise RuntimeError("production Peer GPU activation failed")

        self_final = await _run_stage(
            report,
            "initial_self_utterance",
            _send_utterance(
                application=application,
                channel="self",
                samples=samples,
                events=self_events,
                request=self_request,
                activation_generation=_channel_generation(owner, "self"),
            ),
        )
        self_pid = _require_gpu_session(
            owner,
            channel="self",
            expected_device_id=physical.device_id,
            stage="production Self inference",
        )
        peer_final = await _run_stage(
            report,
            "initial_peer_utterance",
            _send_utterance(
                application=application,
                channel="peer",
                samples=samples,
                events=peer_events,
                request=peer_request,
                activation_generation=_channel_generation(owner, "peer"),
            ),
        )
        shared_pid = _require_gpu_session(
            owner,
            channel="peer",
            expected_device_id=physical.device_id,
            expected_pid=self_pid,
            stage="production Peer inference",
        )
        report["shared_residency"] = _snapshot_fact(owner)
        report["initial_inference"] = {
            "self": _require_final(
                self_final,
                channel="self",
                stage="production Self inference",
            ),
            "peer": _require_final(
                peer_final,
                channel="peer",
                stage="production Peer inference",
            ),
        }

        handoff_generation = _channel_generation(owner, "self")
        retired_start = len(retired_events)
        current_start = len(self_events)
        owned_start, owned_end = _owned_speech_pair(
            samples=samples,
            request=self_request,
            activation_generation=handoff_generation,
        )
        await _run_stage(
            report,
            "handoff_self_speech_start",
            _emit_owned_vad(application=application, channel="self", owned=owned_start),
        )
        handoff_task = asyncio.create_task(
            owner.handoff_provider(
                application.build_self_provider_request(settings, warmup=False),
                start=True,
            )
        )
        try:
            await _run_stage(
                report,
                "handoff_pending_boundary",
                _wait_until(
                    lambda: _channel_pending_handoff(owner, "self") or handoff_task.done(),
                    timeout=60.0,
                ),
            )
            await _run_stage(
                report,
                "handoff_in_flight_speech_end",
                _emit_owned_vad(application=application, channel="self", owned=owned_end),
            )
            handoff = await _run_stage(
                report,
                "handoff_commit",
                handoff_task,
            )
        except BaseException:
            if not handoff_task.done():
                handoff_task.cancel()
                await asyncio.gather(handoff_task, return_exceptions=True)
            raise
        if handoff.status != "applied":
            raise RuntimeError("production Self handoff failed")
        in_flight_final = await _run_stage(
            report,
            "handoff_terminal",
            _wait_final_any(
                (retired_events, retired_start),
                (self_events, current_start),
            ),
        )
        in_flight_sink = (
            "retired"
            if any(_is_final(event) for event in retired_events[retired_start:])
            else "current"
        )
        if _channel_generation(owner, "self") == handoff_generation:
            raise RuntimeError("production Self handoff did not advance generation")
        replacement_final = await _run_stage(
            report,
            "handoff_replacement_utterance",
            _send_utterance(
                application=application,
                channel="self",
                samples=samples,
                events=self_events,
                request=self_request,
                activation_generation=_channel_generation(owner, "self"),
            ),
        )
        if owner.snapshot.gpu.worker_pid != shared_pid:
            raise RuntimeError("production handoff replaced the shared worker")
        in_flight_fact = _require_final(
            in_flight_final,
            channel="self",
            stage="production in-flight handoff",
        )
        report["handoff"] = {
            "in_flight_sink": in_flight_sink,
            "in_flight_terminal_final": in_flight_fact,
            "retired_terminal_final": in_flight_fact if in_flight_sink == "retired" else None,
            "replacement_final": _require_final(
                replacement_final,
                channel="self",
                stage="production replacement handoff",
            ),
            "snapshot": _snapshot_fact(owner),
        }

        failed_pid = owner.snapshot.gpu.worker_pid
        if failed_pid is None:
            raise RuntimeError("production worker PID missing before failure probe")
        os.kill(failed_pid, signal.SIGTERM)
        await _wait_until(lambda: owner.snapshot.gpu.retry_required, timeout=30.0)
        failed_snapshot = _snapshot_fact(owner)
        recovery_diagnostics_start = len(owner.diagnostics)
        await _run_stage(
            report,
            "worker_failure_controller_recovery",
            application.retry_gpu_activation(),
        )
        controller_recovery = _snapshot_fact(owner)
        if owner.snapshot.gpu.worker_pid is not None:
            raise RuntimeError("production Controller recovery eagerly started a GPU worker")
        await _run_stage(
            report,
            "worker_failure_channel_restart",
            asyncio.gather(
                owner.start_channel("self"),
                owner.start_channel("peer"),
            ),
        )
        recovery_self_request = _channel_request(owner, "self")
        recovery_peer_request = _channel_request(owner, "peer")
        recovered_peer_final = await _send_utterance_staged(
            report=report,
            stage="worker_failure_peer_reactivation",
            application=application,
            channel="peer",
            samples=samples,
            events=peer_events,
            request=recovery_peer_request,
            activation_generation=_channel_generation(owner, "peer"),
        )
        recovered_pid = _require_gpu_session(
            owner,
            channel="peer",
            expected_device_id=recovery_peer_request.gpu_device_id,
            stage="production recovered Peer inference",
        )
        recovered_activation = _require_activation_device(
            owner,
            diagnostics_start=recovery_diagnostics_start,
            expected_device_id=physical.device_id,
            stage="production recovered Peer inference",
        )
        if recovered_pid == failed_pid:
            raise RuntimeError("production Controller recovery reused the failed worker")
        recovery_self_final = await _send_utterance_staged(
            report=report,
            stage="worker_failure_self_reactivation",
            application=application,
            channel="self",
            samples=samples,
            events=self_events,
            request=recovery_self_request,
            activation_generation=_channel_generation(owner, "self"),
        )
        _require_gpu_session(
            owner,
            channel="self",
            expected_device_id=recovery_self_request.gpu_device_id,
            expected_pid=recovered_pid,
            stage="production recovered Self inference",
        )
        report["worker_failure_recovery"] = {
            "failed_pid": failed_pid,
            "failed_pid_present_after_detection": _process_present(failed_pid),
            "failed_snapshot": failed_snapshot,
            "controller_recovery": controller_recovery,
            "recovered_pid": recovered_pid,
            "requested_device_id": recovery_peer_request.gpu_device_id,
            "resolved_activation": recovered_activation,
            "recovered_self_final": _require_final(
                recovery_self_final,
                channel="self",
                stage="production recovered Self inference",
            ),
            "recovered_peer_final": _require_final(
                recovered_peer_final,
                channel="peer",
                stage="production recovered Peer inference",
            ),
            "recovered_snapshot": _snapshot_fact(owner),
        }

        await application.channel_reset.reset_provider_channel("self")
        await owner.release_channel("self", mode="abort")
        after_self = _snapshot_fact(owner)
        if owner.snapshot.gpu.active_channels != frozenset({"peer"}):
            raise RuntimeError("production Self release did not retain Peer")
        await application.channel_reset.reset_provider_channel("peer")
        await owner.release_channel("peer", mode="abort")
        after_peer = _snapshot_fact(owner)
        if owner.snapshot.gpu.active_channels or owner.snapshot.gpu.worker_pid is not None:
            raise RuntimeError("production final channel release left GPU resources")
        recovered_pid_present = _process_present(recovered_pid)
        if recovered_pid_present:
            raise RuntimeError("production final channel release left the recovered worker running")
        report["resource_release"] = {
            "after_self": after_self,
            "after_peer": after_peer,
            "recovered_pid_present_after_last_release": recovered_pid_present,
        }
        report["status"] = "passed"
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            await application.close()
        except Exception as exc:
            report["shutdown_failure"] = {
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "traceback": traceback.format_exc(),
            }
            report["status"] = "failed"
        owner_snapshot = _snapshot_fact(owner) if owner is not None else None
        remaining_named_tasks = sorted(
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and any(
                token in task.get_name().casefold()
                for token in ("gpu", "local-asr", "provider-runtime")
            )
        )
        shutdown_passed = bool(
            owner_snapshot is not None
            and owner_snapshot["closed"]
            and not remaining_named_tasks
            and "shutdown_failure" not in report
        )
        report["shutdown"] = {
            "passed": shutdown_passed,
            "owner_snapshot": owner_snapshot,
            "remaining_named_tasks": remaining_named_tasks,
        }
        if report["status"] == "passed" and not shutdown_passed:
            report.update(
                {
                    "status": "failed",
                    "failure_type": "RuntimeError",
                    "failure": "production composition shutdown did not release all owners and tasks",
                }
            )
        if owner is not None:
            report["diagnostics"] = [
                dataclasses.asdict(diagnostic) for diagnostic in owner.diagnostics
            ]
    return report


def run_local_asr_production_composition(
    *,
    audio_path: Path,
    report_path: Path,
    candidate: str,
    expected_gpu_name: str,
) -> int:
    started_at = time.monotonic()
    try:
        report = asyncio.run(
            _execute(
                audio_path=audio_path,
                candidate=candidate,
                expected_gpu_name=expected_gpu_name,
            )
        )
    except BaseException as exc:
        report = {
            "status": "failed",
            "candidate": candidate,
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    else:
        exit_code = 0 if report.get("status") == "passed" else 1
    report["elapsed_seconds"] = time.monotonic() - started_at
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    report_path.write_text(rendered, encoding="utf-8")
    print(_console_safe(rendered, encoding=getattr(sys.stdout, "encoding", None)))
    return exit_code
