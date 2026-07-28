from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

from puripuly_heart.app.ports.local_asr_production_evidence import (
    LocalASRProductionEvidenceFactoryPort,
)
from puripuly_heart.composition.local_asr_production_evidence import (
    compose_local_asr_production_evidence,
)
from puripuly_heart.config.paths import default_settings_path
from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart


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


def _event_fact(event: object) -> dict[str, object]:
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
    return bool(getattr(getattr(event, "transcript", event), "is_final", False))


def _require_final(event: object, *, channel: str, stage: str) -> dict[str, object]:
    fact = _event_fact(event)
    if not fact["is_final"]:
        raise RuntimeError(f"{stage} did not return a final transcript")
    if fact["channel"] != channel:
        raise RuntimeError(f"{stage} returned channel {fact['channel']!r}, expected {channel!r}")
    if not str(fact["text"] or "").strip():
        raise RuntimeError(f"{stage} returned empty text")
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
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    return f'"{pid}"' in result.stdout


async def _wait_until(predicate, *, timeout: float) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


async def _wait_final(events: list[object], start: int) -> object:
    await _wait_until(
        lambda: any(_is_final(event) for event in events[start:]),
        timeout=240.0,
    )
    return next(event for event in events[start:] if _is_final(event))


async def _send_utterance(
    *,
    hub,
    channel: str,
    samples: np.ndarray,
    events: list[object],
) -> object:
    start = len(events)
    utterance_id = uuid4()
    speech_start = SpeechStart(
        utterance_id=utterance_id,
        pre_roll=np.empty(0, np.float32),
        chunk=samples,
    )
    speech_end = SpeechEnd(utterance_id=utterance_id)
    if channel == "self":
        await hub.handle_vad_event(speech_start)
        await hub.handle_vad_event(speech_end)
    else:
        await hub.handle_peer_vad_event(speech_start)
        await hub.handle_peer_vad_event(speech_end)
    return await _wait_final(events, start)


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
    if os.name != "nt" or not getattr(sys, "frozen", False):
        raise RuntimeError("production composition evidence requires the packaged Windows app")
    model_path = local_gpu_model_path()
    if not model_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError({"model": str(model_path), "audio": str(audio_path)})
    samples = _read_audio(audio_path)
    application = composition_factory(
        config_path=default_settings_path(),
    )
    settings = application.load_compatibility_settings()
    provider_type = type(settings.provider.stt)
    settings.provider.stt = provider_type("local_qwen_gpu")
    settings.provider.peer_stt = provider_type("local_qwen_gpu")
    settings.provider.llm = type(settings.provider.llm)("local_llm")
    settings.secrets.backend = type(settings.secrets.backend)("encrypted_file")
    settings.secrets.encrypted_file_path = "release-evidence-secrets.json"
    settings.stt.gpu_device_id = "auto"
    settings.ui.peer_translation_enabled = False
    settings.osc.chatbox_send = False
    os.environ["PURIPULY_HEART_SECRETS_PASSPHRASE"] = uuid4().hex
    self_events: list[object] = []
    peer_events: list[object] = []
    retired_events: list[object] = []
    report: dict[str, object] = {
        "status": "running",
        "candidate": candidate,
        "packaged": True,
        "executable": sys.executable,
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
        hub = application.hub
        owner = application.owner
        await hub.replace_llm_provider(None)
        hub.translation_enabled = False
        if hub.llm is not None:
            raise RuntimeError("production evidence did not disable the external LLM provider")
        report["composition"] = {
            **application.composition_facts(),
            "external_llm_disabled": True,
            "secrets_backend": settings.secrets.backend.value,
        }
        _attach_event_evidence(owner, self_events, peer_events, retired_events)
        await hub.start()
        discovery = await owner.discover_gpu(force=True)
        report["discovery"] = [dataclasses.asdict(item) for item in discovery.gpu.devices]
        physical = next(
            item
            for item in discovery.gpu.devices
            if expected_gpu_name.casefold() in f"{item.name} {item.description}".casefold()
        )
        settings.stt.gpu_device_id = physical.device_id
        report["selected_device"] = dataclasses.asdict(physical)

        self_request = application.build_self_provider_request(settings, warmup=True)
        self_result = await hub.replace_stt_provider_request(self_request, start=True)
        if self_result.status != "applied":
            raise RuntimeError("production Self GPU activation failed")
        self_pid = owner.snapshot.gpu.worker_pid

        peer_request = application.build_peer_provider_request(settings, warmup=True)
        peer_result = await hub.replace_peer_stt_provider_request(
            peer_request,
            start=True,
            on_terminal_failure=None,
        )
        if peer_result.status != "applied":
            raise RuntimeError("production Peer GPU activation failed")
        shared_pid = owner.snapshot.gpu.worker_pid
        if self_pid is None or shared_pid != self_pid:
            raise RuntimeError("production Self and Peer did not share one worker")
        if owner.snapshot.gpu.active_channels != frozenset({"self", "peer"}):
            raise RuntimeError("production Self and Peer residency was not shared")
        report["shared_residency"] = _snapshot_fact(owner)

        self_final = await _send_utterance(
            hub=hub,
            channel="self",
            samples=samples,
            events=self_events,
        )
        peer_final = await _send_utterance(
            hub=hub,
            channel="peer",
            samples=samples,
            events=peer_events,
        )
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

        retired_start = len(retired_events)
        utterance_id = uuid4()
        await hub.handle_vad_event(
            SpeechStart(
                utterance_id=utterance_id,
                pre_roll=np.empty(0, np.float32),
                chunk=samples,
            )
        )
        await hub.handle_vad_event(SpeechEnd(utterance_id=utterance_id))
        handoff = await hub.handoff_stt_provider_request(
            application.build_self_provider_request(settings, warmup=False),
            start=True,
        )
        if handoff.status != "applied":
            raise RuntimeError("production Self handoff failed")
        retired_final = await _wait_final(retired_events, retired_start)
        replacement_final = await _send_utterance(
            hub=hub,
            channel="self",
            samples=samples,
            events=self_events,
        )
        if owner.snapshot.gpu.worker_pid != shared_pid:
            raise RuntimeError("production handoff replaced the shared worker")
        report["handoff"] = {
            "retired_terminal_final": _require_final(
                retired_final,
                channel="self",
                stage="production retired handoff",
            ),
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
        await application.retry_gpu_activation()
        controller_recovery = _snapshot_fact(owner)
        controller_recovered_pid = owner.snapshot.gpu.worker_pid
        if controller_recovered_pid is None or controller_recovered_pid == failed_pid:
            raise RuntimeError("production Controller recovery did not start a fresh worker")
        await hub.resume_self_stt_after_toggle_on()
        peer_reactivation_status = "retained"
        if "peer" not in owner.snapshot.gpu.active_channels:
            peer_reactivation = await hub.replace_peer_stt_provider_request(
                peer_request,
                start=True,
                on_terminal_failure=None,
            )
            peer_reactivation_status = peer_reactivation.status
            if peer_reactivation.status != "applied":
                raise RuntimeError("production Peer reactivation after recovery failed")
        else:
            await hub.start_peer_stt_provider_ingress()
        recovered_pid = owner.snapshot.gpu.worker_pid
        if recovered_pid != controller_recovered_pid:
            raise RuntimeError("production Peer reactivation replaced the recovered worker")
        recovered_final = await _send_utterance(
            hub=hub,
            channel="peer",
            samples=samples,
            events=peer_events,
        )
        report["worker_failure_recovery"] = {
            "failed_pid": failed_pid,
            "failed_pid_present_after_detection": _process_present(failed_pid),
            "failed_snapshot": failed_snapshot,
            "recovered_pid": recovered_pid,
            "controller_recovery": controller_recovery,
            "peer_reactivation_status": peer_reactivation_status,
            "recovered_final": _require_final(
                recovered_final,
                channel="peer",
                stage="production recovered Peer inference",
            ),
            "recovered_snapshot": _snapshot_fact(owner),
        }

        await hub.abort_self_stt_for_toggle_off()
        after_self = _snapshot_fact(owner)
        if owner.snapshot.gpu.active_channels != frozenset({"peer"}):
            raise RuntimeError("production Self release did not retain Peer")
        await hub.abort_peer_stt_for_toggle_off()
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
    print(rendered)
    return exit_code
