from __future__ import annotations

import argparse
import asyncio
import base64
from collections.abc import Mapping
from contextlib import ExitStack
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
from uuid import UUID, uuid5, NAMESPACE_URL
import wave

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soxr

from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.composition.local_asr_production_evidence import compose_local_asr_production_evidence
from puripuly_heart.app.wiring_translation_runtime_configuration import replace_translation_runtime_enabled
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.format import AudioCaptureSpan, float32_to_pcm16le_bytes, pcm16le_bytes_to_float32
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTProviderTurnUpdate
from puripuly_heart.core.runtime_logging import configure_main_logging
from puripuly_heart.core.translation_backend import LlmTranslationBackend
from puripuly_heart.core.vad.gating import SpeechStart, SpeechChunk, SpeechEnd
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
from puripuly_heart.core.local_stt_assets import PARAKEET_V3_MODEL_ID, default_local_stt_model_dir, load_local_stt_asset_manifest
from puripuly_heart.release_evidence.local_asr_production_composition import _segment_settings, _channel_generation

HERE = Path(__file__).resolve().parent
CASES = (
    ("continuation", "I will carry the lantern", "after sunset."),
    ("number_negation", "We need fifteen boxes.", "Not fifty boxes."),
    ("question", "The gate is open.", "Is it safe to enter?"),
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, (UUID, Path)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_samples(path):
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise RuntimeError("Frozen speech must be mono 16 kHz PCM16")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


def generate():
    if (HERE / "inputs.json").exists():
        raise RuntimeError("Frozen inputs already exist; generation never replaces them")
    entries = []
    for case, first, second in CASES:
        for index, phrase in enumerate((first, second)):
            label = f"{case}_{index}"
            path = HERE / f"{label}.wav"
            literal = str(path).replace("'", "''")
            script = (
                "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); "
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "try { $v = $s.GetInstalledVoices() | Where-Object { "
                "$_.Enabled -and $_.VoiceInfo.Culture.Name -eq 'en-US' } | Select-Object -First 1; "
                "if ($null -eq $v) { throw 'No installed offline en-US voice' }; "
                "$s.SelectVoice($v.VoiceInfo.Name); $s.Rate = 0; $s.Volume = 100; "
                f"$s.SetOutputToWaveFile('{literal}'); "
                f"$s.Speak('{phrase.replace(chr(39), chr(39) * 2)}'); "
                "@{Name=$v.VoiceInfo.Name; Culture=$v.VoiceInfo.Culture.Name; Gender=$v.VoiceInfo.Gender.ToString(); Age=$v.VoiceInfo.Age.ToString()} | ConvertTo-Json -Compress "
                "} finally { $s.Dispose() }"
            )
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], capture_output=True, timeout=30, check=True)
            voice = json.loads(result.stdout.decode("utf-8-sig"))
            with wave.open(str(path), "rb") as source:
                rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
                if width != 2 or channels not in (1, 2):
                    raise RuntimeError("SAPI did not produce PCM16")
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768
            samples = soxr.resample(samples, rate, 16000, quality="HQ")
            pcm = np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2")
            with wave.open(str(path), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(16000)
                sink.writeframes(pcm.tobytes())
            entries.append({"label": label, "case": case, "phrase": phrase, "file": path.name, "sha256": digest(path), "samples": len(pcm), "seconds": len(pcm) / 16000, "voice": voice, "sapi_rate": 0, "sapi_volume": 100, "sapi_source_rate_hz": rate})
    save(HERE / "inputs.json", {"provenance": "Public invented synthetic English speech; offline Windows System.Speech SAPI; no microphone, private audio, downloads or remote speech provider", "format": "mono 16000 Hz PCM16 WAV", "normalization": "soxr HQ then PCM16 rounding; no trimming or VAD tuning", "episodes": entries, "within_case_pause_s": 0.18, "between_case_pause_s": 3.0})


class Probe:
    def __init__(self):
        self.origin = time.monotonic()
        self.events = []
        self.labels = {}

    def record(self, event, **fields):
        now = time.monotonic()
        self.events.append({"event": event, "at_s": now - self.origin, "receipt_monotonic_s": now, **plain(fields)})

    def state(self, owner, buffer=None):
        buffer = buffer or owner.merge_buffer
        if buffer is None:
            return None
        attempt = buffer.speculative_attempt
        start = buffer.finalize_wait_started_at
        return {"merge_id": str(buffer.merge_id), "publication_ids": [str(item) for item in buffer.utterance_ids], "acoustic_labels": [self.labels.get(str(item), str(item)) for item in buffer.utterance_ids], "text": owner._merge_text(buffer.parts), "resume_pending": buffer.resume_pending, "resume_confirmed": buffer.resume_confirmed, "resume_id": plain(buffer.resume_utterance_id), "awaiting_vad_end": buffer.awaiting_vad_end, "grace_live": buffer.finalize_wait_task is not None, "grace_start_monotonic_s": start, "grace_deadline_monotonic_s": None if start is None else start + owner.config_snapshot().value.low_latency_finalize_wait_ms / 1000, "blocker": owner._continuation_blocker(buffer), "attempt": None if attempt is None else {"sequence": attempt.sequence, "status": attempt.status.value, "source_text": attempt.source_text, "provider_generation": attempt.provider_generation, "started_at": attempt.started_at, "completed_at": attempt.completed_at, "terminal_action_started": attempt.terminal_action_started}}

    def attach(self, application, stack):
        owner = application.self_vad
        handle = application.owner._handles["self"]
        delegate = handle._event_handler

        async def ingress(event):
            if isinstance(event, (STTProviderTurnTerminal, STTProviderTurnUpdate)):
                self.record("asr_terminal" if isinstance(event, STTProviderTurnTerminal) else "asr_update", label=self.labels.get(str(event.identity.segment.segment_id)), payload=event, state=self.state(owner), source_availability_clock="not exposed by this provider; receipt is first observer boundary")
            if delegate is not None:
                await delegate(event)
            if isinstance(event, (STTProviderTurnTerminal, STTProviderTurnUpdate)):
                self.record("asr_owner_applied", label=self.labels.get(str(event.identity.segment.segment_id)), state=self.state(owner))
        handle._event_handler = ingress
        grace_delegate = type(owner)._restart_post_end_grace

        def grace(owner, buffer):
            before = self.state(owner, buffer)
            grace_delegate(owner, buffer)
            self.record("grace_start_or_rearm", before=before, state=self.state(owner, buffer))
        stack.enter_context(patch.object(type(owner), "_restart_post_end_grace", grace))
        evaluate_delegate = type(owner)._evaluate_speculative_next_action

        async def evaluate(owner, buffer, *, reason):
            self.record("eligibility", reason=reason, state=self.state(owner, buffer))
            await evaluate_delegate(owner, buffer, reason=reason)
        stack.enter_context(patch.object(type(owner), "_evaluate_speculative_next_action", evaluate))
        commit_delegate = type(owner)._commit_merge

        async def commit(owner, buffer, *, reason):
            before = self.state(owner, buffer)
            self.record("commit_requested", reason=reason, state=before)
            await commit_delegate(owner, buffer, reason=reason)
            if owner.merge_buffer is not buffer:
                self.record("source_committed", reason=reason, state=before, runtime_bundle=owner.runtime.utterances.get(buffer.merge_id))
        stack.enter_context(patch.object(type(owner), "_commit_merge", commit))


class ControlledLLM:
    def __init__(self, probe, delay):
        self.probe = probe
        self.delay = delay
        self.invocations = 0

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.invocations += 1
        invocation = self.invocations
        self.probe.record("llm_invoked", invocation=invocation, utterance_id=utterance_id, text=text, target_language=target_language, context=context, delay_s=self.delay)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.probe.record("llm_cancelled", invocation=invocation, utterance_id=utterance_id)
            raise
        self.probe.record("llm_result_ready", invocation=invocation, utterance_id=utterance_id)
        return Translation(utterance_id=utterance_id, text=f"Controlled {target_language} boundary result {invocation}")

    async def close(self):
        self.probe.record("llm_closed")


class ObservedPresenter:
    def __init__(self, probe, clock):
        self.probe = probe
        self.presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)

    async def emit(self, event):
        receipt = await self.presenter.emit(event)
        self.probe.record("overlay_application_receipt", payload=event, receipt=receipt)
        return receipt

    def active_self_overlay_metadata(self):
        return self.presenter.active_self_overlay_metadata()

    def application_receipt(self, publication_id):
        return self.presenter.application_receipt(publication_id)


async def until(predicate, timeout=120):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def execute(args):
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    episodes = inputs["episodes"]
    for entry in episodes:
        if digest(HERE / entry["file"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen input hash mismatch: {entry['file']}")
    probe = Probe()
    application = None
    presenter = None
    logging_sinks = None
    report = {"schema": "self-grace-233/production-synthetic-audio/v1", "status": "running", "policy": "unchanged production baseline", "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner_sha256": digest(Path(__file__)), "interpreter": sys.executable, "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("numpy", "sherpa-onnx", "onnxruntime", "soxr", "sounddevice", "flet")}, "inputs_manifest_sha256": digest(HERE / "inputs.json"), "inputs": inputs, "clock": {"source": "time.monotonic", "receipt": "time.monotonic", "owner": "SystemClock.now = time.monotonic", "real_asyncio_timers": True, "origin_monotonic_s": probe.origin}, "limits": ["Owned synthetic WAV ingress, not a microphone/device/VAD detector measurement", "Fresh local CPU final-only ASR; no stable-contribution provider coverage", "LLM is a controlled scheduling boundary, not a translation quality evaluation", "Overlay receipt is application acceptance through existing presenter, not GUI/OSC/HMD visibility", "Replay of these observed events is not a new live-provider result", "No candidate/default policy mutation"]}
    report["source_files"] = {name: digest(ROOT / name) for name in ("src/puripuly_heart/core/orchestrator/self_translation_channel.py", "src/puripuly_heart/composition/local_asr_production_evidence.py", "src/puripuly_heart/app/wiring/wiring_runtime_pipeline.py", "src/puripuly_heart/providers/stt/local_parakeet_sherpa.py", "src/puripuly_heart/providers/stt/local_qwen_sherpa.py", "src/puripuly_heart/core/stt/scoped_engine.py", "src/puripuly_heart/core/runtime/output.py")}
    settings = AppSettingsVNext()
    settings = dataclasses.replace(settings, intent=dataclasses.replace(settings.intent, stt=dataclasses.replace(settings.intent.stt, provider="local_parakeet_v3", custom_vocabulary_enabled=False), peer_stt=dataclasses.replace(settings.intent.peer_stt, provider="local_parakeet_v3"), languages=dataclasses.replace(settings.intent.languages, source_language="en", target_language="ja", secondary_target_language=""), translation=dataclasses.replace(settings.intent.translation, model="custom_http", http_extension_id="self-grace-233-controlled-only", concurrency_limit=2), secrets=dataclasses.replace(settings.intent.secrets, backend="encrypted_file", encrypted_file_path="unused-isolated-secrets.json"), telemetry=dataclasses.replace(settings.intent.telemetry, enabled=False), osc=dataclasses.replace(settings.intent.osc, connection_mode="off", chatbox_send=False, activation_notice_enabled=False)))
    config_path = HERE / "isolated-settings.json"
    save(config_path, settings)
    report["isolated_profile"] = {"file": config_path.name, "sha256": digest(config_path), "canonical_profile_loaded": False, "secrets_store": "isolated nonexistent encrypted_file; transient invented passphrase", "llm_factory": "controlled boundary injected before construction; no external LLM instance"}
    llm = ControlledLLM(probe, args.llm_delay)
    backend = LlmTranslationBackend(llm)
    audio_hash_labels = {hashlib.sha256(pcm16le_bytes_to_float32(float32_to_pcm16le_bytes(load_samples(HERE / entry["file"]))).tobytes()).hexdigest(): entry["label"] for entry in episodes}
    report["decode_input_f32_sha256"] = {label: value for value, label in audio_hash_labels.items()}
    report["audio_boundary_conversion"] = "Existing ScopedRecognitionEngine float32_to_pcm16le_bytes rounds samples * 32767, then local session decodes PCM16 / 32768"
    native_delegate = LocalQwenSherpaSTTBackend._decode_f32_sync

    def native_decode(backend, recognizer, samples):
        sample_sha256 = hashlib.sha256(np.asarray(samples, dtype=np.float32).tobytes()).hexdigest()
        label = audio_hash_labels.get(sample_sha256)
        probe.record("native_asr_invoked", label=label, samples=len(samples), decode_f32_sha256=sample_sha256, backend=type(backend).__name__, num_threads=backend.num_threads, execution_provider=backend.provider)
        try:
            result = native_delegate(backend, recognizer, samples)
        except BaseException as exc:
            probe.record("native_asr_failed", label=label, error_type=type(exc).__name__)
            raise
        probe.record("native_asr_result_ready", label=label, text=result)
        return result

    os.environ["PURIPULY_HEART_SECRETS_PASSPHRASE"] = "invented-task-local-not-a-user-secret-233"
    try:
        model_dir = default_local_stt_model_dir(PARAKEET_V3_MODEL_ID)
        manifest = load_local_stt_asset_manifest(PARAKEET_V3_MODEL_ID)
        report["model"] = {"id": manifest.model_id, "directory": str(model_dir), "installed_manifest": json.loads((model_dir / manifest.installed_manifest_filename).read_text(encoding="utf-8-sig")), "files": [{"file": asset.relative_path, "sha256": digest(model_dir / asset.relative_path), "expected_sha256": asset.sha256} for asset in manifest.files]}
        if any(item["sha256"] != item["expected_sha256"] for item in report["model"]["files"]):
            raise RuntimeError("Installed local CPU model checksum mismatch")
        with ExitStack() as stack, patch("puripuly_heart.app.wiring.wiring_runtime_pipeline.create_translation_backend", return_value=backend), patch.object(LocalQwenSherpaSTTBackend, "_decode_f32_sync", native_decode):
            logging_sinks = configure_main_logging(log_dir=HERE / "runtime-logs")
            stack.enter_context(patch("puripuly_heart.composition.local_asr_production_evidence.compose_application_runtime", side_effect=lambda **kwargs: compose_application_runtime(runtime_logging_sinks=logging_sinks, **kwargs)))
            report["logging"] = {"log_file": str(logging_sinks.log_file), "scope": "task-local audio/runtime-logs; explicit existing runtime logging sink"}
            application = compose_local_asr_production_evidence(config_path=config_path)
            await application.initialize(settings)
            if application.llm_runtime.provider is not backend:
                await application.llm_runtime.replace_provider(backend, start=False)
            replace_translation_runtime_enabled(application.translation_runtime_configuration, True)
            if application.llm_runtime.provider is not backend or not application.translation_runtime_configuration.snapshot().value.translation_enabled:
                raise RuntimeError("Controlled translation boundary was not enabled")
            probe.attach(application, stack)
            owner = application.self_vad
            report["composition"] = application.composition_facts()
            report["effective_translation_profile"] = plain(application.translation_runtime_configuration.snapshot())
            report["effective_asr_request"] = plain(application.owner._last_requests["self"])
            await application.start_runtime()
            presenter = ObservedPresenter(probe, owner.clock)
            await owner.output_projection.output_runtime.replace_overlay_sink(presenter)
            generation = _channel_generation(application.owner, "self")
            request = application.owner._last_requests["self"]
            ledger = PeerAudioSegmentLedger(activation_generation=generation, settings=_segment_settings(request))
            source_cursor = 0
            sequence = 0
            scheduled_at = time.monotonic() + 0.1
            for index, entry in enumerate(episodes):
                samples = load_samples(HERE / entry["file"])
                utterance_id = uuid5(NAMESPACE_URL, f"self-grace-233/{entry['label']}")
                probe.labels[str(utterance_id)] = entry["label"]
                episode_start = scheduled_at
                for offset in range(0, len(samples), 1600):
                    chunk = samples[offset:offset + 1600]
                    begin = episode_start + offset / 16000
                    end = begin + len(chunk) / 16000
                    await asyncio.sleep(max(0, end - time.monotonic()))
                    capture = AudioCaptureSpan(capture_epoch=generation, callback_sequence=sequence, source_sample_rate_hz=16000, source_start_sample=source_cursor + offset, source_end_sample=source_cursor + offset + len(chunk), source_start_monotonic_s=begin, source_end_monotonic_s=end, normalized_sample_rate_hz=16000, normalized_start_sample=source_cursor + offset, normalized_end_sample=source_cursor + offset + len(chunk))
                    event = SpeechStart(utterance_id=utterance_id, pre_roll=np.empty(0, np.float32), chunk=chunk, chunk_capture=(capture,)) if offset == 0 else SpeechChunk(utterance_id=utterance_id, chunk=chunk, chunk_capture=(capture,))
                    owned = ledger.observe_vad_event(event, now_monotonic_s=time.monotonic())
                    probe.record("vad_start" if offset == 0 else "vad_chunk", label=entry["label"], utterance_id=utterance_id, source=owned.segment, source_schedule_at_s=end - probe.origin)
                    await application.self_vad.handle_vad_event(owned)
                    sequence += 1
                seal = episode_start + len(samples) / 16000
                owned = ledger.observe_vad_event(SpeechEnd(utterance_id=utterance_id), now_monotonic_s=seal)
                probe.record("vad_end", label=entry["label"], utterance_id=utterance_id, source=owned.segment, source_schedule_at_s=seal - probe.origin, state=probe.state(owner))
                await application.self_vad.handle_vad_event(owned)
                source_cursor += len(samples)
                scheduled_at = seal + (inputs["within_case_pause_s"] if index % 2 == 0 else inputs["between_case_pause_s"])
            await until(lambda: sum(event["event"] == "asr_terminal" for event in probe.events) == len(episodes))
            await until(lambda: owner.merge_buffer is None and not owner.runtime.translation_tasks)
            await owner.output_projection.output_runtime.wait_for_peer_output_idle()
            await asyncio.sleep(0.1)
            report["asr_runtime_snapshot"] = plain(application.owner.snapshot)
            report["context_history"] = plain(owner.runtime.translation_history)
            terminals = [event for event in probe.events if event["event"] == "asr_terminal"]
            commits = [event for event in probe.events if event["event"] == "source_committed"]
            report["counts"] = {"input_cases": len(CASES), "input_episodes": len(episodes), "asr_native_invocations": sum(event["event"] == "native_asr_invoked" and event["label"] is not None for event in probe.events), "asr_terminals": len(terminals), "stable_updates": sum(event["event"] == "asr_update" for event in probe.events), "source_commits": len(commits), "llm_invocations": llm.invocations, "llm_ready": sum(event["event"] == "llm_result_ready" for event in probe.events), "llm_cancelled": sum(event["event"] == "llm_cancelled" for event in probe.events)}
            emitted_labels = [label for event in commits for label in event["state"]["acoustic_labels"]]
            expected_labels = [entry["label"] for entry in episodes]
            report["source_accounting"] = {"expected_labels": expected_labels, "committed_labels": emitted_labels, "omitted": sorted(set(expected_labels) - set(emitted_labels)), "duplicated": sorted(label for label in set(emitted_labels) if emitted_labels.count(label) > 1)}
            report["status"] = "passed" if all(event["payload"]["outcome"] == "final" and event["payload"]["text"].strip() for event in terminals) and sorted(emitted_labels) == sorted(expected_labels) and report["counts"]["asr_native_invocations"] == len(episodes) and llm.invocations > 0 else "failed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if application is not None:
            try:
                await asyncio.wait_for(application.close(), timeout=30)
                probe.record("production_runtime_closed")
            except BaseException as exc:
                report["status"] = "failed"
                report["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
        if presenter is not None:
            await presenter.presenter.close()
        if logging_sinks is not None:
            logging_sinks.close()
        report["events"] = probe.events
        save(HERE / args.report, report)
        asr_schedule = [{"event": event["event"], "at_s": event["at_s"], "label": event.get("label"), **({"payload": event["payload"]} if "payload" in event and event["event"] in ("asr_terminal", "asr_update") else {}), **({"source": event["source"], "utterance_id": event["utterance_id"], "source_schedule_at_s": event["source_schedule_at_s"]} if event["event"].startswith("vad_") else {})} for event in probe.events if event["event"] in ("vad_start", "vad_chunk", "vad_end", "asr_terminal", "asr_update")]
        save(HERE / (Path(args.report).stem + "_asr_schedule.json"), {"schema": "self-grace-233/recorded-asr-schedule/v1", "origin_monotonic_s": probe.origin, "report": args.report, "inputs_manifest_sha256": report["inputs_manifest_sha256"], "source_files": report["source_files"], "events": asr_schedule, "replay_limit": "Freeze these observed arrival times and source VAD seal times; map identities consistently and inject through owned/scoped ingress in the existing deterministic fixture; replay is not fresh ASR"})
    print(json.dumps({"status": report["status"], "report": args.report, "counts": report.get("counts"), "error": report.get("error"), "close_error": report.get("close_error")}))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--report", default="baseline.json")
    parser.add_argument("--llm-delay", type=float, default=0.05)
    args = parser.parse_args()
    if Path(args.report).name != args.report or args.llm_delay < 0:
        parser.error("Report must be task-local filename and delay nonnegative")
    if args.generate:
        generate()
    return asyncio.run(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
