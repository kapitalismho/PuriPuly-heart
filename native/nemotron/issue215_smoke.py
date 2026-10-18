"""Focused issue #215 real-model probe; no capture, network or product settings mutation.

Run from repository root with .venv Python. Baseline means the ordinary scoped
CPU batch adapter; new means CausalBatchSession plus real SpeakerRuntime evidence.
All boundaries come from production Silero/delivery or actual model evidence.
Translation execution is deliberately source-only; request preparation/context
admission use their real owners, with a non-executing provider availability port.
This is not a renderer, paid-provider, profile selection or accuracy benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from probe import audio

from puripuly_heart.app.adapters.nemotron import NemotronBackend, inspect_artifact, validate_runtime
from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.orchestrator.peer_translation_channel import PeerTranslationChannelOwner
from puripuly_heart.core.orchestrator.translation_request import TranslationRequestOwner
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnLifecycleOwner,
    TranslationTurnRequest,
)
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.speaker import SpeakerRuntime
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.recognition_units import contribution_transcript
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend


def serial(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: serial(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (UUID, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"samples": value.size}
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_cases(args):
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["privacy"]["public_audio_only"] or manifest["privacy"]["private_user_speech"]:
        raise ValueError("probe requires the authorized public-only corpus")
    approved = {
        "A": "19865499680804b33e8ac1f3a7b96fa8762e31be6a143c9e711bba72984272ad",
        "B": "cccded3f9d5a5bb4c28110b8273e217cc44d5d9e9de95eb70fd19208dce9b4f5",
        "long": "e5664ae10a6273fbe0009068e512f4c416cccbeff19d3d2d11e5fb8d72d03fb7",
        "local_no_trim": "e474e0808b047ccbc2a678a22531adc9599cd07a2a5fccea0e7be8047b777c49",
        "local_correction_trim": "34a41d87d8752cc290945044af90309c36d7076c0a1f236394a095a78d7eadac",
    }
    names = {
        "A": "clips/common-voice-17.0/en/common_voice_en_11289258.mp3",
        "B": "clips/common-voice-17.0/en/common_voice_en_12274.mp3",
        "long": "clips/fleurs/en_us/1009709090964908274.wav",
    }
    inputs, pcm = {}, {}
    for name, relative in names.items():
        source = next(
            t["source"]
            for t in manifest["traces"]
            if t["source"]["relative_audio_path"] == relative
        )
        path = args.corpus / relative
        sha = digest(path)
        if sha != source["source_audio_sha256"] or sha != approved[name]:
            raise ValueError(f"public input hash mismatch: {relative}")
        inputs[name] = {k: v for k, v in source.items() if k != "speaker_id"}
        inputs[name]["verified_sha256"] = sha
        pcm[name] = audio(path)
    old = json.loads((args.correction_root / "local-results-real.json").read_text(encoding="utf-8"))
    for name in ("local_no_trim", "local_correction_trim"):
        path = args.correction_root / "audio" / f"{name}.wav"
        expected = old["audio"][name]
        if digest(path) != expected["sha256"] or expected["sha256"] != approved[name]:
            raise ValueError(f"correction hash mismatch: {name}")
        inputs[name] = {
            **expected,
            "path": f"issue212/audio/{name}.wav",
            "voice": "Microsoft Zira Desktop",
            "kind": "authorized synthetic linguistic speech, not natural conversation",
        }
        inputs[name]["source_text"] = "No." if name == "local_no_trim" else "No, three."
        pcm[name] = audio(path)

    def crop(name):
        # Input preparation only; not a boundary oracle. Keep 160 ms on each side
        # of >0.01 waveform support and document exact crop; no word alignment.
        values = pcm[name]
        active = np.flatnonzero(np.abs(values) > 0.01)
        left, right = max(0, int(active[0]) - 2560), min(values.size, int(active[-1]) + 2561)
        return name, left, right

    def full(name):
        return name, 0, pcm[name].size

    cases = {
        "ordinary_onset": [full("A")],
        "six_second_rollover": [full("long")],
        "clean_A_B": [crop("A"), crop("B")],
        "short_A_B_A": [crop("A"), full("local_no_trim"), crop("A")],
        "distinct_repeated_A_B_A": [crop("A"), crop("B"), crop("A")],
        "linguistic_correction": [full("local_no_trim"), full("local_correction_trim")],
    }
    assembled = {}
    for name, pieces in cases.items():
        blocks, mapping, cursor = [np.zeros(8000, np.float32)], [], 8000
        for source, left, right in pieces:
            blocks.append(pcm[source][left:right])
            mapping.append(
                {
                    "input": source,
                    "input_sample_range": [left, right],
                    "scenario_sample_range": [cursor, cursor + right - left],
                }
            )
            cursor += right - left
        blocks.append(np.zeros(20000, np.float32))
        values = np.concatenate(blocks)
        assembled[name] = (values, mapping)
    return inputs, assembled


class TraceHandler(logging.Handler):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def emit(self, record):
        if record.name.endswith("batch_speaker"):
            self.trace.append(
                {
                    "event": "batch_owner_log",
                    "observed_at": time.monotonic(),
                    "message": record.getMessage(),
                }
            )


class SessionTrace:
    """Transparent adapter instrumentation; does not select ranges or results."""

    def __init__(self, inner, trace, capture_values):
        from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes

        self.inner, self.trace = inner, trace
        self.payloads, self.hashes = {}, {}
        self.capture_pcm = float32_to_pcm16le_bytes(capture_values)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def begin_turn(self, request):
        self.payloads[request.identity] = []
        self.hashes[request.identity] = hashlib.sha256()
        self.trace.append(
            {"event": "native_begin", "at": time.monotonic(), "request": serial(request)}
        )
        await self.inner.begin_turn(request)

    async def turn_events(self):
        async for event in self.inner.turn_events():
            self.trace.append(
                {"event": "native_result", "at": time.monotonic(), "value": serial(event)}
            )
            yield event

    async def send_turn_audio(self, identity, pcm16le, **kwargs):
        self.hashes[identity].update(pcm16le)
        spans = kwargs.get("source_ranges", ())
        expected = b"".join(
            self.capture_pcm[span.normalized_start_sample * 2 : span.normalized_end_sample * 2]
            for span in spans
        )
        role = "source_context" if kwargs["context_only"] else "new_content"
        if not spans:
            role = "unmapped"
        self.payloads[identity].append(
            {
                "samples": len(pcm16le) // 2,
                "pcm16_sha256": hashlib.sha256(pcm16le).hexdigest(),
                "payload_role": role,
                "expected_source_pcm16_sha256": (
                    hashlib.sha256(expected).hexdigest() if spans else None
                ),
                "matches_declared_source_pcm": pcm16le == expected if spans else None,
                "matches_declared_native_input": bool(spans) and pcm16le == expected,
                **serial(kwargs),
            }
        )
        await self.inner.send_turn_audio(identity, pcm16le, **kwargs)

    async def seal_turn(self, identity, **kwargs):
        self.trace.append(
            {
                "event": "native_submit",
                "at": time.monotonic(),
                "identity": serial(identity),
                "pcm16_sha256": self.hashes.pop(identity).hexdigest(),
                "payloads": self.payloads.pop(identity),
                **serial(kwargs),
            }
        )
        await self.inner.seal_turn(identity, **kwargs)


async def case_run(args, name, values, mapping):
    result = {
        "status": "running",
        "case": name,
        "mode": args.mode,
        "input_samples": values.size,
        "input_pcm_f32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "assembly": mapping,
        "trace": [],
        "sources": [],
        "evidence": [],
    }
    trace = result["trace"]
    result["capture_input_samples"] = 0
    result["capture_input_ranges"] = []
    captured_pcm = hashlib.sha256()
    handler = TraceHandler(trace)
    log = logging.getLogger("puripuly_heart.core.stt.batch_speaker")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    backend = create_local_cpu_backend(
        args.asr_model,
        model_root=args.asr_root,
        source_language="en",
        sample_rate_hz=16000,
        stream_label="peer",
    )
    settings = AudioSegmentSettingsSnapshot(
        backend.provider_id,
        (backend.provider_id, args.asr_model),
        ("issue215",),
        "manual",
        "en",
        ("en",),
        16000,
        0.3,
        600,
        500,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    consumption = STTContributionConsumptionLedger()
    config = TranslationRuntimeConfigSnapshot(
        0,
        TranslationRuntimeConfig(
            source_language="en",
            target_language="ko",
            peer_source_language="en",
            peer_target_language="ko",
            peer_translation_enabled=True,
        ),
    )
    peer = ChannelRuntime("peer")
    request_owner = TranslationRequestOwner(
        config_snapshot=lambda: config,
        self_runtime=ChannelRuntime("self"),
        peer_runtime=peer,
        context_resolver=ContextResolver(),
        provider_runtime=SimpleNamespace(
            provider=object(), current_provider_generation=lambda: (None, 0)
        ),
        diagnostics=None,
        presentation=None,
        clock=SystemClock(),
    )

    async def nothing(*_args):
        return None

    async def source_only(*_args):
        return "source_only"

    async def admitted(children):
        requests = tuple(
            PeerTranslationChannelOwner._translation_process_request(child, child.target_language)
            for child in children
        )
        prepared = request_owner.admit_peer(requests)
        for child in children:
            source = child.source_child
            item = prepared.get(child.utterance_id)
            result["sources"].append(
                {
                    "admitted_at": time.monotonic(),
                    "source": serial(source),
                    "prior_context_texts": list(item.context_texts) if item else None,
                    "history_after": serial(peer.translation_history),
                    "effective_equals_raw": source.effective_text == source.raw_text,
                }
            )

    result["source_contract_errors"] = []

    async def rejected(parent_id):
        result["source_contract_errors"].append(f"Source parent rejected: {parent_id}")

    async def terminal(child, outcome):
        if outcome in ("failed", "cancelled"):
            result["source_contract_errors"].append(f"Source child {child.utterance_id}: {outcome}")

    turns = TranslationTurnLifecycleOwner(
        on_child_created=nothing,
        on_child_started=nothing,
        process_child=source_only,
        on_child_terminal=terminal,
        on_parent_closed=nothing,
        on_parent_rejected=rejected,
        on_parent_admitted=admitted,
    )
    publication_order = 0

    async def handle_result(event):
        nonlocal publication_order
        trace.append({"event": "provider_result", "at": time.monotonic(), "value": serial(event)})
        if not isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal)):
            return
        if isinstance(event, STTProviderTurnUpdate) and event.source_release is None:
            return
        snapshot = next((s for s in ledger.snapshots if s.identity == event.identity.segment), None)
        if snapshot is None:
            raise RuntimeError("result has no source ledger snapshot")
        if isinstance(event, STTProviderTurnTerminal):
            consumption.seal_audio(event.identity)
        pieces = consumption.consume_parts(event)
        if pieces:
            publication_order += 1
            transcript = contribution_transcript(
                snapshot,
                event,
                pieces,
                created_at=time.monotonic(),
                publication_order=publication_order,
            )
            await turns.submit(TranslationTurnRequest(transcript, "Peer", "peer", ("ko",), config))

    async def event_sink(event):
        try:
            await handle_result(event)
        except Exception as error:
            result["source_contract_errors"].append(f"{type(error).__name__}: {error}")
            raise

    async def factory(_settings, epoch):
        inner = SessionTrace(
            await backend.open_session(projection=STTSessionProjection("scoped", epoch)),
            trace,
            values,
        )
        if args.mode == "new":
            from puripuly_heart.core.stt.batch_speaker import CausalBatchSession

            result["boundary_policy"] = {
                name: value
                for name, value in vars(CausalBatchSession).items()
                if name.isupper() and isinstance(value, (int, float, str, bool))
            }
            return CausalBatchSession(inner)
        return inner

    engine = ScopedRecognitionEngine(
        factory, channel="peer", event_sink=event_sink, backend_close=backend.close
    )
    nemotron = speaker = None
    started = 0.0
    finished = False

    async def evidence_pump():
        while not finished:
            items = speaker.drain_evidence(2048)
            if items:
                result["evidence"].extend(serial(item) for item in items)
                await engine.observe_speaker_evidence(items)
            await asyncio.sleep(0.005)

    class Sink:
        async def handle_stream_input(self, event):
            result["capture_input_samples"] += event.chunk.size
            captured_pcm.update(event.chunk.tobytes())
            result["capture_input_ranges"].extend(serial(span) for span in event.capture)
            if speaker is not None:
                speaker.observe_input(f"issue215:{name}", event)
            # Local batch consumes mapped VAD payloads, not original capture again.

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
                yield AudioFrameF32(
                    values[offset:end],
                    16000,
                    capture=AudioCaptureSpan(
                        7,
                        sequence,
                        16000,
                        offset,
                        end,
                        started + offset / 16000,
                        started + end / 16000,
                    ),
                )

    pump = None
    try:
        # Real CPU load before playback, not a timing excuse or fabricated decode.
        await backend._ensure_recognizer()
        result["asr"] = {
            "provider_id": backend.provider_id,
            "model_id": backend.model_id,
            "provider": backend.provider,
            "num_threads": backend.num_threads,
            "language_hint": backend.language_hint,
            "hotwords": list(backend.hotwords),
            "feature_dim": backend.feature_dim,
        }
        if args.mode == "new":
            nemotron = NemotronBackend(
                artifact_path=args.model,
                runtime_dir=args.runtime,
                device=args.device,
                profile=args.profile,
            )
            speaker = SpeakerRuntime(nemotron)
            await speaker.start()
            if not speaker.snapshot().ready:
                raise RuntimeError(speaker.snapshot().error)
            result["speaker_initial"] = serial(speaker.snapshot())
            pump = asyncio.create_task(evidence_pump())
        started = time.monotonic()
        result["playback_started_at"] = started
        vad = create_peer_vad_gating(
            engine=SileroVadOnnx(bundled_silero_vad_onnx_path()),
            sample_rate_hz=16000,
            ring_buffer_ms=500,
            speech_threshold=0.3,
            hangover_ms=600,
        )
        result["vad_config"] = {
            "engine": "SileroVadOnnx CPU",
            "model_sha256": digest(bundled_silero_vad_onnx_path()),
            "sample_rate_hz": vad.sample_rate_hz,
            "chunk_samples": vad.chunk_samples,
            "speech_threshold": vad.speech_threshold,
            "continuation_threshold": vad.continuation_threshold,
            "hangover_chunks": vad.hangover_chunks,
            "start_debounce_chunks": vad.start_debounce_chunks,
            "start_commit_chunks": vad.start_commit_chunks,
            "external_delivery_boundaries": vad.external_delivery_boundaries,
            "onset_pre_roll_ms": 500,
            "hard_rollover_pre_roll_ms": 300,
        }
        await run_audio_vad_loop(
            source=Source(),
            vad=vad,
            sink=Sink(),
            target_sample_rate_hz=16000,
            channel_label="peer",
            segment_ledger=ledger,
        )
        deadline = time.monotonic() + 45
        while not engine.is_at_turn_boundary or (
            speaker is not None and speaker.snapshot().pending_items
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("real model result/drain timeout")
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.02)
        await engine.wait_for_event_ingress_drain()
        await turns.wait_for_idle()
        if speaker is not None:
            result["speaker_final"] = serial(speaker.snapshot())
        result["segments"] = serial(ledger.snapshots)
        result["capture_input_pcm_f32_sha256"] = captured_pcm.hexdigest()
        result["original_capture_received_once"] = (
            result["capture_input_samples"] == values.size
            and captured_pcm.hexdigest() == result["input_pcm_f32_sha256"]
        )
        result["status"] = (
            "passed" if result["sources"] and not result["source_contract_errors"] else "failed"
        )
        result["wall_seconds"] = time.monotonic() - started
    except Exception as error:
        result["status"], result["error"] = "failed", f"{type(error).__name__}: {error}"
    finally:
        finished = True
        if pump is not None:
            await pump
        await engine.close_backend()
        await turns.close()
        if speaker is not None:
            await speaker.stop()
            result["worker_stderr"] = bytes(nemotron._stderr).decode(errors="replace")
        log.removeHandler(handler)
    return result


async def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GGML_VK_DISABLE_GRAPH_OPTIMIZE"] = "1"
    os.environ["NEMO_SPEECH_SCHEDULE_CAPTURE"] = "1"
    inputs, cases = input_cases(args)
    from puripuly_heart.core.local_gpu_assets import inspect_local_gpu_install, local_gpu_model_path

    gpu = inspect_local_gpu_install(explicit_opt_in=True, model_root=args.asr_root)
    report = {
        "schema": "issue215-practical-causal-smoke-1",
        "command": sys.argv,
        "python": sys.version,
        "scope": "Production Silero VAD/listen delivery -> real scoped CPU ASR -> normal contribution ledger -> real SourceChild lifecycle/context admission; new path additionally real SpeakerRuntime FP16 Vulkan. No translation execution/renderer claims.",
        "privacy": {
            "public_or_authorized_synthetic_only": True,
            "network_calls": 0,
            "ambient_capture": False,
        },
        "trimming": "OFF",
        "mode": args.mode,
        "profile": args.profile if args.mode == "new" else None,
        "inputs": inputs,
        "nemotron_artifact": inspect_artifact(args.model),
        "runtime": validate_runtime(args.runtime),
        "runtime_manifest_sha256": digest(args.runtime / "runtime-manifest.json"),
        "asr_hashes": {
            str(p.relative_to(args.asr_root)): digest(p)
            for p in sorted((args.asr_root / args.asr_model).rglob("*"))
            if p.is_file()
        },
        "gpu_asr": {
            "status": "blocked" if not gpu.activation_allowed else "not_run",
            "install": serial(gpu),
            "selected_path": str(local_gpu_model_path(args.asr_root)),
            "override": "none supplied; production composition uses local_gpu_model_path()",
            "inference": "not_run",
        },
        "vulkan_summary": subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True
        ).stdout,
        "cases": [],
    }
    report["status_meaning"] = (
        "Case status records model/pipeline execution only; issue acceptance and qualitative gaps are assessed separately."
    )
    import onnxruntime
    import sherpa_onnx

    report["runtime_versions"] = {
        "numpy": np.__version__,
        "onnxruntime": onnxruntime.__version__,
        "sherpa_onnx": sherpa_onnx.__version__,
    }
    report["asr_installed_manifest"] = json.loads(
        (args.asr_root / args.asr_model / "installed-manifest.json").read_text(encoding="utf-8")
    )
    report["execution_source_sha256"] = {
        relative: digest(ROOT / relative)
        for relative in (
            "native/nemotron/issue215_smoke.py",
            "src/puripuly_heart/core/stt/scoped_engine.py",
            "src/puripuly_heart/core/stt/batch_speaker.py",
            "src/puripuly_heart/core/stt/backend.py",
            "src/puripuly_heart/core/stt/recognition_units.py",
            "src/puripuly_heart/domain/recognition.py",
            "src/puripuly_heart/core/speaker/runtime.py",
            "src/puripuly_heart/core/speaker/contracts.py",
            "src/puripuly_heart/app/adapters/nemotron/backend.py",
            "src/puripuly_heart/core/audio/ownership.py",
            "src/puripuly_heart/providers/stt/local_qwen_sherpa.py",
            "src/puripuly_heart/providers/stt/local_parakeet_sherpa.py",
            "src/puripuly_heart/core/audio/listen_delivery.py",
            "src/puripuly_heart/core/runtime/audio_vad_loop.py",
            "src/puripuly_heart/core/orchestrator/translation_turn.py",
            "src/puripuly_heart/core/orchestrator/translation_request.py",
        )
    }

    # Sanitize workstation-specific usernames/root paths, not causal timestamps.
    def save():
        text = json.dumps(serial(report), indent=2)
        for path, label in (
            (str(Path.home()), "<USER_HOME>"),
            (str(Path.home()).replace("\\", "/"), "<USER_HOME>"),
        ):
            text = text.replace(path.replace("\\", "\\\\"), label).replace(path, label)
        args.output.write_text(text + "\n", encoding="utf-8")

    for name in args.cases.split(","):
        values, mapping = cases[name]
        report["cases"].append(await case_run(args, name, values, mapping))
        save()
    return report


if __name__ == "__main__":
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "new"), required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=home / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf",
    )
    parser.add_argument("--runtime", type=Path, default=Path("build/nemotron"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile", choices=("0.64", "1.04"), default="0.64")
    parser.add_argument(
        "--asr-root", type=Path, default=home / "AppData/Local/puripuly-heart/models"
    )
    parser.add_argument("--asr-model", default="parakeet-tdt-0.6b-v3-int8-sherpa")
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
        default="ordinary_onset,six_second_rollover,clean_A_B,short_A_B_A,distinct_repeated_A_B_A,linguistic_correction",
    )
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
