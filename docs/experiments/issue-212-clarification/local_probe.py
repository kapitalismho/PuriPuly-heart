from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

from common import DIRECTORY, ROOT, pcm, settings_data, synthesize

BASELINE = "0f0ba5173da51250bb5926e4a2eead5e68adff81"
JA_REV = "bef18eb066808c90bd0f5df5be685767b0732de8"
JA_URL = f"https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/resolve/{JA_REV}/test_wavs/test_ja_1.wav"
JA_SHA = "09abd330ce706a6e6969fe6bbc8275314af631fcee4e536afd0626570d263fbf"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    (DIRECTORY / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def process_gate():
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "$p = @(Get-Process -Name VRChat,vrserver,vrcompositor,PuriPulyHeartGpuWorker -ErrorAction SilentlyContinue); ConvertTo-Json -Compress -InputObject @($p | Select-Object ProcessName,Id)"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Process safety gate failed: " + result.stderr)
    processes = json.loads(result.stdout)
    if processes:
        raise RuntimeError("Active VR/worker processes; neural work not authorized")
    return {"command": command, "returncode": result.returncode, "matched_processes": processes}


def wav(name, raw):
    path = DIRECTORY / "audio" / (name + ".wav")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(raw)
    return path


def trimmed(name, source):
    raw = pcm(source)
    samples = np.frombuffer(raw, dtype="<i2")
    nonzero = np.flatnonzero(samples != 0)
    start, end = int(nonzero[0]), int(nonzero[-1]) + 1
    path = wav(name, raw[start * 2:end * 2])
    return path, {"parent": str(source.relative_to(ROOT)), "parent_sha256": sha(source), "sample_range": [start, end], "rule": "Remove exact digital-zero leading/trailing samples only; no VAD/quality threshold"}


def audio_metadata(path, mapping=None):
    raw = pcm(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "pcm_sha256": hashlib.sha256(raw).hexdigest(), "samples": len(raw) // 2, "seconds": len(raw) / 32000, "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2, "mapping": mapping}


def prepare_audio():
    no = DIRECTORY / "audio/shared_no.wav"
    correction = DIRECTORY / "audio/shared_correction.wav"
    no_trim, no_map = trimmed("local_no_trim", no)
    correction_trim, correction_map = trimmed("local_correction_trim", correction)
    old = pcm(no_trim)[-4800 * 2:]
    context = wav("local_old_no_300ms_then_correction", old + pcm(correction_trim))
    mapping = {"pieces": [{"input_range": [0, 4800], "source": str(no_trim.relative_to(ROOT)), "source_range": [len(pcm(no_trim)) // 2 - 4800, len(pcm(no_trim)) // 2], "ownership": "already_owned_context_only"}, {"input_range": [4800, 4800 + len(pcm(correction_trim)) // 2], "source": str(correction_trim.relative_to(ROOT)), "source_range": [0, len(pcm(correction_trim)) // 2], "ownership": "new_content"}]}
    ja = DIRECTORY / "audio/local_official_ja_1.wav"
    if not ja.exists():
        with urllib.request.urlopen(JA_URL, timeout=40) as response:
            data = response.read(2_000_000)
        if len(data) != 1248044 or hashlib.sha256(data).hexdigest() != JA_SHA:
            raise ValueError("Pinned public Japanese sample hash/size mismatch")
        ja.write_bytes(data)
    if sha(ja) != JA_SHA:
        raise ValueError("Pinned public Japanese sample hash mismatch")
    # The upstream WAV is normalized below, never silently treated as 16 kHz.
    with wave.open(str(ja), "rb") as reader:
        original = {"rate": reader.getframerate(), "channels": reader.getnchannels(), "width": reader.getsampwidth(), "frames": reader.getnframes()}
        if original["channels"] != 1 or original["width"] != 2:
            raise ValueError("Unexpected official sample format")
        raw = reader.readframes(reader.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if original["rate"] != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(original["rate"], 16000)
        samples = resample_poly(samples, 16000 // divisor, original["rate"] // divisor)
    ja_clip = wav("local_ja_first_6s", np.round(np.clip(samples[:96000], -1, 32767 / 32768) * 32768).astype("<i2").tobytes())
    ko = synthesize("local_ko", "아니요. 삼입니다.", voice="Microsoft Heami Desktop")
    items = [(no, None), (no_trim, no_map), (correction, None), (correction_trim, correction_map), (context, mapping), (ja_clip, {"public_source_url": JA_URL, "public_sha256": JA_SHA, "original_format": original, "selected_normalized_range": [0, 96000], "transcript_reference": "日本語ちゃんと聞き取れてますかちゃんと聞こえてんのちゃんと聞こえてるのじゃあマイクを持ってもらって", "reference_scope": "Full upstream sample; not a claimed exact six-second transcript"}), (ko, {"text": "아니요. 삼입니다.", "voice": "Microsoft Heami Desktop"})]
    return {path.stem: (path, audio_metadata(path, mapping)) for path, mapping in items}


class CapturingRecognizer:
    """Observe the real result during the existing production decode, no fake ASR."""
    def __init__(self, inner):
        self.inner = inner
        self.raw = None

    def create_stream(self):
        return self.inner.create_stream()

    def decode_stream(self, stream):
        self.inner.decode_stream(stream)
        result = stream.result
        fields = {}
        for field in ("text", "tokens", "timestamps", "durations", "lang", "language", "ys", "json"):
            try:
                value = getattr(result, field)
                if callable(value):
                    value = value()
                fields[field] = value
            except AttributeError:
                fields[field] = {"absent": True}
            except Exception as error:
                fields[field] = {"access_error": f"{type(error).__name__}: {error}"}
        self.raw = fields


async def real_probes():
    from puripuly_heart.core.local_stt_assets import default_local_stt_model_root, load_local_stt_asset_manifest, REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    from puripuly_heart.core.local_stt_catalog import inspect_required_cpu_model_installs, resolve_cpu_auto_model
    from puripuly_heart.providers.stt.local_cpu import LocalCPUAutoSTTBackend

    report = {"baseline": BASELINE, "command": sys.argv, "python": sys.version, "executable": sys.executable, "kind": "actual production CPU loading/decode; no external inference", "process_gate": process_gate(), "runtime_versions": {name: importlib.metadata.version(name) for name in ("sherpa-onnx", "sherpa-onnx-core", "numpy")}, "models": [], "decodes": [], "auto": []}
    save("local-results-real.json", report)
    root = default_local_stt_model_root()
    report["model_root"] = str(root)
    for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS:
        manifest = load_local_stt_asset_manifest(model_id)
        files = []
        for file in manifest.files:
            path = root / manifest.install_dirname / file.relative_path
            digest = sha(path)
            files.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest, "expected_sha256": file.sha256, "hash_matches": digest == file.sha256})
        report["models"].append({"id": model_id, "files": files, "sources": {key: value.to_dict() for key, value in manifest.sources.items()}})
    snapshot = inspect_required_cpu_model_installs(root, verify_checksums=False)
    report["actual_auto_install_gate"] = {"available": snapshot.cpu_auto_available, "states": {model.model_id: model.state.status for model in snapshot.models}}
    if not all(file["hash_matches"] for model in report["models"] for file in model["files"]):
        save("local-results-real.json", report)
        raise RuntimeError("Model checksum readiness failed")
    audio = prepare_audio()
    report["audio"] = {name: metadata for name, (_, metadata) in audio.items()}
    english = ["shared_no", "local_no_trim", "shared_correction", "local_correction_trim", "local_old_no_300ms_then_correction"]
    sequences = [("en", english), ("ko", ["local_ko", *english]), ("ja", ["local_ja_first_6s"])]
    for language, names in sequences:
        report["process_gate_before_" + language] = process_gate()
        auto = LocalCPUAutoSTTBackend(source_language=language, model_root=root)
        session = None
        try:
            session = await auto.open_session()
            delegate = auto._delegate
            report["auto"].append({"requested_language": language, "resolver_model": resolve_cpu_auto_model(language), "resolved_model": auto.resolved_model_id, "delegate_class": type(delegate).__name__, "session_class": type(session).__name__, "loaded": auto.is_loaded, "num_threads": delegate.num_threads, "runtime_provider": delegate.provider})
            real = delegate._recognizer
            observer = CapturingRecognizer(real)
            delegate._recognizer = observer
            for name in names:
                path, metadata = audio[name]
                if language == "ko" and name != "local_ko":
                    delegate.language_hint = "English"
                row = {"clip": name, "model": delegate.model_id, "delegate_class": type(delegate).__name__, "language_hint_at_decode": delegate.language_hint, "audio": metadata}
                report["decodes"].append(row)
                save("local-results-real.json", report)
                try:
                    samples = np.frombuffer(pcm(path), dtype="<i2").astype(np.float32) / 32768
                    row["adapter_text"] = await delegate.decode_f32(samples)
                    row["raw_recognizer_result"] = observer.raw
                    row["outcome"] = "empty" if not row["adapter_text"] else "text"
                except Exception as error:
                    row["outcome"] = "failed"
                    row["error"] = f"{type(error).__name__}: {error}"
                save("local-results-real.json", report)
            # Existing Auto binds a delegate rather than resolving each assignment.
            before = auto.resolved_model_id
            auto.source_language = "ja" if language != "ja" else "en"
            report["auto"][-1]["after_source_language_assignment"] = {"language": auto.source_language, "resolved_model": auto.resolved_model_id, "same_delegate": auto._delegate is delegate, "original_model": before, "observation": "Property assignment only; no application transition invoked"}
            delegate._recognizer = real
        except Exception as error:
            report.setdefault("load_failures", []).append({"language": language, "error": f"{type(error).__name__}: {error}"})
        finally:
            if session is not None:
                await session.close()
            await auto.close()
            report.setdefault("cleanup", []).append({"language": language, "auto_closed": auto._closed, "delegate_released": auto._delegate is None})
            session = None
            delegate = None
            real = None
            observer = None
            gc.collect()
            save("local-results-real.json", report)
    for language in ("en", "ja", "ko", "zh", "ar", "uk", "xx"):
        try:
            result = resolve_cpu_auto_model(language)
        except Exception as error:
            result = f"{type(error).__name__}: {error}"
        report.setdefault("auto_resolution", {})[language] = result
    save("local-results-real.json", report)
    print(json.dumps({"decodes": len(report["decodes"]), "load_failures": report.get("load_failures", []), "output": "local-results-real.json"}))


def fixture_module(name, relative_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def gpu_probe():
    from puripuly_heart.app.adapters.gpu_worker_process import DefaultGpuWorkerProcessFactory
    from puripuly_heart.core.local_gpu_assets import local_gpu_model_path

    parent = ROOT.parents[2]
    executable = parent / "build/gpu_worker/PuriPulyHeartGpuWorker.exe"
    report = {"baseline": BASELINE, "command": sys.argv, "process_gate": process_gate(), "selected_worker": str(executable), "worker_size": executable.stat().st_size, "worker_sha256": sha(executable), "binary_source_revision": "unknown; explicit parent staged binary, not certified as candidate source build", "gpu_model_path": str(local_gpu_model_path()), "gpu_model_present": local_gpu_model_path().is_file(), "scope": "Authenticated protocol2 startup and Vulkan discover only; no activate/warmup/transcribe/Nemotron"}
    client = None
    try:
        client = await DefaultGpuWorkerProcessFactory(executable_path=executable, request_timeout_s=15).start(mode="discovery")
        report["authenticated_contract2"] = True
        report["devices"] = [dataclasses.asdict(device) for device in await client.discover()]
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if client is not None:
            await client.close()
            report["cleanup"] = {"closed": client.is_closed, "returncode": client.returncode}
        save("local-results-gpu.json", report)
    print(json.dumps(report, ensure_ascii=True))


async def component_probes():
    from dataclasses import replace
    from types import SimpleNamespace
    from uuid import uuid4
    from puripuly_heart.app.services.local_asr.local_asr_selection import resolve_local_asr_selection
    from puripuly_heart.config.provider_values import STTProviderName
    from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
    from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
    from puripuly_heart.core.stt.backend import STTProviderTurnIdentity, STTProviderTurnRequest, STTProviderTurnTerminal, STTSessionProjection, STTRecognitionUnit, STTRecognitionUnitTerminal
    from puripuly_heart.core.stt.rolling import RollingSTTBackend, RollingProviderDefinition
    from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
    from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
    from puripuly_heart.providers.stt.custom import CustomSTTBackend

    fixtures = fixture_module("local_probe_scoped_fixtures", "tests/core/test_stt_scoped_engine.py")
    custom_fixtures = fixture_module("local_probe_custom_fixtures", "tests/providers/test_custom_stt.py")
    report = {"baseline": BASELINE, "command": sys.argv, "kind": "Bounded actual production owners with explicitly fake recognizer/member/HTTP/WebSocket; no server or acoustic certification", "claims": {}}
    save("local-results-components.json", report)
    report["claims"]["A-selection"] = [dataclasses.asdict(resolve_local_asr_selection(provider, language, cpu_auto_available=available)) for provider, language, available in [("local_cpu_auto", "en", True), ("local_cpu_auto", "ja", True), ("local_cpu_auto", "ko", True), ("local_cpu_auto", "xx", True), ("local_cpu_auto", "en", False), ("local_parakeet_ja", "en", True)]]

    # Real scoped queue + decode lock, fake recognizer only. Freeze request settings,
    # then change session options before queued A can read the mutable backend.
    observed_options = []
    class FakeRecognizer:
        def create_stream(self):
            class Stream:
                result = SimpleNamespace(text="no")
                def __init__(self):
                    self.options = {}
                def set_option(self, name, value):
                    self.options[name] = value
                def accept_waveform(self, rate, samples):
                    self.samples = samples.copy()
            return Stream()
        def decode_stream(self, stream):
            observed_options.append({"options": dict(stream.options), "samples": len(stream.samples)})

    backend = LocalQwenSherpaSTTBackend(model_dir=DIRECTORY, language_hint="English")
    backend._recognizer = FakeRecognizer()
    session = await backend.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-options"))
    captured_events = []
    async def consume():
        async for event in session.turn_events():
            captured_events.append(event)
    consumer = asyncio.create_task(consume())
    await backend._decode_lock.acquire()
    try:
        requests = []
        for order in (1, 2):
            identity = STTProviderTurnIdentity(AudioSegmentIdentity(1, order, uuid4(), 1), "epoch-options", f"options-{order}")
            request_settings = replace(fixtures.settings("local_cpu_auto"), source_language="en" if order == 1 else "ja")
            request = STTProviderTurnRequest(identity, request_settings, channel="peer")
            requests.append(request)
            await session.begin_turn(request)
            await session.send_turn_audio(identity, b"\x01\x00" * 160, payload_sequence=1, source_ranges=(), context_only=False)
            await session.seal_turn(identity, sealed_content_ranges=(), seal_reason="silence", observed_trailing_silence_ms=0)
            if order == 1:
                await backend.reconfigure_session_options(LocalASRSessionOptions("ja", language_hint="Japanese"))
        backend._decode_lock.release()
        await fixtures.wait_until(lambda: len([e for e in captured_events if isinstance(e, STTProviderTurnTerminal)]) == 2, timeout=5)
        terminals = [e for e in captured_events if isinstance(e, STTProviderTurnTerminal)]
        report["claims"]["A-queued-options"] = {"request_languages": [request.settings.source_language for request in requests], "observed_decode_options": observed_options, "terminals": [dataclasses.asdict(event) for event in terminals], "different_equal_text_identities": terminals[0].identity != terminals[1].identity and terminals[0].text == terminals[1].text, "first_request_retains_English_options": observed_options[0]["options"].get("language") == "English", "result": "Current production queued decode reads reconfigured language; immutable request settings do not snapshot native options"}
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()
        await consumer
    save("local-results-components.json", report)

    # Keep old Scribe session alive while next open resolves Gemini; no network.
    old = fixtures.ControlledStreamSession()
    new = fixtures.ControlledStreamSession()
    configured = {"scribe": True}
    class FakeBackend:
        def __init__(self, inner):
            self.inner = inner
        async def open_session(self, **kwargs):
            return self.inner
    rolling = RollingSTTBackend(providers=(
        RollingProviderDefinition(STTProviderName.ELEVENLABS_SCRIBE, lambda: FakeBackend(old), lambda: configured["scribe"]),
        RollingProviderDefinition(STTProviderName.GEMINI_TRANSCRIBE, lambda: FakeBackend(new), lambda: True),
    ))
    old_wrapper = await rolling.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id="old"))
    configured["scribe"] = False
    new_wrapper = await rolling.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id="new"))
    old_stream = RecognitionStreamIdentity("peer", 1, 1, "old", ("rolling",))
    new_stream = replace(old_stream, provider_epoch_id="new")
    await old_wrapper.begin_stream(old_stream)
    await new_wrapper.begin_stream(new_stream)
    old_event = STTRecognitionUnit(RecognitionUnitIdentity(old_stream, uuid4(), 1), "no")
    new_event = STTRecognitionUnit(RecognitionUnitIdentity(new_stream, uuid4(), 1), "no")
    # Use wrapper's existing remapper directly; delayed old source cannot acquire new epoch.
    report["claims"]["A-Rolling-late"] = {"old_member_after_new_open": old_wrapper.provider_name.value, "new_member": new_wrapper.provider_name.value, "old_identity_preserved": old_wrapper._remap_recognition_identity(old_event.identity) == old_event.identity, "old_identity_rejected_by_new": new_wrapper._remap_recognition_identity(old_event.identity) is None, "new_identity_preserved": new_wrapper._remap_recognition_identity(new_event.identity) == new_event.identity, "replay_reuses_identity": old_wrapper._remap_recognition_identity(old_event.identity) == old_wrapper._remap_recognition_identity(old_event.identity), "equal_text_different_identity": old_event.text == new_event.text and old_event.identity != new_event.identity}
    await old_wrapper.end_stream(reason="source_discontinuity")
    report["claims"]["A-Rolling-late"]["delegated_old_controls"] = [call for call in old.calls if call[0] == "end_stream"]
    await old_wrapper.close()
    await new_wrapper.close()
    save("local-results-components.json", report)

    # Explicit discontinuity through engine + Rolling wrapper, not a proposed reducer.
    emitted = []
    inners = []
    wrappers = []
    async def engine_factory(_settings, epoch):
        inner = fixtures.ControlledStreamSession()
        inners.append(inner)
        rb = RollingSTTBackend(providers=(RollingProviderDefinition(STTProviderName.GEMINI_TRANSCRIBE, lambda: FakeBackend(inner), lambda: True),))
        wrapper = await rb.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))
        wrappers.append(wrapper)
        return wrapper
    engine = fixtures.ScopedRecognitionEngine(engine_factory, event_sink=emitted.append, watchdog_resolver=lambda _: fixtures.watchdogs())
    ledger = fixtures.PeerAudioSegmentLedger(activation_generation=3, settings=fixtures.settings("rolling_free"))
    try:
        start, _, end = fixtures.segment_events(ledger, start_sample=0, now=100)
        await engine.handle_stream_input(fixtures.stream_input(ledger, 2, 6, speech_observed=True))
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(end)
        previous = inners[0].stream
        await engine.handle_stream_input(fixtures.OwnedStreamInput(fixtures.CaptureStreamInput(np.empty(0, dtype=np.float32), (), boundary_reason="source_discontinuity"), ledger, ledger.settings, ledger.activation_generation))
        successor = fixtures.PeerAudioSegmentLedger(activation_generation=3, settings=ledger.settings)
        start2, _, _ = fixtures.segment_events(successor, start_sample=10, now=200)
        await engine.handle_owned_vad_event(start2)
        current = inners[-1].stream
        inners[-1].emit(STTRecognitionUnit(RecognitionUnitIdentity(previous, uuid4(), 1), "stale-no"))
        unit = STTRecognitionUnit(RecognitionUnitIdentity(current, uuid4(), 1), "no")
        inners[-1].emit(unit)
        inners[-1].emit(unit)  # replay same identity
        inners[-1].emit(STTRecognitionUnit(RecognitionUnitIdentity(current, uuid4(), 2), "no"))
        await fixtures.wait_until(lambda: len([e for e in emitted if isinstance(e, STTRecognitionUnitTerminal)]) == 2)
        finals = [e for e in emitted if isinstance(e, STTRecognitionUnitTerminal)]
        report["claims"]["A-Rolling-discontinuity"] = {"new_stream": current != previous, "terminal_texts": [e.unit.text for e in finals], "terminal_identities": [dataclasses.asdict(e.unit.identity) for e in finals], "old_controls": [call for call in inners[0].calls if call[0] in ("end_stream", "stop", "close")], "all_old_calls": [str(call) for call in inners[0].calls], "stale_rejected_replay_ignored_repetition_retained": [e.unit.text for e in finals] == ["no", "no"]}
    finally:
        await engine.close()
    save("local-results-components.json", report)

    # Safe configured-target presence only, no endpoint/credential output.
    custom = settings_data().get("stt", {}).get("custom", {})
    report["claims"]["C-configured-target"] = {"mode": custom.get("mode"), "compatibility": custom.get("compatibility"), "endpoint_present": bool(custom.get("endpoint")), "model_present": bool(custom.get("model")), "extra_model_present": bool(custom.get("extra", {}).get("model")), "named_server": "blocked when endpoint/model absent; no endpoint calls performed"}
    import io
    captured_requests = []
    async def handler(url, **kwargs):
        raw_wav = kwargs["files"]["file"][1]
        with wave.open(io.BytesIO(raw_wav), "rb") as reader:
            details = {"rate": reader.getframerate(), "channels": reader.getnchannels(), "width": reader.getsampwidth(), "frames": reader.getnframes(), "pcm_sha256": hashlib.sha256(reader.readframes(reader.getnframes())).hexdigest()}
        captured_requests.append({"url": url, "form": kwargs["data"], "wav": details})
        return custom_fixtures._FakeResponse(200, {"text": " no no ", "language": "en", "words": [{"word": "no", "start": 0, "end": 0.1}], "segments": [{"id": 1, "start": 0, "end": 0.36}], "server_revision": "fixture"})
    client = custom_fixtures._FakeAsyncClient(handler)
    custom_backend = CustomSTTBackend("offline", "openai_transcription", "http://example.invalid", "declared-model", source_language="en", extra={"model": "extra-model"}, http_client_factory=lambda **kwargs: client)
    custom_session = await custom_backend.open_session()
    try:
        raw = pcm(DIRECTORY / "audio/local_no_trim.wav")
        text = await custom_session._request_transcription(raw)
        report["claims"]["C-offline-local-protocol"] = {"requests": captured_requests, "returned_text": text, "input_pcm_sha256": hashlib.sha256(raw).hexdigest(), "upstream_extra_fields": ["words", "segments", "language", "server_revision"], "preserved_value_type": type(text).__name__, "fixture_not_named_server": True}
    finally:
        await custom_session.close()
    ws = custom_fixtures._FakeWebSocket([], hang=True)
    connect_calls = []
    async def connect(url, **kwargs):
        connect_calls.append({"url": url, "headers": kwargs.get("additional_headers", kwargs.get("extra_headers"))})
        return ws
    rt = CustomSTTBackend("realtime", "openai_realtime", "ws://example.invalid", "declared-model", source_language="en", websocket_connect=connect)
    rt_session = await rt.open_session()
    try:
        await rt_session.send_audio(pcm(DIRECTORY / "audio/local_no_trim.wav"))
        messages = [json.loads(message) for message in ws.sent]
        append = next(message for message in messages if message["type"] == "input_audio_buffer.append")
        import base64
        report["claims"]["C-realtime-local-format"] = {"connect": connect_calls, "session_update": messages[0], "append_bytes": len(base64.b64decode(append["audio"])), "append_pcm_sha256": hashlib.sha256(base64.b64decode(append["audio"])).hexdigest(), "adapter_rate": rt.sample_rate_hz, "wire_rate_declared": "input_audio_format" in messages[0]["session"], "fixture_not_named_server": True}
    finally:
        await rt_session.close()
    report["cleanup"] = {"fake_cpu_backend_closed": backend._closed, "fake_http_client_closed": client.closed, "fake_websocket_closed": ws.closed, "engine_closed": True}
    save("local-results-components.json", report)
    print(json.dumps({"output": "local-results-components.json", "claims": list(report["claims"])}))


async def rolling_scoped_probe():
    from uuid import uuid4
    from puripuly_heart.config.provider_values import STTProviderName
    from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
    from puripuly_heart.core.stt.backend import STTProviderTurnIdentity, STTProviderTurnRequest, STTProviderTurnTerminal, STTSessionProjection
    from puripuly_heart.core.stt.rolling import RollingSTTBackend, RollingProviderDefinition
    fixtures = fixture_module("local_probe_scoped_legal_fixtures", "tests/core/test_stt_scoped_engine.py")
    old = fixtures.ControlledScopedSession()  # Scribe correlated-turn contract, not independent units.
    new = fixtures.ControlledStreamSession()
    enabled = {"scribe": True}
    class FakeBackend:
        def __init__(self, inner):
            self.inner = inner
        async def open_session(self, **kwargs):
            return self.inner
    rolling = RollingSTTBackend(providers=(
        RollingProviderDefinition(STTProviderName.ELEVENLABS_SCRIBE, lambda: FakeBackend(old), lambda: enabled["scribe"]),
        RollingProviderDefinition(STTProviderName.GEMINI_TRANSCRIBE, lambda: FakeBackend(new), lambda: True),
    ))
    first = await rolling.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id="scribe-old"))
    identity = STTProviderTurnIdentity(AudioSegmentIdentity(1, 1, uuid4(), 1), "scribe-old", "first")
    request = STTProviderTurnRequest(identity, fixtures.settings("rolling_free"), channel="peer")
    capture = (fixtures.span(1, 0, 4),)
    await first.begin_turn(request)
    await first.send_turn_audio(identity, b"\x01\x00" * 4, payload_sequence=1, source_ranges=capture, context_only=False)
    await first.seal_turn(identity, sealed_content_ranges=capture, seal_reason="hardcut", observed_trailing_silence_ms=0)
    enabled["scribe"] = False
    second = await rolling.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id="gemini-new"))
    old.emit(STTProviderTurnTerminal(identity, outcome="final", text="no", text_authority="authoritative"))
    events = first.turn_events()
    try:
        received = await asyncio.wait_for(anext(events), timeout=1)
        report = {"baseline": BASELINE, "command": sys.argv, "kind": "Actual Rolling wrapper with fake scoped Scribe and independent Gemini member sessions; no wire server", "old_member": first.provider_name.value, "new_member": second.provider_name.value, "old_member_independent": first.independent_recognition_units, "new_member_independent": second.independent_recognition_units, "late_old_terminal": dataclasses.asdict(received), "late_identity_preserved": received.identity == identity, "delegated_control_calls": [str(call) for call in old.calls], "old_seal_count": sum(call[0] == "seal" for call in old.calls), "automatic_successor_seal_count": sum(call[0] == "seal" for call in new.calls), "scope_note": "Corrects the mechanical remapper fixture's artificial independent Scribe capability; does not certify Scribe wire finalization or billing"}
    finally:
        await events.aclose()
        await first.close()
        await second.close()
    report["cleanup"] = {"old_closed": ("close",) in old.calls, "new_closed": ("close",) in new.calls}
    save("local-results-rolling.json", report)
    print(json.dumps({"output": "local-results-rolling.json", "late_identity_preserved": report["late_identity_preserved"], "old_seal_count": report["old_seal_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["real", "components", "gpu", "rolling-scoped"])
    args = parser.parse_args()
    asyncio.run({"real": real_probes, "components": component_probes, "gpu": gpu_probe, "rolling-scoped": rolling_scoped_probe}[args.stage]())
