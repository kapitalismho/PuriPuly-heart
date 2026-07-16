from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.core.local_qwen_runtime import LocalQwenRuntimeBootstrapError
from puripuly_heart.core.local_stt_assets import (
    InstalledLocalSTTManifest,
    LocalSTTManifestInvalidError,
    LocalSTTModelMissingError,
)
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt import local_qwen_sherpa as local_qwen_module
from puripuly_heart.providers.stt.local_qwen_sherpa import (
    LocalQwenSherpaInferenceError,
    LocalQwenSherpaLoadError,
    LocalQwenSherpaSTTBackend,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("stage", ["validation", "recognizer", "inference"])
async def test_local_qwen_synchronous_heavy_stage_runs_inside_to_thread_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    stage: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_threads: list[int] = []

    def blocked() -> None:
        worker_threads.append(threading.get_ident())
        entered.set()
        release.wait(timeout=2)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), stream_label=channel)
    monkeypatch.setattr(local_qwen_module, "validate_local_stt_runtime_ready", lambda _path: None)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_create_recognizer", lambda self: object())
    if stage == "validation":
        monkeypatch.setattr(
            local_qwen_module,
            "validate_local_stt_runtime_ready",
            lambda _path: blocked(),
        )
        operation = asyncio.create_task(backend.open_session())
    elif stage == "recognizer":
        monkeypatch.setattr(
            LocalQwenSherpaSTTBackend,
            "_create_recognizer",
            lambda self: blocked() or object(),
        )
        operation = asyncio.create_task(backend.open_session())
    else:
        backend._recognizer = object()
        monkeypatch.setattr(
            LocalQwenSherpaSTTBackend,
            "_decode_f32_sync",
            lambda self, recognizer, samples: blocked() or "text",
        )
        operation = asyncio.create_task(backend.decode_f32(np.zeros(160, dtype=np.float32)))

    assert await asyncio.to_thread(entered.wait, 1)
    heartbeat = False
    await asyncio.sleep(0)
    heartbeat = True
    release.set()
    result = await operation
    if stage != "inference":
        await result.close()
    assert heartbeat is True
    assert worker_threads and worker_threads[0] != threading.get_ident()


def test_local_qwen_backend_uses_thread_count_3_by_default() -> None:
    assert local_qwen_module.DEFAULT_SHERPA_NUM_THREADS == 3
    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    assert backend.num_threads == 3


def _installed_manifest() -> InstalledLocalSTTManifest:
    return InstalledLocalSTTManifest(
        manifest_version=1,
        model_id="qwen3-asr-0.6b-int8-sherpa",
        engine="sherpa-onnx",
        install_dirname="qwen3-asr-0.6b-int8-sherpa",
        selected_source="huggingface",
        selected_revision="rev-1",
    )


def _install_fake_sherpa(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recognizer_factory,
    qwen3_error: Exception | None = None,
    bootstrap_runtime=None,
) -> dict[str, object]:
    factory_calls: dict[str, object] = {}

    class ConfigNode:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeOfflineQwen3ASRModelConfig(ConfigNode):
        def __init__(self, **kwargs) -> None:
            if qwen3_error is not None:
                raise qwen3_error
            super().__init__(**kwargs)
            factory_calls["qwen3"] = kwargs

    class FakeOfflineModelConfig(ConfigNode):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            factory_calls["model"] = kwargs

    class FakeFeatureExtractorConfig(ConfigNode):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            factory_calls["feat"] = kwargs

    class FakeOfflineRecognizerConfig(ConfigNode):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            factory_calls["recognizer"] = kwargs

    fake_sherpa = ModuleType("sherpa_onnx")
    fake_sherpa.OfflineQwen3ASRModelConfig = FakeOfflineQwen3ASRModelConfig
    fake_sherpa.OfflineModelConfig = FakeOfflineModelConfig
    fake_sherpa.FeatureExtractorConfig = FakeFeatureExtractorConfig
    fake_sherpa.OfflineRecognizerConfig = FakeOfflineRecognizerConfig

    fake_offline_recognizer = ModuleType("sherpa_onnx.offline_recognizer")
    fake_offline_recognizer._Recognizer = recognizer_factory

    if bootstrap_runtime is None:

        def bootstrap_runtime() -> Path:
            return Path("C:/runtime")

    monkeypatch.setattr(local_qwen_module, "ensure_local_qwen_windows_runtime", bootstrap_runtime)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setitem(sys.modules, "sherpa_onnx.offline_recognizer", fake_offline_recognizer)
    return factory_calls


def test_create_local_qwen_sherpa_recognizer_bootstraps_windows_runtime_before_using_sherpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def fake_bootstrap() -> Path:
        order.append("bootstrap")
        return Path("C:/runtime")

    class ConfigNode:
        def __init__(self, **kwargs) -> None:
            order.append(type(self).__name__)
            self.kwargs = kwargs

    class FakeOfflineQwen3ASRModelConfig(ConfigNode):
        pass

    class FakeOfflineModelConfig(ConfigNode):
        pass

    class FakeFeatureExtractorConfig(ConfigNode):
        pass

    class FakeOfflineRecognizerConfig(ConfigNode):
        pass

    class FakeRecognizer:
        def __init__(self, recognizer_config) -> None:
            order.append("recognizer")
            self.recognizer_config = recognizer_config

    fake_sherpa = ModuleType("sherpa_onnx")
    fake_sherpa.OfflineQwen3ASRModelConfig = FakeOfflineQwen3ASRModelConfig
    fake_sherpa.OfflineModelConfig = FakeOfflineModelConfig
    fake_sherpa.FeatureExtractorConfig = FakeFeatureExtractorConfig
    fake_sherpa.OfflineRecognizerConfig = FakeOfflineRecognizerConfig

    fake_offline_recognizer = ModuleType("sherpa_onnx.offline_recognizer")
    fake_offline_recognizer._Recognizer = FakeRecognizer

    monkeypatch.setattr(local_qwen_module, "ensure_local_qwen_windows_runtime", fake_bootstrap)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setitem(sys.modules, "sherpa_onnx.offline_recognizer", fake_offline_recognizer)

    recognizer = local_qwen_module.create_local_qwen_sherpa_recognizer(
        model_dir=Path("/models/qwen"),
        num_threads=3,
    )

    assert isinstance(recognizer, FakeRecognizer)
    assert order[0] == "bootstrap"
    assert order.index("bootstrap") < order.index("FakeOfflineQwen3ASRModelConfig")
    assert order.index("bootstrap") < order.index("recognizer")


def test_create_local_qwen_sherpa_recognizer_rejects_legacy_8000_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: SimpleNamespace())

    with pytest.raises(ValueError, match="16000"):
        local_qwen_module.create_local_qwen_sherpa_recognizer(
            model_dir=Path("/models/qwen"),
            num_threads=3,
            sample_rate_hz=8000,
        )


@pytest.mark.asyncio
async def test_local_qwen_backend_emits_final_transcript_on_speech_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = Path("/models/qwen")
    recognizer_state: dict[str, object] = {}

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: list[tuple[int, object]] = []
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            self.accepted.append((sample_rate, samples))

    class FakeRecognizer:
        def __init__(self) -> None:
            self.streams: list[FakeStream] = []

        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            self.streams.append(stream)
            recognizer_state["stream"] = stream
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            recognizer_state["decoded"] = stream

    class FakeRecognizerEngine(FakeRecognizer):
        def __init__(self, recognizer_config) -> None:
            super().__init__()
            recognizer_state["recognizer_config"] = recognizer_config
            recognizer_state["recognizer"] = self

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    factory_calls = _install_fake_sherpa(
        monkeypatch,
        recognizer_factory=FakeRecognizerEngine,
    )

    backend = LocalQwenSherpaSTTBackend(model_dir=model_dir, sample_rate_hz=16000, num_threads=3)
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00\xff\x7f")
    await session.on_speech_end()

    gen = session.events()
    event = await gen.__anext__()

    assert event.text == "hello local qwen"
    assert event.is_final is True
    assert factory_calls == {
        "qwen3": {
            "conv_frontend": str(model_dir / "conv_frontend.onnx"),
            "encoder": str(model_dir / "encoder.int8.onnx"),
            "decoder": str(model_dir / "decoder.int8.onnx"),
            "tokenizer": str(model_dir / "tokenizer"),
            "max_total_len": 512,
            "max_new_tokens": 128,
            "temperature": 1e-06,
            "top_p": 0.8,
            "seed": 42,
        },
        "model": {
            "qwen3_asr": recognizer_state["recognizer_config"]
            .kwargs["model_config"]
            .kwargs["qwen3_asr"],
            "num_threads": 3,
            "debug": False,
            "provider": "cpu",
        },
        "feat": {
            "sampling_rate": 16000,
            "feature_dim": 128,
        },
        "recognizer": {
            "feat_config": recognizer_state["recognizer_config"].kwargs["feat_config"],
            "model_config": recognizer_state["recognizer_config"].kwargs["model_config"],
            "decoding_method": "greedy_search",
        },
    }
    assert recognizer_state["decoded"] is recognizer_state["stream"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["leşme", "acia"])
async def test_local_qwen_backend_redacts_known_hallucination_text_in_transcript_log_but_emits_final_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    text: str,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text=text)

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = (sample_rate, samples)

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        sample_rate_hz=16000,
        stream_label="self",
    )
    session = await backend.open_session()

    with caplog.at_level(logging.INFO, logger=local_qwen_module.__name__):
        await session.send_audio(b"\x00\x00\xff\x7f")
        await session.on_speech_end()
        await session.stop()

    gen = session.events()
    event = await gen.__anext__()

    assert event.text == text
    assert event.is_final is True
    assert any(
        "[STT][local_qwen][self] Transcript final text_len=" in message
        and "known_hallucination=True" in message
        for message in caplog.messages
    )
    assert not any(text in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_local_qwen_backend_logs_non_matching_transcript_metadata_without_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "hello local qwen"

    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text=text)

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = (sample_rate, samples)

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        sample_rate_hz=16000,
        stream_label="self",
    )
    session = await backend.open_session()

    with caplog.at_level(logging.INFO, logger=local_qwen_module.__name__):
        await session.send_audio(b"\x00\x00\xff\x7f")
        await session.on_speech_end()
        await session.stop()

    gen = session.events()
    event = await gen.__anext__()

    assert event.text == text
    assert event.is_final is True
    assert any(
        "[STT][local_qwen][self] Transcript final text_len=16" in message
        and "known_hallucination=False" in message
        for message in caplog.messages
    )
    assert not any(text in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_local_qwen_session_send_audio_f32_preserves_float32_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer_state: dict[str, object] = {}

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: list[tuple[int, np.ndarray]] = []
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            self.accepted.append((sample_rate, np.asarray(samples, dtype=np.float32)))

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            recognizer_state["stream"] = stream
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            recognizer_state["decoded"] = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=16000)
    session = await backend.open_session()
    original = np.array([0.123456, -0.234567, 0.9999], dtype=np.float32)

    await session.send_audio_f32(original)
    await session.on_speech_end()
    await session.stop()

    stream = recognizer_state["stream"]
    assert isinstance(stream, FakeStream)
    accepted_rate, accepted_samples = stream.accepted[0]
    assert accepted_rate == 16000
    np.testing.assert_array_equal(accepted_samples, original)


@pytest.mark.asyncio
async def test_local_qwen_session_repeated_speech_end_decodes_and_clears_each_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer_state: dict[str, object] = {}
    texts = iter(["first local", "second local"])

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: list[tuple[int, np.ndarray]] = []
            self.result = SimpleNamespace(text=next(texts))

        def accept_waveform(self, sample_rate: int, samples) -> None:
            self.accepted.append((sample_rate, np.asarray(samples, dtype=np.float32).copy()))

    class FakeRecognizer:
        def __init__(self) -> None:
            self.streams: list[FakeStream] = []

        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            self.streams.append(stream)
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    def make_recognizer(_config) -> FakeRecognizer:
        recognizer = FakeRecognizer()
        recognizer_state["recognizer"] = recognizer
        return recognizer

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=make_recognizer)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=16000)
    session = await backend.open_session()
    first = np.array([0.1, 0.2], dtype=np.float32)
    second = np.array([0.3], dtype=np.float32)

    await session.send_audio_f32(first)
    await session.on_speech_end()
    await session.send_audio_f32(second)
    await session.on_speech_end()

    gen = session.events()
    first_event = await gen.__anext__()
    second_event = await gen.__anext__()

    recognizer = recognizer_state["recognizer"]
    assert isinstance(recognizer, FakeRecognizer)
    assert [first_event.text, second_event.text] == ["first local", "second local"]
    assert len(recognizer.streams) == 2
    np.testing.assert_array_equal(recognizer.streams[0].accepted[0][1], first)
    np.testing.assert_array_equal(recognizer.streams[1].accepted[0][1], second)


@pytest.mark.asyncio
async def test_local_qwen_session_clips_float32_before_accept_waveform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer_state: dict[str, object] = {}

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: list[tuple[int, np.ndarray]] = []
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            self.accepted.append((sample_rate, np.asarray(samples, dtype=np.float32)))

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            recognizer_state["stream"] = stream
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            recognizer_state["decoded"] = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=16000)
    session = await backend.open_session()
    original = np.array([1.25, -1.50, 0.25], dtype=np.float32)

    await session.send_audio_f32(original)
    await session.on_speech_end()
    await session.stop()

    stream = recognizer_state["stream"]
    assert isinstance(stream, FakeStream)
    accepted_rate, accepted_samples = stream.accepted[0]
    assert accepted_rate == 16000
    np.testing.assert_array_equal(
        accepted_samples,
        np.array([1.0, -1.0, 0.25], dtype=np.float32),
    )
    np.testing.assert_array_equal(original, np.array([1.25, -1.50, 0.25], dtype=np.float32))


@pytest.mark.asyncio
async def test_local_qwen_backend_sets_stream_language_hint_and_hotwords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer_state: dict[str, object] = {}

    class FakeStream:
        def __init__(self) -> None:
            self.options: dict[str, str] = {}
            self.result = SimpleNamespace(text="hello local qwen")

        def set_option(self, key: str, value: str) -> None:
            self.options[key] = value

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = (sample_rate, samples)

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            recognizer_state["stream"] = stream
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            recognizer_state["decoded"] = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        language_hint="Korean",
        hotwords=("Puripuly", "VRChat"),
    )
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00")
    await session.on_speech_end()
    await session.stop()

    stream = recognizer_state["stream"]
    assert isinstance(stream, FakeStream)
    assert stream.options == {"language": "Korean", "hotwords": "Puripuly,VRChat"}


def test_local_qwen_backend_rejects_legacy_8000_runtime_sample_rate() -> None:
    with pytest.raises(ValueError, match="16000"):
        LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=8000)


@pytest.mark.asyncio
async def test_local_qwen_backend_keeps_16000_input_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognizer_state: dict[str, object] = {}

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: list[tuple[int, np.ndarray]] = []
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            self.accepted.append((sample_rate, np.asarray(samples)))

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            recognizer_state["stream"] = stream
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            recognizer_state["decoded"] = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=16000)
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00\xff\x7f")
    await session.on_speech_end()
    await session.stop()

    stream = recognizer_state["stream"]
    assert isinstance(stream, FakeStream)
    accepted_rate, accepted_samples = stream.accepted[0]
    assert accepted_rate == 16000


@pytest.mark.asyncio
async def test_local_qwen_backend_runtime_validator_runs_only_until_recognizer_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def fake_runtime_ready(model_dir: Path, **_kwargs) -> InstalledLocalSTTManifest:
        calls.append(model_dir)
        return _installed_manifest()

    monkeypatch.setattr(local_qwen_module, "validate_local_stt_runtime_ready", fake_runtime_ready)
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: SimpleNamespace())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=16000)

    await backend.open_session()
    await backend.open_session()

    assert calls == [Path("/models/qwen")]


@pytest.mark.asyncio
async def test_local_qwen_backend_revalidates_runtime_assets_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def fake_runtime_ready(model_dir: Path, **_kwargs) -> InstalledLocalSTTManifest:
        calls.append(model_dir)
        return _installed_manifest()

    monkeypatch.setattr(local_qwen_module, "validate_local_stt_runtime_ready", fake_runtime_ready)
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: SimpleNamespace())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"), sample_rate_hz=16000)

    await backend.open_session()
    await backend.close()
    await backend.open_session()

    assert calls == [Path("/models/qwen"), Path("/models/qwen")]


@pytest.mark.asyncio
async def test_local_qwen_backend_surfaces_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(LocalSTTModelMissingError("missing")),
    )

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))

    with pytest.raises(LocalSTTModelMissingError, match="missing"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_local_qwen_backend_surfaces_invalid_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            LocalSTTManifestInvalidError("manifest invalid")
        ),
    )

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))

    with pytest.raises(LocalSTTManifestInvalidError, match="manifest invalid"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_local_qwen_backend_wraps_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(
        monkeypatch,
        recognizer_factory=lambda _config: None,
        qwen3_error=RuntimeError("load failed"),
    )

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))

    with pytest.raises(LocalQwenSherpaLoadError, match="load failed"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_local_qwen_backend_wraps_runtime_bootstrap_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    monkeypatch.setattr(
        local_qwen_module,
        "create_local_qwen_sherpa_recognizer",
        lambda **_kwargs: (_ for _ in ()).throw(
            LocalQwenRuntimeBootstrapError("runtime bootstrap failed")
        ),
    )

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))

    with pytest.raises(LocalQwenSherpaLoadError, match="runtime bootstrap failed") as exc_info:
        await backend.open_session()

    assert isinstance(exc_info.value.__cause__, LocalQwenRuntimeBootstrapError)


@pytest.mark.asyncio
async def test_local_qwen_backend_preserves_missing_onnxruntime_bootstrap_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(
        monkeypatch,
        recognizer_factory=lambda _config: None,
        bootstrap_runtime=lambda: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'onnxruntime'")
        ),
    )

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))

    with pytest.raises(LocalQwenSherpaLoadError, match="onnxruntime") as exc_info:
        await backend.open_session()

    assert str(exc_info.value) != "failed to import sherpa_onnx"
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.asyncio
async def test_local_qwen_session_surfaces_inference_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream
            raise RuntimeError("decode failed")

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00")
    await session.on_speech_end()

    gen = session.events()
    boundary = await gen.__anext__()
    assert boundary.text == ""
    assert boundary.is_final is True
    with pytest.raises(LocalQwenSherpaInferenceError, match="decode failed"):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_local_qwen_session_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    session = await backend.open_session()

    await session.close()
    await session.close()

    gen = session.events()
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_local_qwen_session_logs_inference_metrics_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    texts = iter(["first local qwen", "second local qwen"])

    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text=next(texts))

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    perf_values = iter([1.0, 1.25, 2.0, 2.2])
    monkeypatch.setattr(local_qwen_module.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        sample_rate_hz=16000,
        stream_label="peer",
    )
    session = await backend.open_session()

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.local_qwen_sherpa"):
        await session.send_audio(b"\x00\x00" * 16000)
        await session.on_speech_end()
        await session.send_audio(b"\x00\x00" * 8000)
        await session.on_speech_end()
        await session.stop()

    messages = [record.getMessage() for record in caplog.records]
    final_messages = [message for message in messages if "Transcript final" in message]
    assert len(final_messages) == 2
    assert (
        "[STT][local_qwen][peer] Transcript final text_len=16 "
        "known_hallucination=False audio_ms=1000.0 inference_ms=250.0 rtf=0.250"
    ) in final_messages
    assert (
        "[STT][local_qwen][peer] Transcript final text_len=17 "
        "known_hallucination=False audio_ms=500.0 inference_ms=200.0 rtf=0.400"
    ) in final_messages
    assert not any("first local qwen" in message for message in messages)
    assert not any("second local qwen" in message for message in messages)
    assert (
        "[STT][local_qwen][peer] Session summary: utterances=2 "
        "total_audio_ms=1500.0 total_inference_ms=450.0 weighted_total_rtf=0.300 mean_rtf=0.325"
    ) in messages


@pytest.mark.asyncio
async def test_local_qwen_logs_decode_diagnostics_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="的答案的答案的答案")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        stream_label="self",
        language_hint="Korean",
        diagnostics_enabled=lambda: True,
    )
    session = await backend.open_session()

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.local_qwen_sherpa"):
        await session.send_audio_f32(np.ones(16000, dtype=np.float32))
        await session.on_speech_end()
        await session.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert any("[AudioDiag][local_qwen][self] decode_start" in message for message in messages)
    assert any("decode_done" in message and "empty_result=False" in message for message in messages)
    assert any("suspicious_script=True" in message for message in messages)
    assert any("suspicious_repetition=True" in message for message in messages)
    audio_diag_messages = [message for message in messages if "[AudioDiag][local_qwen]" in message]
    assert all("的答案的答案的答案" not in message for message in audio_diag_messages)


@pytest.mark.asyncio
async def test_local_qwen_logs_empty_decode_diagnostics_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        stream_label="peer",
        diagnostics_enabled=lambda: True,
    )
    session = await backend.open_session()

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.local_qwen_sherpa"):
        await session.send_audio_f32(np.zeros(16000, dtype=np.float32))
        await session.on_speech_end()
        await session.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert any("[AudioDiag][local_qwen][peer] decode_done" in message for message in messages)
    assert any("empty_result=True" in message and "text_len=0" in message for message in messages)


@pytest.mark.asyncio
async def test_local_qwen_decode_diagnostics_stay_silent_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())
    monkeypatch.setattr(
        local_qwen_module,
        "compute_audio_frame_metrics",
        lambda _frame: (_ for _ in ()).throw(
            AssertionError("disabled local_qwen diagnostics must not compute metrics")
        ),
        raising=False,
    )

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        diagnostics_enabled=lambda: False,
    )
    session = await backend.open_session()

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.local_qwen_sherpa"):
        await session.send_audio_f32(np.ones(16000, dtype=np.float32))
        await session.on_speech_end()

    messages = [record.getMessage() for record in caplog.records]
    assert not any("[AudioDiag][local_qwen]" in message for message in messages)


@pytest.mark.asyncio
async def test_local_qwen_decode_continues_when_diagnostics_predicate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    def failing_diagnostics_enabled() -> bool:
        raise RuntimeError("diagnostics predicate failed")

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        diagnostics_enabled=failing_diagnostics_enabled,
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.ones(16000, dtype=np.float32))
    await session.on_speech_end()

    gen = session.events()
    event = await gen.__anext__()
    assert event.text == "hello local qwen"


@pytest.mark.asyncio
async def test_local_qwen_decode_continues_when_diagnostic_metrics_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())
    monkeypatch.setattr(
        local_qwen_module,
        "compute_audio_frame_metrics",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("metrics failed")),
        raising=False,
    )

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        diagnostics_enabled=lambda: True,
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.ones(16000, dtype=np.float32))
    await session.on_speech_end()

    gen = session.events()
    event = await gen.__anext__()
    assert event.text == "hello local qwen"


@pytest.mark.asyncio
async def test_local_qwen_decode_continues_when_diagnostic_logging_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="hello local qwen")

        def accept_waveform(self, sample_rate: int, samples) -> None:
            _ = sample_rate, samples

    class FakeRecognizer:
        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            _ = stream

    monkeypatch.setattr(
        local_qwen_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())
    original_info = local_qwen_module.logger.info

    def flaky_info(message, *args, **kwargs) -> None:
        if args and isinstance(args[0], str) and args[0].startswith("[AudioDiag][local_qwen]"):
            raise RuntimeError("diagnostic logging failed")
        original_info(message, *args, **kwargs)

    monkeypatch.setattr(local_qwen_module.logger, "info", flaky_info)

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        diagnostics_enabled=lambda: True,
    )
    session = await backend.open_session()

    await session.send_audio_f32(np.ones(16000, dtype=np.float32))
    await session.on_speech_end()

    gen = session.events()
    event = await gen.__anext__()
    assert event.text == "hello local qwen"


@pytest.mark.asyncio
async def test_local_qwen_speech_end_queues_fifo_decode_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    decoded: list[np.ndarray] = []

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        decoded.append(samples_f32.copy())
        started.set()
        await release.wait()
        return str(len(decoded))

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    session = await backend.open_session()
    first = np.ones(160, dtype=np.float32)
    second = np.full(320, 0.5, dtype=np.float32)

    await session.send_audio_f32(first)
    await asyncio.wait_for(session.on_speech_end(), timeout=0.1)
    await asyncio.wait_for(started.wait(), timeout=0.1)
    await session.send_audio_f32(second)
    await asyncio.wait_for(session.on_speech_end(), timeout=0.1)

    assert len(decoded) == 1
    stop_task = asyncio.create_task(session.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    release.set()
    await asyncio.wait_for(stop_task, timeout=1.0)

    events = session.events()
    first_event = await events.__anext__()
    second_event = await events.__anext__()
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()

    assert [first_event.text, second_event.text] == ["1", "2"]
    np.testing.assert_array_equal(decoded[0], first)
    np.testing.assert_array_equal(decoded[1], second)


@pytest.mark.asyncio
async def test_local_qwen_sessions_handoff_finals_before_next_session_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    decoded: list[np.ndarray] = []

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        decoded.append(samples_f32.copy())
        sequence = len(decoded)
        if sequence == 1:
            first_started.set()
            await release_first.wait()
        return f"final-{sequence}"

    async def collect_events(session) -> list[STTBackendTranscriptEvent]:
        return [event async for event in session.events()]

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    old_session = await backend.open_session()
    old_events_task = asyncio.create_task(collect_events(old_session))

    await old_session.send_audio_f32(np.full(160, 1.0, dtype=np.float32))
    await old_session.on_speech_end()
    await asyncio.wait_for(first_started.wait(), timeout=0.1)
    await old_session.send_audio_f32(np.full(160, 2.0, dtype=np.float32))
    await old_session.on_speech_end()

    new_session = await backend.open_session()
    new_events_task = asyncio.create_task(collect_events(new_session))
    await new_session.send_audio_f32(np.full(160, 3.0, dtype=np.float32))
    await asyncio.wait_for(new_session.on_speech_end(), timeout=0.1)
    await asyncio.sleep(0)

    assert len(decoded) == 1

    stop_old_task = asyncio.create_task(old_session.stop())
    stop_new_task = asyncio.create_task(new_session.stop())
    release_first.set()
    await asyncio.wait_for(
        asyncio.gather(stop_old_task, stop_new_task),
        timeout=1.0,
    )
    old_events, new_events = await asyncio.wait_for(
        asyncio.gather(old_events_task, new_events_task),
        timeout=1.0,
    )

    assert [event.text for event in old_events] == ["final-1", "final-2"]
    assert [event.text for event in new_events] == ["final-3"]
    assert [int(samples[0]) for samples in decoded] == [1, 2, 3]


@pytest.mark.asyncio
async def test_local_qwen_close_cancels_and_awaits_decode_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        _ = samples_f32
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ""

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    session = await backend.open_session()
    await session.send_audio_f32(np.ones(160, dtype=np.float32))
    await session.on_speech_end()
    await asyncio.wait_for(started.wait(), timeout=0.1)

    await asyncio.wait_for(session.close(), timeout=0.1)

    assert cancelled.is_set()
    events = session.events()
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()


@pytest.mark.asyncio
async def test_local_qwen_empty_decode_preserves_next_final_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(["", "second final"])

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        _ = samples_f32
        return next(results)

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    session = await backend.open_session()
    await session.send_audio_f32(np.ones(160, dtype=np.float32))
    await session.on_speech_end()
    await session.send_audio_f32(np.full(160, 0.5, dtype=np.float32))
    await session.on_speech_end()
    await session.stop()

    events = session.events()
    first_event = await events.__anext__()
    second_event = await events.__anext__()
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()

    assert first_event.text == ""
    assert first_event.is_final is True
    assert second_event.text == "second final"
    assert second_event.is_final is True


@pytest.mark.asyncio
async def test_local_qwen_speech_end_without_audio_emits_empty_final_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_calls = 0

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        nonlocal decode_calls
        decode_calls += 1
        _ = samples_f32
        return "unexpected"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    session = await backend.open_session()
    await session.on_speech_end()
    await session.stop()

    events = session.events()
    event = await events.__anext__()
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()

    assert decode_calls == 0
    assert event.text == ""
    assert event.is_final is True


@pytest.mark.asyncio
async def test_local_qwen_logs_backlog_warning_once_with_stream_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        _ = samples_f32
        started.set()
        await release.wait()
        return ""

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("/models/qwen"),
        stream_label="peer",
    )
    session = await backend.open_session()

    with caplog.at_level(logging.WARNING, logger=local_qwen_module.__name__):
        await session.send_audio_f32(np.ones(160, dtype=np.float32))
        await session.on_speech_end()
        await asyncio.wait_for(started.wait(), timeout=0.1)
        for _ in range(9):
            await session.send_audio_f32(np.ones(160, dtype=np.float32))
            await session.on_speech_end()

    warnings = [message for message in caplog.messages if "Decode backlog" in message]
    assert len(warnings) == 1
    assert "[STT][local_qwen][peer]" in warnings[0]
    assert "pending_jobs=9" in warnings[0]
    assert "buffered_audio_ms=90.0" in warnings[0]
    assert "threshold=8" in warnings[0]

    release.set()
    await session.stop()
