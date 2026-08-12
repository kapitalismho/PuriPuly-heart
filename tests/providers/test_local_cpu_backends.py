from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    InstalledLocalSTTManifest,
    LocalSTTInstallState,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import (
    LocalCPUInstallSnapshot,
    LocalCPUModelInstall,
)

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt import local_cpu as local_cpu_module
from puripuly_heart.providers.stt import local_parakeet_sherpa as parakeet_module
from puripuly_heart.providers.stt import local_qwen_sherpa as local_qwen_module
from puripuly_heart.providers.stt.local_cpu import (
    LocalCPUAutoSTTBackend,
    LocalCPUAutoUnavailableError,
    create_local_cpu_backend,
)
from puripuly_heart.providers.stt.local_parakeet_sherpa import (
    LocalParakeetJapaneseSherpaSTTBackend,
    LocalParakeetV3SherpaSTTBackend,
    create_local_parakeet_japanese_sherpa_recognizer,
    create_local_parakeet_v3_sherpa_recognizer,
)
from puripuly_heart.providers.stt.local_qwen_sherpa import (
    LocalQwenSherpaInferenceError,
    LocalQwenSherpaSTTBackend,
)


class _ConfigNode:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _TransducerConfig(_ConfigNode):
    def __init__(
        self,
        encoder_filename: str,
        decoder_filename: str,
        joiner_filename: str,
    ) -> None:
        super().__init__(
            encoder_filename=encoder_filename,
            decoder_filename=decoder_filename,
            joiner_filename=joiner_filename,
        )


class _NemoConfig(_ConfigNode):
    def __init__(self, model: str) -> None:
        super().__init__(model=model)


class _Recognizer:
    def __init__(self, config: object) -> None:
        self.config = config


def _fake_sherpa() -> object:
    return SimpleNamespace(
        OfflineTransducerModelConfig=_TransducerConfig,
        OfflineNemoEncDecCtcModelConfig=_NemoConfig,
        OfflineModelConfig=_ConfigNode,
        FeatureExtractorConfig=_ConfigNode,
        OfflineRecognizerConfig=_ConfigNode,
    )


def _ready_snapshot() -> LocalCPUInstallSnapshot:
    installs: list[LocalCPUModelInstall] = []
    for model_id in (PARAKEET_V3_MODEL_ID, PARAKEET_JAPANESE_MODEL_ID, LOCAL_STT_MODEL_ID):
        manifest = load_local_stt_asset_manifest(model_id)
        installed = InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source=next(iter(manifest.sources)),
            selected_revision=next(iter(manifest.sources.values())).revision,
        )
        installs.append(
            LocalCPUModelInstall(
                model_id=model_id,
                state=LocalSTTInstallState(status="ready", installed_manifest=installed),
            )
        )
    return LocalCPUInstallSnapshot(models=tuple(installs))


def test_parakeet_v3_recognizer_uses_transducer_asset_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parakeet_module, "_recognizer_class", lambda: (_fake_sherpa(), _Recognizer))

    recognizer = create_local_parakeet_v3_sherpa_recognizer(
        model_dir=Path("C:/models/parakeet-v3"),
        num_threads=3,
    )

    model_config = recognizer.config.model_config
    assert model_config.transducer.encoder_filename.endswith("encoder.int8.onnx")
    assert model_config.transducer.decoder_filename.endswith("decoder.int8.onnx")
    assert model_config.transducer.joiner_filename.endswith("joiner.int8.onnx")
    assert model_config.tokens.endswith("tokens.txt")
    assert model_config.model_type == "nemo_transducer"
    assert recognizer.config.feat_config.sampling_rate == 16000
    assert recognizer.config.feat_config.feature_dim == 80
    assert recognizer.config.decoding_method == "greedy_search"


def test_parakeet_japanese_recognizer_uses_nemo_ctc_asset_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parakeet_module, "_recognizer_class", lambda: (_fake_sherpa(), _Recognizer))

    recognizer = create_local_parakeet_japanese_sherpa_recognizer(
        model_dir=Path("C:/models/parakeet-ja"),
        num_threads=3,
    )

    model_config = recognizer.config.model_config
    assert model_config.nemo_ctc.model.endswith("model.int8.onnx")
    assert model_config.tokens.endswith("tokens.txt")
    assert recognizer.config.feat_config.sampling_rate == 16000
    assert recognizer.config.feat_config.feature_dim == 80


@pytest.mark.parametrize(
    ("model_id", "backend_type"),
    [
        (LOCAL_STT_MODEL_ID, LocalQwenSherpaSTTBackend),
        (PARAKEET_V3_MODEL_ID, LocalParakeetV3SherpaSTTBackend),
        (PARAKEET_JAPANESE_MODEL_ID, LocalParakeetJapaneseSherpaSTTBackend),
    ],
)
def test_direct_local_cpu_factory_targets_only_selected_model(
    model_id: str,
    backend_type: type[object],
    tmp_path: Path,
) -> None:
    manifest = load_local_stt_asset_manifest(model_id)

    backend = create_local_cpu_backend(
        model_id,
        model_root=tmp_path,
        source_language="ja",
        sample_rate_hz=16000,
        stream_label="peer",
    )

    assert isinstance(backend, backend_type)
    assert backend.model_id == model_id
    assert backend.model_dir == tmp_path / manifest.install_dirname
    assert backend.stream_label == "peer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_type",
    [
        LocalQwenSherpaSTTBackend,
        LocalParakeetV3SherpaSTTBackend,
        LocalParakeetJapaneseSherpaSTTBackend,
    ],
)
async def test_direct_local_cpu_models_preserve_full_audio_on_speech_end(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: type[LocalQwenSherpaSTTBackend],
) -> None:
    decoded: list[np.ndarray] = []

    async def ensure_recognizer(_self: LocalQwenSherpaSTTBackend) -> object:
        return object()

    async def decode(
        _self: LocalQwenSherpaSTTBackend,
        samples_f32: np.ndarray,
    ) -> str:
        decoded.append(samples_f32.copy())
        return "transcript"

    monkeypatch.setattr(backend_type, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(backend_type, "decode_f32", decode)
    backend = backend_type(model_dir=Path("C:/models/direct"))
    session = await backend.open_session()
    samples = np.arange(16_000, dtype=np.float32)

    await session.send_audio_f32(samples)
    await session.on_speech_end(trailing_silence_ms=400)
    event = await anext(session.events())

    assert event == STTBackendTranscriptEvent(text="transcript", is_final=True)
    assert len(decoded) == 1
    assert np.array_equal(decoded[0], samples)
    await session.close()
    await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_type", "model_id"),
    [
        (LocalParakeetV3SherpaSTTBackend, PARAKEET_V3_MODEL_ID),
        (LocalParakeetJapaneseSherpaSTTBackend, PARAKEET_JAPANESE_MODEL_ID),
    ],
)
async def test_parakeet_backend_validates_its_own_strict_manifest_before_load(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: type[LocalQwenSherpaSTTBackend],
    model_id: str,
) -> None:
    validations: list[tuple[Path, str]] = []

    def validate(model_dir: Path, *, manifest: object) -> object:
        validations.append((model_dir, manifest.model_id))
        return object()

    monkeypatch.setattr(local_qwen_module, "validate_local_stt_runtime_ready", validate)
    monkeypatch.setattr(backend_type, "_create_recognizer", lambda self: object())
    model_dir = Path(f"C:/models/{model_id}")
    backend = backend_type(model_dir=model_dir)

    session = await backend.open_session()

    assert validations == [(model_dir, model_id)]
    await session.close()
    await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_type",
    [
        LocalQwenSherpaSTTBackend,
        LocalParakeetV3SherpaSTTBackend,
        LocalParakeetJapaneseSherpaSTTBackend,
    ],
)
@pytest.mark.parametrize("blocked_stage", ["validation", "recognizer"])
async def test_direct_local_cpu_close_prevents_open_resurrection(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: type[LocalQwenSherpaSTTBackend],
    blocked_stage: str,
) -> None:
    stage_started = threading.Event()
    release_stage = threading.Event()
    recognizer = object()

    def validate(self: LocalQwenSherpaSTTBackend) -> None:
        _ = self
        if blocked_stage == "validation":
            stage_started.set()
            release_stage.wait(timeout=2.0)

    def create(self: LocalQwenSherpaSTTBackend) -> object:
        _ = self
        if blocked_stage == "recognizer":
            stage_started.set()
            release_stage.wait(timeout=2.0)
        return recognizer

    monkeypatch.setattr(backend_type, "_validate_runtime_assets", validate)
    monkeypatch.setattr(backend_type, "_create_recognizer", create)
    backend = backend_type(model_dir=Path("C:/models/direct"))
    open_task = asyncio.create_task(backend.open_session())
    assert await asyncio.to_thread(stage_started.wait, 1.0)

    close_task = asyncio.create_task(backend.close())
    await asyncio.sleep(0)

    assert close_task.done() is False
    release_stage.set()
    with pytest.raises(RuntimeError, match="closed"):
        await open_task
    await asyncio.wait_for(close_task, timeout=1.0)

    assert backend._recognizer is None
    with pytest.raises(RuntimeError, match="closed"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_cpu_auto_strict_gate_resolves_once_and_awaits_delegate_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _ready_snapshot()
    verification_modes: list[bool] = []

    def inspect(*_args: object, **kwargs: object) -> LocalCPUInstallSnapshot:
        verification_modes.append(bool(kwargs.get("verify_checksums")))
        return snapshot

    monkeypatch.setattr(
        local_cpu_module,
        "inspect_required_cpu_model_installs",
        inspect,
    )
    factory_calls: list[str] = []

    class Delegate:
        def __init__(self) -> None:
            self.close_calls = 0

        async def open_session(self) -> object:
            return object()

        async def close(self) -> None:
            self.close_calls += 1

    delegate = Delegate()

    def factory(model_id: str, **_kwargs: object) -> object:
        factory_calls.append(model_id)
        return delegate

    backend = LocalCPUAutoSTTBackend(
        source_language="ja-JP",
        stream_label="self",
        backend_factory=factory,
    )

    assert await backend.open_session() is not None
    assert await backend.open_session() is not None
    assert backend.resolved_model_id == PARAKEET_JAPANESE_MODEL_ID
    assert factory_calls == [PARAKEET_JAPANESE_MODEL_ID]
    assert verification_modes == [False]

    await backend.close()

    assert delegate.close_calls == 1
    assert backend.resolved_model_id is None
    with pytest.raises(RuntimeError, match="closed"):
        await backend.open_session()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_language", "expected_model_id"),
    [
        ("en", PARAKEET_V3_MODEL_ID),
        ("ja", PARAKEET_JAPANESE_MODEL_ID),
        ("zh-CN", LOCAL_STT_MODEL_ID),
    ],
)
async def test_cpu_auto_each_delegate_preserves_full_audio_on_speech_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_language: str,
    expected_model_id: str,
) -> None:
    monkeypatch.setattr(
        local_cpu_module,
        "inspect_required_cpu_model_installs",
        lambda *_args, **_kwargs: _ready_snapshot(),
    )
    decoded: list[tuple[str, np.ndarray]] = []

    async def ensure_recognizer(_self: LocalQwenSherpaSTTBackend) -> object:
        return object()

    async def decode(
        self: LocalQwenSherpaSTTBackend,
        samples_f32: np.ndarray,
    ) -> str:
        decoded.append((self.model_id, samples_f32.copy()))
        return "transcript"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode)
    backend = LocalCPUAutoSTTBackend(
        source_language=source_language,
        model_root=tmp_path,
    )
    session = await backend.open_session()
    samples = np.arange(16_000, dtype=np.float32)

    await session.send_audio_f32(samples)
    await session.on_speech_end(trailing_silence_ms=400)
    event = await anext(session.events())

    assert event == STTBackendTranscriptEvent(text="transcript", is_final=True)
    assert backend.resolved_model_id == expected_model_id
    assert len(decoded) == 1
    assert decoded[0][0] == expected_model_id
    assert np.array_equal(decoded[0][1], samples)
    await session.close()
    await backend.close()


@pytest.mark.asyncio
async def test_cpu_auto_rejects_partial_install_without_constructing_direct_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _ready_snapshot()
    invalid = LocalCPUModelInstall(
        model_id=PARAKEET_V3_MODEL_ID,
        state=LocalSTTInstallState(status="invalid", error_message="checksum mismatch"),
    )
    partial = LocalCPUInstallSnapshot(models=(invalid, *snapshot.models[1:]))
    monkeypatch.setattr(
        local_cpu_module,
        "inspect_required_cpu_model_installs",
        lambda *_args, **_kwargs: partial,
    )

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("direct backend must not be constructed")

    backend = LocalCPUAutoSTTBackend(
        source_language="en",
        backend_factory=forbidden_factory,
    )

    with pytest.raises(LocalCPUAutoUnavailableError) as raised:
        await backend.open_session()

    assert raised.value.snapshot is partial
    assert backend.resolved_model_id is None
    await backend.close()


@pytest.mark.asyncio
async def test_cpu_auto_close_during_validation_prevents_late_delegate_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection_started = threading.Event()
    release_inspection = threading.Event()

    def inspect(*_args: object, **_kwargs: object) -> LocalCPUInstallSnapshot:
        inspection_started.set()
        release_inspection.wait(timeout=2.0)
        return _ready_snapshot()

    monkeypatch.setattr(local_cpu_module, "inspect_required_cpu_model_installs", inspect)

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("closed backend must not construct a delegate")

    backend = LocalCPUAutoSTTBackend(
        source_language="en",
        backend_factory=forbidden_factory,
    )
    open_task = asyncio.create_task(backend.open_session())
    await asyncio.to_thread(inspection_started.wait, 1.0)

    await backend.close()
    release_inspection.set()

    with pytest.raises(RuntimeError, match="closed"):
        await open_task


@pytest.mark.asyncio
async def test_cpu_auto_cancelled_open_keeps_inspection_owned_until_close_can_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection_started = threading.Event()
    release_inspection = threading.Event()
    inspection_finished = threading.Event()

    def inspect(*_args: object, **_kwargs: object) -> LocalCPUInstallSnapshot:
        inspection_started.set()
        release_inspection.wait(timeout=2.0)
        inspection_finished.set()
        return _ready_snapshot()

    monkeypatch.setattr(local_cpu_module, "inspect_required_cpu_model_installs", inspect)

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cancelled open must not construct a delegate")

    backend = LocalCPUAutoSTTBackend(
        source_language="en",
        backend_factory=forbidden_factory,
    )
    open_task = asyncio.create_task(backend.open_session())
    assert await asyncio.to_thread(inspection_started.wait, 1.0)
    open_task.cancel()
    await asyncio.sleep(0)

    close_task = asyncio.create_task(backend.close())
    await asyncio.sleep(0.02)

    assert open_task.done() is False
    assert close_task.done() is False
    assert inspection_finished.is_set() is False
    release_inspection.set()
    with pytest.raises(asyncio.CancelledError):
        await open_task
    await asyncio.wait_for(close_task, timeout=1.0)

    assert inspection_finished.is_set() is True


@pytest.mark.asyncio
async def test_cpu_auto_close_during_delegate_open_retires_late_session_and_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_cpu_module,
        "inspect_required_cpu_model_installs",
        lambda *_args, **_kwargs: _ready_snapshot(),
    )
    open_started = asyncio.Event()
    release_open = asyncio.Event()

    class Session:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    class Delegate:
        def __init__(self) -> None:
            self.session = Session()
            self.close_calls = 0

        async def open_session(self) -> object:
            open_started.set()
            await release_open.wait()
            return self.session

        async def close(self) -> None:
            self.close_calls += 1

    delegate = Delegate()
    backend = LocalCPUAutoSTTBackend(
        source_language="en",
        backend_factory=lambda *_args, **_kwargs: delegate,
    )
    open_task = asyncio.create_task(backend.open_session())
    await asyncio.wait_for(open_started.wait(), timeout=0.1)

    close_task = asyncio.create_task(backend.close())
    await asyncio.sleep(0)
    release_open.set()

    with pytest.raises(RuntimeError, match="closed"):
        await open_task
    await close_task
    assert delegate.session.close_calls == 1
    assert delegate.close_calls == 1


@pytest.mark.asyncio
async def test_local_cpu_attempt_diagnostic_separates_queue_wait_and_decode_rtf(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue_now = 1.0
    decode_times = iter((10.0, 10.5))
    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("C:/models/qwen"),
        stream_label="self",
        queue_clock=lambda: queue_now,
        decode_clock=lambda: next(decode_times),
        diagnostics_enabled=lambda: True,
    )

    async def ensure_recognizer() -> object:
        return object()

    async def decode(_samples: np.ndarray) -> str:
        return "private transcript"

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(backend, "decode_f32", decode)

    session = await backend.open_session()
    await session.send_audio_f32(np.ones(16000, dtype=np.float32))
    with caplog.at_level(
        logging.INFO,
        logger="puripuly_heart.providers.stt.local_qwen_sherpa",
    ):
        await session.on_speech_end()
        queue_now = 1.25
        event = await anext(session.events())

    assert event == STTBackendTranscriptEvent(text="private transcript", is_final=True)
    attempt = next(message for message in caplog.messages if "[LocalASR][Attempt]" in message)
    assert "channel=self" in attempt
    assert f"model={LOCAL_STT_MODEL_ID}" in attempt
    assert "backend=CPU" in attempt
    assert "audio_seconds=1.000" in attempt
    assert "decode_seconds=0.500" in attempt
    assert "rtf=0.500000" in attempt
    assert "result=success" in attempt
    assert "queue_wait_seconds=0.250" in attempt
    assert "private transcript" not in attempt
    await session.close()


@pytest.mark.asyncio
async def test_local_cpu_failed_started_attempt_retains_decode_timing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    decode_times = iter((20.0, 20.25))
    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("C:/models/qwen"),
        stream_label="peer",
        queue_clock=lambda: 3.0,
        decode_clock=lambda: next(decode_times),
        diagnostics_enabled=lambda: True,
    )

    async def ensure_recognizer() -> object:
        return object()

    async def decode(_samples: np.ndarray) -> str:
        raise LocalQwenSherpaInferenceError("private failure detail")

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(backend, "decode_f32", decode)
    session = await backend.open_session()
    await session.send_audio_f32(np.ones(8000, dtype=np.float32))

    events = session.events()
    with caplog.at_level(
        logging.INFO,
        logger="puripuly_heart.providers.stt.local_qwen_sherpa",
    ):
        await session.on_speech_end()
        boundary = await anext(events)
        with pytest.raises(LocalQwenSherpaInferenceError, match="private failure detail"):
            await anext(events)

    assert boundary == STTBackendTranscriptEvent(text="", is_final=True)
    attempt = next(message for message in caplog.messages if "[LocalASR][Attempt]" in message)
    assert "channel=peer" in attempt
    assert f"model={LOCAL_STT_MODEL_ID}" in attempt
    assert "audio_seconds=0.500" in attempt
    assert "decode_seconds=0.250" in attempt
    assert "rtf=0.500000" in attempt
    assert "result=failure" in attempt
    assert "private failure detail" not in attempt
    await session.close()


@pytest.mark.asyncio
async def test_local_cpu_expiry_emits_boundary_and_safe_diagnostic_without_decode(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue_now = 0.0
    decode_times = iter((1.0, 1.1))
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    decoded = 0
    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("C:/models/qwen"),
        stream_label="self",
        queue_clock=lambda: queue_now,
        decode_clock=lambda: next(decode_times),
        diagnostics_enabled=lambda: True,
    )

    async def ensure_recognizer() -> object:
        return object()

    async def decode(_samples: np.ndarray) -> str:
        nonlocal decoded
        decoded += 1
        first_started.set()
        await release_first.wait()
        return "first result"

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(backend, "decode_f32", decode)
    session = await backend.open_session()
    await session.send_audio_f32(np.ones(160, dtype=np.float32))
    await session.on_speech_end()
    await asyncio.wait_for(first_started.wait(), timeout=0.1)
    await session.send_audio_f32(np.full(160, 2.0, dtype=np.float32))
    await session.on_speech_end()

    events = session.events()
    with caplog.at_level(
        logging.INFO,
        logger="puripuly_heart.providers.stt.local_qwen_sherpa",
    ):
        queue_now = 12.0
        release_first.set()
        first = await anext(events)
        expired = await anext(events)

    assert decoded == 1
    assert first == STTBackendTranscriptEvent(text="first result", is_final=True)
    assert expired == STTBackendTranscriptEvent(text="", is_final=True)
    expiry = next(message for message in caplog.messages if "[LocalASR][Expiry]" in message)
    assert "channel=self" in expiry
    assert f"model={LOCAL_STT_MODEL_ID}" in expiry
    assert "intended_provider=local_qwen" in expiry
    assert "reason=pending_ttl_exceeded" in expiry
    assert "queue_wait_seconds=12.000" in expiry
    assert "first result" not in expiry
    await session.close()
