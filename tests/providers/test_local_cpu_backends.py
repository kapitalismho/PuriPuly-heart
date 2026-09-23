from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

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

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.stt.backend import (
    STTBackendTranscriptEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
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

SCOPED_PROJECTION = STTSessionProjection(mode="scoped", provider_epoch_id="local-epoch")


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


def _scoped_request(
    order: int = 1,
    *,
    channel: str = "peer",
) -> STTProviderTurnRequest:
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1,
            segment_order=order,
            segment_id=uuid4(),
            capture_epoch=1,
        ),
        provider_epoch_id="local-epoch",
        provider_turn_id=f"local-turn-{order}",
    )
    return STTProviderTurnRequest(
        identity=identity,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="local_qwen",
            provider_signature=("local_qwen",),
            runtime_signature=("local_qwen",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.4,
            vad_hangover_ms=800,
            vad_pre_roll_ms=500,
        ),
        channel=channel,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_type",
    [
        pytest.param(LocalQwenSherpaSTTBackend, id="qwen"),
        pytest.param(LocalParakeetV3SherpaSTTBackend, id="parakeet-v3"),
        pytest.param(LocalParakeetJapaneseSherpaSTTBackend, id="parakeet-ja"),
    ],
)
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_local_cpu_scoped_decode_snapshots_pcm_and_emits_one_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_type: type[LocalQwenSherpaSTTBackend],
    channel: str,
) -> None:
    backend = backend_type(model_dir=tmp_path)

    async def ensure() -> object:
        return object()

    decoded: list[np.ndarray] = []

    async def decode(samples: np.ndarray) -> str:
        decoded.append(samples.copy())
        return "same same"

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure)
    monkeypatch.setattr(backend, "decode_f32", decode)
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    request = _scoped_request(channel=channel)
    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.outcome == "final"
    assert terminal.text == "same same"
    assert len(decoded) == 1
    assert decoded[0].size == 160
    await session.close()
    await backend.close()


@pytest.mark.asyncio
async def test_local_cpu_overlap_preserves_queued_successor_after_predecessor_decode_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = LocalQwenSherpaSTTBackend(model_dir=tmp_path)
    decode_gate = asyncio.Event()
    decode_calls = 0

    async def ensure() -> object:
        return object()

    async def decode(_samples: np.ndarray) -> str:
        nonlocal decode_calls
        decode_calls += 1
        await decode_gate.wait()
        if decode_calls == 1:
            raise RuntimeError("first decode failed")
        return "b-text"

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure)
    monkeypatch.setattr(backend, "decode_f32", decode)
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    first = _scoped_request(1)
    second = _scoped_request(2)

    for request in (first, second):
        await session.begin_turn(request)
        await session.send_turn_audio(
            request.identity,
            b"\x00\x40" * 160,
            payload_sequence=1,
            source_ranges=(),
            context_only=False,
        )
        await session.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="silence",
            observed_trailing_silence_ms=800,
        )

    decode_gate.set()
    events = session.turn_events()
    first_terminal = await asyncio.wait_for(events.__anext__(), timeout=1)
    second_terminal = await asyncio.wait_for(events.__anext__(), timeout=1)

    assert decode_calls == 2
    assert (
        first_terminal.identity,
        first_terminal.outcome,
        first_terminal.epoch_disposition,
    ) == (first.identity, "failed", "retire")
    assert (second_terminal.identity, second_terminal.outcome, second_terminal.text) == (
        second.identity,
        "final",
        "b-text",
    )
    await session.close()
    await backend.close()


@pytest.mark.asyncio
async def test_local_cpu_scoped_empty_error_and_close_terminal_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = LocalQwenSherpaSTTBackend(model_dir=tmp_path)

    async def ensure() -> object:
        return object()

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure)
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    empty = _scoped_request(1)
    await session.begin_turn(empty)
    await session.seal_turn(
        empty.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "empty"
    await session.close()

    async def fail_decode(_samples: np.ndarray) -> str:
        raise RuntimeError("native decode error")

    monkeypatch.setattr(backend, "decode_f32", fail_decode)
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    failed = _scoped_request(2)
    await session.begin_turn(failed)
    await session.send_turn_audio(
        failed.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        failed.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"
    await session.close()

    session = await backend.open_session(projection=SCOPED_PROJECTION)
    closed = _scoped_request(3)
    await session.begin_turn(closed)
    stream = session.turn_events()
    await session.close()
    terminal = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "session_closed"
    await backend.close()


@pytest.mark.asyncio
async def test_local_cpu_active_timeout_holds_handoff_and_model_until_repeated_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = LocalQwenSherpaSTTBackend(
        model_dir=tmp_path,
        active_decode_timeout_s=0.01,
    )
    recognizer = object()
    backend._recognizer = recognizer
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def blocking_owned_call(_operation: object) -> str:
        nonlocal calls
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
            raise
        return "late result"

    monkeypatch.setattr(local_qwen_module, "run_owned_thread_call", blocking_owned_call)
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    request = _scoped_request(1)
    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    with pytest.raises(RuntimeError, match="already sealed"):
        await session.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="duplicate",
            observed_trailing_silence_ms=800,
        )
    await asyncio.wait_for(started.wait(), timeout=1)
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "local_decode_timeout"
    assert terminal.epoch_disposition == "retire"

    backend.active_decode_timeout_s = 1
    replacement = await backend.open_session(projection=SCOPED_PROJECTION)
    replacement_request = _scoped_request(2)
    await replacement.begin_turn(replacement_request)
    await replacement.send_turn_audio(
        replacement_request.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await replacement.seal_turn(
        replacement_request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    await asyncio.sleep(0.02)
    assert calls == 1
    assert backend._recognizer is recognizer

    first_off = asyncio.create_task(session.abort_for_toggle_off())
    second_off = asyncio.create_task(session.abort_for_toggle_off())
    await asyncio.sleep(0)
    assert not first_off.done()
    assert calls == 1
    assert backend._recognizer is recognizer
    release.set()
    await asyncio.wait_for(first_off, timeout=1)
    await asyncio.wait_for(second_off, timeout=1)

    terminal = await asyncio.wait_for(replacement.turn_events().__anext__(), timeout=1)
    assert terminal.identity == replacement_request.identity
    assert terminal.outcome == "final"
    assert terminal.text == "late result"
    assert calls == 2
    await replacement.close()
    assert backend._recognizer is recognizer
    await backend.close()
    assert backend._recognizer is None


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

        async def open_session(self, **_kwargs: object) -> object:
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
@pytest.mark.parametrize(
    ("source_language", "expected_model_id"),
    [
        pytest.param("en", PARAKEET_V3_MODEL_ID, id="parakeet-v3-auto"),
        pytest.param("ja", PARAKEET_JAPANESE_MODEL_ID, id="parakeet-ja-auto"),
        pytest.param("zh-CN", LOCAL_STT_MODEL_ID, id="qwen-auto"),
    ],
)
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_cpu_auto_aliases_delegate_scoped_turn_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_language: str,
    expected_model_id: str,
    channel: str,
) -> None:
    monkeypatch.setattr(
        local_cpu_module,
        "inspect_required_cpu_model_installs",
        lambda *_args, **_kwargs: _ready_snapshot(),
    )

    async def ensure_recognizer(_self: LocalQwenSherpaSTTBackend) -> object:
        return object()

    async def decode(
        self: LocalQwenSherpaSTTBackend,
        _samples_f32: np.ndarray,
    ) -> str:
        return self.model_id

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode)
    backend = LocalCPUAutoSTTBackend(
        source_language=source_language,
        model_root=tmp_path,
    )
    session = await backend.open_session(projection=SCOPED_PROJECTION)
    request = _scoped_request(channel=channel)
    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity,
        b"\x00\x40" * 160,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.identity == request.identity
    assert terminal.outcome == "final"
    assert terminal.text == expected_model_id
    assert backend.resolved_model_id == expected_model_id
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

        async def open_session(self, **_kwargs: object) -> object:
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
async def test_local_cpu_attempt_performance_separates_queue_wait_and_decode_rtf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_now = 1.0
    decode_times = iter((10.0, 10.5))
    basic_logs: list[tuple[str, int]] = []
    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("C:/models/qwen"),
        stream_label="self",
        queue_clock=lambda: queue_now,
        decode_clock=lambda: next(decode_times),
        attempt_log_sink=lambda message, level: basic_logs.append((message, level)),
    )

    async def ensure_recognizer() -> object:
        return object()

    async def decode(_samples: np.ndarray) -> str:
        return "private transcript"

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(backend, "decode_f32", decode)

    session = await backend.open_session()
    await session.send_audio_f32(np.ones(16000, dtype=np.float32))
    await session.on_speech_end()
    queue_now = 1.25
    event = await anext(session.events())
    assert event == STTBackendTranscriptEvent(text="private transcript", is_final=True)
    assert len(basic_logs) == 1
    attempt, level = basic_logs[0]
    assert level == logging.INFO
    assert "[Self · Recognition]" in attempt
    assert "Audio 1.00 s" in attempt
    assert "Decode 0.50 s" in attempt
    assert "RTF 0.500" in attempt
    assert "Result success" in attempt
    assert "Queue 0.25 s" in attempt
    assert "private transcript" not in attempt
    await session.close()


@pytest.mark.asyncio
async def test_local_cpu_zero_audio_does_not_publish_fabricated_attempt_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basic_logs: list[tuple[str, int]] = []
    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("C:/models/qwen"),
        stream_label="self",
        attempt_log_sink=lambda message, level: basic_logs.append((message, level)),
    )

    async def ensure_recognizer() -> object:
        return object()

    async def decode(_samples: np.ndarray) -> str:
        return ""

    monkeypatch.setattr(backend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(backend, "decode_f32", decode)
    session = await backend.open_session()

    await session.on_speech_end()
    event = await anext(session.events())

    assert event == STTBackendTranscriptEvent(text="", is_final=True)
    assert basic_logs == []
    await session.close()


@pytest.mark.asyncio
async def test_local_cpu_failed_started_attempt_retains_decode_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_times = iter((20.0, 20.25))
    basic_logs: list[tuple[str, int]] = []
    backend = LocalQwenSherpaSTTBackend(
        model_dir=Path("C:/models/qwen"),
        stream_label="peer",
        queue_clock=lambda: 3.0,
        decode_clock=lambda: next(decode_times),
        attempt_log_sink=lambda message, level: basic_logs.append((message, level)),
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
    await session.on_speech_end()
    boundary = await anext(events)
    with pytest.raises(LocalQwenSherpaInferenceError, match="private failure detail"):
        await anext(events)

    assert boundary == STTBackendTranscriptEvent(text="", is_final=True)
    assert len(basic_logs) == 1
    attempt, level = basic_logs[0]
    assert level == logging.INFO
    assert "[Peer · Recognition]" in attempt
    assert "Audio 0.50 s" in attempt
    assert "Decode 0.25 s" in attempt
    assert "RTF 0.500" in attempt
    assert "Result failure" in attempt
    await session.close()


@pytest.mark.asyncio
async def test_local_cpu_expiry_emits_boundary_without_decode(
    monkeypatch: pytest.MonkeyPatch,
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
    queue_now = 12.0
    release_first.set()
    first = await anext(events)
    expired = await anext(events)

    assert decoded == 1
    assert first == STTBackendTranscriptEvent(text="first result", is_final=True)
    assert expired == STTBackendTranscriptEvent(text="", is_final=True)
    await session.close()
