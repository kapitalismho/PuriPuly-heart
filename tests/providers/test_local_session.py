from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.core.local_asr.local_stt_assets import (
    InstalledLocalSTTManifest,
    LocalSTTManifestInvalidError,
    LocalSTTModelMissingError,
)
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt import local_session as local_session_module
from puripuly_heart.providers.stt.local_parakeet_sherpa import (
    LocalParakeetJapaneseSherpaSTTBackend,
    LocalParakeetSherpaInferenceError,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("stage", ["validation", "recognizer", "inference"])
async def test_local_session_synchronous_heavy_stage_runs_inside_to_thread_heartbeat(
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

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), stream_label=channel
    )
    monkeypatch.setattr(
        local_session_module, "validate_local_stt_runtime_ready", lambda _path, **_kwargs: None
    )
    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_create_recognizer", lambda self: object()
    )
    if stage == "validation":
        monkeypatch.setattr(
            local_session_module,
            "validate_local_stt_runtime_ready",
            lambda _path, **_kwargs: blocked(),
        )
        operation = asyncio.create_task(backend.open_session())
    elif stage == "recognizer":
        monkeypatch.setattr(
            LocalParakeetJapaneseSherpaSTTBackend,
            "_create_recognizer",
            lambda self: blocked() or object(),
        )
        operation = asyncio.create_task(backend.open_session())
    else:
        backend._recognizer = object()
        monkeypatch.setattr(
            LocalParakeetJapaneseSherpaSTTBackend,
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


def test_local_session_backend_uses_thread_count_2_by_default() -> None:
    assert local_session_module.DEFAULT_SHERPA_NUM_THREADS == 2
    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
    assert backend.num_threads == 2


def _installed_manifest() -> InstalledLocalSTTManifest:
    return InstalledLocalSTTManifest(
        manifest_version=1,
        model_id="parakeet-tdt-ctc-0.6b-ja-int8-sherpa",
        engine="sherpa-onnx",
        install_dirname="parakeet-tdt-ctc-0.6b-ja-int8-sherpa",
        selected_source="huggingface",
        selected_revision="rev-1",
    )


def _install_fake_sherpa(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recognizer_factory,
) -> None:
    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend,
        "_create_recognizer",
        lambda self: recognizer_factory(None),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["leşme", "acia"])
async def test_local_session_backend_emits_final_event_for_known_hallucination_text(
    monkeypatch: pytest.MonkeyPatch,
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"),
        sample_rate_hz=16000,
        stream_label="self",
    )
    session = await backend.open_session()

    await session.send_audio(b"\x00\x00\xff\x7f")
    await session.on_speech_end()
    await session.stop()

    gen = session.events()
    event = await gen.__anext__()

    assert event.text == text
    assert event.is_final is True


@pytest.mark.asyncio
async def test_local_session_session_send_audio_f32_preserves_float32_samples(
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), sample_rate_hz=16000
    )
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
async def test_local_session_session_repeated_speech_end_decodes_and_clears_each_segment(
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=make_recognizer)

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), sample_rate_hz=16000
    )
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
async def test_local_session_session_clips_float32_before_accept_waveform(
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), sample_rate_hz=16000
    )
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


def test_local_session_backend_rejects_legacy_8000_runtime_sample_rate() -> None:
    with pytest.raises(ValueError, match="16000"):
        LocalParakeetJapaneseSherpaSTTBackend(
            model_dir=Path("/models/japanese"), sample_rate_hz=8000
        )


@pytest.mark.asyncio
async def test_local_session_backend_keeps_16000_input_without_resampling(
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), sample_rate_hz=16000
    )
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00\xff\x7f")
    await session.on_speech_end()
    await session.stop()

    stream = recognizer_state["stream"]
    assert isinstance(stream, FakeStream)
    accepted_rate, accepted_samples = stream.accepted[0]
    assert accepted_rate == 16000


@pytest.mark.asyncio
async def test_local_session_backend_runtime_validator_runs_only_until_recognizer_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def fake_runtime_ready(model_dir: Path, **_kwargs) -> InstalledLocalSTTManifest:
        calls.append(model_dir)
        return _installed_manifest()

    monkeypatch.setattr(
        local_session_module, "validate_local_stt_runtime_ready", fake_runtime_ready
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: SimpleNamespace())

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), sample_rate_hz=16000
    )

    await backend.open_session()
    await backend.open_session()

    assert calls == [Path("/models/japanese")]


@pytest.mark.asyncio
async def test_local_session_backend_close_is_terminal_and_replacement_revalidates_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def fake_runtime_ready(model_dir: Path, **_kwargs) -> InstalledLocalSTTManifest:
        calls.append(model_dir)
        return _installed_manifest()

    monkeypatch.setattr(
        local_session_module, "validate_local_stt_runtime_ready", fake_runtime_ready
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: SimpleNamespace())

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"), sample_rate_hz=16000
    )

    await backend.open_session()
    await backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        await backend.open_session()

    replacement = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=Path("/models/japanese"),
        sample_rate_hz=16000,
    )
    await replacement.open_session()

    assert calls == [Path("/models/japanese"), Path("/models/japanese")]


@pytest.mark.asyncio
async def test_local_session_backend_surfaces_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(LocalSTTModelMissingError("missing")),
    )

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))

    with pytest.raises(LocalSTTModelMissingError, match="missing"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_local_session_backend_surfaces_invalid_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            LocalSTTManifestInvalidError("manifest invalid")
        ),
    )

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))

    with pytest.raises(LocalSTTManifestInvalidError, match="manifest invalid"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_local_session_session_surfaces_inference_error(
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00")
    await session.on_speech_end()

    gen = session.events()
    boundary = await gen.__anext__()
    assert boundary.text == ""
    assert boundary.is_final is True
    with pytest.raises(LocalParakeetSherpaInferenceError, match="decode failed"):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_local_session_session_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
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
        local_session_module,
        "validate_local_stt_runtime_ready",
        lambda *args, **kwargs: _installed_manifest(),
    )
    _install_fake_sherpa(monkeypatch, recognizer_factory=lambda _config: FakeRecognizer())

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
    session = await backend.open_session()

    await session.close()
    await session.close()

    gen = session.events()
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_local_session_speech_end_queues_fifo_decode_without_blocking(
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

    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer
    )
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
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
async def test_local_session_queued_payload_survives_mutable_producer_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded: list[np.ndarray] = []
    hold = asyncio.Event()
    started = asyncio.Event()

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        decoded.append(samples_f32.copy())
        started.set()
        await hold.wait()
        return "valid"

    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer
    )
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "decode_f32", decode_f32)
    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
    session = await backend.open_session()
    pcm = bytearray(np.array([-32768, -1, 0, 16384, 32767], dtype="<i2").tobytes())
    direct = np.array([0.125, -0.625], dtype=np.float32)
    await session.send_audio(pcm)
    await session.send_audio_f32(direct)
    pcm[:] = b"\x00" * len(pcm)
    direct[:] = 0
    await session.on_speech_end()
    await asyncio.wait_for(started.wait(), timeout=1)
    hold.set()
    assert (await asyncio.wait_for(anext(session.events()), timeout=1)).text == "valid"
    np.testing.assert_array_equal(
        decoded[0],
        np.array([-1.0, -1 / 32768, 0, 0.5, 32767 / 32768, 0.125, -0.625], dtype=np.float32),
    )
    await session.close()


@pytest.mark.asyncio
async def test_local_session_sessions_handoff_finals_before_next_session_decodes(
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

    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer
    )
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
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
async def test_local_session_close_cancels_and_awaits_decode_worker(
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

    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer
    )
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
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
async def test_local_session_session_close_waits_for_blocking_native_decode_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_started = threading.Event()
    native_release = threading.Event()
    native_finished = threading.Event()

    def blocking_decode(
        self: LocalParakeetJapaneseSherpaSTTBackend,
        recognizer: object,
        samples_f32: np.ndarray,
    ) -> str:
        _ = self, recognizer, samples_f32
        native_started.set()
        native_release.wait(timeout=2.0)
        native_finished.set()
        return "late native result"

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
    backend._recognizer = object()
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "_decode_f32_sync", blocking_decode)
    session = await backend.open_session()
    await session.send_audio_f32(np.ones(160, dtype=np.float32))
    await session.on_speech_end()
    assert await asyncio.to_thread(native_started.wait, 1.0)

    close_task = asyncio.create_task(session.close())
    await asyncio.sleep(0.02)

    assert close_task.done() is False
    assert native_finished.is_set() is False
    native_release.set()
    await asyncio.wait_for(close_task, timeout=1.0)

    assert native_finished.is_set() is True
    events = session.events()
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()
    await backend.close()


@pytest.mark.asyncio
async def test_local_session_provider_close_waits_for_direct_blocking_native_decode_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_started = threading.Event()
    native_release = threading.Event()
    native_finished = threading.Event()

    def blocking_decode(
        self: LocalParakeetJapaneseSherpaSTTBackend,
        recognizer: object,
        samples_f32: np.ndarray,
    ) -> str:
        _ = self, recognizer, samples_f32
        native_started.set()
        native_release.wait(timeout=2.0)
        native_finished.set()
        return "native result"

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
    backend._recognizer = object()
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "_decode_f32_sync", blocking_decode)
    decode_task = asyncio.create_task(backend.decode_f32(np.ones(160, dtype=np.float32)))
    assert await asyncio.to_thread(native_started.wait, 1.0)

    close_task = asyncio.create_task(backend.close())
    await asyncio.sleep(0.02)

    assert close_task.done() is False
    assert native_finished.is_set() is False
    native_release.set()
    assert await asyncio.wait_for(decode_task, timeout=1.0) == "native result"
    await asyncio.wait_for(close_task, timeout=1.0)

    assert native_finished.is_set() is True
    with pytest.raises(RuntimeError, match="closed"):
        await backend.decode_f32(np.ones(160, dtype=np.float32))


@pytest.mark.asyncio
async def test_local_session_empty_decode_preserves_next_final_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(["", "second final"])

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        _ = samples_f32
        return next(results)

    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer
    )
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
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
async def test_local_session_speech_end_without_audio_emits_empty_final_boundary(
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

    monkeypatch.setattr(
        LocalParakeetJapaneseSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer
    )
    monkeypatch.setattr(LocalParakeetJapaneseSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalParakeetJapaneseSherpaSTTBackend(model_dir=Path("/models/japanese"))
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
