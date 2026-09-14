from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from puripuly_heart.providers.stt import gemini_transcribe as gemini_module
from puripuly_heart.providers.stt.gemini_transcribe import (
    GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ,
    GEMINI_TRANSCRIBE_STT_MODEL,
    GeminiTranscribeSTTBackend,
    _build_live_config_sync,
    _GeminiTranscribeLiveSession,
    _prepare_gemini_resources_sync,
)

CONNECT_TIMEOUT_S = 0.05
HANGING_ENTER_S = 0.5
REPLACEMENT_ROUNDS = 5


class _FakeLiveSession:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def receive(self):
        while True:
            item = await self._queue.get()
            if item is _DONE:
                return
            yield item

    def push(self, item) -> None:
        self._queue.put_nowait(item)

    async def close(self) -> None:
        self.closed = True


_DONE = object()


class _FakeLiveContext:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session
        self.exited = False
        self.exit_calls = 0

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        self.exited = True
        await self._session.close()
        return False


class _RecordingFactory:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session
        self.calls: list = []
        self.context: _FakeLiveContext | None = None

    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        self.context = _FakeLiveContext(self._session)
        return self.context


class _HangingLiveContext(_FakeLiveContext):
    async def __aenter__(self):
        await asyncio.sleep(HANGING_ENTER_S)
        return self._session


class _HangingFactory(_RecordingFactory):
    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        self.context = _HangingLiveContext(self._session)
        return self.context


class _RefusingLiveContext(_FakeLiveContext):
    async def __aenter__(self):
        raise ConnectionError("boom-connect")


class _RefusingFactory(_RecordingFactory):
    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        self.context = _RefusingLiveContext(self._session)
        return self.context


class _StubSyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    @property
    def is_closed(self) -> bool:
        return self.closed


class _StubClientAio:
    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _StubClient:
    def __init__(self) -> None:
        self.aio = _StubClientAio()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _StubAsyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.closed = True

    @property
    def is_closed(self) -> bool:
        return self.closed


def _stub_resources(language_codes, custom_vocabulary):
    return gemini_module._GeminiClientResources(
        client=_StubClient(),
        sync_transport=_StubSyncTransport(),
        async_transport=_StubAsyncTransport(),
        config=_build_live_config_sync(language_codes, custom_vocabulary),
    )


class _FinallyGatedLiveContext(_FakeLiveContext):
    def __init__(self, session: _FakeLiveSession, gate, entered_finally) -> None:
        super().__init__(session)
        self._gate = gate
        self._entered_finally = entered_finally
        self.enter_finally_done = False

    async def __aenter__(self):
        try:
            await asyncio.sleep(30)
        finally:
            self._entered_finally.append(True)
            await asyncio.to_thread(self._gate.wait, timeout=10)
            self.enter_finally_done = True
        return self._session


class _FinallyGatedFactory(_RecordingFactory):
    def __init__(self, session: _FakeLiveSession, gate, entered_finally) -> None:
        super().__init__(session)
        self._gate = gate
        self._entered_finally = entered_finally

    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        self.context = _FinallyGatedLiveContext(self._session, self._gate, self._entered_finally)
        return self.context


def _patch_prepare(monkeypatch, made, *, gate=None, entered=None):
    def fake_prepare(api_key, language_codes, custom_vocabulary):
        if entered is not None:
            entered.set()
        if gate is not None:
            assert gate.wait(timeout=10)
        resources = _stub_resources(language_codes, custom_vocabulary)
        made.append(resources)
        return resources

    monkeypatch.setattr(gemini_module, "_prepare_gemini_resources_sync", fake_prepare)


def _make_backend(live, **kwargs):
    factory = _RecordingFactory(live)
    backend = GeminiTranscribeSTTBackend(
        api_key="dummy-offline-key",
        language_codes=("ko-KR",),
        live_connect_factory=factory,
        **kwargs,
    )
    return backend, factory


def _direct_session(factory, *, connect_timeout_s=10.0):
    return _GeminiTranscribeLiveSession(
        api_key="dummy-offline-key",
        language_codes=["ko-KR"],
        custom_vocabulary=[],
        model=GEMINI_TRANSCRIBE_STT_MODEL,
        sample_rate_hz=GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ,
        connect_timeout_s=connect_timeout_s,
        finalize_timeout_s=2.0,
        live_connect_factory=factory,
    )


def _assert_resources_released_once(resources) -> None:
    assert resources.client.aio.aclose_calls == 1
    assert resources.client.close_calls == 1
    assert resources.sync_transport.close_calls == 1
    assert resources.sync_transport.closed is True
    assert resources.async_transport.aclose_calls == 1
    assert resources.async_transport.closed is True


async def _wait_for_predicate(predicate, timeout=5.0) -> None:
    async def ready() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=timeout)


@pytest.mark.asyncio
async def test_offline_sdk_construction_and_dual_close() -> None:
    from google.genai import Client

    live = _FakeLiveSession()
    backend, factory = _make_backend(live)
    stt = await backend.open_session()
    try:
        model, config = factory.calls[0]
        assert model == GEMINI_TRANSCRIBE_STT_MODEL
        raw = config.model_dump(exclude_none=True)
        assert raw["response_modalities"] == ["TEXT"]
        assert raw["input_audio_transcription"]["mode"] == "VERBATIM"
        assert raw["input_audio_transcription"]["language_codes"] == ["ko-KR"]
        assert raw["realtime_input_config"]["automatic_activity_detection"]["disabled"] is True
        resources = stt._client_resources
        assert isinstance(resources.client, Client)
        assert isinstance(resources.sync_transport, httpx.Client)
        assert isinstance(resources.async_transport, httpx.AsyncClient)
        await stt.send_audio(b"\x00\x00" * 160)
        await _wait_for_predicate(lambda: any("activity_start" in call for call in live.sent))
    finally:
        await stt.close()
    assert factory.context is not None
    assert factory.context.exited is True
    assert resources.sync_transport.is_closed
    assert resources.async_transport.is_closed


@pytest.mark.asyncio
async def test_repeated_session_replacement_bounds_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    prepare_calls = 0

    def counting_prepare(api_key, language_codes, custom_vocabulary):
        nonlocal prepare_calls
        prepare_calls += 1
        resources = _stub_resources(language_codes, custom_vocabulary)
        made.append(resources)
        return resources

    monkeypatch.setattr(gemini_module, "_prepare_gemini_resources_sync", counting_prepare)
    live = _FakeLiveSession()
    factory = _RecordingFactory(live)
    backend = GeminiTranscribeSTTBackend(
        api_key="dummy-offline-key",
        language_codes=("ko-KR",),
        live_connect_factory=factory,
    )
    for _ in range(REPLACEMENT_ROUNDS):
        stt = await backend.open_session()
        assert stt._teardown_done is False
        await stt.close()
        assert stt._teardown_done is True
    assert prepare_calls == REPLACEMENT_ROUNDS
    assert len(made) == REPLACEMENT_ROUNDS
    for resources in made:
        _assert_resources_released_once(resources)


@pytest.mark.asyncio
async def test_constructor_failure_reclaims_partial_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    def failing_create(api_key, http_options):
        seen["sync"] = http_options.httpx_client
        seen["async"] = http_options.httpx_async_client
        raise RuntimeError("boom-constructor")

    monkeypatch.setattr(gemini_module, "_create_genai_client_sync", failing_create)
    live = _FakeLiveSession()
    backend, factory = _make_backend(live)
    with pytest.raises(RuntimeError, match="boom-constructor"):
        await backend.open_session()
    assert factory.calls == []
    assert seen["sync"].is_closed
    assert seen["async"].is_closed


@pytest.mark.asyncio
async def test_transport_construction_failure_closes_acquired_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list = []
    real_sync_client = httpx.Client

    def spy_sync_client(*args, **kwargs):
        instance = real_sync_client(*args, **kwargs)
        created.append(instance)
        return instance

    def failing_async_client(*args, **kwargs):
        raise RuntimeError("boom-async-transport")

    monkeypatch.setattr(httpx, "Client", spy_sync_client)
    monkeypatch.setattr(httpx, "AsyncClient", failing_async_client)
    with pytest.raises(RuntimeError, match="boom-async-transport"):
        await asyncio.to_thread(_prepare_gemini_resources_sync, "dummy-offline-key", ["ko-KR"], [])
    assert len(created) == 1
    assert created[0].is_closed


@pytest.mark.asyncio
async def test_handshake_timeout_releases_context_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    live = _FakeLiveSession()
    factory = _HangingFactory(live)
    backend = GeminiTranscribeSTTBackend(
        api_key="dummy-offline-key",
        language_codes=("ko-KR",),
        connect_timeout_s=CONNECT_TIMEOUT_S,
        live_connect_factory=factory,
    )
    with pytest.raises(TimeoutError):
        await backend.open_session()
    assert factory.context is not None
    assert factory.context.exited is True
    assert factory.context.exit_calls == 1
    _assert_resources_released_once(made[0])


@pytest.mark.asyncio
async def test_handshake_refusal_releases_context_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    live = _FakeLiveSession()
    factory = _RefusingFactory(live)
    backend = GeminiTranscribeSTTBackend(
        api_key="dummy-offline-key",
        language_codes=("ko-KR",),
        live_connect_factory=factory,
    )
    with pytest.raises(ConnectionError, match="boom-connect"):
        await backend.open_session()
    assert factory.context is not None
    assert factory.context.exited is True
    assert factory.context.exit_calls == 1
    _assert_resources_released_once(made[0])


@pytest.mark.asyncio
async def test_cancelled_setup_reclaims_late_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    gate = threading.Event()
    entered = threading.Event()
    _patch_prepare(monkeypatch, made, gate=gate, entered=entered)
    live = _FakeLiveSession()
    factory = _RecordingFactory(live)
    stt = _direct_session(factory)
    start_task = asyncio.create_task(stt.start())
    assert await asyncio.to_thread(entered.wait, timeout=5)
    start_task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert factory.calls == []
    _assert_resources_released_once(made[0])
    await stt.close()
    assert stt._live_session is None
    assert stt._client_resources is None
    assert factory.context is None
    remaining = []
    while not stt._event_projection._legacy_events.empty():
        remaining.append(stt._event_projection._legacy_events.get_nowait())
    assert all(item is None for item in remaining)
    assert stt._teardown_done is True


@pytest.mark.asyncio
async def test_repeated_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    live = _FakeLiveSession()
    backend, factory = _make_backend(live)
    stt = await backend.open_session()
    await stt.close()
    await stt.close()
    await stt.close()
    _assert_resources_released_once(made[0])
    assert factory.context is not None
    assert factory.context.exit_calls == 1
    assert live.closed is True


@pytest.mark.asyncio
async def test_repeated_cancel_during_handshake_close_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    live = _FakeLiveSession()
    factory = _HangingFactory(live)
    stt = _direct_session(factory, connect_timeout_s=10.0)
    start_task = asyncio.create_task(stt.start())
    await _wait_for_predicate(lambda: factory.calls != [])
    await asyncio.sleep(0.05)
    for _ in range(3):
        await stt.close()
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    await stt.close()
    _assert_resources_released_once(made[0])
    assert factory.context is not None
    assert factory.context.exit_calls == 1
    assert stt._live_session is None
    assert stt._teardown_done is True


@pytest.mark.asyncio
async def test_gated_async_close_completes_despite_repeated_caller_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    live = _FakeLiveSession()
    backend, factory = _make_backend(live)
    stt = await backend.open_session()
    resources = stt._client_resources
    gate = threading.Event()
    entered: list[str] = []
    orig_aio_aclose = resources.client.aio.aclose
    orig_async_aclose = resources.async_transport.aclose
    orig_client_close = resources.client.close
    orig_sync_close = resources.sync_transport.close

    async def gated_aio_aclose() -> None:
        entered.append("client-aio")
        await asyncio.to_thread(gate.wait)
        await orig_aio_aclose()

    async def gated_async_aclose() -> None:
        entered.append("transport-async")
        await asyncio.to_thread(gate.wait)
        await orig_async_aclose()

    def gated_client_close() -> None:
        entered.append("client-sync")
        assert gate.wait(timeout=10)
        orig_client_close()

    def gated_sync_close() -> None:
        entered.append("transport-sync")
        assert gate.wait(timeout=10)
        orig_sync_close()

    resources.client.aio.aclose = gated_aio_aclose
    resources.async_transport.aclose = gated_async_aclose
    resources.client.close = gated_client_close
    resources.sync_transport.close = gated_sync_close
    close_task = asyncio.create_task(stt.close())
    await _wait_for_predicate(lambda: len(entered) >= 1)
    for _ in range(3):
        close_task.cancel()
        await asyncio.sleep(0)
    assert not close_task.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    _assert_resources_released_once(made[0])
    assert made[0].sync_transport.is_closed
    assert made[0].async_transport.is_closed
    assert stt._teardown_done is True


@pytest.mark.asyncio
async def test_close_waits_for_gated_setup_despite_repeated_requester_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    gate = threading.Event()
    entered = threading.Event()
    _patch_prepare(monkeypatch, made, gate=gate, entered=entered)
    live = _FakeLiveSession()
    factory = _RecordingFactory(live)
    stt = _direct_session(factory)
    start_task = asyncio.create_task(stt.start())
    assert await asyncio.to_thread(entered.wait, timeout=5)
    for _ in range(3):
        start_task.cancel()
        await asyncio.sleep(0)
    close_task = asyncio.create_task(stt.close())
    await asyncio.sleep(0.1)
    assert not close_task.done()
    assert factory.calls == []
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    await close_task
    assert factory.calls == []
    _assert_resources_released_once(made[0])
    assert stt._teardown_done is True


@pytest.mark.asyncio
async def test_close_never_exits_before_enter_finally_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    gate = threading.Event()
    entered_finally: list = []
    live = _FakeLiveSession()
    factory = _FinallyGatedFactory(live, gate, entered_finally)
    stt = _direct_session(factory, connect_timeout_s=10.0)
    start_task = asyncio.create_task(stt.start())
    await _wait_for_predicate(lambda: factory.calls != [])
    await asyncio.sleep(0.05)
    close_task = asyncio.create_task(stt.close())
    await _wait_for_predicate(lambda: bool(entered_finally))
    await asyncio.sleep(0.05)
    assert factory.context is not None
    assert factory.context.exit_calls == 0
    assert not close_task.done()
    gate.set()
    await close_task
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    await stt.close()
    assert factory.context.enter_finally_done is True
    assert factory.context.exit_calls == 1
    _assert_resources_released_once(made[0])


@pytest.mark.asyncio
async def test_factory_sync_raise_releases_client_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)

    def raising_factory(*, model, config):
        raise RuntimeError("boom-factory-sync")

    backend = GeminiTranscribeSTTBackend(
        api_key="dummy-offline-key",
        language_codes=("ko-KR",),
        live_connect_factory=raising_factory,
    )
    with pytest.raises(RuntimeError, match="boom-factory-sync"):
        await backend.open_session()
    assert len(made) == 1
    _assert_resources_released_once(made[0])


@pytest.mark.asyncio
async def test_setup_ready_cancel_race_releases_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for _ in range(5):
        made: list = []
        gate = threading.Event()
        entered = threading.Event()
        _patch_prepare(monkeypatch, made, gate=gate, entered=entered)
        live = _FakeLiveSession()
        factory = _RecordingFactory(live)
        stt = _direct_session(factory)
        start_task = asyncio.create_task(stt.start())
        assert await asyncio.to_thread(entered.wait, timeout=5)
        start_task.cancel()
        gate.set()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
        await stt.close()
        assert len(made) == 1
        _assert_resources_released_once(made[0])
        assert stt._teardown_done is True
