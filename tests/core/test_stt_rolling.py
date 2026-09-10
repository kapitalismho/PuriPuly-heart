from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.rolling import (
    RollingProviderDefinition,
    RollingProviderState,
    RollingQuotaObservability,
    RollingSTTBackend,
    classify_deepgram_error,
    classify_gemini_error,
    classify_scribe_error,
)


class _ScriptedSession:
    def __init__(self, *, error: Exception | None = None, texts: tuple[str, ...] = ("ok",)):
        self._error = error
        self._texts = texts
        self.closed = False
        self.stopped = False

    async def send_audio(self, pcm16le: bytes) -> None:
        _ = pcm16le

    async def on_speech_end(self, *, trailing_silence_ms=None, reason=None) -> None:
        _ = self
        _ = trailing_silence_ms, reason

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        self.closed = True

    async def events(self):
        if self._error is not None:
            raise self._error
        for text in self._texts:
            yield STTBackendTranscriptEvent(text=text, is_final=True)


class _ScriptedBackend:
    def __init__(self, session: _ScriptedSession, *, fail_times: int = 0) -> None:
        self._session = session
        self._fail_times = fail_times
        self.open_count = 0

    async def open_session(self):
        self.open_count += 1
        if self.open_count <= self._fail_times:
            raise self._session._error or RuntimeError("scripted open failure")
        return self._session


def _definition(
    name: STTProviderName,
    session: _ScriptedSession,
    *,
    fail_times: int = 0,
    configured: bool = True,
    classifier=None,
    session_deadline_s: float | None = None,
) -> tuple[RollingProviderDefinition, _ScriptedBackend]:
    backend = _ScriptedBackend(session, fail_times=fail_times)
    return (
        RollingProviderDefinition(
            name=name,
            build_backend=lambda: backend,
            is_configured=lambda: configured,
            classify_error=classifier,
            session_deadline_s=session_deadline_s,
        ),
        backend,
    )


def _make(
    *definitions: RollingProviderDefinition,
    clock: FakeClock | None = None,
) -> RollingSTTBackend:
    return RollingSTTBackend(providers=tuple(definitions), clock=clock or FakeClock())


@pytest.mark.asyncio
async def test_open_session_selects_first_configured_provider() -> None:
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(STTProviderName.GEMINI_TRANSCRIBE, gemini_session)
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, scribe_session)
    rolling = _make(gemini, scribe)

    session = await rolling.open_session()
    assert scribe_backend.open_count == 1
    assert gemini_backend.open_count == 0
    assert session.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await session.close()


@pytest.mark.asyncio
async def test_transient_open_failure_falls_through_for_attempt_only() -> None:
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        scribe_session,
        fail_times=1,
        classifier=lambda exc: "transient",
    )
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE, gemini_session, classifier=lambda exc: "transient"
    )
    rolling = _make(gemini, scribe)

    session = await rolling.open_session()
    assert scribe_backend.open_count == 1
    assert gemini_backend.open_count == 1
    assert session.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await session.close()

    second = await rolling.open_session()
    assert scribe_backend.open_count == 2
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await second.close()


@pytest.mark.asyncio
async def test_gemini_rpm_429_is_transient_not_daily_exhaustion() -> None:
    assert classify_gemini_error(RuntimeError("429 RESOURCE_EXHAUSTED RPM limit")) == "transient"
    assert classify_gemini_error(RuntimeError("quota exceeded per day (RPD)")) == "quota_day"
    assert classify_gemini_error(RuntimeError("401 API key not valid")) == "auth"


@pytest.mark.asyncio
async def test_auth_failure_persists_until_credential_change() -> None:
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        scribe_session,
        fail_times=99,
        classifier=lambda exc: "auth",
    )
    gemini_session = _ScriptedSession()
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, gemini_session)
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    assert first.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await first.close()

    second = await rolling.open_session()
    assert scribe_backend.open_count == 1
    assert second.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await second.close()
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )


@pytest.mark.asyncio
async def test_rebind_member_clears_only_that_member_exclusion() -> None:
    scribe_failed = _ScriptedBackend(
        _ScriptedSession(error=RuntimeError("401")),
        fail_times=99,
    )
    scribe_ready = _ScriptedBackend(_ScriptedSession())
    scribe_backend: _ScriptedBackend | None = scribe_failed
    scribe_configured = True

    def scribe_is_configured() -> bool:
        return scribe_configured

    def scribe_build_backend() -> _ScriptedBackend:
        assert scribe_backend is not None
        return scribe_backend

    def scribe_rebind(api_key: str) -> None:
        nonlocal scribe_backend, scribe_configured
        scribe_configured = bool((api_key or "").strip())
        scribe_backend = scribe_ready if scribe_configured else None

    gemini_session = _ScriptedSession(error=RuntimeError("quota_exceeded"))
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        fail_times=99,
        classifier=lambda exc: "quota",
    )
    scribe = RollingProviderDefinition(
        name=STTProviderName.ELEVENLABS_SCRIBE,
        build_backend=scribe_build_backend,
        is_configured=scribe_is_configured,
        classify_error=lambda exc: "auth",
        rebind=scribe_rebind,
    )
    rolling = _make(gemini, scribe)

    with pytest.raises(RuntimeError, match="quota_exceeded"):
        await rolling.open_session()
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )
    assert gemini_backend.open_count == 1

    assert rolling.rebind_member(STTProviderName.ELEVENLABS_SCRIBE, "rotated-key")
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AVAILABLE
    )
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )

    session = await rolling.open_session()
    assert session.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    assert scribe_ready.open_count == 1
    assert gemini_backend.open_count == 1
    await session.close()


@pytest.mark.asyncio
async def test_daily_quota_exhaustion_persists_but_rpm_does_not() -> None:
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        scribe_session,
        fail_times=99,
        classifier=lambda exc: "quota_day",
    )
    gemini_session = _ScriptedSession()
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, gemini_session)
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    await first.close()
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )
    assert scribe_backend.open_count == 1

    second = await rolling.open_session()
    assert scribe_backend.open_count == 1
    assert second.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await second.close()


@pytest.mark.asyncio
async def test_unconfigured_provider_is_skipped() -> None:
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession(), configured=False
    )
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    session = await rolling.open_session()
    assert gemini_backend.open_count == 0
    assert scribe_backend.open_count == 1
    assert session.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await session.close()


@pytest.mark.asyncio
async def test_no_provider_configured_fails_closed() -> None:
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession(), configured=False)
    rolling = _make(gemini)
    with pytest.raises(RuntimeError, match="No rolling ASR provider is configured"):
        await rolling.open_session()


@pytest.mark.asyncio
async def test_scribe_has_no_local_session_cap() -> None:
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(STTProviderName.GEMINI_TRANSCRIBE, gemini_session)
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, scribe_session)
    rolling = _make(gemini, scribe)

    for _ in range(30):
        session = await rolling.open_session()
        assert session.provider_name is STTProviderName.ELEVENLABS_SCRIBE
        await session.close()

    assert scribe_backend.open_count == 30
    assert gemini_backend.open_count == 0
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AVAILABLE
    )


@pytest.mark.asyncio
async def test_scribe_without_override_uses_controller_deadline() -> None:
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(STTProviderName.GEMINI_TRANSCRIBE, gemini_session)
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    with pytest.raises(AttributeError):
        _ = first.reset_deadline_s
    await first.close()

    second = await rolling.open_session()
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    assert gemini_backend.open_count == 0
    assert scribe_backend.open_count == 2
    await second.close()


@pytest.mark.asyncio
async def test_session_error_event_is_mapped_and_re_raised() -> None:
    gemini_session = _ScriptedSession(error=RuntimeError("quota exceeded per day"))
    gemini, _ = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        classifier=classify_gemini_error,
    )
    rolling = _make(gemini)
    session = await rolling.open_session()
    with pytest.raises(RuntimeError):
        async for _ in session.events():
            pass
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )


@pytest.mark.asyncio
async def test_healthy_rollover_does_not_descend_on_deadline() -> None:
    deadline_hit = asyncio.Event()

    class _DeadlineSession(_ScriptedSession):
        async def events(self):
            await deadline_hit.wait()
            yield STTBackendTranscriptEvent(text="late", is_final=True)

    session = _DeadlineSession()
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        session,
        session_deadline_s=0.05,
    )
    gemini, gemini_backend = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    active = await rolling.open_session()
    await asyncio.sleep(0.1)
    deadline_hit.set()
    again = await rolling.open_session()
    assert again.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    assert gemini_backend.open_count == 0
    assert scribe_backend.open_count == 2
    await active.close()
    await again.close()


def test_scribe_error_classification() -> None:
    assert classify_scribe_error(RuntimeError("auth_error: invalid key")) == "auth"
    assert classify_scribe_error(RuntimeError("quota_exceeded: credits exhausted")) == "quota"
    assert classify_scribe_error(RuntimeError("rate_limited")) == "transient"
    assert classify_scribe_error(RuntimeError("429 Too Many Requests")) == "transient"


def test_deepgram_error_classification() -> None:
    assert classify_deepgram_error(RuntimeError("HTTP 401 Unauthorized")) == "auth"
    assert classify_deepgram_error(RuntimeError("payment required: no balance")) == "quota"
    assert classify_deepgram_error(RuntimeError("connection reset")) == "transient"
    assert classify_deepgram_error(RuntimeError("429 Too Many Requests")) == "transient"


def test_duplicate_providers_rejected() -> None:
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    gemini2, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    with pytest.raises(ValueError, match="duplicate rolling provider"):
        _make(gemini, gemini2)


def test_empty_providers_rejected() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        _make()


def test_priority_order_is_enforced_regardless_of_definition_order() -> None:
    deepgram, deepgram_backend = _definition(STTProviderName.DEEPGRAM, _ScriptedSession())
    gemini, gemini_backend = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(deepgram, gemini, scribe)

    session = asyncio.run(rolling.open_session())
    assert session.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    assert scribe_backend.open_count == 1
    assert gemini_backend.open_count == 0
    assert deepgram_backend.open_count == 0
    asyncio.run(session.close())
    assert [status.name for status in rolling.statuses()] == [
        STTProviderName.ELEVENLABS_SCRIBE,
        STTProviderName.GEMINI_TRANSCRIBE,
        STTProviderName.DEEPGRAM,
    ]


def test_unsupported_provider_rejected() -> None:
    backend = _ScriptedBackend(_ScriptedSession())
    definition = RollingProviderDefinition(
        name=STTProviderName.SONIOX,
        build_backend=lambda: backend,
        is_configured=lambda: True,
    )
    with pytest.raises(ValueError, match="unsupported rolling provider"):
        _make(definition)


def test_gemini_without_estimator_is_allowed() -> None:
    backend = _ScriptedBackend(_ScriptedSession())
    definition = RollingProviderDefinition(
        name=STTProviderName.GEMINI_TRANSCRIBE,
        build_backend=lambda: backend,
        is_configured=lambda: True,
    )
    rolling = _make(definition)
    status = rolling.status(STTProviderName.GEMINI_TRANSCRIBE)
    assert status.state is RollingProviderState.AVAILABLE
    assert status.observability is RollingQuotaObservability.UNKNOWN


def test_status_reports_configured_provider_available_before_first_attempt() -> None:
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    rolling = _make(gemini)
    status = rolling.status(STTProviderName.GEMINI_TRANSCRIBE)
    assert status.state is RollingProviderState.AVAILABLE
    assert status.observability is RollingQuotaObservability.UNKNOWN


def test_scribe_and_deepgram_quota_kind_is_account_quota_not_daily() -> None:
    from puripuly_heart.core.stt.rolling import classify_deepgram_error, classify_scribe_error

    assert classify_scribe_error(RuntimeError("quota_exceeded: monthly credits gone")) == "quota"
    assert classify_deepgram_error(RuntimeError("payment required: balance empty")) == "quota"


def test_explicit_daily_quota_signal_persists_without_day_reset() -> None:
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        scribe_session,
        fail_times=99,
        classifier=lambda exc: "quota_day",
    )
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = asyncio.run(rolling.open_session())
    asyncio.run(first.close())
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )
    assert scribe_backend.open_count == 1

    second = asyncio.run(rolling.open_session())
    assert scribe_backend.open_count == 1
    assert second.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    asyncio.run(second.close())
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )


def test_configuration_loss_hides_but_persists_exclusion_state() -> None:
    scribe_session = _ScriptedSession()
    scribe, _ = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        scribe_session,
        fail_times=99,
        classifier=lambda exc: "auth",
    )
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = asyncio.run(rolling.open_session())
    asyncio.run(first.close())
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )

    definition = rolling._definition(STTProviderName.ELEVENLABS_SCRIBE)
    object.__setattr__(definition, "is_configured", lambda: False)
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.NOT_CONFIGURED
    )
    object.__setattr__(definition, "is_configured", lambda: True)
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )


@pytest.mark.asyncio
async def test_scribe_quota_falls_through_to_deepgram() -> None:
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession(), configured=False
    )
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        _ScriptedSession(),
        fail_times=99,
        classifier=lambda exc: "quota",
    )
    deepgram, deepgram_backend = _definition(STTProviderName.DEEPGRAM, _ScriptedSession())
    rolling = _make(gemini, scribe, deepgram)

    session = await rolling.open_session()
    assert gemini_backend.open_count == 0
    assert scribe_backend.open_count == 1
    assert deepgram_backend.open_count == 1
    assert session.provider_name is STTProviderName.DEEPGRAM
    await session.close()
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )


@pytest.mark.asyncio
async def test_three_provider_chain_scribe_transient_then_gemini_quota_to_deepgram() -> None:
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        _ScriptedSession(),
        fail_times=1,
        classifier=lambda exc: "transient",
    )
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        _ScriptedSession(),
        fail_times=99,
        classifier=lambda exc: "quota",
    )
    deepgram, deepgram_backend = _definition(STTProviderName.DEEPGRAM, _ScriptedSession())
    rolling = _make(gemini, scribe, deepgram)

    first = await rolling.open_session()
    assert first.provider_name is STTProviderName.DEEPGRAM
    assert scribe_backend.open_count == 1
    assert gemini_backend.open_count == 1
    assert deepgram_backend.open_count == 1
    await first.close()

    second = await rolling.open_session()
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    assert scribe_backend.open_count == 2
    await second.close()


@pytest.mark.asyncio
async def test_all_configured_but_excluded_reports_distinct_error() -> None:
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        _ScriptedSession(),
        fail_times=99,
        classifier=lambda exc: "quota_day",
    )
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE,
        _ScriptedSession(),
        fail_times=99,
        classifier=lambda exc: "quota",
    )
    deepgram, deepgram_backend = _definition(
        STTProviderName.DEEPGRAM,
        _ScriptedSession(),
        fail_times=99,
        classifier=lambda exc: "quota",
    )
    rolling = _make(gemini, scribe, deepgram)

    with pytest.raises(RuntimeError):
        await rolling.open_session()
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )
    opens_after_first = (
        gemini_backend.open_count + scribe_backend.open_count + deepgram_backend.open_count
    )

    with pytest.raises(RuntimeError, match="All rolling ASR providers are excluded"):
        await rolling.open_session()
    assert (
        gemini_backend.open_count + scribe_backend.open_count + deepgram_backend.open_count
    ) == opens_after_first


@pytest.mark.asyncio
async def test_healthy_rollover_keeps_provider_available() -> None:
    gemini, _ = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        _ScriptedSession(),
    )
    scribe, _ = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    await first.close()
    second = await rolling.open_session()
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await second.close()
    assert rolling.status(STTProviderName.ELEVENLABS_SCRIBE).state is (
        RollingProviderState.AVAILABLE
    )


class _ScopedScriptedSession(_ScriptedSession):
    def __init__(self) -> None:
        super().__init__()
        self.scoped_calls: list[tuple[object, ...]] = []

    async def begin_turn(self, request) -> None:
        self.scoped_calls.append(("begin", request))

    async def send_turn_audio(
        self,
        identity,
        pcm16le,
        *,
        payload_sequence,
        source_ranges,
        context_only,
    ) -> None:
        self.scoped_calls.append(
            ("audio", identity, pcm16le, payload_sequence, source_ranges, context_only)
        )

    async def seal_turn(
        self,
        identity,
        *,
        sealed_content_ranges,
        seal_reason,
        observed_trailing_silence_ms,
    ) -> None:
        self.scoped_calls.append(
            (
                "seal",
                identity,
                sealed_content_ranges,
                seal_reason,
                observed_trailing_silence_ms,
            )
        )

    async def abort_turn(self, identity, *, reason) -> None:
        self.scoped_calls.append(("abort", identity, reason))

    async def turn_events(self):
        yield "scoped terminal"


@pytest.mark.asyncio
async def test_rolling_session_preserves_scoped_member_protocol() -> None:
    inner = _ScopedScriptedSession()
    definition, _backend = _definition(STTProviderName.DEEPGRAM, inner)
    session = await _make(definition).open_session()
    identity = object()
    request = object()

    await session.begin_turn(request)
    await session.send_turn_audio(
        identity,
        b"audio",
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await session.seal_turn(
        identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    events = [event async for event in session.turn_events()]

    assert [call[0] for call in inner.scoped_calls] == ["begin", "audio", "seal"]
    assert events == ["scoped terminal"]
