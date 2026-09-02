from __future__ import annotations

import asyncio
import time

import pytest

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.rolling import (
    GEMINI_ROLLING_SESSION_TARGET_S,
    GeminiFreeTierEstimator,
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
    estimator: GeminiFreeTierEstimator | None = None,
    session_deadline_s: float | None = None,
) -> tuple[RollingProviderDefinition, _ScriptedBackend]:
    backend = _ScriptedBackend(session, fail_times=fail_times)
    if name is STTProviderName.GEMINI_TRANSCRIBE and estimator is None:
        estimator = GeminiFreeTierEstimator(rpd_baseline=25, wall_time=lambda: time.time())
    return (
        RollingProviderDefinition(
            name=name,
            build_backend=lambda: backend,
            is_configured=lambda: configured,
            classify_error=classifier,
            estimator=estimator,
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
    assert gemini_backend.open_count == 1
    assert scribe_backend.open_count == 0
    assert session.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await session.close()


@pytest.mark.asyncio
async def test_transient_open_failure_falls_through_for_attempt_only() -> None:
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        fail_times=1,
        classifier=lambda exc: "transient",
    )
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(
        STTProviderName.ELEVENLABS_SCRIBE, scribe_session, classifier=lambda exc: "transient"
    )
    rolling = _make(gemini, scribe)

    session = await rolling.open_session()
    assert gemini_backend.open_count == 1
    assert scribe_backend.open_count == 1
    assert session.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await session.close()

    second = await rolling.open_session()
    assert gemini_backend.open_count == 2
    assert second.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await second.close()


@pytest.mark.asyncio
async def test_gemini_rpm_429_is_transient_not_daily_exhaustion() -> None:
    assert classify_gemini_error(RuntimeError("429 RESOURCE_EXHAUSTED RPM limit")) == "transient"
    assert classify_gemini_error(RuntimeError("quota exceeded per day (RPD)")) == "quota_day"
    assert classify_gemini_error(RuntimeError("401 API key not valid")) == "auth"


@pytest.mark.asyncio
async def test_auth_failure_persists_until_credential_change() -> None:
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        fail_times=99,
        classifier=lambda exc: "auth",
    )
    scribe_session = _ScriptedSession()
    scribe, _ = _definition(STTProviderName.ELEVENLABS_SCRIBE, scribe_session)
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    assert first.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await first.close()

    second = await rolling.open_session()
    assert gemini_backend.open_count == 1
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await second.close()
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )


@pytest.mark.asyncio
async def test_daily_quota_exhaustion_persists_but_rpm_does_not() -> None:
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        fail_times=99,
        classifier=lambda exc: "quota_day",
    )
    scribe_session = _ScriptedSession()
    scribe, _ = _definition(STTProviderName.ELEVENLABS_SCRIBE, scribe_session)
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    await first.close()
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )
    assert gemini_backend.open_count == 1

    second = await rolling.open_session()
    assert gemini_backend.open_count == 1
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
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
async def test_gemini_local_estimate_gates_selection_fail_closed() -> None:
    estimator = GeminiFreeTierEstimator(rpd_baseline=1, wall_time=lambda: time.time())
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        estimator=estimator,
    )
    scribe_session = _ScriptedSession()
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, scribe_session)
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    assert first.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    await first.close()
    assert estimator.remaining_sessions() == 0

    status = rolling.status(STTProviderName.GEMINI_TRANSCRIBE)
    assert status.observability is RollingQuotaObservability.ESTIMATED
    assert status.estimate_remaining == 0

    second = await rolling.open_session()
    assert gemini_backend.open_count == 1
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    await second.close()


@pytest.mark.asyncio
async def test_gemini_599_rollover_selects_gemini_again() -> None:
    estimator = GeminiFreeTierEstimator(rpd_baseline=25, wall_time=lambda: time.time())
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        estimator=estimator,
        session_deadline_s=GEMINI_ROLLING_SESSION_TARGET_S,
    )
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = await rolling.open_session()
    assert first.reset_deadline_s == pytest.approx(599.0)
    await first.close()

    second = await rolling.open_session()
    assert second.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    assert scribe_backend.open_count == 0
    assert gemini_backend.open_count == 2
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

    estimator = GeminiFreeTierEstimator(rpd_baseline=25, wall_time=lambda: time.time())
    session = _DeadlineSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        session,
        estimator=estimator,
        session_deadline_s=0.05,
    )
    scribe, scribe_backend = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    active = await rolling.open_session()
    await asyncio.sleep(0.1)
    deadline_hit.set()
    again = await rolling.open_session()
    assert again.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    assert scribe_backend.open_count == 0
    assert gemini_backend.open_count == 2
    await active.close()
    await again.close()


def test_scribe_error_classification() -> None:
    assert classify_scribe_error(RuntimeError("auth_error: invalid key")) == "auth"
    assert classify_scribe_error(RuntimeError("quota_exceeded: credits exhausted")) == "quota"
    assert classify_scribe_error(RuntimeError("rate_limited")) == "transient"


def test_deepgram_error_classification() -> None:
    assert classify_deepgram_error(RuntimeError("HTTP 401 Unauthorized")) == "auth"
    assert classify_deepgram_error(RuntimeError("payment required: no balance")) == "quota"
    assert classify_deepgram_error(RuntimeError("connection reset")) == "transient"


def test_gemini_estimator_resets_on_quota_day_boundary() -> None:
    current = {"t": 1756000000.0}
    estimator = GeminiFreeTierEstimator(rpd_baseline=2, wall_time=lambda: current["t"])
    estimator.on_session_established()
    estimator.on_session_established()
    assert estimator.exhausted() is True
    current["t"] += 24 * 3600
    assert estimator.exhausted() is False
    assert estimator.remaining_sessions() == 2


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
    assert session.provider_name is STTProviderName.GEMINI_TRANSCRIBE
    assert gemini_backend.open_count == 1
    assert scribe_backend.open_count == 0
    assert deepgram_backend.open_count == 0
    asyncio.run(session.close())
    assert [status.name for status in rolling.statuses()] == [
        STTProviderName.GEMINI_TRANSCRIBE,
        STTProviderName.ELEVENLABS_SCRIBE,
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


def test_gemini_without_estimator_rejected() -> None:
    backend = _ScriptedBackend(_ScriptedSession())
    definition = RollingProviderDefinition(
        name=STTProviderName.GEMINI_TRANSCRIBE,
        build_backend=lambda: backend,
        is_configured=lambda: True,
    )
    with pytest.raises(ValueError, match="requires a GeminiFreeTierEstimator"):
        _make(definition)


def test_status_reports_configured_provider_available_before_first_attempt() -> None:
    gemini, _ = _definition(STTProviderName.GEMINI_TRANSCRIBE, _ScriptedSession())
    rolling = _make(gemini)
    status = rolling.status(STTProviderName.GEMINI_TRANSCRIBE)
    assert status.state is RollingProviderState.AVAILABLE
    assert status.observability is RollingQuotaObservability.ESTIMATED


def test_scribe_and_deepgram_quota_kind_is_account_quota_not_daily() -> None:
    from puripuly_heart.core.stt.rolling import classify_deepgram_error, classify_scribe_error

    assert classify_scribe_error(RuntimeError("quota_exceeded: monthly credits gone")) == "quota"
    assert classify_deepgram_error(RuntimeError("payment required: balance empty")) == "quota"


def test_gemini_estimate_reset_clears_persisted_quota_state() -> None:
    current = {"t": 1756000000.0}
    estimator = GeminiFreeTierEstimator(rpd_baseline=1, wall_time=lambda: current["t"])
    gemini_session = _ScriptedSession()
    gemini, gemini_backend = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        fail_times=99,
        classifier=lambda exc: "quota_day",
        estimator=estimator,
    )
    scribe, _ = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = asyncio.run(rolling.open_session())
    asyncio.run(first.close())
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )
    assert gemini_backend.open_count == 1

    current["t"] += 24 * 3600
    second = asyncio.run(rolling.open_session())
    assert gemini_backend.open_count == 2
    assert second.provider_name is STTProviderName.ELEVENLABS_SCRIBE
    asyncio.run(second.close())
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.FREE_QUOTA_EXHAUSTED
    )


def test_configuration_loss_hides_but_persists_exclusion_state() -> None:
    gemini_session = _ScriptedSession()
    gemini, _ = _definition(
        STTProviderName.GEMINI_TRANSCRIBE,
        gemini_session,
        fail_times=99,
        classifier=lambda exc: "auth",
    )
    scribe, _ = _definition(STTProviderName.ELEVENLABS_SCRIBE, _ScriptedSession())
    rolling = _make(gemini, scribe)

    first = asyncio.run(rolling.open_session())
    asyncio.run(first.close())
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )

    definition = rolling._definition(STTProviderName.GEMINI_TRANSCRIBE)
    object.__setattr__(definition, "is_configured", lambda: False)
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.NOT_CONFIGURED
    )
    object.__setattr__(definition, "is_configured", lambda: True)
    assert rolling.status(STTProviderName.GEMINI_TRANSCRIBE).state is (
        RollingProviderState.AUTH_FAILED
    )
