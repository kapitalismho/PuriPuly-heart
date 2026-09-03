"""Free-tier rolling ASR provider selection.

Priority-based failover across configured free cloud ASR providers:

    gemini_transcribe -> elevenlabs_scribe -> deepgram

The rolling backend is a normal STTBackend: it opens one provider session per
attempt, falls through to the next provider on transient failures for that
attempt only, and persists provider exclusion only for conditions that will
not disappear on the next immediate attempt (auth failure, explicit free
quota/credit exhaustion).

Free-only invariant: routing fails closed. A provider with explicit quota
exhaustion or auth failure is skipped instead of silently creating paid usage.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

_ERROR_KIND_AUTH = "auth"
_ERROR_KIND_QUOTA_DAY = "quota_day"
_ERROR_KIND_QUOTA = "quota"
_ERROR_KIND_TRANSIENT = "transient"

_PERSISTENT_EXHAUSTION_STATES = frozenset(
    {
        _ERROR_KIND_QUOTA_DAY,
        _ERROR_KIND_QUOTA,
    }
)

_ROLLING_PRIORITY_ORDER: tuple[STTProviderName, ...] = (
    STTProviderName.GEMINI_TRANSCRIBE,
    STTProviderName.ELEVENLABS_SCRIBE,
    STTProviderName.DEEPGRAM,
)


class RollingProviderState(str, Enum):
    NOT_CONFIGURED = "not_configured"
    AVAILABLE = "available"
    AUTH_FAILED = "auth_failed"
    FREE_QUOTA_EXHAUSTED = "free_quota_exhausted"


class RollingQuotaObservability(str, Enum):
    KNOWN = "known"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RollingProviderStatus:
    name: STTProviderName
    state: RollingProviderState
    observability: RollingQuotaObservability
    estimate_remaining: int | None = None
    estimate_total: int | None = None


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def _compact_error_text(exc: BaseException) -> str:
    return re.sub(r"[^a-z0-9]+", "", _error_text(exc))


def classify_gemini_error(exc: BaseException) -> str:
    compact = _compact_error_text(exc)
    if "401" in compact or "403" in compact or "apikeynotvalid" in compact:
        return _ERROR_KIND_AUTH
    if "unauthorized" in compact or "forbidden" in compact or "permissiondenied" in compact:
        return _ERROR_KIND_AUTH
    if "perday" in compact or "rpd" in compact or "dailyquota" in compact:
        return _ERROR_KIND_QUOTA_DAY
    return _ERROR_KIND_TRANSIENT


def classify_scribe_error(exc: BaseException) -> str:
    compact = _compact_error_text(exc)
    if "autherror" in compact or "401" in compact or "403" in compact:
        return _ERROR_KIND_AUTH
    if "quotaexceeded" in compact:
        return _ERROR_KIND_QUOTA
    return _ERROR_KIND_TRANSIENT


def classify_deepgram_error(exc: BaseException) -> str:
    compact = _compact_error_text(exc)
    text = _error_text(exc)
    if "401" in compact or "403" in compact or "unauthorized" in compact:
        return _ERROR_KIND_AUTH
    if "payment" in text or "credit" in text or "balance" in text or "insufficientfunds" in compact:
        return _ERROR_KIND_QUOTA
    return _ERROR_KIND_TRANSIENT


_ERROR_CLASSIFIER_BY_PROVIDER = {
    STTProviderName.GEMINI_TRANSCRIBE: classify_gemini_error,
    STTProviderName.ELEVENLABS_SCRIBE: classify_scribe_error,
    STTProviderName.DEEPGRAM: classify_deepgram_error,
}


@dataclass(frozen=True, slots=True)
class RollingProviderDefinition:
    name: STTProviderName
    build_backend: Callable[[], STTBackend]
    is_configured: Callable[[], bool]
    session_deadline_s: float | None = None
    classify_error: Callable[[BaseException], str] | None = None

    def classifier(self) -> Callable[[BaseException], str]:
        if self.classify_error is not None:
            return self.classify_error
        return _ERROR_CLASSIFIER_BY_PROVIDER.get(self.name, classify_gemini_error)


@dataclass(slots=True)
class RollingSTTBackend(STTBackend):
    """STTBackend composing free-tier cloud ASR providers in fixed priority.

    Auth failures and account-level quota exclusions persist until this
    backend is rebuilt (application restart). Rotate credentials or refill
    balance, then restart to re-enable the provider.
    """

    providers: tuple[RollingProviderDefinition, ...]
    clock: Clock = field(default_factory=SystemClock)

    _states: dict[STTProviderName, RollingProviderState | None] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("rolling backend requires at least one provider definition")
        seen: set[STTProviderName] = set()
        for definition in self.providers:
            if definition.name not in _ROLLING_PRIORITY_ORDER:
                raise ValueError(f"unsupported rolling provider: {definition.name}")
            if definition.name in seen:
                raise ValueError(f"duplicate rolling provider: {definition.name}")
            seen.add(definition.name)
        object.__setattr__(
            self,
            "providers",
            tuple(
                sorted(
                    self.providers,
                    key=lambda definition: _ROLLING_PRIORITY_ORDER.index(definition.name),
                )
            ),
        )
        self._states = {definition.name: None for definition in self.providers}

    def status(self, name: STTProviderName) -> RollingProviderStatus:
        definition = self._definition(name)
        state = self._effective_state(definition)
        return RollingProviderStatus(
            name=name,
            state=state,
            observability=RollingQuotaObservability.UNKNOWN,
        )

    def statuses(self) -> tuple[RollingProviderStatus, ...]:
        return tuple(self.status(definition.name) for definition in self.providers)

    def _definition(self, name: STTProviderName) -> RollingProviderDefinition:
        for definition in self.providers:
            if definition.name == name:
                return definition
        raise KeyError(name)

    def _mark(self, name: STTProviderName, state: RollingProviderState) -> None:
        if self._states.get(name) == state:
            return
        previous = self._states.get(name)
        previous_label = (
            previous.value if previous is not None else RollingProviderState.AVAILABLE.value
        )
        logger.info(
            "[STT][Rolling] provider=%s state=%s -> %s",
            name.value,
            previous_label,
            state.value,
        )
        self._states[name] = state

    def _provider_state(self, definition: RollingProviderDefinition) -> RollingProviderState:
        return self._effective_state(definition)

    def _effective_state(self, definition: RollingProviderDefinition) -> RollingProviderState:
        name = definition.name
        raw = self._states.get(name)
        if not definition.is_configured():
            if raw is not None:
                logger.warning(
                    "[STT][Rolling] provider=%s configuration lost; runtime state hidden",
                    name.value,
                )
            return RollingProviderState.NOT_CONFIGURED
        if raw is None:
            self._states[name] = RollingProviderState.AVAILABLE
            return RollingProviderState.AVAILABLE
        return raw

    def _is_eligible(self, definition: RollingProviderDefinition) -> bool:
        return self._provider_state(definition) is RollingProviderState.AVAILABLE

    async def open_session(self) -> STTBackendSession:
        attempt_start = self.clock.now()
        last_error: BaseException | None = None
        for definition in self.providers:
            if not self._is_eligible(definition):
                continue
            try:
                backend = definition.build_backend()
                session = await backend.open_session()
            except Exception as exc:
                kind = definition.classifier()(exc)
                self._handle_open_error(definition, exc, kind)
                last_error = exc
                continue
            logger.info(
                "[STT][Rolling] session selected provider=%s elapsed_s=%.3f",
                definition.name.value,
                self.clock.now() - attempt_start,
            )
            return _RollingSession(
                definition=definition,
                inner=session,
                on_session_error=self._handle_session_error,
            )
        if last_error is not None:
            raise last_error
        if any(definition.is_configured() for definition in self.providers):
            logger.warning(
                "[STT][Rolling] all configured providers excluded; statuses=%s",
                [(status.name.value, status.state.value) for status in self.statuses()],
            )
            raise RuntimeError(
                "All rolling ASR providers are excluded (quota/auth); "
                "see statuses() for per-provider state"
            )
        raise RuntimeError(
            "No rolling ASR provider is configured; configure a Gemini, ElevenLabs, "
            "or Deepgram API key"
        )

    def _handle_open_error(
        self,
        definition: RollingProviderDefinition,
        exc: Exception,
        kind: str,
    ) -> None:
        if kind == _ERROR_KIND_AUTH:
            self._mark(definition.name, RollingProviderState.AUTH_FAILED)
            logger.warning(
                "[STT][Rolling] provider=%s open failed kind=%s -> excluded until "
                "credential change",
                definition.name.value,
                kind,
            )
            return
        if kind in _PERSISTENT_EXHAUSTION_STATES:
            self._mark(definition.name, RollingProviderState.FREE_QUOTA_EXHAUSTED)
            logger.warning(
                "[STT][Rolling] provider=%s open failed kind=%s -> excluded until quota reset",
                definition.name.value,
                kind,
            )
            return
        logger.info(
            "[STT][Rolling] provider=%s open failed kind=transient (%s); falling through "
            "for this attempt",
            definition.name.value,
            type(exc).__name__,
        )

    def _handle_session_error(
        self,
        definition: RollingProviderDefinition,
        exc: BaseException,
    ) -> None:
        kind = definition.classifier()(exc)
        if kind == _ERROR_KIND_AUTH:
            self._mark(definition.name, RollingProviderState.AUTH_FAILED)
            return
        if kind in _PERSISTENT_EXHAUSTION_STATES:
            self._mark(definition.name, RollingProviderState.FREE_QUOTA_EXHAUSTED)


@dataclass(slots=True)
class _RollingSession(STTBackendSession):
    """Session wrapper exposing provider identity, deadline, and error mapping."""

    definition: RollingProviderDefinition
    inner: STTBackendSession
    on_session_error: Callable[[RollingProviderDefinition, BaseException], None]

    @property
    def reset_deadline_s(self) -> float:
        if self.definition.session_deadline_s is not None:
            return self.definition.session_deadline_s
        inner_deadline = getattr(self.inner, "reset_deadline_s", None)
        if isinstance(inner_deadline, (int, float)) and inner_deadline > 0:
            return float(inner_deadline)
        raise AttributeError("reset_deadline_s")

    @property
    def provider_name(self) -> STTProviderName:
        return self.definition.name

    async def send_audio(self, pcm16le: bytes) -> None:
        await self.inner.send_audio(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        await self.inner.on_speech_end(
            trailing_silence_ms=trailing_silence_ms,
            reason=reason,
        )

    async def stop(self) -> None:
        await self.inner.stop()

    async def close(self) -> None:
        await self.inner.close()

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        try:
            async for event in self.inner.events():
                yield event
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self.on_session_error(self.definition, exc)
            raise


import asyncio  # placed at bottom to keep the main logic compact
