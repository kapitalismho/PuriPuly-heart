"""Free-tier rolling ASR provider selection.

Priority-based failover across configured free cloud ASR providers:

    elevenlabs_scribe -> gemini_transcribe -> deepgram

The rolling backend is a normal STTBackend: it opens one provider session per
attempt, falls through to the next provider on transient failures for that
attempt only, and persists provider exclusion only for conditions that will
not disappear on the next immediate attempt (auth failure, explicit free
quota/credit exhaustion). Credential rotation rebinds that member in place
without rebuilding the rolling backend or opening sockets.

Free-only invariant: routing fails closed. A provider with explicit quota
exhaustion or auth failure is skipped instead of silently creating paid usage.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import AsyncIterator

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.diagnostics import emit_failure
from puripuly_heart.core.error_messages import stt_failure_report
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime_logging import emit_basic_log
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTIndependentRecognitionSession,
    STTProviderInputTerminal,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
    STTScopedTurnSession,
    STTSessionProjection,
)
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics
from puripuly_heart.domain.recognition import (
    RecognitionStreamIdentity,
    RecognitionUnitIdentity,
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
    STTProviderName.ELEVENLABS_SCRIBE,
    STTProviderName.GEMINI_TRANSCRIBE,
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
    last_failure_diagnostics: ErrorDiagnostics | None = None


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


@dataclass(slots=True)
class RollingMemberBinding:
    name: STTProviderName
    build: Callable[[str], STTBackend]
    backend: STTBackend | None = None
    configured: bool = False

    def apply_key(self, api_key: str) -> None:
        key = (api_key or "").strip()
        self.configured = bool(key)
        self.backend = self.build(key) if self.configured else None

    def is_configured(self) -> bool:
        return self.configured

    def build_backend(self) -> STTBackend:
        if self.backend is None:
            raise RuntimeError(f"rolling provider {self.name.value} was not prepared")
        return self.backend


@dataclass(frozen=True, slots=True)
class RollingProviderDefinition:
    name: STTProviderName
    build_backend: Callable[[], STTBackend]
    is_configured: Callable[[], bool]
    session_deadline_s: float | None = None
    classify_error: Callable[[BaseException], str] | None = None
    rebind: Callable[[str], None] | None = None

    def classifier(self) -> Callable[[BaseException], str]:
        if self.classify_error is not None:
            return self.classify_error
        return _ERROR_CLASSIFIER_BY_PROVIDER.get(self.name, classify_gemini_error)


@dataclass(slots=True)
class RollingSTTBackend(STTBackend):
    """STTBackend composing free-tier cloud ASR providers in fixed priority.

    Auth failures and account-level quota exclusions persist until this
    backend is rebuilt or that member is rebound after a credential change.
    """

    providers: tuple[RollingProviderDefinition, ...]
    clock: Clock = field(default_factory=SystemClock)
    channel: str | None = None

    _states: dict[STTProviderName, RollingProviderState | None] = field(
        init=False, default_factory=dict, repr=False
    )
    _last_failures: dict[STTProviderName, ErrorDiagnostics] = field(
        init=False, default_factory=dict, repr=False
    )
    _fallback_cause: ErrorDiagnostics | None = field(init=False, default=None, repr=False)

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
            last_failure_diagnostics=self._last_failures.get(name),
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

    def rebind_member(self, name: STTProviderName, api_key: str) -> bool:
        try:
            definition = self._definition(name)
        except KeyError:
            return False
        if definition.rebind is None:
            return False
        definition.rebind(api_key)
        self._states[name] = None
        self._last_failures.pop(name, None)
        return True

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        last_error: BaseException | None = None
        for definition in self.providers:
            if not self._is_eligible(definition):
                continue
            try:
                backend = definition.build_backend()
                session = await backend.open_session(projection=projection)
            except Exception as exc:
                kind = definition.classifier()(exc)
                self._handle_open_error(definition, exc, kind, projection.provider_epoch_id)
                last_error = exc
                continue
            return _RollingSession(
                definition=definition,
                inner=session,
                on_session_error=self._handle_session_error,
                on_terminal_observed=self._observe_terminal,
                _channel=self.channel,
            )
        if last_error is not None:
            raise last_error
        if any(definition.is_configured() for definition in self.providers):
            emit_basic_log(
                logger,
                "[Recognition] All configured speech recognition services are unavailable.",
                level=logging.WARNING,
            )
            raise RuntimeError(
                "All rolling ASR providers are excluded (quota/auth); "
                "see statuses() for per-provider state"
            )
        raise RuntimeError(
            "No rolling ASR provider is configured; configure an ElevenLabs, Gemini, "
            "or Deepgram API key"
        )

    def _handle_open_error(
        self,
        definition: RollingProviderDefinition,
        exc: Exception,
        kind: str,
        provider_epoch_id: str | None = None,
    ) -> None:
        diagnostics = stt_failure_report(
            exc, provider=definition.name.value, channel=self.channel, operation="readiness"
        ).diagnostics
        diagnostics = replace(
            diagnostics,
            fields={
                **diagnostics.fields,
                "policy_disposition": "excluded" if kind != _ERROR_KIND_TRANSIENT else "transient",
                **({"provider_epoch_id": provider_epoch_id} if provider_epoch_id else {}),
            },
        )
        exc.failure_diagnostics = diagnostics
        self._last_failures[definition.name] = diagnostics
        self._fallback_cause = self._fallback_cause or diagnostics
        emit_failure(
            logger, diagnostics, severity="warning", disposition="degraded", stage="readiness"
        )
        if kind == _ERROR_KIND_AUTH:
            self._mark(definition.name, RollingProviderState.AUTH_FAILED)
            return
        if kind in _PERSISTENT_EXHAUSTION_STATES:
            self._mark(definition.name, RollingProviderState.FREE_QUOTA_EXHAUSTED)
            return

    def _handle_session_error(
        self,
        definition: RollingProviderDefinition,
        exc: BaseException,
        diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        kind = definition.classifier()(exc)
        diagnostics = (
            diagnostics
            or stt_failure_report(
                exc, provider=definition.name.value, channel=None, operation="receive"
            ).diagnostics
        )
        diagnostics = replace(
            diagnostics,
            fields={
                **diagnostics.fields,
                "policy_disposition": "excluded" if kind != _ERROR_KIND_TRANSIENT else "transient",
            },
        )
        exc.failure_diagnostics = diagnostics
        self._last_failures[definition.name] = diagnostics
        self._fallback_cause = self._fallback_cause or diagnostics
        if kind == _ERROR_KIND_AUTH:
            self._mark(definition.name, RollingProviderState.AUTH_FAILED)
            return
        if kind in _PERSISTENT_EXHAUSTION_STATES:
            self._mark(definition.name, RollingProviderState.FREE_QUOTA_EXHAUSTED)

    def _observe_terminal(
        self,
        definition: RollingProviderDefinition,
        event: STTProviderTurnTerminal | STTProviderInputTerminal,
        channel: str | None,
    ) -> None:
        if event.outcome in {"failed", "degraded"} and event.failure_diagnostics is not None:
            self._last_failures[definition.name] = event.failure_diagnostics
        elif event.outcome in {"final", "empty"} and self._fallback_cause is not None:
            recovered = recognition_failure_diagnostics(
                None,
                provider=definition.name.value,
                channel=channel,
                operation="recovery",
                reason="recovered",
                identity=event.identity,
            )
            emit_failure(
                logger,
                replace(
                    recovered, code="stt.recovered", cause_id=self._fallback_cause.diagnostic_id
                ),
                severity="info",
                disposition="recovered",
                stage="recovery",
            )
            self._fallback_cause = None


@dataclass(slots=True)
class _RollingSession(STTBackendSession):
    """Session wrapper exposing provider identity, deadline, and error mapping."""

    definition: RollingProviderDefinition
    inner: STTBackendSession
    on_session_error: Callable[[RollingProviderDefinition, BaseException, ErrorDiagnostics], None]
    on_terminal_observed: Callable[
        [RollingProviderDefinition, STTProviderTurnTerminal | STTProviderInputTerminal, str | None],
        None,
    ]
    _channel: str | None = None
    _outer_stream: RecognitionStreamIdentity | None = field(default=None, init=False)
    _last_receipt_sequence: int = field(default=0, init=False)
    _unit_identities: OrderedDict[RecognitionUnitIdentity, RecognitionUnitIdentity] = field(
        default_factory=OrderedDict, init=False
    )

    def _record_session_error(
        self,
        exception: BaseException,
        identity: STTProviderTurnIdentity | None = None,
        stream: RecognitionStreamIdentity | None = None,
    ) -> None:
        diagnostics = recognition_failure_diagnostics(
            exception,
            provider=self.definition.name.value,
            channel=self._channel,
            operation="receive",
            reason="provider_event_stream_failed",
            identity=identity,
        )
        stream = stream or self._outer_stream
        if stream is not None:
            diagnostics = replace(
                diagnostics,
                generation=stream.activation_generation,
                fields={**diagnostics.fields, "provider_epoch_id": stream.provider_epoch_id},
            )
        self.on_session_error(self.definition, exception, diagnostics)

    @property
    def max_session_age_s(self) -> float | None:
        return getattr(self.inner, "max_session_age_s", None)

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

    @property
    def accepts_stream_input(self) -> bool:
        return (
            isinstance(self.inner, STTIndependentRecognitionSession)
            and self.inner.accepts_stream_input
        )

    @property
    def independent_recognition_units(self) -> bool:
        return (
            isinstance(self.inner, STTIndependentRecognitionSession)
            and self.inner.independent_recognition_units
        )

    def _scoped_inner(self) -> STTScopedTurnSession:
        if not isinstance(self.inner, STTScopedTurnSession):
            raise TypeError(f"rolling member {self.definition.name.value} is not scoped")
        return self.inner

    async def begin_stream(self, stream: RecognitionStreamIdentity) -> None:
        if not isinstance(self.inner, STTIndependentRecognitionSession):
            raise TypeError("rolling member does not accept independent stream admission")
        if self._outer_stream is not None and self._outer_stream != stream:
            raise RuntimeError("rolling member cannot bind unrelated recognition streams")
        try:
            self._channel = stream.channel
            await self.inner.begin_stream(stream)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc, stream=stream)
            raise
        self._outer_stream = stream

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        self._channel = request.channel
        if self.independent_recognition_units:
            identity = request.identity
            stream = RecognitionStreamIdentity(
                channel=request.channel,
                activation_generation=identity.segment.activation_generation,
                capture_epoch=identity.segment.capture_epoch,
                provider_epoch_id=identity.provider_epoch_id,
                settings_scope=identity.settings_scope,
            )
            if self._outer_stream is None:
                self._outer_stream = stream
            elif self._outer_stream != stream:
                raise RuntimeError("rolling member cannot bind unrelated recognition streams")
        try:
            await self._scoped_inner().begin_turn(request)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc, request.identity)
            raise

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        try:
            await self._scoped_inner().send_turn_audio(
                identity,
                pcm16le,
                payload_sequence=payload_sequence,
                source_ranges=source_ranges,
                context_only=context_only,
            )
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc, identity)
            raise

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        try:
            await self._scoped_inner().seal_turn(
                identity,
                sealed_content_ranges=sealed_content_ranges,
                seal_reason=seal_reason,
                observed_trailing_silence_ms=observed_trailing_silence_ms,
            )
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc, identity)
            raise

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        try:
            await self._scoped_inner().abort_turn(identity, reason=reason)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc, identity)
            raise

    async def send_stream_audio(
        self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
    ) -> None:
        if not isinstance(self.inner, STTIndependentRecognitionSession):
            raise TypeError("rolling member does not accept independent stream audio")
        try:
            await self.inner.send_stream_audio(pcm16le, source_ranges=source_ranges)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc)
            raise

    async def end_stream(self, *, reason: str) -> None:
        if not isinstance(self.inner, STTIndependentRecognitionSession):
            raise TypeError("rolling member does not accept independent stream fences")
        try:
            await self.inner.end_stream(reason=reason)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc)
            raise

    def recognition_source_covers(self, ranges: tuple[AudioCaptureSpan, ...]) -> bool:
        return isinstance(
            self.inner, STTIndependentRecognitionSession
        ) and self.inner.recognition_source_covers(ranges)

    async def turn_events(self) -> AsyncIterator[STTProviderTurnEvent]:
        try:
            async for event in self._scoped_inner().turn_events():
                if isinstance(event, STTProviderTurnTerminal | STTProviderInputTerminal):
                    if (
                        event.outcome in {"failed", "degraded"}
                        and event.failure_diagnostics is None
                    ):
                        event = replace(
                            event,
                            failure_diagnostics=recognition_failure_diagnostics(
                                None,
                                provider=self.definition.name.value,
                                channel=self._channel or "unknown",
                                operation="turn_terminal",
                                reason=event.failure_reason or event.outcome,
                                identity=event.identity,
                            ),
                        )
                    self.on_terminal_observed(self.definition, event, self._channel)
                if isinstance(event, (STTRecognitionUnit, STTRecognitionUnitTerminal)):
                    unit = event if isinstance(event, STTRecognitionUnit) else event.unit
                    identity = self._remap_recognition_identity(unit.identity)
                    if identity is None:
                        continue
                    unit = replace(unit, identity=identity)
                    event = (
                        unit if isinstance(event, STTRecognitionUnit) else replace(event, unit=unit)
                    )
                yield event
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_session_error(exc)
            raise

    def _remap_recognition_identity(
        self, identity: RecognitionUnitIdentity
    ) -> RecognitionUnitIdentity | None:
        outer = self._outer_stream
        inner = identity.stream
        if outer is None or (
            inner.channel != outer.channel
            or inner.activation_generation != outer.activation_generation
            or inner.capture_epoch != outer.capture_epoch
            or inner.provider_epoch_id != outer.provider_epoch_id
            or inner.settings_scope != outer.settings_scope
        ):
            return None
        remapped = self._unit_identities.get(identity)
        if remapped is not None:
            return remapped
        self._last_receipt_sequence += 1
        remapped = RecognitionUnitIdentity(
            stream=outer,
            unit_id=identity.unit_id,
            receipt_sequence=self._last_receipt_sequence,
        )
        self._unit_identities[identity] = remapped
        if len(self._unit_identities) > 4096:
            self._unit_identities.popitem(last=False)
        return remapped

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
                self._record_session_error(exc)
            raise


import asyncio  # placed at bottom to keep the main logic compact
