"""Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    RecoverableSTTSessionError,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTNativeProvenance,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection

logger = logging.getLogger(__name__)

GEMINI_TRANSCRIBE_STT_MODEL = "gemini-3.5-transcribe-live"
GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ = 16000
GEMINI_TRANSCRIBE_FINALIZE_TIMEOUT_S = 2.0


class GeminiTranscribeFinalizeTimeout(RecoverableSTTSessionError):
    pass


@dataclass(slots=True, eq=False)
class _PendingTurn:
    identity: STTProviderTurnIdentity | None = None
    latest_interim: str = ""
    final_emitted: bool = False
    authoritative_received: bool = False
    authoritative_text: str = ""
    activity_end_received: bool = False
    activity_end_ack: asyncio.Event = field(default_factory=asyncio.Event)
    timeout_task: asyncio.Task[None] | None = None
    provenance: list[STTNativeProvenance] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _StartTurn:
    turn: _PendingTurn
    completion: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _EndTurn:
    turn: _PendingTurn
    completion: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _AudioWrite:
    turn: _PendingTurn
    pcm16le: bytes
    completion: asyncio.Future[None]


_STOP = object()


@dataclass(slots=True)
class _GeminiClientResources:
    client: Any
    sync_transport: Any
    async_transport: Any
    config: Any


def _build_live_config_sync(language_codes: Sequence[str], custom_vocabulary: Sequence[str]) -> Any:
    from google.genai import types

    transcription_config_kwargs: dict[str, Any] = {"mode": "VERBATIM"}
    if language_codes:
        transcription_config_kwargs["language_codes"] = list(language_codes)
    if custom_vocabulary:
        transcription_config_kwargs["custom_vocabulary"] = list(custom_vocabulary)
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(**transcription_config_kwargs),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
    )


def _create_transports_sync() -> tuple[Any, Any]:
    import httpx

    sync_transport = httpx.Client(timeout=None, follow_redirects=True)
    try:
        async_transport = httpx.AsyncClient(timeout=None, follow_redirects=True)
    except BaseException:
        with contextlib.suppress(Exception):
            sync_transport.close()
        raise
    return sync_transport, async_transport


def _build_http_options_sync(sync_transport: Any, async_transport: Any) -> Any:
    from google.genai import types

    return types.HttpOptions(httpx_client=sync_transport, httpx_async_client=async_transport)


def _create_genai_client_sync(api_key: str, http_options: Any) -> Any:
    from google import genai

    return genai.Client(api_key=api_key, http_options=http_options)


def _prepare_gemini_resources_sync(
    api_key: str, language_codes: Sequence[str], custom_vocabulary: Sequence[str]
) -> _GeminiClientResources:
    config = _build_live_config_sync(language_codes, custom_vocabulary)
    sync_transport, async_transport = _create_transports_sync()
    try:
        http_options = _build_http_options_sync(sync_transport, async_transport)
        client = _create_genai_client_sync(api_key, http_options)
    except BaseException:
        with contextlib.suppress(Exception):
            sync_transport.close()
        with contextlib.suppress(Exception):
            asyncio.run(async_transport.aclose())
        raise
    return _GeminiClientResources(
        client=client,
        sync_transport=sync_transport,
        async_transport=async_transport,
        config=config,
    )


def gemini_transcribe_language_codes(source_language: str | None) -> list[str]:
    if not source_language:
        return []
    from puripuly_heart.core.language import gemini_transcribe_language_hint

    mapped = gemini_transcribe_language_hint(source_language)
    return [mapped] if mapped else []


def _recv_failure_fields(exc: BaseException) -> tuple[str, object, object, str]:
    exception_class = type(exc).__name__
    api_code = getattr(exc, "code", None)
    api_status = getattr(exc, "status", None)
    return exception_class, api_code, api_status, _recv_message_kind(exc, api_code, api_status)


def _recv_message_kind(exc: BaseException, api_code: object, api_status: object) -> str:
    class_name = type(exc).__name__.lower().replace("_", "")
    status_text = str(api_status or "").lower()
    if "goaway" in class_name or "go_away" in status_text:
        return "go_away"
    if (
        "connection" in class_name
        or "closed" in class_name
        or "websocket" in class_name
        or "unavailable" in status_text
    ):
        return "connection_closed"
    if (
        api_code in {400, 422}
        or "invalid" in status_text
        or "validation" in class_name
        or "invalidargument" in class_name
    ):
        return "validation"
    return "other"


@dataclass(slots=True)
class GeminiTranscribeSTTBackend(STTBackend):
    """Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

    api_key: str
    language_codes: Sequence[str] = ()
    custom_vocabulary: Sequence[str] = ()
    model: str = GEMINI_TRANSCRIBE_STT_MODEL
    sample_rate_hz: int = GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    finalize_timeout_s: float = GEMINI_TRANSCRIBE_FINALIZE_TIMEOUT_S
    live_connect_factory: Callable[[str, Any], Any] | None = None

    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession:
        if self.sample_rate_hz != GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ:
            raise ValueError(
                "sample_rate_hz must be 16000 for Gemini Transcribe Live transcription"
            )
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.finalize_timeout_s <= 0:
            raise ValueError("finalize_timeout_s must be > 0")
        session = _GeminiTranscribeLiveSession(
            api_key=self.api_key,
            language_codes=list(self.language_codes),
            custom_vocabulary=list(self.custom_vocabulary),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            finalize_timeout_s=self.finalize_timeout_s,
            live_connect_factory=self.live_connect_factory,
            projection=projection,
        )
        try:
            await session.start()
        except BaseException:
            with contextlib.suppress(BaseException):
                await session.close()
            raise
        return session

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False

        def _check() -> bool:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status == 200
            except urllib.error.HTTPError as e:
                raise Exception(f"HTTP {e.code}: {e.reason}")
            except Exception as e:
                raise Exception(f"Connection failed: {e}")

        return await asyncio.to_thread(_check)


@dataclass(slots=True)
class _GeminiTranscribeLiveSession(STTBackendSession):
    """Internal session wrapping a google-genai Live API AsyncSession."""

    api_key: str
    language_codes: list[str]
    custom_vocabulary: list[str]
    model: str
    sample_rate_hz: int
    connect_timeout_s: float
    finalize_timeout_s: float
    live_connect_factory: Callable[[str, Any], Any] | None = None
    projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION

    _event_projection: STTSessionEventProjection = field(init=False, repr=False)
    _send_queue: asyncio.Queue[_StartTurn | _EndTurn | _AudioWrite | bytes | object] = field(
        init=False, repr=False
    )
    _live_context: Any = field(init=False, default=None, repr=False)
    _live_session: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _capture_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    _streaming_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    _pending_turns: deque[_PendingTurn] = field(init=False, default_factory=deque, repr=False)
    _protocol_failed: bool = field(init=False, default=False)
    _retirement_due: bool = field(init=False, default=False)
    _retirement_reason: str | None = field(init=False, default=None, repr=False)
    _client_resources: _GeminiClientResources | None = field(init=False, default=None, repr=False)
    _setup_future: Any = field(init=False, default=None, repr=False)
    _setup_executor: Any = field(init=False, default=None, repr=False)
    _handshake_task: asyncio.Task[Any] | None = field(init=False, default=None, repr=False)
    _teardown_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _teardown_done: bool = field(init=False, default=False)
    _scoped_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    allows_interim_timeout_fallback: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self._event_projection = STTSessionEventProjection(self.projection)
        self._send_queue = asyncio.Queue(maxsize=258)

    async def start(self) -> None:
        if self._stopped:
            raise RuntimeError("Gemini Transcribe Live session is closed")
        self._teardown_done = False
        self._teardown_task = None
        self._setup_future = None
        self._setup_executor = None
        self._handshake_task = None
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemini-stt-setup"
        )
        self._setup_executor = executor
        try:
            raw = executor.submit(
                _prepare_gemini_resources_sync,
                self.api_key,
                list(self.language_codes),
                list(self.custom_vocabulary),
            )
        except BaseException:
            self._setup_executor = None
            executor.shutdown(wait=False)
            raise
        self._setup_future = raw
        try:
            resources = await asyncio.wrap_future(raw)
        except BaseException:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise
        if self._stopped:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise RuntimeError("Gemini Transcribe Live session closed during setup")
        self._client_resources = resources
        executor.shutdown(wait=False)
        try:
            factory = self.live_connect_factory
            if factory is None:
                factory = resources.client.aio.live.connect
            live_context = factory(model=self.model, config=resources.config)
        except BaseException:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise
        self._live_context = live_context
        handshake_task = asyncio.create_task(live_context.__aenter__())
        self._handshake_task = handshake_task
        try:
            live_session = await asyncio.wait_for(handshake_task, timeout=self.connect_timeout_s)
        except BaseException:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise
        self._handshake_task = None
        if self._stopped:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise RuntimeError("Gemini Transcribe Live session closed during connect")
        self._live_session = live_session
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _teardown(self) -> None:
        task = self._teardown_task
        if task is None:
            if self._teardown_done:
                return
            task = asyncio.create_task(self._teardown_body())
            self._teardown_task = task
        if task is asyncio.current_task():
            return
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue

    async def _teardown_body(self) -> None:
        raw = self._setup_future
        if raw is not None and not raw.done():
            try:
                await asyncio.wrap_future(raw)
            except asyncio.CancelledError:
                body_task = asyncio.current_task()
                if body_task is not None and body_task.cancelling():
                    raise
            except Exception:
                pass
        outcome = None
        if raw is not None and raw.done():
            try:
                outcome = raw.result()
            except BaseException:
                outcome = None
        if outcome is not None and outcome is not self._client_resources:
            await self._close_resources(outcome)
        executor, self._setup_executor = self._setup_executor, None
        if executor is not None:
            executor.shutdown(wait=False)
        handshake_task, self._handshake_task = self._handshake_task, None
        if handshake_task is not None:
            if not handshake_task.done():
                handshake_task.cancel()
            try:
                await handshake_task
            except asyncio.CancelledError:
                body_task = asyncio.current_task()
                if body_task is not None and body_task.cancelling():
                    raise
            except Exception:
                pass
        for task in (self._send_task, self._recv_task):
            if task is not None:
                task.cancel()
        pending_tasks = [task for task in (self._send_task, self._recv_task) if task is not None]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._send_task = None
        self._recv_task = None
        live_context, self._live_context = self._live_context, None
        live_session, self._live_session = self._live_session, None
        if live_context is not None:
            with contextlib.suppress(Exception):
                await live_context.__aexit__(None, None, None)
        elif live_session is not None:
            with contextlib.suppress(Exception):
                await live_session.close()
        await self._release_client_resources()
        self._teardown_done = True

    async def _close_resources(self, resources: _GeminiClientResources) -> None:
        with contextlib.suppress(Exception):
            await resources.client.aio.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(resources.client.close)
        with contextlib.suppress(Exception):
            await resources.async_transport.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(resources.sync_transport.close)

    async def _release_client_resources(self) -> None:
        resources, self._client_resources = self._client_resources, None
        if resources is None:
            return
        await self._close_resources(resources)

    async def _send_loop(self) -> None:
        item: _StartTurn | _EndTurn | _AudioWrite | bytes | object = _STOP
        try:
            while not self._stopped:
                item = await self._send_queue.get()
                if item is _STOP:
                    return
                if isinstance(item, _StartTurn):
                    from google.genai import types

                    turn = item.turn
                    if not self._turn_can_write(turn):
                        self._resolve_write(
                            item.completion,
                            RuntimeError("Gemini Transcribe Live turn is no longer active"),
                        )
                        continue
                    self._streaming_turn = turn
                    await self._send_realtime(activity_start=types.ActivityStart())
                    if not self._turn_can_write(turn):
                        self._resolve_write(
                            item.completion,
                            RuntimeError("Gemini Transcribe Live turn was retired during write"),
                        )
                        continue
                    self._resolve_write(item.completion, None)
                    continue
                if isinstance(item, _EndTurn):
                    from google.genai import types

                    turn = item.turn
                    if not self._turn_can_write(turn):
                        self._resolve_write(
                            item.completion,
                            RuntimeError("Gemini Transcribe Live turn is no longer active"),
                        )
                        continue
                    self._pending_turns.append(turn)
                    await self._send_realtime(activity_end=types.ActivityEnd())
                    completed = turn.authoritative_received and turn.activity_end_received
                    if self._protocol_failed and not completed:
                        if turn in self._pending_turns:
                            self._pending_turns.remove(turn)
                        self._resolve_write(
                            item.completion,
                            RuntimeError("Gemini Transcribe Live turn was retired during finalize"),
                        )
                        return
                    self._resolve_write(item.completion, None)
                    if turn in self._pending_turns:
                        turn.timeout_task = asyncio.create_task(self._finalize_timeout(turn))
                    await turn.activity_end_ack.wait()
                    if self._streaming_turn is turn:
                        self._streaming_turn = None
                    if self._retirement_due and not self._has_active_turn():
                        self._fail_idle_protocol(self._retirement_reason or "gemini_retirement_due")
                    if self._protocol_failed:
                        return
                    continue
                if isinstance(item, _AudioWrite):
                    if not self._turn_can_write(item.turn):
                        self._resolve_write(
                            item.completion,
                            RuntimeError("Gemini Transcribe Live turn is no longer active"),
                        )
                        continue
                    await self._send_realtime(
                        audio={
                            "data": item.pcm16le,
                            "mime_type": f"audio/pcm;rate={self.sample_rate_hz}",
                        },
                    )
                    if not self._turn_can_write(item.turn):
                        self._resolve_write(
                            item.completion,
                            RuntimeError("Gemini Transcribe Live turn was retired during write"),
                        )
                        continue
                    self._resolve_write(item.completion, None)
                    continue
                if isinstance(item, bytes):
                    await self._send_realtime(
                        audio={
                            "data": item,
                            "mime_type": f"audio/pcm;rate={self.sample_rate_hz}",
                        },
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._resolve_write(getattr(item, "completion", None), exc)
            logger.exception("Gemini Transcribe Live send loop error")
            self._put_event(exc)
            self._scoped_transport_failure("gemini_write_failed")
        finally:
            self._fail_pending_writes()

    def _turn_can_write(self, turn: _PendingTurn) -> bool:
        if self._stopped or self._protocol_failed:
            return False
        if turn.identity is None:
            return True
        return self._scoped_turn is turn

    async def _recv_loop(self) -> None:
        try:
            while not self._stopped:
                live_session = self._live_session
                if live_session is None:
                    return
                async for message in live_session.receive():
                    self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            exception_class, api_code, api_status, message_kind = _recv_failure_fields(exc)
            logger.exception(
                "Gemini Transcribe Live recv loop error exception_class=%s "
                "api_code=%s api_status=%s message_kind=%s",
                exception_class,
                api_code,
                api_status,
                message_kind,
            )
            self._put_event(exc)
            self._scoped_transport_failure("gemini_receive_failed", orderly=False)
        finally:
            self._put_event(None)
            self._scoped_transport_failure("gemini_connection_ended", orderly=True)

    def _handle_message(self, message: Any) -> None:
        if self._protocol_failed:
            return
        if getattr(message, "go_away", None) is not None:
            self._mark_retirement_due("gemini_go_away")
            if not self._has_active_turn():
                self._fail_idle_protocol("gemini_go_away")
                return
        provenance = self._message_provenance(message)
        content = message.server_content
        if content is not None:
            interim = content.interim_input_transcription
            if interim is not None and interim.text:
                text = str(interim.text)
                turn = self._turn_for_interim()
                if turn is not None:
                    turn.latest_interim = text
                    if turn.identity is not None:
                        sequence = self._event_projection.next_update_sequence(turn.identity)
                        if sequence is not None:
                            self._event_projection.put_update(
                                STTProviderTurnUpdate(
                                    identity=turn.identity,
                                    sequence=sequence,
                                    stability="provisional",
                                    assembly="replace",
                                    text=text,
                                    provenance=provenance,
                                )
                            )
            final = content.input_transcription
            if final is not None:
                self._handle_final(str(final.text or ""), provenance)

        voice_activity = getattr(message, "voice_activity", None)
        activity_type = getattr(voice_activity, "voice_activity_type", None)
        activity_value = getattr(activity_type, "value", activity_type)
        if str(activity_value or "").upper() == "ACTIVITY_END":
            self._handle_activity_end_ack(self._message_provenance(message, barrier="activity_end"))

    def _has_active_turn(self) -> bool:
        return (
            self._scoped_turn is not None
            or self._capture_turn is not None
            or self._streaming_turn is not None
            or bool(self._pending_turns)
        )

    def _mark_retirement_due(self, reason: str) -> None:
        self._retirement_due = True
        if self._retirement_reason is None:
            self._retirement_reason = reason

    def _fail_idle_protocol(self, reason: str) -> None:
        if self._protocol_failed:
            return
        self._protocol_failed = True
        self._mark_retirement_due(reason)
        self._event_projection.end_epoch(orderly=False, reason=reason)

    @staticmethod
    def _message_provenance(
        message: Any,
        *,
        barrier: str | None = None,
    ) -> STTNativeProvenance:
        _ = message
        return STTNativeProvenance(barrier=barrier)

    def _turn_for_interim(self) -> _PendingTurn | None:
        return self._streaming_turn

    def _handle_final(self, text: str, provenance: STTNativeProvenance) -> None:
        turn = next(
            (item for item in self._pending_turns if not item.authoritative_received),
            None,
        )
        if turn is None:
            if not self._has_active_turn():
                self._fail_idle_protocol("gemini_unsolicited_authoritative")
            return
        turn.authoritative_received = True
        turn.authoritative_text = text
        turn.provenance.append(provenance)
        if turn.identity is not None:
            sequence = self._event_projection.next_update_sequence(turn.identity)
            if sequence is not None:
                self._event_projection.put_update(
                    STTProviderTurnUpdate(
                        identity=turn.identity,
                        sequence=sequence,
                        stability="stable",
                        assembly="replace",
                        text=text,
                        provenance=provenance,
                    )
                )
        if turn.activity_end_received:
            self._complete_turn(turn)

    def _handle_activity_end_ack(self, provenance: STTNativeProvenance) -> None:
        if not self._pending_turns:
            if not self._has_active_turn():
                self._fail_idle_protocol("gemini_unsolicited_activity_end")
            return
        turn = self._pending_turns[0]
        if not turn.activity_end_received:
            turn.activity_end_received = True
            turn.provenance.append(provenance)
        if turn.authoritative_received:
            self._complete_turn(turn)

    def _complete_turn(self, turn: _PendingTurn) -> None:
        if turn.identity is None:
            if turn not in self._pending_turns:
                return
            self._pending_turns.remove(turn)
            self._cancel_turn_timeout(turn)
            self._emit_turn_final(turn, turn.authoritative_text)
            if self._retirement_due:
                self._protocol_failed = True
            turn.activity_end_ack.set()
            return
        self._complete_scoped_turn(turn)

    def _complete_scoped_turn(self, turn: _PendingTurn) -> None:
        identity = turn.identity
        if identity is None or self._scoped_turn is not turn:
            return
        if turn in self._pending_turns:
            self._pending_turns.remove(turn)
        self._cancel_turn_timeout(turn)
        text = turn.authoritative_text
        disposition = "retire" if self._retirement_due else "reuse"
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final" if text else "empty",
                text=text,
                text_authority="authoritative",
                failure_reason=self._retirement_reason,
                epoch_disposition=disposition,
                provenance=tuple(turn.provenance),
            )
        )
        self._scoped_turn = None
        if disposition == "retire":
            self._protocol_failed = True
        turn.activity_end_ack.set()

    def _emit_turn_final(self, turn: _PendingTurn, text: str) -> None:
        if turn.final_emitted:
            return
        turn.final_emitted = True
        if turn.identity is None:
            self._event_projection.put_legacy(STTBackendTranscriptEvent(text=text, is_final=True))

    def _cancel_turn_timeout(self, turn: _PendingTurn) -> None:
        task = turn.timeout_task
        turn.timeout_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _finalize_timeout(self, turn: _PendingTurn) -> None:
        try:
            await asyncio.sleep(self.finalize_timeout_s)
        except asyncio.CancelledError:
            return
        if self._stopped or self._protocol_failed or turn not in self._pending_turns:
            return
        self._protocol_failed = True
        self._pending_turns.remove(turn)
        if turn.identity is None:
            text = turn.authoritative_text if turn.authoritative_received else turn.latest_interim
            self._emit_turn_final(turn, text)
        else:
            self._timeout_scoped_turn(turn)
        turn.activity_end_ack.set()
        logger.warning(
            "[STT] Gemini Transcribe Live finalize timed out after %.2fs; recycling session",
            self.finalize_timeout_s,
        )
        self._put_event(
            GeminiTranscribeFinalizeTimeout(
                f"Gemini Transcribe Live finalize timed out after {self.finalize_timeout_s:.2f}s"
            )
        )

    @staticmethod
    def _resolve_write(
        completion: asyncio.Future[None] | None,
        error: BaseException | None,
    ) -> None:
        if completion is None or completion.done():
            return
        if error is None:
            completion.set_result(None)
        else:
            completion.set_exception(error)

    def _fail_pending_writes(self) -> None:
        error = RuntimeError("Gemini Transcribe Live writer stopped")
        while not self._send_queue.empty():
            try:
                item = self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._resolve_write(getattr(item, "completion", None), error)

    def _timeout_scoped_turn(self, turn: _PendingTurn) -> None:
        identity = turn.identity
        if identity is None or self._scoped_turn is not turn:
            return
        if turn.authoritative_received:
            text = turn.authoritative_text
            outcome = "final" if text else "empty"
            authority = "authoritative"
        elif turn.latest_interim:
            text = turn.latest_interim
            outcome = "degraded"
            authority = "degraded"
        else:
            text = ""
            outcome = "failed"
            authority = "none"
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                text_authority=authority,
                failure_reason="gemini_finalize_timeout",
                epoch_disposition="retire",
                provenance=tuple(turn.provenance),
            )
        )
        self._scoped_turn = None

    def _scoped_transport_failure(self, reason: str, *, orderly: bool = False) -> None:
        if self._protocol_failed:
            return
        self._protocol_failed = True
        self._mark_retirement_due(reason)
        turn = self._scoped_turn
        identity = turn.identity if turn is not None else None
        provider_turn_id = identity.provider_turn_id if identity is not None else None
        if turn is not None and identity is not None:
            if turn in self._pending_turns:
                self._pending_turns.remove(turn)
            self._cancel_turn_timeout(turn)
            text = turn.authoritative_text if turn.authoritative_received else turn.latest_interim
            self._event_projection.terminal(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="degraded" if text else "failed",
                    text=text,
                    text_authority="degraded" if text else "none",
                    failure_reason=reason,
                    epoch_disposition="retire",
                    provenance=tuple(turn.provenance),
                )
            )
            self._scoped_turn = None
            turn.activity_end_ack.set()
        self._capture_turn = None
        if self._streaming_turn is not None:
            self._streaming_turn.activity_end_ack.set()
        self._event_projection.end_epoch(
            orderly=orderly,
            reason=reason,
            provider_turn_id=provider_turn_id,
        )

    def _require_scoped_turn(self, identity: STTProviderTurnIdentity) -> _PendingTurn:
        self._event_projection.require_open(identity)
        turn = self._scoped_turn
        if turn is None or turn.identity != identity:
            raise RuntimeError("Gemini scoped turn identity mismatch")
        return turn

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._stopped or self._protocol_failed or self._retirement_due:
            raise RuntimeError("Gemini Transcribe Live session is closed")
        if self._scoped_turn is not None or self._capture_turn is not None:
            raise RuntimeError("Gemini allows one unresolved scoped turn")
        self._event_projection.begin(request)
        completion = asyncio.get_running_loop().create_future()
        turn = _PendingTurn(identity=request.identity)
        self._scoped_turn = turn
        self._capture_turn = turn
        await self._send_queue.put(_StartTurn(turn, completion))
        await completion

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        turn = self._require_scoped_turn(identity)
        if self._capture_turn is not turn:
            raise RuntimeError("Gemini scoped turn is sealed")
        self._event_projection.validate_payload(identity, payload_sequence)
        _ = source_ranges, context_only
        completion = asyncio.get_running_loop().create_future()
        await self._send_queue.put(_AudioWrite(turn, pcm16le, completion))
        await completion
        self._event_projection.payload_written(identity, payload_sequence)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        turn = self._require_scoped_turn(identity)
        if self._capture_turn is not turn:
            raise RuntimeError("Gemini scoped turn is already sealed")
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        self._event_projection.seal(identity)
        self._capture_turn = None
        completion = asyncio.get_running_loop().create_future()
        await self._send_queue.put(_EndTurn(turn, completion))
        await completion

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        turn = self._require_scoped_turn(identity)
        self._protocol_failed = True
        if turn in self._pending_turns:
            self._pending_turns.remove(turn)
        self._cancel_turn_timeout(turn)
        turn.activity_end_ack.set()
        self._capture_turn = None
        self._streaming_turn = None
        self._scoped_turn = None
        self._event_projection.terminal(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="cancelled",
                text_authority="none",
                failure_reason=reason,
                epoch_disposition="retire",
            )
        )
        self._event_projection.end_epoch(
            orderly=False,
            reason=reason,
            provider_turn_id=identity.provider_turn_id,
        )

    async def turn_events(self):
        async for event in self._event_projection.turn_events():
            yield event

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or self._protocol_failed:
            return
        if self._send_queue.qsize() >= 256:
            raise RuntimeError("Gemini Transcribe Live audio queue overflow")
        if self._capture_turn is None:
            self._capture_turn = _PendingTurn()
            await self._send_queue.put(_StartTurn(self._capture_turn))
        self._send_queue.put_nowait(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if self._stopped:
            return
        if self._capture_turn is not None:
            turn = self._capture_turn
            self._capture_turn = None
            await self._send_queue.put(_EndTurn(turn))

    async def _send_realtime(self, **kwargs: Any) -> None:
        live_session = self._live_session
        if live_session is None:
            return
        await live_session.send_realtime_input(**kwargs)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._send_queue.put(_STOP)
        if self._pending_turns:
            logger.warning(
                "[STT] Gemini Transcribe Live session closed with unresolved finalize requests count=%s",
                len(self._pending_turns),
            )
        for turn in self._pending_turns:
            self._cancel_turn_timeout(turn)
            turn.activity_end_ack.set()
        self._pending_turns.clear()
        self._capture_turn = None
        self._streaming_turn = None
        self._put_event(None)

    async def close(self) -> None:
        await self.stop()
        current_task = asyncio.current_task()
        try:
            await self._teardown()
        except (asyncio.CancelledError, Exception):
            pass
        self._event_projection.close()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        async for event in self._event_projection.events():
            yield event

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._event_projection.put_legacy(event)
