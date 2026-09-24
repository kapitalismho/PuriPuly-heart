"""Issue 185 outcome G: mocked adverse cases on current remote ASR adapters.

No live provider calls, no protocol-flag changes, and no product mutation.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.stt.backend import (
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTScopedTurnSession,
    STTSessionProjection,
)
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, _RollingSession
from puripuly_heart.providers.stt.custom import (
    _OfflineOpenAITranscriptionSession,
    _StreamingOpenAIRealtimeSession,
)
from puripuly_heart.providers.stt.deepgram import _DeepgramSDKSession, _ThreadWrite
from puripuly_heart.providers.stt.elevenlabs_scribe import _ElevenLabsScribeSession
from puripuly_heart.providers.stt.gemini_transcribe import _GeminiTranscribeLiveSession
from puripuly_heart.providers.stt.qwen_audio import QwenAudioSessionState, _QwenAudioSession
from puripuly_heart.providers.stt.soniox import _STOP as SONIOX_STOP
from puripuly_heart.providers.stt.soniox import _SonioxSession

PCM = b"\x00\x01" * 80
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("g_results.json")
SEAL = {
    "sealed_content_ranges": (),
    "seal_reason": "delivery_pause",
    "observed_trailing_silence_ms": 800,
}


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _versions() -> dict[str, str]:
    found = {"python": sys.version.split()[0]}
    for name in (
        "deepgram-sdk",
        "elevenlabs",
        "google-genai",
        "httpx",
        "websockets",
        "dashscope",
    ):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "not-installed"
    return found


def _request(provider: str, epoch: str, order: int, turn: str) -> STTProviderTurnRequest:
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=AudioSegmentIdentity(1, order, uuid4(), 1),
            provider_epoch_id=epoch,
            provider_turn_id=turn,
            settings_scope=(provider, order),
        ),
        settings=AudioSegmentSettingsSnapshot(
            provider,
            (provider,),
            ("issue-185-g",),
            "peer",
            "en",
            ("en",),
            16000,
            0.5,
            800,
            300,
        ),
        channel="peer",
    )


def _projection(epoch: str) -> STTSessionProjection:
    return STTSessionProjection(mode="scoped", provider_epoch_id=epoch)


def _events(session: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in list(session._event_projection._scoped_events._events):
        if isinstance(event, STTProviderTurnTerminal):
            rows.append(
                {
                    "kind": "terminal",
                    "turn": event.identity.provider_turn_id,
                    "outcome": event.outcome,
                    "text": event.text,
                    "authority": event.text_authority,
                    "reason": event.failure_reason,
                    "epoch_disposition": event.epoch_disposition,
                    "barriers": [item.barrier for item in event.provenance],
                }
            )
        elif isinstance(event, STTProviderTurnUpdate):
            rows.append(
                {
                    "kind": "update",
                    "turn": event.identity.provider_turn_id,
                    "text": event.text,
                    "stability": event.stability,
                }
            )
        elif isinstance(event, STTProviderEpochEnded):
            rows.append(
                {
                    "kind": "epoch_end",
                    "reason": event.reason,
                    "orderly": event.orderly,
                    "turn": event.provider_turn_id,
                }
            )
    return rows


def _terminals(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["kind"] == "terminal"]


async def _attempt(awaitable: object) -> dict[str, object]:
    try:
        await awaitable
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return {"ok": True}


def _case(
    adapter: str, name: str, *, passed: bool, observation: str, **details: object
) -> dict[str, object]:
    return {
        "adapter": adapter,
        "case": name,
        "status": "passed" if passed else "failed",
        "observation": observation,
        **details,
    }


def _overlap_flag(session: object) -> bool:
    return bool(getattr(session, "allows_sealed_turn_overlap", False)) or bool(
        session._event_projection._allows_sealed_turn_overlap
    )


async def _send(session: object, request: STTProviderTurnRequest) -> None:
    await session.send_turn_audio(
        request.identity,
        PCM,
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )


async def _seal(session: object, request: STTProviderTurnRequest) -> None:
    await session.seal_turn(request.identity, **SEAL)


class _DeepgramDrain:
    def __init__(self, session: _DeepgramSDKSession) -> None:
        self._session = session
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._session._audio_q.get(timeout=0.02)
            except queue.Empty:
                continue
            if isinstance(item, _ThreadWrite):
                self._session._resolve_thread_write(item.completion, None)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _deepgram(epoch: str, drain: float) -> tuple[_DeepgramSDKSession, _DeepgramDrain]:
    session = _DeepgramSDKSession(
        api_key="probe",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=0.1,
        keyterms=[],
        drain_timeout_s=drain,
        projection=_projection(epoch),
    )
    session._loop = asyncio.get_running_loop()
    return session, _DeepgramDrain(session)


def _dg_result(text: str, *, is_final: bool, from_finalize: bool) -> SimpleNamespace:
    return SimpleNamespace(
        speech_final=True,
        is_final=is_final,
        from_finalize=from_finalize,
        metadata=SimpleNamespace(from_finalize=from_finalize, request_id="conn"),
        channel=SimpleNamespace(alternatives=[SimpleNamespace(transcript=text)]),
    )


async def _probe_deepgram() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session, drain = _deepgram("dg-1", 30.0)
    request_a = _request("deepgram", "dg-1", 1, "A")
    request_b = _request("deepgram", "dg-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    session._build_transcript_event(_dg_result("ALPHA", is_final=True, from_finalize=False))
    await asyncio.sleep(0)
    before_ack = _terminals(_events(session))
    session._build_transcript_event(_dg_result("", is_final=True, from_finalize=True))
    await asyncio.sleep(0)
    acked = _events(session)
    admitted = await _attempt(session.begin_turn(request_b))
    await _seal(session, request_b)
    session._handle_scoped_result(
        request_a.identity,
        "LATE-A",
        True,
        True,
        STTNativeProvenance(native_request_id="conn", barrier="finalize_ack", from_finalize=True),
    )
    session._build_transcript_event(_dg_result("LATE-ACTIVE", is_final=True, from_finalize=True))
    await asyncio.sleep(0)
    late = _events(session)
    rows.append(
        _case(
            "deepgram",
            "delayed_a_then_late_ack_on_next_seal",
            passed=(
                not blocked["ok"]
                and not before_ack
                and admitted["ok"]
                and _terminals(acked)[0]["turn"] == "A"
                and _terminals(acked)[0]["text"] == "ALPHA"
                and any(
                    row["turn"] == "B" and row["text"] == "LATE-ACTIVE" for row in _terminals(late)
                )
                and not any(row["text"] == "LATE-A" for row in _terminals(late))
                and not _overlap_flag(session)
            ),
            observation=(
                "B cannot begin while A awaits from_finalize. speech_final/is_final is not the barrier. "
                "A retired identity ignores a late ack, but a from_finalize captured against the then-active "
                "identity is attributed to B. The connection request_id is not a turn key."
            ),
            b_while_unresolved=blocked,
            events=late,
            overlap_flag=_overlap_flag(session),
        )
    )
    drain.stop()

    empty, empty_drain = _deepgram("dg-empty", 30.0)
    empty_request = _request("deepgram", "dg-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    empty._build_transcript_event(_dg_result("", is_final=False, from_finalize=True))
    await asyncio.sleep(0)
    empty._build_transcript_event(_dg_result("UNSOLICITED", is_final=True, from_finalize=True))
    await asyncio.sleep(0)
    empty_events = _events(empty)
    rows.append(
        _case(
            "deepgram",
            "empty_and_duplicate_unsolicited",
            passed=(
                _terminals(empty_events)[0]["outcome"] == "empty"
                and _terminals(empty_events)[0]["turn"] == "A"
                and any(
                    row["reason"] == "deepgram_idle_result"
                    for row in empty_events
                    if row["kind"] == "epoch_end"
                )
                and not any(row["text"] == "UNSOLICITED" for row in _terminals(empty_events))
            ),
            observation="Empty from_finalize is authoritative empty for A. A later ack with no active turn retires the epoch.",
            events=empty_events,
        )
    )
    empty_drain.stop()

    timeout, timeout_drain = _deepgram("dg-timeout", 0.05)
    timeout_request = _request("deepgram", "dg-timeout", 1, "A")
    await timeout.begin_turn(timeout_request)
    await _seal(timeout, timeout_request)
    await asyncio.sleep(0.2)
    timeout_rows = _terminals(_events(timeout))
    rows.append(
        _case(
            "deepgram",
            "failure_timeout",
            passed=(
                len(timeout_rows) == 1
                and timeout_rows[0]["turn"] == "A"
                and timeout_rows[0]["outcome"] == "failed"
                and timeout_rows[0]["reason"] == "deepgram_finalize_ack_missing"
                and timeout_rows[0]["epoch_disposition"] == "retire"
            ),
            observation="Missing from_finalize follows the production CloseStream drain and retires A.",
            events=_events(timeout),
        )
    )
    timeout_drain.stop()

    old, old_drain = _deepgram("dg-old", 30.0)
    old_request = _request("deepgram", "dg-old", 1, "A")
    await old.begin_turn(old_request)
    await old.abort_turn(old_request.identity, reason="config_changed")
    new, new_drain = _deepgram("dg-new", 30.0)
    new_begin = await _attempt(new.begin_turn(_request("deepgram", "dg-new", 2, "B")))
    old._build_transcript_event(_dg_result("LATE-A", is_final=True, from_finalize=True))
    await asyncio.sleep(0)
    same_epoch = await _attempt(old.begin_turn(_request("deepgram", "dg-old", 3, "C")))
    rows.append(
        _case(
            "deepgram",
            "cancel_config_change_and_late_after_reconnect",
            passed=not same_epoch["ok"] and new_begin["ok"] and _events(new) == [],
            observation="Abort/config change retires the old epoch. A late ack there does not enter the replacement session.",
            old_events=_events(old),
            new_events=_events(new),
        )
    )
    old_drain.stop()
    new_drain.stop()

    simultaneous, simultaneous_drain = _deepgram("dg-sim", 30.0)
    await simultaneous.begin_turn(_request("deepgram", "dg-sim", 1, "A"))
    second = await _attempt(simultaneous.begin_turn(_request("deepgram", "dg-sim", 2, "B")))
    rows.append(
        _case(
            "deepgram",
            "simultaneous_pending_work",
            passed=not second["ok"] and "unresolved turn" in str(second.get("message")),
            observation="The production projection rejects a second begin before seal or terminal.",
            second=second,
        )
    )
    simultaneous_drain.stop()
    return rows


class _FakeSonioxSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, payload: object) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        return None


def _soniox(epoch: str) -> _SonioxSession:
    session = _SonioxSession(
        api_key="probe",
        model="stt-rt-v5",
        endpoint="wss://example.invalid/transcribe-websocket",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=60.0,
        trailing_silence_ms=0,
        connect_timeout_s=0.1,
        projection=_projection(epoch),
    )
    session._ws = _FakeSonioxSocket()
    session._send_task = asyncio.create_task(session._send_loop())
    return session


async def _stop_soniox(session: _SonioxSession) -> None:
    if session._send_task is not None and not session._send_task.done():
        await session._audio_q.put(SONIOX_STOP)
        await asyncio.wait_for(session._send_task, timeout=1.0)


def _soniox_message(tokens: list[dict[str, object]]) -> str:
    return json.dumps({"request_id": "sx-session", "tokens": tokens})


async def _probe_soniox() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session = _soniox("sx-1")
    request_a = _request("soniox", "sx-1", 1, "A")
    request_b = _request("soniox", "sx-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    session._handle_message(
        _soniox_message([{"text": "ALPHA", "is_final": True}, {"text": "<end>", "is_final": True}])
    )
    before_fin = _terminals(_events(session))
    session._handle_message(_soniox_message([{"text": "<fin>", "is_final": True}]))
    admitted = await _attempt(session.begin_turn(request_b))
    session._handle_message(_soniox_message([{"text": "LATE-A", "is_final": True}]))
    late = _events(session)
    rows.append(
        _case(
            "soniox",
            "delayed_a_then_late_token_on_b",
            passed=(
                not blocked["ok"]
                and not before_fin
                and admitted["ok"]
                and _terminals(late)[0]["turn"] == "A"
                and _terminals(late)[0]["text"] == "ALPHA"
                and any(
                    row["turn"] == "B" and row["text"] == "LATE-A"
                    for row in late
                    if row["kind"] == "update"
                )
                and not _overlap_flag(session)
            ),
            observation=(
                "<end> is not the barrier. <fin> terminals A. A final token arriving after B begins "
                "is appended to B; the message request_id is session-scoped, not per finalize."
            ),
            b_while_unresolved=blocked,
            events=late,
            overlap_flag=_overlap_flag(session),
        )
    )
    await _stop_soniox(session)

    empty = _soniox("sx-empty")
    empty_request = _request("soniox", "sx-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    empty._handle_message(_soniox_message([{"text": "<fin>", "is_final": True}]))
    empty._handle_message(_soniox_message([{"text": "<fin>", "is_final": True}]))
    empty_events = _events(empty)
    rows.append(
        _case(
            "soniox",
            "empty_and_duplicate_fin",
            passed=(
                _terminals(empty_events)[0]["outcome"] == "empty"
                and _terminals(empty_events)[0]["turn"] == "A"
                and any(
                    row["reason"] == "soniox_protocol_ambiguity"
                    for row in empty_events
                    if row["kind"] == "epoch_end"
                )
            ),
            observation="Empty <fin> is authoritative empty for A. A second <fin> is ambiguity and retires the epoch.",
            events=empty_events,
        )
    )
    await _stop_soniox(empty)

    bad = _soniox("sx-bad")
    bad_request = _request("soniox", "sx-bad", 1, "A")
    await bad.begin_turn(bad_request)
    bad._handle_message(_soniox_message([{"text": "<fin>", "is_final": True}]))
    bad_events = _events(bad)
    rows.append(
        _case(
            "soniox",
            "unsolicited_fin_and_failure",
            passed=any(
                row["turn"] == "A" and row["outcome"] == "failed" for row in _terminals(bad_events)
            )
            and any(
                row["reason"] == "soniox_protocol_ambiguity"
                for row in bad_events
                if row["kind"] == "epoch_end"
            ),
            observation="An unsealed <fin> fails A and retires the epoch. It is not an application final. The adapter has no local final timer.",
            events=bad_events,
        )
    )
    await _stop_soniox(bad)

    cancelled = _soniox("sx-cancel")
    cancel_request = _request("soniox", "sx-cancel", 1, "A")
    await cancelled.begin_turn(cancel_request)
    await cancelled.abort_turn(cancel_request.identity, reason="config_changed")
    replacement = _soniox("sx-new")
    replacement_begin = await _attempt(replacement.begin_turn(_request("soniox", "sx-new", 2, "B")))
    cancelled._handle_message(
        _soniox_message([{"text": "LATE-A", "is_final": True}, {"text": "<fin>", "is_final": True}])
    )
    rows.append(
        _case(
            "soniox",
            "cancel_config_change_and_late_after_reconnect",
            passed=(
                not (await _attempt(cancelled.begin_turn(_request("soniox", "sx-cancel", 3, "C"))))[
                    "ok"
                ]
                and replacement_begin["ok"]
                and _events(replacement) == []
            ),
            observation="Cancellation retires the old epoch. Late tokens there do not enter the replacement session.",
            old_events=_events(cancelled),
            new_events=_events(replacement),
        )
    )
    await _stop_soniox(cancelled)
    await _stop_soniox(replacement)

    simultaneous = _soniox("sx-sim")
    await simultaneous.begin_turn(_request("soniox", "sx-sim", 1, "A"))
    second = await _attempt(simultaneous.begin_turn(_request("soniox", "sx-sim", 2, "B")))
    rows.append(
        _case(
            "soniox",
            "simultaneous_pending_work",
            passed=not second["ok"],
            observation="A second begin is rejected. pending_finalize is a counter, not a turn id.",
            second=second,
        )
    )
    await _stop_soniox(simultaneous)
    return rows


class _FakeLive:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.sent.append(kwargs)

    def receive(self) -> object:
        async def _empty():
            if False:
                yield None

        return _empty()

    async def close(self) -> None:
        return None


def _gemini_message(
    *,
    text: str | None = None,
    interim: str | None = None,
    activity_end: bool = False,
) -> SimpleNamespace:
    content = None
    if text is not None or interim is not None:
        content = SimpleNamespace(
            interim_input_transcription=None if interim is None else SimpleNamespace(text=interim),
            input_transcription=None if text is None else SimpleNamespace(text=text),
        )
    activity = SimpleNamespace(voice_activity_type="ACTIVITY_END") if activity_end else None
    return SimpleNamespace(server_content=content, voice_activity=activity, go_away=None)


async def _gemini(
    epoch: str, timeout: float
) -> tuple[_GeminiTranscribeLiveSession, asyncio.Task[None]]:
    session = _GeminiTranscribeLiveSession(
        api_key="probe",
        language_codes=["en"],
        custom_vocabulary=[],
        model="gemini-3.5-transcribe-live",
        sample_rate_hz=16000,
        connect_timeout_s=0.1,
        finalize_timeout_s=timeout,
        projection=_projection(epoch),
    )
    session._live_session = _FakeLive()
    return session, asyncio.create_task(session._send_loop())


async def _stop_gemini(session: _GeminiTranscribeLiveSession, task: asyncio.Task[None]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    for turn in list(session._pending_turns):
        turn.activity_end_ack.set()
        if turn.timeout_task is not None:
            turn.timeout_task.cancel()


async def _probe_gemini() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session, task = await _gemini("gm-1", 30.0)
    request_a = _request("gemini_transcribe", "gm-1", 1, "A")
    request_b = _request("gemini_transcribe", "gm-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    session._handle_message(_gemini_message(text="ALPHA"))
    before_ack = _terminals(_events(session))
    session._handle_message(_gemini_message(text="SECOND"))
    session._handle_message(_gemini_message(activity_end=True))
    await asyncio.sleep(0)
    completed = _events(session)
    admitted = await _attempt(session.begin_turn(request_b))
    rows.append(
        _case(
            "gemini_transcribe",
            "delayed_a_terminal_b_ready",
            passed=(
                not blocked["ok"]
                and not before_ack
                and admitted["ok"]
                and _terminals(completed)[0]["turn"] == "A"
                and _terminals(completed)[0]["text"] == "ALPHA"
                and not any(row["text"] == "SECOND" for row in _terminals(completed))
                and not _overlap_flag(session)
            ),
            observation=(
                "Authoritative text alone does not release A. ActivityEnd plus that text does. "
                "A second authoritative message is not applied to another turn. ActivityStart/End have no turn id."
            ),
            b_while_unresolved=blocked,
            events=completed,
            overlap_flag=_overlap_flag(session),
        )
    )
    await _stop_gemini(session, task)

    empty, empty_task = await _gemini("gm-empty", 30.0)
    empty_request = _request("gemini_transcribe", "gm-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    empty._handle_message(_gemini_message(text=""))
    empty._handle_message(_gemini_message(activity_end=True))
    await asyncio.sleep(0)
    empty._handle_message(_gemini_message(activity_end=True))
    empty_events = _events(empty)
    rows.append(
        _case(
            "gemini_transcribe",
            "empty_and_unsolicited_activity_end",
            passed=(
                _terminals(empty_events)[0]["outcome"] == "empty"
                and _terminals(empty_events)[0]["turn"] == "A"
                and any(
                    row["reason"] == "gemini_unsolicited_activity_end"
                    for row in empty_events
                    if row["kind"] == "epoch_end"
                )
            ),
            observation="Empty authoritative text is an empty terminal for A. A later ActivityEnd retires the idle epoch.",
            events=empty_events,
        )
    )
    await _stop_gemini(empty, empty_task)

    timeout, timeout_task = await _gemini("gm-timeout", 0.05)
    timeout_request = _request("gemini_transcribe", "gm-timeout", 1, "A")
    await timeout.begin_turn(timeout_request)
    await _send(timeout, timeout_request)
    timeout._handle_message(_gemini_message(interim="PARTIAL"))
    await _seal(timeout, timeout_request)
    await asyncio.sleep(0.15)
    timeout_rows = _terminals(_events(timeout))
    rows.append(
        _case(
            "gemini_transcribe",
            "failure_timeout",
            passed=(
                len(timeout_rows) == 1
                and timeout_rows[0]["turn"] == "A"
                and timeout_rows[0]["outcome"] == "degraded"
                and timeout_rows[0]["text"] == "PARTIAL"
                and timeout_rows[0]["epoch_disposition"] == "retire"
            ),
            observation="Finalize timeout keeps interim text only as degraded A and retires the epoch.",
            events=_events(timeout),
        )
    )
    await _stop_gemini(timeout, timeout_task)

    cancelled, cancelled_task = await _gemini("gm-cancel", 30.0)
    cancel_request = _request("gemini_transcribe", "gm-cancel", 1, "A")
    await cancelled.begin_turn(cancel_request)
    await cancelled.abort_turn(cancel_request.identity, reason="config_changed")
    replacement, replacement_task = await _gemini("gm-new", 30.0)
    replacement_begin = await _attempt(
        replacement.begin_turn(_request("gemini_transcribe", "gm-new", 2, "B"))
    )
    cancelled._handle_message(_gemini_message(text="LATE-A", activity_end=True))
    rows.append(
        _case(
            "gemini_transcribe",
            "cancel_config_change_and_late_after_reconnect",
            passed=(
                not (
                    await _attempt(
                        cancelled.begin_turn(_request("gemini_transcribe", "gm-cancel", 3, "C"))
                    )
                )["ok"]
                and replacement_begin["ok"]
                and _events(replacement) == []
                and not any(row["text"] == "LATE-A" for row in _terminals(_events(cancelled)))
            ),
            observation="Abort retires A. A late final on that failed session is not published on the replacement epoch.",
            old_events=_events(cancelled),
            new_events=_events(replacement),
        )
    )
    await _stop_gemini(cancelled, cancelled_task)
    await _stop_gemini(replacement, replacement_task)

    simultaneous, simultaneous_task = await _gemini("gm-sim", 30.0)
    await simultaneous.begin_turn(_request("gemini_transcribe", "gm-sim", 1, "A"))
    second = await _attempt(
        simultaneous.begin_turn(_request("gemini_transcribe", "gm-sim", 2, "B"))
    )
    rows.append(
        _case(
            "gemini_transcribe",
            "simultaneous_pending_work",
            passed=not second["ok"] and "one unresolved" in str(second.get("message")),
            observation="Production begin rejects a second scoped turn. The handler would otherwise take the first pending turn.",
            second=second,
        )
    )
    await _stop_gemini(simultaneous, simultaneous_task)
    return rows


class _FakeScribeConnection:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.commits = 0

    async def send(self, payload: object) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def on(self, *_args: object) -> None:
        return None


def _scribe(epoch: str, *, fail_commit: bool = False) -> _ElevenLabsScribeSession:
    session = _ElevenLabsScribeSession(
        api_key="probe",
        language_code="en",
        keyterms=(),
        model="scribe_v2_realtime",
        sample_rate_hz=16000,
        connect_timeout_s=0.1,
        keepalive_interval_s=60.0,
        projection=_projection(epoch),
    )
    session._connection = _FakeScribeConnection(fail_commit=fail_commit)
    session._last_send_at = 0.0
    return session


async def _probe_scribe() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session = _scribe("sc-1")
    request_a = _request("elevenlabs_scribe", "sc-1", 1, "A")
    request_b = _request("elevenlabs_scribe", "sc-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    session._on_partial({"message_type": "partial_transcript", "text": "partial-A"})
    session._on_partial(
        {"message_type": "committed_transcript_with_timestamps", "text": "stamped-A"}
    )
    before = _terminals(_events(session))
    session._on_committed({"message_type": "committed_transcript", "text": "ALPHA"})
    admitted = await _attempt(session.begin_turn(request_b))
    await _seal(session, request_b)
    session._on_committed({"message_type": "committed_transcript", "text": "LATE-A"})
    events = _events(session)
    rows.append(
        _case(
            "elevenlabs_scribe",
            "delayed_a_then_late_commit_on_b",
            passed=(
                not blocked["ok"]
                and not before
                and admitted["ok"]
                and any(row["turn"] == "A" and row["text"] == "ALPHA" for row in _terminals(events))
                and any(
                    row["turn"] == "B" and row["text"] == "LATE-A" for row in _terminals(events)
                )
                and not _overlap_flag(session)
            ),
            observation=(
                "Partials and timestamp transcripts do not resolve A. committed_transcript does. "
                "A later committed event has no commit id, so after B is sealed it is attributed to B."
            ),
            b_while_unresolved=blocked,
            events=events,
            overlap_flag=_overlap_flag(session),
        )
    )

    empty = _scribe("sc-empty")
    empty_request = _request("elevenlabs_scribe", "sc-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    empty._on_committed({"message_type": "committed_transcript", "text": ""})
    empty._on_committed({"message_type": "committed_transcript", "text": "UNSOLICITED"})
    empty_events = _events(empty)
    rows.append(
        _case(
            "elevenlabs_scribe",
            "empty_and_unsolicited_committed",
            passed=(
                _terminals(empty_events)[0]["outcome"] == "empty"
                and any(
                    row["reason"] == "scribe_unsolicited_committed_transcript"
                    for row in empty_events
                    if row["kind"] == "epoch_end"
                )
            ),
            observation="Empty committed text is authoritative empty. A committed event with no active turn retires the epoch.",
            events=empty_events,
        )
    )

    duplicate = _scribe("sc-dup")
    duplicate_request = _request("elevenlabs_scribe", "sc-dup", 1, "A")
    await duplicate.begin_turn(duplicate_request)

    async def _double_commit() -> None:
        duplicate._on_committed({"text": "FIRST"})
        duplicate._on_committed({"text": "SECOND"})

    duplicate._connection.commit = _double_commit
    await _seal(duplicate, duplicate_request)
    duplicate_events = _events(duplicate)
    rows.append(
        _case(
            "elevenlabs_scribe",
            "duplicate_committed_during_commit_write",
            passed=(
                any(
                    row["reason"] == "scribe_duplicate_committed_transcript"
                    for row in duplicate_events
                    if row["kind"] == "epoch_end"
                )
                and not any(row["text"] == "SECOND" for row in _terminals(duplicate_events))
            ),
            observation="Two committed events while the commit write is in flight fail the epoch instead of opening another turn.",
            events=duplicate_events,
        )
    )

    failed = _scribe("sc-fail", fail_commit=True)
    failed_request = _request("elevenlabs_scribe", "sc-fail", 1, "A")
    await failed.begin_turn(failed_request)
    commit_result = await _attempt(_seal(failed, failed_request))
    rows.append(
        _case(
            "elevenlabs_scribe",
            "failure_timeout",
            passed=(
                not commit_result["ok"]
                and any(
                    row["turn"] == "A" and row["reason"] == "scribe_commit_failed"
                    for row in _terminals(_events(failed))
                )
            ),
            observation="Commit failure terminalizes A and retires the epoch. There is no separate final-wait timer in this adapter.",
            events=_events(failed),
        )
    )

    cancelled = _scribe("sc-cancel")
    cancel_request = _request("elevenlabs_scribe", "sc-cancel", 1, "A")
    await cancelled.begin_turn(cancel_request)
    await cancelled.abort_turn(cancel_request.identity, reason="config_changed")
    replacement = _scribe("sc-new")
    replacement_begin = await _attempt(
        replacement.begin_turn(_request("elevenlabs_scribe", "sc-new", 2, "B"))
    )
    cancelled._on_committed({"text": "LATE-A"})
    rows.append(
        _case(
            "elevenlabs_scribe",
            "cancel_config_change_and_late_after_reconnect",
            passed=(
                not (
                    await _attempt(
                        cancelled.begin_turn(_request("elevenlabs_scribe", "sc-cancel", 3, "C"))
                    )
                )["ok"]
                and replacement_begin["ok"]
                and _events(replacement) == []
                and not any(row["text"] == "LATE-A" for row in _terminals(_events(cancelled)))
            ),
            observation="Cancellation retires the old epoch. A late committed event there does not enter the replacement session.",
            old_events=_events(cancelled),
            new_events=_events(replacement),
        )
    )

    simultaneous = _scribe("sc-sim")
    await simultaneous.begin_turn(_request("elevenlabs_scribe", "sc-sim", 1, "A"))
    second = await _attempt(
        simultaneous.begin_turn(_request("elevenlabs_scribe", "sc-sim", 2, "B"))
    )
    rows.append(
        _case(
            "elevenlabs_scribe",
            "simultaneous_pending_work",
            passed=not second["ok"],
            observation="Only one scoped turn can be open. The adapter reads committed text and ignores any other event fields.",
            second=second,
        )
    )
    return rows


class _FakeQwenSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, payload: object) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        return None


def _qwen(epoch: str, *, finish_timeout: float = 30.0) -> _QwenAudioSession:
    session = _QwenAudioSession(
        api_key="probe",
        language_hints=("en",),
        model="qwen-audio-3.0-asr-flash-streaming",
        endpoint="wss://example.invalid/api-ws/v1/inference",
        sample_rate_hz=16000,
        connect_timeout_s=0.1,
        task_start_timeout_s=30.0,
        task_finish_timeout_s=finish_timeout,
        send_timeout_s=1.0,
        keepalive_interval_s=60.0,
        keepalive_silence_ms=200,
        projection=_projection(epoch),
    )
    session._ws = _FakeQwenSocket()
    session._loop = asyncio.get_running_loop()
    session._state = QwenAudioSessionState.TASK_ACTIVE
    session._task_id = "task-a"
    session._accept_terminals = True
    return session


def _qwen_sentence(task_id: str, sentence_id: str, text: str) -> dict[str, object]:
    return {
        "header": {"event": "result-generated", "task_id": task_id},
        "payload": {
            "output": {
                "sentence": {
                    "sentence_id": sentence_id,
                    "sentence_end": True,
                    "heartbeat": False,
                    "text": text,
                }
            }
        },
    }


async def _probe_qwen() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session = _qwen("qw-1")
    request_a = _request("qwen_audio", "qw-1", 1, "A")
    request_b = _request("qwen_audio", "qw-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    await session.inject_event(_qwen_sentence("task-a", "1", "ALPHA"))
    await session.inject_event(_qwen_sentence("task-a", "1", "DUP"))
    before = _terminals(_events(session))
    await session.inject_event({"header": {"event": "task-finished", "task_id": "task-a"}})
    await asyncio.sleep(0)
    finished = _events(session)
    next_task = session._task_id
    await session.inject_event({"header": {"event": "task-started", "task_id": next_task}})
    admitted = await _attempt(session.begin_turn(request_b))
    await session.inject_event({"header": {"event": "task-finished", "task_id": "task-a"}})
    await session.inject_event(_qwen_sentence("task-a", "9", "LATE-A"))
    events = _events(session)
    rows.append(
        _case(
            "qwen_audio",
            "delayed_a_terminal_b_ready",
            passed=(
                not blocked["ok"]
                and not before
                and admitted["ok"]
                and _terminals(finished)[0]["turn"] == "A"
                and _terminals(finished)[0]["text"] == "ALPHA"
                and not any(row["text"] == "LATE-A" for row in events if row["kind"] == "update")
                and not _overlap_flag(session)
            ),
            observation=(
                "sentence_end is not the turn barrier. task-finished for task-a terminals A. "
                "Stale task-a results after the next task starts are ignored."
            ),
            b_while_unresolved=blocked,
            events=events,
            overlap_flag=_overlap_flag(session),
        )
    )
    if session._start_timeout_task is not None:
        session._start_timeout_task.cancel()

    empty = _qwen("qw-empty")
    empty_request = _request("qwen_audio", "qw-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    await empty.inject_event({"header": {"event": "task-finished", "task_id": "task-a"}})
    await asyncio.sleep(0)
    empty_rows = _terminals(_events(empty))
    rows.append(
        _case(
            "qwen_audio",
            "empty_a",
            passed=len(empty_rows) == 1
            and empty_rows[0]["outcome"] == "empty"
            and empty_rows[0]["turn"] == "A",
            observation="task-finished with no sentence text is authoritative empty for A.",
            events=_events(empty),
        )
    )
    if empty._start_timeout_task is not None:
        empty._start_timeout_task.cancel()

    failed = _qwen("qw-fail", finish_timeout=0.05)
    failed_request = _request("qwen_audio", "qw-fail", 1, "A")
    await failed.begin_turn(failed_request)
    await _seal(failed, failed_request)
    await asyncio.sleep(0.15)
    failed_rows = _terminals(_events(failed))
    rows.append(
        _case(
            "qwen_audio",
            "failure_timeout",
            passed=(
                len(failed_rows) == 1
                and failed_rows[0]["turn"] == "A"
                and failed_rows[0]["outcome"] == "failed"
                and failed_rows[0]["epoch_disposition"] == "retire"
            ),
            observation="Missing task-finished fails A and retires the epoch.",
            events=_events(failed),
        )
    )

    cancelled = _qwen("qw-cancel")
    cancel_request = _request("qwen_audio", "qw-cancel", 1, "A")
    await cancelled.begin_turn(cancel_request)
    await cancelled.abort_turn(cancel_request.identity, reason="config_changed")
    replacement = _qwen("qw-new")
    replacement._task_id = "task-b"
    replacement_begin = await _attempt(
        replacement.begin_turn(_request("qwen_audio", "qw-new", 2, "B"))
    )
    await cancelled.inject_event({"header": {"event": "task-finished", "task_id": "task-a"}})
    rows.append(
        _case(
            "qwen_audio",
            "cancel_config_change_and_late_after_reconnect",
            passed=(
                not (
                    await _attempt(
                        cancelled.begin_turn(_request("qwen_audio", "qw-cancel", 3, "C"))
                    )
                )["ok"]
                and replacement_begin["ok"]
                and any(
                    row["turn"] == "A" and row["outcome"] == "cancelled"
                    for row in _terminals(_events(cancelled))
                )
            ),
            observation="Abort retires A. A replacement session with its own task id does not consume the old task-finished.",
            old_events=_events(cancelled),
            new_events=_events(replacement),
        )
    )

    simultaneous = _qwen("qw-sim")
    await simultaneous.begin_turn(_request("qwen_audio", "qw-sim", 1, "A"))
    second = await _attempt(simultaneous.begin_turn(_request("qwen_audio", "qw-sim", 2, "B")))
    rows.append(
        _case(
            "qwen_audio",
            "simultaneous_pending_work",
            passed=not second["ok"],
            observation="One client task id is bound to the active turn. A second begin is rejected.",
            second=second,
        )
    )
    return rows


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.release: asyncio.Event | None = None
        self.calls = 0

    async def post(self, _url: str, **_kwargs: object) -> _FakeHttpResponse:
        self.calls += 1
        if self.release is not None:
            await self.release.wait()
        return _FakeHttpResponse(self.payload, self.status_code)

    async def aclose(self) -> None:
        return None


def _offline(epoch: str, client: _FakeHttpClient) -> _OfflineOpenAITranscriptionSession:
    session = _OfflineOpenAITranscriptionSession(
        endpoint="https://example.invalid/v1/audio/transcriptions",
        model="whisper-1",
        api_key="probe",
        source_language="en",
        sample_rate_hz=16000,
        http_client_factory=lambda **_kwargs: client,
        projection=_projection(epoch),
    )
    session._client = client
    return session


async def _probe_custom_offline() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    client = _FakeHttpClient({"text": "ALPHA"})
    client.release = asyncio.Event()
    session = _offline("off-1", client)
    request_a = _request("custom_offline", "off-1", 1, "A")
    request_b = _request("custom_offline", "off-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    second_seal = await _attempt(_seal(session, request_a))
    assert client.release is not None
    client.release.set()
    assert session._scoped_task is not None
    await session._scoped_task
    completed = _events(session)
    admitted = await _attempt(session.begin_turn(request_b))
    rows.append(
        _case(
            "custom_offline",
            "delayed_a_terminal_b_ready",
            passed=(
                not blocked["ok"]
                and not second_seal["ok"]
                and admitted["ok"]
                and _terminals(completed)[0]["turn"] == "A"
                and _terminals(completed)[0]["text"] == "ALPHA"
                and not _overlap_flag(session)
            ),
            observation=(
                "The HTTP body is snapshotted into A's task, but begin and a second seal are rejected "
                "while that task is unresolved."
            ),
            b_while_unresolved=blocked,
            second_seal=second_seal,
            events=completed,
            overlap_flag=_overlap_flag(session),
        )
    )

    empty = _offline("off-empty", _FakeHttpClient({"text": "   "}))
    empty_request = _request("custom_offline", "off-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    assert empty._scoped_task is not None
    await empty._scoped_task
    empty_rows = _terminals(_events(empty))
    rows.append(
        _case(
            "custom_offline",
            "empty_a",
            passed=len(empty_rows) == 1
            and empty_rows[0]["outcome"] == "empty"
            and empty_rows[0]["turn"] == "A",
            observation="Whitespace-only HTTP text is authoritative empty for the requesting identity.",
            events=_events(empty),
        )
    )

    failed = _offline("off-fail", _FakeHttpClient({"error": "no"}, status_code=500))
    failed_request = _request("custom_offline", "off-fail", 1, "A")
    await failed.begin_turn(failed_request)
    await _seal(failed, failed_request)
    assert failed._scoped_task is not None
    await failed._scoped_task
    failed_rows = _terminals(_events(failed))
    rows.append(
        _case(
            "custom_offline",
            "failure_http",
            passed=(
                len(failed_rows) == 1
                and failed_rows[0]["turn"] == "A"
                and failed_rows[0]["outcome"] == "failed"
                and failed_rows[0]["text"] == ""
                and failed_rows[0]["epoch_disposition"] == "retire"
            ),
            observation="HTTP failure is a failed terminal for A, not empty text. The 50s total timeout was not waited out.",
            events=_events(failed),
        )
    )

    held = _FakeHttpClient({"text": "LATE-A"})
    held.release = asyncio.Event()
    cancelled = _offline("off-cancel", held)
    cancel_request = _request("custom_offline", "off-cancel", 1, "A")
    await cancelled.begin_turn(cancel_request)
    await _seal(cancelled, cancel_request)
    await cancelled.abort_turn(cancel_request.identity, reason="config_changed")
    replacement = _offline("off-new", _FakeHttpClient({"text": "BETA"}))
    replacement_request = _request("custom_offline", "off-new", 2, "B")
    await replacement.begin_turn(replacement_request)
    await _seal(replacement, replacement_request)
    assert replacement._scoped_task is not None
    await replacement._scoped_task
    rows.append(
        _case(
            "custom_offline",
            "cancel_config_change_and_late_after_reconnect",
            passed=(
                any(
                    row["turn"] == "A" and row["outcome"] == "cancelled"
                    for row in _terminals(_events(cancelled))
                )
                and _terminals(_events(replacement))[0]["text"] == "BETA"
                and not (
                    await _attempt(
                        cancelled.begin_turn(_request("custom_offline", "off-cancel", 3, "C"))
                    )
                )["ok"]
            ),
            observation="Abort cancels A's task and retires the epoch. A new session's response stays on B.",
            old_events=_events(cancelled),
            new_events=_events(replacement),
        )
    )

    simultaneous = _offline("off-sim", _FakeHttpClient({"text": "X"}))
    await simultaneous.begin_turn(_request("custom_offline", "off-sim", 1, "A"))
    second = await _attempt(simultaneous.begin_turn(_request("custom_offline", "off-sim", 2, "B")))
    rows.append(
        _case(
            "custom_offline",
            "simultaneous_pending_work",
            passed=not second["ok"],
            observation="Each HTTP response belongs to its call, but the current session admits only one turn and one scoped task.",
            second=second,
        )
    )
    return rows


class _FakeRealtimeSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        await self._queue.put(None)

    def push(self, event: dict[str, object]) -> None:
        self._queue.put_nowait(json.dumps(event))

    def __aiter__(self) -> _FakeRealtimeSocket:
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


def _realtime(epoch: str) -> tuple[_StreamingOpenAIRealtimeSession, _FakeRealtimeSocket]:
    socket = _FakeRealtimeSocket()
    session = _StreamingOpenAIRealtimeSession(
        endpoint="https://example.invalid/v1/realtime",
        model="gpt-4o-mini-transcribe",
        api_key="probe",
        source_language="en",
        sample_rate_hz=16000,
        extra={"turn_detection": None},
        projection=_projection(epoch),
    )
    session._ws = socket
    session._recv_task = asyncio.create_task(session._receive_loop())
    return session, socket


async def _stop_realtime(session: _StreamingOpenAIRealtimeSession) -> None:
    if session._recv_task is not None and not session._recv_task.done():
        session._recv_task.cancel()
        await asyncio.gather(session._recv_task, return_exceptions=True)
    if session._scoped_final_timeout_task is not None:
        session._scoped_final_timeout_task.cancel()


async def _probe_custom_realtime() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    session, socket = _realtime("rt-1")
    request_a = _request("custom_realtime", "rt-1", 1, "A")
    request_b = _request("custom_realtime", "rt-1", 2, "B")
    await session.begin_turn(request_a)
    await _send(session, request_a)
    await _seal(session, request_a)
    blocked = await _attempt(session.begin_turn(request_b))
    socket.push(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-a",
            "transcript": "TOO-EARLY",
        }
    )
    await asyncio.sleep(0.05)
    early = _events(session)
    admitted = await _attempt(session.begin_turn(request_b))
    rows.append(
        _case(
            "custom_realtime",
            "delayed_a_completion_before_commit",
            passed=not blocked["ok"]
            and any(row["kind"] == "epoch_end" for row in early)
            and not admitted["ok"],
            observation=(
                "B is rejected while the commit barrier is unresolved. A completed event before the committed "
                "item id fails correlation and retires the epoch."
            ),
            b_while_unresolved=blocked,
            b_after_early_completion=admitted,
            events=early,
            overlap_flag=_overlap_flag(session),
        )
    )
    await _stop_realtime(session)

    keyed, keyed_socket = _realtime("rt-keyed")
    keyed_a = _request("custom_realtime", "rt-keyed", 1, "A")
    keyed_b = _request("custom_realtime", "rt-keyed", 2, "B")
    await keyed.begin_turn(keyed_a)
    await _seal(keyed, keyed_a)
    keyed_socket.push({"type": "input_audio_buffer.committed", "item_id": "item-a"})
    keyed_socket.push(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-a",
            "transcript": "ALPHA",
        }
    )
    await asyncio.sleep(0.05)
    keyed_admitted = await _attempt(keyed.begin_turn(keyed_b))
    await _seal(keyed, keyed_b)
    keyed_socket.push({"type": "input_audio_buffer.committed", "item_id": "item-b"})
    keyed_socket.push(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-a",
            "transcript": "LATE-A",
        }
    )
    await asyncio.sleep(0.05)
    keyed_events = _events(keyed)
    rows.append(
        _case(
            "custom_realtime",
            "late_item_after_next_commit",
            passed=(
                keyed_admitted["ok"]
                and any(
                    row["turn"] == "A" and row["text"] == "ALPHA"
                    for row in _terminals(keyed_events)
                )
                and not any(row["text"] == "LATE-A" for row in _terminals(keyed_events))
            ),
            observation=(
                "A late item-a completion does not become B's text. This is not overlap approval: B cannot begin "
                "before A's barrier, and audio appended before committed is not client-keyed."
            ),
            events=keyed_events,
        )
    )
    await _stop_realtime(keyed)

    empty, empty_socket = _realtime("rt-empty")
    empty_request = _request("custom_realtime", "rt-empty", 1, "A")
    await empty.begin_turn(empty_request)
    await _seal(empty, empty_request)
    empty_socket.push({"type": "input_audio_buffer.committed", "item_id": "item-empty"})
    empty_socket.push(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-empty",
            "transcript": "",
        }
    )
    await asyncio.sleep(0.05)
    empty_socket.push(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "UNKEYED"}
    )
    await asyncio.sleep(0.05)
    empty_events = _events(empty)
    rows.append(
        _case(
            "custom_realtime",
            "empty_and_unkeyed_duplicate",
            passed=(
                _terminals(empty_events)[0]["outcome"] == "empty"
                and any(
                    row["reason"] == "ambiguous_unkeyed_terminal"
                    for row in empty_events
                    if row["kind"] == "epoch_end"
                )
            ),
            observation="Empty keyed completion is authoritative empty. A later unkeyed completion retires the epoch.",
            events=empty_events,
        )
    )
    await _stop_realtime(empty)

    import puripuly_heart.providers.stt.custom as custom_module

    previous_timeout = custom_module._STREAM_FINAL_TIMEOUT_S
    custom_module._STREAM_FINAL_TIMEOUT_S = 0.05
    try:
        timeout, _timeout_socket = _realtime("rt-timeout")
        timeout_request = _request("custom_realtime", "rt-timeout", 1, "A")
        await timeout.begin_turn(timeout_request)
        await _seal(timeout, timeout_request)
        await asyncio.sleep(0.15)
        timeout_rows = _terminals(_events(timeout))
        rows.append(
            _case(
                "custom_realtime",
                "failure_timeout",
                passed=(
                    len(timeout_rows) == 1
                    and timeout_rows[0]["turn"] == "A"
                    and timeout_rows[0]["outcome"] == "failed"
                    and timeout_rows[0]["reason"] == "final_timeout"
                    and timeout_rows[0]["epoch_disposition"] == "retire"
                ),
                observation="The production final-wait path fails A and retires the epoch. Only the in-process constant was shortened.",
                events=_events(timeout),
            )
        )
        await _stop_realtime(timeout)
    finally:
        custom_module._STREAM_FINAL_TIMEOUT_S = previous_timeout

    cancelled, _cancelled_socket = _realtime("rt-cancel")
    cancel_request = _request("custom_realtime", "rt-cancel", 1, "A")
    await cancelled.begin_turn(cancel_request)
    await cancelled.abort_turn(cancel_request.identity, reason="config_changed")
    replacement, _replacement_socket = _realtime("rt-new")
    replacement_begin = await _attempt(
        replacement.begin_turn(_request("custom_realtime", "rt-new", 2, "B"))
    )
    rows.append(
        _case(
            "custom_realtime",
            "cancel_config_change_and_late_after_reconnect",
            passed=(
                not (
                    await _attempt(
                        cancelled.begin_turn(_request("custom_realtime", "rt-cancel", 3, "C"))
                    )
                )["ok"]
                and replacement_begin["ok"]
                and _events(replacement) == []
            ),
            observation="Abort closes and retires the old epoch. The replacement session has its own item scope.",
            old_events=_events(cancelled),
            new_events=_events(replacement),
        )
    )
    await _stop_realtime(cancelled)
    await _stop_realtime(replacement)

    simultaneous, _simultaneous_socket = _realtime("rt-sim")
    await simultaneous.begin_turn(_request("custom_realtime", "rt-sim", 1, "A"))
    second = await _attempt(simultaneous.begin_turn(_request("custom_realtime", "rt-sim", 2, "B")))
    rows.append(
        _case(
            "custom_realtime",
            "simultaneous_pending_work",
            passed=not second["ok"],
            observation="The session stores one scoped item id and one pending commit.",
            second=second,
        )
    )
    await _stop_realtime(simultaneous)
    return rows


class _RejectingInner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._open = False

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        self.calls.append(request.identity.provider_turn_id)
        if self._open:
            raise RuntimeError("STT session already has an unresolved turn")
        self._open = True

    async def send_turn_audio(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def seal_turn(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def abort_turn(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def turn_events(self):
        if False:
            yield None

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send_audio(self, _pcm: bytes) -> None:
        return None

    async def on_speech_end(self, **_kwargs: object) -> None:
        return None

    async def events(self):
        if False:
            yield None


async def _probe_rolling() -> list[dict[str, object]]:
    inner = _RejectingInner()
    assert isinstance(inner, STTScopedTurnSession)
    session = _RollingSession(
        definition=RollingProviderDefinition(
            name=STTProviderName.DEEPGRAM,
            build_backend=lambda: None,
            is_configured=lambda: True,
        ),
        inner=inner,
        on_session_error=lambda _definition, _exc: None,
    )
    first = await _attempt(session.begin_turn(_request("rolling_free", "roll-1", 1, "A")))
    second = await _attempt(session.begin_turn(_request("rolling_free", "roll-1", 2, "B")))
    return [
        _case(
            "rolling_free",
            "composition_has_no_independent_correlation",
            passed=first["ok"]
            and not second["ok"]
            and not hasattr(session, "allows_sealed_turn_overlap"),
            observation="Rolling forwards begin to the selected member and adds no turn id or overlap flag.",
            first=first,
            second=second,
            forwarded=inner.calls,
        )
    ]


async def _main() -> dict[str, object]:
    probes = (
        _probe_deepgram,
        _probe_soniox,
        _probe_gemini,
        _probe_scribe,
        _probe_qwen,
        _probe_custom_offline,
        _probe_custom_realtime,
        _probe_rolling,
    )
    cases: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for probe in probes:
        try:
            cases.extend(await probe())
        except Exception as exc:
            errors.append({"probe": probe.__name__, "error": f"{type(exc).__name__}: {exc}"})
    failed = [row for row in cases if row["status"] != "passed"]
    return {
        "revision": _revision(),
        "baseline": "78d90ca9722d3c88e05448bbe7c958892c5b11ec",
        "versions": _versions(),
        "live_provider_calls": "not run",
        "product_changes": "none",
        "case_count": len(cases),
        "failed_count": len(failed),
        "probe_errors": errors,
        "cases": cases,
    }


def main() -> None:
    result = asyncio.run(_main())
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(OUT),
                "case_count": result["case_count"],
                "failed_count": result["failed_count"],
                "probe_errors": result["probe_errors"],
            },
            indent=2,
        )
    )
    if result["failed_count"] or result["probe_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
