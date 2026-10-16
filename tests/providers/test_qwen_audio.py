from __future__ import annotations

import asyncio
import gc
import json
import logging
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.adapters.gpu_worker_process import DefaultGpuWorkerProcessFactory
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    create_stt_backend_from_resolved_config,
)
from puripuly_heart.config.runtime_resolution import STTRuntimeIntent, resolve_stt_config
from puripuly_heart.config.settings_vnext.serialization import from_dict
from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.stt.backend import (
    LEGACY_STT_SESSION_PROJECTION,
    STTContributionConsumptionLedger,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnitTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.qwen_audio import (
    QWEN_AUDIO_MODEL,
    QwenAudioProtocolError,
    QwenAudioSessionState,
    QwenAudioStreamingSTTBackend,
    QwenAudioTaskFailedError,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.incoming: asyncio.Queue[object] = asyncio.Queue()
        self.closed = False
        self.block_audio = False
        self.audio_started = asyncio.Event()
        self.audio_gate = asyncio.Event()
        self.fail_audio = False
        self.fail_text = False

    async def send(self, value: str | bytes) -> None:
        if isinstance(value, bytes):
            if self.block_audio:
                self.block_audio = False
                self.audio_started.set()
                await self.audio_gate.wait()
            if self.fail_audio:
                self.fail_audio = False
                raise RuntimeError("audio send failed")
        elif self.fail_text:
            self.fail_text = False
            raise RuntimeError("text send failed")
        self.sent.append(value)

    async def recv(self) -> object:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put(None)

    async def push(self, value: object) -> None:
        await self.incoming.put(value)


async def open_fake(
    *,
    hotwords: object = (),
    task_start_timeout_s: float = 1,
    task_finish_timeout_s: float = 0.2,
    send_timeout_s: float = 5,
    keepalive_interval_s: float = 15.0,
    keepalive_silence_ms: int = 100,
    language_hints: tuple[str, ...] = ("ko",),
    scoped: bool = False,
) -> tuple[QwenAudioStreamingSTTBackend, object, FakeWebSocket, str]:
    socket = FakeWebSocket()

    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        return socket

    backend = QwenAudioStreamingSTTBackend(
        api_key="test-key",
        language_hints=language_hints,
        hotwords=hotwords,
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=task_start_timeout_s,
        task_finish_timeout_s=task_finish_timeout_s,
        send_timeout_s=send_timeout_s,
        keepalive_interval_s=keepalive_interval_s,
        keepalive_silence_ms=keepalive_silence_ms,
    )
    opening = asyncio.create_task(
        backend.open_session(
            projection=(
                STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
                if scoped
                else LEGACY_STT_SESSION_PROJECTION
            )
        )
    )
    while not socket.sent:
        await asyncio.sleep(0)
    first_id = json.loads(socket.sent[0])["header"]["task_id"]
    await socket.push({"header": {"event": "task-started", "task_id": first_id}})
    session = await opening
    return backend, session, socket, first_id


@pytest.mark.asyncio
async def test_qwen_audio_backend_requires_api_key() -> None:
    backend = QwenAudioStreamingSTTBackend(api_key="")

    with pytest.raises(ValueError, match="api_key"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_qwen_audio_backend_requires_supported_sample_rate() -> None:
    backend = QwenAudioStreamingSTTBackend(api_key="key", sample_rate_hz=44100)

    with pytest.raises(ValueError, match="sample_rate_hz"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_qwen_audio_backend_rejects_empty_verification_key() -> None:
    assert await QwenAudioStreamingSTTBackend.verify_api_key("") is False


async def next_event(session: object):
    return await session.events().__anext__()


def scoped_request(
    provider_id: str,
    *,
    task: str = "turn-1",
    channel: str = "peer",
) -> STTProviderTurnRequest:
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1,
            segment_order=1,
            segment_id=uuid4(),
            capture_epoch=1,
        ),
        provider_epoch_id="epoch-1",
        provider_turn_id=task,
    )
    return STTProviderTurnRequest(
        identity=identity,
        settings=AudioSegmentSettingsSnapshot(
            provider_id=provider_id,
            provider_signature=(provider_id,),
            runtime_signature=(provider_id,),
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
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_scoped_task_uses_native_task_barrier_and_stable_sentence_updates(
    channel: str,
) -> None:
    _, session, socket, task_id = await open_fake(scoped=True)
    request = scoped_request("qwen_audio", channel=channel)

    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity,
        b"pcm",
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
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {
                "output": {
                    "sentence": {
                        "sentence_id": "sentence-1",
                        "sentence_end": True,
                        "text": "same same",
                    }
                }
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})

    event_stream = session.turn_events()
    update = await asyncio.wait_for(event_stream.__anext__(), timeout=1)
    terminal = await asyncio.wait_for(event_stream.__anext__(), timeout=1)
    assert isinstance(update, STTProviderTurnUpdate)
    assert update.text == "same same"
    assert update.provenance.native_task_id == task_id
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.outcome == "final"
    assert terminal.text == "same same"
    assert terminal.provenance[0].native_task_id == task_id
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_scoped_audio_write_failure_terminalizes_and_retires_epoch() -> None:
    _, session, socket, _ = await open_fake(scoped=True)
    request = scoped_request("qwen_audio")
    await session.begin_turn(request)
    socket.fail_audio = True
    await session.send_turn_audio(
        request.identity,
        b"pcm",
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.identity == request.identity
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"
    assert socket.closed


@pytest.mark.asyncio
async def test_scoped_empty_duplicate_late_and_next_native_task_identity() -> None:
    _, session, socket, first_task_id = await open_fake(scoped=True)
    first = scoped_request("qwen_audio", task="turn-1")
    await session.begin_turn(first)
    await session.seal_turn(
        first.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    await socket.push({"header": {"event": "task-finished", "task_id": first_task_id}})
    first_terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert first_terminal.outcome == "empty"
    await socket.push({"header": {"event": "task-finished", "task_id": first_task_id}})
    while True:
        run_messages = [
            json.loads(value)
            for value in socket.sent
            if isinstance(value, str) and json.loads(value)["header"]["action"] == "run-task"
        ]
        if len(run_messages) >= 2:
            break
        await asyncio.sleep(0)
    second_task_id = run_messages[-1]["header"]["task_id"]
    await socket.push({"header": {"event": "task-started", "task_id": second_task_id}})
    while session.state is not QwenAudioSessionState.TASK_ACTIVE:
        await asyncio.sleep(0)
    second = scoped_request("qwen_audio", task="turn-2")
    await session.begin_turn(second)
    await session.seal_turn(
        second.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    with pytest.raises(RuntimeError, match="already sealed"):
        await session.seal_turn(
            second.identity,
            sealed_content_ranges=(),
            seal_reason="duplicate",
            observed_trailing_silence_ms=800,
        )
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": first_task_id},
            "payload": {
                "output": {
                    "sentence": {
                        "sentence_id": "late",
                        "sentence_end": True,
                        "text": "late first task",
                    }
                }
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": first_task_id}})
    await socket.push({"header": {"event": "task-finished", "task_id": second_task_id}})
    second_terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert second_terminal.identity == second.identity
    assert second_terminal.outcome == "empty"
    assert second_terminal.provenance[0].native_task_id == second_task_id
    assert session._event_projection.scoped_event_depth == 0
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_scoped_native_failure_finish_timeout_and_socket_eof() -> None:
    _, session, socket, task_id = await open_fake(scoped=True)
    request = scoped_request("qwen_audio", task="native-failure")
    await session.begin_turn(request)
    await socket.push(
        {
            "header": {
                "event": "task-failed",
                "task_id": task_id,
                "error_code": "DECODE_FAILED",
                "error_message": "native failure",
            }
        }
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"

    _, session, _, _ = await open_fake(task_finish_timeout_s=0.01, scoped=True)
    request = scoped_request("qwen_audio", task="finish-timeout")
    await session.begin_turn(request)
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=800,
    )
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"

    _, session, socket, _ = await open_fake(scoped=True)
    request = scoped_request("qwen_audio", task="socket-eof")
    await session.begin_turn(request)
    await socket.push(None)
    terminal = await asyncio.wait_for(session.turn_events().__anext__(), timeout=1)
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"


@pytest.mark.asyncio
async def test_normal_task_aggregates_only_sentence_finals() -> None:
    _, session, socket, task_id = await open_fake(hotwords=["PuriPuly"])
    await session.send_audio(b"pcm")
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {
                "output": {"sentence": {"sentence_end": False, "sentence_id": 1, "text": "partial"}}
            },
        }
    )
    await session.on_speech_end()
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {
                "output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "첫 문장."}}
            },
        }
    )
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {
                "output": {
                    "sentence": {"sentence_end": True, "sentence_id": 2, "text": "둘째 문장."}
                }
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
    event = await next_event(session)
    assert event.text == "첫 문장.둘째 문장."
    assert event.is_final
    assert json.loads(socket.sent[0])["payload"]["parameters"]["vocabulary"] == {"PuriPuly": 4}
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_queued_audio_flushes_after_next_task_started_and_order_is_preserved() -> None:
    _, session, socket, first_id = await open_fake()
    await session.on_speech_end()
    await session.send_audio(b"next-utterance")
    await socket.push({"header": {"event": "task-finished", "task_id": first_id}})
    while len(socket.sent) < 3:
        await asyncio.sleep(0)
    run = json.loads(socket.sent[2])
    second_id = run["header"]["task_id"]
    assert run["header"]["action"] == "run-task"
    await socket.push({"header": {"event": "task-started", "task_id": second_id}})
    while b"next-utterance" not in socket.sent:
        await asyncio.sleep(0)
    await session.on_speech_end()
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": second_id},
            "payload": {
                "output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "second"}}
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": second_id}})
    first_event = await next_event(session)
    second_event = await next_event(session)
    assert first_event.text == ""
    assert second_event.text == "second"
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_empty_and_task_failed_each_resolve_one_boundary() -> None:
    _, session, socket, task_id = await open_fake()
    await session.on_speech_end()
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
    empty = await next_event(session)
    assert empty.text == ""
    await session.on_speech_end()
    await socket.push(
        {
            "header": {
                "event": "task-failed",
                "task_id": session.task_id,
                "error_code": "CLIENT_ERROR",
                "error_message": "failed",
            }
        }
    )
    failed = await next_event(session)
    assert failed.text == ""
    with pytest.raises(QwenAudioTaskFailedError):
        await next_event(session)
    assert session.state is QwenAudioSessionState.FAILED
    assert socket.closed


@pytest.mark.asyncio
async def test_stale_and_duplicate_events_do_not_consume_next_boundary() -> None:
    _, session, socket, task_id = await open_fake()
    await session.on_speech_end()
    await socket.push(
        {"header": {"event": "result-generated", "task_id": "stale", "event_id": "x"}}
    )
    await socket.push({"header": {"event": "task-finished", "task_id": "stale"}})
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
    first = await next_event(session)
    assert first.text == ""
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_hotword_update_is_applied_on_next_task_and_abort_suppresses_events() -> None:
    _, session, socket, first_id = await open_fake(hotwords=["old"])
    session.update_hotwords(["new"])
    await session.on_speech_end()
    await socket.push({"header": {"event": "task-finished", "task_id": first_id}})
    while len(socket.sent) < 3:
        await asyncio.sleep(0)
    second_id = json.loads(socket.sent[2])["header"]["task_id"]
    assert json.loads(socket.sent[2])["payload"]["parameters"]["vocabulary"] == {"new": 4}
    await socket.push({"header": {"event": "task-started", "task_id": second_id}})
    await session.on_speech_end()
    await session.abort_for_toggle_off()
    assert session.state is QwenAudioSessionState.CLOSING
    assert socket.closed


@pytest.mark.asyncio
async def test_stop_drains_last_task_before_closing() -> None:
    _, session, socket, task_id = await open_fake()
    await session.send_audio(b"last")
    stopping = asyncio.create_task(session.stop())
    await asyncio.sleep(0)
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {
                "output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "last"}}
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
    await stopping
    event = await next_event(session)
    assert event.text == "last"
    assert socket.closed


def test_qwen_audio_language_capabilities_are_model_specific() -> None:
    from puripuly_heart.core.language import (
        get_qwen3_asr_language,
        get_qwen_audio_asr_language,
        is_qwen3_asr_supported,
        is_qwen_audio_asr_supported,
    )

    assert get_qwen3_asr_language("ko") == "ko"
    assert get_qwen_audio_asr_language("ko-KR") == "ko"
    assert is_qwen_audio_asr_supported("tl")
    assert not is_qwen3_asr_supported("tl")


def test_qwen_audio_backend_contract_constants() -> None:
    from puripuly_heart.config.runtime_resolution import (
        QWEN_AUDIO_STT_MODEL,
        STTRuntimeIntent,
        resolve_stt_config,
    )

    assert QWEN_AUDIO_MODEL == "qwen-audio-3.1-asr-flash-streaming"
    assert QWEN_AUDIO_STT_MODEL == QWEN_AUDIO_MODEL
    resolved = resolve_stt_config(
        STTRuntimeIntent(
            provider="qwen_audio",
            qwen_region="singapore",
        )
    )
    assert resolved.model == "qwen-audio-3.1-asr-flash-streaming"
    assert resolved.provider == "qwen_audio"
    assert resolved.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
    assert resolved.region == "singapore"


def _run_task_payload(socket: FakeWebSocket, index: int = 0) -> dict:
    payload = json.loads(socket.sent[index])
    assert payload["header"]["action"] == "run-task"
    assert payload["header"]["streaming"] == "duplex"
    return payload


def _assert_baseline_run_task(payload: dict, *, vocabulary: dict[str, int] | None) -> None:
    body = payload["payload"]
    parameters = body["parameters"]
    expected = {
        "format": "pcm",
        "sample_rate": 16000,
        "semantic_punctuation_enabled": False,
        "max_sentence_silence": 6000,
        "multi_threshold_mode_enabled": False,
        "heartbeat": True,
    }
    if vocabulary is not None:
        expected["vocabulary"] = vocabulary
    assert body["task_group"] == "audio"
    assert body["task"] == "asr"
    assert body["function"] == "recognition"
    assert body["model"] == "qwen-audio-3.1-asr-flash-streaming"
    assert body["input"] == {}
    assert "context" not in body
    assert parameters == expected


@pytest.mark.asyncio
async def test_run_task_omits_context_and_vocabulary_when_hotwords_are_off() -> None:
    _, session, socket, _ = await open_fake(language_hints=(), hotwords=())
    _assert_baseline_run_task(_run_task_payload(socket), vocabulary=None)
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_run_task_maps_instant_hotwords_without_context_or_31_controls() -> None:
    _, session, socket, _ = await open_fake(language_hints=(), hotwords=["PuriPuly", "VRChat"])
    _assert_baseline_run_task(
        _run_task_payload(socket),
        vocabulary={"PuriPuly": 4, "VRChat": 4},
    )
    await session.abort_for_toggle_off()


def _run_task_parameters(socket: FakeWebSocket, index: int = 0) -> dict:
    payload = json.loads(socket.sent[index])
    assert payload["header"]["action"] == "run-task"
    return payload["payload"]["parameters"]


@pytest.mark.asyncio
async def test_auto_detect_without_hints_omits_language_hints_key() -> None:
    _, session, socket, _ = await open_fake(language_hints=())
    parameters = _run_task_parameters(socket)
    assert "language_hints" not in parameters
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_manual_language_sends_single_mapped_hint_in_payload() -> None:
    _, session, socket, _ = await open_fake(language_hints=("ja",))
    assert _run_task_parameters(socket)["language_hints"] == ["ja"]
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_constrained_auto_detect_sends_ordered_hints_in_payload() -> None:
    _, session, socket, _ = await open_fake(language_hints=("ko", "ja", "zh", "en"))
    assert _run_task_parameters(socket)["language_hints"] == ["ko", "ja", "zh", "en"]
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_repeated_tasks_reuse_the_same_language_hints() -> None:
    _, session, socket, first_id = await open_fake(language_hints=("ko", "ja"))
    await session.on_speech_end()
    await socket.push({"header": {"event": "task-finished", "task_id": first_id}})
    await wait_for_condition(lambda: len(socket.sent) >= 3)
    assert _run_task_parameters(socket, 0)["language_hints"] == ["ko", "ja"]
    assert _run_task_parameters(socket, 2)["language_hints"] == ["ko", "ja"]
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_task_finish_timeout_emits_empty_boundary_then_failure() -> None:
    _, session, socket, _ = await open_fake()
    await session.on_speech_end()
    event = await asyncio.wait_for(next_event(session), timeout=1)
    assert event.text == ""
    with pytest.raises(QwenAudioProtocolError, match="task-finished timeout"):
        await asyncio.wait_for(next_event(session), timeout=1)
    assert session.state is QwenAudioSessionState.FAILED
    assert socket.closed


@pytest.mark.asyncio
async def test_vocabulary_rejection_is_diagnosable() -> None:
    _, session, socket, task_id = await open_fake(hotwords=["PuriPuly"])
    await session.on_speech_end()
    await socket.push(
        {
            "header": {
                "event": "task-failed",
                "task_id": task_id,
                "error_code": "VOCABULARY_NOT_SUPPORTED",
                "error_message": "vocabulary is not supported",
            }
        }
    )
    event = await next_event(session)
    assert event.text == ""
    with pytest.raises(QwenAudioTaskFailedError) as failure:
        await next_event(session)
    assert failure.value.hotwords_rejected


@pytest.mark.asyncio
async def test_socket_close_reports_protocol_failure() -> None:
    _, session, socket, _ = await open_fake()
    await socket.close()
    with pytest.raises(QwenAudioProtocolError, match="socket"):
        await asyncio.wait_for(next_event(session), timeout=1)


def test_factory_selects_qwen_audio_protocol_and_source_terms() -> None:
    resolved = resolve_stt_config(
        STTRuntimeIntent(
            provider="qwen_audio",
            source_language="tl",
            qwen_region="singapore",
            custom_vocabulary_enabled=True,
            custom_terms={"tl": ("PuriPuly", "Qwen")},
        )
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "test-key")
    backend = create_stt_backend_from_resolved_config(resolved, secrets=secrets)
    assert isinstance(backend, QwenAudioStreamingSTTBackend)
    assert backend.language_hints == ("tl",)
    assert backend.endpoint.endswith("/api-ws/v1/inference")
    assert tuple(backend.hotwords) == ("PuriPuly", "Qwen")


def test_factory_omits_manual_hint_for_unsupported_language() -> None:
    resolved = resolve_stt_config(STTRuntimeIntent(provider="qwen_audio", source_language="tr"))
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "test-key")

    backend = create_stt_backend_from_resolved_config(resolved, secrets=secrets)

    assert isinstance(backend, QwenAudioStreamingSTTBackend)
    assert backend.language_hints == ()


async def wait_for_condition(predicate) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


def sent_action(socket: FakeWebSocket, action: str, task_id: str | None = None) -> bool:
    for value in socket.sent:
        if not isinstance(value, str):
            continue
        header = json.loads(value).get("header", {})
        if header.get("action") == action and (task_id is None or header.get("task_id") == task_id):
            return True
    return False


@pytest.mark.asyncio
async def test_delayed_flush_serializes_pcm_before_finish_task() -> None:
    _, session, socket, first_id = await open_fake()
    await session.on_speech_end()
    await session.send_audio(b"a")
    await session.send_audio(b"b")
    await socket.push({"header": {"event": "task-finished", "task_id": first_id}})
    await wait_for_condition(lambda: len(socket.sent) >= 3)
    second_id = json.loads(socket.sent[2])["header"]["task_id"]
    socket.block_audio = True
    await socket.push({"header": {"event": "task-started", "task_id": second_id}})
    await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
    await session.send_audio(b"c")
    await session.send_audio(b"d")
    await session.on_speech_end()
    socket.audio_gate.set()
    await wait_for_condition(lambda: sent_action(socket, "finish-task", second_id))
    binary = [value for value in socket.sent if isinstance(value, bytes)]
    assert binary == [b"a", b"b", b"c", b"d"]
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": second_id},
            "payload": {
                "output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "done"}}
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": second_id}})
    assert (await next_event(session)).text == ""
    assert (await next_event(session)).text == "done"
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_admitted_audio_waiter_precedes_speech_end_finish() -> None:
    _, session, socket, _ = await open_fake()
    socket.block_audio = True
    first = asyncio.create_task(session.send_audio(b"first"))
    await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
    second = asyncio.create_task(session.send_audio(b"second"))
    await asyncio.sleep(0)
    speech_end = asyncio.create_task(session.on_speech_end())
    await asyncio.sleep(0)
    assert not speech_end.done()
    socket.audio_gate.set()
    await asyncio.wait_for(asyncio.gather(first, second, speech_end), timeout=1)
    finish_index = next(
        index
        for index, value in enumerate(socket.sent)
        if isinstance(value, str) and json.loads(value)["header"]["action"] == "finish-task"
    )
    assert socket.sent[finish_index - 2 : finish_index] == [b"first", b"second"]
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_stop_drains_two_ordered_nonempty_boundaries() -> None:
    _, session, socket, first_id = await open_fake()
    await session.send_audio(b"first-pcm")
    await session.on_speech_end()
    await session.send_audio(b"second-pcm")
    await session.on_speech_end()
    stopping = asyncio.create_task(session.stop())
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": first_id},
            "payload": {
                "output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "first"}}
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": first_id}})
    await wait_for_condition(lambda: len(socket.sent) >= 4)
    second_id = json.loads(socket.sent[3])["header"]["task_id"]
    await socket.push({"header": {"event": "task-started", "task_id": second_id}})
    await wait_for_condition(lambda: b"second-pcm" in socket.sent)
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": second_id},
            "payload": {
                "output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "second"}}
            },
        }
    )
    await socket.push({"header": {"event": "task-finished", "task_id": second_id}})
    await asyncio.wait_for(stopping, timeout=1)
    assert (await next_event(session)).text == "first"
    assert (await next_event(session)).text == "second"
    assert socket.closed


@pytest.mark.asyncio
async def test_task_start_timeout_is_terminal() -> None:
    socket = FakeWebSocket()

    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        return socket

    backend = QwenAudioStreamingSTTBackend(
        api_key="test-key",
        language_hints=("ko",),
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=0.01,
        task_finish_timeout_s=0.2,
    )
    with pytest.raises(QwenAudioProtocolError, match="task-started timeout"):
        await backend.open_session()
    assert socket.closed


@pytest.mark.asyncio
async def test_initial_connection_failure_is_terminal() -> None:
    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        raise RuntimeError("connect failed")

    backend = QwenAudioStreamingSTTBackend(
        api_key="test-key",
        language_hints=("ko",),
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=1,
        task_finish_timeout_s=0.2,
    )
    with pytest.raises(QwenAudioProtocolError, match="connection failed"):
        await backend.open_session()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "stage", "reason"),
    [
        ("connect", "connect", "connect_failed"),
        ("send", "run_task_send", "run_task_send_failed"),
        ("timeout", "task_start_wait", "task_start_timeout"),
        ("receive", "task_start_wait", "receive_failed"),
        ("eof", "task_start_wait", "receive_eof"),
    ],
)
async def test_startup_failure_diagnostics_exclude_transport_content(
    caplog: pytest.LogCaptureFixture, failure: str, stage: str, reason: str
) -> None:
    secret = "private-api-key"
    host = "private.workspace.example"
    socket = FakeWebSocket()

    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        if failure == "connect":
            raise RuntimeError(f"{secret} {host}")
        return socket

    if failure == "send":

        async def failing_send(value: str | bytes) -> None:
            raise RuntimeError(f"{secret} {host} {value}")

        socket.send = failing_send
    elif failure == "receive":

        async def failing_receive() -> object:
            while not socket.sent:
                await asyncio.sleep(0)
            await asyncio.sleep(0.005)
            raise RuntimeError(f"{secret} {host}")

        socket.recv = failing_receive
    elif failure == "eof":

        async def eof_receive() -> object:
            while not socket.sent:
                await asyncio.sleep(0)
            await asyncio.sleep(0.005)
            return None

        socket.recv = eof_receive

    backend = QwenAudioStreamingSTTBackend(
        api_key=secret,
        endpoint=f"wss://{host}/api-ws/v1/inference",
        websocket_factory=connect,
        task_start_timeout_s=0.02,
    )
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.qwen_audio"):
        with pytest.raises((QwenAudioProtocolError, RuntimeError)):
            await backend.open_session()
    records = [r.getMessage() for r in caplog.records if "[QwenAudio] failure" in r.getMessage()]
    assert len(records) == 1
    assert f"stage={stage}" in records[0]
    assert f"reason={reason}" in records[0]
    assert "startup_ms=" in records[0] and "stage_ms=" in records[0]
    assert secret not in caplog.text and host not in caplog.text
    assert all(isinstance(value, str) for value in socket.sent)
    for value in socket.sent:
        if isinstance(value, str):
            assert json.loads(value)["header"]["task_id"] not in caplog.text


@pytest.mark.asyncio
async def test_close_timing_diagnostic_after_successful_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.qwen_audio"):
        _, session, socket, task_id = await open_fake()
        await session.abort_for_toggle_off()
    assert socket.closed
    assert any("startup stage=task_started" in r.getMessage() for r in caplog.records)
    close_records = [
        r.getMessage() for r in caplog.records if "[QwenAudio] close " in r.getMessage()
    ]
    assert len(close_records) == 1
    assert "elapsed_ms=" in close_records[0] and "outcome=completed" in close_records[0]
    assert task_id not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("send_fails", [False, True])
async def test_task_started_during_run_task_send_keeps_diagnostic_phase(
    caplog: pytest.LogCaptureFixture, send_fails: bool
) -> None:
    secret = "private-send-key"
    host = "private.workspace.example"
    socket = FakeWebSocket()
    send_entered = asyncio.Event()
    release_send = asyncio.Event()

    async def delayed_send(value: str | bytes) -> None:
        assert isinstance(value, str)
        socket.sent.append(value)
        send_entered.set()
        await release_send.wait()
        if send_fails:
            raise RuntimeError(f"{secret} {host}")

    socket.send = delayed_send

    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        return socket

    backend = QwenAudioStreamingSTTBackend(
        api_key=secret,
        endpoint=f"wss://{host}/api-ws/v1/inference",
        websocket_factory=connect,
        task_start_timeout_s=1,
    )
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.qwen_audio"):
        opening = asyncio.create_task(backend.open_session())
        try:
            await asyncio.wait_for(send_entered.wait(), timeout=1)
            task_id = json.loads(socket.sent[0])["header"]["task_id"]
            await socket.push({"header": {"event": "task-started", "task_id": task_id}})
            for _ in range(100):
                if any("startup stage=task_started" in r.getMessage() for r in caplog.records):
                    break
                await asyncio.sleep(0.005)
            else:
                pytest.fail("task-start acknowledgement was not processed")
            assert not opening.done()
            await asyncio.sleep(0.04)
            release_send.set()
            if send_fails:
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(opening, timeout=1)
            else:
                session = await asyncio.wait_for(opening, timeout=1)
                assert session.state is QwenAudioSessionState.TASK_ACTIVE
                await socket.close()
                for _ in range(100):
                    if any("[QwenAudio] failure" in r.getMessage() for r in caplog.records):
                        break
                    await asyncio.sleep(0.005)
                else:
                    pytest.fail("early socket close was not observed")
                await session.close()
        finally:
            release_send.set()
            if not opening.done():
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)

    def fields(message: str) -> dict[str, str]:
        return dict(part.split("=", 1) for part in message.split() if "=" in part)

    ack = next(
        fields(r.getMessage())
        for r in caplog.records
        if "startup stage=task_started" in r.getMessage()
    )
    assert ack["wait_ms"] == "0"
    assert ack["ack_during_send"] == "true"
    failures = [
        fields(r.getMessage()) for r in caplog.records if "[QwenAudio] failure" in r.getMessage()
    ]
    assert len(failures) == 1
    failure = failures[0]
    assert failure["reason"] == ("run_task_send_failed" if send_fails else "receive_eof")
    assert failure["stage"] == ("run_task_send" if send_fails else "active")
    if send_fails:
        assert int(failure["stage_ms"]) >= 30
    else:
        assert int(failure["startup_ms"]) - int(failure["stage_ms"]) >= 20
    assert secret not in caplog.text and host not in caplog.text and task_id not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_open", [False, True])
async def test_abandoned_start_future_never_reports_private_exception(
    cancel_open: bool,
) -> None:
    secret = "private-initial-send-key"
    socket = FakeWebSocket()
    send_entered = asyncio.Event()
    release_send = asyncio.Event()
    loop = asyncio.get_running_loop()
    contexts: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    async def send(value: str | bytes) -> None:
        socket.sent.append(value)
        send_entered.set()
        await release_send.wait()
        raise RuntimeError(f"{secret} {value}")

    socket.send = send

    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        return socket

    backend = QwenAudioStreamingSTTBackend(
        api_key=secret,
        websocket_factory=connect,
        task_start_timeout_s=1,
    )
    try:
        opening = asyncio.create_task(backend.open_session())
        await asyncio.wait_for(send_entered.wait(), timeout=1)
        if cancel_open:
            opening.cancel()
            with pytest.raises(asyncio.CancelledError):
                await opening
        else:
            release_send.set()
            with pytest.raises(RuntimeError, match=secret):
                await opening
        assert socket.closed
        gc.collect()
        await asyncio.sleep(0)
        gc.collect()
        assert contexts == []
    finally:
        release_send.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_heartbeat_result_does_not_create_transcript() -> None:
    _, session, socket, task_id = await open_fake()
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {
                "output": {"sentence": {"heartbeat": True, "sentence_end": True, "sentence_id": 0}}
            },
        }
    )
    await session.on_speech_end()
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
    event = await next_event(session)
    assert event.text == ""
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_audio_send_failure_reaches_single_terminal_event() -> None:
    _, session, socket, _ = await open_fake()
    socket.fail_audio = True
    await session.send_audio(b"broken")
    with pytest.raises(QwenAudioProtocolError, match="audio send failed"):
        await next_event(session)
    assert socket.closed


@pytest.mark.asyncio
async def test_finish_send_failure_reaches_single_terminal_event() -> None:
    _, session, socket, _ = await open_fake()
    socket.fail_text = True
    await session.on_speech_end()
    assert (await next_event(session)).text == ""
    with pytest.raises(QwenAudioProtocolError, match="finish-task send failed"):
        await next_event(session)
    assert socket.closed


@pytest.mark.asyncio
async def test_blocked_audio_send_timeout_closes_session() -> None:
    _, session, socket, _ = await open_fake()
    setattr(session, "send_timeout_s", 0.01)
    socket.block_audio = True
    sending = asyncio.create_task(session.send_audio(b"blocked"))
    await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
    await asyncio.wait_for(sending, timeout=1)
    with pytest.raises(QwenAudioProtocolError, match="audio send failed"):
        await next_event(session)
    assert socket.closed


@pytest.mark.asyncio
async def test_abort_waits_for_blocked_audio_send_without_propagation() -> None:
    _, session, socket, _ = await open_fake()
    socket.block_audio = True
    sending = asyncio.create_task(session.send_audio(b"cancelled"))
    await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
    await session.abort_for_toggle_off()
    await asyncio.wait_for(sending, timeout=1)
    assert socket.closed
    assert b"cancelled" not in socket.sent


@pytest.mark.asyncio
async def test_close_coordinates_with_blocked_audio_send() -> None:
    _, session, socket, _ = await open_fake(task_finish_timeout_s=0.05)
    socket.block_audio = True
    sending = asyncio.create_task(session.send_audio(b"closing"))
    await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
    await asyncio.wait_for(session.close(), timeout=1)
    await asyncio.wait_for(sending, timeout=1)
    assert socket.closed
    assert b"closing" not in socket.sent


@pytest.mark.asyncio
async def test_abort_releases_two_audio_waiters_without_ingress_failures() -> None:
    _, session, socket, _ = await open_fake()
    socket.block_audio = True
    first = asyncio.create_task(session.send_audio(b"first-waiter"))
    await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
    second = asyncio.create_task(session.send_audio(b"second-waiter"))
    await asyncio.sleep(0)
    await session.abort_for_toggle_off()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert socket.closed
    assert b"first-waiter" not in socket.sent
    assert b"second-waiter" not in socket.sent


@pytest.mark.asyncio
async def test_keepalive_sends_zero_silence_after_idle() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.05)
    try:
        audio_frames: list[str | bytes] = []
        for _ in range(200):
            audio_frames = [item for item in socket.sent if isinstance(item, bytes)]
            if audio_frames:
                break
            await asyncio.sleep(0.01)
        assert audio_frames
        assert all(len(frame) == 3200 for frame in audio_frames)
        assert all(set(frame) == {0} for frame in audio_frames)
        assert session.state is QwenAudioSessionState.TASK_ACTIVE
    finally:
        await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_keepalive_stays_quiet_while_audio_flows() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.1)
    try:
        for _ in range(10):
            await session.send_audio(b"\x01\x02")
            await asyncio.sleep(0.02)
        audio_frames = [item for item in socket.sent if isinstance(item, bytes)]
        assert audio_frames
        assert all(frame == b"\x01\x02" for frame in audio_frames)
    finally:
        await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_keepalive_skipped_while_finishing_task() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.05, task_finish_timeout_s=1)
    try:
        await session.on_speech_end()
        assert session.state is QwenAudioSessionState.FINISHING_TASK
        sent_before = len(socket.sent)
        await asyncio.sleep(0.25)
        assert session.state is QwenAudioSessionState.FINISHING_TASK
        assert [item for item in socket.sent[sent_before:] if isinstance(item, bytes)] == []
    finally:
        await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_keepalive_options_must_be_positive() -> None:
    for kwargs in (
        {"keepalive_interval_s": 0},
        {"keepalive_silence_ms": 0},
        {"keepalive_silence_ms": 0.01},
    ):
        backend = QwenAudioStreamingSTTBackend(
            api_key="test-key",
            language_hints=("ko",),
            connect_timeout_s=1,
            task_start_timeout_s=1,
            task_finish_timeout_s=0.2,
            send_timeout_s=5,
            **kwargs,
        )
        with pytest.raises(ValueError):
            await backend.open_session()


@pytest.mark.asyncio
async def test_keepalive_stops_after_close() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.05)
    await session.abort_for_toggle_off()
    sent_count = len(socket.sent)
    await asyncio.sleep(0.25)
    assert len(socket.sent) == sent_count


@pytest.mark.asyncio
async def test_keepalive_skipped_with_queued_audio_boundary_or_flush() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.05)
    try:
        session._audio_queue.append(b"queued")
        session._post_boundary_audio_queue.append(b"post-boundary")
        session._boundary_requested = True
        session._flushing_audio = True
        assert session.state is QwenAudioSessionState.TASK_ACTIVE
        sent_before = len(socket.sent)
        await asyncio.sleep(0.25)
        assert session.state is QwenAudioSessionState.TASK_ACTIVE
        assert [item for item in socket.sent[sent_before:] if isinstance(item, bytes)] == []
    finally:
        await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_keepalive_timer_resets_on_task_started() -> None:
    _, session, socket, task_id = await open_fake(keepalive_interval_s=0.2)
    try:
        for _ in range(300):
            if sum(1 for item in socket.sent if isinstance(item, bytes)) >= 2:
                break
            await asyncio.sleep(0.01)
        keepalive_count = sum(1 for item in socket.sent if isinstance(item, bytes))
        assert keepalive_count >= 2
        await asyncio.sleep(0.1)
        await session.on_speech_end()
        await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
        second_id = None
        for _ in range(300):
            for item in socket.sent:
                if isinstance(item, str):
                    header = json.loads(item).get("header", {})
                    if header.get("action") == "run-task" and header.get("task_id") != task_id:
                        second_id = header.get("task_id")
            if second_id:
                break
            await asyncio.sleep(0.01)
        assert second_id
        await socket.push({"header": {"event": "task-started", "task_id": second_id}})
        await asyncio.sleep(0.19)
        assert session.state is QwenAudioSessionState.TASK_ACTIVE
        assert sum(1 for item in socket.sent if isinstance(item, bytes)) == keepalive_count
    finally:
        await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_keepalive_failure_fails_session() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.05)
    socket.fail_audio = True
    with pytest.raises(QwenAudioProtocolError, match="keepalive send failed"):
        await asyncio.wait_for(next_event(session), timeout=2)
    assert session.state is QwenAudioSessionState.FAILED
    assert socket.closed


@pytest.mark.asyncio
async def test_abort_drains_keepalive_blocked_in_audio_send() -> None:
    _, session, socket, _ = await open_fake(keepalive_interval_s=0.01)
    socket.block_audio = True
    keepalive = session._keepalive_task
    assert keepalive is not None
    try:
        await asyncio.wait_for(socket.audio_started.wait(), timeout=1)
        await asyncio.wait_for(session.abort_for_toggle_off(), timeout=1)
        assert socket.closed
        assert keepalive.done()
        assert not any(isinstance(item, bytes) for item in socket.sent)
    finally:
        keepalive.cancel()
        await asyncio.gather(keepalive, return_exceptions=True)
        await session.abort_for_toggle_off()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sentences", "expected_text", "expected_suffix"),
    [
        (("hello", "world"), "hello world", " world"),
        (("你好", "世界"), "你好世界", "世界"),
    ],
)
async def test_real_qwen_audio_shared_engine_uses_one_sentence_join_projection(
    sentences: tuple[str, str],
    expected_text: str,
    expected_suffix: str,
) -> None:
    socket = FakeWebSocket()

    async def connect(*_args: object, **_kwargs: object) -> FakeWebSocket:
        return socket

    backend = QwenAudioStreamingSTTBackend(
        api_key="test-key",
        language_hints=("en", "zh"),
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=1,
        task_finish_timeout_s=1,
    )

    async def open_session(_settings, provider_epoch_id):
        opening = asyncio.create_task(
            backend.open_session(
                projection=STTSessionProjection("scoped", provider_epoch_id),
            )
        )
        await wait_for_condition(lambda: bool(socket.sent))
        task_id = json.loads(socket.sent[0])["header"]["task_id"]
        await socket.push({"header": {"event": "task-started", "task_id": task_id}})
        return await opening

    provider_settings = AudioSegmentSettingsSnapshot(
        provider_id="qwen_audio",
        provider_signature=("qwen_audio",),
        runtime_signature=("qwen_audio",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en", "zh"),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=provider_settings)
    segment_id = uuid4()
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            np.empty((0,), dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ),
        now_monotonic_s=0.0,
    )
    end = ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=800, reason="silence"),
        now_monotonic_s=1.0,
    )
    events: list[object] = []
    consumption = STTContributionConsumptionLedger()
    consumed: list[str] = []

    def consume_event(event: object) -> None:
        events.append(event)
        if isinstance(event, STTProviderTurnUpdate) and not consumed:
            consumed.append(consumption.consume(event))
        elif isinstance(event, STTProviderTurnTerminal):
            consumed.append(consumption.consume(event))

    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        event_sink=consume_event,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.2,
        ),
    )
    await engine.handle_owned_vad_event(start)
    task_id = json.loads(socket.sent[0])["header"]["task_id"]
    for index, sentence in enumerate(sentences, start=1):
        await socket.push(
            {
                "header": {"event": "result-generated", "task_id": task_id},
                "payload": {
                    "output": {
                        "sentence": {
                            "sentence_id": f"sentence-{index}",
                            "sentence_end": True,
                            "text": sentence,
                        }
                    }
                },
            }
        )
    await wait_for_condition(
        lambda: len([event for event in events if isinstance(event, STTProviderTurnUpdate)]) == 2
    )
    ending = asyncio.create_task(engine.handle_owned_vad_event(end))
    await wait_for_condition(
        lambda: any(
            isinstance(payload, str)
            and json.loads(payload).get("header", {}).get("action") == "finish-task"
            for payload in socket.sent
        )
    )
    await socket.push({"header": {"event": "task-finished", "task_id": task_id}})
    await ending

    updates = [event for event in events if isinstance(event, STTProviderTurnUpdate)]
    terminal = next(event for event in events if isinstance(event, STTProviderTurnTerminal))
    assert [event.text for event in updates] == [sentences[0], expected_text]
    assert terminal.outcome == "final"
    assert terminal.text == expected_text
    assert consumed == [sentences[0], expected_suffix]
    assert [(item.text_start, item.text_end) for item in terminal.included_contributions] == [
        (0, len(sentences[0])),
        (len(sentences[0]), len(expected_text)),
    ]
    await engine.close()


@pytest.mark.asyncio
async def test_peer_language_handoff_finishes_backend_cleanup_before_new_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_stt_scoped_engine import span

    class ServerSocket(FakeWebSocket):
        def __init__(self, text: str, *, gated: bool) -> None:
            super().__init__()
            self.text = text
            self.has_audio = False
            self.close_started = asyncio.Event()
            self.close_gate = asyncio.Event()
            if not gated:
                self.close_gate.set()

        async def send(self, value: str | bytes) -> None:
            await super().send(value)
            if isinstance(value, bytes):
                if not self.has_audio:
                    self.has_audio = True
                    await self.push(
                        {
                            "header": {"event": "result-generated", "task_id": self.task_id},
                            "payload": {
                                "output": {
                                    "sentence": {
                                        "sentence_id": 1,
                                        "sentence_end": True,
                                        "text": self.text,
                                    }
                                }
                            },
                        }
                    )
                return
            header = json.loads(value)["header"]
            task_id = header["task_id"]
            self.task_id = task_id
            if header["action"] == "run-task":
                self.has_audio = False
                await self.push({"header": {"event": "task-started", "task_id": task_id}})
            elif header["action"] == "finish-task":
                await self.push({"header": {"event": "task-finished", "task_id": task_id}})

        async def close(self) -> None:
            self.close_started.set()
            await self.close_gate.wait()
            await super().close()

    old_socket = ServerSocket("你好", gated=True)
    new_socket = ServerSocket("こんにちは", gated=False)
    sockets = iter((old_socket, new_socket))

    async def connect(*_args: object, **_kwargs: object) -> ServerSocket:
        return next(sockets)

    monkeypatch.setattr(network_clients, "external_websocket_connect", connect)
    settings = from_dict(
        {
            "intent": {
                "peer_stt": {"provider": "qwen_audio"},
                "languages": {
                    "peer_source_language": "zh-CN",
                    "peer_source_mode": "manual",
                    "peer_expected_languages": ["zh-CN", "ja"],
                },
                "stt": {"drain_timeout_s": 0.5},
            }
        }
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "test-key")
    factory = SharedSTTProviderFactory(
        secrets=secrets,
        clock=SystemClock(),
        reset_deadline_s=1.0,
        gpu_model_path=Path("."),
    )
    events: list[object] = []
    owner = LocalASRProviderRuntimeOwner(
        provider_factory=factory,
        gpu_runtime_factory=lambda sink: SharedGpuASRRuntime(
            process_factory=DefaultGpuWorkerProcessFactory(),
            clock=factory.clock,
            diagnostic_sink=sink,
        ),
        provisioning=None,
        peer_event_handler=lambda event: asyncio.sleep(0, result=events.append(event)),
    )

    def configuration(mode: str):
        candidate = replace(
            settings,
            intent=replace(
                settings.intent,
                languages=replace(settings.intent.languages, peer_source_mode=mode),
            ),
        )
        config = build_peer_capture_session_config_from_vnext(candidate)
        snapshot = AudioSegmentSettingsSnapshot(
            provider_id=config.provider_id,
            provider_signature=config.provider_signature,
            runtime_signature=config.runtime_signature,
            source_mode=config.delivery_language.source_mode,
            source_language=config.delivery_language.source_language,
            expected_languages=config.delivery_language.expected_languages,
            target_sample_rate_hz=config.target_sample_rate_hz,
            vad_speech_threshold=config.vad_speech_threshold,
            vad_hangover_ms=config.vad_hangover_ms,
            vad_pre_roll_ms=config.vad_pre_roll_ms,
        )
        ledger = PeerAudioSegmentLedger(activation_generation=1, settings=snapshot)
        segment_id = uuid4()
        start = ledger.observe_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=np.ones((8,), dtype=np.float32),
                chunk_capture=(span(0, 0, 8),),
            ),
            now_monotonic_s=0.0,
        )
        end = ledger.observe_vad_event(
            SpeechEnd(segment_id, trailing_silence_ms=500, reason="silence"),
            now_monotonic_s=1.0,
        )
        return build_peer_stt_provider_request(config, gpu_device_id="auto"), start, end

    admission = None
    try:
        await owner.start()
        old_request, old_start, old_end = configuration("manual")
        await owner.replace_provider(old_request, start=True)
        old_engine = owner.current_provider("peer")
        assert isinstance(old_engine, ScopedRecognitionEngine)
        old_engine.event_drain_timeout_s = 0.05
        await owner.handle_owned_vad_event("peer", old_start)
        await owner.handle_owned_vad_event("peer", old_end)
        await wait_for_condition(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in events)
            and any(isinstance(event, STTProviderInputTerminal) for event in events)
            and old_engine._session.state is QwenAudioSessionState.TASK_ACTIVE
        )
        new_request, new_start, new_end = configuration("auto")
        await owner.handoff_provider(new_request, start=True)
        admission = asyncio.create_task(owner.handle_owned_vad_event("peer", new_start))
        await asyncio.wait_for(old_socket.close_started.wait(), timeout=1)
        async with asyncio.timeout(1):
            while not admission.done() and old_engine._backend_close_task is None:
                await asyncio.sleep(0.001)
        if admission.done():
            await admission
        assert not new_socket.sent
        old_socket.close_gate.set()
        await asyncio.wait_for(admission, timeout=1)
        await owner.handle_owned_vad_event("peer", new_end)
        await wait_for_condition(
            lambda: len(
                [event for event in events if isinstance(event, STTRecognitionUnitTerminal)]
            )
            == 2
        )

        terminals = [event for event in events if isinstance(event, STTRecognitionUnitTerminal)]
        assert [(event.outcome, event.unit.text) for event in terminals] == [
            ("final", "你好"),
            ("final", "こんにちは"),
        ]
        assert terminals[0].unit.identity.stream.settings_scope == old_engine.scoped_settings_scope
        assert (
            terminals[1].unit.identity.stream.settings_scope
            != terminals[0].unit.identity.stream.settings_scope
        )
        submitted = [event for event in events if isinstance(event, STTProviderInputTerminal)]
        assert len(submitted) == 2 and all(event.outcome == "submitted" for event in submitted)
        assert old_socket.closed
        assert old_engine.cleanup_debt == 0
        assert owner.snapshot.channel_for("peer").phase == "running"
    finally:
        old_socket.close_gate.set()
        if admission is not None:
            await asyncio.gather(admission, return_exceptions=True)
        await owner.close()
