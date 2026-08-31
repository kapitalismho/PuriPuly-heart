from __future__ import annotations

import asyncio
import json

import pytest

from puripuly_heart.app.wiring.wiring_stt_factory import create_stt_backend_from_resolved_config
from puripuly_heart.config.runtime_resolution import STTRuntimeIntent, resolve_stt_config
from puripuly_heart.core.storage.secrets import InMemorySecretStore
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
) -> tuple[QwenAudioStreamingSTTBackend, object, FakeWebSocket, str]:
    socket = FakeWebSocket()

    async def connect(*args: object, **kwargs: object) -> FakeWebSocket:
        return socket

    backend = QwenAudioStreamingSTTBackend(
        api_key="test-key",
        language="ko",
        hotwords=hotwords,
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=task_start_timeout_s,
        task_finish_timeout_s=task_finish_timeout_s,
        send_timeout_s=send_timeout_s,
    )
    opening = asyncio.create_task(backend.open_session())
    while not socket.sent:
        await asyncio.sleep(0)
    first_id = json.loads(socket.sent[0])['header']['task_id']
    await socket.push({'header': {'event': 'task-started', 'task_id': first_id}})
    session = await opening
    return backend, session, socket, first_id


async def next_event(session: object):
    return await session.events().__anext__()


@pytest.mark.asyncio
async def test_normal_task_aggregates_only_sentence_finals() -> None:
    _, session, socket, task_id = await open_fake(hotwords=["PuriPuly"])
    await session.send_audio(b"pcm")
    await socket.push(
        {
            'header': {'event': 'result-generated', 'task_id': task_id},
            'payload': {'output': {'sentence': {'sentence_end': False, 'sentence_id': 1, 'text': 'partial'}}},
        }
    )
    await session.on_speech_end()
    await socket.push(
        {
            'header': {'event': 'result-generated', 'task_id': task_id},
            'payload': {'output': {'sentence': {'sentence_end': True, 'sentence_id': 1, 'text': '첫 문장.'}}},
        }
    )
    await socket.push(
        {
            'header': {'event': 'result-generated', 'task_id': task_id},
            'payload': {'output': {'sentence': {'sentence_end': True, 'sentence_id': 2, 'text': '둘째 문장.'}}},
        }
    )
    await socket.push({'header': {'event': 'task-finished', 'task_id': task_id}})
    event = await next_event(session)
    assert event.text == "첫 문장.둘째 문장."
    assert event.is_final
    assert json.loads(socket.sent[0])['payload']['parameters']['vocabulary'] == {'PuriPuly': 4}
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_queued_audio_flushes_after_next_task_started_and_order_is_preserved() -> None:
    _, session, socket, first_id = await open_fake()
    await session.on_speech_end()
    await session.send_audio(b"next-utterance")
    await socket.push({'header': {'event': 'task-finished', 'task_id': first_id}})
    while len(socket.sent) < 3:
        await asyncio.sleep(0)
    run = json.loads(socket.sent[2])
    second_id = run['header']['task_id']
    assert run['header']['action'] == 'run-task'
    await socket.push({'header': {'event': 'task-started', 'task_id': second_id}})
    while b"next-utterance" not in socket.sent:
        await asyncio.sleep(0)
    await session.on_speech_end()
    await socket.push(
        {
            'header': {'event': 'result-generated', 'task_id': second_id},
            'payload': {'output': {'sentence': {'sentence_end': True, 'sentence_id': 1, 'text': 'second'}}},
        }
    )
    await socket.push({'header': {'event': 'task-finished', 'task_id': second_id}})
    first_event = await next_event(session)
    second_event = await next_event(session)
    assert first_event.text == ""
    assert second_event.text == "second"
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_empty_and_task_failed_each_resolve_one_boundary() -> None:
    _, session, socket, task_id = await open_fake()
    await session.on_speech_end()
    await socket.push({'header': {'event': 'task-finished', 'task_id': task_id}})
    empty = await next_event(session)
    assert empty.text == ""
    await session.on_speech_end()
    await socket.push(
        {
            'header': {
                'event': 'task-failed',
                'task_id': session.task_id,
                'error_code': 'CLIENT_ERROR',
                'error_message': 'failed',
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
    await socket.push({'header': {'event': 'result-generated', 'task_id': 'stale', 'event_id': 'x'}})
    await socket.push({'header': {'event': 'task-finished', 'task_id': 'stale'}})
    await socket.push({'header': {'event': 'task-finished', 'task_id': task_id}})
    first = await next_event(session)
    assert first.text == ""
    await session.abort_for_toggle_off()


@pytest.mark.asyncio
async def test_hotword_update_is_applied_on_next_task_and_abort_suppresses_events() -> None:
    _, session, socket, first_id = await open_fake(hotwords=["old"])
    session.update_hotwords(["new"])
    await session.on_speech_end()
    await socket.push({'header': {'event': 'task-finished', 'task_id': first_id}})
    while len(socket.sent) < 3:
        await asyncio.sleep(0)
    second_id = json.loads(socket.sent[2])['header']['task_id']
    assert json.loads(socket.sent[2])['payload']['parameters']['vocabulary'] == {'new': 4}
    await socket.push({'header': {'event': 'task-started', 'task_id': second_id}})
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
            'header': {'event': 'result-generated', 'task_id': task_id},
            'payload': {'output': {'sentence': {'sentence_end': True, 'sentence_id': 1, 'text': 'last'}}},
        }
    )
    await socket.push({'header': {'event': 'task-finished', 'task_id': task_id}})
    await stopping
    event = await next_event(session)
    assert event.text == 'last'
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
        QWEN_ASR_STT_MODEL_AUDIO_STREAMING,
        STTRuntimeIntent,
        resolve_stt_config,
    )

    assert QWEN_ASR_STT_MODEL_AUDIO_STREAMING == QWEN_AUDIO_MODEL
    resolved = resolve_stt_config(
        STTRuntimeIntent(
            provider="qwen_asr",
            qwen_asr_model=QWEN_AUDIO_MODEL,
            qwen_region="singapore",
        )
    )
    assert resolved.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
    assert resolved.region == "singapore"


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
            provider="qwen_asr",
            source_language="tl",
            qwen_asr_model=QWEN_AUDIO_MODEL,
            qwen_region="singapore",
            custom_vocabulary_enabled=True,
            custom_terms={"tl": ("PuriPuly", "Qwen")},
        )
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_singapore", "test-key")
    backend = create_stt_backend_from_resolved_config(resolved, secrets=secrets)
    assert isinstance(backend, QwenAudioStreamingSTTBackend)
    assert backend.language == "tl"
    assert backend.endpoint.endswith("/api-ws/v1/inference")
    assert tuple(backend.hotwords) == ("PuriPuly", "Qwen")

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
            "payload": {"output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "done"}}},
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
            "payload": {"output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "first"}}},
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
            "payload": {"output": {"sentence": {"sentence_end": True, "sentence_id": 1, "text": "second"}}},
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
        language="ko",
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
        language="ko",
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=1,
        task_finish_timeout_s=0.2,
    )
    with pytest.raises(QwenAudioProtocolError, match="connection failed"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_heartbeat_result_does_not_create_transcript() -> None:
    _, session, socket, task_id = await open_fake()
    await socket.push(
        {
            "header": {"event": "result-generated", "task_id": task_id},
            "payload": {"output": {"sentence": {"heartbeat": True, "sentence_end": True, "sentence_id": 0}}},
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
    _, session, socket, _ = await open_fake(send_timeout_s=0.01)
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
