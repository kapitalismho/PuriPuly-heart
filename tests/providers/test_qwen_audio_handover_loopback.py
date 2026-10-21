from __future__ import annotations

import asyncio
import hashlib
import json
from uuid import uuid4

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTSessionProjection
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from tests.core.test_self_translation_low_latency import FakeLLMProvider
from tests.core.test_stt_scoped_engine import settings, span, wait_until, watchdogs
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import control, native


def local_inputs(ledger):
    result = []
    expected = []
    for index in range(3):
        start_sample = 16000 + index * 32000
        context_count = 8000 if index == 0 else 4800
        context = np.full(context_count, 0.125, dtype=np.float32)
        first = np.full(8000, (index + 1) / 10, dtype=np.float32)
        second = np.full(8000, 0.125, dtype=np.float32)
        identity = uuid4()
        start = ledger.observe_vad_event(
            SpeechStart(
                identity,
                context,
                first,
                pre_roll_capture=(span(index * 3, start_sample - context_count, start_sample),),
                chunk_capture=(span(index * 3 + 1, start_sample, start_sample + 8000),),
            ),
            now_monotonic_s=float(index),
        )
        chunk = ledger.observe_vad_event(
            SpeechChunk(
                identity,
                second,
                chunk_capture=(span(index * 3 + 2, start_sample + 8000, start_sample + 16000),),
            ),
            now_monotonic_s=index + 0.5,
        )
        end = ledger.observe_vad_event(
            SpeechEnd(identity, trailing_silence_ms=224, reason="silence"),
            now_monotonic_s=index + 1.0,
        )
        result.append((start, chunk, end))
        expected.append(
            b"".join(float32_to_pcm16le_bytes(value) for value in (context, first, second))
        )
    return result, expected


async def run_comparison(channel, overlapping):
    trace = []
    tasks = []
    connections = []
    finishers = set()
    loop = asyncio.get_running_loop()
    started = loop.time()

    def mark(event, task):
        trace.append((event, task["order"], round(loop.time() - started, 6)))

    async def finish(socket, task):
        await asyncio.sleep(0.06 if task["order"] == 1 else 0.005)
        await socket.send(
            json.dumps(native(task["id"], 2, " tail ", begin_time=500, end_time=1000))
        )
        mark("terminal", task)
        await socket.send(json.dumps(control(task["id"], "task-finished")))

    async def handler(socket):
        connections.append(socket)
        task = None
        async for message in socket:
            if isinstance(message, bytes):
                assert task is not None
                task["pcm"].append(message)
                mark("pcm", task)
                if len(task["pcm"]) == 1:
                    await socket.send(
                        json.dumps(native(task["id"], 1, " same ", begin_time=0, end_time=500))
                    )
                continue
            packet = json.loads(message)
            if packet["header"]["action"] == "run-task":
                task = {"id": packet["header"]["task_id"], "order": len(tasks) + 1, "pcm": []}
                tasks.append(task)
                mark("run", task)
                await socket.send(json.dumps(control(task["id"], "task-started")))
                mark("ready", task)
            else:
                mark("finish", task)
                pending = asyncio.create_task(finish(socket, task))
                finishers.add(pending)
                pending.add_done_callback(finishers.discard)

    llm = FakeLLMProvider(delay_s=0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        source_language="en",
        target_language="ko",
        peer_source_language="en",
        peer_target_language="ko",
        low_latency_mode=False,
        peer_translation_enabled=True,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    inputs, expected = local_inputs(ledger)
    terminals = []

    async def receive(event):
        if not isinstance(event, STTProviderTurnTerminal):
            if channel == "self":
                await harness.self_owner.handle_stt_event(event)
            else:
                pytest.fail("Peer published before local task completion")
            return
        terminals.append(event)
        if channel == "self":
            await harness.self_owner.handle_stt_event(event)
        else:
            receipt = ledger.terminalize(
                event.identity.segment.segment_id,
                outcome=event.outcome,
                now_monotonic_s=5,
                provider_epoch_id=event.identity.provider_epoch_id,
                provider_turn_id=event.identity.provider_turn_id,
                text_authority=event.text_authority,
            )
            await harness.peer_owner.handle_provider_turn_terminal(receipt, event)

    async with serve(handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/controlled/{channel}"
        backend = QwenAudioStreamingSTTBackend(
            api_key="offline",
            channel=channel,
            endpoint=endpoint,
            language_hints=("en",),
            sample_rate_hz=16000,
            hotwords=(),
            websocket_factory=connect,
            task_finish_timeout_s=1,
        )
        engine = ScopedRecognitionEngine(
            channel=channel,
            session_factory=lambda _settings, epoch: backend.open_session(
                projection=STTSessionProjection("scoped", epoch)
            ),
            event_sink=receive,
            watchdog_resolver=lambda _: watchdogs(write_timeout_s=1, final_timeout_s=1),
        )
        try:
            await harness.start()
            harness.output_runtime.activate_peer_generation(1)
            for index, segment in enumerate(inputs, 1):
                for event in segment:
                    await engine.handle_owned_vad_event(event)
                if not overlapping:
                    await wait_until(lambda: len(terminals) == index)
            await wait_until(lambda: len(terminals) == 3)
            await harness.translation_turns.wait_for_idle()
            wire = [b"".join(task["pcm"]) for task in tasks]
            assert wire == expected
            assert len(tasks) == 3 and len(connections) == 2
            assert [terminal.text for terminal in terminals] == [" same  tail "] * 3
            assert (
                len(llm.calls) == 3 and [call["text"] for call in llm.calls] == [" same  tail "] * 3
            )
            assert all(len(terminal.recognition_origins) == 2 for terminal in terminals)
            a_finish = next(at for event, order, at in trace if event == "finish" and order == 1)
            a_terminal = next(
                at for event, order, at in trace if event == "terminal" and order == 1
            )
            b_pcm = next(at for event, order, at in trace if event == "pcm" and order == 2)
            assert a_finish < b_pcm
            assert (b_pcm < a_terminal) is overlapping
            return {
                "channel": channel,
                "overlapping": overlapping,
                "connections": len(connections),
                "tasks": len(tasks),
                "input_bytes": [len(pcm) for pcm in wire],
                "pcm_sha256": [hashlib.sha256(pcm).hexdigest() for pcm in wire],
                "completed_segments": len(terminals),
                "native_contributions": sum(len(t.recognition_origins) for t in terminals),
                "translation_calls": len(llm.calls),
                "trace": trace,
            }
        finally:
            await engine.close()
            await harness.stop()
            if finishers:
                await asyncio.gather(*finishers)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_real_websocket_owner_handover_matches_serial_local_input_without_extra_work(channel):
    serial = await run_comparison(channel, False)
    handover = await run_comparison(channel, True)
    for field in (
        "tasks",
        "input_bytes",
        "pcm_sha256",
        "completed_segments",
        "native_contributions",
        "translation_calls",
    ):
        assert serial[field] == handover[field]
    print(
        "QWEN_HANDOVER_SAME_INPUT "
        + json.dumps({"serial": serial, "handover": handover}, sort_keys=True)
    )
