from __future__ import annotations

import asyncio
import json
import wave

import pytest

import scripts.qwen_audio_live as harness
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend

from scripts.qwen_audio_live import (
    RecordingWebSocket,
    _chunks,
    _comparison_pcm,
    _fit_pcm,
    _measure_boundary_to_terminal,
    _nested_blockers,
    _pcm_for_duration,
    _pcm_identity,
    _read_fixture,
    _record_sent_identity,
    _redacted_command,
    _task_records,
    _usage_duration,
    build_cases,
    sanitize_report,
)


def _message(
    event: str,
    task_id: str,
    *,
    usage: float | None = None,
    text: str = "",
    sentence_end: bool = True,
) -> str:
    payload: dict[str, object] = {}
    if usage is not None:
        payload["usage"] = {"duration": usage}
    if text:
        payload["output"] = {
            "sentence": {"sentence_end": sentence_end, "text": text}
        }
    return json.dumps({"header": {"event": event, "task_id": task_id}, "payload": payload})


def test_build_cases_covers_required_repeated_and_conversational_durations() -> None:
    assert [(case["name"], case["duration_s"]) for case in build_cases()] == [
        ("short_0.3s_repeat_1", 0.3),
        ("short_0.3s_repeat_2", 0.3),
        ("short_0.8s_repeat_1", 0.8),
        ("short_0.8s_repeat_2", 0.8),
        ("short_1.2s_repeat_1", 1.2),
        ("short_1.2s_repeat_2", 1.2),
        ("conversational_2s", 2.0),
        ("conversational_4s", 4.0),
        ("conversational_7s", 7.0),
    ]


def test_pcm_generation_is_deterministic_and_duration_exact() -> None:
    first = _pcm_for_duration(0.3)
    second = _pcm_for_duration(0.3)
    assert first == second
    assert len(first) == 0.3 * 16000 * 2


def test_task_records_capture_ids_timestamps_usage_and_transcript() -> None:
    task_id = "task-1"
    socket = RecordingWebSocket(inner=None)
    run_task = json.dumps({"header": {"action": "run-task", "task_id": task_id}})
    finish_task = json.dumps({"header": {"action": "finish-task", "task_id": task_id}})
    socket.sent = [(1.0, 100.0, run_task), (2.0, 101.0, finish_task)]
    socket.received = [
        (1.1, 100.1, _message("task-started", task_id)),
        (1.5, 100.5, _message("result-generated", task_id, text="중간", sentence_end=False)),
        (1.9, 100.9, _message("result-generated", task_id, text="안녕하세요")),
        (2.1, 101.1, _message("task-finished", task_id, usage=0.8, text="무시")),
        (2.2, 101.2, _message("result-generated", task_id, usage=9.0, text="늦은 결과")),
    ]
    records = _task_records(socket)
    assert len(records) == 1
    record = records[0]
    assert record.task_id == task_id
    assert record.run_task_sent_at == "1970-01-01T00:01:40+00:00"
    assert record.task_started_at == "1970-01-01T00:01:40.100000+00:00"
    assert record.finish_task_sent_at == "1970-01-01T00:01:41+00:00"
    assert record.task_finished_at == "1970-01-01T00:01:41.100000+00:00"
    assert record.actual_task_duration_s == 1.0
    assert record.terminal_count == 1
    assert record.transition_latency_s is None
    assert record.usage_duration == 0.8
    assert record.terminal_text == "안녕하세요"
    assert "task_id" not in record.as_dict()


def test_usage_duration_is_extracted_from_nested_provider_payload() -> None:
    usage, raw = _usage_duration(
        {
            "header": {"task_id": "secret-id"},
            "payload": {"output": {"usage": {"duration": "1.2"}}},
        }
    )
    assert usage == 1.2
    assert raw == "1.2"




def test_split_records_conserve_exact_pcm_bytes_and_chunks() -> None:
    first = b"A" * (7 * 16000 * 2)
    second = b"B" * (200 * 16000 * 2 // 1000)
    socket = RecordingWebSocket(inner=None)
    first_id, second_id = "task-1", "task-2"
    socket.sent = [
        (1.0, 100.0, json.dumps({"header": {"action": "run-task", "task_id": first_id}})),
        *[(1.1, 100.1, chunk) for chunk in _chunks(first)],
        (2.0, 101.0, json.dumps({"header": {"action": "finish-task", "task_id": first_id}})),
        (3.0, 102.0, json.dumps({"header": {"action": "run-task", "task_id": second_id}})),
        *[(3.1, 102.1, chunk) for chunk in _chunks(second)],
        (4.0, 103.0, json.dumps({"header": {"action": "finish-task", "task_id": second_id}})),
    ]
    socket.received = [
        (1.2, 100.2, _message("task-started", first_id)),
        (2.1, 101.1, _message("result-generated", first_id, text="first")),
        (2.2, 101.2, _message("task-finished", first_id)),
        (3.2, 102.2, _message("task-started", second_id)),
        (4.1, 103.1, _message("result-generated", second_id, text="second")),
        (4.2, 103.2, _message("task-finished", second_id)),
    ]
    records = _task_records(socket)
    expected = [_pcm_identity(_chunks(first)), _pcm_identity(_chunks(second))]
    assert [record.pcm_bytes_sent for record in records] == [len(first), len(second)]
    assert [record.pcm_chunks_sent for record in records] == [len(_chunks(first)), len(_chunks(second))]
    assert [record.pcm_sent_sha256 for record in records] == [item["sha256"] for item in expected]
    assert [record.pcm_sent_frame_sha256 for record in records] == [item["frame_sha256"] for item in expected]
    assert [record.as_dict()["pcm_sent_sha256"] for record in records] == [item["sha256"] for item in expected]
    assert [record.terminal_count for record in records] == [1, 1]

def test_resampling_and_chunking_are_deterministic(tmp_path) -> None:
    fixture_path = tmp_path / "korean.wav"
    source = b"\x00\x00" * 2205
    with wave.open(str(fixture_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        audio.writeframes(source)
    pcm = _read_fixture(fixture_path)
    assert len(pcm) == 1600 * 2
    assert sum(len(chunk) for chunk in _chunks(pcm)) == len(pcm)
    assert _comparison_pcm(pcm) == _fit_pcm(pcm, 0.8)


@pytest.mark.asyncio
async def test_common_boundary_metric_starts_at_local_boundary_after_pcm_send() -> None:
    order: list[str] = []

    async def boundary() -> None:
        order.append("boundary")

    async def terminal() -> str:
        order.append("terminal")
        return "final"

    event, elapsed = await _measure_boundary_to_terminal(boundary, terminal)
    assert event == "final"
    assert elapsed >= 0
    assert order == ["boundary", "terminal"]


def test_default_invocation_is_dry_and_does_not_read_secrets_or_network(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("live dependency touched during dry run")

    monkeypatch.setattr(harness, "_secret_for_region", fail)
    monkeypatch.setattr(harness.RecordingFactory, "connect", fail)
    report = __import__("asyncio").run(harness.run(["--region", "beijing"]))
    assert report["mode"] == "dry-run"
    assert report["credential_sources"] == {"beijing": "not_read"}
    assert report["regions"] == {}
    assert report["acceptance"]["smoke"]["status"] == "not_run"
    assert report["acceptance"]["billing"]["status"] == "blocked"


def test_nested_failures_and_blockers_are_reportable() -> None:
    blockers = _nested_blockers(
        {
            "status": "measured",
            "qwen3_comparison": {"status": "blocked", "error": {"type": "Exception"}},
            "hotword_on": {"status": "failed", "failures": [{"type": "TimeoutError"}]},
        }
    )
    assert "qwen3_comparison status=blocked" in blockers
    assert "hotword_on status=failed" in blockers
    assert "hotword_on.failures present" in blockers


@pytest.mark.asyncio
async def test_required_child_failures_block_root_acceptance(monkeypatch) -> None:
    async def fake_audio(*args, **kwargs):
        return {
            "region": "beijing",
            "status": "measured",
            "split_byte_accounting": {"conserved": False},
            "failures": [],
        }

    async def fake_qwen3(*args, **kwargs):
        return {"status": "blocked", "error": {"type": "TimeoutError"}}

    async def fake_hotword(*args, **kwargs):
        return {"status": "failed", "error": {"type": "TimeoutError"}}

    monkeypatch.setattr(harness, "_secret_for_region", lambda region: ("test-key", "test"))
    monkeypatch.setattr(harness, "_run_audio_region", fake_audio)
    monkeypatch.setattr(harness, "_run_qwen3_compare", fake_qwen3)
    monkeypatch.setattr(harness, "_run_hotword_transport_case", fake_hotword)
    report = await harness.run(["--live", "--region", "beijing"])
    assert report["acceptance"]["smoke"]["status"] == "blocked"
    assert report["acceptance"]["split_byte_conservation"]["status"] == "blocked"
    assert report["acceptance"]["qwen3_comparison"]["status"] == "blocked"
    assert report["acceptance"]["hotword_transport"]["status"] == "blocked"
    assert report["regions"]["beijing"]["status"] == "blocked"
    assert report["blocked"]


@pytest.mark.asyncio
async def test_hotword_only_region_status_aggregates_child_failure(monkeypatch) -> None:
    outcomes = iter(
        [
            {"status": "measured_transport"},
            {"status": "failed", "error": {"type": "TimeoutError"}},
        ]
    )

    async def fake_hotword(*args, **kwargs):
        return next(outcomes)

    monkeypatch.setattr(harness, "_secret_for_region", lambda region: ("test-key", "test"))
    monkeypatch.setattr(harness, "_run_hotword_transport_case", fake_hotword)
    report = await harness.run(["--live", "--hotword-only", "--region", "beijing"])
    assert report["regions"]["beijing"]["status"] == "blocked"
    assert report["acceptance"]["hotword_transport"]["status"] == "blocked"
    assert report["acceptance"]["smoke"]["status"] == "not_run"


@pytest.mark.asyncio
async def test_abort_failure_blocks_reconnect_and_smoke(monkeypatch) -> None:
    async def fake_reconnect(*args, **kwargs):
        return {
            "status": "failed",
            "abort_status": "failed",
            "first_outcome": {"type": "TimeoutError"},
        }

    monkeypatch.setattr(harness, "_secret_for_region", lambda region: ("test-key", "test"))
    monkeypatch.setattr(harness, "_run_reconnect_probe", fake_reconnect)
    report = await harness.run(["--live", "--reconnect-only", "--region", "beijing"])
    assert report["acceptance"]["reconnect"]["status"] == "blocked"
    assert report["acceptance"]["smoke"]["status"] == "not_run"
    assert any("abort_status=failed" in item for item in report["blocked"])


def test_sanitizer_removes_hostile_identifiers_credentials_errors_and_transcripts() -> None:
    hostile = {
        "header": {"authorization": "Bearer live-secret", "task_id": "12345678-1234-1234-1234-123456789abc"},
        "api_key": "sk-abcdefghijklmnop",
        "error": {"type": "ValueError", "message": "secret transcript"},
        "terminal_text": "안녕하세요",
        "nested": {"transcript": "비밀", "value": "task_0123456789abcdef", "other_id": "session-deadbeef00112233"},
    }
    safe = sanitize_report(hostile)
    encoded = json.dumps(safe, ensure_ascii=False)
    assert "authorization" not in encoded
    assert "live-secret" not in encoded
    assert "task_id" not in encoded
    assert "task_0123456789abcdef" not in encoded
    assert "session-deadbeef00112233" not in encoded
    assert "secret transcript" not in encoded
    assert "안녕하세요" not in encoded
    assert "terminal_text" not in safe
    assert safe["error"] == {"type": "ValueError"}
    retained = sanitize_report({"terminal_text": "안녕하세요"}, retain_transcripts=True)
    assert retained == {"terminal_text": "안녕하세요"}


@pytest.mark.asyncio
async def test_fake_qwen_audio_websocket_split_assigns_exact_bytes_to_tasks() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[object] = []
            self.incoming: asyncio.Queue[object] = asyncio.Queue()
            self.closed = False

        async def send(self, value: object) -> None:
            self.sent.append(value)

        async def recv(self) -> object:
            return await self.incoming.get()

        async def close(self) -> None:
            self.closed = True
            await self.incoming.put(None)

        async def push(self, value: object) -> None:
            await self.incoming.put(value)

    fake = FakeWebSocket()
    recording_factory = harness.RecordingFactory()

    async def connect(*args: object, **kwargs: object) -> harness.RecordingWebSocket:
        recording_factory.socket = harness.RecordingWebSocket(fake)
        return recording_factory.socket

    backend = QwenAudioStreamingSTTBackend(
        api_key="test-key",
        language="ko",
        websocket_factory=connect,
        connect_timeout_s=1,
        task_start_timeout_s=1,
        task_finish_timeout_s=1,
    )

    async def wait_until(predicate) -> None:
        for _ in range(2000):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("fake Qwen Audio operation did not reach expected state")

    opening = asyncio.create_task(backend.open_session())
    await wait_until(lambda: bool(fake.sent))
    first_id = json.loads(fake.sent[0])["header"]["task_id"]
    await fake.push({"header": {"event": "task-started", "task_id": first_id}})
    session = await opening
    events = session.events().__aiter__()

    first = _pcm_for_duration(7.0)
    second = _pcm_for_duration(0.2)
    for chunk in _chunks(first):
        await session.send_audio(chunk)
    await session.on_speech_end()
    for chunk in _chunks(second):
        await session.send_audio(chunk)
    await fake.push(
        {
            "header": {"event": "result-generated", "task_id": first_id},
            "payload": {"output": {"sentence": {"sentence_id": "1", "sentence_end": True, "text": "first"}}},
        }
    )
    await fake.push({"header": {"event": "task-finished", "task_id": first_id}})
    first_event = await events.__anext__()
    assert first_event.text == "first"

    await wait_until(
        lambda: sum(
            1
            for value in fake.sent
            if isinstance(value, str)
            and json.loads(value).get("header", {}).get("action") == "run-task"
        )
        >= 2
    )
    run_tasks = [
        json.loads(value)
        for value in fake.sent
        if isinstance(value, str)
        and json.loads(value).get("header", {}).get("action") == "run-task"
    ]
    second_id = run_tasks[1]["header"]["task_id"]
    await fake.push({"header": {"event": "task-started", "task_id": second_id}})
    await wait_until(
        lambda: sum(len(value) for value in fake.sent if isinstance(value, bytes))
        >= len(first) + len(second)
    )
    await session.on_speech_end()
    await fake.push(
        {
            "header": {"event": "result-generated", "task_id": second_id},
            "payload": {"output": {"sentence": {"sentence_id": "1", "sentence_end": True, "text": "second"}}},
        }
    )
    await fake.push({"header": {"event": "task-finished", "task_id": second_id}})
    second_event = await events.__anext__()
    assert second_event.text == "second"
    await session.abort_for_toggle_off()

    assert recording_factory.socket is not None
    records = _task_records(recording_factory.socket)
    expected = [_pcm_identity(_chunks(first)), _pcm_identity(_chunks(second))]
    sent = [_record_sent_identity(record) for record in records[:2]]
    assert sent == expected
    frames = [b"aa", b"bb", b"cc"]
    assert _pcm_identity(frames[:2]) != _pcm_identity(frames)
    assert _pcm_identity([*frames, frames[-1]]) != _pcm_identity(frames)
    assert _pcm_identity([frames[1], frames[0], frames[2]]) != _pcm_identity(frames)


def test_command_redaction_hides_credential_arguments_in_both_forms() -> None:
    assert _redacted_command(["runner", "--api-key", "secret"]) == [
        "runner",
        "<redacted>",
        "<redacted>",
    ]
    assert _redacted_command(
        [
            "--password=one",
            "--TOKEN",
            "two",
            "--secret=three",
            "--HEADER",
            "four",
            "--authorization=five",
        ]
    ) == ["<redacted>"] * 7
    assert _redacted_command(["--task-timeout", "20"]) == ["--task-timeout", "20"]
