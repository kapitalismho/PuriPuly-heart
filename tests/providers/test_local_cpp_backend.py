from __future__ import annotations

import asyncio
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import puripuly_heart.app.adapters.asr_worker_process as asr_worker_process_module
from puripuly_heart.app.adapters.asr_worker_process import DefaultASRWorkerProcessFactory
from puripuly_heart.core.asr_worker import (
    ASRWorkerActivation,
    ASRWorkerClosedError,
    ASRWorkerDevice,
    ASRWorkerRequestError,
    ASRWorkerTranscription,
)
from puripuly_heart.core.lifecycle import LifecycleDiagnosticsUnavailableError
from puripuly_heart.core.local_asr.local_stt_assets import ORUKEET_MODEL_ID
from puripuly_heart.core.stt.backend import STTSessionProjection
from puripuly_heart.providers.stt.local_cpp import LocalCppSTTBackend
from tests.providers.test_local_cpu_backends import _scoped_request


class NativeWorker:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.is_closed = False
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.running = False
        self.calls = 0
        self.cancelled = []
        self.audio_paths = []
        self.activation = None
        self.events = asyncio.Queue()
        self.audio = []

    async def activate(self, **options):
        self.activation = options
        return ASRWorkerActivation(ASRWorkerDevice("cpu", 0, "CPU", "CPU", "cpu", 0, 0), 0.0, 0.0)

    async def transcribe(self, *, request_id, channel, audio_path, language_hint=None):
        assert audio_path.is_file()
        self.audio_paths.append(audio_path)
        assert channel == self.activation["channel"]
        with wave.open(str(audio_path), "rb") as wav:
            self.audio.append(
                (
                    wav.getparams(),
                    np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy(),
                )
            )
        self.calls += 1
        self.running = True
        self.started.set()
        try:
            await self.finish.wait()
            return ASRWorkerTranscription("completed speech", None, 0.1, 0.2, 2.0)
        finally:
            self.running = False

    async def cancel(self, request_id):
        self.cancelled.append(request_id)

    async def next_event(self):
        value = await self.events.get()
        raise ASRWorkerClosedError("closed") from value

    async def close(self):
        assert not self.running
        self.is_closed = True
        self.events.put_nowait(None)


class WorkerFactory:
    def __init__(self):
        self.workers = []

    async def start(self, **_options):
        worker = NativeWorker(1000 + len(self.workers))
        self.workers.append(worker)
        return worker

    async def close_pending_startups(self):
        return None


def backend(tmp_path, channel, factory, monkeypatch):
    instance = LocalCppSTTBackend(
        model_dir=tmp_path,
        model_id=ORUKEET_MODEL_ID,
        provider_id="local_orukeet",
        stream_label=channel,
        process_factory=factory,
    )
    monkeypatch.setattr(instance, "_validate_runtime_assets", lambda: None)
    return instance


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_cpu_worker_receives_ordered_mono_16khz_pcm16_signal(
    tmp_path: Path, monkeypatch, channel: str
):
    factory = WorkerFactory()
    provider = backend(tmp_path, channel, factory, monkeypatch)
    signals = [
        np.array([-1.0, 0.25, -0.75, 0.0, 1.0, -0.125, 0.75], dtype=np.float32),
        np.array([0.5, -0.25, 0.125, -0.5, 0.0], dtype=np.float32),
    ]
    expected_pcm = [
        [-32768, 8192, -24576, 0, 32767, -4096, 24576],
        [16384, -8192, 4096, -16384, 0],
    ]
    first = asyncio.create_task(provider.decode_f32(signals[0]))
    while not factory.workers:
        await asyncio.sleep(0)
    worker = factory.workers[0]
    worker.finish.set()
    try:
        assert await first == "completed speech"
        assert await provider.decode_f32(signals[1]) == "completed speech"
    finally:
        await provider.close()
    assert len(worker.audio) == len(expected_pcm)
    for turn, ((params, samples), expected) in enumerate(
        zip(worker.audio, expected_pcm, strict=True)
    ):
        assert (params.nchannels, params.framerate, params.sampwidth, params.comptype) == (
            1,
            16000,
            2,
            "NONE",
        )
        assert params.nframes == len(expected)
        np.testing.assert_allclose(
            samples,
            expected,
            rtol=0,
            atol=1,
            err_msg=f"CPU worker staged WAV signal mismatch: channel={channel}, turn={turn}",
        )


async def test_cpu_channels_load_lazily_and_remain_independent_during_stop(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    talk = backend(tmp_path, "self", factory, monkeypatch)
    listen = backend(tmp_path, "peer", factory, monkeypatch)
    assert factory.workers == []
    first = asyncio.create_task(talk.decode_f32(np.ones(160, dtype=np.float32)))
    while not factory.workers:
        await asyncio.sleep(0)
    talk_worker = factory.workers[0]
    await talk_worker.started.wait()
    assert len(factory.workers) == 1
    second = asyncio.create_task(listen.decode_f32(np.ones(160, dtype=np.float32)))
    while len(factory.workers) != 2:
        await asyncio.sleep(0)
    listen_worker = factory.workers[1]
    await listen_worker.started.wait()
    assert talk_worker.running and listen_worker.running
    assert talk_worker.activation["model_path"] == listen_worker.activation["model_path"]
    assert talk_worker.activation["n_threads"] == listen_worker.activation["n_threads"] == 2
    assert talk_worker.activation["backend"] == listen_worker.activation["backend"] == "cpu"
    closing = asyncio.create_task(talk.close())
    await asyncio.sleep(0)
    assert not closing.done()
    assert not talk_worker.is_closed
    listen_worker.finish.set()
    assert await second == "completed speech"
    assert not listen_worker.is_closed
    third = asyncio.create_task(listen.decode_f32(np.ones(160, dtype=np.float32)))
    assert await third == "completed speech"
    talk_worker.finish.set()
    await first
    await closing
    assert talk_worker.is_closed
    assert listen_worker.calls == 2
    assert not listen_worker.is_closed
    await listen.close()
    assert listen_worker.is_closed


async def test_cancelled_started_cpu_decode_drains_before_releasing_model_or_audio(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    provider = backend(tmp_path, "self", factory, monkeypatch)
    decode = asyncio.create_task(provider.decode_f32(np.ones(160, dtype=np.float32)))
    while not factory.workers:
        await asyncio.sleep(0)
    worker = factory.workers[0]
    await worker.started.wait()
    cancel_started = asyncio.Event()
    release_cancel = asyncio.Event()
    cancel = worker.cancel

    async def blocked_cancel(request_id):
        cancel_started.set()
        await release_cancel.wait()
        await cancel(request_id)

    monkeypatch.setattr(worker, "cancel", blocked_cancel)
    decode.cancel()
    closing = asyncio.create_task(provider.close())
    await asyncio.wait_for(cancel_started.wait(), timeout=1)
    decode.cancel()
    await asyncio.sleep(0)
    assert not decode.done()
    assert not closing.done()
    assert worker.audio_paths[0].is_file()
    assert not worker.is_closed
    release_cancel.set()
    while not worker.cancelled:
        await asyncio.sleep(0)
    decode.cancel()
    await asyncio.sleep(0)
    assert not decode.done()
    assert not closing.done()
    assert worker.audio_paths[0].is_file()
    assert len(worker.cancelled) == 1
    worker.finish.set()
    with pytest.raises(asyncio.CancelledError):
        await decode
    await closing
    assert worker.is_closed
    assert not worker.audio_paths[0].exists()


async def test_cpu_queued_cancel_does_not_compute_or_publish_the_retired_turn(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    provider = backend(tmp_path, "peer", factory, monkeypatch)
    session = await provider.open_session(projection=STTSessionProjection("scoped", "local-epoch"))
    worker = factory.workers[0]
    first, queued = _scoped_request(1), _scoped_request(2)
    for request in (first, queued):
        await session.begin_turn(request)
        await session.send_turn_audio(
            request.identity,
            b"\x00\x40" * 160,
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
    await worker.started.wait()
    await session.abort_turn(queued.identity, reason="revoked")
    events = session.turn_events()
    cancelled = await anext(events)
    assert cancelled.identity == queued.identity
    assert cancelled.outcome == "cancelled"
    worker.finish.set()
    final = await anext(events)
    assert final.identity == first.identity
    assert final.outcome == "final"
    await session.close()
    assert worker.calls == 1
    await provider.close()


async def _worker_event(events, name):
    async with asyncio.timeout(5):
        while True:
            event = await events.get()
            if event.name == name:
                return event


def _assert_native_closed(provider, client, audio_paths, factory):
    receipt = client.lifecycle_receipt()
    assert client.returncode is not None
    assert receipt["close_complete"] is True
    assert receipt["primary_code"] is None
    assert receipt["cleanup_failures"] == []
    assert all(
        receipt[name] is True
        for name in ("writer_closed", "server_closed", "tasks_closed", "temporary_directory_closed")
    )
    assert not client.temporary_directory_path.exists()
    assert not client._pending
    assert not client._native_requests
    assert client._scope.active_task_names == ()
    assert not provider.is_loaded
    assert provider.worker_pid is None
    assert provider._scope.active_task_names == ()
    assert all(not path.exists() for path in audio_paths)
    assert factory.startup_cleanup_snapshot()["pending_startups"] == 0


@pytest.mark.parametrize("outcome", ["success", "decode_failure", "cancel", "session_close"])
async def test_handled_native_cpu_outcomes_close_cleanly_and_allow_replacement(
    tmp_path: Path, monkeypatch, outcome: str
):
    existing_tasks = asyncio.all_tasks()
    for name in tuple(os.environ):
        if name.startswith("FAKE_ASR_WORKER_"):
            monkeypatch.delenv(name)
    if outcome == "success":
        monkeypatch.setenv("FAKE_ASR_WORKER_STARTED_SUCCESS", "1")
    elif outcome == "decode_failure":
        monkeypatch.setenv("FAKE_ASR_WORKER_STARTED_FAILURE", "1")
    gate = tmp_path / "release-cancel"
    monkeypatch.setenv("FAKE_ASR_WORKER_BLOCK_CANCEL", str(gate))
    factory = DefaultASRWorkerProcessFactory(
        executable_path=Path(__file__).parents[1] / "fixtures" / "fake_asr_worker.py",
        command_prefix=(sys.executable,),
        heartbeat_timeout_s=5,
    )
    client_type = asr_worker_process_module._DefaultASRWorkerClient
    transcribe = client_type.transcribe
    next_event = client_type.next_event
    audio_paths = []
    request_errors = []
    events = asyncio.Queue()

    async def observe_transcribe(client, **kwargs):
        audio_paths.append(kwargs["audio_path"])
        try:
            return await transcribe(client, **kwargs)
        except ASRWorkerRequestError as exc:
            request_errors.append(exc)
            raise

    async def observe_event(client):
        event = await next_event(client)
        events.put_nowait(event)
        return event

    monkeypatch.setattr(client_type, "transcribe", observe_transcribe)
    monkeypatch.setattr(client_type, "next_event", observe_event)
    provider = backend(tmp_path, "self", factory, monkeypatch)
    session = await provider.open_session()
    client = provider._recognizer
    operations = []
    try:
        if outcome == "cancel":
            request = asyncio.create_task(provider.decode_f32(np.ones(160, dtype=np.float32)))
            operations.append(request)
        else:
            await session.send_audio_f32(np.ones(160, dtype=np.float32))
            await session.on_speech_end()
        await _worker_event(events, "transcribe_started")
        if outcome in {"cancel", "session_close"}:
            if outcome == "cancel":
                request.cancel()
            else:
                closing_session = asyncio.create_task(session.close())
                operations.append(closing_session)
            cancelled = await _worker_event(events, "cancellation_requested")
            assert cancelled.fields["active"] is True
            if outcome == "cancel":
                request.cancel()
                await asyncio.sleep(0)
                request.cancel()
            closing_backend = asyncio.create_task(provider.close())
            operations.append(closing_backend)
            await asyncio.sleep(0)
            assert all(not operation.done() for operation in operations)
            assert all(path.is_file() for path in audio_paths)
            assert client.returncode is None
            gate.touch()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(request, timeout=5)
            else:
                await asyncio.wait_for(closing_session, timeout=5)
            await asyncio.wait_for(closing_backend, timeout=5)
            assert len(request_errors) == 1
            assert request_errors[0].code == "cancelled"
            assert request_errors[0].attempt_started is True
        else:
            transcript_events = session.events()
            event = await asyncio.wait_for(anext(transcript_events), timeout=5)
            assert event.is_final
            if outcome == "decode_failure":
                assert event.text == ""
                with pytest.raises(ASRWorkerRequestError) as error:
                    await asyncio.wait_for(anext(transcript_events), timeout=5)
                assert error.value is request_errors[0]
                assert error.value.code == "decode_failure"
                assert error.value.attempt_started is True
                assert error.value.fields == {
                    "audio_seconds": 1.0,
                    "decode_seconds": 0.25,
                    "rtf": 0.25,
                }
            else:
                assert event.text == "fixture"
                assert request_errors == []
            await transcript_events.aclose()
        await asyncio.wait_for(session.close(), timeout=5)
        await asyncio.wait_for(provider.close(), timeout=5)
        _assert_native_closed(provider, client, audio_paths, factory)
    finally:
        gate.touch()
        if operations:
            await asyncio.wait_for(asyncio.gather(*operations, return_exceptions=True), timeout=5)
        await asyncio.wait_for(session.close(), timeout=5)
        await asyncio.wait_for(provider.close(), timeout=5)
        await factory.close_pending_startups()

    monkeypatch.delenv("FAKE_ASR_WORKER_STARTED_FAILURE", raising=False)
    monkeypatch.setenv("FAKE_ASR_WORKER_STARTED_SUCCESS", "1")
    replacement = backend(tmp_path, "self", factory, monkeypatch)
    replacement_session = await replacement.open_session()
    replacement_client = replacement._recognizer
    try:
        await replacement_session.send_audio_f32(np.ones(160, dtype=np.float32))
        await replacement_session.on_speech_end()
        transcript_events = replacement_session.events()
        event = await asyncio.wait_for(anext(transcript_events), timeout=5)
        assert event.text == "fixture"
        await transcript_events.aclose()
    finally:
        await asyncio.wait_for(replacement_session.close(), timeout=5)
        await asyncio.wait_for(replacement.close(), timeout=5)
        await factory.close_pending_startups()
    _assert_native_closed(replacement, replacement_client, audio_paths, factory)
    await asyncio.sleep(0)
    assert not asyncio.all_tasks() - existing_tasks


async def test_handled_native_cpu_activation_failure_is_not_replayed_during_close(
    tmp_path: Path, monkeypatch
):
    existing_tasks = asyncio.all_tasks()
    for name in tuple(os.environ):
        if name.startswith("FAKE_ASR_WORKER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FAKE_ASR_WORKER_ACTIVATION_FAILURE", "1")
    factory = DefaultASRWorkerProcessFactory(
        executable_path=Path(__file__).parents[1] / "fixtures" / "fake_asr_worker.py",
        command_prefix=(sys.executable,),
        heartbeat_timeout_s=5,
    )
    client_type = asr_worker_process_module._DefaultASRWorkerClient
    activate = client_type.activate
    clients = []
    request_errors = []

    async def observe_activate(client, **kwargs):
        clients.append(client)
        try:
            return await activate(client, **kwargs)
        except ASRWorkerRequestError as exc:
            request_errors.append(exc)
            raise

    monkeypatch.setattr(client_type, "activate", observe_activate)
    provider = backend(tmp_path, "self", factory, monkeypatch)
    try:
        with pytest.raises(ASRWorkerRequestError) as error:
            await asyncio.wait_for(provider.open_session(), timeout=5)
        assert len(clients) == 1
        assert len(request_errors) == 1
        assert error.value is request_errors[0]
        assert error.value.code == "out_of_memory"
        assert error.value.attempt_started is False
        assert error.value.fields == {}
        await asyncio.wait_for(provider.close(), timeout=5)
        _assert_native_closed(provider, clients[0], [], factory)
    finally:
        await asyncio.wait_for(provider.close(), timeout=5)
        await factory.close_pending_startups()
    await asyncio.sleep(0)
    assert not asyncio.all_tasks() - existing_tasks


async def test_native_cpu_resource_cleanup_failure_remains_owned_and_retryable(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    provider = backend(tmp_path, "self", factory, monkeypatch)
    session = await provider.open_session()
    worker = factory.workers[0]
    close = worker.close
    failure = OSError("controlled resource cleanup failure")

    async def fail_close():
        raise failure

    monkeypatch.setattr(worker, "close", fail_close)
    try:
        with pytest.raises(OSError) as error:
            await provider.close()
        assert error.value is failure
        assert provider.is_loaded
        assert not worker.is_closed
    finally:
        monkeypatch.setattr(worker, "close", close)
        await session.close()
        await provider.close()
    assert not provider.is_loaded
    assert worker.is_closed


async def test_native_cpu_unexpected_event_failure_remains_a_background_close_error(
    tmp_path: Path, monkeypatch
):
    factory = WorkerFactory()
    provider = backend(tmp_path, "self", factory, monkeypatch)
    failed = asyncio.Event()

    async def fail_events(worker):
        failed.set()
        raise ValueError("controlled background event failure")

    monkeypatch.setattr(NativeWorker, "next_event", fail_events)
    session = await provider.open_session()
    worker = factory.workers[0]
    await asyncio.wait_for(failed.wait(), timeout=1)
    await session.close()
    with pytest.raises(LifecycleDiagnosticsUnavailableError) as error:
        await provider.close()
    assert worker.is_closed
    assert not provider.is_loaded
    assert len(error.value.diagnostics) == 1
    assert error.value.diagnostics[0].fields["task_name"] == "events"
    assert error.value.diagnostics[0].fields["exception_class"] == "ValueError"
    await provider.close()
