from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import puripuly_heart.core.runtime.gpu_asr as gpu_asr_module
from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerClosedError,
    GpuWorkerDevice,
    GpuWorkerEvent,
    GpuWorkerMode,
    GpuWorkerRequestError,
    GpuWorkerTranscription,
)
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime.gpu_asr import (
    GpuASRChannel,
    GpuASRDecodeDropped,
    GpuASRManualRetryRequired,
    GpuASRRuntimeState,
    GpuASRWorkDiscarded,
    GpuASRWorkExpired,
    GpuDiscoveryState,
    SharedGpuASRRuntime,
)
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend

DEVICE = GpuWorkerDevice(
    device_id="vulkan:0",
    registry_index=0,
    name="Test GPU",
    description="Test GPU",
    device_type="discrete",
    memory_total_bytes=8_000_000_000,
    memory_free_bytes=4_000_000_000,
)
ACTIVATION = GpuWorkerActivation(
    device=DEVICE,
    model_load_seconds=1.25,
    warmup_seconds=0.5,
)


class FakeGpuWorkerClient:
    def __init__(
        self,
        *,
        activation_error: GpuWorkerRequestError | None = None,
        activation_gate: asyncio.Event | None = None,
        discovery_gate: asyncio.Event | None = None,
        transcribe_gate: asyncio.Event | None = None,
        transcribe_error: BaseException | None = None,
        send_error: BaseException | None = None,
    ) -> None:
        self.activation_error = activation_error
        self.activation_gate = activation_gate
        self.discovery_gate = discovery_gate
        self.transcribe_gate = transcribe_gate
        self.transcribe_error = transcribe_error
        self.send_error = send_error
        self.activate_calls: list[tuple[Path, str]] = []
        self.transcribe_calls: list[tuple[str, str, str | None]] = []
        self.cancel_calls: list[str] = []
        self.close_calls = 0
        self.started = asyncio.Event()
        self.transcribe_returning = asyncio.Event()
        self.activation_started = asyncio.Event()
        self._events: asyncio.Queue[GpuWorkerEvent | None] = asyncio.Queue()
        self._closed = False
        self._cancelled_request_ids: set[str] = set()

    @property
    def pid(self) -> int | None:
        return 1234

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def discover(self) -> tuple[GpuWorkerDevice, ...]:
        if self.discovery_gate is not None:
            await self.discovery_gate.wait()
        return (DEVICE,)

    async def activate(self, *, model_path: Path, device_id: str) -> GpuWorkerActivation:
        self.activate_calls.append((model_path, device_id))
        self.activation_started.set()
        if self.activation_gate is not None:
            await self.activation_gate.wait()
        if self.activation_error is not None:
            raise self.activation_error
        return ACTIVATION

    async def transcribe(
        self,
        *,
        request_id: str,
        channel: str,
        audio_path: Path,
        language_hint: str | None = None,
        on_request_sent: Callable[[], None] | None = None,
    ) -> GpuWorkerTranscription:
        assert audio_path.is_file()
        if self.send_error is not None:
            raise self.send_error
        if on_request_sent is not None:
            on_request_sent()
        self.transcribe_calls.append((request_id, channel, language_hint))
        self.started.set()
        if self.transcribe_gate is not None:
            await self.transcribe_gate.wait()
        if request_id in self._cancelled_request_ids:
            raise GpuWorkerRequestError(
                "cancelled",
                {
                    "audio_seconds": 0.1,
                    "decode_seconds": 0.025,
                    "rtf": 0.25,
                },
                attempt_started=True,
            )
        if self._closed:
            raise GpuWorkerClosedError("closed")
        if self.transcribe_error is not None:
            raise self.transcribe_error
        self.transcribe_returning.set()
        return GpuWorkerTranscription(
            text=f"{channel}-{len(self.transcribe_calls)}",
            detected_language="en",
            audio_seconds=0.1,
            decode_seconds=0.05,
            rtf=0.5,
        )

    async def cancel(self, target_request_id: str) -> None:
        self.cancel_calls.append(target_request_id)
        self._cancelled_request_ids.add(target_request_id)
        if self.transcribe_gate is not None:
            self.transcribe_gate.set()

    async def next_event(self) -> GpuWorkerEvent:
        event = await self._events.get()
        if event is None:
            raise GpuWorkerClosedError("closed")
        return event

    async def close(self) -> None:
        self.close_calls += 1
        if self._closed:
            return
        self._closed = True
        if self.transcribe_gate is not None:
            self.transcribe_gate.set()
        await self._events.put(None)

    async def force_close(self) -> None:
        await self.close()


class FakeGpuWorkerFactory:
    def __init__(self, clients: list[FakeGpuWorkerClient]) -> None:
        self.clients = clients
        self.modes: list[GpuWorkerMode] = []

    async def start(self, *, mode: GpuWorkerMode) -> FakeGpuWorkerClient:
        self.modes.append(mode)
        return self.clients.pop(0)


async def _activate(
    runtime: SharedGpuASRRuntime, channel: GpuASRChannel = "self"
) -> GpuWorkerActivation:
    return await runtime.activate_channel(
        channel,
        model_path=Path("model.gguf"),
        model_id="qwen-gpu",
        device_id="vulkan:0",
    )


async def test_one_worker_and_model_are_shared_until_last_channel_deactivates() -> None:
    client = FakeGpuWorkerClient()
    factory = FakeGpuWorkerFactory([client])
    runtime = SharedGpuASRRuntime(process_factory=factory)

    first = await _activate(runtime, "self")
    second = await _activate(runtime, "peer")

    assert first == second == ACTIVATION
    assert factory.modes == ["persistent"]
    assert len(client.activate_calls) == 1
    assert runtime.active_channels == frozenset({"self", "peer"})

    await runtime.deactivate_channel("self")
    assert client.close_calls == 0
    assert runtime.state == GpuASRRuntimeState.READY

    await runtime.deactivate_channel("peer")
    assert client.close_calls == 1
    assert runtime.state == GpuASRRuntimeState.STOPPED
    await runtime.close()


@pytest.mark.parametrize(
    ("stop_order", "restore_order"),
    [
        (("self", "peer"), ("self", "peer")),
        (("self", "peer"), ("peer", "self")),
        (("peer", "self"), ("self", "peer")),
        (("peer", "self"), ("peer", "self")),
    ],
)
async def test_device_change_after_full_quiesce_accepts_both_channel_orders(
    stop_order: tuple[GpuASRChannel, GpuASRChannel],
    restore_order: tuple[GpuASRChannel, GpuASRChannel],
) -> None:
    old_client = FakeGpuWorkerClient()
    new_client = FakeGpuWorkerClient()
    runtime = SharedGpuASRRuntime(process_factory=FakeGpuWorkerFactory([old_client, new_client]))
    await _activate(runtime, "self")
    await _activate(runtime, "peer")

    for channel in stop_order:
        await runtime.deactivate_channel(channel)
    for channel in restore_order:
        await runtime.activate_channel(
            channel,
            model_path=Path("model.gguf"),
            model_id="qwen-gpu",
            device_id="vulkan:1",
        )

    assert old_client.close_calls == 1
    assert new_client.activate_calls == [(Path("model.gguf").resolve(), "vulkan:1")]
    assert runtime.active_channels == frozenset({"self", "peer"})
    assert runtime.configured_device_id == "vulkan:1"
    await runtime.close()


async def test_concurrent_channels_share_one_owned_activation_task() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(activation_gate=gate)
    factory = FakeGpuWorkerFactory([client])
    runtime = SharedGpuASRRuntime(process_factory=factory)

    self_activation = asyncio.create_task(_activate(runtime, "self"))
    await client.activation_started.wait()
    peer_activation = asyncio.create_task(_activate(runtime, "peer"))
    await asyncio.sleep(0)

    assert factory.modes == ["persistent"]
    assert len(client.activate_calls) == 1
    assert runtime.active_channels == frozenset({"self", "peer"})

    gate.set()
    assert await self_activation == ACTIVATION
    assert await peer_activation == ACTIVATION
    await runtime.close()


async def test_close_cancels_and_awaits_in_progress_activation() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(activation_gate=gate)
    runtime = SharedGpuASRRuntime(process_factory=FakeGpuWorkerFactory([client]))

    activation = asyncio.create_task(_activate(runtime))
    await client.activation_started.wait()

    await asyncio.wait_for(runtime.close(), timeout=0.5)

    with pytest.raises(asyncio.CancelledError):
        await activation
    assert client.close_calls == 1


async def test_activation_progress_is_observed_before_activation_completes() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(activation_gate=gate)
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        diagnostic_sink=diagnostics.append,
    )

    activation = asyncio.create_task(_activate(runtime))
    await client.activation_started.wait()
    await client._events.put(
        GpuWorkerEvent(
            name="activation_progress",
            request_id=None,
            fields={"phase": "loading", "progress": 0.25},
        )
    )
    for _ in range(20):
        if diagnostics:
            break
        await asyncio.sleep(0)

    assert diagnostics[-1].kind == "worker_lifecycle"
    assert diagnostics[-1].fields["phase"] == "loading"
    assert activation.done() is False

    gate.set()
    await activation
    await runtime.close()
    assert runtime.state == GpuASRRuntimeState.CLOSED


async def test_global_speech_end_fifo_has_finite_per_channel_public_admission() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(transcribe_gate=gate)
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=100.0),
    )
    await _activate(runtime)
    await _activate(runtime, "peer")
    samples = np.zeros(1600, dtype=np.float32)

    blocker = asyncio.create_task(runtime.submit("self", samples, speech_end_at=99.0))
    await client.started.wait()
    queued = [
        asyncio.create_task(
            runtime.submit(
                "peer" if index % 2 == 0 else "self",
                samples,
                speech_end_at=99.5 + index * 0.001,
            )
        )
        for index in range(16)
    ]
    await asyncio.sleep(0)
    assert runtime.pending_count == 16
    with pytest.raises(GpuASRWorkDiscarded, match="pending_capacity"):
        await runtime.submit("self", samples, speech_end_at=100.0)
    with pytest.raises(GpuASRWorkDiscarded, match="pending_capacity"):
        await runtime.submit("peer", samples, speech_end_at=100.0)

    gate.set()
    await asyncio.gather(blocker, *queued)

    assert len(client.transcribe_calls) == 17
    assert [channel for _request, channel, _hint in client.transcribe_calls[1:4]] == [
        "peer",
        "self",
        "peer",
    ]
    await runtime.close()


async def test_shared_queue_preserves_per_utterance_language_hints() -> None:
    client = FakeGpuWorkerClient()
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=2.0),
    )
    await _activate(runtime)
    await _activate(runtime, "peer")
    samples = np.zeros(1600, dtype=np.float32)

    await asyncio.gather(
        runtime.submit("self", samples, speech_end_at=1.0, language_hint="ko"),
        runtime.submit("peer", samples, speech_end_at=2.0, language_hint=None),
    )

    assert [(channel, hint) for _request, channel, hint in client.transcribe_calls] == [
        ("self", "ko"),
        ("peer", None),
    ]
    await runtime.close()


async def test_pending_work_expires_exactly_at_twelve_seconds_before_start() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(transcribe_gate=gate)
    clock = FakeClock(_now=100.0)
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=clock,
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime)
    await _activate(runtime, "peer")
    samples = np.zeros(1600, dtype=np.float32)

    active = asyncio.create_task(runtime.submit("self", samples, speech_end_at=99.0))
    await client.started.wait()
    expires = asyncio.create_task(runtime.submit("peer", samples, speech_end_at=88.0))
    survives = asyncio.create_task(runtime.submit("self", samples, speech_end_at=88.001))
    await asyncio.sleep(0)
    gate.set()

    await active
    with pytest.raises(GpuASRWorkExpired):
        await expires
    await survives

    assert len(client.transcribe_calls) == 2
    expiry = [item for item in diagnostics if item.kind == "work_expired"]
    assert len(expiry) == 1
    assert expiry[0].fields["channel"] == "peer"
    assert "rtf" not in expiry[0].fields
    await runtime.close()


async def test_active_decode_survives_ttl() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(transcribe_gate=gate)
    clock = FakeClock(_now=10.0)
    runtime = SharedGpuASRRuntime(process_factory=FakeGpuWorkerFactory([client]), clock=clock)
    await _activate(runtime)
    work = asyncio.create_task(
        runtime.submit("self", np.zeros(1600, dtype=np.float32), speech_end_at=10.0)
    )
    await client.started.wait()
    clock.advance(30.0)
    gate.set()

    assert (await work).text == "self-1"
    await runtime.close()


async def test_last_disable_cancels_active_work_and_awaits_worker_close() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(transcribe_gate=gate)
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=10.0),
    )
    await _activate(runtime)
    work = asyncio.create_task(
        runtime.submit("self", np.zeros(1600, dtype=np.float32), speech_end_at=10.0)
    )
    await client.started.wait()

    await runtime.deactivate_channel("self")

    with pytest.raises(GpuASRWorkDiscarded):
        await work
    assert len(client.cancel_calls) == 1
    assert client.close_calls == 1
    assert runtime.state == GpuASRRuntimeState.STOPPED
    await runtime.close()


@pytest.mark.parametrize("shutdown", ["last_disable", "application_close"])
@pytest.mark.parametrize("cancel_hangs", [False, True])
async def test_terminal_shutdown_forces_noncooperative_worker_without_stopping_hang(
    shutdown: str,
    cancel_hangs: bool,
) -> None:
    class NonCooperativeClient(FakeGpuWorkerClient):
        def __init__(self) -> None:
            super().__init__(transcribe_gate=asyncio.Event())
            self.close_gate = asyncio.Event()
            self.force_close_calls = 0

        async def cancel(self, target_request_id: str) -> None:
            self.cancel_calls.append(target_request_id)
            if cancel_hangs:
                await asyncio.Event().wait()

        async def close(self) -> None:
            self.close_calls += 1
            await self.close_gate.wait()

        async def force_close(self) -> None:
            self.force_close_calls += 1
            self._closed = True
            assert self.transcribe_gate is not None
            self.transcribe_gate.set()
            self.close_gate.set()
            await self._events.put(None)

    client = NonCooperativeClient()
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=10.0),
        diagnostic_sink=diagnostics.append,
        channel_cancel_seconds=0.01,
        force_close_seconds=0.01,
    )
    await _activate(runtime)
    work = asyncio.create_task(
        runtime.submit("self", np.zeros(1600, dtype=np.float32), speech_end_at=10.0)
    )
    await client.started.wait()

    if shutdown == "last_disable":
        await asyncio.wait_for(runtime.deactivate_channel("self"), timeout=0.2)
        expected_state = GpuASRRuntimeState.STOPPED
    else:
        await asyncio.wait_for(runtime.close(), timeout=0.2)
        expected_state = GpuASRRuntimeState.CLOSED

    with pytest.raises(GpuASRWorkDiscarded):
        await work
    assert client.force_close_calls == 1
    assert runtime.state == expected_state
    assert runtime.active_channels == frozenset()
    assert runtime.worker_pid is None
    assert runtime._dispatcher_task is None
    assert runtime._event_task is None
    assert runtime._temporary_directory is None
    stopped = [item for item in diagnostics if item.kind == "worker_stopped"]
    assert stopped[-1].fields["forced"] is True
    if shutdown == "last_disable":
        await runtime.close()


@pytest.mark.parametrize(
    ("disabled_channel", "surviving_channel"),
    [("self", "peer"), ("peer", "self")],
)
async def test_channel_disable_discards_only_its_work_and_retains_shared_worker(
    disabled_channel: GpuASRChannel,
    surviving_channel: GpuASRChannel,
) -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(transcribe_gate=gate)
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=100.0),
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime, "self")
    await _activate(runtime, "peer")
    samples = np.zeros(1600, dtype=np.float32)

    active = asyncio.create_task(runtime.submit(disabled_channel, samples, speech_end_at=99.0))
    await client.started.wait()
    discarded_pending = asyncio.create_task(
        runtime.submit(disabled_channel, samples, speech_end_at=99.1)
    )
    surviving_pending = asyncio.create_task(
        runtime.submit(surviving_channel, samples, speech_end_at=99.2)
    )
    await asyncio.sleep(0)

    await runtime.deactivate_channel(disabled_channel)
    await _activate(runtime, disabled_channel)

    with pytest.raises(GpuASRWorkDiscarded, match="channel_disabled"):
        await active
    with pytest.raises(GpuASRWorkDiscarded, match="channel_disabled"):
        await discarded_pending
    fresh = await runtime.submit(disabled_channel, samples, speech_end_at=100.0)
    assert fresh.text == f"{disabled_channel}-3"
    assert (await surviving_pending).text == f"{surviving_channel}-2"
    assert runtime.active_channels == frozenset({"self", "peer"})
    assert runtime.state == GpuASRRuntimeState.READY
    assert client.close_calls == 0
    assert len(client.cancel_calls) == 1
    lifecycle = [item for item in diagnostics if item.kind == "channel_deactivated"]
    assert lifecycle[-1].fields == {
        "channel": disabled_channel,
        "pending_discarded": 1,
        "active_cancelled": True,
        "worker_retained": True,
    }
    await runtime.close()


async def test_channel_disable_rejects_unresolved_worker_result() -> None:
    client = FakeGpuWorkerClient(transcribe_gate=asyncio.Event())
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=100.0),
    )
    await _activate(runtime, "self")
    await _activate(runtime, "peer")
    work = asyncio.create_task(
        runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=99.0,
        )
    )
    await client.started.wait()
    await runtime.deactivate_channel("self")

    with pytest.raises(GpuASRWorkDiscarded, match="channel_disabled"):
        await work
    assert runtime.state == GpuASRRuntimeState.READY
    assert runtime.active_channels == frozenset({"peer"})
    assert client.close_calls == 0
    await runtime.close()


@pytest.mark.parametrize("cancel_hangs", [False, True])
async def test_noncooperative_partial_channel_cancel_forces_bounded_worker_failure(
    cancel_hangs: bool,
) -> None:
    class NonCooperativeClient(FakeGpuWorkerClient):
        def __init__(self) -> None:
            super().__init__(transcribe_gate=asyncio.Event())
            self.close_gate = asyncio.Event()
            self.force_close_calls = 0

        async def cancel(self, target_request_id: str) -> None:
            self.cancel_calls.append(target_request_id)
            if cancel_hangs:
                await asyncio.Event().wait()

        async def close(self) -> None:
            self.close_calls += 1
            await self.close_gate.wait()

        async def force_close(self) -> None:
            self.force_close_calls += 1
            self._closed = True
            assert self.transcribe_gate is not None
            self.transcribe_gate.set()
            self.close_gate.set()
            await self._events.put(None)

    client = NonCooperativeClient()
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=100.0),
        channel_cancel_seconds=0.01,
        force_close_seconds=0.01,
    )
    await _activate(runtime, "self")
    await _activate(runtime, "peer")
    active = asyncio.create_task(
        runtime.submit("self", np.zeros(1600, dtype=np.float32), speech_end_at=99.0)
    )
    await client.started.wait()
    pending = asyncio.create_task(
        runtime.submit("peer", np.zeros(1600, dtype=np.float32), speech_end_at=99.1)
    )
    await asyncio.sleep(0)

    await asyncio.wait_for(runtime.deactivate_channel("self"), timeout=0.2)

    with pytest.raises(GpuASRWorkDiscarded):
        await active
    with pytest.raises(GpuASRWorkDiscarded, match="channel_cancel_timeout"):
        await pending
    assert client.force_close_calls == 1
    assert runtime.state == GpuASRRuntimeState.FAILED
    assert runtime.last_failure_code == "channel_cancel_timeout"
    assert runtime.active_channels == frozenset({"peer"})
    assert runtime.pending_count == 0
    assert runtime.worker_pid is None
    assert runtime._dispatcher_task is None
    assert runtime._event_task is None
    assert runtime._temporary_directory is None
    assert runtime._scope.active_task_names == ()
    await runtime.close()


async def test_ttl_is_rechecked_after_wav_staging_before_worker_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeGpuWorkerClient()
    clock = FakeClock(_now=100.0)
    diagnostics = []
    staged_paths: list[Path] = []
    original_write = gpu_asr_module._write_pcm16_wav

    def stage_and_advance(path: Path, samples: np.ndarray) -> None:
        original_write(path, samples)
        staged_paths.append(path)
        clock.advance(2.0)

    monkeypatch.setattr(gpu_asr_module, "_write_pcm16_wav", stage_and_advance)
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=clock,
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime)

    with pytest.raises(GpuASRWorkExpired):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=90.0,
        )

    assert client.transcribe_calls == []
    assert len(staged_paths) == 1
    expiry = [item for item in diagnostics if item.kind == "work_expired"]
    assert expiry[-1].fields["queue_wait_seconds"] == 12.0
    assert "rtf" not in expiry[-1].fields
    assert [item for item in diagnostics if item.kind == "decode_attempt"] == []
    await runtime.close()
    assert not staged_paths[0].exists()


@pytest.mark.parametrize(
    "failure",
    [
        OSError("staging failed"),
        GpuWorkerClosedError("send failed"),
        GpuWorkerRequestError(
            "audio_invalid",
            {"channel": "self", "backend": "Vulkan"},
            attempt_started=False,
        ),
    ],
)
async def test_prestart_failures_emit_no_attempt_or_rtf(
    failure: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeGpuWorkerClient(
        send_error=failure if isinstance(failure, GpuWorkerClosedError) else None,
        transcribe_error=(failure if isinstance(failure, GpuWorkerRequestError) else None),
    )
    if isinstance(failure, OSError):
        monkeypatch.setattr(
            gpu_asr_module,
            "_write_pcm16_wav",
            lambda _path, _samples: (_ for _ in ()).throw(failure),
        )
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]),
        clock=FakeClock(_now=10.0),
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime)

    with pytest.raises(type(failure)):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=10.0,
        )

    assert [item for item in diagnostics if item.kind == "decode_attempt"] == []
    prestart = [item for item in diagnostics if item.kind == "work_prestart_failed"]
    assert len(prestart) == 1
    assert "rtf" not in prestart[0].fields
    await runtime.close()


async def test_started_native_failure_retains_worker_decode_only_timing() -> None:
    native_failure = GpuWorkerRequestError(
        "decode_failure",
        {
            "audio_seconds": 0.1,
            "decode_seconds": 0.04,
            "rtf": 0.4,
        },
        attempt_started=True,
    )
    diagnostics = []
    recovered = FakeGpuWorkerClient()
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory(
            [FakeGpuWorkerClient(transcribe_error=native_failure), recovered]
        ),
        clock=FakeClock(_now=10.0),
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime)

    with pytest.raises(GpuASRDecodeDropped, match="decode_failure"):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=10.0,
        )

    attempts = [item for item in diagnostics if item.kind == "decode_attempt"]
    assert attempts[-1].fields == {
        "channel": "self",
        "model": "qwen-gpu",
        "backend": "Vulkan",
        "result": "decode_failure",
        "queue_wait_seconds": 0.0,
        "audio_seconds": 0.1,
        "decode_seconds": 0.04,
        "rtf": 0.4,
    }
    await asyncio.wait_for(recovered.activation_started.wait(), timeout=0.5)
    await runtime.close()


async def test_decode_failure_restarts_once_without_replaying_failed_utterance() -> None:
    native_failure = GpuWorkerRequestError(
        "decode_failure",
        {
            "audio_seconds": 7.5,
            "decode_seconds": 2.9,
            "rtf": 0.386,
        },
        attempt_started=True,
    )
    activation_gate = asyncio.Event()
    failed = FakeGpuWorkerClient(transcribe_error=native_failure)
    recovered = FakeGpuWorkerClient(activation_gate=activation_gate)
    factory = FakeGpuWorkerFactory([failed, recovered])
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=factory,
        clock=FakeClock(_now=10.0),
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime, "peer")

    with pytest.raises(GpuASRDecodeDropped, match="decode_failure"):
        await runtime.submit(
            "peer",
            np.zeros(120_000, dtype=np.float32),
            speech_end_at=10.0,
        )

    await asyncio.wait_for(recovered.activation_started.wait(), timeout=0.5)
    next_utterance = asyncio.create_task(
        runtime.submit(
            "peer",
            np.ones(1600, dtype=np.float32),
            speech_end_at=10.0,
        )
    )
    await asyncio.sleep(0)
    assert len(failed.transcribe_calls) == 1
    assert recovered.transcribe_calls == []
    assert runtime.pending_count == 1

    activation_gate.set()
    result = await asyncio.wait_for(next_utterance, timeout=0.5)

    assert result.text == "peer-1"
    assert len(failed.transcribe_calls) == 1
    assert len(recovered.transcribe_calls) == 1
    assert failed.transcribe_calls[0][0] != recovered.transcribe_calls[0][0]
    assert factory.modes == ["persistent", "persistent"]
    assert [item.kind for item in diagnostics].count("worker_recovery_started") == 1
    assert [item.kind for item in diagnostics].count("worker_recovery_ready") == 1
    recovery = next(item for item in diagnostics if item.kind == "worker_recovery_started")
    assert recovery.fields["utterance_retry"] is False
    await runtime.close()


async def test_second_decode_failure_before_success_requires_manual_retry() -> None:
    failure = GpuWorkerRequestError("decode_failure", attempt_started=True)
    first = FakeGpuWorkerClient(transcribe_error=failure)
    second = FakeGpuWorkerClient(transcribe_error=failure)
    factory = FakeGpuWorkerFactory([first, second])
    runtime = SharedGpuASRRuntime(
        process_factory=factory,
        clock=FakeClock(_now=10.0),
    )
    await _activate(runtime)

    with pytest.raises(GpuASRDecodeDropped):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=10.0,
        )
    await asyncio.wait_for(second.activation_started.wait(), timeout=0.5)
    while runtime.state != GpuASRRuntimeState.READY:
        await asyncio.sleep(0)

    with pytest.raises(GpuWorkerRequestError, match="decode_failure"):
        await runtime.submit(
            "self",
            np.ones(1600, dtype=np.float32),
            speech_end_at=10.0,
        )

    assert runtime.state == GpuASRRuntimeState.FAILED
    assert factory.modes == ["persistent", "persistent"]
    with pytest.raises(GpuASRManualRetryRequired, match="decode_failure"):
        await _activate(runtime)
    await runtime.close()


async def test_started_out_of_memory_failure_does_not_auto_restart() -> None:
    failure = GpuWorkerRequestError("out_of_memory", attempt_started=True)
    factory = FakeGpuWorkerFactory(
        [FakeGpuWorkerClient(transcribe_error=failure), FakeGpuWorkerClient()]
    )
    runtime = SharedGpuASRRuntime(
        process_factory=factory,
        clock=FakeClock(_now=10.0),
    )
    await _activate(runtime)

    with pytest.raises(GpuWorkerRequestError, match="out_of_memory"):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=10.0,
        )

    assert runtime.state == GpuASRRuntimeState.FAILED
    assert factory.modes == ["persistent"]
    await runtime.close()


async def test_recovery_activation_failure_discards_new_pending_work() -> None:
    decode_failure = GpuWorkerRequestError("decode_failure", attempt_started=True)
    activation_gate = asyncio.Event()
    failed = FakeGpuWorkerClient(transcribe_error=decode_failure)
    replacement = FakeGpuWorkerClient(
        activation_gate=activation_gate,
        activation_error=GpuWorkerRequestError("out_of_memory"),
    )
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([failed, replacement]),
        clock=FakeClock(_now=10.0),
        diagnostic_sink=diagnostics.append,
    )
    await _activate(runtime)

    with pytest.raises(GpuASRDecodeDropped):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=10.0,
        )
    await asyncio.wait_for(replacement.activation_started.wait(), timeout=0.5)
    pending = asyncio.create_task(
        runtime.submit(
            "self",
            np.ones(1600, dtype=np.float32),
            speech_end_at=10.0,
        )
    )
    await asyncio.sleep(0)
    activation_gate.set()

    with pytest.raises(GpuASRWorkDiscarded, match="worker_recovery_failed"):
        await asyncio.wait_for(pending, timeout=0.5)
    assert runtime.state == GpuASRRuntimeState.FAILED
    assert runtime.last_failure_code == "out_of_memory"
    terminal = [item for item in diagnostics if item.kind == "worker_failed"]
    assert terminal[-1].fields["failure"] == "out_of_memory"
    assert terminal[-1].fields["retry"] == "manual"
    await runtime.close()


async def test_close_cancels_and_awaits_decode_recovery_activation() -> None:
    decode_failure = GpuWorkerRequestError("decode_failure", attempt_started=True)
    activation_gate = asyncio.Event()
    failed = FakeGpuWorkerClient(transcribe_error=decode_failure)
    replacement = FakeGpuWorkerClient(activation_gate=activation_gate)
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([failed, replacement]),
        clock=FakeClock(_now=10.0),
    )
    await _activate(runtime)

    with pytest.raises(GpuASRDecodeDropped):
        await runtime.submit(
            "self",
            np.zeros(1600, dtype=np.float32),
            speech_end_at=10.0,
        )
    await asyncio.wait_for(replacement.activation_started.wait(), timeout=0.5)

    await asyncio.wait_for(runtime.close(), timeout=0.5)

    assert runtime.state == GpuASRRuntimeState.CLOSED
    assert failed.close_calls >= 1
    assert replacement.close_calls >= 1


async def test_gpu_failure_requires_manual_retry_and_never_starts_fallback() -> None:
    failed = FakeGpuWorkerClient(activation_error=GpuWorkerRequestError("out_of_memory"))
    recovered = FakeGpuWorkerClient()
    factory = FakeGpuWorkerFactory([failed, recovered])
    runtime = SharedGpuASRRuntime(process_factory=factory)

    with pytest.raises(GpuASRManualRetryRequired, match="out_of_memory"):
        await _activate(runtime, "self")
    assert runtime.state == GpuASRRuntimeState.FAILED
    assert runtime.last_failure_code == "out_of_memory"
    assert factory.modes == ["persistent"]

    with pytest.raises(GpuASRManualRetryRequired):
        await _activate(runtime, "peer")
    assert factory.modes == ["persistent"]

    await runtime.retry()
    assert runtime.state == GpuASRRuntimeState.READY
    assert factory.modes == ["persistent", "persistent"]
    assert runtime.active_channels == frozenset({"self", "peer"})
    await runtime.close()


async def test_discovery_reports_pending_without_loading_a_model() -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(discovery_gate=gate)
    factory = FakeGpuWorkerFactory([client])
    runtime = SharedGpuASRRuntime(
        process_factory=factory,
        discovery_pending_seconds=0.01,
    )

    discovery = asyncio.create_task(runtime.discover_devices())
    await asyncio.sleep(0.02)
    assert runtime.discovery_state == GpuDiscoveryState.PENDING
    assert client.activate_calls == []
    gate.set()

    assert await discovery == (DEVICE,)
    assert runtime.discovery_state == GpuDiscoveryState.READY
    assert factory.modes == ["discovery"]
    assert client.close_calls == 1
    await runtime.close()


def _scoped_gpu_request(channel: GpuASRChannel, epoch: str) -> STTProviderTurnRequest:
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=AudioSegmentIdentity(
                activation_generation=1,
                segment_order=1,
                segment_id=uuid4(),
                capture_epoch=1,
            ),
            provider_epoch_id=epoch,
            provider_turn_id=f"{epoch}-turn",
        ),
        settings=AudioSegmentSettingsSnapshot(
            provider_id="local_qwen_gpu",
            provider_signature=("local_qwen_gpu",),
            runtime_signature=("local_qwen_gpu",),
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


async def test_real_shared_gpu_runtime_serves_two_scoped_concrete_adapters(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    client = FakeGpuWorkerClient(transcribe_gate=gate)
    runtime = SharedGpuASRRuntime(process_factory=FakeGpuWorkerFactory([client]))
    self_backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="self",
        model_path=tmp_path / "model.gguf",
        model_id="qwen-gpu",
        device_id="vulkan:0",
        source_mode="manual",
    )
    peer_backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel="peer",
        model_path=tmp_path / "model.gguf",
        model_id="qwen-gpu",
        device_id="vulkan:0",
        source_mode="manual",
    )
    self_session = await self_backend.open_session(
        projection=STTSessionProjection("scoped", "self-epoch")
    )
    peer_session = await peer_backend.open_session(
        projection=STTSessionProjection("scoped", "peer-epoch")
    )
    self_request = _scoped_gpu_request("self", "self-epoch")
    peer_request = _scoped_gpu_request("peer", "peer-epoch")
    for session, request in (
        (self_session, self_request),
        (peer_session, peer_request),
    ):
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

    await client.started.wait()
    for _ in range(100):
        if runtime.pending_count == 1:
            break
        await asyncio.sleep(0)
    await self_session.abort_turn(self_request.identity, reason="self_off")
    self_terminal = await asyncio.wait_for(anext(self_session.turn_events()), timeout=0.5)
    assert (self_terminal.outcome, self_terminal.failure_reason) == (
        "cancelled",
        "self_off",
    )
    assert runtime.active_channels == frozenset({"self", "peer"})
    assert runtime.pending_count == 1
    assert runtime.worker_pid == 1234

    gate.set()
    peer_terminal = await asyncio.wait_for(anext(peer_session.turn_events()), timeout=0.5)
    assert (peer_terminal.outcome, peer_terminal.text) == ("final", "peer-2")
    assert [call[1] for call in client.transcribe_calls] == ["self", "peer"]

    await self_session.close()
    await self_backend.close()
    assert runtime.active_channels == frozenset({"peer"})
    assert runtime.worker_pid == 1234
    await peer_session.close()
    await peer_backend.close()
    assert runtime.state == GpuASRRuntimeState.STOPPED
    await runtime.close()


@pytest.mark.parametrize("delete_fails", [False, True])
async def test_scoped_gpu_transcript_precedes_owned_file_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delete_fails: bool
) -> None:
    import threading
    import wave

    class InspectingClient(FakeGpuWorkerClient):
        wav_samples: bytes | None = None

        async def transcribe(self, *, audio_path: Path, **kwargs):
            with wave.open(str(audio_path), "rb") as wav:
                self.wav_samples = wav.readframes(wav.getnframes())
            return await super().transcribe(audio_path=audio_path, **kwargs)

    client = InspectingClient()
    diagnostics = []
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]), diagnostic_sink=diagnostics.append
    )
    backend = LocalGpuSTTBackend(
        runtime=runtime, channel="self", model_path=tmp_path / "model.gguf",
        model_id="qwen-gpu", device_id="vulkan:0",
    )
    session = await backend.open_session(projection=STTSessionProjection("scoped", "probe"))
    request = _scoped_gpu_request("self", "probe")
    raw = bytearray(np.array([-32768, -16385, -1, 0, 1, 16385, 32767], dtype="<i2").tobytes())
    original = bytes(raw)
    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity, raw, payload_sequence=1, source_ranges=(), context_only=False
    )
    raw[:] = b"\0" * len(raw)
    await session.seal_turn(
        request.identity, sealed_content_ranges=(), seal_reason="end",
        observed_trailing_silence_ms=800,
    )
    entered = threading.Event()
    release = threading.Event()
    original_unlink = Path.unlink

    def blocked_unlink(path: Path, *args, **kwargs):
        if path.suffix == ".wav":
            entered.set()
            if not release.wait(3):
                raise TimeoutError("deletion gate expired")
            if delete_fails:
                raise OSError("injected unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", blocked_unlink)
    try:
        terminal = asyncio.create_task(anext(session.turn_events()))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)
        result = await asyncio.wait_for(terminal, timeout=0.5)
        assert result.identity == request.identity
        assert (result.outcome, result.text) == ("final", "self-1")
        assert client.wav_samples == np.rint(
            np.clip(np.frombuffer(original, dtype="<i2").astype(np.float32) / 32768.0, -1, 1)
            * 32767.0
        ).astype("<i2").tobytes()
        shutdown = asyncio.create_task(backend.close())
        await asyncio.sleep(0)
        assert not shutdown.done()
        release.set()
        await asyncio.wait_for(shutdown, timeout=3)
        await runtime.close()
        cleanup = [event for event in diagnostics if event.kind == "file_cleanup"]
        assert len(cleanup) == 1
        assert cleanup[0].fields["failure"] == ("OSError" if delete_fails else None)
        timing = [event for event in diagnostics if event.kind == "decode_timing"]
        assert len(timing) == 1
        assert set(timing[0].fields) >= {
            "gpu_queue_wait_seconds", "wav_stage_seconds",
            "worker_request_to_response_seconds",
            "worker_response_to_transcript_publish_seconds",
        }
    finally:
        release.set()
        monkeypatch.setattr(Path, "unlink", original_unlink)
        await session.close()
        await runtime.close()


@pytest.mark.parametrize("delete_fails", [False, True])
async def test_gpu_cleanup_debt_blocks_next_dispatch_without_blocking_first_result(
    monkeypatch: pytest.MonkeyPatch, delete_fails: bool
) -> None:
    import threading

    client = FakeGpuWorkerClient()
    runtime = SharedGpuASRRuntime(
        process_factory=FakeGpuWorkerFactory([client]), clock=FakeClock(_now=100.0)
    )
    await _activate(runtime, "self")
    await _activate(runtime, "peer")
    entered = threading.Event()
    release = threading.Event()
    original_unlink = Path.unlink

    def gated_unlink(path: Path, *args, **kwargs):
        if path.suffix == ".wav" and not entered.is_set():
            entered.set()
            if not release.wait(3):
                raise TimeoutError("deletion gate expired")
            if delete_fails:
                raise OSError("injected unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", gated_unlink)
    try:
        first = asyncio.create_task(
            runtime.submit("self", np.zeros(1600, dtype=np.float32), speech_end_at=99.0)
        )
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)
        assert (await asyncio.wait_for(first, timeout=0.5)).text == "self-1"
        second = asyncio.create_task(
            runtime.submit("peer", np.ones(1600, dtype=np.float32), speech_end_at=99.1)
        )
        await asyncio.sleep(0)
        assert runtime.pending_count == 1
        assert len(client.transcribe_calls) == 1
        release.set()
        if delete_fails:
            with pytest.raises(GpuASRWorkDiscarded, match="worker_failed"):
                await asyncio.wait_for(second, timeout=3)
            assert runtime.state == GpuASRRuntimeState.FAILED
            assert len(client.transcribe_calls) == 1
        else:
            assert (await asyncio.wait_for(second, timeout=3)).text == "peer-2"
    finally:
        release.set()
        monkeypatch.setattr(Path, "unlink", original_unlink)
        await runtime.close()
