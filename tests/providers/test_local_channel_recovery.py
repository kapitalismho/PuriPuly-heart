from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.adapters.self_capture.self_capture_vad_sink import SelfCaptureVadSinkAdapter
from puripuly_heart.core.asr_worker import ASRWorkerRequestError
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.local_asr.local_stt_assets import FUN_ASR_NANO_MODEL_ID
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import (
    STTChannelAvailability,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.local_cpp import LocalCppSTTBackend
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend
from tests.core.runtime.test_gpu_asr import FakeASRWorkerClient, FakeASRWorkerFactory
from tests.core.runtime.test_self_capture_session import build_owner, config
from tests.core.test_peer_caption_lifecycle import ControlledBackend
from tests.core.test_stt_scoped_engine import segment_events, settings, wait_until, watchdogs
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_local_cpp_backend import NativeWorker


class RecoveryWorkers:
    def __init__(self, *, unsafe_cleanup=False, always_fail=False):
        self.workers = []
        self.unsafe_cleanup = unsafe_cleanup
        self.always_fail = always_fail
        self.allow_cleanup = False

    async def start(self, **_options):
        if self.workers:
            assert self.workers[-1].is_closed
        owner = self
        fail = not self.workers or self.always_fail

        class Worker(NativeWorker):
            async def transcribe(self, **options):
                result = await super().transcribe(**options)
                if fail:
                    raise ASRWorkerRequestError("decode_failure", attempt_started=True)
                return result

            async def close(self):
                if owner.unsafe_cleanup and not owner.allow_cleanup:
                    raise RuntimeError("controlled cleanup unconfirmed")
                await super().close()

        worker = Worker(1000 + len(self.workers))
        worker.finish.set()
        self.workers.append(worker)
        return worker

    async def close_pending_startups(self):
        return None


@pytest.mark.parametrize("always_fail", [False, True])
async def test_fun_asr_decode_failure_crosses_capture_and_real_translation_consumer(
    tmp_path, monkeypatch, always_fail
):
    workers = RecoveryWorkers(always_fail=always_fail)
    backend = LocalCppSTTBackend(
        model_dir=tmp_path,
        model_id=FUN_ASR_NANO_MODEL_ID,
        provider_id="local_fun_asr_nano",
        stream_label="self",
        process_factory=workers,
    )
    monkeypatch.setattr(backend, "_validate_runtime_assets", lambda: None)
    terminals, delays = [], []
    translator = ControlledBackend()
    osc = RecordingOscQueue()
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())

    async def factory(_settings, epoch):
        return await backend.open_session(projection=STTSessionProjection("scoped", epoch))

    async def sleep(delay):
        delays.append(delay)

    async def on_event(event):
        if isinstance(event, STTProviderTurnTerminal):
            terminals.append(event)
        await callbacks.self_event_handler(event)

    engine = ScopedRecognitionEngine(
        session_factory=factory,
        channel="self",
        event_sink=on_event,
        sleep=sleep,
        watchdog_resolver=lambda _: watchdogs(connect_retry_base_s=0.8, connect_retry_max_s=1.6),
        session_lifetime_enabled=False,
    )
    translation = compose_translation_test_harness(stt=engine, llm=translator, osc=osc)
    callbacks.bind_self(translation.self_owner)
    capture, _, provider, sources, loop, _ = build_owner(
        sink=SelfCaptureVadSinkAdapter(lambda: translation.self_owner)
    )
    callbacks.bind_self_capture(capture)
    capture_config = replace(config(local_cpu=True), provider_id="local_fun_asr_nano")
    engine.accepted_settings_scope = (
        capture_config.provider_id,
        capture_config.provider_signature,
        capture_config.runtime_signature,
    )
    try:
        await translation.start()
        engine.bind_event_sink(on_event)
        await capture.apply_intent(capture_config, enabled=True)
        await loop.started.wait()
        guarded = loop.calls[0]["sink"]
        initial_loop = capture.loop_task
        for order in range(3 if always_fail else 2):
            segment_id = uuid4()
            await guarded.handle_vad_event(
                SpeechStart(
                    segment_id,
                    pre_roll=np.empty(0, dtype=np.float32),
                    chunk=np.full(160, 0.1 + order / 10, dtype=np.float32),
                )
            )
            await guarded.handle_vad_event(SpeechEnd(segment_id))
            await wait_until(lambda: len(terminals) == order + 1)
            if not always_fail or order < 2:
                assert capture.snapshot.desired_active and capture.snapshot.effective_active
                assert capture.loop_task is initial_loop
        assert [worker.calls for worker in workers.workers] == [1] * len(workers.workers)
        assert all(worker.is_closed for worker in workers.workers[:-1])
        assert all(
            event.failure_diagnostics.fields["native_classification"] == "decode_failure"
            for event in terminals
            if event.outcome == "failed"
        )
        assert terminals[0].channel_availability is STTChannelAvailability.RECOVERING
        if always_fail:
            await wait_until(lambda: not capture.snapshot.effective_active)
            assert terminals[-1].channel_availability is STTChannelAvailability.UNAVAILABLE
            assert len(workers.workers) == 3
            assert delays == [0.8, 1.6]
            assert translator.calls == []
            assert osc.messages == []
        else:
            await wait_until(lambda: bool(osc.messages))
            assert [request.text for request in translator.calls] == ["completed speech"]
            assert terminals[1].outcome == "final"
            assert terminals[1].text == "completed speech"
            assert len(osc.messages) == 1
            assert delays == [0.8]
            assert sources[0].close_calls == 0
            assert provider.release_calls == []
            assert guarded.ledger.terminal_receipts[0].outcome == "failed"
            assert guarded.ledger.terminal_receipts[1].outcome == "final"
    finally:
        workers.allow_cleanup = True
        await capture.close()
        await engine.close()
        await backend.close()
        await translation.stop()


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_cpu_replacement_refuses_unconfirmed_old_worker_cleanup(
    tmp_path, monkeypatch, channel
):
    workers = RecoveryWorkers(unsafe_cleanup=True)
    backend = LocalCppSTTBackend(
        model_dir=tmp_path,
        model_id=FUN_ASR_NANO_MODEL_ID,
        provider_id="local_fun_asr_nano",
        stream_label=channel,
        process_factory=workers,
    )
    monkeypatch.setattr(backend, "_validate_runtime_assets", lambda: None)
    try:
        with pytest.raises(ASRWorkerRequestError) as failed:
            await backend.decode_f32(np.ones(160, dtype=np.float32))
        assert failed.value.code == "decode_failure"
        for _ in range(2):
            with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
                await backend.decode_f32(np.ones(320, dtype=np.float32))
        assert len(workers.workers) == 1
        assert workers.workers[0].calls == 1
        workers.allow_cleanup = True
        assert await backend.decode_f32(np.ones(480, dtype=np.float32)) == "completed speech"
        assert workers.workers[0].is_closed
        assert [worker.calls for worker in workers.workers] == [1, 1]
    finally:
        workers.allow_cleanup = True
        await backend.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("replacement_fails", [False, True])
async def test_gpu_owned_restart_drop_is_not_an_upper_layer_channel_stop(
    channel, replacement_fails
):
    native_failure = ASRWorkerRequestError("decode_failure", attempt_started=True)
    first = FakeASRWorkerClient(transcribe_error=native_failure)
    replacement = FakeASRWorkerClient(
        transcribe_error=native_failure if replacement_fails else None
    )
    factory = FakeASRWorkerFactory([first, replacement])
    runtime = SharedGpuASRRuntime(process_factory=factory)
    backend = LocalGpuSTTBackend(
        runtime=runtime,
        channel=channel,
        model_path=Path("controlled-model.gguf"),
        model_id="controlled-gpu",
        device_id="auto",
    )
    events, failures, delays = [], [], []

    async def open_session(_settings, epoch):
        return await backend.open_session(projection=STTSessionProjection("scoped", epoch))

    async def sleep(delay):
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=open_session,
        channel=channel,
        event_sink=events.append,
        terminal_failure_sink=failures.append,
        sleep=sleep,
        session_lifetime_enabled=False,
        watchdog_resolver=lambda _: watchdogs(final_timeout_s=1.0),
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("local_qwen_gpu"))
    try:
        for order in range(2):
            start, _, end = segment_events(ledger, start_sample=order * 20, now=order)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
            await wait_until(lambda: len(events) == order + 1)
            if order == 0:
                assert events[0].outcome == "failed"
                assert events[0].channel_availability is STTChannelAvailability.RECOVERING
                assert not failures
                assert (
                    events[0].failure_diagnostics.fields["native_classification"]
                    == "decode_failure"
                )
        assert first.is_closed
        assert len(first.transcribe_calls) == len(replacement.transcribe_calls) == 1
        assert factory.modes == ["persistent", "persistent"]
        assert delays == []
        if replacement_fails:
            assert events[-1].outcome == "failed"
            assert events[-1].channel_availability is STTChannelAvailability.UNAVAILABLE
            assert len(failures) == 1
            assert all(not event.text for event in events)
        else:
            assert events[-1].outcome == "final"
            assert events[-1].text == f"{channel}-1"
            assert not failures
    finally:
        await engine.close()
        await backend.close()
        await runtime.close()
