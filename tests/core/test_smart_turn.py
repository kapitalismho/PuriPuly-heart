from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from puripuly_heart.core.vad.gating import (
    SpeechChunk,
    SpeechEnd,
    SpeechStart,
    VadEvent,
    VadGating,
)
from puripuly_heart.core.vad.smart_turn import (
    SmartTurnEndpointPolicy,
    SmartTurnExperimentConfig,
    SmartTurnOnnxInference,
    SmartTurnPrediction,
    compute_whisper_log_mel_features,
    prepare_smart_turn_audio,
)
from tests.helpers.vad import SequenceVadEngine


@dataclass(slots=True)
class RecordingSink:
    events: list[VadEvent] = field(default_factory=list)

    async def handle_vad_event(self, event: VadEvent) -> None:
        self.events.append(event)


@dataclass(slots=True)
class RecordingInference:
    score: float
    calls: list[np.ndarray] = field(default_factory=list)

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> SmartTurnPrediction:
        assert sample_rate_hz == 16000
        self.calls.append(audio.copy())
        return SmartTurnPrediction(self.score)


class _ResetControl:
    def __init__(self, callback) -> None:
        self._callback = callback

    def reset(self) -> None:
        self._callback()


class BlockingInference:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> SmartTurnPrediction:
        _ = (audio, sample_rate_hz)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        return SmartTurnPrediction(0.99)


def _config(stage: str) -> SmartTurnExperimentConfig:
    return SmartTurnExperimentConfig(
        stage=stage,
        threshold=0.5,
        probe_silence_ms=(224, 416, 608),
        hard_end_ms=800,
    )


async def _send_start_and_silence(
    policy: SmartTurnEndpointPolicy,
    utterance_id,
    count: int,
) -> None:
    await policy.handle_vad_event(
        SpeechStart(
            utterance_id=utterance_id,
            pre_roll=np.empty(0, dtype=np.float32),
            chunk=np.ones(512, dtype=np.float32),
        )
    )
    for _ in range(count):
        await policy.handle_vad_event(
            SpeechChunk(
                utterance_id=utterance_id,
                chunk=np.zeros(512, dtype=np.float32),
                is_speech=False,
            )
        )


def test_prepare_smart_turn_audio_left_pads_and_keeps_latest_eight_seconds() -> None:
    short = np.arange(5, dtype=np.float32)
    prepared = prepare_smart_turn_audio(short, sample_rate_hz=16000)
    assert prepared.shape == (128000,)
    assert np.array_equal(prepared[-5:], short)
    assert np.all(prepared[:-5] == 0)

    long = np.arange(128005, dtype=np.float32)
    prepared_long = prepare_smart_turn_audio(long, sample_rate_hz=16000)
    assert np.array_equal(prepared_long, long[-128000:])


def test_whisper_feature_extractor_returns_onnx_shape() -> None:
    features = compute_whisper_log_mel_features(np.zeros(128000, dtype=np.float32))
    assert features.shape == (80, 800)
    assert features.dtype == np.float32


async def test_active_policy_emits_early_end_at_first_accepted_probe() -> None:
    sink = RecordingSink()
    inference = RecordingInference(score=0.9)
    resets: list[str] = []
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: resets.append("reset")),
        inference=inference,
        config=_config("active"),
    )
    utterance_id = uuid4()

    await _send_start_and_silence(policy, utterance_id, 7)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert resets == ["reset"]
    ends = [event for event in sink.events if isinstance(event, SpeechEnd)]
    assert len(ends) == 1
    assert ends[0].utterance_id == utterance_id
    assert ends[0].trailing_silence_ms == 224
    assert len(inference.calls) == 1
    assert inference.calls[0].shape == (7 * 512 + 512,)


async def test_active_policy_includes_vad_pre_roll_in_probe_snapshot() -> None:
    sink = RecordingSink()
    inference = RecordingInference(score=0.9)
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: None),
        inference=inference,
        config=_config("active"),
    )
    utterance_id = uuid4()

    await policy.handle_vad_event(
        SpeechStart(
            utterance_id=utterance_id,
            pre_roll=np.full(3, -1.0, dtype=np.float32),
            chunk=np.ones(512, dtype=np.float32),
        )
    )
    for _ in range(7):
        await policy.handle_vad_event(
            SpeechChunk(
                utterance_id=utterance_id,
                chunk=np.zeros(512, dtype=np.float32),
                is_speech=False,
            )
        )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert inference.calls[0][:3].tolist() == [-1.0, -1.0, -1.0]
    await policy.close()


async def test_active_policy_repeats_unresolved_probes_with_newer_full_turn_snapshots() -> None:
    sink = RecordingSink()
    inference = RecordingInference(score=0.1)
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: None),
        inference=inference,
        config=_config("active"),
    )
    utterance_id = uuid4()

    await _send_start_and_silence(policy, utterance_id, 20)
    for _ in range(20):
        await asyncio.sleep(0)

    assert [call.size for call in inference.calls] == [8 * 512, 14 * 512, 20 * 512]
    assert not any(isinstance(event, SpeechEnd) for event in sink.events)
    await policy.close()


async def test_active_policy_forwards_vad_hard_boundary_after_all_probes_reject() -> None:
    sink = RecordingSink()
    inference = RecordingInference(score=0.1)
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: None),
        inference=inference,
        config=_config("active"),
    )
    vad = VadGating(
        SequenceVadEngine(probs=[0.9, *([0.0] * 25)]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=800,
    )

    for index in range(26):
        for event in vad.process_chunk(np.full(512, index, dtype=np.float32)):
            await policy.handle_vad_event(event)
        await asyncio.sleep(0)

    ends = [event for event in sink.events if isinstance(event, SpeechEnd)]
    assert len(ends) == 1
    assert ends[0].trailing_silence_ms == 800
    assert vad.in_speech is False
    await policy.close()


async def test_active_policy_enforces_wall_clock_hard_end_without_more_vad_chunks() -> None:
    sink = RecordingSink()
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: None),
        inference=RecordingInference(score=0.1),
        config=SmartTurnExperimentConfig(
            stage="active",
            threshold=0.5,
            probe_silence_ms=(10, 20, 30),
            hard_end_ms=40,
        ),
    )
    utterance_id = uuid4()

    await policy.handle_vad_event(
        SpeechStart(
            utterance_id=utterance_id,
            pre_roll=np.empty(0, dtype=np.float32),
            chunk=np.ones(512, dtype=np.float32),
        )
    )
    await policy.handle_vad_event(
        SpeechChunk(
            utterance_id=utterance_id,
            chunk=np.zeros(512, dtype=np.float32),
            is_speech=False,
        )
    )
    await asyncio.sleep(0.06)

    ends = [event for event in sink.events if isinstance(event, SpeechEnd)]
    assert len(ends) == 1
    assert ends[0].trailing_silence_ms == 40
    await policy.close()


async def test_shadow_policy_does_not_change_vad_boundary() -> None:
    sink = RecordingSink()
    inference = RecordingInference(score=0.9)
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: None),
        inference=inference,
        config=_config("shadow"),
    )
    utterance_id = uuid4()

    await _send_start_and_silence(policy, utterance_id, 7)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not any(isinstance(event, SpeechEnd) for event in sink.events)

    await policy.handle_vad_event(
        SpeechEnd(utterance_id=utterance_id, trailing_silence_ms=512, reason="silence")
    )
    ends = [event for event in sink.events if isinstance(event, SpeechEnd)]
    assert len(ends) == 1
    assert ends[0].trailing_silence_ms == 512


async def test_stale_probe_result_cannot_end_resumed_speech() -> None:
    sink = RecordingSink()
    inference = BlockingInference()
    policy = SmartTurnEndpointPolicy(
        downstream=sink,
        vad_control=_ResetControl(lambda: None),
        inference=inference,
        config=_config("active"),
    )
    utterance_id = uuid4()

    await _send_start_and_silence(policy, utterance_id, 7)
    await asyncio.wait_for(inference.started.wait(), timeout=0.5)
    await policy.handle_vad_event(
        SpeechChunk(
            utterance_id=utterance_id,
            chunk=np.ones(512, dtype=np.float32),
            is_speech=True,
        )
    )
    inference.release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not any(isinstance(event, SpeechEnd) for event in sink.events)


async def test_onnx_inference_feeds_expected_whisper_tensor(monkeypatch, tmp_path) -> None:
    captured: list[dict[str, np.ndarray]] = []

    class FakeSessionOptions:
        execution_mode = None
        inter_op_num_threads = None
        intra_op_num_threads = None
        graph_optimization_level = None

    class FakeSession:
        def __init__(self, _path, *, sess_options) -> None:
            self.options = sess_options

        def run(self, _outputs, inputs):
            captured.append(inputs)
            return [np.array([0.75], dtype=np.float32)]

    fake_onnxruntime = SimpleNamespace(
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=FakeSession,
        SessionOptions=FakeSessionOptions,
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_onnxruntime)
    model_path = tmp_path / "smart-turn.onnx"
    model_path.write_bytes(b"model")
    model = SmartTurnOnnxInference(model_path)

    prediction = await model.predict(np.ones(320, dtype=np.float32), sample_rate_hz=16000)

    assert prediction.score == 0.75
    tensor = captured[0]["input_features"]
    assert tensor.shape == (1, 80, 800)
    assert tensor.dtype == np.float32
