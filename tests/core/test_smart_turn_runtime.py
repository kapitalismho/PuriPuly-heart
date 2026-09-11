from __future__ import annotations

import asyncio
import hashlib
import threading
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio import smart_turn
from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_INPUT_REVISION,
    SMART_TURN_MODEL_SHA256,
    SmartTurnInferenceOwner,
    SmartTurnRequestIdentity,
    prepare_smart_turn_audio,
    smart_turn_language_profile,
)
from puripuly_heart.core.audio.smart_turn_features import compute_whisper_log_mel_features


def test_frozen_language_profiles_and_input_window() -> None:
    assert smart_turn_language_profile("manual", "ko") == ("on", 0.967305183)
    assert smart_turn_language_profile("manual", "ja-JP") == ("on", 0.844703436)
    assert smart_turn_language_profile("manual", "en") == ("on", 0.772239923)
    assert smart_turn_language_profile("manual", "zh-CN") == ("on", 0.925585747)
    assert smart_turn_language_profile("auto", "en") == ("unsupported_auto", None)
    assert smart_turn_language_profile("manual", "fr") == ("unsupported_language", None)

    short = np.arange(16000, dtype=np.float32)
    prepared = prepare_smart_turn_audio(short, sample_rate_hz=16000)
    assert prepared.shape == (128000,)
    np.testing.assert_array_equal(prepared[:112000], 0.0)
    np.testing.assert_array_equal(prepared[-16000:], short)

    long = np.arange(130000, dtype=np.float32)
    np.testing.assert_array_equal(
        prepare_smart_turn_audio(long, sample_rate_hz=16000),
        long[-128000:],
    )


def test_pinned_input_fixture_identity_matches_authoritative_revision() -> None:
    raw = (
        0.25 * np.sin(2.0 * np.pi * 220.0 * np.arange(16000, dtype=np.float64) / 16000.0)
    ).astype(np.float32)
    prepared = prepare_smart_turn_audio(raw, sample_rate_hz=16000)
    features = compute_whisper_log_mel_features(prepared)

    assert hashlib.sha256(raw.tobytes()).hexdigest() == (
        "b30dde97f347b9f64623089f9395a253f69c4213fda97bb045a558cf734a0755"
    )
    assert hashlib.sha256(prepared.tobytes()).hexdigest() == (
        "de37fbd4b1f1b91ea182fccb6e281065ac5304f7e3d932941a8d8e074daab4c7"
    )
    assert hashlib.sha256(features.tobytes()).hexdigest() == (
        "619865d13db4e64a0640e3d613e021f6971b7e53a03b76a8eaa65a54879f4d52"
    )
    assert features.shape == (80, 800)


@pytest.mark.asyncio
async def test_cached_artifact_is_unloaded_until_owned_prepare_actually_starts(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    monkeypatch.setattr(smart_turn, "_sha256_file", lambda _path: SMART_TURN_MODEL_SHA256)
    entered = threading.Event()
    release = threading.Event()

    def factory(_path):
        entered.set()
        assert release.wait(5.0)
        return BlockingInference()

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=factory)
    assert owner.snapshot.availability == "unloaded"
    assert not entered.is_set()

    owner.request_prepare()
    assert owner.snapshot.availability == "loading"
    assert await asyncio.to_thread(entered.wait, 1.0)
    assert owner.snapshot.availability == "loading"
    release.set()
    async with asyncio.timeout(1.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)
    await owner.close()


@pytest.mark.asyncio
async def test_total_prepare_watchdog_reports_timeout_and_reclaims_late_resource(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    monkeypatch.setattr(smart_turn, "_sha256_file", lambda _path: SMART_TURN_MODEL_SHA256)
    entered = threading.Event()
    release = threading.Event()
    resources = []
    calls = 0

    class Inference:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    def factory(_path):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(5.0)
        resource = Inference()
        resources.append(resource)
        return resource

    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=factory,
        prepare_timeout_s=0.01,
    )
    owner.request_prepare()
    assert await asyncio.to_thread(entered.wait, 1.0)
    async with asyncio.timeout(1.0):
        while owner.snapshot.last_error != "TimeoutError":
            await asyncio.sleep(0.001)
    assert owner.snapshot.availability == "error"
    owner.request_prepare()
    assert calls == 1

    release.set()
    await owner.close()
    assert len(resources) == 1
    assert resources[0].close_calls == 1


class BlockingInference:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return 0.9


@pytest.mark.asyncio
async def test_one_executing_resource_has_no_pending_queue_or_replacement(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    monkeypatch.setattr(smart_turn, "_sha256_file", lambda _path: SMART_TURN_MODEL_SHA256)
    inference = BlockingInference()
    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=lambda _path: inference,
    )
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)

    identity = SmartTurnRequestIdentity(
        activation_generation=1,
        segment_id=uuid4(),
        pause_id=1,
        context_revision=1,
        input_revision=SMART_TURN_INPUT_REVISION,
        source_frontier=3584,
        probe_frontier_monotonic_s=0.224,
        complete_deadline_monotonic_s=0.512,
    )
    completions = []

    async def receive(completion) -> None:
        completions.append(completion)

    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "started"
    await inference.started.wait()
    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "busy"
    assert owner.snapshot.busy_skip_count == 1
    assert inference.calls == 1
    inference.release.set()
    await owner.close()
    assert len(completions) == 1
    assert owner.snapshot.inference_count == 1
    assert owner.snapshot.availability == "closed"


@pytest.mark.asyncio
async def test_cancelled_close_retains_blocked_native_setup_until_reclaimed(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    monkeypatch.setattr(smart_turn, "_sha256_file", lambda _path: SMART_TURN_MODEL_SHA256)
    started = threading.Event()
    release = threading.Event()
    constructed = []
    calls = 0

    class Inference:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    def factory(_path):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(5.0):
            raise TimeoutError("setup release was not signaled")
        inference = Inference()
        constructed.append(inference)
        return inference

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=factory)
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while not started.is_set():
            await asyncio.sleep(0.001)

    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    owner.request_prepare()
    assert calls == 1

    release.set()
    await owner.close()
    assert len(constructed) == 1
    assert constructed[0].close_calls == 1
    assert owner.snapshot.availability == "closed"


@pytest.mark.asyncio
async def test_failed_setup_is_not_retried_by_later_pause_requests(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    monkeypatch.setattr(smart_turn, "_sha256_file", lambda _path: SMART_TURN_MODEL_SHA256)
    calls = 0

    def factory(_path):
        nonlocal calls
        calls += 1
        raise RuntimeError("setup failed")

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=factory)
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "error":
            await asyncio.sleep(0.001)
    owner.request_prepare()
    owner.request_prepare()
    await asyncio.sleep(0.01)
    assert calls == 1
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_close_retains_blocked_native_inference_until_reclaimed(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    monkeypatch.setattr(smart_turn, "_sha256_file", lambda _path: SMART_TURN_MODEL_SHA256)
    started = threading.Event()
    release = threading.Event()

    class Inference:
        def __init__(self) -> None:
            self.close_calls = 0

        async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
            def run() -> float:
                started.set()
                if not release.wait(5.0):
                    raise TimeoutError("inference release was not signaled")
                return 0.9

            return await asyncio.to_thread(run)

        def close(self) -> None:
            self.close_calls += 1

    inference = Inference()
    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=lambda _path: inference,
    )
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)
    identity = SmartTurnRequestIdentity(
        activation_generation=1,
        segment_id=uuid4(),
        pause_id=1,
        context_revision=1,
        input_revision=SMART_TURN_INPUT_REVISION,
        source_frontier=3584,
        probe_frontier_monotonic_s=0.224,
        complete_deadline_monotonic_s=0.512,
    )
    completions = []

    async def receive(completion) -> None:
        completions.append(completion)

    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "started"
    async with asyncio.timeout(5.0):
        while not started.is_set():
            await asyncio.sleep(0.001)

    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "unavailable"

    release.set()
    await owner.close()
    assert len(completions) == 1
    assert owner.snapshot.inference_count == 1
    assert inference.close_calls == 1
    assert owner.snapshot.availability == "closed"
