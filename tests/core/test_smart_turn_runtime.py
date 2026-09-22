from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio import smart_turn
from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_COMPLETE_THRESHOLD,
    SMART_TURN_INPUT_REVISION,
    SMART_TURN_SUPPORTED_LANGUAGES,
    SmartTurnInferenceOwner,
    SmartTurnOnnxInference,
    SmartTurnRequestIdentity,
    prepare_smart_turn_audio,
    smart_turn_language_profile,
)
from puripuly_heart.core.audio.smart_turn_features import compute_whisper_log_mel_features
from puripuly_heart.core.language import SUPPORTED_LANGUAGES


def test_product_and_official_language_intersection_profiles_and_input_window() -> None:
    assert SMART_TURN_SUPPORTED_LANGUAGES == {
        "ar",
        "zh",
        "da",
        "nl",
        "de",
        "en",
        "fi",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "no",
        "pl",
        "pt",
        "ru",
        "es",
        "tr",
        "uk",
        "vi",
    }
    selectable_language_bases = {
        language.split("-", 1)[0].lower() for language in SUPPORTED_LANGUAGES
    }
    assert SMART_TURN_SUPPORTED_LANGUAGES <= selectable_language_bases
    for language in SMART_TURN_SUPPORTED_LANGUAGES:
        assert smart_turn_language_profile("manual", language) == (
            "on",
            SMART_TURN_COMPLETE_THRESHOLD,
        )
    assert smart_turn_language_profile("auto", "en") == (
        "on",
        SMART_TURN_COMPLETE_THRESHOLD,
    )
    assert smart_turn_language_profile("auto", "bg") == (
        "on",
        SMART_TURN_COMPLETE_THRESHOLD,
    )
    assert smart_turn_language_profile("manual", "bg") == ("unsupported_language", None)

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
async def test_missing_bundle_fails_locally_without_creating_download_files(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        smart_turn, "SMART_TURN_RESOURCE_RELATIVE_PATH", str(tmp_path / "absent-bundle.onnx")
    )
    owner = SmartTurnInferenceOwner()
    try:
        owner.request_prepare()
        async with asyncio.timeout(2.0):
            while owner.snapshot.availability == "loading":
                await asyncio.sleep(0)
        assert owner.snapshot.availability == "error"
        assert owner.snapshot.last_error == "FileNotFoundError"
        assert list(tmp_path.iterdir()) == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_local_artifact_is_unloaded_until_owned_prepare_actually_starts(
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
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
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
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
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
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
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
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
async def test_failed_setup_is_not_retried_by_later_pause_requests(tmp_path) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
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
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
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


def test_prepare_window_exact_empty_and_invalid_contracts() -> None:
    exact = np.arange(128000, dtype=np.float32)
    prepared = prepare_smart_turn_audio(exact, sample_rate_hz=16000)
    assert prepared.shape == (128000,)
    np.testing.assert_array_equal(prepared, exact)
    assert prepared is not exact
    exact[0] = -1.0
    assert prepared[0] != -1.0
    empty = prepare_smart_turn_audio(np.empty(0, dtype=np.float32), sample_rate_hz=16000)
    assert empty.shape == (128000,)
    np.testing.assert_array_equal(empty, np.zeros(128000, dtype=np.float32))
    with pytest.raises(ValueError):
        prepare_smart_turn_audio(np.zeros(10, dtype=np.float32), sample_rate_hz=8000)
    with pytest.raises(ValueError):
        prepare_smart_turn_audio(np.zeros((2, 8), dtype=np.float32), sample_rate_hz=16000)


def test_feature_helper_right_pads_independently_of_prepare_left_padding() -> None:
    raw = (
        0.25 * np.sin(2.0 * np.pi * 220.0 * np.arange(16000, dtype=np.float64) / 16000.0)
    ).astype(np.float32)
    prepared = prepare_smart_turn_audio(raw, sample_rate_hz=16000)
    direct = compute_whisper_log_mel_features(raw)
    from_prepared = compute_whisper_log_mel_features(prepared)
    assert direct.shape == (80, 800)
    assert from_prepared.shape == (80, 800)
    assert not bool((direct == from_prepared).all())


@pytest.mark.asyncio
async def test_submit_freezes_worker_input_against_caller_mutation(tmp_path) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    class Inference:
        async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
            entered.set()
            await release.wait()
            seen.append(audio.copy())
            return 0.2

    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=lambda _path: Inference(),
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

    source = np.ones(3584, dtype=np.float32)
    assert owner.submit(identity, source, receive) == "started"
    async with asyncio.timeout(5.0):
        await entered.wait()
    source[:] = 0.0
    release.set()
    async with asyncio.timeout(5.0):
        while not completions:
            await asyncio.sleep(0.001)
    assert seen[0] is not source
    np.testing.assert_array_equal(seen[0], np.ones(3584, dtype=np.float32))
    assert completions[0].score == 0.2
    assert completions[0].outcome == "complete"


@pytest.mark.asyncio
async def test_error_and_nonfinite_outcomes_complete_with_counts_and_stay_ready(
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")

    class Inference:
        def __init__(self) -> None:
            self.mode = "error"

        async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
            if self.mode == "error":
                raise RuntimeError("predict failed")
            if self.mode == "nan":
                return float("nan")
            if self.mode == "inf":
                return float("inf")
            return 0.2

    inference = Inference()
    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=lambda _path: inference,
    )
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)

    def make_identity(pause_id: int) -> SmartTurnRequestIdentity:
        return SmartTurnRequestIdentity(
            activation_generation=1,
            segment_id=uuid4(),
            pause_id=pause_id,
            context_revision=1,
            input_revision=SMART_TURN_INPUT_REVISION,
            source_frontier=3584,
            probe_frontier_monotonic_s=0.224,
            complete_deadline_monotonic_s=0.512,
        )

    completions = []

    async def receive(completion) -> None:
        completions.append(completion)

    async def submit_and_wait(mode: str, pause_id: int) -> None:
        inference.mode = mode
        assert (
            owner.submit(make_identity(pause_id), np.zeros(3584, dtype=np.float32), receive)
            == "started"
        )
        async with asyncio.timeout(5.0):
            while len(completions) < pause_id:
                await asyncio.sleep(0.001)

    await submit_and_wait("error", 1)
    assert completions[0].outcome == "error"
    assert completions[0].score is None
    assert owner.snapshot.last_error == "RuntimeError"
    assert owner.snapshot.availability == "ready"
    await submit_and_wait("nan", 2)
    assert completions[1].outcome == "nonfinite"
    assert math.isnan(completions[1].score)
    await submit_and_wait("inf", 3)
    assert completions[2].outcome == "nonfinite"
    assert math.isinf(completions[2].score)
    await submit_and_wait("ok", 4)
    assert completions[3].outcome == "complete"
    assert completions[3].score == 0.2
    assert owner.snapshot.inference_count == 4
    await owner.close()


@pytest.mark.asyncio
async def test_two_cancelled_executions_each_retain_worker_until_released(
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    entered = asyncio.Event()
    release = asyncio.Event()

    class Inference:
        def __init__(self) -> None:
            self.calls = 0

        async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
            self.calls += 1
            entered.set()
            await release.wait()
            return 0.2

    inference = Inference()
    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=lambda _path: inference,
    )
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)

    def make_identity(pause_id: int) -> SmartTurnRequestIdentity:
        return SmartTurnRequestIdentity(
            activation_generation=1,
            segment_id=uuid4(),
            pause_id=pause_id,
            context_revision=1,
            input_revision=SMART_TURN_INPUT_REVISION,
            source_frontier=3584,
            probe_frontier_monotonic_s=0.224,
            complete_deadline_monotonic_s=0.512,
        )

    completions = []

    async def receive(completion) -> None:
        completions.append(completion)

    assert owner.submit(make_identity(1), np.zeros(3584, dtype=np.float32), receive) == "started"
    first = owner._execution_task
    assert first is not None
    async with asyncio.timeout(5.0):
        await entered.wait()
    first.cancel()
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()
    assert inference.calls == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(5.0):
            await first
    assert completions == []
    assert owner.snapshot.inference_count == 0
    entered.clear()
    release.clear()
    assert owner.submit(make_identity(2), np.zeros(3584, dtype=np.float32), receive) == "started"
    second = owner._execution_task
    assert second is not None
    async with asyncio.timeout(5.0):
        await entered.wait()
    second.cancel()
    second.cancel()
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(5.0):
            await second
    assert completions == []
    assert owner.snapshot.inference_count == 0
    assert owner.submit(make_identity(3), np.zeros(3584, dtype=np.float32), receive) == "started"
    async with asyncio.timeout(5.0):
        while not completions:
            await asyncio.sleep(0.001)
    assert completions[0].outcome == "complete"
    assert owner.snapshot.inference_count == 1
    await owner.close()
    assert owner.snapshot.availability == "closed"


@pytest.mark.asyncio
async def test_cancellation_during_completion_return_keeps_owned_lifetime(
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")

    class Inference:
        def __init__(self) -> None:
            self.in_completion = asyncio.Event()
            self.completion_release = asyncio.Event()

        async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
            return 0.2

    inference = Inference()
    owner = SmartTurnInferenceOwner(
        model_path=model_path,
        inference_factory=lambda _path: inference,
    )
    owner.request_prepare()
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)

    def make_identity(pause_id: int) -> SmartTurnRequestIdentity:
        return SmartTurnRequestIdentity(
            activation_generation=1,
            segment_id=uuid4(),
            pause_id=pause_id,
            context_revision=1,
            input_revision=SMART_TURN_INPUT_REVISION,
            source_frontier=3584,
            probe_frontier_monotonic_s=0.224,
            complete_deadline_monotonic_s=0.512,
        )

    completions = []

    async def receive(completion) -> None:
        inference.in_completion.set()
        await inference.completion_release.wait()
        completions.append(completion)

    assert owner.submit(make_identity(1), np.zeros(3584, dtype=np.float32), receive) == "started"
    async with asyncio.timeout(5.0):
        await inference.in_completion.wait()
    task = owner._execution_task
    assert task is not None
    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(5.0):
            await task
    inference.completion_release.set()
    assert completions == []
    assert owner.snapshot.inference_count == 1
    await owner.close()
    assert owner.snapshot.availability == "closed"


@pytest.mark.asyncio
async def test_unready_submit_is_unavailable_and_starts_single_prepare(tmp_path) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    calls = 0

    class Inference:
        async def predict(self, audio: np.ndarray, *, sample_rate_hz: int) -> float:
            return 0.2

    def factory(_path):
        nonlocal calls
        calls += 1
        return Inference()

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=factory)
    completions = []

    async def receive(completion) -> None:
        completions.append(completion)

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
    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "unavailable"
    assert owner.snapshot.availability == "loading"
    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "unavailable"
    async with asyncio.timeout(5.0):
        while calls == 0:
            await asyncio.sleep(0.001)
    assert calls == 1
    owner.record_late()
    owner.record_late()
    assert owner.snapshot.late_count == 2
    async with asyncio.timeout(5.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)
    await owner.close()
    assert owner.snapshot.availability == "closed"
    assert owner.submit(identity, np.zeros(3584, dtype=np.float32), receive) == "unavailable"


@pytest.mark.asyncio
async def test_close_while_preparing_reclaims_and_repeated_close_is_idempotent(
    tmp_path,
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    entered = threading.Event()
    release = threading.Event()

    class Inference:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    constructed = []

    def factory(_path):
        entered.set()
        assert release.wait(5.0)
        inference = Inference()
        constructed.append(inference)
        return inference

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=factory)
    owner.request_prepare()
    assert await asyncio.to_thread(entered.wait, 5.0)
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    async with asyncio.timeout(5.0):
        await closing
    assert owner.snapshot.availability == "closed"
    assert len(constructed) == 1
    assert constructed[0].close_calls == 1
    await owner.close()
    assert owner.snapshot.availability == "closed"
    assert constructed[0].close_calls == 1


@pytest.mark.asyncio
async def test_owner_cancel_during_production_features_abandons_owned_attempt(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    entered_features = threading.Event()
    release_features = threading.Event()
    calls = {"features": 0, "onnx": 0}

    def gated_features(prepared):
        from puripuly_heart.core.audio import smart_turn_features as features_module

        calls["features"] += 1
        entered_features.set()
        assert release_features.wait(10.0)
        return features_module.compute_whisper_log_mel_features(prepared)

    class GatedSession:
        def run(self, _names, feeds):
            calls["onnx"] += 1
            return [np.asarray([0.2], dtype=np.float32)]

    monkeypatch.setattr(smart_turn, "compute_whisper_log_mel_features", gated_features)

    def inference_factory(_path):
        inference = SmartTurnOnnxInference.__new__(SmartTurnOnnxInference)
        inference._session = None
        return inference

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=inference_factory)
    owner.request_prepare()
    async with asyncio.timeout(30.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)
    owner._inference._session = GatedSession()
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
    assert await asyncio.to_thread(entered_features.wait, 10.0)
    task = owner._execution_task
    assert task is not None
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_features.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(30.0):
            await task
    assert completions == []
    assert calls["features"] == 1
    assert owner.snapshot.inference_count == 0
    assert owner._execution_task is None
    await owner.close()
    assert owner.snapshot.availability == "closed"


@pytest.mark.asyncio
async def test_owner_cancel_during_production_onnx_abandons_owned_attempt(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"fixture")
    entered_onnx = threading.Event()
    release_onnx = threading.Event()
    calls = {"features": 0, "onnx": 0}

    def gated_features(prepared):
        from puripuly_heart.core.audio import smart_turn_features as features_module

        calls["features"] += 1
        return features_module.compute_whisper_log_mel_features(prepared)

    class GatedSession:
        def run(self, _names, feeds):
            calls["onnx"] += 1
            entered_onnx.set()
            assert release_onnx.wait(10.0)
            return [np.asarray([0.2], dtype=np.float32)]

    monkeypatch.setattr(smart_turn, "compute_whisper_log_mel_features", gated_features)

    def inference_factory(_path):
        inference = SmartTurnOnnxInference.__new__(SmartTurnOnnxInference)
        inference._session = None
        return inference

    owner = SmartTurnInferenceOwner(model_path=model_path, inference_factory=inference_factory)
    owner.request_prepare()
    async with asyncio.timeout(30.0):
        while owner.snapshot.availability != "ready":
            await asyncio.sleep(0.001)
    owner._inference._session = GatedSession()
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
    assert await asyncio.to_thread(entered_onnx.wait, 10.0)
    task = owner._execution_task
    assert task is not None
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_onnx.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(30.0):
            await task
    assert completions == []
    assert calls == {"features": 1, "onnx": 1}
    assert owner.snapshot.inference_count == 0
    assert owner._execution_task is None
    await owner.close()
    assert owner.snapshot.availability == "closed"
