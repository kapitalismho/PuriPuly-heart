from __future__ import annotations

import argparse
import asyncio
import copy
import threading
from pathlib import Path

import pytest

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.release_evidence.unattended_runtime import (
    QWEN_BASELINE_METRICS,
    QWEN_STAGES,
    REQUIRED_STAGES,
    ProductSourceProbeError,
    _active_peer_publication_gate,
    _evidence_scope,
    _EvidenceDiagnosticsSink,
    _peer_generation_probe,
    _product_process_source_probe,
    compare_baseline,
    execute,
    map_process_report,
    new_report,
    run_deterministic,
    run_local_qwen,
    validate_report,
)


@pytest.mark.asyncio
async def test_deterministic_report_executes_queue_stale_and_cleanup_scenario() -> None:
    report = await run_deterministic()

    validate_report(report)
    assert report["status"] == "passed"
    assert report["stages"]["queue"]["facts"] == {
        "capacity": 2,
        "accepted": 2,
        "processed": 1,
    }
    assert report["stages"]["stale_result"]["facts"]["rejected"] == 1
    assert report["stages"]["cleanup"]["facts"] == {"queue_size": 0, "tasks": 0}


def _native_pass() -> dict[str, object]:
    return {
        "schema": "puripuly-heart/windows-process-isolation/v1",
        "status": "passed",
        "classification": None,
        "supported_target": {
            "system": "Windows",
            "implementation": "CPython",
            "python": "3.12",
            "machine": "AMD64",
            "minimum_windows_build": 19041,
        },
        "host": {
            "system": "Windows",
            "implementation": "CPython",
            "python": "3.12",
            "machine": "AMD64",
            "windows_build": 22631,
        },
        "credential_free": True,
        "network_used": False,
        "thresholds": {},
        "fixture": {
            "ready_and_activation_order_s": {"root_proctap_activations": [0.25, 0.5]},
        },
        "measurements": {},
        "lifecycle": {},
        "capture_construction": {},
    }


def _probe() -> dict[str, object]:
    return {
        "cold_enumeration_ms": 3.0,
        "warm_enumeration_ms": 1.0,
        "cold_count": 10,
        "warm_count": 10,
        "cold_fingerprint": "a" * 64,
        "warm_fingerprint": "a" * 64,
        "cold_invocation": 1,
        "warm_invocation": 2,
        "enumeration_fresh": True,
        "queue_submitted": 3,
        "queue_dropped": 1,
        "stale_submitted": 2,
        "stale_rejected": 2,
        "process_before": 0,
        "process_peak": 3,
        "process_after": 0,
        "threads_before": 4,
        "threads_peak": 8,
        "threads_after": 4,
        "rss_before": 100,
        "rss_peak": 200,
        "rss_after": 110,
        "cleanup_complete": True,
        "activation_ms": 500.0,
    }


@pytest.mark.asyncio
async def test_product_source_probe_uses_owned_callback_queue_and_exact_cleanup() -> None:
    facts = await _product_process_source_probe()

    assert facts["submitted"] == 2
    assert facts["dropped"] == 1
    assert facts["queue_size_before_close"] == 1
    assert all(
        facts[key] for key in ("source_closed", "capture_closed", "watch_closed", "queue_closed")
    )


@pytest.mark.asyncio
async def test_product_source_probe_consumer_failure_still_closes_all_owned_resources() -> None:
    async def fail(_samples) -> None:  # noqa: ANN001
        raise RuntimeError("consumer failed")

    with pytest.raises(ProductSourceProbeError) as raised:
        await _product_process_source_probe(fail)

    assert raised.value.stage == "consumer"
    assert all(
        raised.value.facts[key]
        for key in ("source_closed", "capture_closed", "watch_closed", "queue_closed")
    )


@pytest.mark.asyncio
async def test_product_source_probe_close_failure_is_attributable_cleanup_failure() -> None:
    with pytest.raises(ProductSourceProbeError) as raised:
        await _product_process_source_probe(capture_close_failure=True)

    assert raised.value.stage == "cleanup"
    assert raised.value.facts["capture_closed"] is False
    assert raised.value.facts["source_closed"] is True


@pytest.mark.asyncio
async def test_correlated_product_source_overlap_drop_and_stale_suppression() -> None:
    started = threading.Event()
    release = threading.Event()
    publish, supersede, runtime, peer_facts = _active_peer_publication_gate()

    async def delayed_consumer(_samples) -> None:  # noqa: ANN001
        started.set()
        await asyncio.to_thread(release.wait)
        await publish()

    facts = await _product_process_source_probe(
        delayed_consumer,
        wait_inflight=lambda: asyncio.to_thread(started.wait, 2.0),
        supersede=supersede,
        release_inflight=release.set,
    )

    assert facts["overlap_observed"] is True
    assert facts["dropped"] == 1
    assert peer_facts["attempted"] == 1
    assert peer_facts["published"] == 0
    assert runtime.loop_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["decode", "monitor"])
async def test_scoped_task_failure_keeps_original_error_and_records_safe_diagnostic(
    owner: str,
) -> None:
    sink = _EvidenceDiagnosticsSink()
    scope = _evidence_scope(f"evidence-{owner}-failure", sink)

    async def fail() -> None:
        raise RuntimeError(f"raw {owner} payload")

    task = start_lifecycle_task(scope, fail(), name=owner)
    with pytest.raises(RuntimeError, match=f"raw {owner} payload"):
        await task
    await scope.close()

    assert sink.classifications
    assert all("payload" not in classification for classification in sink.classifications)


@pytest.mark.asyncio
async def test_qwen_cancellation_while_decode_blocked_closes_backend_source_and_scopes(
    tmp_path: Path, monkeypatch
) -> None:
    from puripuly_heart.core.audio.process_source import ProcessAudioCaptureSource
    from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend

    closed = {"backend": 0, "source": 0, "scope": 0}
    supersede_started = asyncio.Event()

    monkeypatch.setattr(
        "puripuly_heart.core.local_stt_assets.validate_local_stt_runtime_ready",
        lambda _path: None,
    )

    async def ensure(self):  # noqa: ANN001
        self._recognizer = object()
        return self._recognizer

    def decode(self, recognizer, samples):  # noqa: ANN001
        return ""

    async def close_backend(self):  # noqa: ANN001
        closed["backend"] += 1
        self._recognizer = None

    original_source_close = ProcessAudioCaptureSource.close
    original_scope_close = LifecycleScope.close

    async def close_source(self):  # noqa: ANN001
        await original_source_close(self)
        closed["source"] += 1

    async def close_scope(self):  # noqa: ANN001
        await original_scope_close(self)
        closed["scope"] += 1

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_decode_f32_sync", decode)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "close", close_backend)
    monkeypatch.setattr(ProcessAudioCaptureSource, "close", close_source)
    monkeypatch.setattr(LifecycleScope, "close", close_scope)

    async def publish() -> None:
        return None

    async def blocked_supersede() -> None:
        supersede_started.set()
        await asyncio.Event().wait()

    fake_runtime = type("Runtime", (), {"loop_task": None})()
    monkeypatch.setattr(
        "puripuly_heart.release_evidence.unattended_runtime._active_peer_publication_gate",
        lambda: (
            publish,
            blocked_supersede,
            fake_runtime,
            {"generation_before": 0, "attempted": 0, "published": 0},
        ),
    )

    task = asyncio.create_task(run_local_qwen(tmp_path))
    await asyncio.wait_for(supersede_started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed["backend"] == 1
    assert closed["source"] == 1
    assert closed["scope"] >= 1


@pytest.mark.asyncio
async def test_peer_generation_probe_uses_runtime_sink_and_suppresses_publication() -> None:
    facts = await _peer_generation_probe()

    assert facts["generation_after"] > facts["generation_before"]
    assert facts["attempted"] == facts["rejected"] == 1
    assert facts["published"] == 0
    assert facts["loop_task_released"] is True


def test_process_success_maps_actual_native_facts() -> None:
    report = map_process_report(_native_pass(), 0, _probe())

    assert report["status"] == "blocked"
    assert report["stages"]["cold_enumeration"]["facts"]["milliseconds"] == 3.0
    assert report["stages"]["activation"]["facts"]["milliseconds"] == 250.0
    assert report["stages"]["queue"]["facts"]["dropped"] == 1
    assert report["stages"]["stale_result"]["facts"]["rejected"] == 2
    assert report["stages"]["cleanup"]["status"] == "passed"


def test_process_incomplete_probe_is_failed() -> None:
    probe = _probe()
    del probe["queue_dropped"]

    report = map_process_report(_native_pass(), 0, probe)

    assert report["status"] == "failed"
    assert report["stages"]["prerequisites"]["classification"] == "process_probe_incomplete"


def test_process_failed_report_attributes_capture_stage() -> None:
    native = _native_pass()
    native["status"] = "failed"
    native["classification"] = "capture_timeout"
    report = map_process_report(native, 1, _probe())

    assert report["status"] == "failed"
    assert report["stages"]["activation"]["classification"] == "capture_timeout"
    assert report["stages"]["prerequisites"]["status"] == "passed"
    assert report["stages"]["cleanup"]["status"] == "blocked"
    assert report["stages"]["cleanup"]["facts"]["harness_owned_complete"] is True


@pytest.mark.parametrize(
    "mutation",
    ["schema", "credential", "network", "host", "field"],
)
def test_process_malformed_security_and_provenance_are_failed(mutation: str) -> None:
    native = _native_pass()
    if mutation == "schema":
        native["schema"] = "wrong"
    elif mutation == "credential":
        native["credential_free"] = False
    elif mutation == "network":
        native["network_used"] = True
    elif mutation == "host":
        native["host"]["machine"] = "ARM64"  # type: ignore[index]
    else:
        native["extra"] = True
    report = map_process_report(native, 0, _probe())

    assert report["status"] == "failed"


def test_qwen_schema_and_passed_fact_validation() -> None:
    report = new_report("local_qwen")
    assert set(REQUIRED_STAGES + QWEN_STAGES) <= report["stages"].keys()
    report["stages"]["load"] = {"status": "passed", "classification": "bad", "facts": {}}
    with pytest.raises(ValueError, match="no facts"):
        validate_report(report)


def test_qwen_stale_schema_requires_observed_facts_for_pass() -> None:
    report = new_report("local_qwen")
    report["stages"]["stale_result"] = {
        "status": "passed",
        "classification": "benchmark_generation_gate_observed",
        "facts": {},
    }
    with pytest.raises(ValueError, match="no facts"):
        validate_report(report)

    report["stages"]["stale_result"]["facts"] = {
        "submitted": 2,
        "rejected": 2,
        "active_generation": 2,
    }
    validate_report(report)


def _comparable_report() -> dict[str, object]:
    report = new_report("local_qwen")
    report["stages"]["load"]["facts"] = {"milliseconds": 100.0}
    report["stages"]["inference"]["facts"] = {"max_milliseconds": 200.0}
    report["stages"]["rtf"]["facts"] = {"max_value": 0.2}
    report["stages"]["rss"]["facts"] = {"delta_bytes": 1000}
    return report


def test_baseline_requires_approval_identity_and_every_metric_without_mutation() -> None:
    report = _comparable_report()
    baseline = {
        "approved": True,
        "identity": {"environment": report["environment"], "model": report["model"]},
        "metrics": {"load_ms": 100, "inference_ms": 200, "rtf": 0.2, "rss_delta_bytes": 1000},
    }
    original = copy.deepcopy(baseline)

    assert compare_baseline(report, None)["status"] == "absent"
    assert compare_baseline(report, {**baseline, "approved": False})["status"] == "unapproved"
    incompatible = copy.deepcopy(baseline)
    incompatible["identity"]["environment"] = {}
    assert compare_baseline(report, incompatible)["status"] == "incompatible"
    incomplete = copy.deepcopy(baseline)
    del incomplete["metrics"][QWEN_BASELINE_METRICS[-1]]
    assert compare_baseline(report, incomplete)["status"] == "not_comparable"
    assert compare_baseline(report, baseline)["status"] == "passed"
    assert baseline == original


def test_process_baseline_requires_all_timing_and_memory_metrics() -> None:
    report = map_process_report(_native_pass(), 0, _probe())
    baseline = {
        "approved": True,
        "identity": {"environment": report["environment"], "model": None},
        "metrics": {
            "cold_enumeration_ms": 3.0,
            "warm_enumeration_ms": 1.0,
            "activation_ms": 250.0,
            "peak_rss_bytes": 200,
        },
    }

    assert compare_baseline(report, baseline)["status"] == "passed"
    del baseline["metrics"]["activation_ms"]
    assert compare_baseline(report, baseline)["status"] == "not_comparable"


@pytest.mark.asyncio
async def test_runner_rejects_same_baseline_alias_without_mutating_baseline(tmp_path: Path) -> None:
    output = tmp_path / "same.json"
    output.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        target="deterministic",
        evidence=output,
        model_dir=None,
        baseline=output,
        regression_allowance=0.1,
        thresholds=tmp_path / "thresholds.json",
    )

    original = output.read_bytes()

    with pytest.raises(ValueError, match="baseline_evidence_same_file"):
        await execute(args)
    assert output.read_bytes() == original


@pytest.mark.asyncio
async def test_malformed_baseline_is_written_as_non_pass(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text("not json", encoding="utf-8")
    args = argparse.Namespace(
        target="deterministic",
        evidence=output,
        model_dir=None,
        baseline=baseline,
        regression_allowance=0.1,
        thresholds=tmp_path / "thresholds.json",
    )

    report = await execute(args)

    assert output.is_file()
    assert report["baseline"]["status"] == "malformed"
