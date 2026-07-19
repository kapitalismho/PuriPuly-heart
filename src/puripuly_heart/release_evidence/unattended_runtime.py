from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

SCHEMA = "puripuly-heart.unattended-runtime-evidence.v2"
STATUSES = {"passed", "failed", "blocked"}
REQUIRED_STAGES = (
    "prerequisites",
    "cold_enumeration",
    "warm_enumeration",
    "activation",
    "queue",
    "thread",
    "stale_result",
    "cleanup",
)
QWEN_STAGES = ("load", "inference", "rtf", "recognizer_count", "queue_drop", "rss")
NATIVE_PROCESS_SCHEMA = "puripuly-heart/windows-process-isolation/v1"
QWEN_BASELINE_METRICS = ("load_ms", "inference_ms", "rtf", "rss_delta_bytes")
PROCESS_BASELINE_METRICS = (
    "cold_enumeration_ms",
    "warm_enumeration_ms",
    "activation_ms",
    "peak_rss_bytes",
)


def _stage(status: str, classification: str, **facts: Any) -> dict[str, Any]:
    return {"status": status, "classification": classification, "facts": facts}


def _safe_classification(prefix: str, exc: BaseException) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", type(exc).__name__)[:80]
    return f"{prefix}_{name}"


class ProductSourceProbeError(RuntimeError):
    def __init__(self, stage: str, facts: dict[str, Any]) -> None:
        super().__init__(stage)
        self.stage = stage
        self.facts = facts


class _EvidenceDiagnosticsSink:
    def __init__(self) -> None:
        self.classifications: list[str] = []

    async def emit_diagnostic(self, event: Any) -> None:
        diagnostics = getattr(event, "diagnostics", None)
        code = getattr(diagnostics, "code", "lifecycle_task_failed")
        self.classifications.append(str(code)[:80])


def _evidence_scope(name: str, sink: _EvidenceDiagnosticsSink | None = None) -> LifecycleScope:
    return LifecycleScope(name, diagnostics_sink=sink or _EvidenceDiagnosticsSink())


async def _product_process_source_probe(
    consumer: Any = None,
    *,
    wait_inflight: Any = None,
    supersede: Any = None,
    release_inflight: Any = None,
    capture_close_failure: bool = False,
) -> dict[str, Any]:
    from puripuly_heart.config.process_capture_platform import ProcessCapturePlatformAvailability
    from puripuly_heart.config.process_capture_resolution import ResolvedProcessCaptureIdentity
    from puripuly_heart.core.audio.process_source import ProcessAudioCaptureSource

    @dataclass
    class Capture:
        on_data: Any
        started: bool = False
        closed: bool = False

        def start(self) -> None:
            self.started = True

        def close(self) -> None:
            if capture_close_failure:
                raise RuntimeError("fixture_capture_close_failed")
            self.closed = True

    class CaptureFactory:
        capture: Capture | None = None

        def create(self, *, pid: int, on_data: Any) -> Capture:
            self.capture = Capture(on_data)
            return self.capture

    @dataclass
    class Watch:
        identity_verified: bool = True
        closed: bool = False

        def close(self) -> None:
            self.closed = True

    class Watcher:
        watch_result = Watch()

        def watch(self, identity: Any, on_terminal: Any) -> Watch:
            return self.watch_result

    facts: dict[str, Any] = {
        "submitted": 0,
        "dropped": 0,
        "queue_size_before_close": 0,
        "consumed": 0,
        "overlap_observed": False,
        "source_closed": False,
        "capture_closed": False,
        "watch_closed": False,
        "queue_closed": False,
    }
    capture_factory = CaptureFactory()
    watcher = Watcher()
    source = None
    capture = None
    consumer_task = None
    failure_stage = None
    try:
        identity = ResolvedProcessCaptureIdentity(
            pid=os.getpid(), target=object(), instance_id="fixture"
        )
        source = ProcessAudioCaptureSource(
            identity=identity,
            watcher=watcher,
            max_queue_frames=1,
            capture_factory=capture_factory,
            platform_availability=lambda: ProcessCapturePlatformAvailability(True, None),
        )
        capture = capture_factory.capture
        if capture is None:
            raise RuntimeError("product_source_capture_missing")
        frame = np.zeros((16, 2), dtype=np.float32).tobytes()
        capture.on_data(frame, 16)
        facts["submitted"] += 1
        if consumer is None:
            capture.on_data(frame, 16)
            facts["submitted"] += 1
        else:
            product_frame = await anext(source.frames())
            scope = _evidence_scope("evidence-product-source")
            try:
                consumer_task = start_lifecycle_task(
                    scope, consumer(product_frame.samples), name="consumer"
                )
                if wait_inflight is not None:
                    in_flight = await wait_inflight()
                    if in_flight is False:
                        raise TimeoutError("consumer_inflight_timeout")
                    facts["overlap_observed"] = not consumer_task.done()
                capture.on_data(frame, 16)
                capture.on_data(frame, 16)
                facts["submitted"] += 2
                if supersede is not None:
                    await supersede()
                if release_inflight is not None:
                    release_inflight()
                await consumer_task
            finally:
                await scope.close()
            facts["consumed"] = 1
        facts["dropped"] = source.queue_drop_count
        facts["queue_size_before_close"] = source._queue.sync_q.qsize()
    except Exception:
        failure_stage = "consumer"
    finally:
        if release_inflight is not None:
            release_inflight()
        if consumer_task is not None and not consumer_task.done():
            await asyncio.gather(consumer_task, return_exceptions=True)
        if source is not None:
            try:
                await source.close()
            except Exception:
                failure_stage = "cleanup"
        facts["source_closed"] = bool(source is not None and source._closed)
        facts["capture_closed"] = bool(capture is not None and capture.closed)
        facts["watch_closed"] = watcher.watch_result.closed
        facts["queue_closed"] = bool(source is not None and source._queue.closed)
        if not all(
            facts[key]
            for key in ("source_closed", "capture_closed", "watch_closed", "queue_closed")
        ):
            failure_stage = "cleanup"
    if failure_stage is not None:
        raise ProductSourceProbeError(failure_stage, facts)
    return facts


def _peer_probe_owner(hub: object) -> Any:
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner

    class Admission:
        async def admit(self, _config: object) -> object:
            raise AssertionError("admission is outside this probe")

    class Resolver:
        async def resolve(self, _target: object) -> object:
            raise AssertionError("target resolution is outside this probe")

    class Provider:
        def is_ready(self, _config: object) -> bool:
            return False

        async def release(self, **_kwargs: object) -> None:
            return None

    return PeerCaptureSessionOwner(
        admission=Admission(),
        target_resolver=Resolver(),
        provider=Provider(),
        clock=SystemClock(),
        provider_request_factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("provider construction is outside this probe")
        ),
        source_factory=lambda *_args: None,
        vad_factory=lambda *_args: None,
        run_audio_loop=lambda **_kwargs: asyncio.sleep(0),
        vad_sink=hub,
    )


async def _peer_generation_probe() -> dict[str, Any]:

    class Hub:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def handle_peer_vad_event(self, event: object) -> None:
            self.events.append(event)

    hub = Hub()
    runtime = _peer_probe_owner(hub)
    generation = runtime._generation
    sink = runtime.guard_vad_sink(generation)
    await runtime.close()
    attempted = 1
    await sink.handle_vad_event(object())
    return {
        "attempted": attempted,
        "published": len(hub.events),
        "rejected": attempted - len(hub.events),
        "generation_before": generation,
        "generation_after": runtime._generation,
        "loop_task_released": runtime.loop_task is None,
    }


def _active_peer_publication_gate() -> tuple[Any, Any, Any, dict[str, Any]]:
    from puripuly_heart.core.peer_capture import PeerCaptureSessionState

    class Hub:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def handle_peer_vad_event(self, event: object) -> None:
            self.events.append(event)

    hub = Hub()
    runtime = _peer_probe_owner(hub)
    runtime._desired_active = True
    runtime._state = PeerCaptureSessionState.RUNNING
    runtime._loop_task = asyncio.current_task()
    generation = runtime._generation
    sink = runtime.guard_vad_sink(generation)
    facts = {"generation_before": generation, "attempted": 0, "published": 0}

    async def publish() -> None:
        facts["attempted"] += 1
        await sink.handle_vad_event(object())
        facts["published"] = len(hub.events)

    async def supersede() -> None:
        await runtime.close()
        facts["generation_after"] = runtime._generation

    return publish, supersede, runtime, facts


def environment_identity() -> dict[str, Any]:
    windows_build = getattr(getattr(sys, "getwindowsversion", lambda: None)(), "build", None)
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "windows_build": windows_build,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "cpu_logical_count": os.cpu_count(),
    }


def model_identity(model_dir: Path | None) -> dict[str, Any] | None:
    if model_dir is None:
        return None
    manifest = model_dir / "installed-manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
    return {"name": model_dir.name, "installed_manifest_sha256": digest}


def new_report(target: str, model_dir: Path | None = None) -> dict[str, Any]:
    names = REQUIRED_STAGES + (QWEN_STAGES if target == "local_qwen" else ())
    return {
        "schema": SCHEMA,
        "target": target,
        "status": "blocked",
        "environment": environment_identity(),
        "model": model_identity(model_dir),
        "stages": {name: _stage("blocked", "not_run") for name in names},
        "baseline": {"status": "not_required", "comparisons": None},
    }


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != SCHEMA or report.get("status") not in STATUSES:
        raise ValueError("invalid evidence envelope")
    required = REQUIRED_STAGES + (QWEN_STAGES if report.get("target") == "local_qwen" else ())
    stages = report.get("stages")
    if not isinstance(stages, dict) or set(required) - set(stages):
        raise ValueError("incomplete evidence stages")
    for name in required:
        stage = stages[name]
        if not isinstance(stage, dict) or stage.get("status") not in STATUSES:
            raise ValueError(f"invalid stage status: {name}")
        if not isinstance(stage.get("classification"), str) or not isinstance(
            stage.get("facts"), dict
        ):
            raise ValueError(f"invalid stage facts: {name}")
        if stage["status"] == "passed" and not stage["facts"]:
            raise ValueError(f"passed stage has no facts: {name}")


def _metrics(report: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, float]]:
    stages = report["stages"]
    if report["target"] == "process_capture":
        values = {
            "cold_enumeration_ms": stages["cold_enumeration"]["facts"].get("milliseconds"),
            "warm_enumeration_ms": stages["warm_enumeration"]["facts"].get("milliseconds"),
            "activation_ms": stages["activation"]["facts"].get("milliseconds"),
            "peak_rss_bytes": stages["thread"]["facts"].get("peak_rss_bytes"),
        }
        return PROCESS_BASELINE_METRICS, {
            key: float(value) for key, value in values.items() if isinstance(value, (int, float))
        }
    values = {
        "load_ms": stages["load"]["facts"].get("milliseconds"),
        "inference_ms": stages["inference"]["facts"].get("max_milliseconds"),
        "rtf": stages["rtf"]["facts"].get("max_value"),
        "rss_delta_bytes": stages["rss"]["facts"].get("delta_bytes"),
    }
    return QWEN_BASELINE_METRICS, {
        key: float(value) for key, value in values.items() if isinstance(value, (int, float))
    }


def compare_baseline(
    report: dict[str, Any], baseline: dict[str, Any] | None, *, allowance: float = 0.10
) -> dict[str, Any]:
    if baseline is None:
        return {"status": "absent", "comparisons": None}
    if not isinstance(baseline, dict) or baseline.get("approved") is not True:
        return {"status": "unapproved", "comparisons": None}
    identity = {"environment": report["environment"], "model": report["model"]}
    if baseline.get("identity") != identity:
        return {"status": "incompatible", "comparisons": None}
    required_metrics, current = _metrics(report)
    approved = baseline.get("metrics")
    if not isinstance(approved, dict) or any(
        metric not in current or not isinstance(approved.get(metric), (int, float))
        for metric in required_metrics
    ):
        return {"status": "not_comparable", "comparisons": None}
    comparisons = {}
    regressed = False
    for metric in required_metrics:
        limit = float(approved[metric]) * (1.0 + allowance)
        passed = current[metric] <= limit
        regressed = regressed or not passed
        comparisons[metric] = {
            "current": current[metric],
            "approved": float(approved[metric]),
            "limit": limit,
            "passed": passed,
        }
    return {"status": "regressed" if regressed else "passed", "comparisons": comparisons}


def _finalize(report: dict[str, Any]) -> None:
    statuses = [stage["status"] for stage in report["stages"].values()]
    report["status"] = (
        "failed" if "failed" in statuses else "blocked" if "blocked" in statuses else "passed"
    )
    if (
        report["target"] in {"local_qwen", "process_capture"}
        and report["baseline"]["status"] != "passed"
    ):
        if report["baseline"]["status"] == "regressed":
            report["status"] = "failed"
        elif report["status"] == "passed":
            report["status"] = "blocked"
    validate_report(report)


async def run_deterministic() -> dict[str, Any]:
    report = new_report("deterministic")
    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue(maxsize=2)
    accepted = dropped = processed = rejected = 0
    generation = 1

    async def delayed_item(item_generation: int, value: str, delay: float) -> None:
        nonlocal accepted, dropped
        await asyncio.sleep(delay)
        try:
            queue.put_nowait((item_generation, value))
            accepted += 1
        except asyncio.QueueFull:
            dropped += 1

    scope = _evidence_scope("evidence-deterministic")
    try:
        tasks = [
            start_lifecycle_task(scope, delayed_item(1, "stale", 0.001), name="stale"),
            start_lifecycle_task(scope, delayed_item(2, "current-a", 0.002), name="current-a"),
            start_lifecycle_task(scope, delayed_item(2, "current-b", 0.003), name="current-b"),
        ]
        generation = 2
        await asyncio.gather(*tasks)
    finally:
        await scope.close()
    while not queue.empty():
        item_generation, _value = queue.get_nowait()
        if item_generation != generation:
            rejected += 1
        else:
            processed += 1
    remaining_tasks = sum(not task.done() for task in tasks)
    report["stages"] = {
        "prerequisites": _stage("passed", "fixture_available", network=False, credentials=False),
        "cold_enumeration": _stage("passed", "fixture_enumerated", identities=["a", "b"]),
        "warm_enumeration": _stage("passed", "stable_identity", identities=["a", "b"]),
        "activation": _stage("passed", "generation_advanced", generation=generation),
        "queue": _stage(
            "passed", "bounded_queue_exercised", capacity=2, accepted=accepted, processed=processed
        ),
        "thread": _stage(
            "passed", "async_tasks_joined", created=len(tasks), remaining=remaining_tasks
        ),
        "stale_result": _stage("passed", "generation_rejected", rejected=rejected),
        "cleanup": _stage(
            "passed", "resources_released", queue_size=queue.qsize(), tasks=remaining_tasks
        ),
    }
    _finalize(report)
    return report


async def run_local_qwen(model_dir: Path) -> dict[str, Any]:
    report = new_report("local_qwen", model_dir)
    stages = report["stages"]
    try:
        from puripuly_heart.core.local_stt_assets import (
            LocalQwenSherpaLoadError,
            LocalSTTManifestInvalidError,
            LocalSTTModelMissingError,
            validate_local_stt_runtime_ready,
        )

        known_prerequisites = (
            LocalQwenSherpaLoadError,
            LocalSTTManifestInvalidError,
            LocalSTTModelMissingError,
            ImportError,
        )
        await asyncio.to_thread(validate_local_stt_runtime_ready, model_dir)
        import psutil

        from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
    except Exception as exc:
        known = "known_prerequisites" in locals() and isinstance(exc, known_prerequisites)
        status = "blocked" if known else "failed"
        stages["prerequisites"] = _stage(
            status, _safe_classification("prerequisite", exc), known=known
        )
        _finalize(report)
        return report

    stages["prerequisites"] = _stage("passed", "installed_model_ready", manifest=report["model"])
    stages["cold_enumeration"] = _stage("passed", "model_resolved", identity=report["model"])
    stages["warm_enumeration"] = _stage("passed", "model_identity_stable", identity=report["model"])
    creation = {"count": 0}
    decode_gate_started = threading.Event()
    decode_gate_release = threading.Event()
    decode_gate_active = {"value": False}

    class ObservedBackend(LocalQwenSherpaSTTBackend):
        def _create_recognizer(self) -> object:
            recognizer = super()._create_recognizer()
            creation["count"] += 1
            return recognizer

        def _decode_f32_sync(self, recognizer: object, samples_f32: np.ndarray) -> str:
            if decode_gate_active["value"]:
                decode_gate_started.set()
                if not decode_gate_release.wait(timeout=10.0):
                    raise TimeoutError("correlated_decode_gate_timeout")
            return super()._decode_f32_sync(recognizer, samples_f32)

    backend = ObservedBackend(model_dir=model_dir)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    threads_before = process.num_threads()
    cancellation: asyncio.CancelledError | None = None
    try:
        started = time.perf_counter()
        await backend._ensure_recognizer()
        load_ms = (time.perf_counter() - started) * 1000.0
        stages["load"] = _stage("passed", "recognizer_loaded", milliseconds=load_ms)
        stages["activation"] = _stage(
            "passed", "recognizer_active", recognizer_present=backend._recognizer is not None
        )
        fixture = np.zeros(16000, dtype=np.float32)
        correlation_id = "local-qwen-source-generation-1"
        publish, supersede, peer_runtime, peer_facts = _active_peer_publication_gate()

        async def correlated_consumer(samples: np.ndarray) -> None:
            decode_gate_active["value"] = True
            try:
                await backend.decode_f32(samples)
                await publish()
            finally:
                decode_gate_active["value"] = False

        source_probe = await _product_process_source_probe(
            correlated_consumer,
            wait_inflight=lambda: asyncio.to_thread(decode_gate_started.wait, 10.0),
            supersede=supersede,
            release_inflight=decode_gate_release.set,
        )

        decode_generation = 1

        async def timed_decode(generation: int) -> tuple[int, float, float]:
            submitted_at = time.perf_counter()
            async with backend._decode_lock:
                entered_at = time.perf_counter()
                recognizer = await backend._ensure_recognizer()
                await asyncio.to_thread(backend._decode_f32_sync, recognizer, fixture)
            completed_at = time.perf_counter()
            wait_ms = (entered_at - submitted_at) * 1000.0
            decode_ms = (completed_at - entered_at) * 1000.0
            return generation, wait_ms, decode_ms

        scope = _evidence_scope("evidence-qwen-decodes")
        try:
            decode_tasks = [
                start_lifecycle_task(scope, timed_decode(decode_generation), name="decode-first"),
                start_lifecycle_task(scope, timed_decode(decode_generation), name="decode-second"),
            ]
            await asyncio.sleep(0)
            decode_generation += 1
            await asyncio.gather(*decode_tasks)
        finally:
            await scope.close()
        decoded = [task.result() for task in decode_tasks]
        stale_rejected = sum(generation != decode_generation for generation, _, _ in decoded)
        waits = [wait_ms for _, wait_ms, _ in decoded]
        decodes = [decode_ms for _, _, decode_ms in decoded]
        stages["inference"] = _stage(
            "passed",
            "back_to_back_completed",
            decode_milliseconds=decodes,
            max_milliseconds=max(decodes),
        )
        stages["rtf"] = _stage(
            "passed",
            "measured",
            values=[value / 1000.0 for value in decodes],
            max_value=max(decodes) / 1000.0,
        )
        create_count = creation["count"]
        stages["recognizer_count"] = _stage(
            "passed" if create_count == 1 else "failed", "factory_observed", value=create_count
        )
        stages["queue"] = _stage(
            "passed",
            "product_bounded_source_and_decode_lock_observed",
            submitted=source_probe["submitted"],
            completed=len(decoded),
            wait_milliseconds=waits,
            lock_waiters_after=len(backend._decode_lock._waiters or ()),
            correlation_id=correlation_id,
        )
        stages["queue_drop"] = _stage(
            "passed" if source_probe["dropped"] > 0 else "failed",
            "product_source_queue_drop_count",
            submitted=source_probe["submitted"],
            dropped=source_probe["dropped"],
            correlation_id=correlation_id,
        )
        stages["stale_result"] = _stage(
            "passed" if peer_facts["published"] == 0 and peer_facts["attempted"] == 1 else "failed",
            "peer_runtime_generation_sink_observed",
            decode_completions=len(decode_tasks),
            harness_rejected=stale_rejected,
            correlated_source_submissions=source_probe["submitted"],
            overlap_observed=source_probe["overlap_observed"],
            attempted=peer_facts["attempted"],
            published=peer_facts["published"],
            product_rejected=peer_facts["attempted"] - peer_facts["published"],
            generation_before=peer_facts["generation_before"],
            generation_after=peer_facts["generation_after"],
            correlation_id=correlation_id,
        )
        stages["thread"] = _stage(
            "passed",
            "thread_counts_observed",
            before=threads_before,
            after_decode=process.num_threads(),
        )
        rss_after = process.memory_info().rss
        stages["rss"] = _stage(
            "passed",
            "measured",
            before_bytes=rss_before,
            after_bytes=rss_after,
            delta_bytes=max(0, rss_after - rss_before),
        )
    except asyncio.CancelledError as exc:
        cancellation = exc
    except Exception as exc:
        pending = next(
            (name for name in ("load", "inference") if stages[name]["status"] == "blocked"),
            "inference",
        )
        stages[pending] = _stage("failed", _safe_classification("runtime", exc), stage=pending)
        if isinstance(exc, ProductSourceProbeError):
            stages["cleanup"] = _stage(
                "failed",
                f"product_source_{exc.stage}_failed",
                **exc.facts,
            )
    try:
        await backend.close()
        gc.collect()
        await asyncio.sleep(0)
        released = backend._recognizer is None
        harness_reference_released = "recognizer" not in locals()
        threads_after = process.num_threads()
        source_released = "source_probe" in locals() and all(
            source_probe[key]
            for key in ("source_closed", "capture_closed", "watch_closed", "queue_closed")
        )
        peer_released = "peer_runtime" in locals() and peer_runtime.loop_task is None
        resources_released = (
            released and harness_reference_released and source_released and peer_released
        )
        if stages["cleanup"]["status"] != "failed":
            stages["cleanup"] = _stage(
                "passed" if resources_released else "failed",
                "backend_close_observed",
                recognizer_released=released,
                harness_strong_reference=harness_reference_released is False,
                threads_before=threads_before,
                threads_after=threads_after,
                rss_after_bytes=process.memory_info().rss,
                source_released=source_released,
                peer_loop_task_released=peer_released,
            )
        else:
            stages["cleanup"]["facts"].update(
                recognizer_released=released,
                source_released=source_released,
                peer_loop_task_released=peer_released,
            )
    except Exception as exc:
        stages["cleanup"] = _stage(
            "failed", _safe_classification("cleanup", exc), recognizer_released=False
        )
    if cancellation is not None:
        raise cancellation
    _finalize(report)
    return report


def validate_native_process_envelope(native: Any, exit_code: int) -> None:
    base_keys = {
        "schema",
        "status",
        "classification",
        "supported_target",
        "host",
        "credential_free",
        "network_used",
        "thresholds",
        "measurements",
        "lifecycle",
        "capture_construction",
    }
    if not isinstance(native, dict) or native.get("schema") != NATIVE_PROCESS_SCHEMA:
        raise ValueError("native_schema_invalid")
    native_status = native.get("status")
    expected_keys = base_keys | (
        {"fixture"}
        if native_status in {"passed", "failed"} and native.get("measurements") is not None
        else set()
    )
    if set(native) != expected_keys:
        raise ValueError("native_fields_invalid")
    expected_code = 0 if native_status == "passed" else 2 if native_status == "blocked" else 1
    if native_status not in STATUSES or exit_code != expected_code:
        raise ValueError("native_status_invalid")
    if native.get("credential_free") is not True or native.get("network_used") is not False:
        raise ValueError("native_security_invalid")
    supported = native.get("supported_target")
    host = native.get("host")
    if not isinstance(supported, dict) or not isinstance(host, dict):
        raise ValueError("native_provenance_invalid")
    for field in ("system", "implementation", "python", "machine"):
        if host.get(field) != supported.get(field):
            raise ValueError("native_host_incompatible")
    minimum_build = supported.get("minimum_windows_build")
    if not isinstance(minimum_build, int) or not isinstance(host.get("windows_build"), int):
        raise ValueError("native_build_invalid")
    if host["windows_build"] < minimum_build:
        raise ValueError("native_host_incompatible")


def map_process_report(
    native: Any, exit_code: int, probe: dict[str, Any] | None = None
) -> dict[str, Any]:
    report = new_report("process_capture")
    stages = report["stages"]
    try:
        validate_native_process_envelope(native, exit_code)
    except Exception as exc:
        stages["prerequisites"] = _stage(
            "failed", _safe_classification("native_envelope", exc), exit_code=exit_code
        )
        _finalize(report)
        return report
    probe = probe or {}
    required_probe = {
        "cold_enumeration_ms",
        "warm_enumeration_ms",
        "cold_count",
        "warm_count",
        "cold_fingerprint",
        "warm_fingerprint",
        "cold_invocation",
        "warm_invocation",
        "enumeration_fresh",
        "queue_submitted",
        "queue_dropped",
        "stale_submitted",
        "stale_rejected",
        "process_before",
        "process_peak",
        "process_after",
        "threads_before",
        "threads_peak",
        "threads_after",
        "rss_before",
        "rss_peak",
        "rss_after",
        "cleanup_complete",
        "activation_ms",
    }
    if required_probe - set(probe):
        stages["prerequisites"] = _stage(
            "failed", "process_probe_incomplete", missing=sorted(required_probe - set(probe))
        )
        _finalize(report)
        return report
    native_status = native["status"]
    stages["prerequisites"] = _stage(
        "passed",
        "native_v1_validated",
        schema=native["schema"],
        credential_free=True,
        network_used=False,
    )
    stages["cold_enumeration"] = _stage(
        "passed",
        "product_snapshot_port_measured",
        milliseconds=probe["cold_enumeration_ms"],
        count=probe["cold_count"],
        fingerprint=probe["cold_fingerprint"],
        invocation=probe["cold_invocation"],
    )
    stages["warm_enumeration"] = _stage(
        "passed" if probe["enumeration_fresh"] else "failed",
        "product_snapshot_port_measured",
        milliseconds=probe["warm_enumeration_ms"],
        count=probe["warm_count"],
        fingerprint=probe["warm_fingerprint"],
        invocation=probe["warm_invocation"],
        fresh_invocation=probe["enumeration_fresh"],
    )
    stages["queue"] = _stage(
        "passed",
        "bounded_probe_measured",
        submitted=probe["queue_submitted"],
        dropped=probe["queue_dropped"],
    )
    stages["thread"] = _stage(
        "passed",
        "host_counts_sampled",
        process_before=probe["process_before"],
        process_peak=probe["process_peak"],
        process_after=probe["process_after"],
        threads_before=probe["threads_before"],
        threads_peak=probe["threads_peak"],
        threads_after=probe["threads_after"],
        peak_rss_bytes=probe["rss_peak"],
    )
    stages["stale_result"] = _stage(
        "passed" if probe["stale_rejected"] == probe["stale_submitted"] else "failed",
        "generation_probe_measured",
        submitted=probe["stale_submitted"],
        rejected=probe["stale_rejected"],
    )
    stages["cleanup"] = _stage(
        "passed" if probe["cleanup_complete"] else "failed",
        "host_cleanup_sampled",
        process_after=probe["process_after"],
        threads_after=probe["threads_after"],
        rss_after=probe["rss_after"],
        complete=probe["cleanup_complete"],
        source=probe.get("source_cleanup"),
        peer_runtime=probe.get("generation_probe"),
    )
    if native_status != "passed":
        classification = str(native.get("classification") or "native_unclassified")
        harness_cleanup = stages["cleanup"]["facts"]
        stages["cleanup"] = _stage(
            "blocked",
            "native_owned_cleanup_unreported",
            harness_owned_complete=harness_cleanup["complete"],
            source=harness_cleanup["source"],
            peer_runtime=harness_cleanup["peer_runtime"],
            global_threads_after=harness_cleanup["threads_after"],
            global_rss_after=harness_cleanup["rss_after"],
        )
        stages["activation"] = _stage(
            native_status,
            classification,
            milliseconds=probe["activation_ms"],
            native_status=native_status,
        )
        _finalize(report)
        return report
    activation = native["fixture"]["ready_and_activation_order_s"]
    activation_times = activation.get("root_proctap_activations")
    valid_activation = isinstance(activation_times, list) and bool(activation_times)
    stages["activation"] = _stage(
        "passed" if valid_activation else "failed",
        "native_activation_timing_mapped",
        milliseconds=min(activation_times) * 1000.0 if valid_activation else None,
        native_activation_seconds=activation_times,
    )
    _finalize(report)
    return report


async def run_process(evidence_path: Path, thresholds_path: Path) -> dict[str, Any]:
    from puripuly_heart.release_evidence.windows_process_isolation import run

    native_path = evidence_path.with_suffix(".native.json")
    try:
        import psutil

        from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots

        process = psutil.Process()
        snapshots_port = PsutilCurrentUserProcessSnapshots()
        invocation_count = 0

        def enumerate_processes() -> tuple[float, int, str, int]:
            nonlocal invocation_count
            invocation_count += 1
            started = time.perf_counter()
            snapshots = tuple(snapshots_port.snapshots())
            elapsed = (time.perf_counter() - started) * 1000.0
            redacted = "|".join(
                f"{item.pid}:{item.parent_pid}:{int(item.is_current_user)}:{item.instance_id or ''}"
                for item in sorted(snapshots, key=lambda item: item.pid)
            )
            return (
                elapsed,
                len(snapshots),
                hashlib.sha256(redacted.encode()).hexdigest(),
                invocation_count,
            )

        cold_ms, cold_count, cold_fingerprint, cold_invocation = await asyncio.to_thread(
            enumerate_processes
        )
        warm_ms, warm_count, warm_fingerprint, warm_invocation = await asyncio.to_thread(
            enumerate_processes
        )
        source_probe = await _product_process_source_probe()
        generation_probe = await _peer_generation_probe()
        children_before = len(process.children(recursive=True))
        threads_before = process.num_threads()
        rss_before = process.memory_info().rss
        samples: list[tuple[int, int, int]] = []
        monitoring = True

        async def monitor() -> None:
            while monitoring:
                samples.append(
                    (
                        len(process.children(recursive=True)),
                        process.num_threads(),
                        process.memory_info().rss,
                    )
                )
                await asyncio.sleep(0.01)

        scope = _evidence_scope("evidence-process-monitor")
        try:
            monitor_task = start_lifecycle_task(scope, monitor(), name="host-sampler")
            activation_started = time.perf_counter()
            try:
                code = await run(native_path, thresholds_path)
                activation_ms = (time.perf_counter() - activation_started) * 1000.0
            finally:
                monitoring = False
            await monitor_task
        finally:
            await scope.close()
        native = json.loads(native_path.read_text(encoding="utf-8"))
        children_after = len(process.children(recursive=True))
        threads_after = process.num_threads()
        rss_after = process.memory_info().rss
        probe = {
            "cold_enumeration_ms": cold_ms,
            "warm_enumeration_ms": warm_ms,
            "cold_count": cold_count,
            "warm_count": warm_count,
            "cold_fingerprint": cold_fingerprint,
            "warm_fingerprint": warm_fingerprint,
            "cold_invocation": cold_invocation,
            "warm_invocation": warm_invocation,
            "enumeration_fresh": cold_invocation == 1 and warm_invocation == 2,
            "queue_submitted": source_probe["submitted"],
            "queue_dropped": source_probe["dropped"],
            "stale_submitted": generation_probe["attempted"],
            "stale_rejected": generation_probe["rejected"],
            "process_before": children_before,
            "process_peak": max([children_before, *(sample[0] for sample in samples)]),
            "process_after": children_after,
            "threads_before": threads_before,
            "threads_peak": max([threads_before, *(sample[1] for sample in samples)]),
            "threads_after": threads_after,
            "rss_before": rss_before,
            "rss_peak": max([rss_before, *(sample[2] for sample in samples)]),
            "rss_after": rss_after,
            "cleanup_complete": (
                children_after <= children_before
                and source_probe["source_closed"]
                and source_probe["capture_closed"]
                and source_probe["watch_closed"]
                and source_probe["queue_closed"]
                and generation_probe["loop_task_released"]
                and monitor_task.done()
            ),
            "activation_ms": activation_ms,
            "source_cleanup": source_probe,
            "generation_probe": generation_probe,
        }
    except Exception as exc:
        if "monitor_task" in locals() and not monitor_task.done():
            monitoring = False
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
        report = new_report("process_capture")
        report["stages"]["prerequisites"] = _stage(
            "failed",
            _safe_classification("native_report", exc),
            native_report_written=native_path.is_file(),
        )
        _finalize(report)
        return report
    return map_process_report(native, code, probe)


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    report = new_report(args.target.replace("-", "_"), getattr(args, "model_dir", None))
    try:
        same_file = bool(args.baseline and os.path.samefile(args.baseline, args.evidence))
    except FileNotFoundError:
        same_file = bool(args.baseline and args.baseline.resolve() == args.evidence.resolve())
    if same_file:
        raise ValueError("baseline_evidence_same_file")
    try:
        if args.target == "deterministic":
            report = await run_deterministic()
        elif args.target == "local-qwen":
            report = await run_local_qwen(args.model_dir)
        else:
            report = await run_process(args.evidence, args.thresholds)
        baseline = None
        if args.baseline:
            try:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            except Exception as exc:
                report["baseline"] = {
                    "status": "malformed",
                    "comparisons": None,
                    "classification": _safe_classification("baseline", exc),
                }
            else:
                report["baseline"] = compare_baseline(
                    report, baseline, allowance=args.regression_allowance
                )
        elif report["target"] in {"local_qwen", "process_capture"}:
            report["baseline"] = compare_baseline(report, None, allowance=args.regression_allowance)
        _finalize(report)
    except Exception as exc:
        report["stages"]["prerequisites"] = _stage(
            "failed", _safe_classification("runner", exc), report_finalized=True
        )
        _finalize(report)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=("deterministic", "process-capture", "local-qwen"), required=True
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--regression-allowance", type=float, default=0.10)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("scripts/release/windows-process-isolation-thresholds.json"),
    )
    args = parser.parse_args()
    if args.target == "local-qwen" and args.model_dir is None:
        from puripuly_heart.core.local_stt_assets import default_local_stt_model_dir

        args.model_dir = default_local_stt_model_dir()
    report = asyncio.run(execute(args))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
