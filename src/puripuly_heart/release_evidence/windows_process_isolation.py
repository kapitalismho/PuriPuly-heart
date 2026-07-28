from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import ntpath
import os
import platform
import queue
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from puripuly_heart.app.services.peer_process_capture_retry import (
    PeerProcessCaptureRetryOwner,
)
from puripuly_heart.config.process_capture_platform import (
    PROCESS_CAPTURE_MIN_WINDOWS_BUILD,
    get_process_capture_platform_availability,
)
from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget

EVIDENCE_SCHEMA = "puripuly-heart/windows-process-isolation/v1"
SAMPLE_RATE_HZ = 48000
CHANNELS = 2
TARGET_FREQUENCY_HZ = 700.0
CONTROL_FREQUENCY_HZ = 1300.0
EMITTER_AMPLITUDE = 0.18
CAPTURE_SECONDS = 3.0
PROTOCOL_VERSION = 1
WORKER_MODULE = "puripuly_heart.release_evidence.windows_process_isolation"
GUI_PROCESS_RETRY_ACTION = PeerProcessCaptureRetryOwner.retry

EvidenceStatus = Literal["passed", "failed", "blocked"]


@dataclass(frozen=True, slots=True)
class IsolationThresholds:
    target_present_amplitude_min: float
    control_excluded_amplitude_max: float
    control_to_target_ratio_max: float


@dataclass(frozen=True, slots=True)
class IsolationMeasurements:
    target_amplitude: float
    control_amplitude: float
    control_to_target_ratio: float | None
    sample_frames: int


@dataclass(frozen=True, slots=True)
class FixtureMessage:
    event: Literal["ready", "error"]
    role: Literal["target_root", "target_child", "control"]
    pid: int
    child_pid: int | None = None
    child_role: Literal["target_child"] | None = None
    classification: str | None = None
    protocol_version: int = PROTOCOL_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> FixtureMessage:
        raw = json.loads(value)
        expected = {
            "event",
            "role",
            "pid",
            "child_pid",
            "child_role",
            "classification",
            "protocol_version",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("invalid fixture message fields")
        message = cls(**raw)
        if message.protocol_version != PROTOCOL_VERSION or message.pid <= 0:
            raise ValueError("invalid fixture message values")
        return message


def load_thresholds(path: Path) -> IsolationThresholds:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {
        "target_present_amplitude_min",
        "control_excluded_amplitude_max",
        "control_to_target_ratio_max",
    }:
        raise ValueError("invalid isolation threshold fields")
    thresholds = IsolationThresholds(**raw)
    if (
        thresholds.target_present_amplitude_min <= 0
        or thresholds.control_excluded_amplitude_max < 0
        or not 0 <= thresholds.control_to_target_ratio_max < 1
    ):
        raise ValueError("invalid isolation threshold values")
    return thresholds


def measure_isolation(
    samples: np.ndarray, *, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> IsolationMeasurements:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CHANNELS or values.shape[0] == 0:
        raise ValueError("capture samples must be non-empty stereo frames")
    mono = values.mean(axis=1)
    target = _windowed_tone_amplitude(mono, TARGET_FREQUENCY_HZ, sample_rate_hz)
    control = _windowed_tone_amplitude(mono, CONTROL_FREQUENCY_HZ, sample_rate_hz)
    ratio = control / target if target > np.finfo(np.float64).eps else None
    return IsolationMeasurements(
        target_amplitude=target,
        control_amplitude=control,
        control_to_target_ratio=ratio,
        sample_frames=int(values.shape[0]),
    )


def isolation_passes(measurements: IsolationMeasurements, thresholds: IsolationThresholds) -> bool:
    return (
        measurements.target_amplitude >= thresholds.target_present_amplitude_min
        and measurements.control_amplitude <= thresholds.control_excluded_amplitude_max
        and measurements.control_to_target_ratio is not None
        and measurements.control_to_target_ratio <= thresholds.control_to_target_ratio_max
    )


def lifecycle_passes(
    *,
    events: Sequence[str],
    warning_reason: str | None,
    loop_task_done_at_warning: bool,
    process_source_pids: Sequence[int],
    closed_source_pids: set[int],
    first_pid: int,
    retry_pid: int,
    no_automatic_reconnect: bool,
    gui_retry_succeeded: bool,
    gui_warning_cleared: bool,
) -> bool:
    required_events = ("provider_closed", "source_closed", "typed_warning")
    try:
        event_indexes = tuple(events.index(event) for event in required_events)
    except ValueError:
        return False
    return (
        warning_reason == "process_target_exited"
        and event_indexes == tuple(sorted(event_indexes))
        and loop_task_done_at_warning
        and list(process_source_pids) == [first_pid, retry_pid]
        and first_pid in closed_source_pids
        and no_automatic_reconnect
        and gui_retry_succeeded
        and gui_warning_cleared
        and first_pid != retry_pid
    )


async def invoke_gui_process_retry(action: object) -> bool:
    return await GUI_PROCESS_RETRY_ACTION(action)


def build_fixture_capture_target(executable: str) -> ResolvedDesktopAudioCaptureTarget:
    normalized = ntpath.normcase(ntpath.abspath(executable))
    return ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="generic_executable",
        executable_identity=normalized,
    )


def validate_direct_child_topology(
    *,
    root_pid: int,
    child_pid: int,
    child_ppid: int,
    descendant_pids: set[int],
    control_pid: int,
) -> bool:
    return (
        root_pid > 0
        and child_pid > 0
        and child_ppid == root_pid
        and child_pid in descendant_pids
        and control_pid not in descendant_pids
    )


def _tone_amplitude(samples: np.ndarray, frequency_hz: float, sample_rate_hz: int) -> float:
    indexes = np.arange(samples.size, dtype=np.float64)
    basis = np.exp(-2j * np.pi * frequency_hz * indexes / sample_rate_hz)
    return float(2.0 * abs(np.dot(samples, basis)) / samples.size)


def _windowed_tone_amplitude(
    samples: np.ndarray, frequency_hz: float, sample_rate_hz: int
) -> float:
    window_frames = sample_rate_hz // 10
    complete_frames = samples.size - (samples.size % window_frames)
    if complete_frames == 0:
        return _tone_amplitude(samples, frequency_hz, sample_rate_hz)
    amplitudes = [
        _tone_amplitude(window, frequency_hz, sample_rate_hz)
        for window in samples[:complete_frames].reshape((-1, window_frames))
    ]
    return float(np.percentile(amplitudes, 90))


def classify_native_capability(
    *,
    module_available: Callable[[str], bool] | None = None,
    audio_output_probe: Callable[[], None] | None = None,
) -> str | None:
    availability = get_process_capture_platform_availability()
    if not availability.available:
        return availability.reason or "unsupported_platform"
    available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    for module_name in ("proctap", "psutil", "sounddevice"):
        if not available(module_name):
            return f"native_dependency_unavailable:{module_name}"
    probe = audio_output_probe or _probe_audio_output
    try:
        probe()
    except Exception:
        return "native_audio_output_unavailable"
    return None


def build_blocked_evidence(
    classification: str, thresholds: IsolationThresholds
) -> dict[str, object]:
    return _build_empty_evidence("blocked", classification, thresholds)


def build_failed_evidence(
    classification: str, thresholds: IsolationThresholds
) -> dict[str, object]:
    return _build_empty_evidence("failed", classification, thresholds)


def _build_empty_evidence(
    status: EvidenceStatus,
    classification: str,
    thresholds: IsolationThresholds,
) -> dict[str, object]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": status,
        "classification": classification,
        "supported_target": {
            "system": "Windows",
            "implementation": "CPython",
            "python": "3.12",
            "machine": "AMD64",
            "minimum_windows_build": PROCESS_CAPTURE_MIN_WINDOWS_BUILD,
        },
        "host": _safe_host_facts(),
        "credential_free": True,
        "network_used": False,
        "thresholds": asdict(thresholds),
        "measurements": None,
        "lifecycle": None,
        "capture_construction": {"process_sources": 0, "device_loopback_sources": 0},
    }


def classify_fixture_failure(exc: Exception) -> str:
    from puripuly_heart.core.audio.process_source import (
        ProcessAudioCaptureSetupError,
        ProcessAudioCaptureUnavailableError,
    )

    if isinstance(exc, ProcessAudioCaptureUnavailableError):
        return "process_capture_unavailable"
    if isinstance(exc, ProcessAudioCaptureSetupError):
        return "process_capture_setup_failed"
    if isinstance(exc, ImportError):
        return "fixture_import_failed"
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "fixture_timeout"
    if isinstance(exc, RuntimeError):
        code = str(exc)
        if code == "direct_child_capture_timeout":
            return "direct_child_capture_timeout"
        if code == "native_capture_timeout":
            return "peer_runtime_capture_timeout"
        if code == "peer_runtime_faulted_before_frames":
            return "peer_runtime_faulted_before_frames"
        if code.startswith("peer_runtime_faulted_before_frames:"):
            reason = code.partition(":")[2]
            if reason in {
                "process_target_unavailable",
                "process_setup_failed",
                "process_target_exited",
                "process_source_failed",
                "process_provider_failed",
                "peer_runtime_failed",
            }:
                return f"peer_runtime_faulted_before_frames_{reason}"
        if code.startswith("peer_runtime_loop_exception:"):
            exception_name = code.partition(":")[2]
            if exception_name.replace("_", "").isalnum():
                return f"peer_runtime_loop_exception_{exception_name[:80]}"
        if code == "peer_runtime_loop_completed_before_frames":
            return "peer_runtime_loop_completed_before_frames"
        if code == "target_exit_timeout":
            return "target_exit_timeout"
        if code == "direct_child_topology_invalid":
            return "fixture_topology_invalid"
        if code.endswith("_startup_timeout") or code.endswith("_startup_failed"):
            return "emitter_startup_failed"
    return "fixture_execution_failed"


def write_evidence(path: Path, evidence: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_host_facts() -> dict[str, object]:
    build = getattr(getattr(sys, "getwindowsversion", lambda: None)(), "build", None)
    return {
        "system": platform.system(),
        "implementation": platform.python_implementation(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "machine": platform.machine(),
        "windows_build": build,
    }


def _probe_audio_output() -> None:
    import sounddevice

    device = sounddevice.query_devices(kind="output")
    if int(device["max_output_channels"]) < CHANNELS:
        raise RuntimeError("stereo output unavailable")


class _EmitterProcess:
    def __init__(
        self,
        process: subprocess.Popen[str],
        ready: FixtureMessage,
        ready_monotonic_s: float,
    ) -> None:
        self.process = process
        self.ready = ready
        self.ready_monotonic_s = ready_monotonic_s

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        descendants = []
        try:
            import psutil

            descendants = psutil.Process(self.process.pid).children(recursive=True)
        except Exception:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        for descendant in descendants:
            try:
                descendant.terminate()
            except Exception:
                pass
        if descendants:
            try:
                import psutil

                _, alive = psutil.wait_procs(descendants, timeout=5)
                for descendant in alive:
                    descendant.kill()
                psutil.wait_procs(alive, timeout=5)
            except Exception:
                pass


def _worker_command(*args: str) -> list[str]:
    executable = getattr(sys, "_base_executable", None) or sys.executable
    return [str(Path(executable).resolve()), "-m", WORKER_MODULE, *args]


def _worker_environment(runtime_dir: Path) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "TEMP": str(runtime_dir),
        "TMP": str(runtime_dir),
        "PYTHONIOENCODING": "utf-8",
    }
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join((source_root, sysconfig.get_paths()["purelib"]))
    return environment


def _start_emitter(role: Literal["target_root", "control"], runtime_dir: Path) -> _EmitterProcess:
    process = subprocess.Popen(
        _worker_command("--worker", role),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=_worker_environment(runtime_dir),
        cwd=runtime_dir,
    )
    assert process.stdout is not None
    lines: queue.Queue[str] = queue.Queue(maxsize=1)
    reader = threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True)
    reader.start()
    try:
        line = lines.get(timeout=10)
    except queue.Empty:
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError(f"{role}_startup_timeout") from None
    if not line:
        process.wait(timeout=5)
        raise RuntimeError(f"{role}_startup_failed")
    message = FixtureMessage.from_json(line)
    if message.event != "ready" or message.role != role:
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError(message.classification or f"{role}_startup_failed")
    return _EmitterProcess(process, message, time.monotonic())


def _emit_tone(role: Literal["target_child", "control"], frequency_hz: float) -> None:
    import sounddevice

    phase = 0

    def callback(outdata, frames, _time_info, status) -> None:  # noqa: ANN001
        nonlocal phase
        if status:
            raise RuntimeError("audio_output_status")
        positions = np.arange(phase, phase + frames, dtype=np.float64)
        wave = (
            EMITTER_AMPLITUDE * np.sin(2 * np.pi * frequency_hz * positions / SAMPLE_RATE_HZ)
        ).astype(np.float32)
        outdata[:] = np.repeat(wave[:, None], CHANNELS, axis=1)
        phase += frames

    try:
        with sounddevice.OutputStream(
            samplerate=SAMPLE_RATE_HZ,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
        ):
            print(FixtureMessage(event="ready", role=role, pid=os.getpid()).to_json(), flush=True)
            while True:
                time.sleep(1)
    except Exception:
        print(
            FixtureMessage(
                event="error",
                role=role,
                pid=os.getpid(),
                classification="native_audio_output_unavailable",
            ).to_json(),
            flush=True,
        )
        raise SystemExit(3)


def _run_target_root() -> None:
    child = subprocess.Popen(
        _worker_command("--worker", "target_child"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=os.environ.copy(),
    )
    try:
        assert child.stdout is not None
        child_message = FixtureMessage.from_json(child.stdout.readline())
        if child_message.event != "ready" or child_message.role != "target_child":
            print(
                FixtureMessage(
                    event="error",
                    role="target_root",
                    pid=os.getpid(),
                    classification=child_message.classification,
                ).to_json(),
                flush=True,
            )
            raise SystemExit(3)
        print(
            FixtureMessage(
                event="ready",
                role="target_root",
                pid=os.getpid(),
                child_pid=child_message.pid,
                child_role=child_message.role,
            ).to_json(),
            flush=True,
        )
        child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


async def _run_native(thresholds: IsolationThresholds, runtime_dir: Path) -> dict[str, object]:
    import psutil

    from puripuly_heart.config.process_capture_resolution import ResolvedProcessCaptureIdentity
    from puripuly_heart.config.resolved import (
        ResolvedCredentialRequirement,
        ResolvedSTTConfig,
    )
    from puripuly_heart.config.settings import STTProviderName
    from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent
    from puripuly_heart.core.audio.process_identity import PsutilProcessIdentityWatcher
    from puripuly_heart.core.audio.process_source import (
        ProcessAudioCaptureSource,
        ProcTapProcessAudioCaptureFactory,
        verify_proctap_1_0_3_process_specific,
    )
    from puripuly_heart.core.runtime.peer_channel import (
        PeerChannelRuntime,
        PeerChannelRuntimeState,
        PeerRuntimeConfig,
        PeerRuntimeFailureReason,
    )

    class Clock:
        def now(self) -> float:
            return time.monotonic()

    class Provider:
        def __init__(self, events: list[str]) -> None:
            self.events = events
            self.closed = False

        async def close_backend(self) -> None:
            self.closed = True
            self.events.append("provider_closed")

    class Hub:
        def __init__(self) -> None:
            self.peer_stt = None

        async def replace_peer_stt_provider(self, provider, *, start=True) -> None:  # noqa: ANN001
            previous, self.peer_stt = self.peer_stt, provider
            if previous is not None and previous is not provider:
                await previous.close_backend()

    events: list[str] = []
    samples: list[np.ndarray] = []
    loop_failure_types: list[str] = []
    process_source_pids: list[int] = []
    source_closed: set[int] = set()
    native_mode_observations: list[bool] = []
    activation_times: list[float] = []
    current_root: _EmitterProcess | None = None
    fixture_started = time.monotonic()

    class ObservedCapture:
        def __init__(self, capture) -> None:  # noqa: ANN001
            self.capture = capture

        def start(self) -> None:
            activation_times.append(time.monotonic() - fixture_started)
            self.capture.start()

        def close(self) -> None:
            self.capture.close()

    class ObservedCaptureFactory:
        def __init__(self) -> None:
            self.delegate = ProcTapProcessAudioCaptureFactory()

        def create(self, *, pid: int, on_data) -> ObservedCapture:  # noqa: ANN001
            capture = self.delegate.create(pid=pid, on_data=on_data)
            native_mode_observations.append(verify_proctap_1_0_3_process_specific(capture))
            return ObservedCapture(capture)

    observed_capture_factory = ObservedCaptureFactory()

    class ObservedSource:
        def __init__(self, source: ProcessAudioCaptureSource, pid: int) -> None:
            self.source = source
            self.pid = pid

        @property
        def terminal_reason(self) -> str | None:
            return self.source.terminal_reason

        def frames(self):  # noqa: ANN201
            return self.source.frames()

        async def close(self) -> None:
            await self.source.close()
            source_closed.add(self.pid)
            events.append("source_closed")

    def source_factory(_config: PeerRuntimeConfig) -> ObservedSource:
        if current_root is None or current_root.process.poll() is not None:
            raise RuntimeError("target root unavailable")
        process = psutil.Process(current_root.ready.pid)
        identity = ResolvedProcessCaptureIdentity(
            pid=process.pid,
            target=ProcessCaptureTargetIntent.generic_executable(sys.executable),
            instance_id=f"{process.pid}:{process.create_time()}",
        )
        process_source_pids.append(process.pid)
        return ObservedSource(
            ProcessAudioCaptureSource(
                identity=identity,
                watcher=PsutilProcessIdentityWatcher(),
                capture_factory=observed_capture_factory,
            ),
            process.pid,
        )

    async def collect_loop(*, source, **_kwargs) -> None:  # noqa: ANN001
        try:
            async for frame in source.frames():
                samples.append(frame.samples.copy())
        except Exception as exc:
            loop_failure_types.append(type(exc).__name__)
            raise

    credential = ResolvedCredentialRequirement(source="none", required=False, reference=None)
    backend = ResolvedSTTConfig(
        channel="peer",
        source_language="en",
        provider=STTProviderName.DEEPGRAM,
        model=None,
        endpoint=None,
        region=None,
        credential=credential,
        input_host_api=None,
        input_device=None,
        output_device=None,
        sample_rate_hz=SAMPLE_RATE_HZ,
        channels=CHANNELS,
        ring_buffer_ms=1000,
        drain_timeout_s=1.0,
        vad_speech_threshold=0.5,
        vad_hangover_ms=0,
        vad_pre_roll_ms=0,
        low_latency_enabled=False,
        low_latency_merge_gap_ms=0,
        low_latency_spec_retry_max=0,
        custom_vocabulary_enabled=False,
        custom_terms=MappingProxyType({}),
        provider_options=MappingProxyType({}),
    )
    target = build_fixture_capture_target(sys.executable)
    config = PeerRuntimeConfig(
        backend=backend,
        output_device="",
        vad_threshold=0.5,
        vad_hangover_ms=0,
        vad_pre_roll_ms=0,
        provider_signature=("fixture",),
        runtime_signature=("fixture",),
        capture_target=target,
    )
    diagnostics = []
    loop_task_at_warning: list[bool] = []
    initial_loop_task = None

    def diagnostic_sink(diagnostic) -> None:  # noqa: ANN001
        events.append("typed_warning")
        diagnostics.append(diagnostic)
        loop_task_at_warning.append(initial_loop_task is not None and initial_loop_task.done())

    runtime = PeerChannelRuntime(
        hub=Hub(),
        clock=Clock(),
        stt_factory=lambda _config, _failure: Provider(events),
        source_factory=source_factory,
        vad_factory=lambda _config, _path: object(),
        vad_model_resolver=lambda: runtime_dir / "unused-vad.onnx",
        run_audio_loop=collect_loop,
        diagnostic_sink=diagnostic_sink,
    )
    control: _EmitterProcess | None = None
    first_pid = 0
    first_child_pid = 0
    first_child_role: str | None = None
    second_pid = 0
    direct_child_measurement: IsolationMeasurements | None = None
    direct_child_activation_s: float | None = None
    target_child_ready_s: float | None = None
    control_ready_s: float | None = None
    retry_root_ready_s: float | None = None
    child_relation_verified = False
    child_os_ppid = 0
    control_non_descendant_verified = False
    try:
        current_root = _start_emitter("target_root", runtime_dir)
        control = _start_emitter("control", runtime_dir)
        first_pid = current_root.ready.pid
        first_child_pid = current_root.ready.child_pid or 0
        first_child_role = current_root.ready.child_role
        target_child_ready_s = current_root.ready_monotonic_s - fixture_started
        control_ready_s = control.ready_monotonic_s - fixture_started
        root_process = psutil.Process(first_pid)
        child_process = psutil.Process(first_child_pid)
        child_os_ppid = child_process.ppid()
        root_descendants = {process.pid for process in root_process.children(recursive=True)}
        child_relation_verified = validate_direct_child_topology(
            root_pid=first_pid,
            child_pid=first_child_pid,
            child_ppid=child_os_ppid,
            descendant_pids=root_descendants,
            control_pid=control.ready.pid,
        )
        control_non_descendant_verified = control.ready.pid not in root_descendants
        if not child_relation_verified or not control_non_descendant_verified:
            raise RuntimeError("direct_child_topology_invalid")

        direct_samples: list[np.ndarray] = []

        def on_direct_child_data(data: bytes, frames: int) -> None:
            derived_frames = len(data) // (CHANNELS * 4)
            if frames == -1:
                frames = derived_frames
            if frames <= 0 or derived_frames != frames:
                return
            direct_samples.append(
                np.frombuffer(data, dtype="<f4").reshape((frames, CHANNELS)).copy()
            )

        direct_capture = ProcTapProcessAudioCaptureFactory().create(
            pid=first_child_pid,
            on_data=on_direct_child_data,
        )
        direct_native_mode = verify_proctap_1_0_3_process_specific(direct_capture)
        direct_child_activation_s = time.monotonic() - fixture_started
        try:
            direct_capture.start()
            await asyncio.sleep(1.2)
        finally:
            direct_capture.close()
        if not direct_samples:
            raise RuntimeError("direct_child_capture_timeout")
        direct_child_measurement = measure_isolation(np.concatenate(direct_samples, axis=0))

        await runtime.apply_policy(config=config, desired_active=True)
        initial_loop_task = runtime.loop_task
        deadline = time.monotonic() + CAPTURE_SECONDS + 5
        required_frames = int(CAPTURE_SECONDS * SAMPLE_RATE_HZ)
        while sum(frame.shape[0] for frame in samples) < required_frames:
            if runtime.state == PeerChannelRuntimeState.FAULTED:
                failure = runtime.last_failure
                reason = failure.reason.value if failure is not None else "unknown"
                if loop_failure_types:
                    raise RuntimeError(f"peer_runtime_loop_exception:{loop_failure_types[-1]}")
                raise RuntimeError(f"peer_runtime_faulted_before_frames:{reason}")
            if initial_loop_task is not None and initial_loop_task.done():
                raise RuntimeError("peer_runtime_loop_completed_before_frames")
            if time.monotonic() >= deadline:
                raise RuntimeError("native_capture_timeout")
            await asyncio.sleep(0.05)
        measured_samples = np.concatenate(samples, axis=0)[:required_frames]
        measurements = measure_isolation(measured_samples)
        current_root.stop()
        deadline = time.monotonic() + 8
        while runtime.state != PeerChannelRuntimeState.FAULTED:
            if time.monotonic() >= deadline:
                raise RuntimeError("target_exit_timeout")
            await asyncio.sleep(0.05)
        sources_after_exit = len(process_source_pids)
        await runtime.apply_policy(config=config, desired_active=True)
        no_automatic_reconnect = len(process_source_pids) == sources_after_exit
        current_root = _start_emitter("target_root", runtime_dir)
        second_pid = current_root.ready.pid
        retry_root_ready_s = current_root.ready_monotonic_s - fixture_started

        retry_settings = object()
        retry_warning = ["process_target_exited"]
        gui_action = PeerProcessCaptureRetryOwner(
            settings_provider=lambda: retry_settings,
            runtime_provider=lambda: runtime,
            should_be_active=lambda _settings: True,
            ensure_ready=lambda: asyncio.sleep(0, result=True),
            build_config=lambda _settings: config,
            on_retry_succeeded=lambda: retry_warning.__setitem__(0, None),
            sync_effective_flags=lambda _settings: None,
            refresh_consumers=lambda: None,
        )
        retried = await invoke_gui_process_retry(gui_action)
        await asyncio.sleep(0.1)
        lifecycle_passed_result = lifecycle_passes(
            events=events,
            warning_reason=(
                diagnostics[-1].reason.value
                if diagnostics
                and diagnostics[-1].reason is PeerRuntimeFailureReason.PROCESS_TARGET_EXITED
                else None
            ),
            loop_task_done_at_warning=bool(loop_task_at_warning and loop_task_at_warning[-1]),
            process_source_pids=process_source_pids,
            closed_source_pids=source_closed,
            first_pid=first_pid,
            retry_pid=second_pid,
            no_automatic_reconnect=no_automatic_reconnect,
            gui_retry_succeeded=retried,
            gui_warning_cleared=retry_warning[0] is None,
        )
        passed = isolation_passes(measurements, thresholds) and lifecycle_passed_result
        activation_order_verified = (
            len(activation_times) == 2
            and target_child_ready_s is not None
            and control_ready_s is not None
            and direct_child_activation_s is not None
            and retry_root_ready_s is not None
            and target_child_ready_s
            < control_ready_s
            < direct_child_activation_s
            < activation_times[0]
            and retry_root_ready_s <= activation_times[1]
        )
        fixture_contract_passed = (
            first_child_pid > 0
            and child_relation_verified
            and control_non_descendant_verified
            and activation_order_verified
            and direct_native_mode
            and all(native_mode_observations)
            and direct_child_measurement.target_amplitude >= thresholds.target_present_amplitude_min
        )
        passed = passed and fixture_contract_passed
        provider_close_before_warning = (
            "provider_closed" in events
            and "typed_warning" in events
            and events.index("provider_closed") < events.index("typed_warning")
        )
        source_close_before_warning = (
            "source_closed" in events
            and "typed_warning" in events
            and events.index("source_closed") < events.index("typed_warning")
        )
        return {
            "schema": EVIDENCE_SCHEMA,
            "status": "passed" if passed else "failed",
            "classification": None if passed else "measured_contract_failed",
            "supported_target": {
                "system": "Windows",
                "implementation": "CPython",
                "python": "3.12",
                "machine": "AMD64",
                "minimum_windows_build": PROCESS_CAPTURE_MIN_WINDOWS_BUILD,
            },
            "host": _safe_host_facts(),
            "credential_free": True,
            "network_used": False,
            "thresholds": asdict(thresholds),
            "measurements": asdict(measurements),
            "fixture": {
                "target_root_pid": first_pid,
                "target_child_pid": first_child_pid,
                "control_pid": control.ready.pid,
                "retry_root_pid": second_pid,
                "child_ready_role": first_child_role,
                "child_ready_pid_matches": first_child_pid == child_process.pid,
                "child_os_ppid": child_os_ppid,
                "child_descendant_relation_verified": child_relation_verified,
                "control_non_descendant_verified": control_non_descendant_verified,
                "activation_order_verified": activation_order_verified,
                "ready_and_activation_order_s": {
                    "target_child_ready": target_child_ready_s,
                    "control_ready": control_ready_s,
                    "direct_child_proctap_activation": direct_child_activation_s,
                    "retry_root_ready": retry_root_ready_s,
                    "root_proctap_activations": activation_times,
                },
                "signals_hz": {"target": TARGET_FREQUENCY_HZ, "control": CONTROL_FREQUENCY_HZ},
                "selected_tree_child_included": (
                    measurements.target_amplitude >= thresholds.target_present_amplitude_min
                ),
                "unrelated_control_excluded": (
                    measurements.control_amplitude <= thresholds.control_excluded_amplitude_max
                    and measurements.control_to_target_ratio is not None
                    and measurements.control_to_target_ratio
                    <= thresholds.control_to_target_ratio_max
                ),
                "direct_child_diagnostic": {
                    "target_amplitude": direct_child_measurement.target_amplitude,
                    "sample_frames": direct_child_measurement.sample_frames,
                    "native_process_specific_mode": direct_native_mode,
                },
            },
            "lifecycle": {
                "typed_warning": diagnostics[-1].reason.value if diagnostics else None,
                "warning_after_provider_source_task_teardown": lifecycle_passed_result,
                "provider_close_before_warning": provider_close_before_warning,
                "source_close_before_warning": source_close_before_warning,
                "task_completion_before_warning": bool(
                    loop_task_at_warning and loop_task_at_warning[-1]
                ),
                "automatic_reconnect": not no_automatic_reconnect,
                "retry_action": "PeerProcessCaptureRetryOwner.retry",
                "retry_succeeded": retried,
                "fresh_pid": first_pid != second_pid,
            },
            "capture_construction": {
                "process_sources": len(process_source_pids) + 1,
                "device_loopback_sources": 0,
                "source_pids": process_source_pids,
                "direct_diagnostic_pid": first_child_pid,
                "native_process_specific_observations": [
                    direct_native_mode,
                    *native_mode_observations,
                ],
                "frame_source_kind": (
                    "process"
                    if direct_native_mode and all(native_mode_observations)
                    else "unverified"
                ),
            },
        }
    finally:
        await runtime.close()
        if current_root is not None:
            current_root.stop()
        if control is not None:
            control.stop()


def _threshold_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "release"
        / "windows-process-isolation-thresholds.json"
    )


async def run(evidence_path: Path, thresholds_path: Path) -> int:
    thresholds = load_thresholds(thresholds_path)
    classification = classify_native_capability()
    if classification is not None:
        evidence = build_blocked_evidence(classification, thresholds)
        write_evidence(evidence_path, evidence)
        return 2
    with tempfile.TemporaryDirectory(prefix="puripuly-isolation-") as directory:
        try:
            evidence = await _run_native(thresholds, Path(directory))
        except Exception as exc:
            evidence = build_failed_evidence(classify_fixture_failure(exc), thresholds)
        write_evidence(evidence_path, evidence)
    if evidence["status"] == "passed":
        return 0
    return 2 if evidence["status"] == "blocked" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="windows-process-isolation-evidence")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--thresholds", type=Path, default=_threshold_path())
    parser.add_argument(
        "--worker", choices=("target_root", "target_child", "control"), help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker == "target_root":
        _run_target_root()
        return 0
    if args.worker == "target_child":
        _emit_tone("target_child", TARGET_FREQUENCY_HZ)
        return 0
    if args.worker == "control":
        _emit_tone("control", CONTROL_FREQUENCY_HZ)
        return 0
    if args.evidence is None:
        raise SystemExit("--evidence is required")
    return asyncio.run(run(args.evidence, args.thresholds))


if __name__ == "__main__":
    raise SystemExit(main())
