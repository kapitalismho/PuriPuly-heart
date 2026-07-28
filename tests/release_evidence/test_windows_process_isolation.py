from __future__ import annotations

import asyncio
import json
import ntpath
import sys
from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.app.services.peer_process_capture_retry import (
    PeerProcessCaptureRetryOwner,
)
from puripuly_heart.release_evidence.windows_process_isolation import (
    CHANNELS,
    CONTROL_FREQUENCY_HZ,
    EVIDENCE_SCHEMA,
    GUI_PROCESS_RETRY_ACTION,
    SAMPLE_RATE_HZ,
    TARGET_FREQUENCY_HZ,
    WORKER_MODULE,
    FixtureMessage,
    IsolationThresholds,
    _worker_command,
    build_blocked_evidence,
    build_fixture_capture_target,
    classify_native_capability,
    invoke_gui_process_retry,
    isolation_passes,
    lifecycle_passes,
    load_thresholds,
    measure_isolation,
    run,
    validate_direct_child_topology,
)

THRESHOLDS = IsolationThresholds(
    target_present_amplitude_min=0.05,
    control_excluded_amplitude_max=0.005,
    control_to_target_ratio_max=0.1,
)


def _stereo_tones(*, target_amplitude: float, control_amplitude: float) -> np.ndarray:
    indexes = np.arange(SAMPLE_RATE_HZ, dtype=np.float64)
    mono = target_amplitude * np.sin(
        2 * np.pi * TARGET_FREQUENCY_HZ * indexes / SAMPLE_RATE_HZ
    ) + control_amplitude * np.sin(2 * np.pi * CONTROL_FREQUENCY_HZ * indexes / SAMPLE_RATE_HZ)
    return np.repeat(mono[:, None], CHANNELS, axis=1).astype(np.float32)


def test_fixture_protocol_round_trips_deterministically_and_rejects_unknown_fields() -> None:
    message = FixtureMessage(
        event="ready",
        role="target_root",
        pid=41,
        child_pid=42,
        child_role="target_child",
        classification=None,
    )

    encoded = message.to_json()

    assert encoded == (
        '{"child_pid":42,"child_role":"target_child","classification":null,'
        '"event":"ready","pid":41,'
        '"protocol_version":1,"role":"target_root"}'
    )
    assert FixtureMessage.from_json(encoded) == message
    with pytest.raises(ValueError, match="fields"):
        FixtureMessage.from_json(encoded[:-1] + ',"token":"secret"}')


def test_worker_command_uses_stable_module_name_in_nested_root_process() -> None:
    command = _worker_command("--worker", "target_child")

    assert command[1:] == ["-m", WORKER_MODULE, "--worker", "target_child"]
    assert Path(command[0]) == Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    assert "__main__" not in command


def test_topology_contract_rejects_intermediate_launcher() -> None:
    assert validate_direct_child_topology(
        root_pid=10,
        child_pid=12,
        child_ppid=10,
        descendant_pids={12},
        control_pid=20,
    )
    assert not validate_direct_child_topology(
        root_pid=10,
        child_pid=12,
        child_ppid=11,
        descendant_pids={11, 12},
        control_pid=20,
    )


def test_threshold_math_requires_measured_target_and_control_exclusion() -> None:
    passing = measure_isolation(_stereo_tones(target_amplitude=0.18, control_amplitude=0.001))
    callback_only = measure_isolation(_stereo_tones(target_amplitude=0.0, control_amplitude=0.0))
    leaking_control = measure_isolation(
        _stereo_tones(target_amplitude=0.18, control_amplitude=0.04)
    )

    assert passing.target_amplitude == pytest.approx(0.18, abs=1e-6)
    assert passing.control_amplitude == pytest.approx(0.001, abs=1e-6)
    assert isolation_passes(passing, THRESHOLDS) is True
    assert isolation_passes(callback_only, THRESHOLDS) is False
    assert isolation_passes(leaking_control, THRESHOLDS) is False


def test_lifecycle_contract_requires_ordered_teardown_fresh_pid_and_gui_retry() -> None:
    facts = dict(
        events=["provider_closed", "source_closed", "typed_warning"],
        warning_reason="process_target_exited",
        loop_task_done_at_warning=True,
        process_source_pids=[101, 202],
        closed_source_pids={101},
        first_pid=101,
        retry_pid=202,
        no_automatic_reconnect=True,
        gui_retry_succeeded=True,
        gui_warning_cleared=True,
    )

    assert lifecycle_passes(**facts) is True
    assert lifecycle_passes(**{**facts, "events": ["typed_warning", "source_closed"]}) is False
    assert lifecycle_passes(**{**facts, "retry_pid": 101}) is False
    assert lifecycle_passes(**{**facts, "no_automatic_reconnect": False}) is False


@pytest.mark.asyncio
async def test_fixture_invokes_the_committed_gui_retry_action_contract() -> None:
    class Runtime:
        async def retry_process_capture(self, *, config) -> bool:  # noqa: ANN001
            assert config == "fresh-resolved-config"
            return True

    settings = object()
    warning = ["process_target_exited"]
    action = PeerProcessCaptureRetryOwner(
        settings_provider=lambda: settings,
        runtime_provider=Runtime,
        should_be_active=lambda current: current is settings,
        ensure_ready=lambda: asyncio.sleep(0, result=True),
        build_config=lambda current: (
            "fresh-resolved-config" if current is settings else "stale-config"
        ),
        on_retry_succeeded=lambda: warning.__setitem__(0, None),
        sync_effective_flags=lambda _settings: None,
        refresh_consumers=lambda: None,
    )

    assert GUI_PROCESS_RETRY_ACTION is PeerProcessCaptureRetryOwner.retry
    assert await invoke_gui_process_retry(action) is True
    assert warning[0] is None
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "puripuly_heart"
        / "release_evidence"
        / "windows_process_isolation.py"
    ).read_text(encoding="utf-8")
    assert "puripuly_heart.ui.controller" not in source
    assert '"retry_action": "PeerProcessCaptureRetryOwner.retry"' in source


def test_fixture_builds_the_committed_resolved_process_target_contract() -> None:
    executable = r"C:\Program Files\Python312\python.exe"

    target = build_fixture_capture_target(executable)

    assert target.kind == "process"
    assert target.process_kind == "generic_executable"
    assert target.executable_identity == ntpath.normcase(executable)
    assert target.executable_basename is None


def test_checked_in_thresholds_and_blocked_artifact_are_strict_and_secret_safe(
    tmp_path: Path,
) -> None:
    threshold_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "release"
        / "windows-process-isolation-thresholds.json"
    )
    artifact_path = tmp_path / "artifact.json"

    loaded = load_thresholds(threshold_path)
    artifact = build_blocked_evidence("native_dependency_unavailable:proctap", loaded)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    serialized = json.dumps(artifact, sort_keys=True)

    assert loaded == THRESHOLDS
    assert artifact["schema"] == EVIDENCE_SCHEMA
    assert artifact["status"] == "blocked"
    assert artifact["measurements"] is None
    assert artifact["capture_construction"] == {
        "process_sources": 0,
        "device_loopback_sources": 0,
    }
    assert all(
        word not in serialized.casefold() for word in ("api_key", "password", "credential_value")
    )


def test_capability_classification_is_explicit_for_platform_dependency_and_audio(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "puripuly_heart.release_evidence.windows_process_isolation.get_process_capture_platform_availability",
        lambda: type("Availability", (), {"available": True, "reason": None})(),
    )

    assert (
        classify_native_capability(
            module_available=lambda name: name != "proctap", audio_output_probe=lambda: None
        )
        == "native_dependency_unavailable:proctap"
    )

    def blocked_audio() -> None:
        raise RuntimeError("no output")

    assert (
        classify_native_capability(
            module_available=lambda _name: True, audio_output_probe=blocked_audio
        )
        == "native_audio_output_unavailable"
    )


@pytest.mark.asyncio
async def test_run_records_classified_block_instead_of_false_pass(
    tmp_path: Path, monkeypatch
) -> None:
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "target_present_amplitude_min": 0.05,
                "control_excluded_amplitude_max": 0.005,
                "control_to_target_ratio_max": 0.1,
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    monkeypatch.setattr(
        "puripuly_heart.release_evidence.windows_process_isolation.classify_native_capability",
        lambda: "unsupported_python",
    )

    exit_code = await run(evidence, thresholds)
    artifact = json.loads(evidence.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert artifact["status"] == "blocked"
    assert artifact["classification"] == "unsupported_python"
    assert artifact["measurements"] is None


@pytest.mark.asyncio
async def test_supported_host_fixture_exception_is_failed_not_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "target_present_amplitude_min": 0.05,
                "control_excluded_amplitude_max": 0.005,
                "control_to_target_ratio_max": 0.1,
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    monkeypatch.setattr(
        "puripuly_heart.release_evidence.windows_process_isolation.classify_native_capability",
        lambda: None,
    )

    async def fail_native(_thresholds, _runtime_dir) -> dict[str, object]:  # noqa: ANN001
        raise ImportError("raw import detail")

    monkeypatch.setattr(
        "puripuly_heart.release_evidence.windows_process_isolation._run_native",
        fail_native,
    )

    exit_code = await run(evidence, thresholds)
    artifact = json.loads(evidence.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert artifact["status"] == "failed"
    assert artifact["classification"] == "fixture_import_failed"
    assert "raw import detail" not in json.dumps(artifact)


@pytest.mark.parametrize(
    ("code", "classification"),
    [
        ("direct_child_capture_timeout", "direct_child_capture_timeout"),
        ("native_capture_timeout", "peer_runtime_capture_timeout"),
        ("peer_runtime_faulted_before_frames", "peer_runtime_faulted_before_frames"),
        ("peer_runtime_loop_completed_before_frames", "peer_runtime_loop_completed_before_frames"),
    ],
)
def test_capture_timeout_classification_preserves_attributable_stage(
    code: str, classification: str
) -> None:
    from puripuly_heart.release_evidence.windows_process_isolation import classify_fixture_failure

    assert classify_fixture_failure(RuntimeError(code)) == classification
