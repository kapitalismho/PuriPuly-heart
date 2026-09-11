from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bench_ovr_hmd_measurement as measurement


def test_live_guard_refuses_preexisting_overlay_without_stopping_it() -> None:
    processes = {
        "vrserver.exe",
        "vrcompositor.exe",
        "PuriPulyHeartOverlay.exe",
    }

    with pytest.raises(measurement.MeasurementError, match="already running"):
        measurement.validate_live_guard(processes, confirmed_hmd_ready=True)

    assert "PuriPulyHeartOverlay.exe" in processes


def test_startup_contract_rejects_extra_or_changed_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = dict(measurement.EXPECTED_STARTUP_CONTRACT)
    mismatched["unexpected_capability"] = {"version": 1}
    monkeypatch.setattr(
        measurement.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(mismatched),
            stderr="",
        ),
    )

    with pytest.raises(measurement.MeasurementError, match="startup contract mismatch"):
        measurement._check_startup_contract(Path("candidate.exe"))


def test_manual_observation_cannot_upgrade_failed_software_run(tmp_path: Path) -> None:
    run_report = tmp_path / "run-failed.json"
    run_report.write_text(
        json.dumps(
            {
                "schema": measurement.RUN_SCHEMA,
                "run_id": "failed-run",
                "session": "session-local",
                "software": {"outcome": "failed", "failure_reason": "runtime_crashed"},
            }
        ),
        encoding="utf-8",
    )

    observation_path = measurement.record_observation(
        run_report,
        result="no_issue",
        note="No visual issue noticed before the software failure.",
        uncertainty="manual observation; timing unknown",
    )
    observation = json.loads(observation_path.read_text(encoding="utf-8"))

    assert observation["software_run_outcome"] == "failed"
    assert observation["overall_outcome"] == "failed"
    assert observation["latency_claim"] == "not_measured"
    assert observation["cannot_upgrade_failed_software_run"] is True


@pytest.mark.asyncio
async def test_offline_run_closes_owned_bridge_tasks_without_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preparation = {
        "session": "cleanup-session",
        "pair": {"protocol": 8},
        "provenance": {"accepted_source": measurement.ACCEPTED_SOURCE},
    }
    monkeypatch.setattr(
        measurement,
        "load_prepared_stage",
        lambda stage: (preparation, tmp_path / "unused.exe", tmp_path / "unused.dll"),
    )

    async def short_sequence(*args, **kwargs):
        return ([{"step": "synthetic", "outcome": "applied"}], 0.01)

    monkeypatch.setattr(measurement, "_run_fixed_sequence", short_sequence)

    report_path = await measurement.run_measurement(
        tmp_path,
        live=False,
        hold_seconds=3.0,
        idle_seconds=30.0,
        run_timeout_seconds=2.0,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["software"]["outcome"] == "pass"
    assert report["software"]["cleanup"] == "complete"
    assert report["software"]["owned_child_exit"] == "not_applicable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("receipt", "expected_outcome", "expected_reason"),
    [
        (
            {
                "graceful_request": "sent",
                "acknowledged": True,
                "terminate_requested": False,
                "kill_requested": False,
                "forced": False,
                "exit_confirmed": True,
                "exit_code": 0,
                "reader_cleanup": "complete",
                "graceful_completed": True,
                "cleanup_succeeded": True,
                "terminal_cause": None,
                "stdout_events": [{"type": "shutdown_complete"}],
                "stderr_diagnostics": [],
            },
            "pass",
            None,
        ),
        (
            {
                "acknowledged": True,
                "forced": False,
                "exit_confirmed": True,
                "exit_code": 1,
                "cleanup_succeeded": False,
                "terminal_cause": "runtime_exit_nonzero",
            },
            "failed",
            "runtime_exit_nonzero",
        ),
        (
            {
                "acknowledged": False,
                "forced": True,
                "exit_confirmed": True,
                "exit_code": 0,
                "cleanup_succeeded": False,
                "terminal_cause": "shutdown_forced",
            },
            "failed",
            "shutdown_forced",
        ),
        (
            {
                "acknowledged": True,
                "forced": False,
                "exit_confirmed": False,
                "exit_code": None,
                "cleanup_succeeded": False,
                "terminal_cause": "termination_unconfirmed",
            },
            "failed",
            "termination_unconfirmed",
        ),
    ],
)
async def test_live_run_requires_confirmed_normal_shutdown_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    receipt: dict[str, object],
    expected_outcome: str,
    expected_reason: str | None,
) -> None:
    preparation = {
        "session": "synthetic-live-session",
        "pair": {"protocol": 8},
        "provenance": {"accepted_source": measurement.ACCEPTED_SOURCE},
    }

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            self.last_process = SimpleNamespace(returncode=receipt.get("exit_code"))

    class FakeManager:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            self.state = "off"
            self.failure_reason: str | None = None

        async def start(self) -> None:
            self.state = "connected"

        def mark_shutdown_requested(self, *, request_sent: bool = True) -> None:
            _ = request_sent

        async def stop(self) -> None:
            terminal_cause = receipt.get("terminal_cause")
            self.failure_reason = terminal_cause if isinstance(terminal_cause, str) else None
            self.state = "off" if terminal_cause is None else "failed"

        def shutdown_receipt(self) -> dict[str, object]:
            return dict(receipt)

    async def short_sequence(*args: object, **kwargs: object):
        _ = (args, kwargs)
        return ([{"step": "synthetic", "outcome": "applied"}], 0.01)

    monkeypatch.setattr(
        measurement,
        "load_prepared_stage",
        lambda stage: (preparation, tmp_path / "synthetic.exe", tmp_path / "synthetic.dll"),
    )
    monkeypatch.setattr(measurement, "MeasurementProcessRunner", FakeRunner)
    monkeypatch.setattr(measurement, "OverlayProcessManager", FakeManager)
    monkeypatch.setattr(measurement, "_run_fixed_sequence", short_sequence)
    monkeypatch.setattr(measurement.secrets, "token_hex", lambda size: "receipt")

    if expected_outcome == "pass":
        report_path = await measurement.run_measurement(
            tmp_path,
            live=True,
            hold_seconds=3.0,
            idle_seconds=30.0,
            run_timeout_seconds=2.0,
        )
    else:
        with pytest.raises(measurement.MeasurementError, match=expected_reason):
            await measurement.run_measurement(
                tmp_path,
                live=True,
                hold_seconds=3.0,
                idle_seconds=30.0,
                run_timeout_seconds=2.0,
            )
        report_path = tmp_path / "run-live-off-receipt.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["software"]["outcome"] == expected_outcome
    assert report["software"]["failure_reason"] == expected_reason
    assert report["software"]["cleanup"] == ("complete" if expected_outcome == "pass" else "failed")
    assert report["software"]["shutdown"] == receipt


def test_experiment_cli_requires_explicit_arm_and_exposes_only_approved_arms() -> None:
    parser = measurement.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["dry-run", "--stage", "prepared"])

    off = parser.parse_args(["dry-run", "--stage", "prepared", "--arm", "off"])
    cached = parser.parse_args(["live", "--stage", "prepared", "--arm", "cached_frame_rehandoff"])
    assert off.arm == "off"
    assert cached.arm == "cached_frame_rehandoff"


@pytest.mark.parametrize("hold", ["0.05", "2.999", "3.001", "30", "nan", "inf", "-inf"])
def test_live_hold_is_exactly_preregistered_three_seconds(hold: str) -> None:
    parser = measurement.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "live",
                "--stage",
                "prepared",
                "--arm",
                "off",
                "--hold-seconds",
                hold,
            ]
        )

    offline = parser.parse_args(
        [
            "dry-run",
            "--stage",
            "prepared",
            "--arm",
            "off",
            "--hold-seconds",
            "0.05",
        ]
    )
    assert offline.hold_seconds == 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hold",
    [0.05, 2.999, 3.001, 30.0, float("nan"), float("inf"), float("-inf")],
)
async def test_live_hold_contract_is_rejected_before_stage_or_process_launch(
    hold: float,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        measurement.MeasurementError,
        match="exactly 3.0 seconds",
    ):
        await measurement.run_measurement(
            tmp_path / "not-loaded",
            live=True,
            hold_seconds=hold,
            idle_seconds=30.0,
            run_timeout_seconds=120.0,
            arm="off",
        )


@pytest.mark.asyncio
async def test_offline_arm_report_is_experiment_only_without_claiming_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preparation = {
        "session": "paired-session",
        "pair": {"protocol": 8},
        "provenance": {"accepted_source": measurement.ACCEPTED_SOURCE},
    }
    monkeypatch.setattr(
        measurement,
        "load_prepared_stage",
        lambda stage: (preparation, tmp_path / "unused.exe", tmp_path / "unused.dll"),
    )

    async def short_sequence(*args, **kwargs):
        return ([{"step": "synthetic", "outcome": "applied"}], 0.01)

    monkeypatch.setattr(measurement, "_run_fixed_sequence", short_sequence)
    report_path = await measurement.run_measurement(
        tmp_path,
        live=False,
        hold_seconds=3.0,
        idle_seconds=30.0,
        run_timeout_seconds=2.0,
        arm="cached_frame_rehandoff",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["experiment"]["arm"] == "cached_frame_rehandoff"
    assert report["experiment"]["experiment_only"] is True
    assert report["experiment"]["qualifies_for_r2_conformance"] is False
    assert report["experiment"]["discrimination"] == "not_observed"
    assert report["software"]["diagnostics"]["outcome"] == "written"
