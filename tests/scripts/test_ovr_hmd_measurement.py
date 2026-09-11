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
        "pair": {"protocol": 7},
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
