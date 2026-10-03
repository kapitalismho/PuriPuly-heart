from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bench_ovr_hmd_measurement as measurement

sys.path.insert(0, str(Path(measurement.__file__).parent))
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from scripts import hmd_issue206_scenarios as scenarios


@pytest.mark.parametrize(
    "confirmed,names",
    [
        (False, set()),
        (True, set()),
        (True, {"vrserver.exe", "vrcompositor.exe", "PuriPulyHeartOverlay.exe"}),
    ],
)
def test_live_guard_fail_closed(confirmed, names):
    with pytest.raises(measurement.MeasurementError):
        measurement.validate_live_guard(names, confirmed_hmd_ready=confirmed)


def test_direct_launch_guard_precedes_loading_and_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(
        measurement,
        "load_prepared_stage",
        lambda p: pytest.fail("identity must not load before missing confirmation guard"),
    )
    with pytest.raises(measurement.MeasurementError, match="confirm-hmd-ready"):
        measurement.launch_arm(
            tmp_path, arm="candidate", scenario="stable", anchor="head_locked", live=True
        )


@pytest.mark.asyncio
async def test_direct_runtime_guard_cannot_be_bypassed():
    with pytest.raises(measurement.MeasurementError, match="confirm-hmd-ready"):
        await scenarios.run_measurement(SimpleNamespace(live=True, confirm_hmd_ready=False))


def test_startup_contract_acceptance_and_extra_capability_rejection(monkeypatch):
    payload = dict(measurement.EXPECTED_STARTUP_CONTRACT)
    monkeypatch.setattr(
        measurement.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
    )
    assert measurement._check_startup_contract(Path("native.exe"))["vr_initialization"] is False
    payload["unexpected"] = True
    with pytest.raises(measurement.MeasurementError, match="mismatch"):
        measurement._check_startup_contract(Path("native.exe"))


def test_complete_tree_identity_detects_source_and_lock_tampering(tmp_path):
    (tmp_path / "src").mkdir()
    code = tmp_path / "src" / "owner.py"
    lock = tmp_path / "uv.lock"
    code.write_text("before", encoding="utf-8")
    lock.write_text("lock", encoding="utf-8")
    original = measurement._tree_identity(tmp_path)
    code.write_text("after", encoding="utf-8")
    assert measurement._tree_identity(tmp_path) != original
    original = measurement._tree_identity(tmp_path)
    lock.write_text("changed lock", encoding="utf-8")
    assert measurement._tree_identity(tmp_path) != original


@pytest.mark.parametrize(
    "mode,outcome", [("offline_dry_run", "pass"), ("live", "failed"), ("live", "pass")]
)
def test_observation_never_upgrades_offline_or_failed_run(tmp_path, mode, outcome):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "schema": measurement.RUN_SCHEMA,
                "run_id": "owned-run",
                "mode": mode,
                "software": {"outcome": outcome},
            }
        ),
        encoding="utf-8",
    )
    if mode != "live":
        with pytest.raises(measurement.MeasurementError, match="offline"):
            measurement.record_observation(
                path, result="no_issue", note="readable", uncertainty="manual"
            )
    else:
        observation = measurement.record_observation(
            path, result="no_issue", note="readable", uncertainty="manual"
        )
        payload = json.loads(observation.read_text(encoding="utf-8"))
        assert payload["overall_outcome"] == (
            "failed" if outcome == "failed" else "observation_recorded"
        )
        assert payload["new_run_performed"] is False
        assert json.loads(path.read_text(encoding="utf-8"))["software"]["outcome"] == outcome


def test_stop_is_run_correlated_and_does_not_terminate_processes(tmp_path):
    run = tmp_path / "runs" / "candidate-stable-1234"
    run.mkdir(parents=True)
    (run / "owned.json").write_text("{}", encoding="utf-8")
    assert measurement.request_stop(tmp_path, run.name).is_file()
    with pytest.raises(measurement.MeasurementError):
        measurement.request_stop(tmp_path, "../unrelated")


@pytest.mark.parametrize("duration,timeout", [(61, 90), (0, 90), (10, 121), (float("nan"), 90)])
def test_duration_and_timeout_bounded_before_launch(tmp_path, duration, timeout):
    with pytest.raises(measurement.MeasurementError):
        measurement.launch_arm(
            tmp_path,
            arm="candidate",
            scenario="sustained",
            anchor="head_locked",
            duration=duration,
            timeout=timeout,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", measurement.SCENARIOS)
async def test_actual_scenario_engine_semantics_and_owned_cleanup(name):
    presenter = OverlayPresenter(calibration=OverlayCalibration(), native_retry_enabled=True)
    records = []
    args = SimpleNamespace(arm="candidate", scenario=name, live=False, duration=1)
    restart_calls = []

    async def restart():
        restart_calls.append(True)
        await presenter.begin_native_retry_epoch(enabled=True)

    engine = scenarios.ScenarioEngine(presenter, args, records, restart)
    try:
        async with asyncio.timeout(20):
            await engine.run()
        assert records
        assert any(r["stage"] == "provider_call" for r in records)
        assert any(r["stage"] == "Presenter_application" for r in records)
        assert len(engine.provider.calls) <= 4
        if name in ("independent", "stable", "final_only"):
            assert engine.checks["before_ui_release"]["source_applied"] is True
            assert engine.checks["identity_preserved"] is True
            assert len(engine.provider.calls) == 1
        if name == "stable_burst":
            assert engine.checks["stable_updates"] == 4
        if name == "restart_reconnect":
            assert len(restart_calls) == 2
    finally:
        await engine.close()
        await presenter.close()
    assert engine.consumer.done()
    assert all(task.done() for task in engine.ingress)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_real_owner_engine_cancellation_and_timeout_cleanup(termination):
    presenter = OverlayPresenter(calibration=OverlayCalibration(), native_retry_enabled=True)
    engine = scenarios.ScenarioEngine(
        presenter,
        SimpleNamespace(arm="candidate", scenario="expiry", live=False, duration=1),
        [],
        None,
    )
    task = asyncio.create_task(engine.run())
    try:
        await scenarios.until(lambda: bool(engine.provider.calls))
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await engine.close()
        await presenter.close()
    assert engine.consumer.done()


@pytest.mark.parametrize(
    "change",
    [{"forced": True}, {"exit_confirmed": False}, {"exit_code": 1}, {"acknowledged": False}],
)
def test_shutdown_requires_normal_owned_exit(change):
    receipt = {
        "cleanup_succeeded": True,
        "exit_confirmed": True,
        "graceful_completed": True,
        "acknowledged": True,
        "forced": False,
        "exit_code": 0,
        "terminal_cause": None,
    }
    assert scenarios.shutdown_ok(receipt)
    receipt.update(change)
    assert not scenarios.shutdown_ok(receipt)


def test_import_guard_rejects_current_checkout_for_exported_arm(tmp_path):
    with pytest.raises(measurement.MeasurementError, match="import separation"):
        scenarios.validate_arm_imports(tmp_path / "source" / "baseline")


def test_sustained_requires_software_success_and_correlated_no_issue_observation(tmp_path):
    for name in measurement.SCENARIOS[:-1]:
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        report = run / "report.json"
        report.write_text(
            json.dumps(
                {
                    "mode": "live",
                    "arm": "candidate",
                    "anchor": "head_locked",
                    "scenario": name,
                    "software": {"outcome": "pass"},
                }
            ),
            encoding="utf-8",
        )
        (run / "observation.json").write_text(
            json.dumps({"result": "no_issue", "run_report_sha256": measurement._sha256(report)}),
            encoding="utf-8",
        )
    measurement.validate_sustained_guard(tmp_path, "candidate", "head_locked")
    with pytest.raises(measurement.MeasurementError):
        measurement.validate_sustained_guard(tmp_path, "candidate", "spatial_locked")
    (tmp_path / "runs" / "expiry" / "observation.json").unlink()
    with pytest.raises(measurement.MeasurementError):
        measurement.validate_sustained_guard(tmp_path, "candidate", "head_locked")


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["stop", "cancel", "timeout", "success", "bad_shutdown"])
async def test_complete_runtime_cleanup_and_truthful_receipts_with_explicit_fake_native(
    tmp_path, monkeypatch, termination
):
    run_id = "explicit-fake-native-" + termination
    run = tmp_path / "runs" / run_id
    run.mkdir(parents=True)
    (run / "owned.json").write_text("{}", encoding="utf-8")
    lock = tmp_path / "active-live.lock"
    lock.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    monkeypatch.setattr(measurement, "LIVE_LOCK", lock)
    monkeypatch.setattr(
        measurement,
        "load_prepared_stage",
        lambda p: {
            "control": {},
            "native": {},
            "sources": {"candidate": {"tree": {"sha256": "explicit-test-boundary"}}},
        },
    )
    monkeypatch.setattr(scenarios, "validate_arm_imports", lambda p: "explicit-test-boundary")
    monkeypatch.setattr(
        measurement, "inspect_process_names", lambda: {"vrserver.exe", "vrcompositor.exe"}
    )
    monkeypatch.setattr(
        measurement, "environment_inventory", lambda: {"os": "explicit-test-boundary"}
    )
    for key in tuple(scenarios.os.environ):
        if key.startswith("PURIPULY_"):
            monkeypatch.delenv(key)
    managers = []

    class FakeManager:
        def __init__(self, **kwargs):
            self.state = "off"
            self.failure_reason = None
            assert kwargs["bridge_messages_authenticated"] is True
            assert kwargs["handoff_experiment"] == "off"
            managers.append(self)

        async def start(self):
            self.state = "connected"

        async def stop(self):
            self.state = "off"

        def shutdown_receipt(self):
            return {
                "cleanup_succeeded": True,
                "exit_confirmed": True,
                "graceful_completed": True,
                "acknowledged": True,
                "forced": termination == "bad_shutdown",
                "exit_code": 0,
                "terminal_cause": None,
            }

    monkeypatch.setattr(scenarios, "OverlayProcessManager", FakeManager)
    args = SimpleNamespace(
        stage=tmp_path,
        arm="candidate",
        anchor="head_locked",
        run_id=run_id,
        live=True,
        confirm_hmd_ready=True,
        duration=1,
        timeout=1 if termination == "timeout" else 20,
        scenario="independent" if termination in ("success", "bad_shutdown") else "expiry",
        device="explicit-test-boundary",
        firmware="unknown",
        connection="unknown",
    )
    task = asyncio.create_task(scenarios.run_measurement(args))
    if termination in ("stop", "cancel"):
        await scenarios.until(lambda: bool(managers) and managers[0].state == "connected")
        await asyncio.sleep(0.05)
        if termination == "stop":
            measurement.request_stop(tmp_path, run_id)
        else:
            task.cancel()
    if termination == "success":
        await task
    elif termination == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(measurement.MeasurementError):
            await task
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["physical_hmd"]["result"] == "not_observed"
    assert managers[0].state == "off"
    assert report["software"]["outcome"] == ("pass" if termination == "success" else "failed")
    assert report["software"]["cleanup"] == (
        "failed" if termination == "bad_shutdown" else "complete"
    )
    if termination in ("stop", "timeout", "cancel"):
        assert (
            report["software"]["failure_reason"]
            == {"stop": "operator_stop", "timeout": "run_timeout", "cancel": "cancelled"}[
                termination
            ]
        )


@pytest.mark.parametrize(
    "surface", ["source", "lock", "native", "harness", "interpreter", "metadata"]
)
def test_prepared_stage_loader_rejects_identity_tampering(tmp_path, monkeypatch, surface):
    control = tmp_path / "control"
    control.mkdir()
    control_script = control / "bench_ovr_hmd_measurement.py"
    control_script.write_bytes(Path(measurement.__file__).read_bytes())
    (control / "hmd_issue206_scenarios.py").write_text("helper", encoding="utf-8")
    native = tmp_path / "runtime" / "run"
    native.mkdir(parents=True)
    executable = native / "PuriPulyHeartOverlay.exe"
    executable.write_bytes(b"explicit-test-native")
    monkeypatch.setattr(measurement, "SOURCE_EXE_SHA256", measurement._sha256(executable))
    monkeypatch.setattr(
        measurement, "_interpreter_identity", lambda: {"identity": "explicit-test-interpreter"}
    )
    identities = {}
    for arm, revision in measurement.ARMS.items():
        source = tmp_path / "source" / arm
        source.mkdir(parents=True)
        (source / "uv.lock").write_text("locked", encoding="utf-8")
        (source / "owner.py").write_text(arm, encoding="utf-8")
        archive = source.with_suffix(".tar")
        archive.write_bytes(arm.encode())
        identities[arm] = {
            "revision": revision,
            "archive_sha256": measurement._sha256(archive),
            "tree": measurement._tree_identity(source),
            "lock_sha256": measurement._sha256(source / "uv.lock"),
        }
    payload = {
        "schema": measurement.SCHEMA,
        "sources": identities,
        "control": measurement._tree_identity(control),
        "runtime": measurement._tree_identity(tmp_path / "runtime"),
        "startup": {"stdout": measurement.EXPECTED_STARTUP_CONTRACT},
        "interpreter": measurement._interpreter_identity(),
    }
    report = tmp_path / "preparation.json"
    measurement._write_json(report, payload)
    (tmp_path / "preparation.sha256").write_text(measurement._sha256(report), encoding="ascii")
    monkeypatch.setattr(
        measurement,
        "_check_startup_contract",
        lambda p: pytest.fail("inspect must not execute native"),
    )
    assert measurement.load_prepared_stage(tmp_path)["schema"] == measurement.SCHEMA
    if surface == "source":
        (tmp_path / "source" / "baseline" / "owner.py").write_text("changed", encoding="utf-8")
    elif surface == "lock":
        (tmp_path / "source" / "candidate" / "uv.lock").write_text("changed", encoding="utf-8")
    elif surface == "native":
        executable.write_bytes(b"changed-native")
    elif surface == "harness":
        (control / "hmd_issue206_scenarios.py").write_text("changed", encoding="utf-8")
    elif surface == "interpreter":
        monkeypatch.setattr(measurement, "_interpreter_identity", lambda: {"identity": "changed"})
    else:
        report.write_text("{}", encoding="utf-8")
    with pytest.raises(measurement.MeasurementError, match="identity"):
        measurement.load_prepared_stage(tmp_path)


@pytest.mark.asyncio
async def test_direct_live_refuses_inherited_experiment_overrides(monkeypatch):
    monkeypatch.setattr(
        measurement, "inspect_process_names", lambda: {"vrserver.exe", "vrcompositor.exe"}
    )
    monkeypatch.setenv("PURIPULY_SKIP_VR_PREFLIGHT", "1")
    args = SimpleNamespace(live=True, confirm_hmd_ready=True, scenario="stable")
    with pytest.raises(measurement.MeasurementError, match="overrides refused"):
        await scenarios.run_measurement(args)
