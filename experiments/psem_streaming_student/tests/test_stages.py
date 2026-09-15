from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "stages.py"
SPEC = importlib.util.spec_from_file_location("psem_streaming_stages", MODULE_PATH)
assert SPEC and SPEC.loader
stages = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stages)


def write_plan(path: Path, stage_values: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "stages": stage_values}), encoding="utf-8")


def stage(stage_id: str, code: str, *outputs: str) -> dict[str, object]:
    return {
        "id": stage_id,
        "argv": ["{python}", "-c", code, "{run_root}"],
        "timeout_seconds": 20,
        "required_outputs": list(outputs),
    }


def wait_for(path: Path, process: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if process is not None and process.poll() is not None:
            raise AssertionError(f"driver exited early: {process.communicate()}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def invoke(command: str, state_root: Path, plan: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            command,
            "--state-root",
            str(state_root),
            "--plan",
            str(plan),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_pause_persists_at_stage_boundary_and_resume_skips_completed(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    first = "import pathlib,sys,time; r=pathlib.Path(sys.argv[1]); (r/'first-started').write_text('1');\nwhile not (r/'release').exists(): time.sleep(.01)\n(r/'first-result').write_text('done')"
    second = "import pathlib,sys; r=pathlib.Path(sys.argv[1]); p=r/'second-count'; p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"
    write_plan(
        plan, [stage("first", first, "first-result"), stage("second", second, "second-count")]
    )
    driver = subprocess.Popen(
        [
            sys.executable,
            str(MODULE_PATH),
            "run",
            "--state-root",
            str(state_root),
            "--plan",
            str(plan),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wait_for(state_root / "first-started", driver)
    paused = invoke("pause", state_root, plan)
    assert paused.returncode == 0
    assert json.loads(paused.stdout)["pause_requested"] is True
    (state_root / "release").write_text("go", encoding="utf-8")
    stdout, stderr = driver.communicate(timeout=20)
    assert driver.returncode == 0, stderr
    boundary = json.loads(stdout)
    assert boundary["status"] == "PAUSED"
    assert boundary["completed_stages"] == ["first"]
    assert (state_root / "first-result").read_text() == "done"
    assert not (state_root / "second-count").exists()
    restarted_status = invoke("status", state_root, plan)
    assert json.loads(restarted_status.stdout) == boundary
    resumed = invoke("resume", state_root, plan)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["status"] == "COMPLETED"
    assert (state_root / "second-count").read_text() == "1"
    state = json.loads((state_root / "state.json").read_text())
    assert state["completed_stages"] == ["first", "second"]
    assert [item["stage_id"] for item in state["stage_outcomes"]] == ["first", "second"]
    assert len(list((state_root / "logs").glob("first.attempt-*.stdout.log"))) == 1


def test_pause_wins_before_next_stage_launch(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    write_plan(
        plan_path,
        [
            stage(
                "only",
                "import pathlib,sys; (pathlib.Path(sys.argv[1])/'ran').write_text('x')",
                "ran",
            )
        ],
    )
    state, frozen = stages.initialize(state_root, plan_path)
    result = stages.pause(state_root, plan_path)
    assert result["status"] == "PAUSED"
    assert stages.status(state_root, plan_path)["next_stage"] == "only"
    stages.execute_stage(frozen["stages"][0], state_root, stages.paths(state_root), state)
    assert not (state_root / "ran").exists()
    assert not (state_root / "logs").exists()


def test_duplicate_driver_is_rejected(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    blocking = "import pathlib,sys,time; r=pathlib.Path(sys.argv[1]); (r/'started').write_text('1');\nwhile not (r/'release').exists(): time.sleep(.01)"
    write_plan(plan, [stage("blocking", blocking)])
    driver = subprocess.Popen(
        [
            sys.executable,
            str(MODULE_PATH),
            "run",
            "--state-root",
            str(state_root),
            "--plan",
            str(plan),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wait_for(state_root / "started", driver)
    duplicate = invoke("resume", state_root, plan)
    assert duplicate.returncode == 2
    assert "lock is already held" in duplicate.stderr
    (state_root / "release").write_text("go")
    driver.communicate(timeout=20)
    assert driver.returncode == 0


def test_failure_is_retained_and_does_not_advance(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    write_plan(
        plan,
        [
            stage(
                "bad", "import sys; print('failure evidence', file=sys.stderr); raise SystemExit(7)"
            ),
            stage(
                "never",
                "import pathlib,sys; (pathlib.Path(sys.argv[1])/'advanced').write_text('bad')",
            ),
        ],
    )
    result = invoke("run", state_root, plan)
    summary = json.loads(result.stdout)
    assert summary["status"] == "FAILED"
    assert result.returncode == 1
    assert not (state_root / "advanced").exists()
    state = json.loads((state_root / "state.json").read_text())
    assert state["stage_outcomes"][0]["return_code"] == 7
    stderr_receipt = Path(state["stage_outcomes"][0]["stderr"]["path"])
    assert "failure evidence" in stderr_receipt.read_text()
    blocked = invoke("resume", state_root, plan)
    assert blocked.returncode == 2
    assert "failed stage blocks continuation" in blocked.stderr
    assert len(state["stage_outcomes"]) == 1


def test_control_uses_frozen_plan_but_resume_rejects_external_plan_change(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    write_plan(
        plan,
        [
            stage(
                "one",
                "import pathlib,sys; (pathlib.Path(sys.argv[1])/'one').write_text('1')",
                "one",
            )
        ],
    )
    stages.initialize(state_root, plan)
    write_plan(
        plan,
        [
            stage(
                "one",
                "import pathlib,sys; (pathlib.Path(sys.argv[1])/'changed').write_text('1')",
                "changed",
            )
        ],
    )
    paused = invoke("pause", state_root, plan)
    assert paused.returncode == 0
    assert json.loads(paused.stdout)["status"] == "PAUSED"
    assert invoke("status", state_root, plan).returncode == 0
    changed = invoke("resume", state_root, plan)
    assert changed.returncode == 2
    assert "requested plan identity differs" in changed.stderr
    plan.unlink()
    assert invoke("status", state_root, plan).returncode == 0
    missing = invoke("resume", state_root, plan)
    assert missing.returncode == 2
    assert "file does not exist" in missing.stderr
    assert not (state_root / "changed").exists()


def test_state_root_file_returns_structured_error_without_clobbering(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "occupied"
    state_root.write_text("owned", encoding="utf-8")
    write_plan(plan, [stage("one", "raise SystemExit(0)")])
    result = invoke("run", state_root, plan)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"].startswith("cannot create state root")
    assert state_root.read_text(encoding="utf-8") == "owned"


def test_manifest_rejects_unknown_keys_and_invalid_timeout(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    value = stage("one", "raise SystemExit(0)")
    value["timeouts_seconds"] = 5
    write_plan(plan, [value])
    typo = invoke("run", state_root, plan)
    assert typo.returncode == 2
    assert "unknown keys: timeouts_seconds" in typo.stderr
    assert not (state_root / "state.json").exists()
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stages": [stage("one", "raise SystemExit(0)")],
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )
    unknown = invoke("run", state_root, plan)
    assert unknown.returncode == 2
    assert "plan has unknown keys: unexpected" in unknown.stderr
    value = stage("one", "raise SystemExit(0)")
    value["timeout_seconds"] = float("inf")
    write_plan(plan, [value])
    nonfinite = invoke("run", state_root, plan)
    assert nonfinite.returncode == 2
    assert "timeout_seconds must be an integer" in nonfinite.stderr


def test_frozen_only_initialization_recovers_once_and_mismatch_refuses(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "recoverable"
    state_root.mkdir()
    one = stage(
        "one",
        "import pathlib,sys; p=pathlib.Path(sys.argv[1])/'count'; p.write_text('1')",
        "count",
    )
    write_plan(plan, [one])
    frozen = stages.validate_plan(json.loads(plan.read_text(encoding="utf-8")))
    stages.atomic_json(state_root / "frozen_plan.json", frozen)
    recovered = invoke("run", state_root, plan)
    assert recovered.returncode == 0, recovered.stderr
    assert (state_root / "count").read_text(encoding="utf-8") == "1"
    second = invoke("run", state_root, plan)
    assert second.returncode == 2
    assert (state_root / "count").read_text(encoding="utf-8") == "1"

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    stages.atomic_json(mismatch_root / "frozen_plan.json", frozen)
    write_plan(plan, [stage("changed", "raise SystemExit(0)")])
    mismatch = invoke("run", mismatch_root, plan)
    assert mismatch.returncode == 2
    assert "recoverable frozen initialization" in mismatch.stderr
    assert not (mismatch_root / "state.json").exists()


def test_interrupted_driver_exposes_untrusted_live_child_metadata(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    state_root = tmp_path / "state"
    blocking = "import pathlib,sys,time; r=pathlib.Path(sys.argv[1]); (r/'started').write_text('1');\nwhile not (r/'release').exists(): time.sleep(.01)\n(r/'uncommitted').write_text('done'); (r/'cleanup-ready').write_text('1'); deadline=time.monotonic()+10\nwhile time.monotonic()<deadline: time.sleep(.1)"
    write_plan(plan, [stage("orphan", blocking, "uncommitted")])
    driver = subprocess.Popen(
        [
            sys.executable,
            str(MODULE_PATH),
            "run",
            "--state-root",
            str(state_root),
            "--plan",
            str(plan),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wait_for(state_root / "started", driver)
    driver.terminate()
    driver.communicate(timeout=10)
    interrupted = invoke("resume", state_root, plan)
    assert interrupted.returncode == 2
    current = json.loads(invoke("status", state_root, plan).stdout)
    child = current["last_recorded_child"]
    assert current["status"] == "INTERRUPTED"
    assert child["stage_id"] == "orphan"
    assert isinstance(child["pid"], int)
    assert "may still be running" in current["interruption_warning"]
    assert current["completed_stages"] == []
    (state_root / "release").write_text("go", encoding="utf-8")
    wait_for(state_root / "uncommitted")
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    wait_for(state_root / "cleanup-ready")
    assert state["stage_outcomes"] == []
    assert state["completed_stages"] == []
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(child["pid"]), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        try:
            os.kill(child["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
