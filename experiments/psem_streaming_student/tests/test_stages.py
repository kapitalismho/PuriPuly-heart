from __future__ import annotations

import importlib.util
import json
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
    stages.initialize(state_root, plan_path)
    result = stages.pause(state_root, plan_path)
    assert result["status"] == "PAUSED"
    assert stages.status(state_root, plan_path)["next_stage"] == "only"
    assert not (state_root / "ran").exists()


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


def test_plan_change_refuses_resume(tmp_path: Path) -> None:
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
    result = invoke("resume", state_root, plan)
    assert result.returncode == 2
    assert "plan identity differs" in result.stderr
    assert not (state_root / "changed").exists()
