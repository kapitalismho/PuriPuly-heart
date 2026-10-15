from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci import check_test_selection as checker
from tests.helpers.paths import REPO_ROOT


def _source(root: Path, name: str, content: str = "def test_example():\n    assert True\n") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _workflow(root: Path, command: str = "uv run --locked pytest -ra tests/unit") -> None:
    path = root / checker.WORKFLOW
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"jobs": {"offline": {"steps": [{"run": command}]}}}),
        encoding="utf-8",
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    _source(tmp_path, "tests/unit/test_selected.py")
    _source(tmp_path, "tests/integration/test_live.py")
    _workflow(tmp_path)
    return tmp_path


def test_actual_workflow_accounts_for_current_test_sources() -> None:
    report = checker.check_selection(REPO_ROOT)
    assert report.selected
    assert all(not name.startswith("tests/integration/") for name in report.selected)
    assert report.excluded
    assert all(name.startswith("tests/integration/") for name in report.excluded)


def test_pr_python_setup_and_invocations_are_locked_without_changing_release_default() -> None:
    workflow = yaml.safe_load((REPO_ROOT / checker.WORKFLOW).read_text(encoding="utf-8"))
    action = yaml.safe_load(
        (REPO_ROOT / ".github/actions/setup-uv-environment/action.yml").read_text(encoding="utf-8")
    )
    assert action["inputs"]["lock-mode"]["default"] == "frozen"
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if step.get("uses") == "./.github/actions/setup-uv-environment":
                assert step["with"]["lock-mode"] == "locked"
            command = step.get("run", "").strip()
            if command.startswith("uv run "):
                assert command.split()[:3] == ["uv", "run", "--locked"]


def test_pr_job_responsibilities_remain_nine_stable_checks() -> None:
    workflow = yaml.safe_load((REPO_ROOT / checker.WORKFLOW).read_text(encoding="utf-8"))
    names = set()
    for job in workflow["jobs"].values():
        if job["name"] == "${{ matrix.name }}":
            names.update(row["name"] for row in job["strategy"]["matrix"]["include"])
        else:
            names.add(job["name"])
    assert names == {
        "Lint",
        "Contracts",
        "Architecture",
        "Runtime",
        "Providers",
        "UI",
        "Tooling",
        "Broker",
        "Overlay",
    }


def test_selected_and_justified_live_exclusion_are_accepted(project: Path) -> None:
    report = checker.check_selection(project)
    assert report.selected == {"tests/unit/test_selected.py"}
    assert report.excluded == {"tests/integration/test_live.py"}
    assert "credential" in checker.INTENTIONAL_EXCLUSIONS["tests/integration"]


def test_new_ordinary_file_cannot_disappear(project: Path) -> None:
    _source(project, "tests/new_area/test_omitted.py")
    with pytest.raises(checker.SelectionError, match="tests/new_area/test_omitted.py"):
        checker.check_selection(project)


def test_default_pytest_suffix_pattern_is_also_protected(project: Path) -> None:
    _source(project, "tests/new_area/omitted_test.py")
    with pytest.raises(checker.SelectionError, match="omitted_test.py"):
        checker.check_selection(project)


def test_declared_matrix_and_folded_node_commands_are_read(project: Path) -> None:
    path = project / checker.WORKFLOW
    path.write_text(
        """jobs:
  offline:
    strategy:
      matrix:
        include:
          - name: Unit
            pytest-args: tests/unit
    steps:
      - run: uv run --locked pytest -ra --junitxml=results.xml ${{ matrix.pytest-args }}
  supplemental:
    steps:
      - run: >
          uv run --locked pytest -ra
          tests/unit/test_selected.py::test_example
""",
        encoding="utf-8",
    )
    assert checker.check_selection(project).selected == {"tests/unit/test_selected.py"}


def test_workflow_edit_changes_selection_without_another_manifest(project: Path) -> None:
    _source(project, "tests/another/test_added.py")
    _workflow(project, "uv run --locked pytest tests/unit tests/another")
    assert "tests/another/test_added.py" in checker.check_selection(project).selected


@pytest.mark.parametrize("target", ["tests/missing", "tests/unit/test_missing.py"])
def test_missing_declared_target_is_an_error(project: Path, target: str) -> None:
    _workflow(project, f"uv run --locked pytest tests/unit {target}")
    with pytest.raises(checker.SelectionError, match="Missing or invalid declared test target"):
        checker.check_selection(project)


def test_empty_declared_directory_is_an_error(project: Path) -> None:
    (project / "tests/empty").mkdir()
    _workflow(project, "uv run --locked pytest tests/unit tests/empty")
    with pytest.raises(checker.SelectionError, match="unexpectedly empty: tests/empty"):
        checker.check_selection(project)


def test_missing_protected_root_is_an_error(tmp_path: Path) -> None:
    _workflow(tmp_path)
    with pytest.raises(checker.SelectionError, match="Missing protected test root"):
        checker.check_selection(tmp_path)


def test_empty_protected_root_is_an_error(tmp_path: Path) -> None:
    _workflow(tmp_path)
    (tmp_path / "tests").mkdir()
    with pytest.raises(checker.SelectionError, match="Protected test root is unexpectedly empty"):
        checker.check_selection(tmp_path)


@pytest.mark.parametrize("source", ["jobs: [", "", "jobs: {}", "jobs: {}\njobs: {}\n"])
def test_broken_or_empty_workflow_cannot_look_clean(project: Path, source: str) -> None:
    (project / checker.WORKFLOW).write_text(source, encoding="utf-8")
    with pytest.raises(checker.SelectionError):
        checker.check_selection(project)


def test_missing_workflow_is_an_error(project: Path) -> None:
    (project / checker.WORKFLOW).unlink()
    with pytest.raises(checker.SelectionError, match="Cannot inspect workflow/test source"):
        checker.check_selection(project)


def test_python_parse_failure_is_an_error_even_in_excluded_root(project: Path) -> None:
    _source(project, "tests/integration/test_malformed.py", "def broken(:\n")
    with pytest.raises(checker.SelectionError, match="Cannot inspect workflow/test source"):
        checker.check_selection(project)


def test_discovery_error_is_not_an_empty_success(project: Path, monkeypatch) -> None:
    def unreadable(_root, *, onerror):
        onerror(PermissionError("fixture discovery denied"))
        return iter(())

    monkeypatch.setattr(checker.os, "walk", unreadable)
    with pytest.raises(checker.SelectionError, match="fixture discovery denied"):
        checker.check_selection(project)


def test_stale_exclusion_is_an_error(project: Path) -> None:
    (project / "tests/integration/test_live.py").unlink()
    with pytest.raises(checker.SelectionError, match="unexpectedly empty: tests/integration"):
        checker.check_selection(project)


def test_blanket_integration_selection_is_rejected(project: Path) -> None:
    _workflow(project, "uv run --locked pytest tests")
    with pytest.raises(
        checker.SelectionError, match="Intentional integration exclusion was selected"
    ):
        checker.check_selection(project)


def test_node_only_selection_does_not_claim_whole_file_coverage(project: Path) -> None:
    _workflow(project, "uv run --locked pytest tests/unit/test_selected.py::test_example")
    with pytest.raises(checker.SelectionError, match="Offline test files absent"):
        checker.check_selection(project)


def test_missing_supplemental_test_node_is_an_error(project: Path) -> None:
    _workflow(project, "uv run --locked pytest tests/unit tests/unit/test_selected.py::test_absent")
    with pytest.raises(checker.SelectionError, match="Missing or unsupported declared test node"):
        checker.check_selection(project)


@pytest.mark.parametrize(
    "command",
    [
        "uv run --locked pytest",
        "uv run pytest tests/unit",
        "uv run --locked pytest --ignore=tests/unit tests/unit",
        "uv run --locked pytest tests/unit && echo success",
        "uv run --locked pytest ${{ inputs.tests }}",
        "uv run --locked pytest tests/unit 'unterminated",
        "uv run --locked pytest ../tests/unit",
    ],
)
def test_unrecognized_command_forms_fail_closed(project: Path, command: str) -> None:
    _workflow(project, command)
    with pytest.raises(checker.SelectionError):
        checker.check_selection(project)


def test_unsupported_matrix_shape_is_an_error(project: Path) -> None:
    (project / checker.WORKFLOW).write_text(
        """jobs:
  offline:
    strategy:
      matrix:
        pytest-args: [tests/unit]
    steps:
      - run: uv run --locked pytest ${{ matrix.pytest-args }}
""",
        encoding="utf-8",
    )
    with pytest.raises(checker.SelectionError, match="Only explicit matrix include"):
        checker.check_selection(project)
