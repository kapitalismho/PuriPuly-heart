from __future__ import annotations

import py_compile
from importlib.machinery import PathFinder
from importlib.util import cache_from_source
from pathlib import Path
from pkgutil import iter_modules

import pytest
from setuptools.config.pyprojecttoml import read_configuration
from setuptools.discovery import find_package_path

from tests.helpers.ast_sources import assert_no_forbidden_imports
from tests.helpers.paths import REPO_ROOT
from tests.helpers.task_creation import (
    ASYNCIO_CREATE_TASK,
    TaskDisposition,
    TaskIdentity,
    assert_task_ownership,
    scan_task_creation_sites,
)


def test_chatbot_tasks_keep_reviewed_bounded_ownership() -> None:
    sites = scan_task_creation_sites(REPO_ROOT, ("benchmarks/chatbot",))
    assert sites
    assert_task_ownership(
        sites,
        {
            TaskIdentity(
                "benchmarks/chatbot/execution.py", "AttemptScheduler.run", ASYNCIO_CREATE_TASK
            ): TaskDisposition(
                "named_owner",
                1,
                "AttemptScheduler retains bounded per-attempt tasks, stops admission on failure/cancellation, "
                "and shield-drains each attempt's model close and checkpoint before returning",
            )
        },
    )


def test_production_sources_do_not_import_benchmarks() -> None:
    for path in (REPO_ROOT / "src").rglob("*.py"):
        assert_no_forbidden_imports(path, ("benchmarks",))


def _assert_benchmark_not_packaged(root: Path) -> None:
    configuration = read_configuration(root / "pyproject.toml")["tool"]["setuptools"]
    names = [*(configuration.get("packages") or ()), *(configuration.get("py-modules") or ())]
    assert names
    for name in names:
        path = Path(find_package_path(name, configuration.get("package-dir", {}), root))
        assert not (
            name == "benchmarks"
            or name.startswith("benchmarks.")
            or path.resolve().is_relative_to((root / "benchmarks").resolve())
        ), f"Benchmark included in production packaging: {name}"


def _assert_retired_simulator_not_importable(root: Path) -> None:
    parent = root / "src/puripuly_heart/release_evidence"
    name = "puripuly_heart.release_evidence.agent_chat_simulator"
    spec = PathFinder.find_spec(name, [str(parent)])
    if spec is None:
        return
    assert spec.loader is None, f"Retired simulator remains importable: {name}"
    for location in spec.submodule_search_locations or ():
        package = Path(location)
        for directory in (package, *package.rglob("*")):
            if directory.is_dir() and "__pycache__" not in directory.relative_to(package).parts:
                modules = list(iter_modules([str(directory)]))
                assert not modules, f"Retired simulator modules remain importable: {modules}"


def test_benchmark_stays_outside_production_package_discovery() -> None:
    _assert_benchmark_not_packaged(REPO_ROOT)


def test_retired_simulator_has_no_importable_implementation_or_entrypoint() -> None:
    _assert_retired_simulator_not_importable(REPO_ROOT)


@pytest.fixture
def synthetic_project(tmp_path: Path) -> Path:
    for package in ("src/puripuly_heart", "benchmarks/chatbot"):
        directory = tmp_path / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "synthetic-boundary"\nversion = "0.0.0"\n'
        '[tool.setuptools]\npackage-dir = {"" = "src"}\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("cached", [False, True], ids=["empty-namespace", "ignored-bytecode"])
def test_retired_boundary_accepts_inert_namespace_residue(
    synthetic_project: Path, cached: bool
) -> None:
    retired = synthetic_project / "src/puripuly_heart/release_evidence/agent_chat_simulator"
    retired.mkdir(parents=True)
    if cached:
        for name in ("__init__", "__main__", "runner"):
            source = retired / f"{name}.py"
            source.write_text(
                'raise AssertionError("Retired code must not execute")\n', encoding="utf-8"
            )
            py_compile.compile(source, cfile=cache_from_source(str(source)), doraise=True)
            source.unlink()
    _assert_retired_simulator_not_importable(synthetic_project)
    _assert_benchmark_not_packaged(synthetic_project)


@pytest.mark.parametrize(
    "relative_path",
    [
        "agent_chat_simulator.py",
        "agent_chat_simulator/__init__.py",
        "agent_chat_simulator/__main__.py",
        "agent_chat_simulator/runner.py",
        "agent_chat_simulator/__main__.pyc",
        "agent_chat_simulator/nested/runner.py",
    ],
)
def test_retired_boundary_rejects_importable_code(
    synthetic_project: Path, relative_path: str
) -> None:
    target = synthetic_project / "src/puripuly_heart/release_evidence" / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    source = target.with_suffix(".py")
    source.write_text('raise AssertionError("Retired code must not execute")\n', encoding="utf-8")
    if target.suffix == ".pyc":
        py_compile.compile(source, cfile=target, doraise=True)
        source.unlink()
    with pytest.raises(AssertionError, match="Retired simulator.*importable"):
        _assert_retired_simulator_not_importable(synthetic_project)


@pytest.mark.parametrize("alias", [False, True], ids=["root-discovery", "renamed-package"])
def test_packaging_boundary_rejects_benchmark_inclusion(
    synthetic_project: Path, alias: bool
) -> None:
    configuration = (
        '[tool.setuptools]\npackages = ["hidden_chatbot"]\n'
        'package-dir = {"hidden_chatbot" = "benchmarks/chatbot"}\n'
        if alias
        else '[tool.setuptools.packages.find]\nwhere = ["."]\n'
    )
    (synthetic_project / "pyproject.toml").write_text(
        '[project]\nname = "synthetic-boundary"\nversion = "0.0.0"\n' + configuration,
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="Benchmark included in production packaging"):
        _assert_benchmark_not_packaged(synthetic_project)
