from __future__ import annotations

import tomllib

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


def test_benchmark_stays_outside_production_package_discovery() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = project["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["packages"]["find"]["where"] == ["src"]
    assert not (REPO_ROOT / "src/puripuly_heart/release_evidence/agent_chat_simulator").exists()
