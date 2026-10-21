from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.architecture.test_lifecycle_task_guard import REVIEWED_TASK_CREATION
from tests.helpers.ast_sources import SourceScanError
from tests.helpers.task_creation import (
    ASYNCIO_CREATE_TASK,
    ASYNCIO_ENSURE_FUTURE,
    ASYNCIO_TASK,
    BARE_RUN_TASK,
    LOOP_CREATE_TASK,
    RUN_TASK,
    TaskDisposition,
    TaskIdentity,
    assert_task_ownership,
    scan_task_creation_sites,
    task_creation_sites_from_source,
)

MODULE = "src/puripuly_heart/fixture.py"


def _review(function: str, maximum: int = 1) -> dict[TaskIdentity, TaskDisposition]:
    return {
        TaskIdentity(MODULE, function, ASYNCIO_CREATE_TASK): TaskDisposition(
            "legacy_exception",
            maximum,
            "Fixture local caller retains and awaits bounded work; removal is permitted",
        )
    }


@pytest.mark.parametrize(
    ("module", "source", "classification"),
    [
        (
            "src/puripuly_heart/core/lifecycle.py",
            "import asyncio\nclass LifecycleScope:\n def create_task(self):\n  asyncio.create_task(work())\n",
            "owner_primitive",
        ),
        (
            "src/puripuly_heart/core/runtime/clipboard.py",
            "import asyncio\nclass ClipboardRuntime:\n def submit_from_loop(self):\n  asyncio.create_task(work())\n",
            "named_owner",
        ),
        (
            "src/puripuly_heart/core/stt/scoped_engine.py",
            "import asyncio\nclass ScopedRecognitionEngine:\n def _normalization_diagnostic(self):\n  asyncio.create_task(work())\n",
            "legacy_exception",
        ),
    ],
)
def test_exact_reviewed_identity_has_intended_classification(
    module, source, classification
) -> None:
    sites = task_creation_sites_from_source(source, module)
    result = assert_task_ownership(sites, REVIEWED_TASK_CREATION)
    assert len(result) == 1
    assert result[0][1] == classification


@pytest.mark.parametrize(
    ("source", "family", "callee"),
    [
        (
            "from asyncio import create_task as spawn\ndef fresh():\n schedule = spawn\n schedule(work())",
            ASYNCIO_CREATE_TASK,
            "schedule",
        ),
        (
            "import asyncio\ndef fresh():\n schedule = asyncio.create_task\n schedule(work())",
            ASYNCIO_CREATE_TASK,
            "schedule",
        ),
        (
            "import flet as ft\nfrom typing import Optional\ndef fresh(page: Optional[ft.Page]):\n page.run_task(work)",
            RUN_TASK,
            "page.run_task",
        ),
        (
            "import asyncio\ndef fresh(loop: 'asyncio.AbstractEventLoop | None'):\n loop.create_task(work())",
            LOOP_CREATE_TASK,
            "loop.create_task",
        ),
        (
            "import asyncio as aio\ndef fresh():\n aio.create_task(work())",
            ASYNCIO_CREATE_TASK,
            "aio.create_task",
        ),
        (
            "from asyncio import create_task as spawn\ndef fresh():\n spawn(work())",
            ASYNCIO_CREATE_TASK,
            "spawn",
        ),
        (
            "from asyncio import create_task\ndef fresh():\n create_task(work())",
            ASYNCIO_CREATE_TASK,
            "create_task",
        ),
        (
            "from asyncio import get_running_loop as running\ndef fresh():\n loop = running()\n loop.create_task(work())",
            LOOP_CREATE_TASK,
            "loop.create_task",
        ),
        (
            "import asyncio as aio\ndef fresh():\n aio.get_event_loop().create_task(work())",
            LOOP_CREATE_TASK,
            "aio.get_event_loop().create_task",
        ),
        (
            "import asyncio\ndef fresh(loop: asyncio.AbstractEventLoop):\n loop.create_task(work())",
            LOOP_CREATE_TASK,
            "loop.create_task",
        ),
        ("from asyncio import Task as Spawn\ndef fresh():\n Spawn(work())", ASYNCIO_TASK, "Spawn"),
        (
            "from asyncio import ensure_future as spawn\ndef fresh():\n spawn(work())",
            ASYNCIO_ENSURE_FUTURE,
            "spawn",
        ),
        (
            "import flet as ft\ndef fresh(page: ft.Page):\n page.run_task(work)",
            RUN_TASK,
            "page.run_task",
        ),
        (
            "import flet as ft\ndef fresh(page: ft.Page):\n runner = getattr(page, 'run_task', None)\n runner(work)",
            BARE_RUN_TASK,
            "runner",
        ),
        (
            "from typing import Callable\ndef fresh(run_task: Callable):\n run_task(work)",
            BARE_RUN_TASK,
            "run_task",
        ),
        (
            "from puripuly_heart.ui.flet_runtime import control_page as page_of\ndef fresh(control):\n page = page_of(control)\n page.run_task(work)",
            RUN_TASK,
            "page.run_task",
        ),
        (
            "import flet as ft\nclass View(ft.Column):\n def fresh(self):\n  page = getattr(self, 'page', None)\n  page.run_task(work)",
            RUN_TASK,
            "page.run_task",
        ),
        (
            "import flet as ft\nclass View:\n def __init__(self, page: ft.Page):\n  self._page = page\n def fresh(self):\n  self._page.run_task(work)",
            RUN_TASK,
            "self._page.run_task",
        ),
    ],
)
def test_recognized_alias_or_binding_is_rejected_in_unreviewed_function(
    source, family, callee
) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert len(sites) == 1
    assert sites[0].identity.family == family
    assert sites[0].callee == callee
    assert sites[0].identity.function in {"fresh", "View.fresh"}
    with pytest.raises(AssertionError) as failure:
        assert_task_ownership(sites, {})
    message = str(failure.value)
    assert MODULE in message
    assert sites[0].identity.function in message
    assert family in message
    assert callee in message
    assert "LifecycleScope/start_lifecycle_task" in message


@pytest.mark.parametrize(
    "source",
    [
        "def fresh(service):\n service.run_task(work)",
        "def fresh(run_task):\n run_task(work)",
        "def fresh(service):\n runner = getattr(service, 'run_task', None)\n runner(work)",
        "def fresh(loop):\n loop.create_task(work())",
        "import asyncio\ndef fresh(asyncio):\n asyncio.create_task(work())",
        "from asyncio import create_task as spawn\ndef fresh(spawn):\n spawn(work())",
        "from asyncio import create_task as spawn\ndef fresh():\n spawn(work())\n spawn = unrelated",
        "from asyncio import create_task as spawn\ndef fresh():\n spawn = unrelated\n spawn(work())",
        "from asyncio import create_task as spawn\ndef fresh():\n for spawn in services:\n  spawn(work())",
        "from asyncio import create_task as spawn\ndef fresh():\n with service as spawn:\n  spawn(work())",
        "from asyncio import create_task as spawn\ndef fresh():\n [spawn(work()) for spawn in services]",
        "from asyncio import create_task as spawn\ndef fresh():\n callback = lambda spawn: spawn(work())",
        "from asyncio import create_task as spawn\ndef fresh():\n try:\n  service()\n except Exception as spawn:\n  spawn(work())",
        "import flet as ft\ndef fresh(pages: list[ft.Page]):\n pages.run_task(work)",
        "import flet as ft\nfrom typing import Callable\ndef fresh(callback: Callable[[], ft.Page]):\n callback.run_task(work)",
        "from typing import Callable\ndef fresh(run_task: list[Callable]):\n run_task(work)",
    ],
)
def test_unrelated_suffixes_and_lexical_shadowing_are_not_task_bindings(source) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert sites == []
    assert assert_task_ownership(sites, {}) == []


def test_nested_function_and_comprehension_scope_preserve_real_outer_alias() -> None:
    source = """from asyncio import create_task as spawn
class View:
 def approved(self):
  [spawn(work()) for spawn in services]
  spawn(work())
  def fresh():
   spawn(work())
"""
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity.function for site in sites] == ["View.approved", "View.approved.fresh"]
    with pytest.raises(AssertionError, match="View.approved.fresh"):
        assert_task_ownership(sites, _review("View.approved"))


def test_same_count_substitution_rejects_new_identity() -> None:
    old = task_creation_sites_from_source(
        "import asyncio\ndef approved():\n asyncio.create_task(work())", MODULE
    )
    new = task_creation_sites_from_source(
        "from asyncio import create_task as spawn\ndef fresh():\n spawn(work())", MODULE
    )
    assert len(old) == len(new) == 1
    assert_task_ownership(old, _review("approved"))
    with pytest.raises(AssertionError, match="fresh.*unreviewed raw creation"):
        assert_task_ownership(new, _review("approved"))


def test_same_identity_growth_is_rejected_and_removal_is_accepted() -> None:
    source = "import asyncio\ndef approved():\n asyncio.create_task(first())\n asyncio.create_task(second())"
    sites = task_creation_sites_from_source(source, MODULE)
    with pytest.raises(AssertionError, match="multiplicity 2 exceeds reviewed upper bound 1"):
        assert_task_ownership(sites, _review("approved"))
    assert len(assert_task_ownership(sites, _review("approved", 2))) == 2
    removed = task_creation_sites_from_source("def approved():\n return None", MODULE)
    assert assert_task_ownership(removed, _review("approved")) == []


@pytest.mark.parametrize(
    ("module", "owner", "page_expression"),
    [
        ("src/puripuly_heart/ui/desktop_overlay.py", "FletDesktopRendererWindow", "self._page"),
        (
            "src/puripuly_heart/ui/presentation_adapter.py",
            "FletUiPresentationAdapter",
            "getattr(self._app, 'page', None)",
        ),
    ],
)
def test_exact_known_injected_binding_catches_a_new_function(
    module, owner, page_expression
) -> None:
    source = (
        f"class {owner}:\n def fresh(self):\n"
        f"  page = {page_expression}\n  runner = getattr(page, 'run_task', None)\n  runner(work)\n"
    )
    sites = task_creation_sites_from_source(source, module)
    assert len(sites) == 1
    assert sites[0].identity.family == BARE_RUN_TASK
    with pytest.raises(AssertionError, match=f"{owner}.fresh"):
        assert_task_ownership(sites, REVIEWED_TASK_CREATION)
    assert task_creation_sites_from_source(source, MODULE) == []


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio\ndef fresh(task=asyncio.create_task(work())):\n pass",
        "from asyncio import create_task as spawn\ndef fresh(task=spawn(work())):\n spawn = unrelated",
    ],
)
def test_definition_default_creation_is_not_omitted(source) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert len(sites) == 1
    with pytest.raises(AssertionError, match="fresh"):
        assert_task_ownership(sites, {})


@pytest.mark.parametrize(
    "disposition",
    [
        TaskDisposition("raw_rejected", 1, "Not an owning disposition"),
        TaskDisposition("legacy_exception", 0, "No lawful upper bound"),
        TaskDisposition("named_owner", 1, ""),
    ],
)
def test_invalid_review_metadata_cannot_accept_source(disposition) -> None:
    source = "import asyncio\ndef approved():\n asyncio.create_task(work())"
    sites = task_creation_sites_from_source(source, MODULE)
    with pytest.raises(AssertionError, match="task disposition|task bound or lifetime rationale"):
        assert_task_ownership(sites, {sites[0].identity: disposition})


@pytest.mark.parametrize("source", ["def broken(:", "\x00", None])
def test_parser_inputs_fail_closed(source) -> None:
    with pytest.raises(SourceScanError, match=MODULE):
        task_creation_sites_from_source(source, MODULE)


def test_discovery_rejects_undeclared_missing_and_empty_roots(tmp_path) -> None:
    with pytest.raises(SourceScanError, match="No protected source roots"):
        scan_task_creation_sites(tmp_path, ())
    with pytest.raises(SourceScanError, match="Missing protected source root"):
        scan_task_creation_sites(tmp_path, ("protected",))
    (tmp_path / "protected").mkdir()
    with pytest.raises(SourceScanError, match="Empty protected source root"):
        scan_task_creation_sites(tmp_path, ("protected",))


def test_actual_discovery_and_rule_accept_valid_source_but_reject_new_file(tmp_path) -> None:
    protected = tmp_path / "src" / "puripuly_heart"
    protected.mkdir(parents=True)
    fixture = protected / "fixture.py"
    fixture.write_text(
        "import asyncio\ndef approved():\n asyncio.create_task(work())", encoding="utf-8"
    )
    roots = ("src/puripuly_heart",)
    sites = scan_task_creation_sites(tmp_path, roots)
    assert len(assert_task_ownership(sites, _review("approved"))) == 1
    (protected / "new.py").write_text(
        "import asyncio\ndef fresh():\n asyncio.create_task(work())", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="new.py.*fresh"):
        assert_task_ownership(scan_task_creation_sites(tmp_path, roots), _review("approved"))


@pytest.mark.parametrize("content", [b"def broken(:", b"\xff"])
def test_discovered_parse_and_encoding_failures_are_checker_errors(tmp_path, content) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "bad.py").write_bytes(content)
    with pytest.raises(SourceScanError, match="protected/bad.py"):
        scan_task_creation_sites(tmp_path, ("protected",))


def test_discovery_error_is_not_a_clean_result(tmp_path, monkeypatch) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()

    def denied_walk(root, *, onerror):
        onerror(PermissionError("synthetic discovery denial"))
        return iter(())

    monkeypatch.setattr(os, "walk", denied_walk)
    with pytest.raises(SourceScanError, match="Cannot discover protected source protected"):
        scan_task_creation_sites(tmp_path, ("protected",))


def test_discovered_read_error_is_not_a_clean_result(tmp_path, monkeypatch) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    fixture = protected / "blocked.py"
    fixture.write_text("pass", encoding="utf-8")
    original = Path.read_text

    def denied_read(path, *args, **kwargs):
        if path == fixture:
            raise PermissionError("synthetic read denial")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_read)
    with pytest.raises(SourceScanError, match="Cannot read protected source protected/blocked.py"):
        scan_task_creation_sites(tmp_path, ("protected",))
