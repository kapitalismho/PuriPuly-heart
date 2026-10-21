from __future__ import annotations

import pytest

from tests.helpers.task_creation import (
    ASYNCIO_CREATE_TASK,
    RUN_TASK,
    TaskIdentity,
    assert_task_ownership,
    task_creation_sites_from_source,
)

MODULE = "src/puripuly_heart/probe.py"


@pytest.mark.parametrize(
    ("source", "function"),
    [
        (
            "import asyncio\ndef fresh():\n callback = lambda task=asyncio.create_task(work()): task",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n @decorate(asyncio.create_task(work()))\n class Inner: pass",
            "fresh.Inner",
        ),
        (
            "import asyncio\ndef fresh():\n registry[asyncio.create_task(work())] = True",
            "fresh",
        ),
        ("import asyncio\ndef fresh():\n asyncio.create_task(work())", "fresh"),
        (
            "import asyncio\ndef fresh():\n callback = lambda *, task=asyncio.create_task(work()): task",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n class Inner(base(asyncio.create_task(work()))): pass",
            "fresh.Inner",
        ),
        (
            "import asyncio\ndef fresh():\n class Inner(metaclass=meta(asyncio.create_task(work()))): pass",
            "fresh.Inner",
        ),
        (
            "import asyncio\ndef fresh():\n class Inner(option=asyncio.create_task(work())): pass",
            "fresh.Inner",
        ),
        (
            "import asyncio\ndef fresh():\n asyncio.create_task(work()).attribute = True",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n registry[0], registry[asyncio.create_task(work())] = values",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n registry[asyncio.create_task(work())]: bool = True",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n for registry[asyncio.create_task(work())] in values: pass",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n with resource() as registry[asyncio.create_task(work())]: pass",
            "fresh",
        ),
        (
            "import asyncio\ndef fresh():\n items = [value for registry[asyncio.create_task(work())] in values]",
            "fresh",
        ),
    ],
    ids=[
        "review-lambda-default",
        "review-class-decorator",
        "review-assignment-target",
        "review-positive-control",
        "lambda-keyword-default",
        "class-base",
        "class-metaclass",
        "class-keyword",
        "attribute-target",
        "destructured-target",
        "annotated-target",
        "loop-target",
        "with-target",
        "comprehension-target",
    ],
)
def test_executable_expression_children_are_recognized_and_rejected(source, function) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity for site in sites] == [
        TaskIdentity(MODULE, function, ASYNCIO_CREATE_TASK)
    ]
    assert sites[0].callee == "asyncio.create_task"
    with pytest.raises(AssertionError) as failure:
        assert_task_ownership(sites, {})
    message = str(failure.value)
    assert MODULE in message
    assert function in message
    assert "asyncio.create_task" in message
    assert "unreviewed raw creation" in message
    assert "LifecycleScope/start_lifecycle_task" in message


@pytest.mark.parametrize(
    ("source", "functions"),
    [
        (
            "import asyncio\nclass Worker:\n asyncio = unrelated\n def fresh(self):\n  asyncio.create_task(work())",
            ["Worker.fresh"],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n def fresh(self):\n  spawn(work())\n",
            [],
        ),
        (
            "asyncio = unrelated\nclass Worker:\n import asyncio\n def fresh(self):\n  asyncio.create_task(work())",
            [],
        ),
        (
            "from asyncio import create_task as spawn\nclass Worker:\n def spawn(*args):\n  return None\n def fresh(self):\n  spawn(work())",
            ["Worker.fresh"],
        ),
        (
            "def factory():\n from asyncio import create_task as spawn\n class Worker:\n  spawn = unrelated\n  def fresh(self):\n   spawn(work())",
            ["factory.Worker.fresh"],
        ),
        (
            "import asyncio\nclass Outer:\n asyncio = unrelated\n class Worker:\n  def fresh(self):\n   asyncio.create_task(work())",
            ["Outer.Worker.fresh"],
        ),
        (
            "def spawn(*args):\n return None\nclass Outer:\n from asyncio import create_task as spawn\n class Worker:\n  def fresh(self):\n   spawn(work())",
            [],
        ),
        (
            "from asyncio import create_task as spawn\nclass Worker:\n def fresh(self):\n  spawn = unrelated\n  spawn(work())",
            [],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n task = spawn(work())\n def fresh(self):\n  spawn(work())",
            ["Worker"],
        ),
        (
            "import asyncio\nclass Worker:\n asyncio = unrelated\n task = asyncio.create_task(work())",
            [],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n def fresh(self, task=spawn(work())):\n  spawn(work())",
            ["Worker.fresh"],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n @decorate(spawn(work()))\n def fresh(self):\n  spawn(work())",
            ["Worker.fresh"],
        ),
        (
            "import asyncio\nclass Worker:\n asyncio = unrelated\n callback = lambda: asyncio.create_task(work())",
            ["Worker"],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n callback = lambda: spawn(work())",
            [],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n callback = lambda task=spawn(work()): spawn(work())",
            ["Worker"],
        ),
        (
            "import asyncio\nclass Worker:\n asyncio = unrelated\n tasks = [asyncio.create_task(work()) for item in values]",
            ["Worker"],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n tasks = [spawn(work()) for item in values]",
            [],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n tasks = [item for item in spawn(work())]",
            ["Worker"],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n tasks = [item for outer in values for item in spawn(work())]",
            [],
        ),
        (
            "from asyncio import create_task as spawn\nclass Worker:\n spawn = unrelated\n tasks = [item for item in spawn(work())]",
            [],
        ),
        (
            "from asyncio import create_task as spawn\nclass Worker:\n spawn = unrelated\n tasks = {item: spawn(work()) for item in values}",
            ["Worker"],
        ),
        (
            "def spawn(*args):\n return None\nclass Worker:\n from asyncio import create_task as spawn\n tasks = (spawn(work()) for item in values)",
            [],
        ),
        (
            "from asyncio import create_task as spawn\nclass Worker:\n spawn = unrelated\n tasks = {spawn(work()) for item in values}",
            ["Worker"],
        ),
    ],
    ids=[
        "review-module-asyncio-hidden-by-class",
        "review-harmless-module-spawn-class-import",
        "class-import-not-visible-in-method",
        "class-method-not-visible-in-other-method",
        "method-closes-over-enclosing-function",
        "nested-class-method-bypasses-both-class-namespaces",
        "nested-class-import-not-visible-in-method",
        "method-local-shadowing-remains-effective",
        "class-body-import-remains-visible",
        "class-body-own-shadowing-remains-effective",
        "method-default-uses-definition-scope",
        "method-decorator-uses-definition-scope",
        "lambda-body-bypasses-class-shadowing",
        "lambda-body-does-not-use-class-import",
        "lambda-default-uses-class-import-body-does-not",
        "comprehension-body-bypasses-class-shadowing",
        "comprehension-body-does-not-use-class-import",
        "outermost-iterable-uses-definition-scope",
        "subsequent-iterable-bypasses-class-import",
        "outermost-iterable-respects-class-shadowing",
        "dict-comprehension-body-bypasses-class-shadowing",
        "generator-body-does-not-use-class-import",
        "set-comprehension-body-bypasses-class-shadowing",
    ],
)
def test_class_definition_and_lexical_body_scopes_remain_distinct(source, functions) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity.function for site in sites] == functions
    assert all(site.identity.module == MODULE for site in sites)
    assert all(site.identity.family == ASYNCIO_CREATE_TASK for site in sites)
    if functions:
        with pytest.raises(AssertionError, match="unreviewed raw creation"):
            assert_task_ownership(sites, {})
    else:
        assert assert_task_ownership(sites, {}) == []


@pytest.mark.parametrize(
    "source",
    [
        """import flet as ft
class Worker(ft.Column):
 ft = unrelated
 def fresh(self):
  self.page.run_task(work)
""",
        """import asyncio
import flet as ft
class Worker:
 asyncio = unrelated
 def __init__(self, page: ft.Page):
  self._page = page
 def fresh(self):
  self._page.run_task(work)
""",
    ],
    ids=["flet-base-resolved-in-definition-scope", "typed-constructor-instance-attribute"],
)
def test_class_namespace_bypass_preserves_explicit_page_attribute_inference(source) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity for site in sites] == [TaskIdentity(MODULE, "Worker.fresh", RUN_TASK)]
    with pytest.raises(AssertionError, match="Worker.fresh"):
        assert_task_ownership(sites, {})
