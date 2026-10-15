from __future__ import annotations

import pytest

from tests.helpers.task_creation import (
    ASYNCIO_CREATE_TASK,
    LOOP_CREATE_TASK,
    TaskIdentity,
    assert_task_ownership,
    task_creation_sites_from_source,
)

MODULE = "src/puripuly_heart/probe.py"


def test_returned_callback_resolves_later_enclosing_assigned_alias() -> None:
    source = """import asyncio
def factory():
 def inner():
  spawn(job())
 spawn = asyncio.create_task
 return inner
"""
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity for site in sites] == [
        TaskIdentity(MODULE, "factory.inner", ASYNCIO_CREATE_TASK)
    ]
    with pytest.raises(AssertionError, match="factory.inner.*unreviewed raw creation"):
        assert_task_ownership(sites, {})


@pytest.mark.parametrize(
    ("source", "function", "family"),
    [
        (
            "import asyncio\ndef factory():\n callback = lambda: spawn(job())\n spawn = asyncio.create_task\n return callback",
            "factory",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n async def inner():\n  spawn(job())\n spawn = asyncio.create_task\n return inner",
            "factory.inner",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n def middle():\n  def inner():\n   spawn(job())\n  return inner\n spawn = asyncio.create_task\n return middle",
            "factory.middle.inner",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n def middle():\n  return lambda: spawn(job())\n spawn = asyncio.create_task\n return middle",
            "factory.middle",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n def middle():\n  def inner():\n   spawn(job())\n  spawn = asyncio.create_task\n  return inner\n return middle",
            "factory.middle.inner",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "def factory():\n def inner():\n  spawn(job())\n from asyncio import create_task as spawn\n return inner",
            "factory.inner",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n def inner():\n  loop.create_task(job())\n loop = asyncio.get_running_loop()\n return inner",
            "factory.inner",
            LOOP_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n class Worker:\n  spawn = unrelated\n  def inner(self):\n   spawn(job())\n spawn = asyncio.create_task\n return Worker",
            "factory.Worker.inner",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "import asyncio\ndef factory():\n callbacks = [lambda: spawn(job()) for item in values]\n spawn = asyncio.create_task\n return callbacks",
            "factory",
            ASYNCIO_CREATE_TASK,
        ),
        (
            "def factory():\n def inner():\n  spawn(job())\n return inner\nimport asyncio\nspawn = asyncio.create_task",
            "factory.inner",
            ASYNCIO_CREATE_TASK,
        ),
    ],
    ids=[
        "lambda-body",
        "async-callback-body",
        "multiple-enclosing-functions",
        "lambda-with-multiple-enclosing-functions",
        "alias-in-nearest-enclosing-function",
        "later-imported-alias",
        "later-loop-binding",
        "class-method-captures-enclosing-function-not-class",
        "lambda-inside-comprehension",
        "later-module-binding",
    ],
)
def test_completed_enclosing_lexical_bindings_expose_unreviewed_creation(
    source, function, family
) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity for site in sites] == [TaskIdentity(MODULE, function, family)]
    with pytest.raises(AssertionError) as failure:
        assert_task_ownership(sites, {})
    message = str(failure.value)
    assert MODULE in message
    assert function in message
    assert family in message
    assert "LifecycleScope/start_lifecycle_task" in message


@pytest.mark.parametrize(
    "source",
    [
        "import asyncio\ndef factory():\n spawn = asyncio.create_task\n def inner():\n  spawn(job())\n spawn = unrelated\n return inner",
        "import asyncio\ndef factory():\n spawn = asyncio.create_task\n callback = lambda: spawn(job())\n spawn = unrelated\n return callback",
        "import asyncio\ndef factory():\n def inner(spawn):\n  spawn(job())\n spawn = asyncio.create_task\n return inner",
        "import asyncio\ndef factory():\n callback = lambda spawn: spawn(job())\n spawn = asyncio.create_task\n return callback",
        "import asyncio\ndef factory():\n def inner():\n  spawn(job())\n  spawn = unrelated\n spawn = asyncio.create_task\n return inner",
        "import asyncio\ndef factory():\n def middle():\n  def inner():\n   spawn(job())\n  spawn = unrelated\n  return inner\n spawn = asyncio.create_task\n return middle",
        "import asyncio\ndef factory():\n callbacks = [lambda: spawn(job()) for spawn in values]\n spawn = asyncio.create_task\n return callbacks",
        "import asyncio\ndef factory():\n def inner():\n  spawn(job())\n spawn = asyncio.create_task\n spawn = unrelated\n return inner",
    ],
    ids=[
        "callback-unrelated-final-reassignment",
        "lambda-unrelated-final-reassignment",
        "callback-parameter-shadow",
        "lambda-parameter-shadow",
        "callback-local-shadow-before-assignment",
        "nearest-enclosing-function-shadows-outer-alias",
        "comprehension-target-shadows-outer-alias",
        "later-alias-then-unrelated-final-reassignment",
    ],
)
def test_unrelated_reassignment_and_lexical_shadowing_do_not_create_false_positives(
    source,
) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert sites == []
    assert assert_task_ownership(sites, {}) == []


@pytest.mark.parametrize(
    ("source", "function", "line"),
    [
        (
            "import asyncio\ndef factory():\n spawn = unrelated\n def inner(task=spawn(job())):\n  spawn(job())\n spawn = asyncio.create_task\n return inner",
            "factory.inner",
            5,
        ),
        (
            "import asyncio\ndef factory():\n spawn = asyncio.create_task\n def inner(task=spawn(job())):\n  spawn(job())\n spawn = unrelated\n return inner",
            "factory.inner",
            4,
        ),
        (
            "import asyncio\ndef factory():\n spawn = unrelated\n @decorate(spawn(job()))\n def inner():\n  spawn(job())\n spawn = asyncio.create_task\n return inner",
            "factory.inner",
            6,
        ),
        (
            "import asyncio\ndef factory():\n spawn = asyncio.create_task\n @decorate(spawn(job()))\n def inner():\n  spawn(job())\n spawn = unrelated\n return inner",
            "factory.inner",
            4,
        ),
        (
            "import asyncio\ndef factory():\n spawn = unrelated\n callback = (\n  lambda task=spawn(job()):\n  spawn(job())\n )\n spawn = asyncio.create_task\n return callback",
            "factory",
            6,
        ),
        (
            "import asyncio\ndef factory():\n spawn = asyncio.create_task\n callback = (\n  lambda task=spawn(job()):\n  spawn(job())\n )\n spawn = unrelated\n return callback",
            "factory",
            5,
        ),
    ],
    ids=[
        "callback-default-eager-body-late-bound",
        "callback-default-eager-before-unrelated-reassignment",
        "callback-decorator-eager-body-late-bound",
        "callback-decorator-eager-before-unrelated-reassignment",
        "lambda-default-eager-body-late-bound",
        "lambda-default-eager-before-unrelated-reassignment",
    ],
)
def test_defaults_and_decorators_keep_definition_order_while_bodies_are_late_bound(
    source, function, line
) -> None:
    sites = task_creation_sites_from_source(source, MODULE)
    assert [site.identity for site in sites] == [
        TaskIdentity(MODULE, function, ASYNCIO_CREATE_TASK)
    ]
    assert sites[0].line == line
    with pytest.raises(AssertionError, match="unreviewed raw creation"):
        assert_task_ownership(sites, {})
