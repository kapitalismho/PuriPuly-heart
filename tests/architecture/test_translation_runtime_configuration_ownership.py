from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from tests.helpers.ast_sources import assert_no_forbidden_imports
from tests.helpers.paths import SOURCE_ROOT

CONFIG_FIELD_NAMES = {field.name for field in fields(TranslationRuntimeConfig)}


def test_channel_owners_and_context_resolver_do_not_store_configuration_fields() -> None:
    assert not {field.name for field in fields(PeerTranslationChannelOwner)} & CONFIG_FIELD_NAMES
    assert not {field.name for field in fields(ContextResolver)} & CONFIG_FIELD_NAMES


def test_peer_algorithms_read_explicit_configuration_snapshots() -> None:
    path = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    direct_reads = [
        (node.lineno, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in CONFIG_FIELD_NAMES
    ]

    assert direct_reads == []


def test_peer_owner_consumes_snapshot_port_without_configuration_mutation() -> None:
    path = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PeerTranslationChannelOwner"
    )
    methods = {
        node.name for node in owner.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    fields = {
        node.target.id
        for node in owner.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "config_snapshot" in fields
    assert "translation_runtime_configuration" not in fields
    assert {"__getattribute__", "__setattr__"}.isdisjoint(methods)


def test_each_application_configuration_mutator_performs_one_owner_transform() -> None:
    path = SOURCE_ROOT / "app" / "wiring" / "wiring_translation_runtime_configuration.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    expected = {
        "replace_translation_runtime_settings",
        "replace_translation_runtime_effective_flags",
        "replace_translation_runtime_enabled",
    }
    operation_counts = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in expected:
            continue
        operation_counts[node.name] = {
            operation: sum(
                1
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == operation
            )
            for operation in ("snapshot", "replace", "transform")
        }

    assert operation_counts == {
        name: {
            "snapshot": 0,
            "replace": 0,
            "transform": 1,
        }
        for name in expected
    }


def test_configuration_boundary_has_no_ui_dependency() -> None:
    for relative_path in (
        Path("core/orchestrator/configuration.py"),
        Path("core/orchestrator/context.py"),
    ):
        assert_no_forbidden_imports(SOURCE_ROOT / relative_path, ("flet", "puripuly_heart.ui"))


def test_configuration_mutation_boundary_consumes_only_projected_settings_values() -> None:
    path = SOURCE_ROOT / "app" / "wiring" / "wiring_translation_runtime_configuration.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "puripuly_heart.config.settings" not in imports
    assert "TranslationRuntimeSettingsValues" in {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
