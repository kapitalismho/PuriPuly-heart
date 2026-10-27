from __future__ import annotations

import ast

from tests.helpers.paths import SOURCE_ROOT

OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
CALLBACKS_PATH = SOURCE_ROOT / "core" / "orchestrator" / "translation_channel_callbacks.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _methods(node: ast.ClassDef) -> set[str]:
    return {
        child.name
        for child in node.body
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _fields(node: ast.ClassDef) -> set[str]:
    return {
        child.target.id
        for child in node.body
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
    }


def test_peer_owner_has_explicit_ingress_lifecycle_and_dependencies() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    owner = _class(tree, "PeerTranslationChannelOwner")
    methods = _methods(owner)
    fields = _fields(owner)

    assert {
        "handle_peer_owned_vad_event",
        "handle_stt_event",
        "handle_retired_stt_event",
        "handle_stt_event_loop_exception",
        "reset_provider_channel",
        "clear_language_runtime_state",
        "on_child_created",
        "process_child",
        "on_child_started",
        "on_child_terminal",
        "on_parent_closed",
        "on_parent_rejected",
        "submit_translation_output",
        "open_ingress",
        "close_ingress",
        "close",
    }.issubset(methods)
    assert {
        "runtime",
        "config_snapshot",
        "translation_turns",
        "local_asr_runtime",
        "translation_requests",
        "output_projection",
        "diagnostics",
        "clock",
    }.issubset(fields)
    assert {
        "translation_runtime_configuration",
        "llm",
        "stt",
        "peer_stt",
        "output_runtime",
        "_peer_stt_task",
    }.isdisjoint(fields)


def test_durable_callbacks_dispatch_peer_work_through_public_owner_contract() -> None:
    source = CALLBACKS_PATH.read_text(encoding="utf-8")

    assert "PeerTranslationChannelOwner" in source
    assert "self._require_peer().handle_stt_event(event)" in source
    assert "self._require_peer().handle_retired_stt_event(event)" in source
    assert "self._require_peer().process_child(" in source
    assert "self._require_peer().on_child_terminal(child, outcome)" in source
    assert "self._require_peer().submit_translation_output(submission)" in source
    assert "ClientHub" not in source
