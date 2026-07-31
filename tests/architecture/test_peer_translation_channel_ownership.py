from __future__ import annotations

import ast

from tests.helpers.paths import SOURCE_ROOT

OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
CALLBACKS_PATH = SOURCE_ROOT / "core" / "orchestrator" / "translation_channel_callbacks.py"
PIPELINE_PATH = SOURCE_ROOT / "app" / "wiring" / "wiring_runtime_pipeline.py"


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


def test_pipeline_constructs_and_exposes_one_direct_peer_translation_owner() -> None:
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.count("PeerTranslationChannelOwner") == 1
    assert "callbacks.bind_peer(peer_translation_channel)" in source
    assert "peer_translation_channel=peer_translation_channel" in source
    assert (
        "peer_translation_channel,\n        local_asr_runtime,\n"
        "        peer_translation_channel,"
    ) in source
    assert "ClientHub" not in source


def test_peer_owner_has_explicit_ingress_lifecycle_and_dependencies() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    owner = _class(tree, "PeerTranslationChannelOwner")
    methods = _methods(owner)
    fields = _fields(owner)

    assert {
        "handle_peer_vad_event",
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


def test_production_peer_consumers_bind_directly_without_hub_residue() -> None:
    paths = (
        SOURCE_ROOT / "app" / "wiring" / "wiring_runtime_pipeline.py",
        SOURCE_ROOT / "app" / "services" / "settings" / "settings_runtime_effects.py",
        SOURCE_ROOT / "composition" / "application_runtime.py",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "pipeline.hub" not in source
        assert "components.hub" not in source
        assert "ClientHub" not in source
        assert "core.orchestrator.hub" not in source

    assert not (SOURCE_ROOT / "core" / "orchestrator" / "hub.py").exists()
    assert not (SOURCE_ROOT / "core" / "orchestrator" / "hub_callbacks.py").exists()


def test_durable_callbacks_dispatch_peer_work_through_public_owner_contract() -> None:
    source = CALLBACKS_PATH.read_text(encoding="utf-8")

    assert "PeerTranslationChannelOwner" in source
    assert "self._require_peer().handle_stt_event(event)" in source
    assert "self._require_peer().handle_retired_stt_event(event)" in source
    assert "self._require_peer().process_child(" in source
    assert "self._require_peer().on_child_terminal(child, outcome)" in source
    assert "self._require_peer().submit_translation_output(submission)" in source
    assert "ClientHub" not in source


def test_peer_owner_preserves_explicit_chatbox_denial_attempts() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert 'self.output_projection.chatbox_is_denied("peer")' in source
    assert "self.output_projection.publish_peer_chatbox_denial(" in source
    assert 'channel="peer"' in source
    assert 'channel="self"' not in source
