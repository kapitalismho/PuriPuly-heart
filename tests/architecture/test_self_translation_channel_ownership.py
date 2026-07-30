from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "self_translation_channel.py"
PEER_OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
CALLBACKS_PATH = SOURCE_ROOT / "core" / "orchestrator" / "translation_channel_callbacks.py"
PIPELINE_PATH = SOURCE_ROOT / "app" / "wiring_runtime_pipeline.py"


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


def test_pipeline_constructs_and_exposes_one_direct_self_translation_owner() -> None:
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.count("SelfTranslationChannelOwner") == 1
    assert "callbacks.bind_self(self_translation_channel)" in source
    assert "callbacks.bind_peer(peer_translation_channel)" in source
    assert "self_translation_channel=self_translation_channel" in source
    assert (
        "self_translation_channel,\n        local_asr_runtime,\n        self_translation_channel,"
        in source
    )


def test_self_translation_owner_has_explicit_ingress_lifecycle_and_dependencies() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    owner = _class(tree, "SelfTranslationChannelOwner")
    methods = _methods(owner)
    fields = _fields(owner)

    assert {
        "handle_vad_event",
        "handle_stt_event",
        "submit_text",
        "reset_provider_channel",
        "clear_language_runtime_state",
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
    assert {"llm", "stt", "peer_stt", "output_runtime"}.isdisjoint(fields)


def test_peer_owner_has_no_self_translation_algorithms_state_or_runtime_reference() -> None:
    source = PEER_OWNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = _class(tree, "PeerTranslationChannelOwner")
    methods = _methods(owner)
    fields = _fields(owner)

    assert {
        "handle_vad_event",
        "submit_text",
        "mark_promo_eligible",
        "_send_stt_connected_notification",
        "_handle_low_latency_final",
        "_sync_overlay_active_self",
        "_commit_merge",
        "_run_spec_translation",
    }.isdisjoint(methods)
    assert {
        "direct_self_runtime",
        "self_runtime",
        "_utterances",
        "_translation_tasks",
        "_utterance_sources",
        "_utterance_start_times",
        "_translation_history",
        "_speech_ended_ids",
        "_merge_buffer",
        "_stt_task",
        "_last_promo_time",
        "_promo_eligible",
        "active_chatbox_channel",
    }.isdisjoint(fields)
    assert "self.self_runtime" not in source
    assert "direct_self_runtime" not in source


def test_production_self_consumers_do_not_route_through_hub() -> None:
    application_runtime_path = SOURCE_ROOT / "composition" / "application_runtime.py"
    paths = (
        SOURCE_ROOT / "app" / "adapters" / "ui_runtime.py",
        SOURCE_ROOT / "app" / "services" / "settings_runtime_effects.py",
        application_runtime_path,
    )
    forbidden = (
        "pipeline.hub.submit_text",
        "pipeline.hub.handle_vad_event",
        "pipeline.hub.mark_promo_eligible",
        "pipeline.hub.self_runtime",
        "context_provider=lambda: pipeline.hub",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        for residue in forbidden:
            assert residue not in source, f"{path.relative_to(REPO_ROOT)}: {residue}"
    application_runtime_source = application_runtime_path.read_text(encoding="utf-8")
    assert "cast(object, pipeline.hub)" not in application_runtime_source
    assert "pipeline.hub" not in application_runtime_source


def test_durable_callbacks_dispatch_self_and_peer_to_distinct_owners() -> None:
    source = CALLBACKS_PATH.read_text(encoding="utf-8")

    assert "self._require_self().handle_stt_event(event)" in source
    assert "self._require_peer().handle_stt_event(event)" in source
    assert "self._require_self().process_child(" in source
    assert "self._require_peer().process_child(" in source
    assert "self._require_self().submit_translation_output(submission)" in source
    assert "self._require_peer().submit_translation_output(submission)" in source
