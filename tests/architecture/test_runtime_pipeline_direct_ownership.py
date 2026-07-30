from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
PIPELINE_PATH = SOURCE_ROOT / "app" / "wiring_runtime_pipeline.py"
HUB_PATH = SOURCE_ROOT / "core" / "orchestrator" / "hub.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_pipeline_constructs_direct_durable_owners_and_hub_constructs_none() -> None:
    pipeline_tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    hub_tree = ast.parse(HUB_PATH.read_text(encoding="utf-8"))
    pipeline_calls = [
        _call_name(node) for node in ast.walk(pipeline_tree) if isinstance(node, ast.Call)
    ]
    hub_calls = {_call_name(node) for node in ast.walk(hub_tree) if isinstance(node, ast.Call)}

    assert pipeline_calls.count("OutputRuntime") == 1
    assert pipeline_calls.count("ChannelRuntime") == 2
    assert pipeline_calls.count("ContextResolver") == 1
    assert pipeline_calls.count("TranslationTurnLifecycleOwner") == 1
    assert pipeline_calls.count("ProviderRuntimeHandle") == 1
    assert {
        "OutputRuntime",
        "ChannelRuntime",
        "ContextResolver",
        "TranslationTurnLifecycleOwner",
        "ProviderRuntimeHandle",
    }.isdisjoint(hub_calls)


def test_hub_has_no_composite_lifecycle_or_provider_alias_surface() -> None:
    tree = ast.parse(HUB_PATH.read_text(encoding="utf-8"))
    hub = _class(tree, "ClientHub")
    methods = {
        node.name for node in hub.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    fields = {
        node.target.id
        for node in hub.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert {"start", "stop"}.isdisjoint(methods)
    assert {
        "replace_stt_provider_request",
        "handoff_stt_provider_request",
        "replace_peer_stt_provider_request",
        "handoff_peer_stt_provider_request",
        "replace_llm_provider",
    }.isdisjoint(methods)
    assert {"stt", "peer_stt", "llm"}.isdisjoint(fields)


def test_components_record_is_frozen_and_contains_no_lifecycle_policy() -> None:
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    components = _class(tree, "RuntimePipelineComponents")

    assert any(
        isinstance(decorator, ast.Call)
        and _call_name(decorator) == "dataclass"
        and any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in components.decorator_list
    )
    assert not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) for node in components.body
    )


def test_cut_over_consumers_do_not_reach_nested_hub_owners() -> None:
    paths = (
        SOURCE_ROOT / "app" / "adapters" / "application_runtime_shutdown.py",
        SOURCE_ROOT / "app" / "adapters" / "peer_application_state.py",
        SOURCE_ROOT / "app" / "adapters" / "peer_capture_provider.py",
        SOURCE_ROOT / "app" / "adapters" / "self_capture_provider.py",
        SOURCE_ROOT / "app" / "services" / "settings_runtime_effects.py",
        SOURCE_ROOT / "app" / "wiring_local_asr_application.py",
        SOURCE_ROOT / "app" / "wiring_managed_account.py",
        SOURCE_ROOT / "app" / "wiring_peer_application.py",
        SOURCE_ROOT / "app" / "wiring_provider_runtime.py",
        SOURCE_ROOT / "composition" / "application_startup.py",
        SOURCE_ROOT / "composition" / "application_state.py",
        SOURCE_ROOT / "composition" / "local_asr_production_evidence.py",
        SOURCE_ROOT / "release_evidence" / "local_asr_production_composition.py",
    )
    forbidden = (
        "hub.start(",
        "hub.stop(",
        "hub.llm",
        "hub.local_asr_provider_runtime",
        "hub.provider_runtime_handles",
        "stop_hub_owned_runtimes",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        for residue in forbidden:
            assert residue not in source, f"{path.relative_to(REPO_ROOT)}: {residue}"
