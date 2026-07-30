from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
PIPELINE_PATH = SOURCE_ROOT / "app" / "wiring_runtime_pipeline.py"
PEER_OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "translation_request.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_production_composes_one_translation_request_owner() -> None:
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    calls = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    source = PIPELINE_PATH.read_text(encoding="utf-8")

    assert calls.count("TranslationRequestOwner") == 1
    assert "provider_runtime=llm_runtime" in source
    assert "context_resolver=context_resolver" in source
    assert "presentation=translation_output_projection" in source
    assert "translation_requests=translation_requests" in source
    assert "translation_requests=translation_requests" in source


def test_peer_owner_contains_no_translation_request_algorithm_or_provider_reference() -> None:
    tree = ast.parse(PEER_OWNER_PATH.read_text(encoding="utf-8"))
    owner = _class(tree, "PeerTranslationChannelOwner")
    methods = {
        node.name for node in owner.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    fields = {
        node.target.id
        for node in owner.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    source = PEER_OWNER_PATH.read_text(encoding="utf-8")

    assert {
        "_format_system_prompt",
        "_detected_language_for_llm",
        "_request_source_language",
        "_capture_llm_provider_request",
        "_raise_if_stale_llm_provider_request",
        "_normalize_translation",
        "_translate_text",
        "_build_translation_process_result",
    }.isdisjoint(methods)
    assert "_llm_provider_runtime" not in fields
    assert "ProviderRuntimeHandle" not in source
    assert "LLMProvider" not in source
    assert "render_translation_prompt_template" not in source
    assert "map_detected_language_for_llm" not in source
    assert "llm.translate(" not in source
    assert "provider.translate(" not in source


def test_request_side_effects_are_owned_only_by_translation_request_owner() -> None:
    owner_source = OWNER_PATH.read_text(encoding="utf-8")
    production_request_calls: list[tuple[Path, int]] = []
    prompt_render_calls: list[tuple[Path, int]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "translate":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "provider":
                    production_request_calls.append((path, node.lineno))
            if _call_name(node) == "render_translation_prompt_template":
                prompt_render_calls.append((path, node.lineno))

    assert [path for path, _line in production_request_calls] == [OWNER_PATH, OWNER_PATH]
    assert [path for path, _line in prompt_render_calls] == [OWNER_PATH]
    assert "current_provider_generation()" in owner_source
    assert "is_current_provider_generation(" in owner_source
    assert "resolve_for_request(" in owner_source


def test_translation_request_owner_has_no_lifecycle_or_output_delivery_authority() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    owner = _class(tree, "TranslationRequestOwner")
    methods = {
        node.name for node in owner.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert {"start", "close", "stop", "publish_translation_result"}.isdisjoint(methods)
    assert "OutputRuntime" not in source
    assert "asyncio.create_task" not in source
    assert "replace_provider(" not in source
    assert "close(" not in source
