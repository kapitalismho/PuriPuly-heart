from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.ast_sources import call_name as _call_name
from tests.helpers.paths import SOURCE_ROOT

PEER_OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
OWNER_PATH = SOURCE_ROOT / "core" / "orchestrator" / "translation_request.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


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
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.name for node in ast.walk(tree) if isinstance(node, ast.alias)
    }

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
    assert {
        "ProviderRuntimeHandle",
        "LLMProvider",
        "TranslationBackend",
        "TranslationBackendRequest",
        "LlmTranslationBackend",
        "render_translation_prompt_template",
        "map_detected_language_for_llm",
    }.isdisjoint(names)


def _request_preparation_sites(
    source_root: Path,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    request_root = source_root / "core" / "orchestrator"
    production_request_calls: list[tuple[Path, int]] = []
    prompt_render_calls: list[tuple[Path, int]] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                path.is_relative_to(request_root)
                and _call_name(node) == "TranslationBackendRequest"
            ):
                production_request_calls.append((path, node.lineno))
            if _call_name(node) == "render_translation_prompt_template":
                prompt_render_calls.append((path, node.lineno))
    return production_request_calls, prompt_render_calls


def test_request_preparation_and_prompt_rendering_belong_to_request_owner() -> None:
    production_request_calls, prompt_render_calls = _request_preparation_sites(SOURCE_ROOT)

    assert {path for path, _line in production_request_calls} == {OWNER_PATH}
    assert [path for path, _line in prompt_render_calls] == [OWNER_PATH]


def test_request_preparation_guard_allows_port_delegation_but_detects_backend_bypass(
    tmp_path,
) -> None:
    orchestrator = tmp_path / "core" / "orchestrator"
    orchestrator.mkdir(parents=True)
    owner = orchestrator / "translation_request.py"
    owner.write_text(
        "request = TranslationBackendRequest(text='synthetic')\n"
        "prompt = render_translation_prompt_template(template)\n",
        encoding="utf-8",
    )
    channel = orchestrator / "peer.py"
    channel.write_text(
        'description = "TranslationBackendRequest(text=source)"\n'
        "async def process(requests, direct_request):\n"
        "    return await requests.translate(direct_request)\n",
        encoding="utf-8",
    )
    requests, prompts = _request_preparation_sites(tmp_path)
    assert requests == [(owner, 1)]
    assert prompts == [(owner, 2)]

    channel.write_text(
        "renamed_backend.translate(TranslationBackendRequest(text='synthetic'))\n",
        encoding="utf-8",
    )
    requests, prompts = _request_preparation_sites(tmp_path)
    assert requests == [(channel, 1), (owner, 1)]
    assert prompts == [(owner, 2)]


def test_translation_request_owner_has_no_lifecycle_or_output_delivery_authority() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    owner = _class(tree, "TranslationRequestOwner")
    methods = {
        node.name for node in owner.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    calls = {_call_name(node) for node in ast.walk(owner) if isinstance(node, ast.Call)}

    assert {"start", "close", "stop", "publish_translation_result"}.isdisjoint(methods)
    assert "OutputRuntime" not in names
    assert {"create_task", "replace_provider", "close"}.isdisjoint(calls)
