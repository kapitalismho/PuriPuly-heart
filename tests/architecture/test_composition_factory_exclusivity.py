from __future__ import annotations

import ast

import pytest

from tests.helpers.ast_sources import call_name
from tests.helpers.paths import SOURCE_ROOT

COMPOSITION_PATH = SOURCE_ROOT / "composition" / "application_runtime.py"

CASES = [
    {
        "id": "github_star_prompt",
        "factory_call": "compose_github_star_prompt_owner",
    },
    {
        "id": "vrc_mic_sync",
        "factory_call": "compose_vrc_mic_sync",
    },
    {
        "id": "vrchat_osc_presence",
        "factory_call": "create_vrchat_osc_presence_probe_owner",
    },
    {
        "id": "gpu_recovery",
        "factory_call": "create_gpu_provider_recovery_application_owner",
    },
]


@pytest.fixture(scope="module")
def composition_tree() -> ast.Module:
    return ast.parse(COMPOSITION_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composition_calls_factory_exactly_once(case, composition_tree) -> None:
    calls = [call_name(node) for node in ast.walk(composition_tree) if isinstance(node, ast.Call)]
    assert calls.count(case["factory_call"]) == 1
