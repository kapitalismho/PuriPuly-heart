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
        "retired_names": (
            "build_ui_prompt_clipboard_state_settings_path_patch",
            "_persist_order24_state_mutation",
            "_github_star_prompt_translation_connection_for",
            "_github_star_prompt_has_managed_connection",
            "_github_star_prompt_has_user_owned_cloud_connection",
        ),
    },
    {
        "id": "vrc_mic_sync",
        "factory_call": "compose_vrc_mic_sync",
        "retired_names": ("_stop_vrc_mic_receiver",),
    },
    {
        "id": "vrchat_osc_presence",
        "factory_call": "create_vrchat_osc_presence_probe_owner",
        "retired_names": (),
    },
    {
        "id": "gpu_recovery",
        "factory_call": "create_gpu_provider_recovery_application_owner",
        "retired_names": (
            "_gpu_provider_recovery_lock",
            "_get_gpu_provider_recovery_lock",
            "_apply_gpu_runtime_owner_recovery_locked",
            "_execute_gpu_provider_recovery_retry",
            "_build_gpu_recovery_request",
            "_abort_provider_recoveries",
            "_resume_gpu_provider_consumers",
            "_gpu_provider_recovery_execution",
            "_complete_gpu_provider_recovery",
            "_desired_gpu_channels",
            "_gpu_provider_recovery_channel_plans",
        ),
    },
]


@pytest.fixture(scope="module")
def composition_tree() -> ast.Module:
    return ast.parse(COMPOSITION_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composition_calls_factory_exactly_once(case, composition_tree) -> None:
    calls = [call_name(node) for node in ast.walk(composition_tree) if isinstance(node, ast.Call)]
    assert calls.count(case["factory_call"]) == 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composition_has_no_retired_state_or_algorithms(case, composition_tree) -> None:
    referenced = (
        {node.id for node in ast.walk(composition_tree) if isinstance(node, ast.Name)}
        | {node.attr for node in ast.walk(composition_tree) if isinstance(node, ast.Attribute)}
        | {
            node.name
            for node in ast.walk(composition_tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
    )
    assert set(case["retired_names"]).isdisjoint(referenced)
