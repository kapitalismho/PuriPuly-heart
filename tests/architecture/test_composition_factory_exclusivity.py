from __future__ import annotations

import pytest

from tests.helpers.paths import SOURCE_ROOT

COMPOSITION_PATH = SOURCE_ROOT / "composition" / "application_runtime.py"

CASES = [
    {
        "id": "github_star_prompt",
        "factory_call": "compose_github_star_prompt_owner(",
        "required_snippets": (
            "settings=settings",
            "transaction_result_sink=(require_settings_application().results.set)",
        ),
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
        "factory_call": "compose_vrc_mic_sync(",
        "required_snippets": (
            "configure_vrc_mic=lambda *, enabled: "
            "(require_vrc_mic_sync().configure(enabled=enabled))",
        ),
        "retired_names": ("def _stop_vrc_mic_receiver(",),
    },
    {
        "id": "vrchat_osc_presence",
        "factory_call": "create_vrchat_osc_presence_probe_owner(",
        "required_snippets": (
            "presence_provider=lambda: vrchat_osc_presence",
            "port_provider=vrchat_probe_port",
            "publish_notice=presentation.set_dashboard_vrchat_osc_notice",
        ),
        "retired_names": (),
    },
    {
        "id": "gpu_recovery",
        "factory_call": "create_gpu_provider_recovery_application_owner(",
        "required_snippets": (
            "await require_gpu_recovery().recover(",
            "lambda: gpu_recovery_request(",
        ),
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


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composition_calls_factory_exactly_once(case) -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")
    assert source.count(case["factory_call"]) == 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composition_contains_required_snippets(case) -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")
    for snippet in case["required_snippets"]:
        assert snippet in source


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_composition_has_no_retired_names(case) -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")
    for retired_name in case["retired_names"]:
        assert retired_name not in source
