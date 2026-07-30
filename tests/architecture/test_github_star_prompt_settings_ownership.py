from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSITION_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"


def test_application_composes_github_prompt_through_settings_owner() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert source.count("compose_github_star_prompt_owner(") == 1
    assert "settings=settings" in source
    assert "transaction_result_sink=(require_settings_application().results.set)" in source
    assert "build_ui_prompt_clipboard_state_settings_path_patch" not in source
    assert "_persist_order24_state_mutation" not in source
    assert "_github_star_prompt_translation_connection_for" not in source
    assert "_github_star_prompt_has_managed_connection" not in source
    assert "_github_star_prompt_has_user_owned_cloud_connection" not in source
