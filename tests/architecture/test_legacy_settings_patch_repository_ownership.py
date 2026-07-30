from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSITION_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
UI_RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"
PROVIDER_SETTINGS_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "provider" / "provider_settings.py"
)
SETTINGS_APPLICATION_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "settings_application.py"
)
OPENROUTER_PKCE_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "openrouter_pkce_flow.py"
)
MANAGED_ACCOUNT_WIRING_PATH = ROOT / "src" / "puripuly_heart" / "app" / "wiring_managed_account.py"


def test_application_composes_settings_patch_repository_owners() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")
    ui_runtime_source = UI_RUNTIME_PATH.read_text(encoding="utf-8")
    provider_settings_source = PROVIDER_SETTINGS_PATH.read_text(encoding="utf-8")
    settings_application_source = SETTINGS_APPLICATION_PATH.read_text(encoding="utf-8")
    openrouter_pkce_source = OPENROUTER_PKCE_PATH.read_text(encoding="utf-8")
    managed_account_source = MANAGED_ACCOUNT_WIRING_PATH.read_text(encoding="utf-8")

    assert "_ControllerSettingsPatchRepository" not in source
    assert "def _settings_snapshot_values" not in source
    assert "_legacy_settings_patch_repository" not in source
    assert "create_legacy_patch_repository(" not in source
    assert managed_account_source.count("settings.create_legacy_patch_repository(") == 1
    assert provider_settings_source.count("self.settings.create_legacy_patch_repository(") == 3
    assert settings_application_source.count("self.settings.create_legacy_patch_repository(") == 1
    assert openrouter_pkce_source.count("self.settings.create_legacy_patch_repository(") == 1
    assert "class ProviderApplicationOwner" in provider_settings_source
    assert "class SettingsApplicationOwner" in settings_application_source
    assert "class OpenRouterPkceApplicationOwner" in openrouter_pkce_source
    assert "SettingsTransactionResultOwner" in settings_application_source
    assert "last_settings_mutation_result" not in source
    assert "def _compensate_failed_local_asr_settings_apply(" not in source
    assert "persist_settings=self._persist_settings_at_controller_boundary" not in source
    assert "begin_canonical_mutation=" not in source
    assert "_persist_settings_at_controller_boundary" not in source
    assert "_update_canonical_settings_from_legacy_delta" not in source
    assert "_canonical_vnext_after_legacy_delta" not in source
    assert "_remember_canonical_legacy_projection" not in source
    assert "_begin_canonical_mutation" not in source
    assert "_rollback_canonical_mutation" not in source
    assert "_complete_canonical_mutation" not in source
    for method_name in (
        "_apply_order22_order23_order24_settings_via_mutation_services",
        "_apply_stt_language_audio_settings_via_mutation_service",
        "_apply_overlay_osc_output_settings_via_mutation_service",
        "_mutate_order24_settings_patch",
        "_apply_ui_prompt_clipboard_state_settings_via_mutation_service",
        "_resync_committed_order22_settings_after_strict_save_failure",
        "_resync_committed_order22_provider_runtime_after_strict_save_failure",
        "_resync_committed_order23_settings_after_strict_save_failure",
        "_resync_committed_order24_settings_after_strict_save_failure",
        "_apply_settings_direct",
        "_save_settings",
        "persist_settings",
    ):
        assert f"def {method_name}(" not in source

    language_change_source = ui_runtime_source.split(
        "    async def on_dashboard_language_change(",
        maxsplit=1,
    )[1].split("\n    def ", maxsplit=1)[0]
    assert "apply_language_selection" in language_change_source
    assert ".begin(" not in language_change_source
    assert ".apply_legacy_delta(" not in language_change_source
