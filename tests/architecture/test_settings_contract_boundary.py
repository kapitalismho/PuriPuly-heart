from __future__ import annotations

import ast
import pathlib

from puripuly_heart.ui.settings import contract as settings_contract
from puripuly_heart.ui.settings import renderer as settings_renderer
from tests.helpers.ast_sources import imported_modules as _imported_modules
from tests.helpers.paths import SOURCE_ROOT

FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.app.services",
    "puripuly_heart.app.wiring",
    "puripuly_heart.config",
)

SETTINGS_CONTRACT_MODULES = (settings_contract, settings_renderer)

G14_SURFACE_INTENT_FIELDS = (
    "settings_changed",
    "show_snackbar",
    "runtime_log_basic",
    "runtime_log_detailed",
)
G14_PROVIDER_INTENT_FIELDS = (
    "providers_changed",
    "request_openrouter_pkce",
    "verify_api_key",
    "provider_secret_change",
    "secret_cleared",
    "local_llm_secret_changed",
    "gpu_discovery_requested",
    "custom_stt_secret_changed",
)
G14_OWNED_VIEW_CALLBACKS = (
    "on_settings_changed",
    "on_providers_changed",
    "on_request_openrouter_pkce",
    "on_verify_api_key",
    "on_provider_secret_change",
    "on_secret_cleared",
    "on_local_llm_secret_changed",
    "on_gpu_discovery_requested",
    "on_custom_stt_secret_changed",
)
G14_OWNED_VIEW_SINKS = (
    "show_snackbar",
    "runtime_log_basic",
    "runtime_log_detailed",
)
G15_GENERAL_INTENT_FIELDS = (
    "start_microphone_test",
    "telemetry_consent_change",
    "list_loopback_capture_options",
    "list_loopback_process_options",
    "list_loopback_device_options",
    "current_loopback_capture_option",
    "apply_loopback_capture_option",
    "loopback_capture_summary",
    "osc_effective_ports",
)
G15_PROMPT_INTENT_FIELDS = ("prompt_apply_settings",)
G15_OVERLAY_INTENT_FIELDS = (
    "desktop_overlay_lock_change",
    "desktop_overlay_size_change",
    "desktop_overlay_recovery_action",
    "desktop_overlay_position_reset",
    "view_logs",
    "calibration_begin",
    "calibration_change",
    "calibration_apply",
    "calibration_cancel",
)
G15_OWNED_VIEW_CALLBACKS = (
    "on_prompt_apply_settings",
    "on_start_microphone_test",
    "on_telemetry_consent_change",
    "on_list_loopback_capture_options",
    "on_list_loopback_process_options",
    "on_list_loopback_device_options",
    "on_current_loopback_capture_option",
    "on_apply_loopback_capture_option",
    "on_loopback_capture_summary",
    "on_desktop_overlay_lock_change",
    "on_desktop_overlay_size_change",
    "on_desktop_overlay_recovery_action",
    "on_desktop_overlay_position_reset",
    "on_view_logs",
    "on_overlay_calibration_begin",
    "on_overlay_calibration_change",
    "on_overlay_calibration_apply",
    "on_overlay_calibration_cancel",
)


def _settings_view_attribute_assignments(attribute_owner: str) -> list[str]:
    tree = ast.parse((SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8"))
    assigned: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == attribute_owner
            ):
                assigned.append(target.attr)
    return assigned


def test_settings_contract_and_renderer_stay_above_backend_owners() -> None:
    for module in SETTINGS_CONTRACT_MODULES:
        path = pathlib.Path(module.__file__)
        for imported in _imported_modules(path):
            assert not imported.startswith(
                FORBIDDEN_IMPORT_PREFIXES
            ), f"{path.name} must not import backend implementation: {imported}"


def test_settings_contract_modules_do_not_reach_into_the_view() -> None:
    for module in SETTINGS_CONTRACT_MODULES:
        imported = _imported_modules(pathlib.Path(module.__file__))
        assert not any(name.startswith("puripuly_heart.ui.views") for name in imported)


def test_settings_view_implements_the_explicit_contract() -> None:
    from puripuly_heart.ui.views.settings import SettingsView

    for method in (
        "bind_settings_intents",
        "self_stt_control",
        "peer_stt_control",
        "translation_provider_control",
        "translation_connection_control",
        "translation_fallback_control",
        "gpu_device_control",
        "local_llm_connection_control",
        "custom_stt_connection_control",
        "managed_key_control",
        "peer_expected_language_control",
        "api_keys_control",
    ):
        assert callable(getattr(SettingsView, method)), method


def test_settings_state_sink_protocol_covers_every_settings_view_push() -> None:
    from puripuly_heart.ui.views.settings import SettingsView

    sink_methods = {
        name
        for name in vars(settings_contract.SettingsProviderStateSink)
        if not name.startswith("_")
    }
    for name in sink_methods:
        assert callable(getattr(SettingsView, name, None)), name


def test_production_settings_surface_uses_an_external_slot_provider() -> None:
    source = (SOURCE_ROOT / "ui" / "views" / "settings.py").read_text(encoding="utf-8")
    assert "SettingsApiSurfaceSlots.from_slot_provider(self)" in source
    assert "compose_settings_api_surface(" in source
    assert "placeholder_factory=self._wrap_empty_unit_card" in source


def test_translator_app_wires_every_settings_intent_through_one_path() -> None:
    assigned = set(_settings_view_attribute_assignments("view_settings"))
    for owned in (
        *G14_OWNED_VIEW_CALLBACKS,
        *G14_OWNED_VIEW_SINKS,
        *G15_OWNED_VIEW_CALLBACKS,
    ):
        assert owned not in assigned, f"{owned} must be bound through bind_settings_intents"
    assert not any(name.startswith("on_") for name in assigned)

    app_source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert app_source.count("bind_settings_intents(") == 1


def test_settings_intent_groups_expose_the_accepted_field_sets() -> None:
    assert (
        tuple(settings_contract.SettingsSurfaceIntents.__dataclass_fields__)
        == G14_SURFACE_INTENT_FIELDS
    )
    assert (
        tuple(settings_contract.SettingsProviderIntents.__dataclass_fields__)
        == G14_PROVIDER_INTENT_FIELDS
    )
    assert (
        tuple(settings_contract.SettingsGeneralIntents.__dataclass_fields__)
        == G15_GENERAL_INTENT_FIELDS
    )
    assert (
        tuple(settings_contract.SettingsPromptIntents.__dataclass_fields__)
        == G15_PROMPT_INTENT_FIELDS
    )
    assert (
        tuple(settings_contract.SettingsOverlayIntents.__dataclass_fields__)
        == G15_OVERLAY_INTENT_FIELDS
    )


def test_g15_surfaces_consume_the_shared_renderer_without_private_g14_access() -> None:
    import inspect

    source = (SOURCE_ROOT / "ui" / "views" / "settings.py").read_text(encoding="utf-8")
    for call in (
        "compose_settings_general_surface(",
        "compose_settings_prompt_surface(",
        "compose_settings_overlay_surface(",
    ):
        assert call in source

    expected = {
        settings_renderer.compose_settings_general_surface: "SettingsGeneralSurfaceSlots",
        settings_renderer.compose_settings_prompt_surface: "SettingsPromptSurfaceSlots",
        settings_renderer.compose_settings_overlay_surface: "SettingsOverlaySurfaceSlots",
    }
    for function, slot_type in expected.items():
        parameters = inspect.signature(function).parameters
        assert parameters["slots"].annotation == slot_type
        assert not any("Api" in str(parameter.annotation) for parameter in parameters.values())

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {
                "compose_settings_general_surface",
                "compose_settings_prompt_surface",
                "compose_settings_overlay_surface",
            }
        ):
            rendered = ast.dump(node)
            assert "_api_surface" not in rendered
            assert "_self_stt_card" not in rendered
