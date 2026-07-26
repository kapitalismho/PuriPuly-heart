from __future__ import annotations

import ast
import pathlib

import puripuly_heart
from puripuly_heart.ui.settings import contract as settings_contract
from puripuly_heart.ui.settings import renderer as settings_renderer

SOURCE_ROOT = pathlib.Path(puripuly_heart.__file__).resolve().parent
FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.ui.controller",
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
)
G14_OWNED_VIEW_SINKS = (
    "show_snackbar",
    "runtime_log_basic",
    "runtime_log_detailed",
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


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


def test_translator_app_wires_g14_settings_intents_through_one_path() -> None:
    assigned = set(_settings_view_attribute_assignments("view_settings"))
    for owned in (*G14_OWNED_VIEW_CALLBACKS, *G14_OWNED_VIEW_SINKS):
        assert owned not in assigned, f"{owned} must be bound through bind_settings_intents"

    app_source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert app_source.count("bind_settings_intents(") == 1


def test_settings_intent_groups_expose_the_accepted_field_sets() -> None:
    surface_fields = tuple(settings_contract.SettingsSurfaceIntents.__dataclass_fields__)
    provider_fields = tuple(settings_contract.SettingsProviderIntents.__dataclass_fields__)
    assert surface_fields == G14_SURFACE_INTENT_FIELDS
    assert provider_fields == G14_PROVIDER_INTENT_FIELDS
