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
        "cloud_free_tier_control",
        "speaker_diarization_control",
        "gpu_device_control",
        "gpu_llm_control",
        "model_assets_control",
        "local_llm_connection_control",
        "custom_stt_connection_control",
        "managed_key_control",
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


def test_settings_view_has_no_whole_settings_escape_hatch() -> None:
    path = SOURCE_ROOT / "ui" / "views" / "settings.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = _imported_modules(path)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert "puripuly_heart.config.settings" not in imported
    assert "AppSettings" not in names
    assert "SettingsSnapshot" not in names
    assert "_settings" not in attributes
    assert "_provider_settings_draft" not in attributes


def test_settings_presentation_sinks_do_not_carry_whole_settings() -> None:
    source = (
        SOURCE_ROOT / "app" / "services" / "capture" / "peer_capture_target_application.py"
    ).read_text(encoding="utf-8")

    assert "SettingsPresentationSink = Callable[[GeneralSettingsSnapshot], None]" in source
    assert "settings_presentation_sink(next_settings)" not in source
    wiring_source = (SOURCE_ROOT / "app" / "wiring" / "wiring_peer_application.py").read_text(
        encoding="utf-8"
    )
    assert "settings_presentation_sink: Callable[[GeneralSettingsSnapshot], None]" in wiring_source
