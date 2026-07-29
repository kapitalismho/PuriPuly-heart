from pathlib import Path

import puripuly_heart.composition.ui_application as composition_module
from puripuly_heart.app.services.ui_application import UiApplicationBoundary


def test_composition_forwards_options_without_passing_a_flet_page(monkeypatch) -> None:
    captured: dict[str, object] = {}
    presentation = object()
    logging_sinks = object()
    presence = object()
    runtime_components = object()

    class Backend:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.installed_runtime = None

        def install_runtime_composition(self, components: object) -> None:
            self.installed_runtime = components

        def _log_error(self, message: str) -> None:
            _ = message

    backends: list[Backend] = []

    def controller_factory(**kwargs):
        backend = Backend(**kwargs)
        backends.append(backend)
        return backend

    monkeypatch.setattr(composition_module, "GuiController", controller_factory)
    monkeypatch.setattr(
        composition_module,
        "compose_gui_runtime_components",
        lambda backend: runtime_components,
    )

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=Path("settings.json"),
        allow_stable_settings_import=True,
        runtime_logging_sinks=logging_sinks,
        vrchat_osc_presence=presence,
    )

    [backend] = backends
    assert isinstance(application, UiApplicationBoundary)
    assert application.wraps(backend)
    assert backend.installed_runtime is runtime_components
    settings_owner = captured.pop("settings_owner")
    provider_settings_owner = captured.pop("provider_settings_owner")
    assert provider_settings_owner.settings is settings_owner
    assert captured == {
        "page": None,
        "app": presentation,
        "config_path": Path("settings.json"),
        "allow_stable_settings_import": True,
        "runtime_logging_sinks": logging_sinks,
        "vrchat_osc_presence": presence,
    }
