from pathlib import Path

import puripuly_heart.composition.ui_application as composition_module
from puripuly_heart.app.services.ui_application import UiApplicationBoundary


def test_composition_forwards_options_without_passing_a_flet_page(monkeypatch) -> None:
    captured: dict[str, object] = {}
    backend = object()
    presentation = object()
    logging_sinks = object()
    presence = object()

    def controller_factory(**kwargs):
        captured.update(kwargs)
        return backend

    monkeypatch.setattr(composition_module, "GuiController", controller_factory)

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=Path("settings.json"),
        allow_stable_settings_import=True,
        runtime_logging_sinks=logging_sinks,
        vrchat_osc_presence=presence,
    )

    assert isinstance(application, UiApplicationBoundary)
    assert application.wraps(backend)
    assert captured == {
        "page": None,
        "app": presentation,
        "config_path": Path("settings.json"),
        "allow_stable_settings_import": True,
        "runtime_logging_sinks": logging_sinks,
        "vrchat_osc_presence": presence,
    }
