from pathlib import Path
from types import SimpleNamespace

import puripuly_heart.composition.ui_application as composition_module
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


def test_composition_forwards_explicit_options_without_a_flet_page(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()
    presentation = object()
    logging_sinks = object()
    presence = object()

    def compose(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(composition_module, "compose_application_runtime", compose)

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=Path("settings.json"),
        runtime_logging_sinks=logging_sinks,
        vrchat_osc_presence=presence,
    )

    assert application is expected
    assert captured == {
        "presentation": presentation,
        "config_path": Path("settings.json"),
        "runtime_logging_sinks": logging_sinks,
        "vrchat_osc_presence": presence,
    }
    assert "page" not in captured


def test_real_composition_returns_the_application_boundary(tmp_path: Path) -> None:
    presentation = FletUiPresentationAdapter(
        SimpleNamespace(debug_ui_preview=False),
    )

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=tmp_path / "settings.json",
    )

    assert isinstance(application, UiApplicationBoundary)
