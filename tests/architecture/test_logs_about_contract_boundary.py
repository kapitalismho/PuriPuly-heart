from __future__ import annotations

import pathlib

from puripuly_heart.ui.about import contract as about_contract
from puripuly_heart.ui.about import renderer as about_renderer
from puripuly_heart.ui.logs import contract as logs_contract
from puripuly_heart.ui.logs import renderer as logs_renderer
from tests.helpers.ast_sources import imported_modules as _imported_modules
from tests.helpers.paths import SOURCE_ROOT

FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.app.services",
    "puripuly_heart.app.wiring",
    "puripuly_heart.config",
)

CONTRACT_MODULES = (logs_contract, logs_renderer, about_contract, about_renderer)


def test_logs_and_about_contracts_stay_above_backend_owners() -> None:
    for module in CONTRACT_MODULES:
        path = pathlib.Path(module.__file__)
        for imported in _imported_modules(path):
            assert not imported.startswith(
                FORBIDDEN_IMPORT_PREFIXES
            ), f"{path.name} must not import backend implementation: {imported}"


def test_logs_and_about_contracts_do_not_reach_into_the_view() -> None:
    for module in CONTRACT_MODULES:
        imported = _imported_modules(pathlib.Path(module.__file__))
        assert not any(name.startswith("puripuly_heart.ui.views") for name in imported)


def test_about_view_implements_the_explicit_state_sink() -> None:
    from puripuly_heart.ui.views.about import AboutView

    for name in vars(about_contract.AboutStateSink):
        if name.startswith("_"):
            continue
        assert callable(getattr(AboutView, name, None)), name


def test_production_logs_and_about_surfaces_use_the_renderers() -> None:
    logs_source = (SOURCE_ROOT / "ui" / "views" / "logs.py").read_text(encoding="utf-8")
    assert "compose_logs_surface(" in logs_source
    assert "LogsSurfaceSlots(" in logs_source

    about_source = (SOURCE_ROOT / "ui" / "views" / "about.py").read_text(encoding="utf-8")
    assert "compose_about_surface(" in about_source
    assert "AboutSurfaceSlots(" in about_source


def test_renderers_do_not_own_runtime_state_or_callbacks() -> None:
    for module in (logs_renderer, about_renderer):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        assert "on_click" not in source
        assert "webbrowser" not in source
        assert "asyncio" not in source
