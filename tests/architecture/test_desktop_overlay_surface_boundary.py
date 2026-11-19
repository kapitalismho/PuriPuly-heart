from __future__ import annotations

import pathlib

from puripuly_heart.ui.desktop_overlay_surface import contract as overlay_contract
from puripuly_heart.ui.desktop_overlay_surface import renderer as overlay_renderer
from tests.helpers.ast_sources import imported_modules as _imported_modules

SURFACE_MODULES = (overlay_contract, overlay_renderer)

FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.runtime",
    "puripuly_heart.ui.app",
    "puripuly_heart.ui.views",
    "puripuly_heart.app.services",
    "puripuly_heart.app.wiring",
)
FORBIDDEN_EXACT_IMPORTS = ("puripuly_heart.ui.desktop_overlay",)
FORBIDDEN_RUNNER_IMPORTS = ("websockets", "argparse", "asyncio")


def test_overlay_surface_modules_do_not_import_the_runner_or_views() -> None:
    for module in SURFACE_MODULES:
        path = pathlib.Path(module.__file__)
        for imported in _imported_modules(path):
            assert not imported.startswith(
                FORBIDDEN_IMPORT_PREFIXES
            ), f"{path.name} must not import {imported}"
            assert (
                imported not in FORBIDDEN_EXACT_IMPORTS
            ), f"{path.name} must not import {imported}"


def test_overlay_surface_modules_do_not_absorb_the_runner_transport() -> None:
    for module in SURFACE_MODULES:
        imported = _imported_modules(pathlib.Path(module.__file__))
        for forbidden in FORBIDDEN_RUNNER_IMPORTS:
            assert forbidden not in imported, f"{module.__name__} must stay free of {forbidden}"


def test_contract_module_does_not_import_flet() -> None:
    assert "flet" not in _imported_modules(pathlib.Path(overlay_contract.__file__))
