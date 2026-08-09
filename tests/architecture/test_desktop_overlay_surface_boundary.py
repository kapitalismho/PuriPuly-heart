from __future__ import annotations

import pathlib

from puripuly_heart.ui import desktop_overlay as desktop_overlay_module
from puripuly_heart.ui.desktop_overlay_surface import contract as overlay_contract
from puripuly_heart.ui.desktop_overlay_surface import renderer as overlay_renderer
from tests.helpers.ast_sources import imported_modules as _imported_modules
from tests.helpers.paths import SOURCE_ROOT

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

PUBLIC_SURFACE_NAMES = (
    "build_desktop_caption_plan",
    "build_desktop_caption_surface",
    "build_desktop_transparent_sizing_host",
    "build_desktop_empty_lock_action",
    "build_desktop_overlay_preview_catalog",
    "desktop_empty_lock_action_label",
    "desktop_overlay_preview_fixture_data_sources",
    "preview_fixture_secret_findings",
)
CONTRACT_NAMES = (
    "DesktopCaptionPlan",
    "DesktopCaptionSlot",
    "DesktopCaptionLine",
    "DesktopCaptionVisualState",
    "DesktopCaptionSizePreset",
    "DESKTOP_CAPTION_MAPPING_TABLE",
    "DesktopOverlayPreviewCatalog",
    "DesktopOverlayPreviewFixture",
)
RUNTIME_OWNED_NAMES = (
    "FletDesktopRendererWindow",
    "RendererWindow",
    "LifecycleSink",
    "run_renderer",
    "run_preview",
)


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


def test_contract_module_holds_no_flet_composition() -> None:
    source = pathlib.Path(overlay_contract.__file__).read_text(encoding="utf-8")
    assert "invoke_control_method" not in source
    assert "ft.Container" not in source


def test_desktop_overlay_keeps_the_public_surface_and_runner_ownership() -> None:
    for name in (*PUBLIC_SURFACE_NAMES, *CONTRACT_NAMES):
        assert getattr(desktop_overlay_module, name, None) is not None, name
    for name in RUNTIME_OWNED_NAMES:
        assert name in vars(desktop_overlay_module), f"{name} must stay owned by the runner module"


def test_public_surface_functions_are_owned_by_the_renderer_module() -> None:
    for name in PUBLIC_SURFACE_NAMES:
        function = getattr(desktop_overlay_module, name)
        assert function is getattr(overlay_renderer, name)
        assert function.__module__ == overlay_renderer.__name__


def test_visibility_confirmation_boundary_stays_on_the_runner() -> None:
    source = (SOURCE_ROOT / "ui" / "desktop_overlay.py").read_text(encoding="utf-8")
    assert "_confirm_window_visible" in source
    assert "confirm_window_visible(" in source
    assert "reveal_window(" not in source
