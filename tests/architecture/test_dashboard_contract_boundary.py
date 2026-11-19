from __future__ import annotations

import pathlib

from puripuly_heart.ui.dashboard import contract as dashboard_contract
from puripuly_heart.ui.dashboard import renderer as dashboard_renderer
from tests.helpers.ast_sources import imported_modules as _imported_modules

FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.app.services",
)


def test_dashboard_contract_and_renderer_stay_above_backend_owners() -> None:
    from puripuly_heart.ui.dashboard import capture as dashboard_capture
    from puripuly_heart.ui.dashboard import capture_notices as dashboard_capture_notices

    for module in (
        dashboard_contract,
        dashboard_renderer,
        dashboard_capture,
        dashboard_capture_notices,
    ):
        path = pathlib.Path(module.__file__)
        for imported in _imported_modules(path):
            assert not imported.startswith(
                FORBIDDEN_IMPORT_PREFIXES
            ), f"{path.name} must not import backend implementation: {imported}"


def test_capture_controls_consume_the_contract_without_dashboard_view_access() -> None:
    from puripuly_heart.ui.dashboard import capture as dashboard_capture
    from puripuly_heart.ui.dashboard import capture_notices as dashboard_capture_notices

    for module in (dashboard_capture, dashboard_capture_notices):
        imported = _imported_modules(pathlib.Path(module.__file__))
        assert not any(name.startswith("puripuly_heart.ui.views") for name in imported)


def test_dashboard_view_implements_the_explicit_contract() -> None:
    from puripuly_heart.ui.views.dashboard import DashboardView

    for method in (
        "bind_dashboard_intents",
        "self_capture_control",
        "peer_capture_control",
        "overlay_control",
    ):
        assert callable(getattr(DashboardView, method)), method
