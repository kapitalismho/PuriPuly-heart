from __future__ import annotations

import ast
import pathlib

import puripuly_heart
from puripuly_heart.ui.about import contract as about_contract
from puripuly_heart.ui.about import renderer as about_renderer
from puripuly_heart.ui.logs import contract as logs_contract
from puripuly_heart.ui.logs import renderer as logs_renderer

SOURCE_ROOT = pathlib.Path(puripuly_heart.__file__).resolve().parent
FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.app.services",
    "puripuly_heart.app.wiring",
    "puripuly_heart.config",
)

CONTRACT_MODULES = (logs_contract, logs_renderer, about_contract, about_renderer)

LOGS_INTENT_FIELDS = ("runtime_logging_mode_change",)
LOGS_OWNED_VIEW_CALLBACKS = ("on_mode_change",)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _view_attribute_assignments(attribute_owner: str) -> list[str]:
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


def test_logs_intents_expose_the_accepted_field_set() -> None:
    assert tuple(logs_contract.LogsIntents.__dataclass_fields__) == LOGS_INTENT_FIELDS


def test_logs_view_implements_the_explicit_contract() -> None:
    from puripuly_heart.ui.views.logs import LogsView

    assert callable(getattr(LogsView, "bind_logs_intents", None))
    for name in vars(logs_contract.LogsStateSink):
        if name.startswith("_"):
            continue
        assert getattr(LogsView, name, None) is not None, name


def test_about_view_implements_the_explicit_state_sink() -> None:
    from puripuly_heart.ui.views.about import AboutView

    for name in vars(about_contract.AboutStateSink):
        if name.startswith("_"):
            continue
        assert callable(getattr(AboutView, name, None)), name


def test_translator_app_wires_logs_intents_through_one_path() -> None:
    assigned = set(_view_attribute_assignments("view_logs"))
    for owned in LOGS_OWNED_VIEW_CALLBACKS:
        assert owned not in assigned, f"{owned} must be bound through bind_logs_intents"
    assert not any(name.startswith("on_") for name in assigned)

    app_source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert app_source.count("bind_logs_intents(") == 1


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
