from __future__ import annotations

import ast
import inspect
from pathlib import Path

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UIEventBridgePort, UiPresentationPort
from puripuly_heart.app.services.ui_application import (
    UI_APPLICATION_USER_INTENT_METHODS,
    UiApplicationBoundary,
)
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "app.py"
CONTROLLER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"

UI_APPLICATION_NON_INTENT_MEMBERS = {
    "application_shutdown_callbacks",
    "bind_application_lifecycle",
    "build_managed_openrouter_byok_target_settings",
    "cancel_managed_auth_task",
    "clear_managed_auth_task",
    "close_github_star_prompt_runtime",
    "close_managed_auth_tasks",
    "compatibility_settings",
    "current_loopback_capture_option_value",
    "dashboard_managed_auth_action",
    "dashboard_managed_auth_prompt_kind",
    "emit_application_shutdown_diagnostic",
    "get_event_language_codes",
    "is_current_github_star_prompt_generation",
    "list_loopback_capture_options",
    "list_loopback_device_options",
    "list_loopback_process_options",
    "local_llm_selected",
    "log_basic",
    "log_detailed",
    "loopback_capture_summary",
    "managed_auth_task_names",
    "managed_auth_tasks_open",
    "merge_settings_tab_apply_with_current_languages",
    "merge_settings_view_change_with_current",
    "overlay_calibration",
    "overlay_peer_presentation_state",
    "refresh_settings_after_openrouter_pkce_success",
    "refresh_settings_projection",
    "should_show_github_star_prompt",
    "state",
    "stop",
    "stop_github_star_prompt_ingress",
    "supports_discord_managed_auth_reopen",
    "translation_enable_succeeded",
}


def _contract_members(contract: type[object]) -> set[str]:
    return {
        name
        for name, member in contract.__dict__.items()
        if not name.startswith("_") and (inspect.isfunction(member) or isinstance(member, property))
    }


def _imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _assert_contract_signatures(contract: type[object], implementation: type[object]) -> None:
    for name, contract_member in contract.__dict__.items():
        if name.startswith("_"):
            continue
        implementation_member = inspect.getattr_static(implementation, name)
        if isinstance(contract_member, property):
            assert isinstance(implementation_member, property)
            assert inspect.signature(contract_member.fget) == inspect.signature(
                implementation_member.fget
            )
        elif inspect.isfunction(contract_member):
            assert inspect.signature(contract_member) == inspect.signature(implementation_member)


def test_ui_application_contract_covers_every_translator_app_boundary_access() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    accessed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "application"
    }

    contract = _contract_members(UiApplicationPort)
    implementation = _contract_members(UiApplicationBoundary)

    assert accessed <= contract
    assert contract <= implementation
    assert "__getattr__" not in UiApplicationBoundary.__dict__
    assert not hasattr(UiApplicationBoundary, "backend")


def test_ui_boundary_implementations_match_every_declared_contract_signature() -> None:
    _assert_contract_signatures(UiApplicationPort, UiApplicationBoundary)
    _assert_contract_signatures(UiPresentationPort, FletUiPresentationAdapter)


def test_ui_event_bridge_boundary_declares_every_consumed_operation() -> None:
    assert _contract_members(UIEventBridgePort) == {
        "close",
        "report_overlay_state",
        "run",
        "wait_started",
    }

    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    assert "_ui_event_bridge: UIEventBridgePort | None" in source
    assert "def _start_ui_event_bridge_task(self, bridge: UIEventBridgePort)" in source


def test_every_ui_application_member_is_classified_as_guarded_intent_or_safe_operation() -> None:
    contract = _contract_members(UiApplicationPort)

    assert UI_APPLICATION_USER_INTENT_METHODS.isdisjoint(UI_APPLICATION_NON_INTENT_MEMBERS)
    assert contract == UI_APPLICATION_USER_INTENT_METHODS | UI_APPLICATION_NON_INTENT_MEMBERS
    assert all(
        hasattr(getattr(UiApplicationBoundary, name), "__wrapped__")
        for name in UI_APPLICATION_USER_INTENT_METHODS
    )


def test_production_gui_constructor_wires_one_explicit_boundary_in_each_direction() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    translator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TranslatorApp"
    )
    initializer = next(
        node
        for node in translator.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    main_gui = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "main_gui"
    )
    initializer_source = ast.get_source_segment(source, initializer)
    main_gui_source = ast.get_source_segment(source, main_gui)

    assert initializer_source.count("FletUiPresentationAdapter(self)") == 1
    assert initializer_source.count("application_factory(") == 1
    assert "presentation=self._presentation_adapter" in initializer_source
    assert "page=" not in initializer_source.split("application_factory(", 1)[1].split(")", 1)[0]
    assert "from puripuly_heart.ui.controller import GuiController" not in source
    assert not any(
        isinstance(node, ast.Name) and node.id == "GuiController" for node in ast.walk(tree)
    )
    assert 'application = getattr(app, "application", None)' in main_gui_source
    assert "await application.start()" in main_gui_source
    assert 'UiApplicationBoundary(getattr(app, "controller", None))' not in main_gui_source


def test_translator_app_has_no_operational_controller_or_hub_reach_through() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    direct_controller_accesses: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in {"__init__", "application"}:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and child.attr == "controller"
                and isinstance(child.ctx, ast.Load)
            ):
                direct_controller_accesses.append((node.name, child.lineno))

    source = APP_PATH.read_text(encoding="utf-8")
    assert direct_controller_accesses == []
    assert ".hub" not in source
    assert "getattr(self.application" not in source


def test_translator_app_imports_only_the_approved_backend_boundary_surface() -> None:
    imports = _imports(APP_PATH)
    forbidden_prefixes = (
        "puripuly_heart.app.wiring",
        "puripuly_heart.config.settings",
        "puripuly_heart.core.orchestrator",
        "puripuly_heart.core.runtime",
        "puripuly_heart.core.runtime_logging",
        "puripuly_heart.core.storage",
        "puripuly_heart.providers",
    )

    assert not any(module.startswith(forbidden_prefixes) for module in imports)
    assert "puripuly_heart.app.ports.ui_application" in imports
    assert "puripuly_heart.app.services.ui_application" in imports


def test_controller_presentation_access_is_explicit_and_adapter_is_closed() -> None:
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    accessed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and node.value.attr == "app"
        ):
            accessed.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "self"
            and node.args[0].attr == "app"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            accessed.add(node.args[1].value)

    assert accessed <= _contract_members(UiPresentationPort)
    assert _contract_members(UiPresentationPort) <= _contract_members(FletUiPresentationAdapter)
    assert "__getattr__" not in FletUiPresentationAdapter.__dict__
    assert FletUiPresentationAdapter.__annotations__["_app"] == "UiPresentationPort"
    assert not hasattr(FletUiPresentationAdapter, "app")


def test_application_runtime_has_no_flet_or_ui_module_dependencies() -> None:
    imports = _imports(CONTROLLER_PATH)

    assert "flet" not in imports
    assert not any(module.startswith("puripuly_heart.ui") for module in imports)
