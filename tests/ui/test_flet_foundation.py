from __future__ import annotations

import ast
import asyncio
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import flet as ft
import pytest

import puripuly_heart.ui.app as app_module
from puripuly_heart.app.ports.ui_application import UiApplicationState
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCoordinator,
)
from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper
from puripuly_heart.ui.fonts import (
    FONT_FAMILY_NOTO_SANS,
    FONT_FAMILY_NOTO_SANS_CJK_JP,
    FONT_FAMILY_NOTO_SANS_CJK_KR,
    FONT_FAMILY_NOTO_SANS_CJK_SC,
    FONT_FAMILY_NOTO_SANS_CJK_TC,
    assets_dir,
    font_asset_path,
    noto_cjk_family_for_ui_locale,
    register_fonts,
)
from puripuly_heart.ui.foundation.adapter import FletFoundationAdapter
from puripuly_heart.ui.foundation.preview import (
    FoundationPreviewSurface,
    foundation_preview_copy,
)
from puripuly_heart.ui.foundation.primitives import FoundationCard
from puripuly_heart.ui.foundation.resources import (
    DEFAULT_FOUNDATION_RESOURCES,
    FoundationResourceLocator,
)
from puripuly_heart.ui.foundation.runtime import FletFoundationRuntime
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_DIVIDER,
    COLOR_ERROR,
    COLOR_NEUTRAL,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_WARNING,
)
from tests.helpers.paths import REPO_ROOT

FOUNDATION_ROOT = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "foundation"
LOCALES = ("en", "ko", "zh-CN", "ja", "ru")
FOUNDATION_I18N_KEYS = {
    "debug_preview.foundation_primitives",
    "foundation.preview.title",
    "foundation.preview.body",
    "foundation.preview.ready",
    "foundation.preview.action",
    "foundation.preview.unavailable",
}


class RecordingApplication:
    def __init__(self) -> None:
        self.state_calls = 0

    def state(self) -> UiApplicationState:
        self.state_calls += 1
        return UiApplicationState(
            config_path=Path("settings.json"),
            runtime_logging_mode="detailed",
            translation_enabled=True,
            stt_state=None,
            peer_translation_eula_accepted=False,
            microphone_test_active=False,
            provider_name="gemini",
            overlay_target="none",
            desktop_overlay_captions_locked=False,
            managed_auth_referral_bonus_applied=False,
        )


class RecordingPresentation:
    def __init__(self, *, debug_ui_preview: bool) -> None:
        self.debug_ui_preview = debug_ui_preview
        self.locale_calls = 0

    def apply_locale(self) -> None:
        self.locale_calls += 1


class TaskPage:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[object]] = []

    def run_task(self, coroutine, *args):
        task = asyncio.create_task(coroutine(*args))
        self.tasks.append(task)
        return task


def _baseline_source(relative_path: str) -> str:
    result = subprocess.run(
        [
            "git",
            "show",
            f"{FOUNDATION_DESIGN_TOKENS.accepted_production_revision}:{relative_path}",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def test_foundation_tokens_match_accepted_production_palette_and_fixed_delta() -> None:
    baseline_theme = _baseline_source("src/puripuly_heart/ui/theme.py")
    palette_constants = {
        "COLOR_BACKGROUND": FOUNDATION_DESIGN_TOKENS.palette.background,
        "COLOR_SURFACE": FOUNDATION_DESIGN_TOKENS.palette.surface,
        "COLOR_ON_BACKGROUND": FOUNDATION_DESIGN_TOKENS.palette.on_background,
        "COLOR_PRIMARY": FOUNDATION_DESIGN_TOKENS.palette.primary,
        "COLOR_ERROR": FOUNDATION_DESIGN_TOKENS.palette.error,
        "COLOR_SUCCESS": FOUNDATION_DESIGN_TOKENS.palette.success,
        "COLOR_WARNING": FOUNDATION_DESIGN_TOKENS.palette.warning,
        "COLOR_DIVIDER": FOUNDATION_DESIGN_TOKENS.palette.divider,
        "COLOR_SECONDARY": FOUNDATION_DESIGN_TOKENS.palette.secondary,
        "COLOR_NEUTRAL": FOUNDATION_DESIGN_TOKENS.palette.neutral,
    }

    for constant, value in palette_constants.items():
        assert f'{constant} = "{value}"' in baseline_theme

    assert (
        COLOR_BACKGROUND,
        COLOR_SURFACE,
        COLOR_ON_BACKGROUND,
        COLOR_PRIMARY,
        COLOR_ERROR,
        COLOR_SUCCESS,
        COLOR_WARNING,
        COLOR_DIVIDER,
        COLOR_SECONDARY,
        COLOR_NEUTRAL,
    ) == tuple(palette_constants.values())
    assert app_module.DEFAULT_WINDOW_WIDTH == 1136
    assert app_module.DEFAULT_WINDOW_HEIGHT == 850
    assert FOUNDATION_DESIGN_TOKENS.window.resizable is False
    assert FOUNDATION_DESIGN_TOKENS.window.maximizable is False


def test_foundation_card_reuses_accepted_shared_card_geometry() -> None:
    content = ft.Text("content")
    foundation = FoundationCard(content, width=400)
    accepted = SharedCardWrapper(content, expand=False, height=None, padding=24)

    assert foundation.bgcolor == accepted.bgcolor
    assert foundation.border_radius == accepted.border_radius == 16
    assert foundation.border == accepted.border
    assert foundation.shadow == accepted.shadow
    assert foundation.clip_behavior == accepted.clip_behavior
    assert foundation.width == 400
    assert foundation.content.padding == 24


def test_foundation_adapter_consumes_only_application_and_presentation_ports() -> None:
    application = RecordingApplication()
    presentation = RecordingPresentation(debug_ui_preview=True)
    adapter = FletFoundationAdapter(application, presentation)

    snapshot = adapter.snapshot()
    adapter.apply_locale()

    assert snapshot.config_path == Path("settings.json")
    assert snapshot.runtime_logging_mode == "detailed"
    assert snapshot.translation_enabled is True
    assert snapshot.debug_preview_enabled is True
    assert application.state_calls == 1
    assert presentation.locale_calls == 1
    assert "flet" not in inspect.getsource(app_module.UiApplicationPort).casefold()


@pytest.mark.asyncio
async def test_foundation_runtime_is_bound_to_application_shutdown_and_rejects_late_work() -> None:
    page = TaskPage()
    adapter = FletFoundationAdapter(
        RecordingApplication(),
        RecordingPresentation(debug_ui_preview=False),
    )
    runtime = FletFoundationRuntime(page, adapter)
    lifecycle = ApplicationShutdownCoordinator(runtime.application_shutdown_callbacks())
    runtime.bind_application_lifecycle(lifecycle)
    cancelled = asyncio.Event()

    async def work() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = runtime.run_page_task(work, intent_name="foundation-test-work")
    await asyncio.sleep(0)

    assert task is page.tasks[0]
    assert runtime.snapshot.accepting_tasks is True
    assert runtime.snapshot.tracked_task_count == 1

    snapshot = await lifecycle.shutdown()

    assert snapshot.state == "completed"
    assert cancelled.is_set()
    assert runtime.snapshot.close_completed is True
    assert runtime.snapshot.tracked_task_count == 0
    assert runtime.run_page_task(work, intent_name="late-foundation-test-work") is None
    assert len(page.tasks) == 1


def test_foundation_runtime_requires_one_explicit_lifecycle_owner() -> None:
    runtime = FletFoundationRuntime(
        TaskPage(),
        FletFoundationAdapter(
            RecordingApplication(),
            RecordingPresentation(debug_ui_preview=False),
        ),
    )
    first = ApplicationShutdownCoordinator()
    second = ApplicationShutdownCoordinator()

    with pytest.raises(RuntimeError, match="not bound"):
        runtime.run_page_task(lambda: None)

    runtime.bind_application_lifecycle(first)
    runtime.bind_application_lifecycle(first)

    with pytest.raises(RuntimeError, match="another lifecycle"):
        runtime.bind_application_lifecycle(second)


def test_foundation_resource_locator_is_cwd_independent_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    icon = DEFAULT_FOUNDATION_RESOURCES.require_file(FOUNDATION_DESIGN_TOKENS.icon_asset)

    assert icon.is_file()
    assert icon.parent.name == "icons"
    assert assets_dir() == DEFAULT_FOUNDATION_RESOURCES.assets_root
    assert font_asset_path("NanumSquareRound") == "/fonts/NanumSquareRoundEB.ttf"
    assert font_asset_path(FONT_FAMILY_NOTO_SANS) == "/fonts/NotoSansCJK-Medium.ttc"
    assert font_asset_path(FONT_FAMILY_NOTO_SANS_CJK_KR) == "/fonts/NotoSansCJK-Medium.ttc"
    assert font_asset_path(FONT_FAMILY_NOTO_SANS_CJK_JP) == "/fonts/NotoSansCJK-Medium.ttc"
    assert font_asset_path(FONT_FAMILY_NOTO_SANS_CJK_SC) == "/fonts/NotoSansCJK-Medium.ttc"
    assert font_asset_path(FONT_FAMILY_NOTO_SANS_CJK_TC) == "/fonts/NotoSansCJK-Medium.ttc"
    assert not {
        "write",
        "write_bytes",
        "write_text",
        "mkdir",
        "unlink",
        "rename",
        "replace",
    } & set(dir(FoundationResourceLocator))

    for unsafe in ("", "..", "../icon.ico", "/icons/icon.ico", r"C:\icons\icon.ico"):
        with pytest.raises(ValueError):
            DEFAULT_FOUNDATION_RESOURCES.asset_url(unsafe)


def test_register_fonts_maps_desktop_overlay_caption_families_to_bundled_noto_cjk() -> None:
    page = SimpleNamespace()

    register_fonts(page)

    assert page.fonts[FONT_FAMILY_NOTO_SANS] == "/fonts/NotoSansCJK-Medium.ttc"
    assert page.fonts[FONT_FAMILY_NOTO_SANS_CJK_KR] == "/fonts/NotoSansCJK-Medium.ttc"
    assert page.fonts[FONT_FAMILY_NOTO_SANS_CJK_JP] == "/fonts/NotoSansCJK-Medium.ttc"
    assert page.fonts[FONT_FAMILY_NOTO_SANS_CJK_SC] == "/fonts/NotoSansCJK-Medium.ttc"
    assert page.fonts[FONT_FAMILY_NOTO_SANS_CJK_TC] == "/fonts/NotoSansCJK-Medium.ttc"
    assert page.fonts["NanumSquareRound"] == "/fonts/NanumSquareRoundEB.ttf"


def test_noto_cjk_family_follows_ui_locale() -> None:
    assert noto_cjk_family_for_ui_locale("ko") == FONT_FAMILY_NOTO_SANS_CJK_KR
    assert noto_cjk_family_for_ui_locale("ja") == FONT_FAMILY_NOTO_SANS_CJK_JP
    assert noto_cjk_family_for_ui_locale("zh-CN") == FONT_FAMILY_NOTO_SANS_CJK_SC
    assert noto_cjk_family_for_ui_locale("en") == FONT_FAMILY_NOTO_SANS_CJK_JP
    assert noto_cjk_family_for_ui_locale("ru") == FONT_FAMILY_NOTO_SANS_CJK_JP


def test_foundation_preview_copy_has_distinct_inputs_for_all_five_locales() -> None:
    copies = {locale: foundation_preview_copy(locale) for locale in LOCALES}

    assert all(copy.title != "foundation.preview.title" for copy in copies.values())
    assert all(copy.body != "foundation.preview.body" for copy in copies.values())
    assert len({copy.title for copy in copies.values()}) == len(LOCALES)

    for locale in LOCALES:
        bundle_path = REPO_ROOT / "src" / "puripuly_heart" / "data" / "i18n" / f"{locale}.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert FOUNDATION_I18N_KEYS <= set(bundle)


def test_foundation_preview_surface_is_deterministic_and_static() -> None:
    first = FoundationPreviewSurface("en")
    second = FoundationPreviewSurface("en")

    assert first.copy == second.copy
    assert first.preview_card.bgcolor == COLOR_SURFACE
    row = first.preview_card.content.content.controls[2]
    assert row.controls[0].content.value == "Ready"
    assert row.controls[1].content == "Sample action"
    assert row.controls[2].content == "Unavailable"
    assert row.controls[2].disabled is True
    assert row.controls[1].on_click is None


def test_foundation_preview_action_is_hidden_without_flag_and_has_no_external_calls() -> None:
    app = app_module.TranslatorApp.__new__(app_module.TranslatorApp)
    app.page = SimpleNamespace(
        opened=[],
        show_dialog=lambda dialog: app.page.opened.append(dialog),
    )
    app._foundation_adapter = SimpleNamespace(debug_preview_enabled=False)

    app._preview_foundation_primitives()

    assert app.page.opened == []

    app._foundation_adapter = SimpleNamespace(debug_preview_enabled=True)
    app._preview_foundation_primitives()

    assert len(app.page.opened) == 1
    assert isinstance(app.page.opened[0].content, FoundationPreviewSurface)
    method_source = inspect.getsource(app_module.TranslatorApp._preview_foundation_primitives)
    assert ".show_dialog(dialog)" in method_source
    assert ".open(dialog)" not in method_source
    assert "self.application" not in method_source
    assert "self.controller" not in method_source


def test_foundation_modules_remain_below_the_ui_boundary_and_do_not_cut_over_views() -> None:
    forbidden_foundation_imports = (
        "puripuly_heart.config",
        "puripuly_heart.core.orchestrator",
        "puripuly_heart.providers",
    )
    for path in FOUNDATION_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        assert not any(module.startswith(forbidden_foundation_imports) for module in imports)

    for root_name in ("app", "config", "core", "providers"):
        root = REPO_ROOT / "src" / "puripuly_heart" / root_name
        for path in root.rglob("*.py"):
            assert "puripuly_heart.ui.foundation" not in path.read_text(encoding="utf-8")

    main_source = (REPO_ROOT / "src" / "puripuly_heart" / "main.py").read_text(encoding="utf-8")
    app_source = (REPO_ROOT / "src" / "puripuly_heart" / "ui" / "app.py").read_text(
        encoding="utf-8"
    )
    assert "from puripuly_heart.ui.app import main_gui" in main_source
    assert "self.view_dashboard = DashboardView()" in app_source
    assert "self.view_settings = SettingsView()" in app_source
    assert "self.view_logs = LogsView()" in app_source
    assert "self.view_about = AboutView()" in app_source
