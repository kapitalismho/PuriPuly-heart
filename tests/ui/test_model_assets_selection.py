from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import flet as ft
import pytest

from puripuly_heart.ui.components.model_assets_dialog import ModelAssetsDialog
from puripuly_heart.ui.i18n import t


@dataclass(frozen=True)
class Entry:
    model_id: str
    name: str
    description_key: str
    purpose: str
    size_bytes: int
    status: str = "missing"
    progress_percent: int | None = None
    failure_code: str | None = None
    importable: bool = False
    required_files: tuple[str, ...] = ()


class Page:
    def __init__(self) -> None:
        self.shown: list[object] = []
        self.popped = 0

    def show_dialog(self, dialog) -> None:
        self.shown.append(dialog)

    def pop_dialog(self) -> None:
        self.popped += 1


class Application:
    def __init__(self) -> None:
        self.entries = [
            Entry("asr", "Qwen CPU", "model_assets.description.qwen_cpu", "asr", 2 * 1024**3),
            Entry(
                "gemma",
                "Gemma E4B",
                "model_assets.description.gemma_e4b",
                "translation",
                4 * 1024**3,
                importable=True,
                required_files=("model.gguf", "draft.gguf"),
            ),
        ]
        self.calls: list[tuple] = []

    def model_asset_entries(self):
        return tuple(self.entries)

    async def refresh_model_assets(self):
        self.calls.append(("refresh",))
        return tuple(self.entries)

    async def download_model_asset(self, model_id, *, source=None):
        self.calls.append(("download", model_id, source))

    async def import_model_asset(self, model_id, directory):
        self.calls.append(("import", model_id, directory))

    async def cancel_model_asset(self, model_id):
        self.calls.append(("cancel", model_id))
        return True

    async def delete_model_asset(self, model_id):
        self.calls.append(("delete", model_id))

    def set(self, model_id: str, **changes) -> None:
        self.entries = [
            replace(entry, **changes) if entry.model_id == model_id else entry
            for entry in self.entries
        ]


def _texts(control) -> list[str]:
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, ft.Text) and node.value and getattr(node, "visible", True):
            found.append(node.value)
        for name in ("content", "controls"):
            child = getattr(node, name, None)
            if isinstance(child, list):
                for item in child:
                    walk(item)
            elif child is not None and hasattr(child, "__dict__"):
                walk(child)

    walk(control)
    return found


def _action(dialog: ModelAssetsDialog, label: str) -> ft.Container:
    for card in dialog._actions.controls:
        if label in _texts(card):
            return card
    raise AssertionError(f"missing action {label}: {_texts(dialog._actions)}")


def _dialog(app: Application, tasks: list, **kwargs) -> ModelAssetsDialog:
    return ModelAssetsDialog(Page(), app, tasks.append, **kwargs)


async def _run(tasks: list) -> None:
    while tasks:
        task = tasks.pop(0)
        if task.__name__ == "_refresh_and_watch":
            continue
        await task()


@pytest.mark.asyncio
async def test_modal_lists_models_by_purpose_and_shows_actions_for_the_selection() -> None:
    app = Application()
    tasks: list = []
    dialog = _dialog(app, tasks)

    dialog.open()

    texts = _texts(dialog.dialog.content)
    assert t("settings.model_assets") in texts
    assert t("model_assets.group.asr") in texts
    assert t("model_assets.group.translation") in texts
    assert f"2.0 GB · {t('model_assets.status.missing')}" in texts
    assert dialog.selected_model_id == "asr"
    assert t("model_assets.description.qwen_cpu") in texts
    assert texts.count("Qwen CPU") == 1
    assert [label for card in dialog._actions.controls for label in _texts(card)] == [
        t("model_assets.action.download")
    ]

    dialog.select("gemma")
    actions = _texts(dialog._actions)
    assert t("model_assets.action.import") in actions
    assert t("model_assets.import_files", files="model.gguf, draft.gguf") in actions


@pytest.mark.asyncio
async def test_download_and_two_step_delete_call_the_application() -> None:
    app = Application()
    app.set("asr", status="installed")
    tasks: list = []
    dialog = _dialog(app, tasks)
    dialog.open()

    _action(dialog, t("model_assets.action.redownload")).on_click(None)
    await _run(tasks)
    assert ("download", "asr", None) in app.calls

    _action(dialog, t("model_assets.action.delete")).on_click(None)
    await _run(tasks)
    assert not any(call[0] == "delete" for call in app.calls)
    _action(dialog, t("model_assets.action.delete_confirm")).on_click(None)
    await _run(tasks)
    assert ("delete", "asr") in app.calls


@pytest.mark.asyncio
async def test_progress_replaces_actions_and_other_models_wait_for_the_running_task() -> None:
    app = Application()
    app.set("asr", status="downloading", progress_percent=42)
    tasks: list = []
    dialog = _dialog(app, tasks)
    dialog.open()

    assert t("model_assets.status.downloading", percent=42) in _texts(dialog._actions)
    _action(dialog, t("model_assets.action.cancel")).on_click(None)
    await _run(tasks)
    assert ("cancel", "asr") in app.calls

    dialog.select("gemma")
    download = _action(dialog, t("model_assets.action.download"))
    assert download.disabled is True
    assert t("model_assets.busy_other") in _texts(dialog._actions)


@pytest.mark.asyncio
async def test_failure_is_shown_with_a_known_reason_or_the_generic_one() -> None:
    app = Application()
    app.set("asr", failure_code="model_in_use")
    dialog = _dialog(app, [])
    dialog.open()
    assert dialog._failure.value == t("model_assets.error.model_in_use")
    assert dialog._failure.visible is True

    app.set("asr", failure_code="mirror_redirect_external")
    dialog.render()
    assert dialog._failure.value == t("model_assets.error.download_failed")


@pytest.mark.asyncio
async def test_import_picks_a_folder_and_closing_never_cancels_work() -> None:
    app = Application()
    tasks: list = []

    class Picker:
        async def get_directory_path(self):
            return "C:/models"

    dialog = _dialog(app, tasks, picker_factory=Picker)
    dialog.open()
    dialog.select("gemma")

    _action(dialog, t("model_assets.action.import")).on_click(None)
    await _run(tasks)
    assert ("import", "gemma", Path("C:/models")) in app.calls

    dialog.close()
    assert dialog.is_open is False
    assert not any(call[0] == "cancel" for call in app.calls)
