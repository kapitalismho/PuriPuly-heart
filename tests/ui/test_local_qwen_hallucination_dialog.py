from __future__ import annotations

# ruff: noqa: I001

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

import flet as ft  # noqa: E402

from puripuly_heart.ui.theme import (  # noqa: E402
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
)


LOCAL_QWEN_GUIDANCE_KEYS = {
    "local_qwen_hallucination.body",
    "local_qwen_hallucination.open_guide",
    "local_qwen_hallucination.close",
    "debug_preview.local_qwen_hallucination_modal",
}

EXPECTED_EN_BODY = (
    "The local speech recognition model isn't working right now.\n"
    "Please switch to a cloud speech recognition service.\n\n"
    "We recommend Deepgram. You can use it for free with welcome credits.\n"
    "Open the GitHub guide and follow the setup instructions."
)


class FakePage:
    def __init__(self) -> None:
        self.dialog = None
        self.opened: list[object] = []
        self.closed: list[object] = []

    def open(self, dialog) -> None:
        self.dialog = dialog
        self.opened.append(dialog)

    def close(self, dialog) -> None:
        self.closed.append(dialog)
        if self.dialog is dialog:
            self.dialog = None


def _dialog_module():
    try:
        return importlib.import_module(
            "puripuly_heart.ui.components.local_qwen_hallucination_dialog"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"production Local Qwen guidance dialog module is missing: {exc}")


def _unwrap_glow_content(dialog: ft.AlertDialog):
    stack = dialog.content
    assert stack.__class__.__name__ == "Stack"
    assert len(stack.controls) == 2
    foreground = stack.controls[1]
    return foreground.content


def _content_column(dialog: ft.AlertDialog):
    return _unwrap_glow_content(dialog).content


def _body_column(dialog: ft.AlertDialog):
    for control in _content_column(dialog).controls:
        if control.__class__.__name__ == "Column":
            return control
    raise AssertionError("dialog content did not include a body column")


def _action_row(dialog: ft.AlertDialog):
    return _content_column(dialog).controls[-1]


def test_local_qwen_guidance_dialog_delegates_to_warm_document_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog_module = _dialog_module()
    captured: dict[str, object] = {}
    requested_keys: list[str] = []

    def fake_t(key: str) -> str:
        requested_keys.append(key)
        return {
            "local_qwen_hallucination.body": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "local_qwen_hallucination.open_guide": "Open guide",
            "local_qwen_hallucination.close": "Close",
        }[key]

    def fake_open_warm_document_dialog(page, **kwargs):
        captured["page"] = page
        captured.update(kwargs)
        return SimpleNamespace(dialog="opened-dialog")

    monkeypatch.setattr(dialog_module, "t", fake_t)
    monkeypatch.setattr(
        dialog_module,
        "open_warm_document_dialog",
        fake_open_warm_document_dialog,
    )

    page = FakePage()
    dialog = dialog_module.LocalQwenHallucinationDialog(
        page,
        on_open_guide=lambda: None,
        on_close=lambda: None,
    )

    dialog.open()

    assert captured["page"] is page
    assert captured["body_paragraphs"] == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]
    assert captured["primary_label"] == "Open guide"
    assert captured["secondary_label"] == "Close"
    assert getattr(captured["primary_action"], "__self__", None) is dialog
    assert getattr(captured["primary_action"], "__func__", None) is getattr(
        dialog._on_open_guide,
        "__func__",
        None,
    )
    assert getattr(captured["secondary_action"], "__self__", None) is dialog
    assert getattr(captured["secondary_action"], "__func__", None) is getattr(
        dialog._on_close,
        "__func__",
        None,
    )
    assert captured["glow_factory"] is dialog_module.create_glow_stack
    assert requested_keys == [
        "local_qwen_hallucination.body",
        "local_qwen_hallucination.open_guide",
        "local_qwen_hallucination.close",
    ]
    assert not any(key.endswith(".title") for key in requested_keys)
    assert not any("warning" in key for key in requested_keys)


def test_local_qwen_guidance_dialog_renders_large_readable_two_action_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog_module = _dialog_module()
    monkeypatch.setattr(
        dialog_module,
        "t",
        lambda key: {
            "local_qwen_hallucination.body": EXPECTED_EN_BODY,
            "local_qwen_hallucination.open_guide": "Open guide",
            "local_qwen_hallucination.close": "Close",
        }[key],
    )
    page = FakePage()
    events: list[str] = []

    dialog = dialog_module.LocalQwenHallucinationDialog(
        page,
        on_open_guide=lambda: events.append("guide"),
        on_close=lambda: events.append("close"),
    )

    dialog.open()

    opened_dialog = page.opened[-1]
    assert opened_dialog is dialog._dialog
    assert getattr(opened_dialog, "title", None) is None
    assert getattr(opened_dialog, "actions", None) in (None, [])

    modal_content = _unwrap_glow_content(opened_dialog)
    assert modal_content.width == 720
    assert modal_content.height is None

    body_column = _body_column(opened_dialog)
    body_text = body_column.controls[0]
    assert len(body_column.controls) == 1
    assert body_text.value == EXPECTED_EN_BODY
    assert body_text.size == 24
    assert body_text.selectable is True

    action_row = _action_row(opened_dialog)
    assert [button.__class__.__name__ for button in action_row.controls] == [
        "TextButton",
        "TextButton",
    ]
    assert [button.content for button in action_row.controls] == [
        "Close",
        "Open guide",
    ]
    assert [button.style.color[ft.ControlState.DEFAULT] for button in action_row.controls] == [
        COLOR_NEUTRAL_DARK,
        COLOR_NEUTRAL_DARK,
    ]
    assert [button.style.color[ft.ControlState.HOVERED] for button in action_row.controls] == [
        COLOR_PRIMARY,
        COLOR_PRIMARY,
    ]
    assert [button.style.animation_duration for button in action_row.controls] == [0, 0]

    action_row.controls[1].on_click(None)

    assert events == ["guide"]
    assert page.closed == [opened_dialog]


def test_local_qwen_guidance_i18n_keys_exist_in_all_locale_bundles() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    i18n_dir = repo_root / "src" / "puripuly_heart" / "data" / "i18n"

    for locale in ("en.json", "ko.json", "ja.json", "zh-CN.json"):
        bundle = json.loads((i18n_dir / locale).read_text(encoding="utf-8"))
        missing = LOCAL_QWEN_GUIDANCE_KEYS - set(bundle)
        assert not missing, f"{locale} missing {sorted(missing)}"
        for key in LOCAL_QWEN_GUIDANCE_KEYS:
            assert isinstance(bundle[key], str)
            assert bundle[key].strip(), f"{locale} has blank {key}"

    english = json.loads((i18n_dir / "en.json").read_text(encoding="utf-8"))
    assert english["local_qwen_hallucination.body"] == EXPECTED_EN_BODY
    assert english["local_qwen_hallucination.open_guide"] == "Open guide"
    assert english["local_qwen_hallucination.close"] == "Close"
    assert english["debug_preview.local_qwen_hallucination_modal"] == "Local Qwen warning"
    assert "welcome credits" in english["local_qwen_hallucination.body"]
