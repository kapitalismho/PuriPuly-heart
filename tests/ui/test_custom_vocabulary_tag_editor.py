from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

ft = pytest.importorskip("flet")


def _editor_class():
    return _editor_module().CustomVocabularyTagEditor


def _editor_module():
    try:
        module = import_module("puripuly_heart.ui.components.settings.custom_vocabulary_tag_editor")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in RED run
        pytest.fail(f"CustomVocabularyTagEditor module is missing: {exc}")

    try:
        module.CustomVocabularyTagEditor
    except AttributeError:  # pragma: no cover - exercised in RED run
        pytest.fail("CustomVocabularyTagEditor class is missing")
    return module


def _make_editor():
    return _editor_class()()


def _chip_controls(editor) -> list[ft.Container]:
    return list(editor._chips_wrap.controls)


def _chip_term_text(chip: ft.Container) -> ft.Text:
    term_text = chip.content
    assert isinstance(term_text, ft.Text)
    return term_text


def _iter_controls(control: ft.Control):
    yield control
    content = getattr(control, "content", None)
    if content is not None:
        yield from _iter_controls(content)
    for child in getattr(control, "controls", []) or []:
        yield from _iter_controls(child)


def test_set_terms_renders_compact_chips_above_native_input() -> None:
    editor = _make_editor()
    module = _editor_module()

    editor.set_terms(["Puripuly", "VRChat"])

    chips = _chip_controls(editor)
    assert editor._chips_wrap.wrap is True
    assert editor._empty_text.visible is False
    assert len(chips) == 2
    assert [_chip_term_text(chip).value for chip in chips] == ["Puripuly", "VRChat"]
    assert [chip.data for chip in chips] == ["Puripuly", "VRChat"]
    assert editor._chips_wrap.visible is True
    assert list(editor.controls) == [editor._chips_wrap, editor._input_field]
    assert editor._input_field not in editor._chips_wrap.controls
    assert not hasattr(editor, "_token_wrap")
    assert not hasattr(editor, "_token_field")
    assert _chip_term_text(chips[0]).width is None
    assert _chip_term_text(chips[0]).size == module._CHIP_TEXT_SIZE
    assert chips[0].padding.left == module._CHIP_HORIZONTAL_PADDING
    assert chips[0].padding.right == module._CHIP_HORIZONTAL_PADDING
    assert chips[0].padding.top == module._CHIP_VERTICAL_PADDING
    assert chips[0].padding.bottom == module._CHIP_VERTICAL_PADDING
    assert callable(chips[0].on_click)
    assert callable(chips[0].on_hover)
    assert chips[0].tooltip is None
    assert _chip_term_text(chips[0]).tooltip is None
    assert not any(isinstance(node, ft.IconButton) for node in _iter_controls(editor))


def test_empty_terms_show_only_native_input_without_placeholder_help() -> None:
    editor = _make_editor()
    editor.set_empty_text("No hints yet.")
    editor.set_placeholder("Type hint, then Space")

    editor.set_terms([])

    assert _chip_controls(editor) == []
    assert editor._empty_text.value == "No hints yet."
    assert editor._empty_text.visible is False
    assert editor._chips_wrap.visible is False
    assert list(editor.controls) == [editor._input_field]
    assert editor._input_field.hint_text == ""


def test_input_uses_native_focus_styling_without_outer_token_field() -> None:
    editor = _make_editor()

    assert not hasattr(editor, "_token_field")
    assert editor._input_field.border != ft.InputBorder.NONE  # noqa: SLF001
    assert editor._input_field.focus_color != ft.Colors.TRANSPARENT  # noqa: SLF001
    assert editor._input_field.focused_bgcolor != ft.Colors.TRANSPARENT  # noqa: SLF001
    assert editor._input_field.focused_border_color != ft.Colors.TRANSPARENT  # noqa: SLF001


def test_input_size_matches_api_key_fields() -> None:
    editor = _make_editor()
    module = _editor_module()

    assert editor._input_field.border_radius == module._INPUT_FIELD_RADIUS  # noqa: SLF001
    assert editor._input_field.dense is not True  # noqa: SLF001
    assert editor._input_field.content_padding is None  # noqa: SLF001


def test_space_delimited_input_change_commits_tokens_and_clears_input() -> None:
    editor = _make_editor()
    added: list[list[str]] = []
    editor.on_add_terms = added.append
    editor._input_field.value = "  Puripuly VRChat\nSoniox  "

    editor._input_field.on_change(None)

    assert added == [["Puripuly", "VRChat", "Soniox"]]
    assert editor._input_field.value == ""


def test_input_change_waits_for_separator_before_committing_token() -> None:
    editor = _make_editor()
    added: list[list[str]] = []
    editor.on_add_terms = added.append
    editor._input_field.value = "Puripuly"

    editor._input_field.on_change(None)

    assert added == []
    assert editor._input_field.value == "Puripuly"


def test_submit_commits_current_token_without_trailing_space() -> None:
    editor = _make_editor()
    added: list[list[str]] = []
    editor.on_add_terms = added.append
    editor._input_field.value = "Puripuly"

    editor._input_field.on_submit(None)

    assert added == [["Puripuly"]]
    assert editor._input_field.value == ""


def test_blur_commits_current_token_for_tab_navigation() -> None:
    editor = _make_editor()
    added: list[list[str]] = []
    editor.on_add_terms = added.append
    editor._input_field.value = "VRChat"

    editor._input_field.on_blur(None)

    assert added == [["VRChat"]]
    assert editor._input_field.value == ""


def test_token_input_preserves_input_when_no_add_callback_is_installed() -> None:
    editor = _make_editor()
    editor._input_field.value = "Puripuly VRChat "

    editor._input_field.on_change(None)

    assert editor._input_field.value == "Puripuly VRChat "


def test_token_input_ignores_empty_whitespace_segments() -> None:
    editor = _make_editor()
    added: list[list[str]] = []
    editor.on_add_terms = added.append
    editor._input_field.value = "Puripuly  VRChat\n\n Soniox "

    editor._input_field.on_change(None)

    assert added == [["Puripuly", "VRChat", "Soniox"]]


def test_clicking_chip_calls_remove_callback_with_visible_term() -> None:
    editor = _make_editor()
    removed: list[str] = []
    editor.on_remove_term = removed.append
    editor.set_remove_label_template("Remove {term}")
    editor.set_terms(["Puripuly"])

    chip = _chip_controls(editor)[0]
    chip.on_click(None)

    assert removed == ["Puripuly"]
    assert chip.tooltip is None
    assert not any(isinstance(node, ft.IconButton) for node in _iter_controls(chip))


def test_chip_hover_matches_dashboard_active_button_colors() -> None:
    module = _editor_module()
    editor = _make_editor()
    editor.set_terms(["Puripuly"])
    chip = _chip_controls(editor)[0]
    term_text = _chip_term_text(chip)

    chip.on_hover(SimpleNamespace(data="true", control=chip))

    assert chip.bgcolor == module.COLOR_PRIMARY
    assert chip.border.top.color == module.COLOR_PRIMARY
    assert term_text.color == ft.Colors.WHITE

    chip.on_hover(SimpleNamespace(data="false", control=chip))

    assert chip.bgcolor == module.COLOR_PRIMARY_CONTAINER
    assert chip.border.top.color == module.COLOR_DIVIDER
    assert term_text.color == module.COLOR_ON_PRIMARY_CONTAINER


def test_locale_setters_update_placeholder_empty_add_and_existing_remove_labels() -> None:
    editor = _make_editor()
    editor.set_terms(["Puripuly"])

    editor.set_placeholder("힌트 추가")
    editor.set_empty_text("아직 추가된 힌트가 없어요.")
    editor.set_add_label("추가")
    editor.set_remove_label_template("{term} 삭제")

    assert editor._input_field.hint_text == ""
    assert editor._empty_text.value == "아직 추가된 힌트가 없어요."
    assert _chip_controls(editor)[0].tooltip is None


def test_clear_input_clears_unsubmitted_add_text() -> None:
    editor = _make_editor()
    editor._input_field.value = "draft hint"

    editor.clear_input()

    assert editor._input_field.value == ""


def test_long_hint_text_is_constrained_without_hover_tooltips() -> None:
    editor = _make_editor()
    long_term = "A very long Speech Recognition Hint " * 8

    editor.set_terms([long_term])

    chip = _chip_controls(editor)[0]
    term_text = _chip_term_text(chip)

    assert chip.clip_behavior == ft.ClipBehavior.HARD_EDGE
    assert chip.tooltip is None
    assert term_text.value == long_term
    assert term_text.tooltip is None
    assert term_text.semantics_label == long_term
    assert term_text.width is not None
    assert term_text.width <= 240
    assert term_text.max_lines == 1
    assert term_text.overflow == ft.TextOverflow.ELLIPSIS


def test_component_is_exported_from_settings_components_package() -> None:
    editor_class = _editor_class()
    settings_components = import_module("puripuly_heart.ui.components.settings")

    assert settings_components.CustomVocabularyTagEditor is editor_class
    assert "CustomVocabularyTagEditor" in settings_components.__all__
