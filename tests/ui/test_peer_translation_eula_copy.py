from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("flet")

import puripuly_heart.ui.components.peer_translation_eula_dialog as dialog_module  # noqa: E402
from puripuly_heart.ui.components.peer_translation_eula_dialog import (  # noqa: E402
    PeerTranslationEulaDialog,
)

EXPECTED_EULA_BODIES = {
    "ko": (
        "PuriPuly는 상대방의 음성을 번역하기 위해 데스크톱 오디오를 루프백으로 캡처합니다.\n"
        "이는 타인의 동의 없이 음성이 처리되거나, 자신이 참여하지 않은 대화의 음성이 처리될 수 있음을 의미합니다.\n"
        "이러한 행위는 국가나 지역에 따라 관련 법률에 저촉될 수 있습니다.\n\n"
        "사용에 따른 법적 책임은 사용자에게 있습니다.\n"
        "가능한 한 자신이 참여한 대화에서, 상대방의 동의를 받은 상태로 사용해주세요.\n\n"
        "PuriPuly는 상대방의 오디오, 전사문, 번역 결과를 저장하지 않습니다.\n"
        "다만 설정된 외부 음성 인식 및 번역 제공자가 데이터를 처리할 수 있습니다."
    ),
    "zh-CN": (
        "PuriPuly 为了翻译对方的声音，会通过环回捕获桌面音频。\n"
        "这意味着，可能会在未经他人同意的情况下处理其声音，也可能会处理你并未参与的对话中的声音。\n"
        "此类行为可能因国家或地区不同而违反相关法律。\n\n"
        "使用产生的法律责任由用户自行承担。\n"
        "请尽可能只在自己参与的对话中，并在对方同意的情况下使用。\n\n"
        "PuriPuly 不会存储对方的音频、转写文本或翻译结果。\n"
        "但你配置的外部语音识别和翻译服务商可能会处理这些数据。"
    ),
    "en": (
        "PuriPuly captures desktop audio via loopback to translate the other person's voice.\n"
        "This means it may process someone's audio without their consent, or audio from conversations you are not part of.\n"
        "Depending on your region, this could violate local laws.\n\n"
        "You are legally responsible for how you use this tool.\n"
        "Whenever possible, please use it only in conversations you're actively participating in, and make sure you have the other person's consent.\n\n"
        "PuriPuly does not store the other person's audio, transcripts, or translation results.\n"
        "However, your configured external speech recognition and translation providers may process the data."
    ),
}

LEGACY_SPLIT_EULA_BODY_KEYS = {
    "peer_translation_eula.body.capture",
    "peer_translation_eula.body.responsibility",
    "peer_translation_eula.body.storage",
}


def test_peer_translation_eula_body_copy_is_localized_exactly() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    i18n_dir = repo_root / "src" / "puripuly_heart" / "data" / "i18n"

    for locale, expected_body in EXPECTED_EULA_BODIES.items():
        bundle = json.loads((i18n_dir / f"{locale}.json").read_text(encoding="utf-8"))
        assert bundle["peer_translation_eula.body"] == expected_body


def test_peer_translation_eula_body_copy_uses_single_body_key_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    i18n_dir = repo_root / "src" / "puripuly_heart" / "data" / "i18n"

    for locale in EXPECTED_EULA_BODIES:
        bundle = json.loads((i18n_dir / f"{locale}.json").read_text(encoding="utf-8"))
        unexpected = LEGACY_SPLIT_EULA_BODY_KEYS & set(bundle)
        assert not unexpected, f"{locale} still has split EULA keys: {sorted(unexpected)}"


def test_peer_translation_eula_dialog_renders_single_full_body_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_keys: list[str] = []

    def fake_t(key: str) -> str:
        requested_keys.append(key)
        return f"value:{key}"

    class FakePage:
        def __init__(self) -> None:
            self.dialog = None

        def show_dialog(self, dialog) -> None:
            self.dialog = dialog

        def pop_dialog(self):
            dialog = self.dialog
            self.dialog = None
            return dialog

    monkeypatch.setattr(dialog_module, "t", fake_t)
    monkeypatch.setattr(dialog_module, "create_glow_stack", lambda content: content)

    page = FakePage()
    PeerTranslationEulaDialog(page, on_accept=lambda: None).open()

    assert "peer_translation_eula.body" in requested_keys
    assert "peer_translation_eula.body.capture" not in requested_keys
    assert "peer_translation_eula.body.responsibility" not in requested_keys
    assert "peer_translation_eula.body.storage" not in requested_keys

    modal_content = page.dialog.content
    body_column = next(
        control
        for control in modal_content.content.controls
        if control.__class__.__name__ == "Column"
    )
    assert [control.value for control in body_column.controls] == [
        "value:peer_translation_eula.body"
    ]
