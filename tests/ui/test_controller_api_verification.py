from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui import app as app_module
from puripuly_heart.ui import i18n as i18n_module


def test_action_snackbar_helper_removed_from_app_source() -> None:
    app_source = Path(app_module.__file__).read_text(encoding="utf-8")

    assert "def show_action_snackbar(" not in app_source


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN"])
def test_obsolete_local_stt_prompt_keys_are_removed(locale: str) -> None:
    bundle = i18n_module._load_bundle(locale)

    assert "local_stt.download_prompt_missing" not in bundle
    assert "local_stt.download_prompt_invalid" not in bundle
    assert "local_stt.download_prompt_failed" not in bundle
    assert "local_stt.download_action" not in bundle
