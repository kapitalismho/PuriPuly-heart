from __future__ import annotations

import pytest

from puripuly_heart.app.ports.ui_models import ManagedGemmaDashboardNotice
from puripuly_heart.ui.dashboard import capture_notices
from puripuly_heart.ui.dashboard.capture_notices import (
    GPU_NOTICE_KEYS,
    LOCAL_ASR_NOTICE_KEYS,
    MANAGED_GEMMA_NOTICE_KEYS,
    gpu_capture_action_label,
    gpu_capture_notice,
    local_asr_capture_notice,
    managed_gemma_action_label,
    managed_gemma_capture_notice,
)
from puripuly_heart.ui.gpu_notice import GpuDashboardNotice


@pytest.fixture(autouse=True)
def _stub_i18n(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_t(key, *, default=None, **kwargs):
        if not key:
            return default if default is not None else ""
        rendered = f"i18n:{key}"
        if kwargs:
            rendered += ":" + ",".join(f"{name}={value}" for name, value in sorted(kwargs.items()))
        return rendered

    monkeypatch.setattr(capture_notices, "t", fake_t)


def test_no_local_asr_notice_without_status_or_for_unknown_status() -> None:
    assert local_asr_capture_notice(status=None) is None
    assert local_asr_capture_notice(status="something-else") is None


@pytest.mark.parametrize("status", sorted(LOCAL_ASR_NOTICE_KEYS))
def test_every_local_asr_status_renders_text_and_tone(status: str) -> None:
    notice = local_asr_capture_notice(status=status)

    assert notice is not None
    assert notice.text == f"i18n:{LOCAL_ASR_NOTICE_KEYS[status]}"
    assert notice.tone in {"info", "warning", "error"}
    assert notice.action is None


def test_local_asr_download_progress_uses_percent_variants() -> None:
    generic = local_asr_capture_notice(status="downloading", percent=42)
    assert generic is not None
    assert generic.text == "i18n:dashboard.local_stt_notice_downloading_progress:percent=42"

    targeted = local_asr_capture_notice(
        status="downloading", percent=42, model_id="qwen3-asr-0.6b-int8-sherpa"
    )
    assert targeted is not None
    assert targeted.text == (
        "i18n:dashboard.local_stt_notice_downloading_progress_model:"
        "model=i18n:local_stt.model.qwen3-asr-0.6b-int8-sherpa,percent=42"
    )

    without_percent = local_asr_capture_notice(
        status="downloading", model_id="qwen3-asr-0.6b-int8-sherpa"
    )
    assert without_percent is not None
    assert without_percent.text.startswith("i18n:dashboard.local_stt_notice_downloading_model:")


def test_local_asr_unknown_model_falls_back_to_the_model_id() -> None:
    notice = local_asr_capture_notice(status="missing", model_id="mystery-model")

    assert notice is not None
    assert notice.text == "i18n:dashboard.local_stt_notice_missing_model:model=mystery-model"


def test_local_asr_model_is_ignored_for_non_model_statuses() -> None:
    notice = local_asr_capture_notice(status="starting", model_id="qwen3-asr-0.6b-int8-sherpa")

    assert notice is not None
    assert notice.text == "i18n:dashboard.local_stt_notice_starting"


def test_no_gpu_notice_without_notice_or_for_unknown_status() -> None:
    assert gpu_capture_notice(None) is None
    assert gpu_capture_notice(GpuDashboardNotice(status="ready")) is None


@pytest.mark.parametrize("status", sorted(GPU_NOTICE_KEYS))
def test_every_gpu_status_renders_text_tone_and_action(status: str) -> None:
    notice = gpu_capture_notice(GpuDashboardNotice(status=status, action="install"))

    assert notice is not None
    assert notice.tone in {"info", "warning", "error"}
    assert notice.action == "install"
    if status == "installing":
        assert notice.text == "i18n:dashboard.gpu_notice.installing:percent=0"
        assert notice.tone == "info"
    else:
        assert notice.text == f"i18n:{GPU_NOTICE_KEYS[status]}"


def test_gpu_install_progress_renders_percent() -> None:
    notice = gpu_capture_notice(GpuDashboardNotice(status="installing", progress_percent=77))

    assert notice is not None
    assert notice.text == "i18n:dashboard.gpu_notice.installing:percent=77"


def test_gpu_action_label_is_localized_only_when_an_action_exists() -> None:
    assert gpu_capture_action_label(None) is None
    assert gpu_capture_action_label("repair") == "i18n:dashboard.gpu_action.repair"


@pytest.mark.parametrize("status", sorted(MANAGED_GEMMA_NOTICE_KEYS))
def test_every_managed_gemma_status_renders_text_and_tone(status: str) -> None:
    notice = managed_gemma_capture_notice(
        ManagedGemmaDashboardNotice(status=status, progress_percent=42)
    )

    assert notice is not None
    assert notice.tone in {"info", "warning", "error"}
    if status == "downloading":
        assert notice.text == ("i18n:dashboard.managed_gemma_notice.downloading:percent=42")
    else:
        assert notice.text == f"i18n:{MANAGED_GEMMA_NOTICE_KEYS[status]}"


def test_managed_gemma_notice_actions_are_localized() -> None:
    assert managed_gemma_action_label(None) is None
    assert managed_gemma_action_label("cancel") == ("i18n:dashboard.managed_gemma_action.cancel")
