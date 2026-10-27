from __future__ import annotations

import pytest

from puripuly_heart.app.ports.ui_models import ManagedGemmaDashboardNotice
from puripuly_heart.ui.dashboard import capture_notices
from puripuly_heart.ui.dashboard.capture_notices import (
    gpu_capture_action_label,
    gpu_capture_notice,
    local_asr_capture_notice,
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


@pytest.mark.parametrize(
    ("status", "expected_text", "tone"),
    [
        ("self_loading", "i18n:dashboard.local_stt_notice_self_loading", "info"),
        ("peer_loading", "i18n:dashboard.local_stt_notice_peer_loading", "info"),
        ("start_failed", "i18n:dashboard.local_stt_notice_start_failed", "error"),
        ("missing", "i18n:dashboard.local_stt_notice_missing_model:model=", "warning"),
        ("invalid", "i18n:dashboard.local_stt_notice_invalid_model:model=", "warning"),
        ("downloading", "i18n:dashboard.local_stt_notice_downloading_model:model=", "info"),
        (
            "download_failed",
            "i18n:dashboard.local_stt_notice_download_failed_model:model=",
            "error",
        ),
    ],
)
def test_local_asr_status_renders_text_and_tone(status: str, expected_text: str, tone: str) -> None:
    notice = local_asr_capture_notice(status=status)

    assert notice is not None
    assert notice.text == expected_text
    assert notice.tone == tone
    assert notice.action is None


def test_local_asr_download_progress_uses_percent_variants() -> None:
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
    notice = local_asr_capture_notice(status="self_loading", model_id="qwen3-asr-0.6b-int8-sherpa")

    assert notice is not None
    assert notice.text == "i18n:dashboard.local_stt_notice_self_loading"


def test_no_gpu_notice_without_notice_or_for_unknown_status() -> None:
    assert gpu_capture_notice(None) is None
    assert gpu_capture_notice(GpuDashboardNotice(status="ready")) is None


@pytest.mark.parametrize(
    ("status", "expected_text", "tone"),
    [
        ("discovery_failed", "i18n:dashboard.gpu_notice.discovery_failed", "error"),
        ("unsupported", "i18n:dashboard.gpu_notice.unsupported", "warning"),
        ("unavailable_device", "i18n:dashboard.gpu_notice.unavailable_device", "warning"),
        ("not_installed", "i18n:dashboard.gpu_notice.not_installed", "warning"),
        ("invalid", "i18n:dashboard.gpu_notice.invalid", "warning"),
        ("installing", "i18n:dashboard.gpu_notice.installing:percent=0", "info"),
        ("install_failed", "i18n:dashboard.gpu_notice.install_failed", "error"),
        ("activation_failed", "i18n:dashboard.gpu_notice.activation_failed", "error"),
    ],
)
def test_gpu_status_renders_text_tone_and_action(
    status: str, expected_text: str, tone: str
) -> None:
    notice = gpu_capture_notice(GpuDashboardNotice(status=status, action="install"))

    assert notice is not None
    assert notice.text == expected_text
    assert notice.tone == tone
    assert notice.action == "install"


def test_gpu_install_progress_renders_percent() -> None:
    notice = gpu_capture_notice(GpuDashboardNotice(status="installing", progress_percent=77))

    assert notice is not None
    assert notice.text == "i18n:dashboard.gpu_notice.installing:percent=77"


def test_gpu_action_label_is_localized_only_when_an_action_exists() -> None:
    assert gpu_capture_action_label(None) is None
    assert gpu_capture_action_label("restart") == "i18n:dashboard.gpu_action.restart"
    assert gpu_capture_action_label("install") is None


@pytest.mark.parametrize(
    ("status", "expected_text", "tone"),
    [
        ("checking", "i18n:dashboard.managed_gemma_notice.checking", "info"),
        ("downloading", "i18n:dashboard.managed_gemma_notice.downloading:percent=42", "info"),
        ("preparing", "i18n:dashboard.managed_gemma_notice.preparing", "info"),
        ("failed", "i18n:dashboard.managed_gemma_notice.failed", "error"),
        ("cancelled", "i18n:dashboard.managed_gemma_notice.cancelled", "warning"),
    ],
)
def test_managed_gemma_status_renders_text_and_tone(
    status: str, expected_text: str, tone: str
) -> None:
    notice = managed_gemma_capture_notice(
        ManagedGemmaDashboardNotice(status=status, progress_percent=42)
    )

    assert notice is not None
    assert notice.text == expected_text
    assert notice.tone == tone
    assert notice.action is None


@pytest.mark.parametrize(
    ("gpu_only", "phase", "key"),
    [
        (True, "runtime", "dashboard.managed_gemma_notice.gpu_only_failed"),
        (False, "runtime", "dashboard.managed_gemma_notice.failed"),
        (True, "download", "dashboard.managed_gemma_notice.failed"),
        (False, "download", "dashboard.managed_gemma_notice.failed"),
        (True, "runtime_files", "dashboard.managed_gemma_notice.runtime_files_failed"),
        (False, "runtime_files", "dashboard.managed_gemma_notice.runtime_files_failed"),
    ],
)
def test_managed_failure_notice_uses_application_presentation_hint(
    gpu_only: bool, phase: str, key: str
) -> None:
    notice = managed_gemma_capture_notice(
        ManagedGemmaDashboardNotice(
            status="failed",
            backend="gpu",
            gpu_only=gpu_only,
            failure_phase=phase,
        )
    )
    assert notice is not None
    assert notice.tone == "error"
    assert notice.text == f"i18n:{key}"
