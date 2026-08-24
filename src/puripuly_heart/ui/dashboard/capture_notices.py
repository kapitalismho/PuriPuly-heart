from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.ui_models import (
    ManagedGemmaDashboardNotice,
    ManagedGemmaNoticeAction,
)
from puripuly_heart.ui.gpu_notice import GpuDashboardNotice, GpuNoticeAction
from puripuly_heart.ui.i18n import t

LOCAL_ASR_MODEL_LABEL_KEYS = {
    "parakeet-tdt-0.6b-v3-int8-sherpa": "local_stt.model.parakeet-tdt-0.6b-v3-int8-sherpa",
    "parakeet-tdt-ctc-0.6b-ja-int8-sherpa": "local_stt.model.parakeet-tdt-ctc-0.6b-ja-int8-sherpa",
    "qwen3-asr-0.6b-int8-sherpa": "local_stt.model.qwen3-asr-0.6b-int8-sherpa",
}
LOCAL_ASR_TARGETED_NOTICE_KEYS = {
    "missing": "dashboard.local_stt_notice_missing_model",
    "invalid": "dashboard.local_stt_notice_invalid_model",
    "downloading": "dashboard.local_stt_notice_downloading_model",
    "download_failed": "dashboard.local_stt_notice_download_failed_model",
}
LOCAL_ASR_NOTICE_KEYS = {
    "starting": "dashboard.local_stt_notice_starting",
    "self_loading": "dashboard.local_stt_notice_self_loading",
    "peer_loading": "dashboard.local_stt_notice_peer_loading",
    "start_failed": "dashboard.local_stt_notice_start_failed",
    "missing": "dashboard.local_stt_notice_missing",
    "invalid": "dashboard.local_stt_notice_invalid",
    "downloading": "dashboard.local_stt_notice_downloading",
    "download_failed": "dashboard.local_stt_notice_download_failed",
}
LOCAL_ASR_NOTICE_TONES = {
    "starting": "info",
    "self_loading": "info",
    "peer_loading": "info",
    "start_failed": "error",
    "missing": "warning",
    "invalid": "warning",
    "downloading": "info",
    "download_failed": "error",
}
GPU_NOTICE_KEYS = {
    "discovery_failed": "dashboard.gpu_notice.discovery_failed",
    "unsupported": "dashboard.gpu_notice.unsupported",
    "unavailable_device": "dashboard.gpu_notice.unavailable_device",
    "not_installed": "dashboard.gpu_notice.not_installed",
    "invalid": "dashboard.gpu_notice.invalid",
    "installing": "dashboard.gpu_notice.installing",
    "install_failed": "dashboard.gpu_notice.install_failed",
    "activation_failed": "dashboard.gpu_notice.activation_failed",
}
GPU_NOTICE_TONES = {
    "discovery_failed": "error",
    "unsupported": "warning",
    "unavailable_device": "warning",
    "not_installed": "warning",
    "invalid": "warning",
    "install_failed": "error",
    "activation_failed": "error",
}
GPU_ACTION_KEYS = {
    "install": "dashboard.gpu_action.install",
    "repair": "dashboard.gpu_action.repair",
    "reinstall": "dashboard.gpu_action.reinstall",
    "rediscover": "dashboard.gpu_action.rediscover",
    "restart": "dashboard.gpu_action.restart",
}
MANAGED_GEMMA_NOTICE_KEYS = {
    "checking": "dashboard.managed_gemma_notice.checking",
    "downloading": "dashboard.managed_gemma_notice.downloading",
    "preparing": "dashboard.managed_gemma_notice.preparing",
    "failed": "dashboard.managed_gemma_notice.failed",
    "cancelled": "dashboard.managed_gemma_notice.cancelled",
}
MANAGED_GEMMA_NOTICE_TONES = {
    "checking": "info",
    "downloading": "info",
    "preparing": "info",
    "failed": "error",
    "cancelled": "warning",
}
MANAGED_GEMMA_ACTION_KEYS = {
    "cancel": "dashboard.managed_gemma_action.cancel",
}


@dataclass(frozen=True, slots=True)
class CaptureNotice:
    text: str
    tone: str | None
    action: GpuNoticeAction | None = None


def local_asr_capture_notice(
    *,
    status: str | None,
    percent: int | None = None,
    model_id: str | None = None,
) -> CaptureNotice | None:
    if status is None:
        return None
    notice_key = LOCAL_ASR_NOTICE_KEYS.get(status)
    if notice_key is None:
        return None

    targeted = model_id is not None and status in LOCAL_ASR_TARGETED_NOTICE_KEYS
    if targeted:
        model = t(LOCAL_ASR_MODEL_LABEL_KEYS.get(model_id, ""), default=model_id)
        text = (
            t("dashboard.local_stt_notice_downloading_progress_model", model=model, percent=percent)
            if status == "downloading" and percent is not None
            else t(LOCAL_ASR_TARGETED_NOTICE_KEYS[status], model=model)
        )
    else:
        text = (
            t("dashboard.local_stt_notice_downloading_progress", percent=percent)
            if status == "downloading" and percent is not None
            else t(notice_key)
        )
    return CaptureNotice(text=text, tone=LOCAL_ASR_NOTICE_TONES.get(status))


def gpu_capture_notice(notice: GpuDashboardNotice | None) -> CaptureNotice | None:
    if notice is None or notice.status not in GPU_NOTICE_KEYS:
        return None
    key = GPU_NOTICE_KEYS[notice.status]
    text = t(key, percent=notice.progress_percent or 0) if notice.status == "installing" else t(key)
    return CaptureNotice(
        text=text,
        tone=GPU_NOTICE_TONES.get(notice.status, "info"),
        action=notice.action,
    )


def gpu_capture_action_label(action: GpuNoticeAction | None) -> str | None:
    if action is None:
        return None
    return t(GPU_ACTION_KEYS[action])


def managed_gemma_capture_notice(
    notice: ManagedGemmaDashboardNotice | None,
) -> CaptureNotice | None:
    if notice is None or notice.status not in MANAGED_GEMMA_NOTICE_KEYS:
        return None
    key = MANAGED_GEMMA_NOTICE_KEYS[notice.status]
    text = (
        t(key, percent=notice.progress_percent or 0) if notice.status == "downloading" else t(key)
    )
    return CaptureNotice(
        text=text,
        tone=MANAGED_GEMMA_NOTICE_TONES[notice.status],
        action=None,
    )


def managed_gemma_action_label(
    action: ManagedGemmaNoticeAction | None,
) -> str | None:
    if action is None:
        return None
    return t(MANAGED_GEMMA_ACTION_KEYS[action])


__all__ = [
    "GPU_ACTION_KEYS",
    "GPU_NOTICE_KEYS",
    "GPU_NOTICE_TONES",
    "LOCAL_ASR_MODEL_LABEL_KEYS",
    "LOCAL_ASR_NOTICE_KEYS",
    "LOCAL_ASR_NOTICE_TONES",
    "LOCAL_ASR_TARGETED_NOTICE_KEYS",
    "MANAGED_GEMMA_ACTION_KEYS",
    "MANAGED_GEMMA_NOTICE_KEYS",
    "MANAGED_GEMMA_NOTICE_TONES",
    "CaptureNotice",
    "gpu_capture_action_label",
    "gpu_capture_notice",
    "local_asr_capture_notice",
    "managed_gemma_action_label",
    "managed_gemma_capture_notice",
]
