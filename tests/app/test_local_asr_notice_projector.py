from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.local_asr_application import (
    LocalASRApplicationSettings,
    LocalASRNoticeProjector,
)
from puripuly_heart.core.local_asr.local_asr_provisioning import (
    LocalASRModelProvisioningState,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_asr.local_stt_assets import (
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.core.self_capture import SelfCaptureSessionState


def _ready_provisioning() -> LocalASRProvisioningSnapshot:
    return LocalASRProvisioningSnapshot(
        models=tuple(
            LocalASRModelProvisioningState(
                model_id=model_id,
                backend="cpu",
                integrity="ready",
            )
            for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
        ),
        required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
        gpu_model_id="gpu",
    )


def _settings() -> LocalASRApplicationSettings:
    return LocalASRApplicationSettings(
        locale="en",
        self_provider="local_fun_asr_mlt_nano",
        peer_provider="deepgram",
        self_source_language="ko",
        peer_source_language="en",
        self_gpu_provider=False,
        peer_gpu_provider=False,
        peer_requested=False,
        peer_activation_requested=False,
    )


def _sync(*, desired_active: bool, state: SelfCaptureSessionState) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []
    projector = LocalASRNoticeProjector(
        settings_provider=_settings,
        self_capture_provider=lambda: SimpleNamespace(
            snapshot=SimpleNamespace(desired_active=desired_active, state=state)
        ),
        peer=lambda: SimpleNamespace(activation_starting=False, model_loading=False),
        provisioning_snapshot=_ready_provisioning,
        sink=lambda **kwargs: seen.append(kwargs),
    )
    projector.sync()
    return seen


@pytest.mark.parametrize(
    "state",
    (SelfCaptureSessionState.STARTING, SelfCaptureSessionState.ADMISSION_PENDING),
)
def test_background_self_prepare_does_not_report_talk_starting(
    state: SelfCaptureSessionState,
) -> None:
    seen = _sync(desired_active=False, state=state)

    assert seen == [
        {
            "status": None,
            "model_id": None,
            "percent": None,
            "starting": False,
        }
    ]


@pytest.mark.parametrize(
    "state",
    (SelfCaptureSessionState.STARTING, SelfCaptureSessionState.ADMISSION_PENDING),
)
def test_user_requested_self_start_reports_loading_and_starting(
    state: SelfCaptureSessionState,
) -> None:
    seen = _sync(desired_active=True, state=state)

    assert seen == [
        {
            "status": "self_loading",
            "model_id": None,
            "percent": None,
            "starting": True,
        }
    ]
