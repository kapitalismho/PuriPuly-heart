from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import flet as ft
import pytest

from puripuly_heart.config.process_capture_resolution import (
    ProcessCaptureTargetUnavailableError,
)
from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget
from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.config.settings_vnext.facade import load_settings, save_settings_with_result
from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRModelProvisioningState,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.core.runtime.peer_channel import (
    PeerRuntimeDiagnostic,
    PeerRuntimeFailureReason,
)
from puripuly_heart.core.self_capture import (
    SelfCaptureFailureReason,
    SelfCaptureProviderStatus,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.components.settings.settings_modal import OptionItem, SettingsModal
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.overlay_peer_contract import (
    build_overlay_peer_consumer_contract,
    build_overlay_peer_consumer_contract_from_state,
    is_process_capture_warning_reason,
)
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views.dashboard import DashboardView
from puripuly_heart.ui.views.settings import SettingsView


def _presentation(
    app: object,
    *,
    page: object | None = None,
) -> FletUiPresentationAdapter:
    if page is not None:
        setattr(app, "page", page)
    return FletUiPresentationAdapter(app)


class ReadyProvisioningPort:
    def __init__(self) -> None:
        self.snapshot = LocalASRProvisioningSnapshot(
            models=(
                LocalASRModelProvisioningState(PARAKEET_V3_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(PARAKEET_JAPANESE_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(LOCAL_STT_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(LOCAL_QWEN_GPU_MODEL_ID, "gpu", "ready"),
            ),
            required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
            gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
        )

    @property
    def diagnostics(self):
        return ()

    async def inspect_cpu(self, model_ids=None, *, verify_checksums=False):
        _ = (model_ids, verify_checksums)
        return self.snapshot

    async def inspect_gpu(self, *, explicit_intent, verify_checksums=False):
        _ = (explicit_intent, verify_checksums)
        return self.snapshot

    def start_install(self, request):
        raise AssertionError(f"unexpected provisioning install: {request}")

    async def report_model_validation_failure(self, model_id, *, failure_type):
        _ = (model_id, failure_type)
        return self.snapshot

    async def cancel_install(self, backend):
        _ = backend

    async def close(self):
        return


class RecordingControllerSelfOwner:
    def __init__(self) -> None:
        self.apply_calls: list[bool] = []
        self.generation = 0
        self.fail_start = False
        self.loop_task = None
        self.source = None
        self.cleanup_source = None
        self.vad = None
        self.last_cleanup_exception = None

    async def apply_intent(
        self,
        config: object,
        *,
        enabled: bool,
        **kwargs: object,
    ) -> SelfCaptureSessionSnapshot:
        _ = (config, kwargs)
        self.apply_calls.append(enabled)
        self.generation += 1
        failed = enabled and self.fail_start
        active = enabled and not failed
        self.loop_task = object() if active else None
        self.source = object() if active else None
        self.vad = object() if active else None
        return SelfCaptureSessionSnapshot(
            state=(
                SelfCaptureSessionState.FAULTED
                if failed
                else SelfCaptureSessionState.RUNNING if active else SelfCaptureSessionState.STOPPED
            ),
            provider_status=(
                SelfCaptureProviderStatus.FAILED
                if failed
                else (
                    SelfCaptureProviderStatus.READY
                    if active
                    else SelfCaptureProviderStatus.DETACHED
                )
            ),
            desired_active=active,
            effective_active=active,
            generation=self.generation,
            provider_id="local_qwen",
            runtime_signature=("runtime",),
            failure_reason=(SelfCaptureFailureReason.SOURCE_OPEN_FAILED if failed else None),
            admission_reason=None,
            has_source=active,
            has_vad=active,
            has_loop_task=active,
            cleanup_debt=0,
            closed=False,
        )


def _runtime_hub_stub(
    *,
    self_available: bool = True,
    peer_available: bool = False,
    **values: object,
) -> SimpleNamespace:
    async def abort_self_stt_for_toggle_off() -> None:
        return None

    async def drain_self_stt_for_toggle_off(
        *,
        release_backend_after: float | None = None,
    ) -> None:
        _ = release_backend_after

    async def warmup_stt_channel(channel: str) -> None:
        _ = channel

    return SimpleNamespace(
        stt=object() if self_available else None,
        peer_stt=object() if peer_available else None,
        has_stt_provider=lambda channel: (self_available if channel == "self" else peer_available),
        abort_self_stt_for_toggle_off=abort_self_stt_for_toggle_off,
        drain_self_stt_for_toggle_off=drain_self_stt_for_toggle_off,
        warmup_stt_channel=warmup_stt_channel,
        mark_promo_eligible=lambda: None,
        **values,
    )


@pytest.fixture(autouse=True)
def _use_ready_local_asr_provisioning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        controller_module,
        "create_local_asr_provisioning_owner",
        lambda **_kwargs: ReadyProvisioningPort(),
    )


def _load_locale_keys(locale: str) -> set[str]:
    from puripuly_heart.ui import i18n as i18n_module

    path = Path(i18n_module.__file__).resolve().parents[1] / "data" / "i18n" / f"{locale}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data)


def _collect_text_values(control: ft.Control) -> list[str]:
    """Recursively collect all string values from a control tree."""
    values: list[str] = []
    if hasattr(control, "value") and isinstance(control.value, str) and control.value:
        values.append(control.value)
    children = getattr(control, "controls", None)
    if children:
        for child in children:
            values.extend(_collect_text_values(child))
    inner = getattr(control, "content", None)
    if inner is not None and inner is not control:
        values.extend(_collect_text_values(inner))
    return values


PROCESS_WARNING_KEYS = [
    "settings.peer_translation.warning.process_unavailable_no_process",
    "settings.peer_translation.warning.process_unavailable_ambiguous",
    "settings.peer_translation.warning.process_unavailable_ineligible",
    "settings.peer_translation.warning.process_unavailable_unsupported_platform",
    "settings.peer_translation.warning.process_setup_failed",
    "settings.peer_translation.warning.process_target_exited",
    "settings.peer_translation.warning.process_source_failed",
    "settings.peer_translation.warning.process_provider_failed",
    "settings.peer_translation.warning.process_capture_failed",
    "settings.desktop_audio.section.process",
    "settings.desktop_audio.section.device",
    "settings.desktop_audio.process.vrchat",
    "settings.desktop_audio.process.discord_stable",
    "settings.desktop_audio.process.discord_ptb",
    "settings.desktop_audio.process.discord_canary",
]


def test_process_warning_i18n_keys_have_locale_parity() -> None:
    bundles = {locale: _load_locale_keys(locale) for locale in ("en", "ko", "ja", "zh-CN")}
    for key in PROCESS_WARNING_KEYS:
        for locale, keys in bundles.items():
            assert key in keys, f"{locale} missing {key}"


def test_activation_starting_i18n_keys_have_locale_parity() -> None:
    keys = (
        "settings.peer_translation.status.starting",
        "dashboard.local_stt_notice_starting",
        "dashboard.local_stt_notice_start_failed",
    )
    bundles = {locale: _load_locale_keys(locale) for locale in ("en", "ko", "ja", "zh-CN")}
    for key in keys:
        assert all(key in locale_keys for locale_keys in bundles.values())


def test_peer_contract_exposes_starting_before_readiness() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_activation_starting=True,
    )
    assert contract.peer.state == "starting"
    assert contract.peer.status_text == t("settings.peer_translation.status.starting")


def test_peer_contract_keeps_starting_visible_during_effective_model_transition() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=True,
        peer_activation_starting=True,
    )

    assert contract.peer.state == "starting"


def test_peer_contract_treats_overlay_startup_as_peer_starting() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="starting",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
    )

    assert contract.peer.state == "starting"
    assert contract.peer.helper_text == ""


def test_peer_contract_keeps_process_failure_warning_during_overlay_startup() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="starting",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_provider_failed",
    )

    assert contract.peer.state == "warning"
    assert contract.peer.warning_reason == "process_provider_failed"


@pytest.mark.asyncio
async def test_process_identity_resolution_is_fresh_and_does_not_block_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("x"),
    )
    target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )
    config = SimpleNamespace(
        capture_target=target,
        backend=SimpleNamespace(sample_rate_hz=16000),
    )
    calls: list[int] = []
    release = threading.Event()

    class Resolver:
        def __init__(self, *, snapshots):  # noqa: ANN001
            _ = snapshots

        def resolve_for_start(self, _target):  # noqa: ANN001
            calls.append(threading.get_ident())
            release.wait(timeout=1)
            return SimpleNamespace(identity=object(), unavailable_reason=None)

    monkeypatch.setattr(controller_module, "ProcessCaptureResolver", Resolver)
    monkeypatch.setattr(
        GuiController,
        "_create_process_peer_audio_source",
        lambda _self, _config, *, resolution: resolution.identity,
    )
    activation = asyncio.create_task(
        controller._create_peer_audio_source_from_runtime_config(config)
    )
    await asyncio.sleep(0)
    heartbeat = False
    await asyncio.sleep(0)
    heartbeat = True
    release.set()
    assert await activation is not None
    assert heartbeat is True
    assert len(calls) == 1
    assert calls[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_idle_process_preparation_is_bounded_once_and_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued: list[object] = []
    page = SimpleNamespace(run_task=lambda callback: queued.append(callback))
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("x"),
    )
    calls = 0

    class FailingSnapshots:
        def snapshots(self):
            nonlocal calls
            calls += 1
            raise RuntimeError("cold enumeration failed")

    monkeypatch.setattr(controller_module, "PsutilCurrentUserProcessSnapshots", FailingSnapshots)
    controller._schedule_process_discovery_idle_preparation()
    controller._schedule_process_discovery_idle_preparation()
    assert len(queued) == 1
    await queued[0]()
    assert calls == 1

    fresh_resolutions = 0

    class FreshResolver:
        def __init__(self, *, snapshots):  # noqa: ANN001
            _ = snapshots

        def resolve_for_start(self, _target):  # noqa: ANN001
            nonlocal fresh_resolutions
            fresh_resolutions += 1
            return SimpleNamespace(identity=object(), unavailable_reason=None)

    target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )
    monkeypatch.setattr(controller_module, "ProcessCaptureResolver", FreshResolver)
    monkeypatch.setattr(
        GuiController,
        "_create_process_peer_audio_source",
        lambda _self, _config, *, resolution: resolution.identity,
    )
    await controller._create_peer_audio_source_from_runtime_config(
        SimpleNamespace(capture_target=target, backend=SimpleNamespace(sample_rate_hz=16000))
    )
    assert calls == 1
    assert fresh_resolutions == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("final_enabled", [False, True])
async def test_peer_starting_is_published_before_delayed_readiness_and_latest_intent_wins(
    monkeypatch: pytest.MonkeyPatch,
    final_enabled: bool,
) -> None:
    contracts = []
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(
            SimpleNamespace(
                view_dashboard=SimpleNamespace(
                    set_overlay_peer_contract=contracts.append,
                )
            )
        ),
        config_path=Path("x"),
    )
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.settings.ui.overlay_enabled = True
    controller.overlay_state = "connected"
    controller.hub = _runtime_hub_stub(
        self_available=False,
        peer_translation_enabled=False,
        integrated_context_enabled=False,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_ready(_self, **_kwargs) -> bool:  # noqa: ANN001
        entered.set()
        await release.wait()
        return True

    async def no_refresh(_self) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr(GuiController, "_ensure_peer_local_stt_ready", delayed_ready)
    monkeypatch.setattr(GuiController, "_refresh_overlay_runtime_dependencies", no_refresh)
    enabling = asyncio.create_task(controller.set_peer_translation_enabled(True))
    await entered.wait()
    assert contracts[-1].peer.state == "starting"
    await controller.set_peer_translation_enabled(False)
    latest = None
    if final_enabled:
        latest = asyncio.create_task(controller.set_peer_translation_enabled(True))
        await asyncio.sleep(0)
        assert contracts[-1].peer.state == "starting"
    release.set()
    await asyncio.gather(enabling, *([latest] if latest is not None else []))
    assert controller.settings.ui.peer_translation_enabled is final_enabled
    assert controller._peer_activation_starting is False


@pytest.mark.asyncio
async def test_self_toggle_path_delegates_only_to_composed_capture_owner() -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("x"),
    )
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.hub = _runtime_hub_stub()
    owner = RecordingControllerSelfOwner()
    controller._self_capture_owner = owner
    await controller.set_stt_enabled(True)
    await controller.set_stt_enabled(False)
    await controller.set_stt_enabled(True)

    assert owner.apply_calls == [True, False, True]
    assert controller._stt_desired is True
    assert controller._mic_task is owner.loop_task


@pytest.mark.asyncio
async def test_self_microphone_start_failure_becomes_effective_off_failure_notice() -> None:
    dash = SimpleNamespace(
        enabled=[],
        notices=[],
        set_stt_enabled=lambda enabled: dash.enabled.append(enabled),
        set_local_stt_notice=lambda status, percent=None: dash.notices.append((status, percent)),
    )
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace(view_dashboard=dash)),
        config_path=Path("x"),
    )
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.hub = _runtime_hub_stub()
    owner = RecordingControllerSelfOwner()
    owner.fail_start = True
    controller._self_capture_owner = owner
    await controller.set_stt_enabled(True)
    assert controller._stt_desired is False
    assert controller._stt_activation_failed is True
    assert dash.enabled[-1] is False
    assert dash.notices[-1][0] == "start_failed"


@pytest.mark.asyncio
async def test_peer_post_readiness_runtime_completion_cannot_publish_after_supersession(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disclosures: list[str] = []
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("x"),
    )
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.SONIOX
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.settings.ui.overlay_enabled = True
    controller.overlay_state = "connected"
    controller.hub = _runtime_hub_stub(
        peer_available=True,
        peer_translation_enabled=False,
        integrated_context_enabled=False,
        enqueue_peer_translation_disclosure=disclosures.append,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    policy_calls: list[bool] = []

    class Runtime:
        async def apply_policy(self, *, config, desired_active):  # noqa: ANN001
            _ = config
            policy_calls.append(desired_active)
            if desired_active:
                entered.set()
                await release.wait()

    controller._peer_runtime = Runtime()  # type: ignore[assignment]
    monkeypatch.setattr(
        GuiController,
        "_ensure_peer_local_stt_ready",
        lambda self, **_kwargs: asyncio.sleep(0, result=True),
    )
    monkeypatch.setattr(
        GuiController,
        "_peer_runtime_should_be_active",
        lambda self, settings: settings.ui.peer_translation_enabled,
    )
    enabling = asyncio.create_task(controller.set_peer_translation_enabled(True))
    await entered.wait()
    controller._peer_activation_generation += 1
    controller.settings.ui.peer_translation_enabled = False
    controller._peer_activation_starting = False
    await controller._peer_runtime.apply_policy(
        config=controller._build_peer_runtime_config(controller.settings),
        desired_active=False,
    )
    release.set()
    await enabling
    assert policy_calls == [True, False]
    assert controller.settings.ui.peer_translation_enabled is False
    assert controller.hub.peer_translation_enabled is False
    assert disclosures == []


def test_settings_modal_renders_process_section_before_device_and_hides_descriptions() -> None:
    options = [
        OptionItem(
            value="process:vrchat:c:\\vrchat\\vrchat.exe",
            label="VRChat",
            description="should stay hidden",
            section="Applications",
        ),
        OptionItem(
            value="process:generic:c:\\apps\\game\\game.exe",
            label="Game (2)",
            description="hidden",
            disabled=True,
            section="Applications",
        ),
        OptionItem(value="device:", label="Auto", section="Output devices"),
        OptionItem(value="device:Speakers", label="Speakers", section="Output devices"),
    ]
    modal = SettingsModal(
        page=SimpleNamespace(open=lambda *_a, **_k: None, close=lambda *_a, **_k: None),
        title="Loopback Audio",
        options=options,
        on_select=lambda _value: None,
        show_description=False,
        two_column=True,
    )
    option_list = modal._build_option_list("process:vrchat:c:\\vrchat\\vrchat.exe")

    assert isinstance(option_list, ft.Row)
    assert len(option_list.controls) == 2

    left_column = option_list.controls[0].content
    right_column = option_list.controls[1].content

    left_labels = _collect_text_values(left_column)
    right_labels = _collect_text_values(right_column)

    assert "Applications" in left_labels
    assert "VRChat" in left_labels
    assert "Game (2)" in left_labels
    assert "Output devices" in right_labels
    assert "Auto" in right_labels
    assert "Speakers" in right_labels
    assert "should stay hidden" not in left_labels + right_labels
    assert "hidden" not in left_labels + right_labels


def test_settings_modal_renders_unsectioned_options_without_loading_section() -> None:
    modal = SettingsModal(
        page=SimpleNamespace(open=lambda *_a, **_k: None, close=lambda *_a, **_k: None),
        title="Provider",
        options=[
            OptionItem(value="first", label="First"),
            OptionItem(value="second", label="Second"),
        ],
        on_select=lambda _value: None,
    )

    option_list = modal._build_option_list("first")

    assert [control.content.value for control in option_list.controls] == ["First", "Second"]


def test_settings_modal_only_replaces_explicit_loading_section() -> None:
    modal = SettingsModal(
        page=SimpleNamespace(open=lambda *_a, **_k: None, close=lambda *_a, **_k: None),
        title="Loopback Audio",
        options=[
            OptionItem(value="", label="", section="Applications"),
            OptionItem(value="device:", label="Auto", section="Output devices"),
        ],
        on_select=lambda _value: None,
    )
    modal._loading_section = "Applications"

    controls = modal._build_option_items("device:")

    assert len(controls) == 4
    assert controls[0].content.controls[-1].value == "Applications"
    assert controls[1].content.controls[0].__class__.__name__ == "ProgressRing"
    assert controls[2].content.controls[-1].value == "Output devices"
    assert controls[3].content.value == "Auto"


def test_settings_modal_two_column_shows_loading_in_left_column() -> None:
    modal = SettingsModal(
        page=SimpleNamespace(open=lambda *_a, **_k: None, close=lambda *_a, **_k: None),
        title="Loopback Audio",
        options=[
            OptionItem(value="", label="", section="Applications"),
            OptionItem(value="device:", label="Auto", section="Output devices"),
        ],
        on_select=lambda _value: None,
        two_column=True,
    )
    modal._loading_section = "Applications"

    option_list = modal._build_option_list("device:")

    assert isinstance(option_list, ft.Row)
    left_items = option_list.controls[0].content.controls
    right_items = option_list.controls[1].content.controls

    assert left_items[0].content.controls[-1].value == "Applications"
    assert left_items[1].content.controls[0].__class__.__name__ == "ProgressRing"
    assert right_items[0].content.controls[-1].value == "Output devices"
    assert right_items[1].content.value == "Auto"


def test_settings_modal_replace_options_updates_both_columns() -> None:
    modal = SettingsModal(
        page=SimpleNamespace(open=lambda *_a, **_k: None, close=lambda *_a, **_k: None),
        title="Loopback Audio",
        options=[
            OptionItem(value="", label="", section="Applications"),
            OptionItem(value="device:", label="Auto", section="Output devices"),
        ],
        on_select=lambda _value: None,
        two_column=True,
    )
    modal._loading_section = "Applications"
    option_list = modal._build_option_list("device:")
    assert isinstance(option_list, ft.Row)

    modal.replace_options(
        [
            OptionItem(value="process:vrchat", label="VRChat", section="Applications"),
            OptionItem(value="device:", label="Auto", section="Output devices"),
        ]
    )

    left_labels = _collect_text_values(option_list.controls[0].content)
    right_labels = _collect_text_values(option_list.controls[1].content)
    assert "VRChat" in left_labels
    assert "Auto" in right_labels


def test_process_warning_helper_text_is_localized_and_retry_classified() -> None:
    set_locale("en")
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_target_exited",
    )
    assert contract.peer.state == "warning"
    assert is_process_capture_warning_reason(contract.peer.warning_reason)
    assert contract.peer.helper_text == t("settings.peer_translation.warning.process_target_exited")


def test_dashboard_process_warning_click_toggles_off_instead_of_retrying() -> None:
    view = DashboardView.__new__(DashboardView)
    view._overlay_peer_contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_setup_failed",
    )
    retries: list[bool] = []
    toggles: list[bool] = []
    view.on_retry_peer_process_capture = lambda: retries.append(True)
    view.on_toggle_peer_translation = lambda enabled: toggles.append(enabled)
    view._toggle_peer_translation()
    assert retries == []
    assert toggles == [False]


def test_dashboard_normal_peer_toggle_still_inverts_intent() -> None:
    view = DashboardView.__new__(DashboardView)
    view._overlay_peer_contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=True,
        peer_warning_reason=None,
    )
    toggles: list[bool] = []
    view.on_retry_peer_process_capture = lambda: None
    view.on_toggle_peer_translation = lambda enabled: toggles.append(enabled)
    view._toggle_peer_translation()
    assert toggles == [False]


def test_controller_encodes_and_decodes_loopback_capture_options() -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    process = ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe")
    encoded = controller._encode_process_capture_option(process)
    decoded = controller._decode_capture_option(encoded)
    assert decoded.kind == "process"
    assert decoded.process is not None
    assert decoded.process.kind == "generic_executable"
    assert controller._decode_capture_option("device:Speakers").kind == "named_output_device"
    assert controller._decode_capture_option("device:").kind == "default_output_device"


def test_controller_list_loopback_options_puts_process_before_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )

    class FakeResolver:
        def __init__(self, *, snapshots):  # noqa: ANN001
            _ = snapshots

        def enumerate_candidates(self):
            return (
                SimpleNamespace(
                    name="VRChat",
                    target=ProcessCaptureTargetIntent.vrchat(r"C:\VRChat\VRChat.exe"),
                    enabled=True,
                ),
                SimpleNamespace(
                    name="Game (2)",
                    target=ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe"),
                    enabled=False,
                ),
            )

    monkeypatch.setattr(controller_module, "ProcessCaptureResolver", FakeResolver)
    monkeypatch.setattr(
        GuiController,
        "_enumerate_loopback_device_names",
        staticmethod(lambda: ["Speakers (Loopback)"]),
    )
    options = controller.list_loopback_capture_options()
    assert options[0].section == t("settings.desktop_audio.section.process")
    assert options[0].label == "VRChat"
    assert options[0].description == ""
    assert options[1].disabled is True
    assert options[1].label == "Game (2)"
    device_index = next(i for i, option in enumerate(options) if option.value.startswith("device:"))
    assert device_index > 1
    assert options[device_index].section == t("settings.desktop_audio.section.device")


def test_controller_process_options_sorts_disabled_after_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )

    class FakeResolver:
        def __init__(self, *, snapshots):  # noqa: ANN001
            _ = snapshots

        def enumerate_candidates(self):
            return (
                SimpleNamespace(
                    name="Game (2)",
                    target=ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe"),
                    enabled=False,
                ),
                SimpleNamespace(
                    name="VRChat",
                    target=ProcessCaptureTargetIntent.vrchat(r"C:\VRChat\VRChat.exe"),
                    enabled=True,
                ),
            )

    monkeypatch.setattr(controller_module, "ProcessCaptureResolver", FakeResolver)
    monkeypatch.setattr(
        GuiController,
        "_enumerate_loopback_device_names",
        staticmethod(lambda: []),
    )
    options = controller.list_loopback_process_options()
    assert options[0].label == "VRChat"
    assert options[0].disabled is False
    assert options[1].label == "Game (2)"
    assert options[1].disabled is True


def test_controller_process_diagnostic_sets_peer_warning_reason() -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller._on_peer_runtime_diagnostic(
        PeerRuntimeDiagnostic(
            reason=PeerRuntimeFailureReason.PROCESS_TARGET_UNAVAILABLE,
            capture_kind="process",
            process_unavailable_reason="no_process",
        )
    )
    assert controller._peer_process_warning_reason == "process_unavailable_no_process"
    state = controller.overlay_peer_presentation_state()
    assert state is not None
    contract = build_overlay_peer_consumer_contract_from_state(state)
    assert contract is not None
    assert contract.peer.warning_reason == "process_unavailable_no_process"


@pytest.mark.asyncio
async def test_controller_retry_clears_process_warning_only_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller._peer_process_warning_reason = "process_target_exited"

    class Runtime:
        def __init__(self, result: bool) -> None:
            self.result = result

        async def retry_process_capture(self, *, config):  # noqa: ANN001
            _ = config
            return self.result

    controller._peer_runtime = Runtime(False)  # type: ignore[assignment]
    monkeypatch.setattr(
        GuiController, "_peer_runtime_should_be_active", lambda self, settings: True
    )
    monkeypatch.setattr(
        GuiController,
        "_ensure_peer_local_stt_ready",
        lambda self: __import__("asyncio").sleep(0, result=True),
    )
    assert await controller.retry_peer_process_capture() is False
    assert controller._peer_process_warning_reason == "process_target_exited"

    controller._peer_runtime = Runtime(True)  # type: ignore[assignment]
    assert await controller.retry_peer_process_capture() is True
    assert controller._peer_process_warning_reason is None


def test_loopback_summary_prefers_localized_process_name() -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    settings = AppSettings()
    settings.desktop_audio.runtime_capture_target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )
    set_locale("en")
    assert controller.loopback_capture_summary(settings) == t(
        "settings.desktop_audio.process.vrchat"
    )
    for channel, basename, key in (
        ("stable", "Discord.exe", "settings.desktop_audio.process.discord_stable"),
        ("ptb", "DiscordPTB.exe", "settings.desktop_audio.process.discord_ptb"),
        ("canary", "DiscordCanary.exe", "settings.desktop_audio.process.discord_canary"),
    ):
        settings.desktop_audio.runtime_capture_target = ResolvedDesktopAudioCaptureTarget(
            kind="process",
            process_kind="discord",
            discord_channel=channel,
            executable_basename=basename,
        )
        assert controller.loopback_capture_summary(settings) == t(key)
    settings.desktop_audio.runtime_capture_target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"c:\apps\game\game.exe",
    )
    assert controller.loopback_capture_summary(settings) == "game"


def test_list_options_preserves_saved_process_when_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    saved = ProcessCaptureTargetIntent.vrchat(r"C:\VRChat\VRChat.exe")
    controller.settings.desktop_audio.runtime_capture_target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )

    class EmptyResolver:
        def __init__(self, *, snapshots):  # noqa: ANN001
            _ = snapshots

        def enumerate_candidates(self):
            return ()

    monkeypatch.setattr(controller_module, "ProcessCaptureResolver", EmptyResolver)
    monkeypatch.setattr(
        GuiController,
        "_enumerate_loopback_device_names",
        staticmethod(lambda: ["Speakers"]),
    )
    options = controller.list_loopback_capture_options()
    current = controller.current_loopback_capture_option_value()
    process_options = [option for option in options if option.value.startswith("process:")]
    assert len(process_options) == 1
    assert process_options[0].value == current
    assert process_options[0].label == t("settings.desktop_audio.process.vrchat")
    assert process_options[0].description == ""
    assert process_options[0].disabled is False
    assert "\\" not in process_options[0].label
    assert "pid" not in process_options[0].label.lower()
    assert controller.settings.desktop_audio.runtime_capture_target.kind == "process"
    assert (
        controller.settings.desktop_audio.runtime_capture_target.executable_identity
        == r"c:\vrchat\vrchat.exe"
    )
    _ = saved


def test_settings_capture_target_refresh_preserves_unrelated_drafts() -> None:
    view = SettingsView.__new__(SettingsView)
    baseline_settings = AppSettings()
    provider_draft = AppSettings()
    provider_draft.system_prompt = "provider draft"
    view._settings = baseline_settings
    view._provider_settings_draft = provider_draft
    view.has_provider_changes = True
    view.has_pending_prompt_changes = True
    view._desktop_overlay_pending_size_preset = "large"
    view._audio_settings = SimpleNamespace(desktop_output_device="Old device")
    view._loopback_audio_text = SimpleNamespace(
        content=SimpleNamespace(value="", size=None),
        page=None,
        update=lambda: None,
    )
    view.on_loopback_capture_summary = lambda: "VRChat"
    saved = AppSettings()
    saved.desktop_audio.output_device = "Saved device"

    view.refresh_loopback_capture_target(saved)

    assert view._settings is baseline_settings
    assert view._provider_settings_draft is provider_draft
    assert view._provider_settings_draft.system_prompt == "provider draft"
    assert view.has_provider_changes is True
    assert view.has_pending_prompt_changes is True
    assert view._desktop_overlay_pending_size_preset == "large"
    assert view._audio_settings.desktop_output_device == "Saved device"
    assert view._loopback_audio_text.content.value == "VRChat"


@pytest.mark.parametrize(
    ("initial_target", "committed_target", "committed_output"),
    [
        (
            ResolvedDesktopAudioCaptureTarget(kind="named_output_device", device_name="Speakers"),
            ResolvedDesktopAudioCaptureTarget(
                kind="process",
                process_kind="vrchat",
                executable_identity=r"c:\vrchat\vrchat.exe",
            ),
            "",
        ),
        (
            ResolvedDesktopAudioCaptureTarget(
                kind="process",
                process_kind="vrchat",
                executable_identity=r"c:\vrchat\vrchat.exe",
            ),
            ResolvedDesktopAudioCaptureTarget(kind="named_output_device", device_name="Headset"),
            "Headset",
        ),
    ],
)
def test_settings_capture_target_rebase_updates_all_retained_apply_sources(
    initial_target: ResolvedDesktopAudioCaptureTarget,
    committed_target: ResolvedDesktopAudioCaptureTarget,
    committed_output: str,
) -> None:
    view = SettingsView.__new__(SettingsView)
    retained = AppSettings()
    retained.desktop_audio.output_device = initial_target.device_name or ""
    retained.desktop_audio.runtime_capture_target = initial_target
    retained.stt.vad_speech_threshold = 0.31
    retained.system_prompt = "retained prompt"
    draft = copy.deepcopy(retained)
    draft.stt.vad_speech_threshold = 0.73
    draft.system_prompt = "draft prompt"
    view._settings = retained
    view._provider_settings_draft = draft
    view.has_provider_changes = True
    view.has_pending_prompt_changes = True
    view._audio_settings = SimpleNamespace(
        desktop_output_device=retained.desktop_audio.output_device
    )
    view._loopback_audio_text = SimpleNamespace(
        content=SimpleNamespace(value="", size=None),
        page=None,
        update=lambda: None,
    )
    view.on_loopback_capture_summary = lambda: "Committed target"
    committed = AppSettings()
    committed.desktop_audio.output_device = committed_output
    committed.desktop_audio.runtime_capture_target = committed_target

    view.refresh_loopback_capture_target(committed)
    view.refresh_loopback_capture_target(committed)

    for rebased in (retained, draft):
        assert rebased.desktop_audio.output_device == committed_output
        assert rebased.desktop_audio.runtime_capture_target == committed_target
    assert retained.stt.vad_speech_threshold == 0.31
    assert draft.stt.vad_speech_threshold == 0.73
    assert retained.system_prompt == "retained prompt"
    assert draft.system_prompt == "draft prompt"
    assert view.has_provider_changes is True
    assert view.has_pending_prompt_changes is True


@pytest.mark.asyncio
async def test_failed_process_warning_survives_unrelated_draft_apply_without_device_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    initial = AppSettings()
    initial.desktop_audio.output_device = "Speakers"
    assert save_settings_with_result(path, initial).ok
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=path,
    )
    controller.settings = load_settings(path)
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    view = SettingsView.__new__(SettingsView)
    view._settings = copy.deepcopy(controller.settings)
    view._provider_settings_draft = copy.deepcopy(controller.settings)
    view._provider_settings_draft.stt.vad_speech_threshold = 0.77
    view._provider_settings_draft.system_prompt = "pending prompt"
    view.has_provider_changes = True
    view.has_pending_prompt_changes = True
    view._audio_settings = SimpleNamespace(desktop_output_device="Speakers")
    view._loopback_audio_text = SimpleNamespace(
        content=SimpleNamespace(value="", size=None),
        page=None,
        update=lambda: None,
    )
    view.on_loopback_capture_summary = lambda: controller.loopback_capture_summary()
    controller.app = _presentation(SimpleNamespace(view_settings=view))

    async def refresh_peer_stt_runtime(_self) -> None:
        return None

    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", refresh_peer_stt_runtime)
    monkeypatch.setattr(GuiController, "_sync_effective_hub_flags", lambda *_args: None)
    monkeypatch.setattr(GuiController, "_refresh_overlay_peer_consumers", lambda *_args: None)
    await controller.apply_loopback_capture_option("process:vrchat:c:\\vrchat\\vrchat.exe")
    controller._on_peer_runtime_diagnostic(
        PeerRuntimeDiagnostic(
            reason=PeerRuntimeFailureReason.PROCESS_TARGET_UNAVAILABLE,
            capture_kind="process",
            process_unavailable_reason="no_process",
        )
    )
    warning_reason = controller._peer_process_warning_reason
    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert pending.desktop_audio.runtime_capture_target is not None
    assert pending.desktop_audio.runtime_capture_target.kind == "process"
    assert save_settings_with_result(path, pending).ok
    reloaded = load_settings(path)
    reloaded.ui.peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    reloaded.ui.peer_translation_eula_accepted = (
        controller.settings.ui.peer_translation_eula_accepted
    )
    controller.settings = reloaded

    persisted = load_settings(path)
    config = controller._build_peer_runtime_config(controller.settings)
    assert persisted.desktop_audio.runtime_capture_target.kind == "process"
    assert config.capture_target.kind == "process"
    assert controller._peer_process_warning_reason == warning_reason
    assert controller.peer_warning_action_is_retry() is True
    monkeypatch.setattr(
        controller_module,
        "DesktopLoopbackAudioSource",
        lambda **_kwargs: pytest.fail("device fallback constructed"),
    )

    class UnavailableResolver:
        def __init__(self, *, snapshots):
            _ = snapshots

        def resolve_for_start(self, _target):
            return SimpleNamespace(identity=None, unavailable_reason="no_process")

    monkeypatch.setattr(controller_module, "ProcessCaptureResolver", UnavailableResolver)
    with pytest.raises(ProcessCaptureTargetUnavailableError):
        await controller._create_peer_audio_source_from_runtime_config(config)


@pytest.mark.asyncio
async def test_controller_capture_target_apply_uses_narrow_settings_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    saved = AppSettings()
    saved.desktop_audio.runtime_capture_target = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )
    refreshed: list[AppSettings] = []
    view_settings = SimpleNamespace(
        refresh_loopback_capture_target=refreshed.append,
        load_from_settings=lambda *_args, **_kwargs: pytest.fail("full settings reload"),
    )
    controller.app = _presentation(SimpleNamespace(view_settings=view_settings))

    async def refresh_peer_stt_runtime(_self) -> None:
        return None

    def sync_effective_hub_flags(_self, _settings) -> None:
        return None

    def refresh_overlay_peer_consumers(_self) -> None:
        return None

    monkeypatch.setattr(
        type(controller._get_settings_owner()),
        "update_capture_target",
        lambda *_args: saved,
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", refresh_peer_stt_runtime)
    monkeypatch.setattr(GuiController, "_sync_effective_hub_flags", sync_effective_hub_flags)
    monkeypatch.setattr(
        GuiController,
        "_refresh_overlay_peer_consumers",
        refresh_overlay_peer_consumers,
    )

    await controller.apply_loopback_capture_option("process:vrchat:c:\\vrchat\\vrchat.exe")

    assert controller.settings is saved
    assert controller.settings.desktop_audio.runtime_capture_target.kind == "process"
    assert refreshed == [saved]


def test_dashboard_process_warning_clears_on_success_without_touching_other_warnings() -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]

    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_target_exited",
    )
    view.set_overlay_peer_contract(warning)
    assert view._process_capture_warning_active is True
    assert displays[-1] == warning.peer.helper_text

    success = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=True,
        peer_warning_reason=None,
    )
    view.set_overlay_peer_contract(success)
    assert view._process_capture_warning_active is False
    assert displays[-1] == ""


def test_dashboard_process_warning_clears_on_peer_disabled() -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]

    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_setup_failed",
    )
    view.set_overlay_peer_contract(warning)
    disabled = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=False,
        peer_effective_enabled=False,
        peer_warning_reason=None,
    )
    view.set_overlay_peer_contract(disabled)
    assert view._process_capture_warning_active is False
    assert displays[-1] == ""


def test_dashboard_process_warning_does_not_clear_stt_warning_content() -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = True
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]

    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_source_failed",
    )
    view.set_overlay_peer_contract(warning)
    view.set_display_text(t("dashboard.warn_stt_key"))
    success = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=True,
        peer_warning_reason=None,
    )
    view.set_overlay_peer_contract(success)
    assert view._process_capture_warning_active is False
    assert displays == [warning.peer.helper_text, t("dashboard.warn_stt_key")]
    assert view._stt_showing_warning is True


@pytest.mark.parametrize("peer_enabled", [True, False])
def test_dashboard_process_warning_does_not_clear_newer_display_content(
    peer_enabled: bool,
) -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]
    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_source_failed",
    )
    view.set_overlay_peer_contract(warning)
    view.set_display_text("newer unrelated content")
    resolved = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=peer_enabled,
        peer_effective_enabled=peer_enabled,
        peer_warning_reason=None,
    )
    view.set_overlay_peer_contract(resolved)
    assert displays[-1] == "newer unrelated content"
    assert view._current_display_text == "newer unrelated content"


def test_dashboard_process_warning_does_not_clear_newer_matching_primary_text() -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]
    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_source_failed",
    )
    view.set_overlay_peer_contract(warning)
    view.set_display_text(warning.peer.helper_text)
    view.set_overlay_peer_contract(
        build_overlay_peer_consumer_contract(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=True,
            peer_effective_enabled=True,
            peer_warning_reason=None,
        )
    )
    assert displays[-1] == warning.peer.helper_text


def test_dashboard_process_warning_does_not_clear_after_status_transition() -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []
    statuses: list[str] = []
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]
    view._sync_stt_button_state = lambda: None  # type: ignore[method-assign]
    view._sync_translation_button_state = lambda: None  # type: ignore[method-assign]
    view._refresh_language_card = lambda: None  # type: ignore[method-assign]
    view._ui_font = lambda: None  # type: ignore[method-assign]
    view._source_lang_code = "en"
    view._capture_controls = SimpleNamespace(apply_locale=lambda: None)
    view.trans_button = SimpleNamespace(set_label=lambda _label: None)
    view.language_card = SimpleNamespace(set_row_labels=lambda *_labels: None)
    view.display_card = SimpleNamespace(
        set_status=lambda status, **_kwargs: statuses.append(status),
        apply_locale=lambda **_kwargs: None,
    )

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_target_exited",
    )
    view.set_overlay_peer_contract(warning)
    view.set_status("connecting")
    view.set_overlay_peer_contract(
        build_overlay_peer_consumer_contract(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=False,
            peer_effective_enabled=False,
            peer_warning_reason=None,
        )
    )
    assert statuses == ["connecting"]
    assert displays == [warning.peer.helper_text]


@pytest.mark.parametrize("peer_enabled", [True, False])
def test_dashboard_unchanged_process_warning_does_not_reacquire_invalidated_primary(
    peer_enabled: bool,
) -> None:
    previous_locale = get_locale()
    set_locale("en")
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []
    statuses: list[str] = []
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]
    view._sync_stt_button_state = lambda: None  # type: ignore[method-assign]
    view._sync_translation_button_state = lambda: None  # type: ignore[method-assign]
    view._refresh_language_card = lambda: None  # type: ignore[method-assign]
    view._ui_font = lambda: None  # type: ignore[method-assign]
    view._source_lang_code = "en"
    view._capture_controls = SimpleNamespace(apply_locale=lambda: None)
    view.trans_button = SimpleNamespace(set_label=lambda _label: None)
    view.language_card = SimpleNamespace(set_row_labels=lambda *_labels: None)
    view.display_card = SimpleNamespace(
        set_status=lambda status, **_kwargs: statuses.append(status),
        apply_locale=lambda **_kwargs: None,
    )

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    warning = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_source_failed",
    )
    view.set_overlay_peer_contract(warning)
    owned_revision = view._process_capture_warning_display_revision
    view.set_overlay_peer_contract(warning)
    assert displays == [warning.peer.helper_text]
    assert view._process_capture_warning_display_revision == owned_revision

    view.set_display_text(warning.peer.helper_text)
    view.set_status("connecting")
    set_locale("ko")
    view.apply_locale()
    view.set_overlay_peer_contract(
        build_overlay_peer_consumer_contract(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=True,
            peer_effective_enabled=False,
            peer_warning_reason="process_source_failed",
        )
    )
    assert statuses == ["connecting"]
    assert displays == [warning.peer.helper_text, warning.peer.helper_text]

    view.set_overlay_peer_contract(
        build_overlay_peer_consumer_contract(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=peer_enabled,
            peer_effective_enabled=peer_enabled,
            peer_warning_reason=None,
        )
    )
    assert displays == [warning.peer.helper_text, warning.peer.helper_text]
    set_locale(previous_locale)


def test_dashboard_changed_process_warning_reclaims_primary_with_new_guidance() -> None:
    view = DashboardView.__new__(DashboardView)
    view._stt_showing_warning = False
    view._translation_showing_warning = False
    view._process_capture_warning_active = False
    view._process_capture_warning_reason = None
    view._process_capture_warning_locale = None
    view._process_capture_warning_text = ""
    view._current_display_text = None
    view._primary_display_revision = 0
    view._process_capture_warning_display_revision = None
    view._overlay_peer_contract = None
    displays: list[str] = []
    view._sync_overlay_peer_buttons = lambda: None  # type: ignore[method-assign]

    def set_display_text(text: str, **_kwargs: object) -> None:
        view._primary_display_revision += 1
        view._current_display_text = text
        displays.append(text)

    view.set_display_text = set_display_text  # type: ignore[method-assign]
    unavailable = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_unavailable_no_process",
    )
    changed = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_warning_reason="process_target_exited",
    )
    view.set_overlay_peer_contract(unavailable)
    view.set_display_text("newer peer content")
    view.set_overlay_peer_contract(changed)
    changed_helper = replace(
        changed,
        peer=replace(changed.peer, helper_text="Changed process warning guidance."),
    )
    view.set_display_text("newer system content")
    view.set_overlay_peer_contract(changed_helper)
    assert displays == [
        unavailable.peer.helper_text,
        "newer peer content",
        changed.peer.helper_text,
        "newer system content",
        changed_helper.peer.helper_text,
    ]
    view.set_overlay_peer_contract(
        build_overlay_peer_consumer_contract(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=True,
            peer_effective_enabled=True,
            peer_warning_reason=None,
        )
    )
    assert displays[-1] == ""


@pytest.mark.asyncio
async def test_app_queues_process_capture_retry_and_apply_actions() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    calls: list[tuple[str, str | None]] = []
    queued: list[object] = []
    app._log_basic = lambda _message: None  # type: ignore[method-assign]
    app.controller = SimpleNamespace(
        retry_peer_process_capture=lambda: _record_async_call(calls, "retry"),
        apply_loopback_capture_option=lambda value: _record_async_call(calls, "apply", value),
    )
    app._queue_settings_mutation_task = queued.append  # type: ignore[method-assign]

    app._on_retry_peer_process_capture()
    app._on_apply_loopback_capture_option("process:discord:stable")

    assert len(queued) == 2
    for task in queued:
        await task()
    assert calls == [("retry", None), ("apply", "process:discord:stable")]


async def _record_async_call(
    calls: list[tuple[str, str | None]],
    action: str,
    value: str | None = None,
) -> None:
    calls.append((action, value))


@pytest.mark.asyncio
async def test_peer_chatbox_deny_still_holds() -> None:
    from tests.core.test_peer_channel_routing import (
        test_peer_desktop_transcripts_are_routed_to_peer_runtime_and_never_sent_to_chatbox,
    )

    await test_peer_desktop_transcripts_are_routed_to_peer_runtime_and_never_sent_to_chatbox()
