from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)


@dataclass(frozen=True, slots=True)
class SelfCaptureApplicationSettings:
    config: SelfCaptureSessionConfig
    provider_id: str
    qwen_region: str | None


@dataclass(slots=True)
class SelfCaptureApplicationOwner:
    settings_provider: Callable[[], SelfCaptureApplicationSettings | None]
    runtime_available: Callable[[], bool]
    capture_owner: Callable[[], SelfCaptureSessionOwner]
    capture_owner_if_created: Callable[[], SelfCaptureSessionOwner | None]
    persist_manual_fallback: Callable[[], bool]
    reset_local_pending: Callable[[], None]
    clear_gpu_pending: Callable[[], None]
    overlay_state_provider: Callable[[], str]
    mark_promo_eligible: Callable[[], None]
    dashboard_enabled_sink: Callable[[bool], None]
    dashboard_needs_key_sink: Callable[[bool], None]
    dashboard_needs_key: Callable[[bool], bool]
    state_sink: Callable[[SelfCaptureSessionSnapshot], None]
    sync_effective_flags: Callable[[], None]
    sync_local_notice: Callable[[], None]
    log_basic: Callable[[str], None]
    log_diagnostic: Callable[[str, int], None]
    restart_requested: bool = False
    force_immediate: bool = False

    async def set_enabled(self, enabled: bool, *, force_immediate: bool = False) -> None:
        if enabled and not self.persist_manual_fallback():
            self.dashboard_enabled_sink(False)
            return
        self.log_basic(f"[STT] Toggle request: enabled={enabled}")
        owner = self.capture_owner_if_created()
        self.log_diagnostic(
            "[STT] Toggle detail: "
            f"desired_before={owner is not None and owner.snapshot.desired_active} "
            f"overlay_state={self.overlay_state_provider()}",
            logging.INFO,
        )
        self.force_immediate = force_immediate
        if not enabled:
            self.reset_local_pending()
            self.clear_gpu_pending()
        settings = self.settings_provider()
        if enabled and settings is not None:
            provider = settings.provider_id
            self.log_basic(f"[STT] Enabled with provider: {provider}")
            if provider == "qwen_audio" and settings.qwen_region is not None:
                self.log_diagnostic(
                    f"[STT] Provider detail: provider={provider} " f"region={settings.qwen_region}",
                    logging.INFO,
                )
        if enabled and self.runtime_available():
            self.mark_promo_eligible()
        snapshot = await self.run_switch(desired=enabled)
        if snapshot is None or snapshot.state is not SelfCaptureSessionState.ADMISSION_PENDING:
            self.dashboard_enabled_sink(
                bool(
                    snapshot is not None
                    and snapshot.desired_active
                    and snapshot.state is not SelfCaptureSessionState.FAULTED
                    and snapshot.has_loop_task
                )
            )
        self.sync_local_notice()

    async def run_switch(
        self,
        *,
        desired: bool | None = None,
    ) -> SelfCaptureSessionSnapshot | None:
        settings = self.settings_provider()
        if settings is None:
            self.log_diagnostic(
                "[STT] Enable requested before Self translation owner is ready",
                logging.WARNING,
            )
            return None
        restart = self.restart_requested
        force_immediate = self.force_immediate
        self.restart_requested = False
        self.force_immediate = False
        owner = self.capture_owner()
        next_desired = owner.snapshot.desired_active if desired is None else desired
        snapshot = await owner.apply_intent(
            settings.config,
            enabled=next_desired,
            restart=restart,
            force_immediate=force_immediate,
            explicit_toggle_off=not next_desired,
        )
        self.state_sink(snapshot)
        return snapshot

    async def replace_provider(self, *, smooth_local: bool = False) -> None:
        owner = self.capture_owner_if_created()
        self.log_diagnostic(
            "[STT] Replacing runtime provider detail: "
            f"desired={owner is not None and owner.snapshot.desired_active} "
            f"mic_task_active={owner is not None and owner.loop_task is not None}",
            logging.INFO,
        )
        settings = self.settings_provider()
        if settings is None or not self.runtime_available():
            return
        requested_provider = settings.provider_id
        current_provider = (
            getattr(owner.snapshot, "provider_id", None) if owner is not None else None
        )
        self.log_diagnostic(
            "[STT][Runtime] provider refresh requested: "
            f"current={current_provider or 'none'} requested={requested_provider}",
            logging.INFO,
        )
        owner = self.capture_owner()
        config = settings.config
        if owner.snapshot.desired_active:
            snapshot = await owner.apply_intent(
                config,
                enabled=True,
                restart=not smooth_local,
                explicit_toggle_off=False,
            )
        else:
            snapshot = await owner.prepare_provider(config)
        if (
            snapshot.provider_status is SelfCaptureProviderStatus.READY
            and snapshot.failure_reason is None
            and snapshot.runtime_signature == config.runtime_signature
        ):
            if snapshot.desired_active and snapshot.effective_active:
                self.log_diagnostic(
                    "[STT][Runtime] provider handoff committed: "
                    f"provider={getattr(snapshot, 'provider_id', config.provider_id)}",
                    logging.INFO,
                )
            else:
                self.log_diagnostic(
                    "[STT][Runtime] provider prepared: "
                    f"provider={getattr(snapshot, 'provider_id', config.provider_id)}",
                    logging.INFO,
                )
        elif snapshot.provider_status is SelfCaptureProviderStatus.PENDING:
            self.log_diagnostic(
                "[STT][Runtime] provider handoff pending: "
                f"requested={config.provider_id} reason={snapshot.admission_reason or 'boundary'}",
                logging.INFO,
            )
        self.state_sink(snapshot)
        self.project_availability(snapshot)
        self.restart_requested = False
        if (
            snapshot.failure_reason is not None
            or snapshot.runtime_signature != config.runtime_signature
        ):
            raise RuntimeError("Self STT runtime did not apply the requested configuration")

    def project_availability(self, snapshot: SelfCaptureSessionSnapshot) -> bool:
        available = snapshot.provider_status is SelfCaptureProviderStatus.READY
        settings = self.settings_provider()
        if settings is not None:
            self.sync_effective_flags()
        self.dashboard_needs_key_sink(self.dashboard_needs_key(available))
        if not available:
            self.dashboard_enabled_sink(False)
        return available


__all__ = [
    "SelfCaptureApplicationOwner",
    "SelfCaptureApplicationSettings",
]
