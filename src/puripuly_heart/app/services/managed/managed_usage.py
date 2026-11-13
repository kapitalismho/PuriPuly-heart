from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace

from puripuly_heart.core.diagnostics import emit_failure, exception_diagnostics, safe_exception_fields
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.openrouter.openrouter_handoff import is_effectively_exhausted
from puripuly_heart.core.openrouter.openrouter_metadata import OpenRouterKeyMetadata

from .managed_status_refresh import ManagedStatusRefreshOwner

ManagedUsageStateProvider = Callable[[], "ManagedUsageState"]
ManagedUsageReleaseServiceProvider = Callable[[], object | None]
ManagedUsageMetadataFetcher = Callable[[], Awaitable["ManagedUsageMetadataResult"]]
ManagedUsagePendingSink = Callable[[bool], None]
ManagedUsageViewSink = Callable[["ManagedUsageViewState"], None]
ManagedUsageDisableTranslationSink = Callable[[bool], None]
ManagedUsageAutoShowFounderLetterProvider = Callable[[OpenRouterKeyMetadata | None], bool]
ManagedUsageWarningSink = Callable[[str, BaseException | None], None]


@dataclass(frozen=True, slots=True)
class ManagedUsageState:
    settings_available: bool
    managed_key_visible: bool
    release_settings_available: bool
    installation_id: str | None
    entitlement_ref: str | None
    ingress_frozen: bool


@dataclass(frozen=True, slots=True)
class ManagedUsageViewState:
    visible: bool
    remaining_percent: int | None


@dataclass(frozen=True, slots=True)
class ManagedUsageMetadataResult:
    key_available: bool
    metadata: OpenRouterKeyMetadata | None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0
    scope: ManagedUsageState | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        excess = max(0, len(self.secondary_diagnostics) - 4)
        if excess:
            object.__setattr__(self, "secondary_diagnostics", self.secondary_diagnostics[:4])
            object.__setattr__(self, "secondary_omitted_count", self.secondary_omitted_count + excess)


@dataclass(slots=True)
class ManagedUsageOwner:
    state_provider: ManagedUsageStateProvider
    release_service_provider: ManagedUsageReleaseServiceProvider
    metadata_fetcher: ManagedUsageMetadataFetcher
    pending_sink: ManagedUsagePendingSink
    view_sink: ManagedUsageViewSink
    disable_translation_sink: ManagedUsageDisableTranslationSink
    auto_show_founder_letter_provider: ManagedUsageAutoShowFounderLetterProvider
    warning_sink: ManagedUsageWarningSink
    usage_metadata: OpenRouterKeyMetadata | None = None
    usage_metadata_entitlement_ref: str | None = None
    refresh_owner: ManagedStatusRefreshOwner = field(init=False, repr=False)
    last_result: ManagedUsageMetadataResult | None = field(init=False, default=None)
    superseded_result: ManagedUsageMetadataResult | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.refresh_owner = ManagedStatusRefreshOwner(
            diagnostics_sink=self._on_refresh_diagnostic,
        )

    @property
    def remaining_percent(self) -> int | None:
        return self.remaining_percent_for(self.usage_metadata)

    @property
    def is_exhausted(self) -> bool:
        return is_effectively_exhausted(self.usage_metadata)

    @staticmethod
    def remaining_percent_for(metadata: OpenRouterKeyMetadata | None) -> int | None:
        if (
            metadata is None
            or metadata.limit_usd is None
            or metadata.remaining_usd is None
            or metadata.limit_usd <= 0
        ):
            return None
        return max(0, min(100, round((metadata.remaining_usd / metadata.limit_usd) * 100)))

    def set_managed_key_state(self, *, visible: bool, remaining_percent: int | None) -> None:
        self.view_sink(ManagedUsageViewState(visible=visible, remaining_percent=remaining_percent))

    def schedule_usage_refresh(self) -> bool:
        if self.state_provider().ingress_frozen:
            return False
        return self.refresh_owner.schedule_trial_usage_refresh(self.refresh_best_effort)

    def delegate_ready(self) -> None:
        if self.state_provider().ingress_frozen:
            return
        self.pending_sink(False)
        self.schedule_usage_refresh()

    async def refresh_best_effort(self) -> ManagedUsageMetadataResult | None:
        try:
            return await self.refresh()
        except Exception as exc:
            self._warn("[ManagedAuth] Usage refresh failed", exc)
            return self.last_result

    async def refresh(
        self, *, auto_show_founder_letter: bool = True
    ) -> ManagedUsageMetadataResult | None:
        state = self.state_provider()
        if state.ingress_frozen:
            return
        if not state.settings_available or not state.managed_key_visible:
            self.clear_usage_metadata()
            self.pending_sink(False)
            self.set_managed_key_state(visible=False, remaining_percent=None)
            return
        if not state.release_settings_available:
            self.clear_usage_metadata()
            self.pending_sink(False)
            self.set_managed_key_state(visible=True, remaining_percent=None)
            return
        entitlement_ref = self.sync_usage_metadata_scope(state)
        try:
            metadata_result = await self.metadata_fetcher()
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            if not isinstance(diagnostics, ErrorDiagnostics):
                diagnostics = exception_diagnostics(
                    exc,
                    component="managed_usage",
                    operation="managed_usage_refresh",
                    code="managed_usage_refresh_failed",
                    fields={"phase": "managed_usage_refresh"},
                )
            if type(exc).__module__ == "builtins":
                exc.diagnostics = diagnostics
            self._publish_result(
                ManagedUsageMetadataResult(False, self.usage_metadata, diagnostics, scope=state),
                self.state_provider(),
            )
            raise
        metadata_result = replace(metadata_result, scope=state)
        if not self._publish_result(metadata_result, self.state_provider()):
            self.clear_usage_metadata()
            return
        if metadata_result.key_available:
            self.pending_sink(False)
        metadata = metadata_result.metadata
        self.usage_metadata = metadata
        self.usage_metadata_entitlement_ref = entitlement_ref
        self.set_managed_key_state(visible=True, remaining_percent=self.remaining_percent)
        if (
            auto_show_founder_letter
            and is_effectively_exhausted(metadata)
            and self.auto_show_founder_letter_provider(metadata)
        ):
            self.disable_translation_sink(True)
        return metadata_result

    @staticmethod
    def _scope_current(admitted: ManagedUsageState, current: ManagedUsageState) -> bool:
        return (
            not current.ingress_frozen
            and current.settings_available
            and current.managed_key_visible
            and current.release_settings_available
            and current.installation_id == admitted.installation_id
            and current.entitlement_ref == admitted.entitlement_ref
        )

    def _publish_result(
        self, result: ManagedUsageMetadataResult, current: ManagedUsageState
    ) -> bool:
        if result.scope is not None and self._scope_current(result.scope, current):
            self.last_result = result
            return True
        self.superseded_result = result
        for diagnostics in (
            *((result.diagnostics,) if result.diagnostics is not None else ()),
            *result.secondary_diagnostics,
        ):
            emit_failure(
                logging.getLogger(__name__), diagnostics, severity="warning", disposition="superseded"
            )
        return False

    def current_result(self, state: ManagedUsageState | None = None) -> ManagedUsageMetadataResult | None:
        result = self.last_result
        current = state or self.state_provider()
        return (
            result
            if result is not None
            and result.scope is not None
            and self._scope_current(result.scope, current)
            else None
        )

    async def should_route_to_founder_letter(self) -> bool:
        if not self.state_provider().settings_available:
            return False
        try:
            await self.refresh(auto_show_founder_letter=False)
        except Exception as exc:
            self._warn("[ManagedAuth] Usage refresh failed", exc, notify=False)
        if not is_effectively_exhausted(self.usage_metadata):
            return False
        self.disable_translation_sink(True)
        return True

    def clear_usage_metadata(self) -> None:
        self.usage_metadata = None
        self.usage_metadata_entitlement_ref = None
        self.last_result = None

    def sync_usage_metadata_scope(
        self,
        state: ManagedUsageState | None = None,
    ) -> str | None:
        current = state or self.state_provider()
        if not current.settings_available:
            self.clear_usage_metadata()
            return None
        if current.entitlement_ref != self.usage_metadata_entitlement_ref:
            self.usage_metadata = None
            self.usage_metadata_entitlement_ref = current.entitlement_ref
        if self.last_result is not None and self.current_result(current) is None:
            self.last_result = None
        return current.entitlement_ref

    async def close(self) -> None:
        await self.refresh_owner.close()

    def stop_ingress(self) -> None:
        self.refresh_owner.stop_ingress()

    def _on_refresh_diagnostic(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None,
    ) -> None:
        self._warn(
            "[ManagedAuth] Background refresh failed "
            f"event={event} kind={metadata.get('kind')} "
            f"error_type={metadata.get('error_type')}",
            exception,
        )

    def _warn(
        self, message: str, exception: BaseException | None = None, *, notify: bool = True
    ) -> None:
        diagnostics = getattr(exception, "diagnostics", None)
        observation = None
        if exception is not None:
            if not isinstance(diagnostics, ErrorDiagnostics):
                diagnostics = exception_diagnostics(
                    exception,
                    component="managed_usage",
                    operation="managed_usage_refresh",
                    code="managed_usage_refresh_failed",
                    fields={"phase": "managed_usage_refresh"},
                )
            observation = next(
                (
                    result for result in (self.last_result, self.superseded_result)
                    if result is not None
                    and result.diagnostics is not None
                    and result.diagnostics.diagnostic_id == diagnostics.diagnostic_id
                ),
                None,
            )
            emit_failure(
                logging.getLogger(__name__), diagnostics, severity="warning", disposition="degraded"
            )
        if not notify:
            return
        try:
            self.warning_sink(message, exception)
        except Exception as exc:
            secondary = ErrorDiagnostics(
                component="managed_usage",
                operation="auth_notification",
                code="managed_usage_notification_failed",
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={"phase": "auth_notification", **safe_exception_fields(exc)},
                cause_id=(
                    diagnostics.diagnostic_id if isinstance(diagnostics, ErrorDiagnostics) else None
                ),
            )
            disposition = "degraded"
            if observation is not None:
                updated = replace(
                    observation,
                    secondary_diagnostics=(*observation.secondary_diagnostics, secondary),
                )
                if self.last_result is observation and self.current_result() is observation:
                    self.last_result = updated
                else:
                    self.superseded_result = updated
                    disposition = "superseded"
            emit_failure(
                logging.getLogger(__name__), secondary, severity="warning", disposition=disposition
            )


__all__ = [
    "ManagedUsageOwner",
    "ManagedUsageMetadataResult",
    "ManagedUsageState",
    "ManagedUsageViewState",
]
