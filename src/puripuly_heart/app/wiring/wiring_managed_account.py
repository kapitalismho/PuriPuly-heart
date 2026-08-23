from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.managed_auth import ManagedAuthOwner
from puripuly_heart.app.services.managed_gemma_translation import ManagedGemmaTranslationOwner
from puripuly_heart.app.services.managed_usage import (
    ManagedUsageMetadataResult,
    ManagedUsageOwner,
    ManagedUsageState,
    ManagedUsageViewState,
)
from puripuly_heart.app.services.openrouter_pkce_flow import (
    OpenRouterPkceApplicationOwner,
    OpenRouterPkceFlowOwner,
)
from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeOwner
from puripuly_heart.app.services.provider_settings import ProviderSettingsOwner
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.app.services.translation_enable import TranslationEnableOwner
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    TranslationConnection,
    TranslationModel,
    normalize_owned_referral_id,
)
from puripuly_heart.core.hardware_fingerprint import get_raw_hardware_fingerprint
from puripuly_heart.core.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseService,
    UnavailableManagedOpenRouterReleaseClient,
)
from puripuly_heart.core.openrouter_credentials import resolve_openrouter_credentials
from puripuly_heart.core.openrouter_handoff import should_auto_show_founder_letter
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEClient
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationPort,
)
from puripuly_heart.core.telemetry import TranslationSuccessTelemetryClientPort

from .wiring_managed_auth_factory import (
    ManagedAuthRuntimeAdapter,
    ManagedTranslationRuntimeAdapter,
    build_managed_identity_state_port,
    build_openrouter_credential_runtime_config,
    build_openrouter_release_runtime_config,
)
from .wiring_managed_gemma import managed_gemma_selection
from .wiring_secrets_factory import create_secret_store
from .wiring_translation_runtime_configuration import (
    replace_translation_runtime_enabled,
)

logger = logging.getLogger(__name__)

_MANAGED_CONNECTIONS = frozenset(
    {
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
    }
)


class ManagedTranslationContextPort(Protocol):
    def clear_context(self) -> None: ...


class ManagedLlmRuntimePort(Protocol):
    provider: object | None


@dataclass(slots=True)
class ManagedTranslationRuntimeAccess:
    llm_runtime_provider: Callable[[], ManagedLlmRuntimePort | None]
    context_provider: Callable[[], ManagedTranslationContextPort | None]
    translation_runtime_configuration_provider: Callable[
        [],
        TranslationRuntimeConfigurationPort | None,
    ]
    rebuild_llm: Callable[[], Awaitable[object]]

    def presence(self) -> tuple[bool, bool]:
        runtime = self.llm_runtime_provider()
        return runtime is not None, runtime is not None and runtime.provider is not None

    def snapshot(self) -> tuple[bool, bool, object | None]:
        runtime = self.llm_runtime_provider()
        config_owner = self.translation_runtime_configuration_provider()
        translation_enabled = (
            config_owner.snapshot().value.translation_enabled if config_owner is not None else False
        )
        return (
            runtime is not None,
            translation_enabled,
            runtime.provider if runtime is not None else None,
        )

    async def ensure(self, mode: str) -> bool:
        runtime = self.llm_runtime_provider()
        if runtime is None:
            return False
        if mode == "always" or (mode == "if_missing" and runtime.provider is None):
            await self.rebuild_llm()
        return runtime.provider is not None

    def set_enabled(self, enabled: bool) -> None:
        config_owner = self.translation_runtime_configuration_provider()
        if config_owner is not None:
            replace_translation_runtime_enabled(config_owner, enabled)

    def clear_context(self) -> None:
        context = self.context_provider()
        if context is not None:
            context.clear_context()


@dataclass(slots=True)
class ManagedOpenRouterReleaseRuntime:
    settings: SettingsOwner
    config_path: Path
    callback_received: Callable[[], None]
    diagnostic_sink: Callable[[str, BaseException], None] = field(
        default=lambda _message, _exception: None,
        repr=False,
    )
    service: ManagedOpenRouterReleaseService | None = field(default=None, repr=False)
    telemetry_client: TranslationSuccessTelemetryClientPort | None = field(
        default=None,
        repr=False,
    )
    _retired_services: list[ManagedOpenRouterReleaseService] = field(
        default_factory=list,
        repr=False,
    )

    def selected(self, settings: AppSettings | None = None) -> bool:
        current = settings if settings is not None else self.settings.current
        return bool(
            current is not None
            and current.provider.llm == LLMProviderName.OPENROUTER
            and current.translation.connection in _MANAGED_CONNECTIONS
            and current.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        )

    def release_settings(self) -> AppSettings | None:
        current = self.settings.current
        return current if self.selected(current) else None

    async def rebuild(self, *, secrets: object) -> ManagedOpenRouterReleaseService | None:
        await self.retry_retired()
        replacement, telemetry_client = self._create(secrets=secrets)
        previous = self.service
        self.service = replacement
        self.telemetry_client = telemetry_client
        if previous is not None and previous is not replacement:
            try:
                await previous.close()
            except BaseException as exc:
                self._retired_services.append(previous)
                with contextlib.suppress(Exception):
                    self.diagnostic_sink(
                        "[Managed OpenRouter] Retired release service close failed",
                        exc,
                    )
                raise
        return replacement

    async def retry_retired(self) -> None:
        failures: list[BaseException] = []
        remaining: list[ManagedOpenRouterReleaseService] = []
        for service in self._retired_services:
            try:
                await service.close()
            except BaseException as exc:
                remaining.append(service)
                failures.append(exc)
                with contextlib.suppress(Exception):
                    self.diagnostic_sink(
                        "[Managed OpenRouter] Retired release service retry failed",
                        exc,
                    )
        self._retired_services = remaining
        if failures:
            raise BaseExceptionGroup(
                "managed OpenRouter retired release cleanup failed",
                failures,
            )

    def _create(
        self,
        *,
        secrets: object,
    ) -> tuple[
        ManagedOpenRouterReleaseService | None,
        TranslationSuccessTelemetryClientPort | None,
    ]:
        current = self.settings.current
        release_settings = self.release_settings()
        if current is None or release_settings is None:
            return None, None

        from puripuly_heart import __version__

        try:
            client = HttpManagedOpenRouterBrokerClient(
                base_url=current.openrouter.broker_base_url,
            )
            telemetry_client: TranslationSuccessTelemetryClientPort | None = client
        except ValueError as exc:
            logger.warning(
                "[Managed OpenRouter] Invalid broker base URL %r; using unavailable fallback: %s",
                current.openrouter.broker_base_url,
                exc,
            )
            client = UnavailableManagedOpenRouterReleaseClient()
            telemetry_client = None

        return (
            ManagedOpenRouterReleaseService(
                openrouter_config=build_openrouter_release_runtime_config(release_settings),
                managed_state=build_managed_identity_state_port(
                    current,
                    self.settings.managed_identity_persistence_callback(current),
                ),
                secrets=secrets,
                client=client,
                raw_hardware_fingerprint_provider=get_raw_hardware_fingerprint,
                app_version=__version__,
                on_discord_callback_received=self.callback_received,
            ),
            telemetry_client,
        )

    async def close(self) -> None:
        service = self.service
        failures: list[BaseException] = []
        if service is not None:
            try:
                await service.close()
            except BaseException as exc:
                failures.append(exc)
                with contextlib.suppress(Exception):
                    self.diagnostic_sink(
                        "[Managed OpenRouter] Active release service close failed",
                        exc,
                    )
            else:
                if self.service is service:
                    self.service = None
                    self.telemetry_client = None
        try:
            await self.retry_retired()
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise BaseExceptionGroup(
                "managed OpenRouter release cleanup failed",
                failures,
            )


@dataclass(slots=True)
class ManagedUsageRuntimeAdapter:
    settings: SettingsOwner
    release: ManagedOpenRouterReleaseRuntime
    config_path: Path
    verifier: ProviderVerifierPort
    ingress_provider: Callable[[], bool]

    def state(self) -> ManagedUsageState:
        current = self.settings.current
        if current is None:
            return ManagedUsageState(
                settings_available=False,
                managed_key_visible=False,
                release_settings_available=False,
                installation_id=None,
                entitlement_ref=None,
                referral_id=None,
                ingress_frozen=self.ingress_provider(),
            )
        active_ref = current.managed_identity.active_managed_credential_ref
        entitlement_ref = active_ref.strip() if isinstance(active_ref, str) else None
        return ManagedUsageState(
            settings_available=True,
            managed_key_visible=self.release.selected(current),
            release_settings_available=self.release.release_settings() is not None,
            installation_id=current.managed_identity.installation_id.strip() or None,
            entitlement_ref=entitlement_ref or None,
            referral_id=normalize_owned_referral_id(current.managed_identity.referral_id),
            ingress_frozen=self.ingress_provider(),
        )

    async def fetch_metadata(self) -> ManagedUsageMetadataResult:
        current = self.settings.current
        release_settings = self.release.release_settings()
        if current is None or release_settings is None:
            return ManagedUsageMetadataResult(key_available=False, metadata=None)
        try:
            secrets = create_secret_store(current.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config(release_settings),
                secrets=secrets,
            )
        except Exception:
            return ManagedUsageMetadataResult(key_available=False, metadata=None)
        if not resolution.api_key:
            return ManagedUsageMetadataResult(key_available=False, metadata=None)
        metadata = await self.verifier.fetch_openrouter_key_metadata(resolution.api_key)
        return ManagedUsageMetadataResult(key_available=True, metadata=metadata)

    def should_auto_show_founder_letter(self, metadata: object) -> bool:
        current = self.settings.current
        if current is None:
            return False
        return should_auto_show_founder_letter(
            build_managed_identity_state_port(current, lambda _settings: None),
            metadata,
        )


@dataclass(frozen=True, slots=True)
class ManagedAccountComponents:
    release: ManagedOpenRouterReleaseRuntime
    auth: ManagedAuthOwner
    usage: ManagedUsageOwner
    translation: TranslationEnableOwner
    pkce_flow: OpenRouterPkceFlowOwner
    pkce: OpenRouterPkceApplicationOwner


def compose_managed_account(
    *,
    config_path: Path,
    settings: SettingsOwner,
    provider_settings: ProviderSettingsOwner,
    provider_runtime: ProviderRuntimeOwner,
    verifier: ProviderVerifierPort,
    results: SettingsTransactionResultOwner,
    runtime: ManagedTranslationRuntimeAccess,
    ingress_provider: Callable[[], bool],
    pending_sink: Callable[[bool], None],
    usage_view_sink: Callable[[ManagedUsageViewState], None],
    dashboard_sink: Callable[[bool], None],
    starting_sink: Callable[[bool], None] | None = None,
    message_sink: Callable[[str, Mapping[str, object]], None],
    qq_dialog_sink: Callable[[], None],
    founder_dialog: Callable[[], bool],
    failure_route: Callable[[str], None],
    log_basic: Callable[[str], None],
    log_detailed: Callable[[str], None],
    log_error: Callable[[str], None],
    basic_warning_sink: Callable[[str], None],
    detailed_warning_sink: Callable[[str, BaseException | None], None],
    runtime_state_changed: Callable[[], None] | None = None,
    managed_gemma: ManagedGemmaTranslationOwner | None = None,
    sync_local_translation_demand: Callable[[], Awaitable[None]] | None = None,
) -> ManagedAccountComponents:
    auth_owner: ManagedAuthOwner | None = None

    def callback_received() -> None:
        if auth_owner is not None:
            auth_owner.on_callback_received()

    release = ManagedOpenRouterReleaseRuntime(
        settings=settings,
        config_path=config_path,
        callback_received=callback_received,
        diagnostic_sink=detailed_warning_sink,
    )
    auth_adapter = ManagedAuthRuntimeAdapter(
        config_path=config_path,
        secret_store_factory=create_secret_store,
        settings_provider=lambda: settings.current,
        settings_sink=lambda value: setattr(settings, "current", value),
        release_service_provider=lambda: release.service,
        persistence_callback_factory=settings.managed_identity_persistence_callback,
        settings_repository_factory=lambda base, committed, surface: (
            settings.create_legacy_patch_repository(
                base_settings=base,
                committed_settings=committed,
                surface=surface,
                save_failure_sink=log_error,
            )
        ),
        settings_owner_complete=settings.complete,
        runtime_presence_provider=runtime.presence,
        ingress_provider=ingress_provider,
    )
    usage_adapter = ManagedUsageRuntimeAdapter(
        settings=settings,
        release=release,
        config_path=config_path,
        verifier=verifier,
        ingress_provider=ingress_provider,
    )
    usage_owner: ManagedUsageOwner | None = None

    def current_usage() -> ManagedUsageOwner:
        if usage_owner is None:
            raise RuntimeError("managed usage composition is incomplete")
        return usage_owner

    auth_owner = ManagedAuthOwner(
        state_provider=auth_adapter.state,
        pending_sink=pending_sink,
        qq_executor=auth_adapter.execute_qq,
        discord_executor=auth_adapter.execute_discord,
        runtime_ensurer=runtime.ensure,
        usage_view_sink=lambda referral_id, pass_status: current_usage().set_view_state(
            visible=True,
            remaining_percent=None,
            referral_id=referral_id or current_usage().current_referral_id,
            pass_status=pass_status,
        ),
        usage_refresh_sink=lambda: current_usage().schedule_usage_refresh(),
        message_sink=message_sink,
        result_sink=results.set,
        log_sink=log_error,
    )

    def warning_sink(message: str, exception: BaseException | None) -> None:
        if message.startswith("[ManagedAuth] Background refresh failed"):
            detailed_warning_sink(message, exception)
            return
        basic_warning_sink(message)

    translation_adapter = ManagedTranslationRuntimeAdapter(
        auth=auth_adapter,
        settings_provider=lambda: settings.current,
        release_service_provider=lambda: release.service,
        runtime_snapshot_provider=runtime.snapshot,
        ingress_provider=ingress_provider,
        founder_dialog=founder_dialog,
        persist_settings=lambda: settings.save_current(
            failure_sink=lambda exc: log_error(f"Failed to save settings: {exc}")
        ),
    )
    translation_owner: TranslationEnableOwner | None = None

    async def warmup_translation() -> None:
        current = settings.current
        if (
            current is not None
            and current.translation.model
            in {TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_12B}
            and managed_gemma is not None
        ):
            if sync_local_translation_demand is not None:
                await sync_local_translation_demand()
            else:
                await managed_gemma.prepare(managed_gemma_selection(current))
            return
        await translation_adapter.warmup()

    async def teardown_translation() -> None:
        if sync_local_translation_demand is not None:
            await sync_local_translation_demand()
            return
        if managed_gemma is not None:
            await managed_gemma.deactivate(linger=True)

    def disable_translation(reopen: bool) -> None:
        if translation_owner is not None:
            translation_owner.disable_for_managed_exhaustion(
                reopen_founder_letter=reopen,
            )

    usage_owner = ManagedUsageOwner(
        state_provider=usage_adapter.state,
        release_service_provider=lambda: release.service,
        metadata_fetcher=usage_adapter.fetch_metadata,
        pending_sink=auth_owner.set_pending,
        view_sink=usage_view_sink,
        disable_translation_sink=disable_translation,
        auto_show_founder_letter_provider=usage_adapter.should_auto_show_founder_letter,
        normalize_referral_id=normalize_owned_referral_id,
        warning_sink=warning_sink,
    )
    translation_owner = TranslationEnableOwner(
        state_provider=translation_adapter.state,
        managed_prepare=translation_adapter.prepare,
        founder_route=usage_owner.should_route_to_founder_letter,
        pending_sink=auth_owner.set_pending,
        runtime_ensurer=runtime.ensure,
        usage_refresh_sink=usage_owner.schedule_usage_refresh,
        usage_refresh_now=lambda: usage_owner.refresh(auto_show_founder_letter=False),
        runtime_sink=lambda enabled: _set_runtime_enabled(
            runtime,
            enabled,
            runtime_state_changed,
        ),
        dashboard_sink=dashboard_sink,
        starting_sink=starting_sink,
        clear_context=runtime.clear_context,
        warmup=warmup_translation,
        message_sink=message_sink,
        qq_dialog_sink=qq_dialog_sink,
        result_sink=results.set,
        log_basic=log_basic,
        log_detailed=log_detailed,
        log_error=log_error,
        founder_letter_sink=translation_adapter.show_founder_letter,
        teardown=teardown_translation,
    )
    pkce_flow = OpenRouterPkceFlowOwner(
        client_factory=lambda: OpenRouterPKCEClient(
            callback_origin="http://localhost:3000",
        )
    )
    pkce = OpenRouterPkceApplicationOwner(
        flow=pkce_flow,
        verifier=verifier,
        settings=settings,
        provider_settings=provider_settings,
        provider_runtime=provider_runtime,
        secret_store_factory=lambda current: SyncSecretStoreAdapter(
            create_secret_store(current.secrets, config_path=config_path)
        ),
        failure_message_sink=lambda key: message_sink(key, {}),
        failure_diagnostics_sink=log_error,
        failure_route=failure_route,
        results=results,
    )
    return ManagedAccountComponents(
        release=release,
        auth=auth_owner,
        usage=usage_owner,
        translation=translation_owner,
        pkce_flow=pkce_flow,
        pkce=pkce,
    )


def _set_runtime_enabled(
    runtime: ManagedTranslationRuntimeAccess,
    enabled: bool,
    state_changed: Callable[[], None] | None,
) -> None:
    runtime.set_enabled(enabled)
    if state_changed is not None:
        state_changed()


__all__ = [
    "ManagedAccountComponents",
    "ManagedOpenRouterReleaseRuntime",
    "ManagedTranslationRuntimeAccess",
    "ManagedUsageRuntimeAdapter",
    "compose_managed_account",
]
