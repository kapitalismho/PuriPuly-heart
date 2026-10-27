from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.broker_client import (
    BrokerIssueRequest,
    BrokerIssueResult,
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckRequest,
    ManagedKeyDeliveryAckResult,
    ManagedOperationResumeRequest,
    ManagedOperationStatusRequest,
    ManagedOperationStatusResult,
)
from puripuly_heart.app.ports.discord_auth import DiscordAuthRequest, DiscordAuthResult
from puripuly_heart.app.ports.managed_identity import (
    ManagedIdentityPreflightRequest,
    ManagedIdentityPreflightResult,
)
from puripuly_heart.app.ports.managed_identity_state import (
    ManagedIdentitySnapshot,
    ManagedIdentityStatePort,
)
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsCommitResult,
    SettingsRepositoryPort,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.github_star_prompt import (
    github_star_prompt_utc_timestamp,
)
from puripuly_heart.app.services.managed.managed_auth import (
    ManagedAuthExecutionResult,
    ManagedAuthState,
)
from puripuly_heart.app.services.managed.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    ManagedAuthClaimGuard,
)
from puripuly_heart.app.services.managed.managed_connection_auth import (
    ManagedConnectionAuthRequest,
    ManagedConnectionAuthService,
)
from puripuly_heart.app.services.managed.managed_key_delivery_ack import (
    ManagedKeyDeliveryAckService,
)
from puripuly_heart.app.services.qq_managed_auth import (
    QqManagedAuthRequest,
    QqManagedAuthService,
)
from puripuly_heart.app.services.translation_enable import (
    ManagedTranslationPreparation,
    TranslationEnableState,
)
from puripuly_heart.config.llm_profiles import (
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.provider_values import (
    OpenRouterCredentialSource,
    OpenRouterSelectionAlias,
    parse_openrouter_llm_model,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)
from puripuly_heart.core.diagnostics import safe_exception_fields, sanitize_diagnostics
from puripuly_heart.core.discord.discord_managed_oauth import (
    DiscordOAuthBrowserLaunchError,
    DiscordOAuthStateError,
    register_discord_oauth_expected_state,
    run_discord_oauth_callback_flow,
)
from puripuly_heart.core.discord.discord_oauth_loopback import (
    DiscordOAuthCallbackError,
    DiscordOAuthLoopbackClosedError,
    DiscordOAuthLoopbackListener,
)
from puripuly_heart.core.hardware_fingerprint import compute_hardware_hash
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.managed_identity import (
    ManagedIdentityBundle,
    ensure_managed_identity_bundle,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    SEVERITY_ERROR,
    TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    DiagnosticCategory,
    ErrorDiagnostics,
    TransactionResult,
    UserMessageRef,
)
from puripuly_heart.core.openrouter.managed_openrouter_release import (
    MANAGED_OPENROUTER_TRIAL_BUDGET_USD,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseError,
    OpenRouterReleaseRuntimeConfig,
    format_managed_openrouter_diagnostics,
)
from puripuly_heart.core.openrouter.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    OpenRouterCredentialRuntimeConfig,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.openrouter.openrouter_handoff import mark_founder_letter_shown
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.qq_auth_diagnostics import exception_fields, log_qq_auth
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.storage.secrets import SecretStore

HardwareFingerprintProvider = Callable[[], str | Awaitable[str]]
DiscordOAuthListenerFactory = Callable[[], DiscordOAuthLoopbackListener]
DiscordOAuthCallbackRunner = Callable[
    [DiscordOAuthLoopbackListener, str, str],
    Awaitable[tuple[str, str]],
]


class ManagedIdentityRecord(Protocol):
    installation_id: str
    release_token: str | None
    release_token_expires_at: str | None
    verified_hardware_hash: str | None
    verified_hardware_hash_salt_version: int | None
    active_managed_credential_ref: str | None
    active_managed_expires_at: str | None
    founder_letter_seen_credential_ref: str | None
    local_managed_claim_sources: tuple[str, ...]
    pending_delivery_ack_source: str | None
    pending_delivery_ack_delivery_id: str | None
    pending_delivery_ack_managed_credential_ref: str | None
    pending_delivery_ack_expires_at: str | None
    pending_managed_operation_id: str | None
    pending_managed_operation_source: str | None
    pending_managed_operation_installation_id: str | None
    pending_managed_operation_state: str | None


class ManagedIdentityStateAdapter:
    def __init__(self, identity: ManagedIdentityRecord, persist: Callable[..., None]) -> None:
        object.__setattr__(self, "_settings_bag", None)
        managed = getattr(identity, "managed_identity", None)
        if managed is not None:
            object.__setattr__(self, "_settings_bag", identity)
            identity = managed
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_persist", persist)

    def __getattr__(self, name: str) -> object:
        return getattr(self._identity, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._identity, name, value)

    def persist(self) -> None:
        if self._settings_bag is not None:
            self._persist(self._settings_bag)
        else:
            self._persist()

    def snapshot(self) -> ManagedIdentitySnapshot:
        from dataclasses import fields

        return ManagedIdentitySnapshot(
            **{item.name: getattr(self, item.name) for item in fields(ManagedIdentitySnapshot)}
        )

    def restore(self, snapshot: ManagedIdentitySnapshot) -> None:
        from dataclasses import fields

        for item in fields(snapshot):
            setattr(self, item.name, getattr(snapshot, item.name))


def build_managed_identity_state_port(
    identity: ManagedIdentityRecord,
    persist: Callable[[], None],
) -> ManagedIdentityStatePort:
    """Build a ``ManagedIdentityStatePort`` adapter at the wiring boundary."""

    return cast(ManagedIdentityStatePort, ManagedIdentityStateAdapter(identity, persist))


class CanonicalManagedIdentityStateAdapter:
    def __init__(self, owner: SettingsOwner) -> None:
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_changes", {})

    def __getattr__(self, name: str) -> object:
        changes = self._changes
        if name in changes:
            return changes[name]
        return getattr(self._owner.require_canonical().state.managed_connection, name)

    def __setattr__(self, name: str, value: object) -> None:
        current = self._owner.require_canonical().state.managed_connection
        if not hasattr(current, name):
            raise AttributeError(name)
        self._changes[name] = value

    def persist(self) -> None:
        current = self._owner.require_canonical()
        managed = current.state.managed_connection
        updated = replace(
            current,
            state=replace(current.state, managed_connection=replace(managed, **self._changes)),
        )
        self._owner.persist_managed_identity(updated, bound_managed_snapshot=managed)
        self._changes.clear()

    def snapshot(self) -> ManagedIdentitySnapshot:
        from dataclasses import fields

        return ManagedIdentitySnapshot(
            **{item.name: getattr(self, item.name) for item in fields(ManagedIdentitySnapshot)}
        )

    def restore(self, snapshot: ManagedIdentitySnapshot) -> None:
        for name in tuple(self._changes):
            value = getattr(snapshot, name)
            current = getattr(self._owner.require_canonical().state.managed_connection, name)
            if value == current:
                self._changes.pop(name, None)
            else:
                self._changes[name] = value


def managed_identity_state_port_from_vnext(
    owner: SettingsOwner,
    settings: AppSettingsVNext,
) -> ManagedIdentityStatePort:
    return cast(ManagedIdentityStatePort, CanonicalManagedIdentityStateAdapter(owner))


@dataclass(slots=True)
class ManagedIdentityPreflightAdapter:
    managed_state: ManagedIdentityStatePort
    secrets: SecretStore
    _bundle: ManagedIdentityBundle | None = None

    async def preflight_managed_identity(
        self,
        request: ManagedIdentityPreflightRequest,
    ) -> ManagedIdentityPreflightResult:
        _ = request
        try:
            bundle = await self.ensure_bundle()
        except Exception:
            return ManagedIdentityPreflightResult(
                succeeded=False,
                local_public_key=None,
                local_identity_revision=None,
                message=_message("discord_auth.error.retry"),
                diagnostics=_diagnostics(
                    component="managed_identity_preflight",
                    operation="preflight_managed_identity",
                    code="managed_identity_preflight_failed",
                    category=DIAGNOSTIC_CATEGORY_AUTH,
                ),
            )
        return ManagedIdentityPreflightResult(
            succeeded=True,
            local_public_key=bundle.device_public_key,
            local_identity_revision=bundle.installation_id,
            message=None,
            diagnostics=None,
        )

    async def ensure_bundle(self) -> ManagedIdentityBundle:
        if self._bundle is None:
            self._bundle = await run_owned_thread_call(
                partial(ensure_managed_identity_bundle, self.managed_state, self.secrets)
            )
        return self._bundle


@dataclass(slots=True)
class DiscordOAuthAuthAdapter:
    identity: ManagedIdentityPreflightAdapter
    client: object
    app_version: str
    raw_hardware_fingerprint_provider: HardwareFingerprintProvider | None
    hardware_hash_provider: HardwareFingerprintProvider | None
    oauth_runtime: OAuthRuntime
    listener_factory: DiscordOAuthListenerFactory
    callback_runner: DiscordOAuthCallbackRunner
    on_callback_received: Callable[[], None] | None = None

    async def start_discord_auth(self, request: DiscordAuthRequest) -> DiscordAuthResult:
        _ = request
        listener: DiscordOAuthLoopbackListener | None = None
        try:
            bundle = await self.identity.ensure_bundle()
            try:
                listener = self.listener_factory()
            except OSError as exc:
                return _discord_auth_failure(
                    "discord_loopback_unavailable",
                    message_key="discord_auth.error.loopback_unavailable",
                    operation="discord_start",
                    stage="listener_start",
                    error=exc,
                )
            self.oauth_runtime.attach_loopback_listener(listener, listener_name="discord-loopback")
            start_response = await self._start_discord_oauth(bundle, listener)
            if start_response.redirect_uri != listener.redirect_uri:
                return _discord_auth_failure("discord_redirect_mismatch")
            register_discord_oauth_expected_state(listener, start_response.authorization_url)
            try:
                code, state = await self.callback_runner(
                    listener,
                    start_response.authorization_url,
                    start_response.oauth_session_expires_at,
                )
            except TimeoutError as exc:
                return _discord_auth_failure(
                    "discord_oauth_timeout",
                    operation="discord_callback",
                    stage="callback_wait",
                    error=exc,
                )
            except DiscordOAuthBrowserLaunchError as exc:
                return _discord_auth_failure(
                    "discord_browser_launch_failed",
                    message_key="discord_auth.error.browser_launch_failed",
                    operation="discord_callback",
                    stage="browser_launch",
                    error=exc,
                )
            except (DiscordOAuthCallbackError, DiscordOAuthLoopbackClosedError) as exc:
                return _discord_auth_failure(
                    (
                        "discord_oauth_callback_error"
                        if isinstance(exc, DiscordOAuthCallbackError)
                        else "discord_oauth_callback_closed"
                    ),
                    operation="discord_callback",
                    stage="callback_wait",
                    error=exc,
                )
            if self.on_callback_received is not None:
                with contextlib.suppress(Exception):
                    self.on_callback_received()
            hardware_hash = await self._hardware_hash(start_response)
            return DiscordAuthResult(
                succeeded=True,
                discord_user_id=None,
                message=None,
                diagnostics=None,
                authorization_code=code,
                oauth_state=state,
                redirect_uri=listener.redirect_uri,
                issue_nonce=start_response.issue_nonce,
                hardware_hash=hardware_hash,
                hardware_hash_salt_version=start_response.fingerprint_salt_version,
            )
        except ManagedOpenRouterReleaseError as exc:
            return _discord_auth_failure_from_release_error(exc)
        except DiscordOAuthStateError as exc:
            return _discord_auth_failure(
                "discord_oauth_state_invalid",
                operation="discord_start",
                stage="discord_start",
                error=exc,
            )
        except Exception as exc:
            return _discord_auth_failure("discord_auth_exception", error=exc)
        finally:
            if listener is not None:
                await self.oauth_runtime.close_loopback_listener(
                    listener,
                    listener_name="discord-loopback",
                )

    async def _start_discord_oauth(
        self,
        bundle: ManagedIdentityBundle,
        listener: DiscordOAuthLoopbackListener,
    ) -> ManagedOpenRouterDiscordStartSuccess:
        start = getattr(self.client, "start_discord_oauth")
        return await start(
            installation_id=bundle.installation_id,
            device_public_key=bundle.device_public_key,
            redirect_uri=listener.redirect_uri,
            app_version=self.app_version,
        )

    async def _hardware_hash(self, start_response: ManagedOpenRouterDiscordStartSuccess) -> str:
        if self.raw_hardware_fingerprint_provider is not None:
            raw = await _resolve_provider_without_blocking_event_loop(
                self.raw_hardware_fingerprint_provider
            )
            return compute_hardware_hash(
                fingerprint_salt=start_response.fingerprint_salt.salt,
                raw_fingerprint=raw,
            )
        if self.hardware_hash_provider is not None:
            hardware_hash = await _resolve_provider_without_blocking_event_loop(
                self.hardware_hash_provider
            )
            normalized_hardware_hash = _normalize_optional_text(hardware_hash)
            if normalized_hardware_hash is None:
                raise RuntimeError("hardware hash provider returned an invalid value")
            return normalized_hardware_hash
        raise RuntimeError("managed hardware fingerprint provider is not configured")


@dataclass(slots=True)
class DiscordManagedBrokerClientAdapter:
    identity: ManagedIdentityPreflightAdapter
    client: object
    openrouter_config: OpenRouterReleaseRuntimeConfig
    app_version: str
    signed_at_provider: Callable[[], str]
    last_issue_response: ManagedOpenRouterIssueSuccess | None = None

    async def issue_managed_connection(self, request: BrokerIssueRequest) -> BrokerIssueResult:
        missing = _missing_discord_issue_request_fields(request)
        if missing:
            return _broker_issue_failure("discord_issue_material_missing")
        try:
            bundle = await self.identity.ensure_bundle()
            issue_request = bundle.sign_discord_issue_request(
                code=request.authorization_code or "",
                state=request.oauth_state or "",
                redirect_uri=request.redirect_uri or "",
                hardware_hash=request.hardware_hash or "",
                hardware_hash_salt_version=request.hardware_hash_salt_version or 0,
                app_version=self.app_version,
                reason="llm_start",
                budget_usd=MANAGED_OPENROUTER_TRIAL_BUDGET_USD,
                model=_resolve_managed_issue_model(self.openrouter_config),
                issue_nonce=request.issue_nonce or "",
                signed_at=self.signed_at_provider(),
            )
            issue_request["delivery_ack_supported"] = True
            if request.operation_id:
                issue_request["operation_id"] = request.operation_id
            if request.resume_token:
                issue_request["resume_token"] = request.resume_token
            issue = await getattr(self.client, "issue_discord_managed_key")(issue_request)
        except ManagedOpenRouterReleaseError as exc:
            return _broker_issue_failure_from_release_error(exc)
        except Exception:
            return _broker_issue_failure("discord_issue_exception", unknown_outcome=True)
        self.last_issue_response = issue
        return BrokerIssueResult(
            succeeded=True,
            broker_connection_id=issue.managed_credential_ref,
            managed_secret_key=issue.openrouter_api_key,
            remote_key_revision=issue.managed_credential_ref,
            message=None,
            diagnostics=None,
            managed_credential_ref=issue.managed_credential_ref,
            expires_at=issue.expires_at,
            openrouter_user_id=issue.openrouter_user_id,
            delivery_ack=_delivery_ack_metadata_from_issue(issue, source="discord"),
        )

    async def assert_qq_managed_identity(self, request: object) -> object:
        assert_qq = getattr(self.client, "assert_qq_managed_identity")
        return await assert_qq(request)

    async def acknowledge_managed_key_delivery(
        self,
        request: ManagedKeyDeliveryAckRequest,
    ) -> object:
        ack = getattr(self.client, "acknowledge_managed_key_delivery")
        return await ack(request)

    async def get_managed_operation_status(
        self, request: ManagedOperationStatusRequest
    ) -> ManagedOperationStatusResult:
        status = getattr(self.client, "get_managed_operation_status", None)
        if not callable(status):
            raise ManagedOpenRouterReleaseError(
                code="managed_operation_status_unsupported",
                error_class="retryable",
                message="Broker client does not support managed operation status",
                operation="managed_operation_status",
            )
        return await status(request)

    async def resume_managed_operation(
        self, request: ManagedOperationResumeRequest
    ) -> ManagedOperationStatusResult:
        resume = getattr(self.client, "resume_managed_operation", None)
        if not callable(resume):
            raise ManagedOpenRouterReleaseError(
                code="managed_operation_resume_unsupported",
                error_class="retryable",
                message="Broker client does not support managed operation resume",
                operation="managed_operation_resume",
            )
        return await resume(request)


def _delivery_ack_metadata_from_issue(
    issue: ManagedOpenRouterIssueSuccess,
    *,
    source: str,
) -> ManagedKeyDeliveryAckMetadata | None:
    if not issue.delivery_ack_required:
        return None
    if not (issue.delivery_id and issue.managed_credential_ref and issue.delivery_ack_token):
        return None
    return ManagedKeyDeliveryAckMetadata(
        source=source,
        delivery_id=issue.delivery_id,
        managed_credential_ref=issue.managed_credential_ref,
        expires_at=issue.delivery_ack_expires_at,
        delivery_ack_token=issue.delivery_ack_token,
    )


async def _resolve_provider_without_blocking_event_loop(
    provider: HardwareFingerprintProvider,
) -> str:
    if inspect.iscoroutinefunction(provider):
        return await _resolve_maybe_awaitable(provider())
    return await _resolve_maybe_awaitable(await asyncio.to_thread(provider))


async def _resolve_maybe_awaitable(value: str | Awaitable[str]) -> str:
    if inspect.isawaitable(value):
        resolved = await value
    else:
        resolved = value
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError("hardware fingerprint provider returned an invalid value")
    return resolved


def _missing_discord_issue_request_fields(request: BrokerIssueRequest) -> bool:
    return not bool(
        request.authorization_code
        and request.oauth_state
        and request.redirect_uri
        and request.issue_nonce
        and request.hardware_hash
        and request.hardware_hash_salt_version is not None
    )


def _discord_auth_failure_from_release_error(
    error: ManagedOpenRouterReleaseError,
) -> DiscordAuthResult:
    return DiscordAuthResult(
        succeeded=False,
        discord_user_id=None,
        message=_message(_discord_message_key_for_release_error(error)),
        diagnostics=_diagnostics(
            component="discord_managed_auth",
            operation=error.operation or "discord_auth",
            code=error.code,
            category=DIAGNOSTIC_CATEGORY_AUTH,
            subcode=error.subcode,
            retry_after_ms=error.retry_after_ms,
            cause=error.failure_diagnostics,
        ),
    )


def _discord_auth_failure(
    code: str,
    *,
    message_key: str = "discord_auth.error.retry",
    operation: str = "discord_auth",
    stage: str | None = None,
    error: Exception | None = None,
) -> DiscordAuthResult:
    diagnostics = _diagnostics(
        component="managed_auth",
        operation=operation,
        code=code,
        category=DIAGNOSTIC_CATEGORY_AUTH,
    )
    diagnostics = sanitize_diagnostics(
        replace(
            diagnostics,
            fields={
                **diagnostics.fields,
                **(safe_exception_fields(error) if error is not None else {}),
                **({"stage": stage} if stage is not None else {}),
            },
        )
    )
    return DiscordAuthResult(
        succeeded=False,
        discord_user_id=None,
        message=_message(message_key),
        diagnostics=diagnostics,
    )


def _broker_issue_failure_from_release_error(
    error: ManagedOpenRouterReleaseError,
) -> BrokerIssueResult:
    return BrokerIssueResult(
        succeeded=False,
        broker_connection_id=None,
        managed_secret_key=None,
        remote_key_revision=None,
        message=_message(_discord_message_key_for_release_error(error)),
        diagnostics=_diagnostics(
            component="managed_openrouter_broker_client",
            operation=error.operation or "discord_issue",
            code=error.code,
            category=DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
            subcode=error.subcode,
            retry_after_ms=error.retry_after_ms,
            cause=error.failure_diagnostics,
        ),
        unknown_outcome=error.error_class == "retryable"
        and error.subcode not in {"oauth_session_expired", "discord_oauth_session_expired"}
        and not (
            error.failure_diagnostics is not None
            and error.failure_diagnostics.fields.get("dispatch_certainty") == "not_dispatched"
        ),
    )


def _broker_issue_failure(code: str, *, unknown_outcome: bool = False) -> BrokerIssueResult:
    return BrokerIssueResult(
        succeeded=False,
        broker_connection_id=None,
        managed_secret_key=None,
        remote_key_revision=None,
        message=_message("discord_auth.error.retry"),
        diagnostics=_diagnostics(
            component="managed_openrouter_broker_client",
            operation="discord_issue",
            code=code,
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
        ),
        unknown_outcome=unknown_outcome,
    )


def _discord_message_key_for_release_error(error: ManagedOpenRouterReleaseError) -> str:
    if error.code == "discord_browser_launch_failed":
        return "discord_auth.error.browser_launch_failed"
    if error.failure_diagnostics is not None and (
        error.failure_diagnostics.fields.get("configuration_error")
        in {
            "unsupported_proxy_configuration",
            "invalid_proxy_configuration",
            "invalid_tls_configuration",
        }
    ):
        return "discord_auth.error.network_configuration"
    if error.subcode == "discord_email_unverified":
        return "discord_auth.error.email_unverified"
    if error.subcode == "discord_account_too_new":
        return "discord_auth.error.account_too_new"
    if error.subcode == "discord_lifetime_used":
        return "discord_auth.error.lifetime_used"
    if error.subcode == "hardware_duplicate":
        return "discord_auth.error.hardware_duplicate"
    if error.subcode == "global_cap_reached":
        return "discord_auth.error.daily_cap"
    if error.subcode in {"oauth_session_expired", "discord_oauth_session_expired"}:
        return "discord_auth.error.expired"
    if error.code == "discord_loopback_unavailable":
        return "discord_auth.error.loopback_unavailable"
    return "discord_auth.error.retry"


def _message(key: str) -> UserMessageRef:
    return UserMessageRef(key=key, params={}, severity=SEVERITY_ERROR)


def _diagnostics(
    *,
    component: str,
    operation: str,
    code: str,
    category: DiagnosticCategory,
    subcode: str | None = None,
    retry_after_ms: int | None = None,
    cause: ErrorDiagnostics | None = None,
) -> ErrorDiagnostics:
    fields: dict[str, str | int | float | bool | None] = {"phase": operation}
    if subcode is not None:
        fields["subcode"] = subcode
    if cause is not None:
        safe = sanitize_diagnostics(cause)
        return replace(
            safe,
            fields={**safe.fields, **fields},
            retry_after_ms=retry_after_ms if retry_after_ms is not None else safe.retry_after_ms,
        )
    return ErrorDiagnostics(
        component=component,
        operation=operation,
        code=code,
        category=category,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=retry_after_ms,
        fields=fields,
    )


def _resolve_managed_issue_model(config: OpenRouterReleaseRuntimeConfig) -> str:
    selection_alias = config.selection_alias
    if selection_alias is None:
        selection_alias = openrouter_alias_for_fields(
            model=config.llm_model.value,
            source=config.selected_source.value,
        )
    profile = get_openrouter_llm_profile(
        selection_alias.value if hasattr(selection_alias, "value") else selection_alias
    )
    if profile is not None and profile.openrouter_model is not None:
        return profile.openrouter_model
    return config.llm_model.value


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def build_openrouter_credential_runtime_config_from_vnext(
    settings: AppSettingsVNext,
) -> OpenRouterCredentialRuntimeConfig:
    translation = settings.intent.translation
    identity = settings.state.managed_connection
    return OpenRouterCredentialRuntimeConfig(
        selected_source=OpenRouterCredentialSource(translation.openrouter_selected_source),
        installation_id=identity.installation_id or None,
        managed_credential_kind=_managed_credential_kind_from_connection(translation.connection),
        active_managed_credential_ref=identity.active_managed_credential_ref,
        active_managed_expires_at=identity.active_managed_expires_at,
        pending_managed_operation_id=identity.pending_managed_operation_id,
        pending_managed_operation_source=identity.pending_managed_operation_source,
        pending_delivery_ack_source=identity.pending_delivery_ack_source,
    )


def build_openrouter_credential_runtime_config(
    settings: AppSettingsVNext,
) -> OpenRouterCredentialRuntimeConfig:
    return build_openrouter_credential_runtime_config_from_vnext(settings)


def build_openrouter_release_runtime_config_from_vnext(
    settings: AppSettingsVNext,
) -> OpenRouterReleaseRuntimeConfig:
    translation = settings.intent.translation
    alias = translation.openrouter_selection_alias
    return OpenRouterReleaseRuntimeConfig(
        llm_model=parse_openrouter_llm_model(translation.openrouter_model),
        selected_source=OpenRouterCredentialSource(translation.openrouter_selected_source),
        selection_alias=None if alias is None else OpenRouterSelectionAlias(alias),
        managed_credential_kind=_managed_credential_kind_from_connection(translation.connection),
    )


def build_openrouter_release_runtime_config(
    settings: AppSettingsVNext,
) -> OpenRouterReleaseRuntimeConfig:
    return build_openrouter_release_runtime_config_from_vnext(settings)


def _managed_credential_kind_from_connection(connection: str) -> str:
    if connection == TranslationConnection.MANAGED_CHINA.value:
        return "qq"
    return "standard"


def managed_release_service_for_openrouter_config(
    managed_release_service: object | None,
    *,
    openrouter_config: OpenRouterReleaseRuntimeConfig,
    managed_state: object | None = None,
) -> object | None:
    if managed_release_service is None:
        return None

    from puripuly_heart.core.openrouter.managed_openrouter_release import (
        ManagedOpenRouterReleaseService,
    )

    if not isinstance(managed_release_service, ManagedOpenRouterReleaseService):
        return managed_release_service

    if managed_release_service.openrouter_config == openrouter_config:
        return managed_release_service

    return ManagedOpenRouterReleaseService(
        openrouter_config=openrouter_config,
        managed_state=managed_state or managed_release_service.managed_state,
        secrets=managed_release_service.secrets,
        client=managed_release_service.client,
        app_version=managed_release_service.app_version,
        raw_hardware_fingerprint_provider=managed_release_service.raw_hardware_fingerprint_provider,
        hardware_hash_provider=managed_release_service._legacy_hardware_hash_provider,
        signed_at_provider=managed_release_service.signed_at_provider,
        monotonic_ms_provider=managed_release_service.monotonic_ms_provider,
    )


DISCORD_AUTH_ERROR_KEY_BY_SUBCODE = {
    "discord_email_unverified": "discord_auth.error.email_unverified",
    "discord_account_too_new": "discord_auth.error.account_too_new",
    "discord_lifetime_used": "discord_auth.error.lifetime_used",
    "hardware_duplicate": "discord_auth.error.hardware_duplicate",
    "global_cap_reached": "discord_auth.error.daily_cap",
    "oauth_session_expired": "discord_auth.error.expired",
    "discord_oauth_session_expired": "discord_auth.error.expired",
    "loopback_unavailable": "discord_auth.error.loopback_unavailable",
}

ManagedAuthSecretStoreFactory = Callable[..., object]
ManagedAuthReleaseServiceProvider = Callable[[], object | None]
ManagedAuthRuntimePresenceProvider = Callable[[], tuple[bool, bool]]
ManagedAuthIngressProvider = Callable[[], bool]
ManagedAuthSettingsRepositoryFactory = Callable[..., SettingsRepositoryPort]


@dataclass(slots=True)
class CurrentManagedSettingsRepository:
    settings: SettingsOwner
    factory: ManagedAuthSettingsRepositoryFactory

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        current = self.settings.require_canonical()
        repository = self.factory(current, current, "managed_connection_auth")
        return await repository.save(request)


@dataclass(slots=True)
class ManagedAuthRuntimeAdapter:
    config_path: Path
    secret_store_factory: ManagedAuthSecretStoreFactory
    settings: SettingsOwner
    release_service_provider: ManagedAuthReleaseServiceProvider
    settings_repository_factory: ManagedAuthSettingsRepositoryFactory
    runtime_presence_provider: ManagedAuthRuntimePresenceProvider
    ingress_provider: ManagedAuthIngressProvider

    async def logout_local_managed(self, provider: str) -> dict[str, object]:
        if provider not in {"qq", "discord"}:
            raise ValueError("unsupported managed account provider")
        current = self.settings.require_canonical()
        state = current.state.managed_connection
        if state.pending_delivery_ack_source or state.pending_managed_operation_id:
            return {"status": "action_required", "action": "resolve_pending_managed_authorization"}
        keys = (
            (OPENROUTER_MANAGED_QQ_API_KEY_SECRET,)
            if provider == "qq"
            else (
                OPENROUTER_MANAGED_API_KEY_SECRET,
                OPENROUTER_MANAGED_USER_ID_SECRET,
                OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
            )
        )
        store = SyncSecretStoreAdapter(
            self.secret_store_factory(current.intent.secrets, config_path=self.config_path)
        )
        snapshots = [await store.snapshot_secret(key) for key in keys]
        if not snapshots[0].existed:
            return {"status": "rejected", "provider": provider, "reason": "not_authorized"}
        cleared = []
        try:
            for snapshot in snapshots:
                await store.clear_secret(snapshot.key)
                cleared.append(snapshot)
            translation = current.intent.translation
            active_account = managed_openrouter_selected_from_vnext(current) and (
                (provider == "qq" and translation.connection == "managed_china")
                or (provider == "discord" and translation.connection == "managed")
            )
            updated_state = replace(
                state,
                active_managed_credential_ref=(
                    None if active_account else state.active_managed_credential_ref
                ),
                active_managed_expires_at=(
                    None if active_account else state.active_managed_expires_at
                ),
                founder_letter_seen_credential_ref=(
                    None if active_account else state.founder_letter_seen_credential_ref
                ),
            )
            updated = replace(
                current, state=replace(current.state, managed_connection=updated_state)
            )
            repository = self.settings.create_canonical_patch_repository(
                base_settings=current, committed_settings=updated, surface="managed_logout"
            )
            commit = await repository.save(
                SettingsCommitRequest(
                    values={"state": {"managed_connection": asdict(updated_state)}},
                    expected_revision=None,
                    reason="managed_logout",
                )
            )
            if not commit.succeeded:
                for snapshot in reversed(cleared):
                    await store.restore_secret(snapshot)
                return {"status": "persistence_failed", "provider": provider}
        except BaseException:
            for snapshot in reversed(cleared):
                await store.restore_secret(snapshot)
            raise
        return {"status": "applied", "provider": provider, "scope": "local_only"}

    def state(self) -> ManagedAuthState:
        current = self.settings.canonical
        canonical = self.settings.projected_canonical()
        runtime_owner_available, runtime_available = self.runtime_presence_provider()
        if canonical is None or current is None:
            return ManagedAuthState(
                settings_available=False,
                managed_selected=False,
                managed_china=False,
                local_key_available=False,
                release_service_available=False,
                runtime_available=runtime_available,
                ingress_frozen=self.ingress_provider(),
            )
        managed_selected = managed_openrouter_selected_from_vnext(canonical)
        return ManagedAuthState(
            settings_available=True,
            managed_selected=managed_selected,
            managed_china=(
                current.intent.translation.connection == TranslationConnection.MANAGED_CHINA.value
            ),
            local_key_available=(
                self._local_key_available(canonical) if managed_selected else False
            ),
            release_service_available=self.release_service_provider() is not None,
            runtime_available=runtime_available if runtime_owner_available else False,
            ingress_frozen=self.ingress_provider(),
        )

    async def execute_qq(
        self,
        qq_identity: str,
        credential: str,
    ) -> ManagedAuthExecutionResult:
        current = self.settings.canonical
        canonical = self.settings.canonical
        release_service = self.release_service_provider()
        broker_client = getattr(release_service, "client", None)
        if current is None or canonical is None or broker_client is None:
            log_qq_auth("preflight", "configuration_unavailable")
            return ManagedAuthExecutionResult(
                succeeded=False,
                message_key="qq_managed_auth.configuration_unavailable",
            )
        try:
            secret_store = self.secret_store_factory(
                canonical.intent.secrets,
                config_path=self.config_path,
            )
        except Exception as exc:
            log_qq_auth("secret_storage", "local_storage_unavailable", fields=exception_fields(exc))
            return ManagedAuthExecutionResult(
                succeeded=False, message_key="qq_managed_auth.local_storage_unavailable"
            )
        secret_store_port = SyncSecretStoreAdapter(secret_store)
        managed_state = managed_identity_state_port_from_vnext(self.settings, current)
        await secret_store_port.clear_secret("openrouter_managed_qq_status_auth")
        await secret_store_port.clear_secret("openrouter_managed_qq_operation_auth")
        if not managed_state.installation_id and (
            managed_state.pending_managed_operation_source in {None, "qq"}
            and managed_state.pending_delivery_ack_source in {None, "qq"}
        ):
            initialized = await ManagedIdentityPreflightAdapter(
                managed_state=managed_state, secrets=secret_store
            ).preflight_managed_identity(
                ManagedIdentityPreflightRequest(
                    local_secret_key=OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
                    correlation_id=None,
                    metadata={},
                )
            )
            if not initialized.succeeded:
                return ManagedAuthExecutionResult(
                    succeeded=False, message_key="qq_managed_auth.local_storage_unavailable"
                )
        result = await QqManagedAuthService(
            broker_client=broker_client,
            secret_store=secret_store_port,
            managed_state=managed_state,
            claim_guard=ManagedAuthClaimGuard(
                managed_state=managed_state,
                secret_store=secret_store_port,
            ),
        ).authenticate(
            QqManagedAuthRequest(
                qq_identity=qq_identity,
                credential=credential,
                asserted_at=github_star_prompt_utc_timestamp(),
                metadata={"flow": "qq_managed_auth_dialog"},
            )
        )
        delivery_ack_pending = result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
        if _settings_mutation_committed(result):
            return ManagedAuthExecutionResult(
                succeeded=True,
                transaction_result=result,
                runtime_rebuild="always",
            )
        message = result.message
        return ManagedAuthExecutionResult(
            succeeded=False,
            transaction_result=result,
            delivery_ack_pending=delivery_ack_pending,
            message_key=(message.key if message is not None else "qq_managed_auth.error.retry"),
            message_kwargs=dict(message.params) if message is not None else {},
        )

    async def execute_discord(
        self,
        on_callback_received: Callable[[], None] | None,
        on_recovery_started: Callable[[], None] | None = None,
        *,
        authorization_url_sink: Callable[[str], None] | None = None,
        open_browser: bool = True,
    ) -> ManagedAuthExecutionResult:
        release_service = self.release_service_provider()
        current = self.settings.canonical
        if release_service is None or current is None:
            return ManagedAuthExecutionResult(succeeded=False)
        if not _supports_transaction_auth(release_service):
            if authorization_url_sink is not None or not open_browser:
                return ManagedAuthExecutionResult(
                    succeeded=False,
                    message_key="discord_auth.error.action_required",
                )
            return await self._execute_legacy_discord(
                release_service,
            )
        return await self._execute_transaction_discord(
            release_service,
            on_callback_received=on_callback_received,
            on_recovery_started=on_recovery_started,
            authorization_url_sink=authorization_url_sink,
            open_browser=open_browser,
        )

    async def _execute_transaction_discord(
        self,
        release_service: object,
        *,
        on_callback_received: Callable[[], None] | None,
        on_recovery_started: Callable[[], None] | None = None,
        authorization_url_sink: Callable[[str], None] | None = None,
        open_browser: bool = True,
    ) -> ManagedAuthExecutionResult:
        current = self.settings.canonical
        canonical = self.settings.canonical
        if current is None or canonical is None:
            return ManagedAuthExecutionResult(succeeded=False)
        updated = current
        secret_store = self.secret_store_factory(
            canonical.intent.secrets,
            config_path=self.config_path,
        )
        secret_store_port = SyncSecretStoreAdapter(secret_store)
        managed_state = managed_identity_state_port_from_vnext(self.settings, updated)
        identity = ManagedIdentityPreflightAdapter(
            managed_state=managed_state,
            secrets=secret_store,
        )
        discord_auth = DiscordOAuthAuthAdapter(
            identity=identity,
            client=release_service.client,
            app_version=release_service.app_version,
            raw_hardware_fingerprint_provider=release_service.raw_hardware_fingerprint_provider,
            hardware_hash_provider=getattr(
                release_service,
                "_legacy_hardware_hash_provider",
                None,
            ),
            oauth_runtime=release_service.oauth_runtime,
            listener_factory=release_service.discord_oauth_listener_factory,
            callback_runner=(
                release_service.discord_oauth_callback_runner
                if authorization_url_sink is None and open_browser
                else partial(
                    run_discord_oauth_callback_flow,
                    open_browser=open_browser,
                    authorization_url_sink=authorization_url_sink,
                )
            ),
            on_callback_received=on_callback_received,
        )
        broker = DiscordManagedBrokerClientAdapter(
            identity=identity,
            client=release_service.client,
            openrouter_config=release_service.openrouter_config,
            app_version=release_service.app_version,
            signed_at_provider=release_service.signed_at_provider,
        )
        ack_results: list[ManagedKeyDeliveryAckResult] = []

        def _notify_managed_operation_progress(phase: str) -> None:
            if phase == "recovering" and on_recovery_started is not None:
                with contextlib.suppress(Exception):
                    on_recovery_started()

        result = await ManagedConnectionAuthService(
            local_identity=identity,
            discord_auth=discord_auth,
            broker_client=broker,
            secret_store=secret_store_port,
            settings_repository=CurrentManagedSettingsRepository(
                settings=self.settings, factory=self.settings_repository_factory
            ),
            claim_guard=ManagedAuthClaimGuard(
                managed_state=managed_state,
                secret_store=secret_store_port,
            ),
            delivery_ack_service=ManagedKeyDeliveryAckService(
                broker_client=broker,
                secret_store=secret_store_port,
                managed_state=managed_state,
            ),
            managed_state=managed_state,
        ).authorize(
            ManagedConnectionAuthRequest(
                local_secret_key=OPENROUTER_MANAGED_API_KEY_SECRET,
                settings_values=_managed_connection_auth_settings_values(
                    self.settings.project(updated, authoritative=True)
                ),
                expected_settings_revision=None,
                reason="managed_connection_auth",
                correlation_id=None,
                broker_metadata={"flow": "managed_connection_auth"},
                progress_sink=_notify_managed_operation_progress,
                ack_result_sink=ack_results.append,
            )
        )
        delivery_ack_pending = result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
        if not _settings_mutation_committed(result):
            message = result.message
            diagnostics = result.diagnostics
            return ManagedAuthExecutionResult(
                succeeded=False,
                transaction_result=result,
                delivery_ack_pending=delivery_ack_pending,
                message_key=(message.key if message is not None else "discord_auth.error.retry"),
                message_kwargs=dict(message.params) if message is not None else {},
                error_class=getattr(diagnostics, "category", None),
            )
        return ManagedAuthExecutionResult(
            succeeded=True,
            transaction_result=result,
            runtime_rebuild="always",
        )

    async def _execute_legacy_discord(
        self,
        release_service: object,
    ) -> ManagedAuthExecutionResult:
        try:
            claim_guard = self.claim_guard()
            claim_result = await claim_guard.preflight(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
        except Exception:
            return ManagedAuthExecutionResult(succeeded=False, log_failure=False)
        if claim_result is not None:
            message = claim_result.message
            return ManagedAuthExecutionResult(
                succeeded=False,
                transaction_result=claim_result,
                message_key=(message.key if message is not None else "discord_auth.error.retry"),
                message_kwargs=dict(message.params) if message is not None else {},
                log_failure=False,
            )
        try:
            result = await release_service.prepare_for_translation()
        except Exception as exc:
            return ManagedAuthExecutionResult(
                succeeded=False,
                log_message=f"[ManagedAuth] Discord auth start failed: {exc}",
                log_failure=False,
            )
        if result.behavior == ManagedOpenRouterReleaseBehavior.READY and result.local_key_available:
            with contextlib.suppress(Exception):
                claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
                claim_guard.managed_state.persist()
            return ManagedAuthExecutionResult(
                succeeded=True,
                runtime_rebuild="if_missing",
            )
        diagnostics = result.diagnostics
        return ManagedAuthExecutionResult(
            succeeded=False,
            message_key=_discord_auth_message_key(result),
            message_kwargs=dict(result.message_kwargs),
            error_class=getattr(diagnostics, "error_class", None),
        )

    def claim_guard(self) -> ManagedAuthClaimGuard:
        current = self.settings.canonical
        canonical = self.settings.canonical
        if current is None or canonical is None:
            raise RuntimeError("Settings are not loaded")
        secret_store = self.secret_store_factory(
            canonical.intent.secrets,
            config_path=self.config_path,
        )
        secret_store_port = SyncSecretStoreAdapter(secret_store)
        managed_state = managed_identity_state_port_from_vnext(self.settings, current)
        return ManagedAuthClaimGuard(
            managed_state=managed_state,
            secret_store=secret_store_port,
        )

    def _local_key_available(self, settings: AppSettingsVNext) -> bool:
        managed = settings.state.managed_connection
        if managed.pending_managed_operation_id or managed.pending_delivery_ack_source:
            return False
        try:
            secrets = self.secret_store_factory(
                settings.intent.secrets,
                config_path=self.config_path,
            )
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config_from_vnext(settings),
                secrets=secrets,
                request_intent="TRANS",
            )
        except Exception:
            return False
        return resolution.api_key is not None


def _supports_transaction_auth(release_service: object) -> bool:
    return all(
        hasattr(release_service, attr)
        for attr in (
            "app_version",
            "client",
            "discord_oauth_callback_runner",
            "discord_oauth_listener_factory",
            "oauth_runtime",
            "openrouter_config",
            "signed_at_provider",
        )
    )


def _discord_auth_message_key(result: object) -> str:
    diagnostics = getattr(result, "diagnostics", None)
    subcode = getattr(diagnostics, "subcode", None)
    if subcode is not None:
        mapped_key = DISCORD_AUTH_ERROR_KEY_BY_SUBCODE.get(subcode)
        if mapped_key is not None:
            return mapped_key
    if getattr(diagnostics, "code", None) == "discord_loopback_unavailable":
        return DISCORD_AUTH_ERROR_KEY_BY_SUBCODE["loopback_unavailable"]
    return getattr(result, "message_key", "discord_auth.error.retry")


def managed_openrouter_selected_from_vnext(settings: AppSettingsVNext) -> bool:
    translation = settings.intent.translation
    return (
        translation.connection
        in {
            TranslationConnection.MANAGED.value,
            TranslationConnection.MANAGED_CHINA.value,
        }
        and translation.openrouter_selected_source == OpenRouterCredentialSource.MANAGED.value
    )


def _managed_connection_auth_settings_values(
    settings: AppSettingsVNext,
) -> dict[str, Any]:
    translation = settings.intent.translation
    managed = settings.state.managed_connection
    return {
        "intent": {
            "translation": {
                "connection": translation.connection,
                "model": translation.model,
            },
            "openrouter": {
                "selected_source": translation.openrouter_selected_source,
                "llm_model": translation.openrouter_model,
                "selection_alias": translation.openrouter_selection_alias,
            },
        },
        "state": {
            "managed_connection": {
                "installation_id": managed.installation_id,
                "active_managed_credential_ref": managed.active_managed_credential_ref,
                "active_managed_expires_at": managed.active_managed_expires_at,
                "founder_letter_seen_credential_ref": (managed.founder_letter_seen_credential_ref),
                "local_managed_claim_sources": list(managed.local_managed_claim_sources),
                "pending_managed_operation_id": managed.pending_managed_operation_id,
                "pending_managed_operation_source": managed.pending_managed_operation_source,
                "pending_managed_operation_installation_id": (
                    managed.pending_managed_operation_installation_id
                ),
                "pending_managed_operation_state": managed.pending_managed_operation_state,
            }
        },
    }


def _llm_provider_name_from_vnext(settings: AppSettingsVNext) -> str:
    from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
        provider_llm_for_translation,
    )

    translation = settings.intent.translation
    return provider_llm_for_translation(translation.model, translation.connection)


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


ManagedTranslationReleaseServiceProvider = Callable[[], object | None]
ManagedTranslationRuntimeSnapshotProvider = Callable[
    [],
    tuple[bool, bool, object | None],
]
ManagedTranslationIngressProvider = Callable[[], bool]
ManagedTranslationFounderDialog = Callable[[], bool]
ManagedTranslationPersistSettings = Callable[[], object]


@dataclass(slots=True)
class ManagedTranslationRuntimeAdapter:
    auth: ManagedAuthRuntimeAdapter
    settings: SettingsOwner
    release_service_provider: ManagedTranslationReleaseServiceProvider
    runtime_snapshot_provider: ManagedTranslationRuntimeSnapshotProvider
    ingress_provider: ManagedTranslationIngressProvider
    founder_dialog: ManagedTranslationFounderDialog
    persist_settings: ManagedTranslationPersistSettings

    def state(self) -> TranslationEnableState:
        canonical = self.settings.projected_canonical()
        runtime_available, translation_enabled, llm = self.runtime_snapshot_provider()
        auth_state = self.auth.state()
        is_custom_http = (
            canonical is not None
            and canonical.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
        )
        qwen_region = None if canonical is None else canonical.intent.translation.qwen.region
        return TranslationEnableState(
            runtime_available=runtime_available,
            translation_enabled=translation_enabled,
            llm_available=llm is not None,
            settings_available=canonical is not None,
            provider_name=(
                "custom_http"
                if is_custom_http
                else (None if canonical is None else _llm_provider_name_from_vnext(canonical))
            ),
            qwen_region=None if is_custom_http else qwen_region,
            managed_selected=auth_state.managed_selected and not is_custom_http,
            managed_china=auth_state.managed_china and not is_custom_http,
            managed_local_key_available=auth_state.local_key_available and not is_custom_http,
            managed_release_service_available=(
                auth_state.release_service_available and not is_custom_http
            ),
            ingress_frozen=self.ingress_provider(),
        )

    async def prepare(self, allow_authorization: bool = True) -> ManagedTranslationPreparation:
        canonical = self.settings.canonical
        current = self.settings.canonical
        service = self.release_service_provider()
        if (
            canonical is None
            or current is None
            or service is None
            or canonical.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
        ):
            return ManagedTranslationPreparation(ready=True)
        claim_guard = None
        if not self.auth.state().managed_china:
            try:
                claim_guard = self.auth.claim_guard()
                claim_result = await claim_guard.preflight(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
            except Exception:
                return ManagedTranslationPreparation(
                    ready=False,
                    message_key="discord_auth.error.retry",
                )
            if claim_result is not None:
                message = claim_result.message
                return ManagedTranslationPreparation(
                    ready=False,
                    transaction_result=claim_result,
                    message_key=(
                        message.key if message is not None else "discord_auth.error.retry"
                    ),
                    message_kwargs=(dict(message.params) if message is not None else {}),
                )
        result = await service.prepare_for_translation(
            allow_authorization=allow_authorization,
        )
        if result.behavior == ManagedOpenRouterReleaseBehavior.READY and result.local_key_available:
            if claim_guard is not None:
                with contextlib.suppress(Exception):
                    claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
                    claim_guard.managed_state.persist()
            return ManagedTranslationPreparation(ready=True)
        return ManagedTranslationPreparation(
            ready=False,
            message_key=result.message_key,
            message_kwargs=dict(result.message_kwargs),
            diagnostics_text=format_managed_openrouter_diagnostics(result.diagnostics),
            show_qq_dialog=(
                result.message_key == "qq_managed_auth.required" and self.auth.state().managed_china
            ),
        )

    def show_founder_letter(self) -> None:
        current = self.settings.canonical
        if current is None or not self.founder_dialog():
            return
        managed_state = managed_identity_state_port_from_vnext(self.settings, current)
        mark_founder_letter_shown(managed_state)
        with contextlib.suppress(Exception):
            managed_state.persist()
            self.persist_settings()

    async def warmup(self) -> None:
        _runtime_available, _translation_enabled, llm = self.runtime_snapshot_provider()
        if isinstance(llm, SemaphoreLLMProvider):
            llm = llm.inner
        warmup = getattr(llm, "warmup", None)
        if not callable(warmup):
            return
        with contextlib.suppress(Exception):
            result = warmup()
            if inspect.isawaitable(result):
                await result
