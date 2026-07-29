from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.broker_client import (
    BrokerIssueRequest,
    BrokerIssueResult,
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckRequest,
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
from puripuly_heart.app.ports.settings_repository import SettingsRepositoryPort
from puripuly_heart.app.services.github_star_prompt import (
    github_star_prompt_utc_timestamp,
)
from puripuly_heart.app.services.managed_auth import (
    ManagedAuthExecutionResult,
    ManagedAuthState,
)
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    ManagedAuthClaimGuard,
)
from puripuly_heart.app.services.managed_connection_auth import (
    ManagedConnectionAuthRequest,
    ManagedConnectionAuthService,
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
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    TranslationConnection,
    normalize_owned_referral_id,
)
from puripuly_heart.core.discord_oauth_loopback import (
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
from puripuly_heart.core.managed_openrouter_release import (
    MANAGED_OPENROUTER_TRIAL_BUDGET_USD,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseError,
    OpenRouterReleaseRuntimeConfig,
    format_managed_openrouter_diagnostics,
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
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OpenRouterCredentialRuntimeConfig,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.openrouter_handoff import mark_founder_letter_shown
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.storage.secrets import SecretStore

HardwareFingerprintProvider = Callable[[], str | Awaitable[str]]
DiscordOAuthListenerFactory = Callable[[], DiscordOAuthLoopbackListener]
DiscordOAuthCallbackRunner = Callable[
    [DiscordOAuthLoopbackListener, str, str],
    Awaitable[tuple[str, str]],
]


@dataclass(slots=True)
class ManagedIdentityStateAdapter:
    """Boundary adapter that exposes ``AppSettings`` managed-identity state as a
    ``ManagedIdentityStatePort``.

    Reads and writes proxy directly to ``settings.managed_identity`` so that
    mutations are visible to subsequent reads before ``persist`` is called.
    ``persist`` delegates to the supplied persistence callable, which receives
    the wrapped ``AppSettings`` instance.
    """

    _settings: AppSettings
    _persist: Callable[[AppSettings], None]

    @property
    def installation_id(self) -> str:
        return self._settings.managed_identity.installation_id

    @installation_id.setter
    def installation_id(self, value: str) -> None:
        self._settings.managed_identity.installation_id = value

    @property
    def release_token(self) -> str | None:
        return self._settings.managed_identity.release_token

    @release_token.setter
    def release_token(self, value: str | None) -> None:
        self._settings.managed_identity.release_token = value

    @property
    def release_token_expires_at(self) -> str | None:
        return self._settings.managed_identity.release_token_expires_at

    @release_token_expires_at.setter
    def release_token_expires_at(self, value: str | None) -> None:
        self._settings.managed_identity.release_token_expires_at = value

    @property
    def verified_hardware_hash(self) -> str | None:
        return self._settings.managed_identity.verified_hardware_hash

    @verified_hardware_hash.setter
    def verified_hardware_hash(self, value: str | None) -> None:
        self._settings.managed_identity.verified_hardware_hash = value

    @property
    def verified_hardware_hash_salt_version(self) -> int | None:
        return self._settings.managed_identity.verified_hardware_hash_salt_version

    @verified_hardware_hash_salt_version.setter
    def verified_hardware_hash_salt_version(self, value: int | None) -> None:
        self._settings.managed_identity.verified_hardware_hash_salt_version = value

    @property
    def active_managed_credential_ref(self) -> str | None:
        return self._settings.managed_identity.active_managed_credential_ref

    @active_managed_credential_ref.setter
    def active_managed_credential_ref(self, value: str | None) -> None:
        self._settings.managed_identity.active_managed_credential_ref = value

    @property
    def active_managed_expires_at(self) -> str | None:
        return self._settings.managed_identity.active_managed_expires_at

    @active_managed_expires_at.setter
    def active_managed_expires_at(self, value: str | None) -> None:
        self._settings.managed_identity.active_managed_expires_at = value

    @property
    def founder_letter_seen_credential_ref(self) -> str | None:
        return self._settings.managed_identity.founder_letter_seen_credential_ref

    @founder_letter_seen_credential_ref.setter
    def founder_letter_seen_credential_ref(self, value: str | None) -> None:
        self._settings.managed_identity.founder_letter_seen_credential_ref = value

    @property
    def referral_id(self) -> str | None:
        return self._settings.managed_identity.referral_id

    @referral_id.setter
    def referral_id(self, value: str | None) -> None:
        self._settings.managed_identity.referral_id = value

    @property
    def local_managed_claim_sources(self) -> tuple[str, ...]:
        return self._settings.managed_identity.local_managed_claim_sources

    @local_managed_claim_sources.setter
    def local_managed_claim_sources(self, value: tuple[str, ...]) -> None:
        self._settings.managed_identity.local_managed_claim_sources = value

    @property
    def pending_delivery_ack_source(self) -> str | None:
        return getattr(self._settings.managed_identity, "pending_delivery_ack_source", None)

    @pending_delivery_ack_source.setter
    def pending_delivery_ack_source(self, value: str | None) -> None:
        self._settings.managed_identity.pending_delivery_ack_source = value

    @property
    def pending_delivery_ack_delivery_id(self) -> str | None:
        return getattr(self._settings.managed_identity, "pending_delivery_ack_delivery_id", None)

    @pending_delivery_ack_delivery_id.setter
    def pending_delivery_ack_delivery_id(self, value: str | None) -> None:
        self._settings.managed_identity.pending_delivery_ack_delivery_id = value

    @property
    def pending_delivery_ack_managed_credential_ref(self) -> str | None:
        return getattr(
            self._settings.managed_identity,
            "pending_delivery_ack_managed_credential_ref",
            None,
        )

    @pending_delivery_ack_managed_credential_ref.setter
    def pending_delivery_ack_managed_credential_ref(self, value: str | None) -> None:
        self._settings.managed_identity.pending_delivery_ack_managed_credential_ref = value

    @property
    def pending_delivery_ack_expires_at(self) -> str | None:
        return getattr(self._settings.managed_identity, "pending_delivery_ack_expires_at", None)

    @pending_delivery_ack_expires_at.setter
    def pending_delivery_ack_expires_at(self, value: str | None) -> None:
        self._settings.managed_identity.pending_delivery_ack_expires_at = value

    def persist(self) -> None:
        self._persist(self._settings)

    def snapshot(self) -> ManagedIdentitySnapshot:
        managed = self._settings.managed_identity
        return ManagedIdentitySnapshot(
            installation_id=managed.installation_id,
            release_token=managed.release_token,
            release_token_expires_at=managed.release_token_expires_at,
            verified_hardware_hash=managed.verified_hardware_hash,
            verified_hardware_hash_salt_version=managed.verified_hardware_hash_salt_version,
            active_managed_credential_ref=managed.active_managed_credential_ref,
            active_managed_expires_at=managed.active_managed_expires_at,
            founder_letter_seen_credential_ref=managed.founder_letter_seen_credential_ref,
            referral_id=managed.referral_id,
            local_managed_claim_sources=managed.local_managed_claim_sources,
            pending_delivery_ack_source=getattr(managed, "pending_delivery_ack_source", None),
            pending_delivery_ack_delivery_id=getattr(
                managed, "pending_delivery_ack_delivery_id", None
            ),
            pending_delivery_ack_managed_credential_ref=getattr(
                managed,
                "pending_delivery_ack_managed_credential_ref",
                None,
            ),
            pending_delivery_ack_expires_at=getattr(
                managed,
                "pending_delivery_ack_expires_at",
                None,
            ),
        )

    def restore(self, snapshot: ManagedIdentitySnapshot) -> None:
        managed = self._settings.managed_identity
        managed.installation_id = snapshot.installation_id
        managed.release_token = snapshot.release_token
        managed.release_token_expires_at = snapshot.release_token_expires_at
        managed.verified_hardware_hash = snapshot.verified_hardware_hash
        managed.verified_hardware_hash_salt_version = snapshot.verified_hardware_hash_salt_version
        managed.active_managed_credential_ref = snapshot.active_managed_credential_ref
        managed.active_managed_expires_at = snapshot.active_managed_expires_at
        managed.founder_letter_seen_credential_ref = snapshot.founder_letter_seen_credential_ref
        managed.referral_id = snapshot.referral_id
        managed.local_managed_claim_sources = snapshot.local_managed_claim_sources
        managed.pending_delivery_ack_source = snapshot.pending_delivery_ack_source
        managed.pending_delivery_ack_delivery_id = snapshot.pending_delivery_ack_delivery_id
        managed.pending_delivery_ack_managed_credential_ref = (
            snapshot.pending_delivery_ack_managed_credential_ref
        )
        managed.pending_delivery_ack_expires_at = snapshot.pending_delivery_ack_expires_at


def build_managed_identity_state_port(
    settings: AppSettings,
    persist: Callable[[AppSettings], None],
) -> ManagedIdentityStatePort:
    """Build a ``ManagedIdentityStatePort`` adapter at the wiring boundary."""

    return ManagedIdentityStateAdapter(settings, persist)


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
            self._bundle = await asyncio.to_thread(
                ensure_managed_identity_bundle,
                self.managed_state,
                self.secrets,
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
    referral_id: str | None = None
    on_callback_received: Callable[[], None] | None = None

    async def start_discord_auth(self, request: DiscordAuthRequest) -> DiscordAuthResult:
        _ = request
        listener: DiscordOAuthLoopbackListener | None = None
        try:
            bundle = await self.identity.ensure_bundle()
            listener = self.listener_factory()
            self.oauth_runtime.attach_loopback_listener(listener, listener_name="discord-loopback")
            start_response = await self._start_discord_oauth(bundle, listener)
            if start_response.redirect_uri != listener.redirect_uri:
                return _discord_auth_failure("discord_redirect_mismatch")
            code, state = await self.callback_runner(
                listener,
                start_response.authorization_url,
                start_response.oauth_session_expires_at,
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
        except (
            DiscordOAuthCallbackError,
            DiscordOAuthLoopbackClosedError,
            TimeoutError,
        ):
            return _discord_auth_failure("discord_callback_failed")
        except Exception:
            return _discord_auth_failure("discord_auth_exception")
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
            referral_id=self.referral_id,
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
            issue = await getattr(self.client, "issue_discord_managed_key")(issue_request)
        except ManagedOpenRouterReleaseError as exc:
            return _broker_issue_failure_from_release_error(exc)
        except Exception:
            return _broker_issue_failure("discord_issue_exception")
        self.last_issue_response = issue
        apply_discord_issue_result_to_managed_state(self.identity.managed_state, issue)
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
            referral_id=issue.referral_id,
            referral_bonus_applied=issue.referral_bonus_applied,
            pass_status=issue.pass_status,
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


def apply_discord_issue_result_to_managed_state(
    managed_state: ManagedIdentityStatePort,
    issue: ManagedOpenRouterIssueSuccess,
) -> None:
    current_ref = managed_state.active_managed_credential_ref
    next_ref = (
        _normalize_optional_text(issue.managed_credential_ref)
        or current_ref
        or _normalize_optional_text(issue.expires_at)
        or managed_state.installation_id
        or "managed-entitlement"
    )
    if current_ref != next_ref:
        managed_state.founder_letter_seen_credential_ref = None
    managed_state.active_managed_credential_ref = next_ref
    managed_state.active_managed_expires_at = _normalize_optional_text(issue.expires_at)
    referral_id = _normalize_owned_referral_id(issue.referral_id)
    if referral_id is not None:
        managed_state.referral_id = referral_id
    managed_state.release_token = None
    managed_state.release_token_expires_at = None
    managed_state.verified_hardware_hash = None
    managed_state.verified_hardware_hash_salt_version = None


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
        ),
    )


def _discord_auth_failure(code: str) -> DiscordAuthResult:
    return DiscordAuthResult(
        succeeded=False,
        discord_user_id=None,
        message=_message("discord_auth.error.retry"),
        diagnostics=_diagnostics(
            component="discord_managed_auth",
            operation="discord_auth",
            code=code,
            category=DIAGNOSTIC_CATEGORY_AUTH,
        ),
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
        ),
    )


def _broker_issue_failure(code: str) -> BrokerIssueResult:
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
    )


def _discord_message_key_for_release_error(error: ManagedOpenRouterReleaseError) -> str:
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
    if error.subcode == "oauth_session_expired":
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
) -> ErrorDiagnostics:
    fields: dict[str, str | int | float | bool | None] = {"phase": operation}
    if subcode is not None:
        fields["subcode"] = subcode
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


def _normalize_owned_referral_id(value: object) -> str | None:
    from puripuly_heart.config.settings import normalize_owned_referral_id

    return normalize_owned_referral_id(value)


def build_openrouter_credential_runtime_config(
    settings: AppSettings,
) -> OpenRouterCredentialRuntimeConfig:
    """Build a narrow OpenRouter credential runtime DTO from legacy settings."""

    return OpenRouterCredentialRuntimeConfig(
        selected_source=settings.openrouter.selected_source,
        installation_id=settings.managed_identity.installation_id,
        managed_credential_kind=_managed_credential_kind_for_settings(settings),
        active_managed_credential_ref=settings.managed_identity.active_managed_credential_ref,
        active_managed_expires_at=settings.managed_identity.active_managed_expires_at,
    )


def build_openrouter_release_runtime_config(
    settings: AppSettings,
) -> OpenRouterReleaseRuntimeConfig:
    """Build a narrow OpenRouter release runtime DTO from legacy settings."""

    return OpenRouterReleaseRuntimeConfig(
        llm_model=settings.openrouter.llm_model,
        selected_source=settings.openrouter.selected_source,
        selection_alias=settings.openrouter.selection_alias,
        managed_credential_kind=_managed_credential_kind_for_settings(settings),
    )


def _managed_credential_kind_for_settings(settings: AppSettings) -> str:
    if settings.translation.connection == TranslationConnection.MANAGED_CHINA:
        return "qq"
    return "standard"


def _managed_release_service_for_alias(
    managed_release_service: object | None,
    *,
    alias_settings: AppSettings,
) -> object | None:
    if managed_release_service is None:
        return None

    from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterReleaseService

    if not isinstance(managed_release_service, ManagedOpenRouterReleaseService):
        return managed_release_service

    desired_config = build_openrouter_release_runtime_config(alias_settings)
    if managed_release_service.openrouter_config == desired_config:
        return managed_release_service

    return ManagedOpenRouterReleaseService(
        openrouter_config=desired_config,
        managed_state=ManagedIdentityStateAdapter(
            alias_settings,
            lambda _settings: managed_release_service.managed_state.persist(),
        ),
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
    "loopback_unavailable": "discord_auth.error.loopback_unavailable",
}

ManagedAuthSettingsProvider = Callable[[], AppSettings | None]
ManagedAuthSettingsSink = Callable[[AppSettings], None]
ManagedAuthSecretStoreFactory = Callable[..., object]
ManagedAuthReleaseServiceProvider = Callable[[], object | None]
ManagedAuthPersistenceCallbackFactory = Callable[
    [AppSettings],
    Callable[[AppSettings], None],
]
ManagedAuthSettingsRepositoryFactory = Callable[
    [AppSettings, AppSettings, str],
    SettingsRepositoryPort,
]
ManagedAuthSettingsOwnerComplete = Callable[[], None]
ManagedAuthRuntimePresenceProvider = Callable[[], tuple[bool, bool]]
ManagedAuthIngressProvider = Callable[[], bool]


@dataclass(slots=True)
class ManagedAuthRuntimeAdapter:
    config_path: Path
    secret_store_factory: ManagedAuthSecretStoreFactory
    settings_provider: ManagedAuthSettingsProvider
    settings_sink: ManagedAuthSettingsSink
    release_service_provider: ManagedAuthReleaseServiceProvider
    persistence_callback_factory: ManagedAuthPersistenceCallbackFactory
    settings_repository_factory: ManagedAuthSettingsRepositoryFactory
    settings_owner_complete: ManagedAuthSettingsOwnerComplete
    runtime_presence_provider: ManagedAuthRuntimePresenceProvider
    ingress_provider: ManagedAuthIngressProvider

    def state(self) -> ManagedAuthState:
        settings = self.settings_provider()
        runtime_owner_available, runtime_available = self.runtime_presence_provider()
        if settings is None:
            return ManagedAuthState(
                settings_available=False,
                managed_selected=False,
                managed_china=False,
                local_key_available=False,
                release_service_available=False,
                runtime_available=runtime_available,
                ingress_frozen=self.ingress_provider(),
            )
        managed_selected = _managed_openrouter_selected(settings)
        return ManagedAuthState(
            settings_available=True,
            managed_selected=managed_selected,
            managed_china=(settings.translation.connection == TranslationConnection.MANAGED_CHINA),
            local_key_available=(
                self._local_key_available(settings) if managed_selected else False
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
        settings = self.settings_provider()
        release_service = self.release_service_provider()
        broker_client = getattr(release_service, "client", None)
        if settings is None or broker_client is None:
            return ManagedAuthExecutionResult(
                succeeded=False,
                message_key="qq_managed_auth.error.retry",
            )
        secret_store = self.secret_store_factory(
            settings.secrets,
            config_path=self.config_path,
        )
        secret_store_port = SyncSecretStoreAdapter(secret_store)
        managed_state = build_managed_identity_state_port(
            settings,
            self.persistence_callback_factory(settings),
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
        if _settings_mutation_committed(result):
            runtime_owner_available, runtime_available = self.runtime_presence_provider()
            return ManagedAuthExecutionResult(
                succeeded=True,
                transaction_result=result,
                runtime_rebuild=(
                    "if_missing" if runtime_owner_available and not runtime_available else "never"
                ),
            )
        message = result.message
        return ManagedAuthExecutionResult(
            succeeded=False,
            transaction_result=result,
            message_key=(message.key if message is not None else "qq_managed_auth.error.retry"),
            message_kwargs=dict(message.params) if message is not None else {},
        )

    async def execute_discord(
        self,
        referral_id: str | None,
        on_callback_received: Callable[[], None] | None,
    ) -> ManagedAuthExecutionResult:
        release_service = self.release_service_provider()
        settings = self.settings_provider()
        if release_service is None or settings is None:
            return ManagedAuthExecutionResult(succeeded=False)
        if not _supports_transaction_auth(release_service):
            return await self._execute_legacy_discord(
                release_service,
                settings,
                referral_id=referral_id,
            )
        return await self._execute_transaction_discord(
            release_service,
            settings,
            referral_id=referral_id,
            on_callback_received=on_callback_received,
        )

    async def _execute_transaction_discord(
        self,
        release_service: object,
        settings: AppSettings,
        *,
        referral_id: str | None,
        on_callback_received: Callable[[], None] | None,
    ) -> ManagedAuthExecutionResult:
        updated = copy.deepcopy(settings)
        secret_store = self.secret_store_factory(
            updated.secrets,
            config_path=self.config_path,
        )
        secret_store_port = SyncSecretStoreAdapter(secret_store)
        managed_state = build_managed_identity_state_port(
            updated,
            self.persistence_callback_factory(updated),
        )
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
            callback_runner=release_service.discord_oauth_callback_runner,
            referral_id=referral_id,
            on_callback_received=on_callback_received,
        )
        broker = DiscordManagedBrokerClientAdapter(
            identity=identity,
            client=release_service.client,
            openrouter_config=release_service.openrouter_config,
            app_version=release_service.app_version,
            signed_at_provider=release_service.signed_at_provider,
        )
        result = await ManagedConnectionAuthService(
            local_identity=identity,
            discord_auth=discord_auth,
            broker_client=broker,
            secret_store=secret_store_port,
            settings_repository=self.settings_repository_factory(
                settings,
                updated,
                "managed_connection_auth",
            ),
            claim_guard=ManagedAuthClaimGuard(
                managed_state=managed_state,
                secret_store=secret_store_port,
            ),
        ).authorize(
            ManagedConnectionAuthRequest(
                local_secret_key=OPENROUTER_MANAGED_API_KEY_SECRET,
                settings_values=_managed_connection_auth_settings_values(updated),
                expected_settings_revision=None,
                reason="managed_connection_auth",
                correlation_id=None,
                broker_metadata={"flow": "managed_connection_auth"},
            )
        )
        self.settings_owner_complete()
        if result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING:
            self.settings_sink(updated)
        if not _settings_mutation_committed(result):
            message = result.message
            diagnostics = result.diagnostics
            return ManagedAuthExecutionResult(
                succeeded=False,
                transaction_result=result,
                delivery_ack_pending=(
                    result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
                ),
                message_key=(message.key if message is not None else "discord_auth.error.retry"),
                message_kwargs=dict(message.params) if message is not None else {},
                error_class=getattr(diagnostics, "category", None),
            )
        issue = broker.last_issue_response
        self.settings_sink(updated)
        return ManagedAuthExecutionResult(
            succeeded=True,
            transaction_result=result,
            referral_bonus_applied=bool(getattr(issue, "referral_bonus_applied", False)),
            referral_id=normalize_owned_referral_id(getattr(issue, "referral_id", None)),
            pass_status=getattr(issue, "pass_status", None),
            runtime_rebuild="always",
        )

    async def _execute_legacy_discord(
        self,
        release_service: object,
        settings: AppSettings,
        *,
        referral_id: str | None,
    ) -> ManagedAuthExecutionResult:
        try:
            claim_guard = self.claim_guard(settings)
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
            result = await release_service.prepare_for_translation(referral_id=referral_id)
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
                referral_bonus_applied=(getattr(result, "referral_bonus_applied", False) is True),
                referral_id=normalize_owned_referral_id(getattr(result, "referral_id", None)),
                pass_status=getattr(result, "pass_status", None),
                runtime_rebuild="if_missing",
            )
        diagnostics = result.diagnostics
        return ManagedAuthExecutionResult(
            succeeded=False,
            message_key=_discord_auth_message_key(result),
            message_kwargs=dict(result.message_kwargs),
            error_class=getattr(diagnostics, "error_class", None),
        )

    def claim_guard(self, settings: AppSettings) -> ManagedAuthClaimGuard:
        secret_store = self.secret_store_factory(
            settings.secrets,
            config_path=self.config_path,
        )
        secret_store_port = SyncSecretStoreAdapter(secret_store)
        managed_state = build_managed_identity_state_port(
            settings,
            self.persistence_callback_factory(settings),
        )
        return ManagedAuthClaimGuard(
            managed_state=managed_state,
            secret_store=secret_store_port,
        )

    def _local_key_available(self, settings: AppSettings) -> bool:
        try:
            secrets = self.secret_store_factory(
                settings.secrets,
                config_path=self.config_path,
            )
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config(settings),
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


def _managed_openrouter_selected(settings: AppSettings) -> bool:
    return bool(
        settings.provider.llm == LLMProviderName.OPENROUTER
        and settings.translation.connection
        in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
        and settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    )


def _managed_connection_auth_settings_values(
    settings: AppSettings,
) -> dict[str, Any]:
    managed = settings.managed_identity
    selection_alias = settings.openrouter.selection_alias
    return {
        "intent": {
            "translation": {
                "connection": settings.translation.connection.value,
                "model": settings.translation.model.value,
            },
            "openrouter": {
                "selected_source": settings.openrouter.selected_source.value,
                "llm_model": settings.openrouter.llm_model.value,
                "selection_alias": (selection_alias.value if selection_alias is not None else None),
            },
        },
        "state": {
            "managed_connection": {
                "installation_id": managed.installation_id,
                "active_managed_credential_ref": (managed.active_managed_credential_ref),
                "active_managed_expires_at": managed.active_managed_expires_at,
                "founder_letter_seen_credential_ref": (managed.founder_letter_seen_credential_ref),
                "referral_id": managed.referral_id,
                "local_managed_claim_sources": list(managed.local_managed_claim_sources),
            }
        },
    }


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


ManagedTranslationSettingsProvider = Callable[[], AppSettings | None]
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
    settings_provider: ManagedTranslationSettingsProvider
    release_service_provider: ManagedTranslationReleaseServiceProvider
    runtime_snapshot_provider: ManagedTranslationRuntimeSnapshotProvider
    ingress_provider: ManagedTranslationIngressProvider
    founder_dialog: ManagedTranslationFounderDialog
    persist_settings: ManagedTranslationPersistSettings

    def state(self) -> TranslationEnableState:
        settings = self.settings_provider()
        runtime_available, translation_enabled, llm = self.runtime_snapshot_provider()
        auth_state = self.auth.state()
        return TranslationEnableState(
            runtime_available=runtime_available,
            translation_enabled=translation_enabled,
            llm_available=llm is not None,
            settings_available=settings is not None,
            provider_name=(settings.provider.llm.value if settings is not None else None),
            qwen_region=(settings.qwen.region.value if settings is not None else None),
            managed_selected=auth_state.managed_selected,
            managed_china=auth_state.managed_china,
            managed_local_key_available=auth_state.local_key_available,
            managed_release_service_available=(auth_state.release_service_available),
            ingress_frozen=self.ingress_provider(),
        )

    async def prepare(self) -> ManagedTranslationPreparation:
        settings = self.settings_provider()
        service = self.release_service_provider()
        if settings is None or service is None:
            return ManagedTranslationPreparation(ready=True)
        claim_guard = None
        if not self.auth.state().managed_china:
            try:
                claim_guard = self.auth.claim_guard(settings)
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
        result = await service.prepare_for_translation()
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
        settings = self.settings_provider()
        if settings is None or not self.founder_dialog():
            return
        mark_founder_letter_shown(
            build_managed_identity_state_port(
                settings,
                lambda _settings: None,
            )
        )
        with contextlib.suppress(Exception):
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
