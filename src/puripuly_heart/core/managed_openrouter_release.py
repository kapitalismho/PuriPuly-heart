from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol
from uuid import UUID

from puripuly_heart.config.llm_profiles import (
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.settings import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    AppSettings,
    OpenRouterCredentialSource,
    TranslationConnection,
    normalize_owned_referral_id,
)
from puripuly_heart.core.discord_managed_oauth import run_discord_oauth_callback_flow
from puripuly_heart.core.discord_oauth_loopback import (
    DiscordOAuthCallbackError,
    DiscordOAuthLoopbackClosedError,
    DiscordOAuthLoopbackListener,
    bind_first_available,
)
from puripuly_heart.core.hardware_fingerprint import compute_hardware_hash
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.managed_auth_claims import record_local_managed_claim_source
from puripuly_heart.core.managed_identity import (
    ensure_managed_identity_bundle,
    load_existing_managed_identity_bundle,
    regenerate_managed_identity_bundle,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_DELIVERY_ACK_TOKEN_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_DELIVERY_ACK_TOKEN_SECRET,
    best_effort_store_managed_openrouter_user_identifier,
    clear_temporary_managed_release_state,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.openrouter_handoff import store_managed_entitlement_snapshot
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.domain.models import Translation

logger = logging.getLogger(__name__)

MANAGED_OPENROUTER_TRIAL_BUDGET_USD = 0.07
BINDING_MISMATCH_SUBCODES = {
    "device_public_key_registered",
    "installation_binding_mismatch",
}
QQ_ASSERT_CREDENTIAL_SUBCODES = frozenset(
    {
        "qq_credential_invalid",
        "qq_credential_mismatch",
    }
)
QQ_ASSERT_LIFETIME_USED_SUBCODE = "qq_lifetime_used"
QQ_ASSERT_RATE_LIMITED_SUBCODE = "ip_rate_limited"
HardwareFingerprintProvider = Callable[[], str | Awaitable[str]]
DiscordOAuthListenerFactory = Callable[[], DiscordOAuthLoopbackListener]
DiscordOAuthCallbackRunner = Callable[
    [DiscordOAuthLoopbackListener, str, str],
    Awaitable[tuple[str, str]],
]


def _default_signed_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


class ManagedOpenRouterReleaseBehavior(str, Enum):
    READY = "ready"
    RETRY = "retry"
    RESTART = "restart"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterReleaseDiagnostics:
    operation: str | None = None
    code: str | None = None
    error_class: str | None = None
    subcode: str | None = None
    retry_after_ms: int | None = None
    message: str | None = None


def format_managed_openrouter_diagnostics(
    diagnostics: ManagedOpenRouterReleaseDiagnostics | None,
) -> str:
    if diagnostics is None:
        return ""
    parts: list[str] = []
    if diagnostics.operation is not None:
        parts.append(f"operation={diagnostics.operation}")
    if diagnostics.code is not None:
        parts.append(f"code={diagnostics.code}")
    if diagnostics.error_class is not None:
        parts.append(f"class={diagnostics.error_class}")
    if diagnostics.subcode is not None:
        parts.append(f"subcode={diagnostics.subcode}")
    if diagnostics.retry_after_ms is not None:
        parts.append(f"retry_after_ms={diagnostics.retry_after_ms}")
    if diagnostics.message is not None:
        parts.append(f"message={diagnostics.message}")
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class TalkTogetherPassStatus:
    pass_id: str
    invite_count: int
    invite_limit: int
    bonus_translations_per_friend: int = 200


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterReleaseResult:
    behavior: ManagedOpenRouterReleaseBehavior
    message_key: str
    message_kwargs: Mapping[str, object] = field(default_factory=dict)
    diagnostics: ManagedOpenRouterReleaseDiagnostics | None = None
    retry_after_ms: int | None = None
    api_key: str | None = field(default=None, repr=False)
    local_key_available: bool = False
    pending_issue: bool = False
    single_flight_reused: bool = False
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: TalkTogetherPassStatus | None = None


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterFingerprintSalt:
    version: int
    salt: str


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterChallengeSuccess:
    challenge: str
    challenge_expires_at: str
    fingerprint_salt: ManagedOpenRouterFingerprintSalt


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterDiscordStartSuccess:
    authorization_url: str
    redirect_uri: str
    oauth_session_expires_at: str
    issue_nonce: str
    fingerprint_salt: ManagedOpenRouterFingerprintSalt
    fingerprint_salt_version: int


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterVerifySuccess:
    release_token: str
    release_token_expires_at: str


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAck:
    issue_source: str
    delivery_id: str
    managed_credential_ref: str
    ack_token: str = field(repr=False)
    expires_at: str


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAckResult:
    status: str
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: TalkTogetherPassStatus | None = None


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterIssueSuccess:
    openrouter_api_key: str = field(repr=False)
    managed_credential_ref: str | None = None
    expires_at: str | None = None
    openrouter_user_id: str | None = None
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: TalkTogetherPassStatus | None = None
    delivery_ack: ManagedKeyDeliveryAck | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterQqAssertSuccess:
    status: str
    qq_subject_ref: str = field(repr=False)
    issue: ManagedOpenRouterIssueSuccess | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterTrialStatusSuccess:
    referral_id: str | None = None
    pass_status: TalkTogetherPassStatus | None = None


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterStatusRefreshResult:
    referral_id: str | None
    pass_status: TalkTogetherPassStatus | None = None
    succeeded: bool = True


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterPreflightStop:
    reason: str


@dataclass(frozen=True, slots=True)
class ManagedOpenRouterReleaseError(Exception):
    code: str
    error_class: str
    message: str
    operation: str | None = None
    subcode: str | None = None
    retry_after_ms: int | None = None
    managed_lifecycle: str | None = None

    def __str__(self) -> str:
        return self.message or self.code

    def to_diagnostics(self) -> ManagedOpenRouterReleaseDiagnostics:
        return ManagedOpenRouterReleaseDiagnostics(
            operation=self.operation,
            code=self.code,
            error_class=self.error_class,
            subcode=self.subcode,
            retry_after_ms=self.retry_after_ms,
            message=self.message,
        )


class ManagedOpenRouterReleaseClient(Protocol):
    async def challenge(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        app_version: str,
    ) -> ManagedOpenRouterChallengeSuccess | ManagedOpenRouterPreflightStop: ...

    async def verify(self, request: dict[str, str]) -> ManagedOpenRouterVerifySuccess: ...

    async def issue(self, request: dict[str, object]) -> ManagedOpenRouterIssueSuccess: ...

    async def start_discord_oauth(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        redirect_uri: str,
        app_version: str,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterDiscordStartSuccess: ...

    async def issue_discord_managed_key(
        self,
        request: dict[str, object],
    ) -> ManagedOpenRouterIssueSuccess: ...

    async def assert_qq_credential(
        self,
        *,
        qq_identity: str,
        credential: str,
        asserted_at: str,
    ) -> ManagedOpenRouterQqAssertSuccess: ...

    async def acknowledge_managed_key_delivery(
        self,
        delivery_ack: ManagedKeyDeliveryAck,
    ) -> ManagedKeyDeliveryAckResult: ...

    async def get_trial_status(
        self,
        *,
        installation_id: str,
        timestamp: str,
        signature: str,
    ) -> ManagedOpenRouterTrialStatusSuccess: ...


@dataclass(slots=True)
class UnavailableManagedOpenRouterReleaseClient:
    async def challenge(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        app_version: str,
    ) -> ManagedOpenRouterPreflightStop:
        _ = installation_id, device_public_key, app_version
        return ManagedOpenRouterPreflightStop(reason="unavailable")

    async def verify(self, request: dict[str, str]) -> ManagedOpenRouterVerifySuccess:
        _ = request
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )

    async def issue(self, request: dict[str, object]) -> ManagedOpenRouterIssueSuccess:
        _ = request
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )

    async def start_discord_oauth(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        redirect_uri: str,
        app_version: str,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterDiscordStartSuccess:
        _ = installation_id, device_public_key, redirect_uri, app_version, referral_id
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )

    async def issue_discord_managed_key(
        self,
        request: dict[str, object],
    ) -> ManagedOpenRouterIssueSuccess:
        _ = request
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )

    async def assert_qq_credential(
        self,
        *,
        qq_identity: str,
        credential: str,
        asserted_at: str,
    ) -> ManagedOpenRouterQqAssertSuccess:
        _ = qq_identity, credential, asserted_at
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )

    async def get_trial_status(
        self,
        *,
        installation_id: str,
        timestamp: str,
        signature: str,
    ) -> ManagedOpenRouterTrialStatusSuccess:
        _ = installation_id, timestamp, signature
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )

    async def acknowledge_managed_key_delivery(
        self,
        delivery_ack: ManagedKeyDeliveryAck,
    ) -> ManagedKeyDeliveryAckResult:
        _ = delivery_ack
        raise ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
        )


@dataclass(slots=True)
class ManagedOpenRouterReleaseService:
    settings: AppSettings
    secrets: SecretStore
    client: ManagedOpenRouterReleaseClient
    persist_settings: Callable[[AppSettings], None]
    app_version: str
    raw_hardware_fingerprint_provider: HardwareFingerprintProvider | None = None
    hardware_hash_provider: InitVar[HardwareFingerprintProvider | None] = None
    signed_at_provider: Callable[[], str] = _default_signed_at
    monotonic_ms_provider: Callable[[], int] = _default_monotonic_ms
    discord_oauth_listener_factory: DiscordOAuthListenerFactory = bind_first_available
    discord_oauth_callback_runner: DiscordOAuthCallbackRunner = run_discord_oauth_callback_flow
    on_discord_callback_received: Callable[[], None] | None = None
    _prepare_task: asyncio.Task[ManagedOpenRouterReleaseResult] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _issue_task: asyncio.Task[ManagedOpenRouterReleaseResult] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _status_refresh_tasks: set[asyncio.Task[object]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _retry_after_deadline_ms: int | None = field(init=False, default=None, repr=False)
    _retry_after_diagnostics: ManagedOpenRouterReleaseDiagnostics | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _legacy_hardware_hash_provider: HardwareFingerprintProvider | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self, hardware_hash_provider: HardwareFingerprintProvider | None) -> None:
        self._legacy_hardware_hash_provider = hardware_hash_provider

    def _start_shared_task(
        self,
        attr_name: str,
        coro: Awaitable[ManagedOpenRouterReleaseResult],
    ) -> asyncio.Task[ManagedOpenRouterReleaseResult]:
        task = asyncio.create_task(coro)
        setattr(self, attr_name, task)

        def _clear(finished_task: asyncio.Task[ManagedOpenRouterReleaseResult]) -> None:
            if getattr(self, attr_name) is finished_task:
                setattr(self, attr_name, None)

        task.add_done_callback(_clear)
        return task

    async def _await_shared_task(
        self,
        task: asyncio.Task[ManagedOpenRouterReleaseResult],
        *,
        single_flight_reused: bool,
    ) -> ManagedOpenRouterReleaseResult:
        result = await asyncio.shield(task)
        if single_flight_reused:
            return replace(result, single_flight_reused=True)
        return result

    async def prepare_for_translation(
        self,
        *,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        if _is_managed_china_connection(self.settings):
            resolution = resolve_openrouter_credentials(
                self.settings,
                secrets=self.secrets,
                request_intent="TRANS",
            )
            if resolution.selected_source != OpenRouterCredentialSource.MANAGED:
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="managed_release.stop",
                )
            if resolution.api_key is not None:
                pending_ack_result = await self._retry_pending_delivery_ack_before_ready(
                    resolution.api_key
                )
                if pending_ack_result is not None:
                    return pending_ack_result
                return self._ready_with_api_key(resolution.api_key)
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        if self._issue_task is not None and not self._issue_task.done():
            return await self._await_shared_task(self._issue_task, single_flight_reused=True)
        if self._prepare_task is not None and not self._prepare_task.done():
            return await self._await_shared_task(self._prepare_task, single_flight_reused=True)

        task = self._start_shared_task(
            "_prepare_task",
            self._run_prepare_flow(referral_id=referral_id),
        )
        return await self._await_shared_task(task, single_flight_reused=False)

    async def prepare_from_qq_assertion(
        self,
        *,
        qq_identity: str,
        credential: str,
        issue_persistence_allowed: Callable[[], bool] | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        logger.info("[QQAuth] prepare_from_qq_assertion: starting QQ assertion flow")
        resolution = resolve_openrouter_credentials(
            self.settings,
            secrets=self.secrets,
            request_intent="TRANS",
        )
        if resolution.selected_source != OpenRouterCredentialSource.MANAGED:
            logger.warning("[QQAuth] prepare_from_qq_assertion: selected source is not Managed")
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if not _is_managed_china_connection(self.settings):
            logger.warning(
                "[QQAuth] prepare_from_qq_assertion: selected connection is not Managed China"
            )
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_before_ready(
                resolution.api_key
            )
            if pending_ack_result is not None:
                return pending_ack_result
            return self._ready_with_api_key(resolution.api_key)

        retry_result = self._result_for_retry_after_window()
        if retry_result is not None:
            logger.info(
                "[QQAuth] prepare_from_qq_assertion: in retry_after window, returning retry"
            )
            return retry_result

        try:
            logger.info("[QQAuth] prepare_from_qq_assertion: calling client.assert_qq_credential")
            qq_assert_response = await self.client.assert_qq_credential(
                qq_identity=qq_identity,
                credential=credential,
                asserted_at=self.signed_at_provider(),
            )
            logger.info(
                "[QQAuth] prepare_from_qq_assertion: broker returned status=%s issued=%s",
                qq_assert_response.status,
                qq_assert_response.issue is not None,
            )
        except ManagedOpenRouterReleaseError as exc:
            safe_error = _sanitize_qq_assert_release_error(exc)
            logger.warning(
                "[QQAuth] prepare_from_qq_assertion: broker error code=%s class=%s subcode=%s retry_after_ms=%s",
                safe_error.code,
                safe_error.error_class,
                safe_error.subcode,
                safe_error.retry_after_ms,
            )
            qq_error_result = self._handle_qq_assert_release_error(safe_error)
            if qq_error_result is not None:
                return qq_error_result
            return self._handle_release_error(safe_error, operation="qq_assert")

        if qq_assert_response.issue is None:
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        normalized_qq_api_key = _normalize_optional_text(
            qq_assert_response.issue.openrouter_api_key
        )
        if normalized_qq_api_key is None:
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        if issue_persistence_allowed is not None and not issue_persistence_allowed():
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        return await self._persist_qq_managed_issue_success(
            qq_assert_response.issue,
            api_key=normalized_qq_api_key,
        )

    async def ensure_key_for_llm_start(self) -> ManagedOpenRouterReleaseResult:
        resolution = resolve_openrouter_credentials(self.settings, secrets=self.secrets)
        if resolution.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_before_ready(
                resolution.api_key
            )
            if pending_ack_result is not None:
                return pending_ack_result
            return self._ready_with_api_key(resolution.api_key)
        if _is_managed_china_connection(self.settings):
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        if self._prepare_task is not None and not self._prepare_task.done():
            prepare_result = await self._await_shared_task(
                self._prepare_task,
                single_flight_reused=True,
            )
            if prepare_result.behavior != ManagedOpenRouterReleaseBehavior.READY:
                return prepare_result
            if prepare_result.local_key_available or prepare_result.api_key is not None:
                return prepare_result

        if self._issue_task is not None and not self._issue_task.done():
            return await self._await_shared_task(self._issue_task, single_flight_reused=True)

        if _normalize_optional_text(self.settings.managed_identity.release_token) is None:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
            )

        retry_result = self._result_for_retry_after_window()
        if retry_result is not None:
            return retry_result

        return await self._await_or_start_issue_flow()

    async def refresh_managed_status(self) -> ManagedOpenRouterStatusRefreshResult:
        """Best-effort signed-request status refresh for owned Pass ID and live pass status."""

        observed_referral_id = normalize_owned_referral_id(
            self.settings.managed_identity.referral_id
        )
        if self._closed:
            return ManagedOpenRouterStatusRefreshResult(
                referral_id=observed_referral_id,
                succeeded=False,
            )
        if self.settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterStatusRefreshResult(
                referral_id=observed_referral_id,
                succeeded=False,
            )

        current_task = asyncio.current_task()
        if current_task is not None:
            self._status_refresh_tasks.add(current_task)
        try:
            try:
                bundle = load_existing_managed_identity_bundle(self.settings, self.secrets)
            except Exception:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=observed_referral_id,
                    succeeded=False,
                )
            if bundle is None:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=observed_referral_id,
                    succeeded=False,
                )

            try:
                signed_request = bundle.sign_status_request(timestamp=self.signed_at_provider())
                status_response = await self.client.get_trial_status(
                    installation_id=signed_request["installation_id"],
                    timestamp=signed_request["timestamp"],
                    signature=signed_request["signature"],
                )
            except Exception:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=normalize_owned_referral_id(
                        self.settings.managed_identity.referral_id
                    ),
                    succeeded=False,
                )

            if self._closed:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=normalize_owned_referral_id(
                        self.settings.managed_identity.referral_id
                    ),
                    succeeded=False,
                )
            returned_referral_id = normalize_owned_referral_id(
                getattr(status_response, "referral_id", None)
            )
            latest_referral_id = normalize_owned_referral_id(
                self.settings.managed_identity.referral_id
            )
            pass_status = getattr(status_response, "pass_status", None)
            if returned_referral_id is None:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=latest_referral_id,
                    pass_status=None,
                    succeeded=True,
                )
            if latest_referral_id != observed_referral_id:
                if pass_status is not None and pass_status.pass_id != latest_referral_id:
                    pass_status = None
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=latest_referral_id,
                    pass_status=pass_status,
                    succeeded=True,
                )
            if returned_referral_id != latest_referral_id:
                previous_referral_id = self.settings.managed_identity.referral_id
                self.settings.managed_identity.referral_id = returned_referral_id
                try:
                    self.persist_settings(self.settings)
                except Exception:
                    self.settings.managed_identity.referral_id = previous_referral_id
                    return ManagedOpenRouterStatusRefreshResult(
                        referral_id=latest_referral_id,
                        succeeded=False,
                    )

            final_referral_id = normalize_owned_referral_id(
                self.settings.managed_identity.referral_id
            )
            if pass_status is not None and pass_status.pass_id != final_referral_id:
                pass_status = None
            return ManagedOpenRouterStatusRefreshResult(
                referral_id=final_referral_id,
                pass_status=pass_status,
                succeeded=True,
            )
        finally:
            if current_task is not None:
                self._status_refresh_tasks.discard(current_task)

    async def refresh_owned_referral_id_from_status(self) -> str | None:
        """Best-effort signed status refresh for the persisted owned Referral ID."""

        return (await self.refresh_managed_status()).referral_id

    async def _run_prepare_flow(
        self,
        *,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        resolution = resolve_openrouter_credentials(
            self.settings,
            secrets=self.secrets,
            request_intent="TRANS",
        )
        if resolution.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_before_ready(
                resolution.api_key
            )
            if pending_ack_result is not None:
                return pending_ack_result
            return self._ready_with_api_key(resolution.api_key)
        if _is_managed_china_connection(self.settings):
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        bundle = ensure_managed_identity_bundle(
            self.settings,
            self.secrets,
            persist_settings=self.persist_settings,
        )
        retry_result = self._result_for_retry_after_window()
        if retry_result is not None:
            return retry_result
        if _normalize_optional_text(self.settings.managed_identity.release_token) is not None:
            if self._verified_snapshot() is None:
                clear_temporary_managed_release_state(self.settings)
                self.persist_settings(self.settings)
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                    message_key="managed_release.restart",
                )
            return await self._await_or_start_issue_flow()

        listener: DiscordOAuthLoopbackListener | None = None
        try:
            try:
                listener = self.discord_oauth_listener_factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return self._handle_release_error(
                    _discord_listener_release_error(exc),
                    operation="discord_start",
                )
            try:
                start_response = await self.client.start_discord_oauth(
                    installation_id=bundle.installation_id,
                    device_public_key=bundle.device_public_key,
                    redirect_uri=listener.redirect_uri,
                    app_version=self.app_version,
                    referral_id=referral_id,
                )
            except ManagedOpenRouterReleaseError as exc:
                return self._handle_release_error(exc, operation="discord_start")

            if start_response.redirect_uri != listener.redirect_uri:
                return self._handle_release_error(
                    ManagedOpenRouterReleaseError(
                        code="discord_redirect_mismatch",
                        error_class="terminal",
                        message="Discord OAuth broker returned a different redirect URI",
                        operation="discord_start",
                    ),
                    operation="discord_start",
                )

            try:
                code, state = await self.discord_oauth_callback_runner(
                    listener,
                    start_response.authorization_url,
                    start_response.oauth_session_expires_at,
                )
            except asyncio.CancelledError:
                raise
            except ManagedOpenRouterReleaseError as exc:
                return self._handle_release_error(exc, operation="discord_callback")
            except (
                DiscordOAuthCallbackError,
                DiscordOAuthLoopbackClosedError,
                TimeoutError,
            ) as exc:
                return self._handle_release_error(
                    _discord_callback_release_error(exc),
                    operation="discord_callback",
                )
            self._notify_discord_callback_received()
            try:
                hardware_hash = await self._resolve_hardware_hash(
                    fingerprint_salt=start_response.fingerprint_salt,
                )
            except Exception:
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="managed_release.stop",
                )

            issue_request = bundle.sign_discord_issue_request(
                code=code,
                state=state,
                redirect_uri=listener.redirect_uri,
                hardware_hash=hardware_hash,
                hardware_hash_salt_version=start_response.fingerprint_salt_version,
                app_version=self.app_version,
                reason="llm_start",
                budget_usd=MANAGED_OPENROUTER_TRIAL_BUDGET_USD,
                model=_resolve_managed_issue_model(self.settings),
                issue_nonce=start_response.issue_nonce,
                signed_at=self.signed_at_provider(),
            )
            try:
                issue_response = await self.client.issue_discord_managed_key(issue_request)
            except ManagedOpenRouterReleaseError as exc:
                return self._handle_release_error(exc, operation="discord_issue")

            return await self._persist_managed_issue_success(issue_response)
        finally:
            if listener is not None:
                listener.close()

    def _notify_discord_callback_received(self) -> None:
        if self.on_discord_callback_received is None:
            return
        with contextlib.suppress(Exception):
            self.on_discord_callback_received()

    async def _await_or_start_issue_flow(self) -> ManagedOpenRouterReleaseResult:
        resolution = resolve_openrouter_credentials(
            self.settings,
            secrets=self.secrets,
            request_intent="TRANS",
        )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_before_ready(
                resolution.api_key
            )
            if pending_ack_result is not None:
                return pending_ack_result
            return self._ready_with_api_key(resolution.api_key)
        if _is_managed_china_connection(self.settings):
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        if self._issue_task is not None and not self._issue_task.done():
            return await self._await_shared_task(self._issue_task, single_flight_reused=True)

        retry_result = self._result_for_retry_after_window()
        if retry_result is not None:
            return retry_result

        task = self._start_shared_task("_issue_task", self._run_issue_flow())
        return await self._await_shared_task(task, single_flight_reused=False)

    async def _run_issue_flow(self) -> ManagedOpenRouterReleaseResult:
        if _is_managed_china_connection(self.settings):
            self._clear_retry_after()
            return _managed_china_key_unavailable_result()

        bundle = ensure_managed_identity_bundle(
            self.settings,
            self.secrets,
            persist_settings=self.persist_settings,
        )
        release_token = _normalize_optional_text(self.settings.managed_identity.release_token)
        if release_token is None:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
            )
        verified_snapshot = self._verified_snapshot()
        if verified_snapshot is None:
            clear_temporary_managed_release_state(self.settings)
            self.persist_settings(self.settings)
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
            )
        verified_hardware_hash, _verified_hardware_hash_salt_version = verified_snapshot
        issue_request = bundle.sign_issue_request(
            release_token=release_token,
            reason="llm_start",
            hardware_hash=verified_hardware_hash,
            budget_usd=MANAGED_OPENROUTER_TRIAL_BUDGET_USD,
            model=_resolve_managed_issue_model(self.settings),
            signed_at=self.signed_at_provider(),
        )
        try:
            issue_response = await self.client.issue(issue_request)
        except ManagedOpenRouterReleaseError as exc:
            return self._handle_release_error(exc, operation="issue")

        return await self._persist_managed_issue_success(issue_response)

    async def _persist_managed_issue_success(
        self,
        issue_response: ManagedOpenRouterIssueSuccess,
        *,
        secret_key: str = OPENROUTER_MANAGED_API_KEY_SECRET,
    ) -> ManagedOpenRouterReleaseResult:
        try:
            self.secrets.set(secret_key, issue_response.openrouter_api_key)
        except Exception:
            previous_release_token = self.settings.managed_identity.release_token
            previous_release_token_expires_at = (
                self.settings.managed_identity.release_token_expires_at
            )
            previous_verified_hardware_hash = self.settings.managed_identity.verified_hardware_hash
            previous_verified_hardware_hash_salt_version = (
                self.settings.managed_identity.verified_hardware_hash_salt_version
            )
            try:
                self.secrets.delete(secret_key)
            except Exception:
                pass
            clear_temporary_managed_release_state(self.settings)
            try:
                self.persist_settings(self.settings)
            except Exception:
                self.settings.managed_identity.release_token = previous_release_token
                self.settings.managed_identity.release_token_expires_at = (
                    previous_release_token_expires_at
                )
                self.settings.managed_identity.verified_hardware_hash = (
                    previous_verified_hardware_hash
                )
                self.settings.managed_identity.verified_hardware_hash_salt_version = (
                    previous_verified_hardware_hash_salt_version
                )
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        best_effort_store_managed_openrouter_user_identifier(
            self.settings,
            secrets=self.secrets,
            openrouter_user_id=issue_response.openrouter_user_id,
        )
        store_managed_entitlement_snapshot(
            self.settings,
            managed_credential_ref=issue_response.managed_credential_ref,
            expires_at=issue_response.expires_at,
        )
        record_local_managed_claim_source(self.settings, MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
        returned_referral_id = normalize_owned_referral_id(issue_response.referral_id)
        if returned_referral_id is not None:
            self.settings.managed_identity.referral_id = returned_referral_id
        final_referral_id = normalize_owned_referral_id(self.settings.managed_identity.referral_id)
        pass_status = issue_response.pass_status
        if pass_status is not None and pass_status.pass_id != final_referral_id:
            pass_status = None
        if issue_response.delivery_ack is not None:
            try:
                self._store_pending_delivery_ack(issue_response.delivery_ack)
            except Exception:
                self._delete_secret_safely(secret_key)
                self._clear_pending_delivery_ack_state(delete_token=True)
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="managed_release.stop",
                )
        clear_temporary_managed_release_state(self.settings)
        try:
            self.persist_settings(self.settings)
        except Exception:
            self._delete_secret_safely(secret_key)
            self._clear_pending_delivery_ack_state(delete_token=True)
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if issue_response.delivery_ack is not None:
            ack_result = await self._acknowledge_pending_delivery_ack(
                issue_response.delivery_ack,
                api_key=issue_response.openrouter_api_key,
            )
            if ack_result.behavior != ManagedOpenRouterReleaseBehavior.READY:
                return ack_result
            return ack_result
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key=issue_response.openrouter_api_key,
            local_key_available=True,
            referral_bonus_applied=issue_response.referral_bonus_applied is True,
            referral_id=final_referral_id,
            pass_status=pass_status,
        )

    async def _persist_qq_managed_issue_success(
        self,
        issue_response: ManagedOpenRouterIssueSuccess,
        *,
        api_key: str,
    ) -> ManagedOpenRouterReleaseResult:
        try:
            self.secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, api_key)
        except Exception:
            try:
                self.secrets.delete(OPENROUTER_MANAGED_QQ_API_KEY_SECRET)
            except Exception:
                pass
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )

        store_managed_entitlement_snapshot(
            self.settings,
            managed_credential_ref=issue_response.managed_credential_ref,
            expires_at=issue_response.expires_at,
        )
        record_local_managed_claim_source(self.settings, MANAGED_AUTH_CLAIM_SOURCE_QQ)
        if issue_response.delivery_ack is not None:
            try:
                self._store_pending_delivery_ack(issue_response.delivery_ack)
            except Exception:
                self._delete_secret_safely(OPENROUTER_MANAGED_QQ_API_KEY_SECRET)
                self._clear_pending_delivery_ack_state(delete_token=True)
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="managed_release.stop",
                )
        try:
            self.persist_settings(self.settings)
        except Exception:
            self._delete_secret_safely(OPENROUTER_MANAGED_QQ_API_KEY_SECRET)
            self._clear_pending_delivery_ack_state(delete_token=True)
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if issue_response.delivery_ack is not None:
            ack_result = await self._acknowledge_pending_delivery_ack(
                issue_response.delivery_ack,
                api_key=api_key,
            )
            if ack_result.behavior != ManagedOpenRouterReleaseBehavior.READY:
                return ack_result
            return ack_result
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key=api_key,
            local_key_available=True,
        )

    async def _retry_pending_delivery_ack_before_ready(
        self,
        api_key: str,
    ) -> ManagedOpenRouterReleaseResult | None:
        pending_ack = self._pending_delivery_ack_from_settings()
        if pending_ack is None:
            if self._has_pending_delivery_ack_metadata():
                source = self.settings.managed_identity.pending_delivery_ack_source or "discord"
                self._delete_local_managed_key_for_ack_source(source)
                self._clear_pending_delivery_ack_state(delete_token=True)
                self.persist_settings(self.settings)
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                    message_key="managed_release.restart",
                    diagnostics=ManagedOpenRouterReleaseDiagnostics(
                        operation="delivery_ack",
                        code="delivery_ack_state_incomplete",
                        error_class="terminal",
                        message="managed key delivery acknowledgement state is incomplete",
                    ),
                )
            return None
        ack_result = await self._acknowledge_pending_delivery_ack(
            pending_ack,
            api_key=api_key,
        )
        if ack_result.behavior == ManagedOpenRouterReleaseBehavior.READY:
            return None
        return ack_result

    async def _acknowledge_pending_delivery_ack(
        self,
        delivery_ack: ManagedKeyDeliveryAck,
        *,
        api_key: str,
    ) -> ManagedOpenRouterReleaseResult:
        try:
            ack_result = await self.client.acknowledge_managed_key_delivery(delivery_ack)
        except ManagedOpenRouterReleaseError as exc:
            diagnostics = exc.to_diagnostics()
            if diagnostics.operation is None:
                diagnostics = replace(diagnostics, operation="delivery_ack")
            if exc.error_class in {"terminal", "security_fail"}:
                self._delete_local_managed_key_for_ack_source(delivery_ack.issue_source)
                self._clear_pending_delivery_ack_state(delete_token=True)
                self.persist_settings(self.settings)
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                    message_key="managed_release.restart",
                    diagnostics=diagnostics,
                )
            retry_after_ms = _normalize_retry_after_ms(exc.retry_after_ms)
            if retry_after_ms is not None:
                self._retry_after_deadline_ms = self.monotonic_ms_provider() + retry_after_ms
                self._retry_after_diagnostics = diagnostics
            else:
                self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key=(
                    "managed_release.retry_after_ms"
                    if retry_after_ms is not None
                    else "managed_release.retry"
                ),
                message_kwargs=(
                    {"retry_after_ms": retry_after_ms} if retry_after_ms is not None else {}
                ),
                diagnostics=replace(diagnostics, retry_after_ms=retry_after_ms),
                retry_after_ms=retry_after_ms,
                api_key=api_key,
                local_key_available=True,
                pending_issue=True,
            )
        except Exception:
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="managed_release.retry",
                diagnostics=ManagedOpenRouterReleaseDiagnostics(
                    operation="delivery_ack",
                    code="delivery_ack_pending",
                    error_class="retryable",
                    message="managed key delivery acknowledgement is pending",
                ),
                api_key=api_key,
                local_key_available=True,
                pending_issue=True,
            )

        returned_referral_id = normalize_owned_referral_id(ack_result.referral_id)
        if returned_referral_id is not None:
            self.settings.managed_identity.referral_id = returned_referral_id
        final_referral_id = normalize_owned_referral_id(self.settings.managed_identity.referral_id)
        pass_status = ack_result.pass_status
        if pass_status is not None and pass_status.pass_id != final_referral_id:
            pass_status = None
        if not self._clear_persisted_pending_delivery_ack_state(delivery_ack):
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="managed_release.retry",
                diagnostics=ManagedOpenRouterReleaseDiagnostics(
                    operation="delivery_ack",
                    code="delivery_ack_clear_pending_failed",
                    error_class="retryable",
                    message="managed key delivery acknowledgement cleanup is pending",
                ),
                api_key=api_key,
                local_key_available=True,
                pending_issue=True,
            )
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key=api_key,
            local_key_available=True,
            referral_bonus_applied=ack_result.referral_bonus_applied is True,
            referral_id=final_referral_id,
            pass_status=pass_status,
        )

    def _store_pending_delivery_ack(self, delivery_ack: ManagedKeyDeliveryAck) -> None:
        source = _normalize_delivery_ack_source(delivery_ack.issue_source)
        if source is None:
            raise ValueError("delivery ACK source must be discord or qq")
        self.secrets.set(_delivery_ack_token_secret_key(source), delivery_ack.ack_token)
        self.settings.managed_identity.pending_delivery_ack_source = source
        self.settings.managed_identity.pending_delivery_ack_id = delivery_ack.delivery_id
        self.settings.managed_identity.pending_delivery_ack_managed_credential_ref = (
            delivery_ack.managed_credential_ref
        )
        self.settings.managed_identity.pending_delivery_ack_expires_at = delivery_ack.expires_at

    def _pending_delivery_ack_from_settings(self) -> ManagedKeyDeliveryAck | None:
        source = _normalize_delivery_ack_source(
            self.settings.managed_identity.pending_delivery_ack_source
        )
        delivery_id = _normalize_optional_text(
            self.settings.managed_identity.pending_delivery_ack_id
        )
        managed_credential_ref = _normalize_optional_text(
            self.settings.managed_identity.pending_delivery_ack_managed_credential_ref
        )
        expires_at = _normalize_optional_text(
            self.settings.managed_identity.pending_delivery_ack_expires_at
        )
        if (
            source is None
            or delivery_id is None
            or managed_credential_ref is None
            or expires_at is None
        ):
            return None
        ack_token = _normalize_optional_text(
            self.secrets.get(_delivery_ack_token_secret_key(source))
        )
        if ack_token is None:
            return None
        return ManagedKeyDeliveryAck(
            issue_source=source,
            delivery_id=delivery_id,
            managed_credential_ref=managed_credential_ref,
            ack_token=ack_token,
            expires_at=expires_at,
        )

    def _has_pending_delivery_ack_metadata(self) -> bool:
        return any(
            _normalize_optional_text(value) is not None
            for value in (
                self.settings.managed_identity.pending_delivery_ack_source,
                self.settings.managed_identity.pending_delivery_ack_id,
                self.settings.managed_identity.pending_delivery_ack_managed_credential_ref,
                self.settings.managed_identity.pending_delivery_ack_expires_at,
            )
        )

    def _clear_pending_delivery_ack_state(self, *, delete_token: bool) -> None:
        source = _normalize_delivery_ack_source(
            self.settings.managed_identity.pending_delivery_ack_source
        )
        self.settings.managed_identity.pending_delivery_ack_source = None
        self.settings.managed_identity.pending_delivery_ack_id = None
        self.settings.managed_identity.pending_delivery_ack_managed_credential_ref = None
        self.settings.managed_identity.pending_delivery_ack_expires_at = None
        if delete_token:
            for candidate_source in (source, "discord", "qq"):
                if candidate_source is not None:
                    self._delete_secret_safely(_delivery_ack_token_secret_key(candidate_source))

    def _clear_persisted_pending_delivery_ack_state(
        self, delivery_ack: ManagedKeyDeliveryAck
    ) -> bool:
        previous_source = self.settings.managed_identity.pending_delivery_ack_source
        previous_id = self.settings.managed_identity.pending_delivery_ack_id
        previous_ref = self.settings.managed_identity.pending_delivery_ack_managed_credential_ref
        previous_expires_at = self.settings.managed_identity.pending_delivery_ack_expires_at
        self._clear_pending_delivery_ack_state(delete_token=False)
        try:
            self.persist_settings(self.settings)
        except Exception:
            self.settings.managed_identity.pending_delivery_ack_source = previous_source
            self.settings.managed_identity.pending_delivery_ack_id = previous_id
            self.settings.managed_identity.pending_delivery_ack_managed_credential_ref = (
                previous_ref
            )
            self.settings.managed_identity.pending_delivery_ack_expires_at = previous_expires_at
            return False
        self._delete_secret_safely(_delivery_ack_token_secret_key(delivery_ack.issue_source))
        return True

    def _delete_local_managed_key_for_ack_source(self, source: str) -> None:
        normalized_source = _normalize_delivery_ack_source(source)
        if normalized_source == "qq":
            self._delete_secret_safely(OPENROUTER_MANAGED_QQ_API_KEY_SECRET)
        elif normalized_source == "discord":
            self._delete_secret_safely(OPENROUTER_MANAGED_API_KEY_SECRET)

    def _delete_secret_safely(self, key: str) -> None:
        try:
            self.secrets.delete(key)
        except Exception:
            pass

    def _ready_with_api_key(self, api_key: str) -> ManagedOpenRouterReleaseResult:
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key=api_key,
            local_key_available=True,
        )

    def _handle_release_error(
        self,
        error: ManagedOpenRouterReleaseError,
        *,
        operation: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        diagnostics = error.to_diagnostics()
        if diagnostics.operation is None and operation is not None:
            diagnostics = replace(diagnostics, operation=operation)
        if error.error_class == "security_fail" and error.subcode in BINDING_MISMATCH_SUBCODES:
            try:
                regenerate_managed_identity_bundle(
                    self.settings,
                    self.secrets,
                    persist_settings=self.persist_settings,
                )
            except Exception:
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="managed_release.stop",
                    diagnostics=diagnostics,
                )
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
                diagnostics=diagnostics,
            )

        if error.managed_lifecycle == "revoked":
            clear_temporary_managed_release_state(self.settings)
            self.persist_settings(self.settings)
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.revoked_contact",
                diagnostics=diagnostics,
            )

        if error.code == "issuance_suspended":
            retry_after_ms = _normalize_retry_after_ms(error.retry_after_ms)
            self._clear_retry_after()
            message_kwargs: dict[str, object] = {}
            if retry_after_ms is not None:
                message_kwargs["retry_after_ms"] = retry_after_ms
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="managed_release.brake",
                message_kwargs=message_kwargs,
                diagnostics=replace(diagnostics, retry_after_ms=retry_after_ms),
                retry_after_ms=retry_after_ms,
            )

        if (
            error.error_class == "security_fail"
            or error.code
            in {
                "challenge_expired",
            }
            or error.subcode == "release_token_expired"
        ):
            clear_temporary_managed_release_state(self.settings)
            self.persist_settings(self.settings)
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
                diagnostics=diagnostics,
            )

        if error.error_class == "terminal":
            clear_temporary_managed_release_state(self.settings)
            self.persist_settings(self.settings)
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key=(
                    "managed_release.not_eligible"
                    if error.code == "trial_not_eligible"
                    else "managed_release.stop"
                ),
                diagnostics=diagnostics,
            )

        retry_after_ms = _normalize_retry_after_ms(error.retry_after_ms)
        if retry_after_ms is not None:
            self._retry_after_deadline_ms = self.monotonic_ms_provider() + retry_after_ms
            self._retry_after_diagnostics = diagnostics
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="managed_release.retry_after_ms",
                message_kwargs={"retry_after_ms": retry_after_ms},
                diagnostics=replace(diagnostics, retry_after_ms=retry_after_ms),
                retry_after_ms=retry_after_ms,
            )

        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="managed_release.retry",
            diagnostics=diagnostics,
        )

    def _handle_qq_assert_release_error(
        self,
        error: ManagedOpenRouterReleaseError,
    ) -> ManagedOpenRouterReleaseResult | None:
        diagnostics = error.to_diagnostics()
        if diagnostics.operation is None:
            diagnostics = replace(diagnostics, operation="qq_assert")

        if error.subcode in QQ_ASSERT_CREDENTIAL_SUBCODES:
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="qq_auth.error.credential_mismatch",
                diagnostics=diagnostics,
            )

        if error.subcode == QQ_ASSERT_LIFETIME_USED_SUBCODE:
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="qq_auth.error.lifetime_used",
                diagnostics=diagnostics,
            )

        if error.subcode == QQ_ASSERT_RATE_LIMITED_SUBCODE:
            retry_after_ms = _normalize_retry_after_ms(error.retry_after_ms)
            diagnostics = replace(diagnostics, retry_after_ms=retry_after_ms)
            message_kwargs: dict[str, object] = {}
            if retry_after_ms is not None:
                self._retry_after_deadline_ms = self.monotonic_ms_provider() + retry_after_ms
                self._retry_after_diagnostics = diagnostics
                message_kwargs["retry_after_ms"] = retry_after_ms
            else:
                self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="qq_auth.error.retry",
                message_kwargs=message_kwargs,
                diagnostics=diagnostics,
                retry_after_ms=retry_after_ms,
            )

        return None

    async def _resolve_hardware_hash(
        self,
        *,
        fingerprint_salt: ManagedOpenRouterFingerprintSalt,
    ) -> str:
        if self.raw_hardware_fingerprint_provider is not None:
            raw_hardware_fingerprint = await _resolve_provider_without_blocking_event_loop(
                self.raw_hardware_fingerprint_provider
            )
            return compute_hardware_hash(
                fingerprint_salt=fingerprint_salt.salt,
                raw_fingerprint=raw_hardware_fingerprint,
            )
        if self._legacy_hardware_hash_provider is not None:
            hardware_hash = await _resolve_provider_without_blocking_event_loop(
                self._legacy_hardware_hash_provider
            )
            normalized_hardware_hash = _normalize_optional_text(hardware_hash)
            if normalized_hardware_hash is None:
                raise ValueError("hardware_hash_provider must return a non-empty string")
            return normalized_hardware_hash
        raise RuntimeError("managed hardware fingerprint provider is not configured")

    def _verified_snapshot(self) -> tuple[str, int] | None:
        verified_hardware_hash = _normalize_optional_text(
            self.settings.managed_identity.verified_hardware_hash
        )
        verified_hardware_hash_salt_version = (
            self.settings.managed_identity.verified_hardware_hash_salt_version
        )
        if verified_hardware_hash is None or verified_hardware_hash_salt_version is None:
            return None
        return verified_hardware_hash, verified_hardware_hash_salt_version

    def _result_for_retry_after_window(self) -> ManagedOpenRouterReleaseResult | None:
        if self._retry_after_deadline_ms is None:
            return None
        now_ms = self.monotonic_ms_provider()
        if now_ms >= self._retry_after_deadline_ms:
            self._clear_retry_after()
            return None
        remaining_ms = self._retry_after_deadline_ms - now_ms
        diagnostics = self._retry_after_diagnostics
        message_key = "managed_release.retry_after_ms"
        if diagnostics is not None:
            diagnostics = replace(diagnostics, retry_after_ms=remaining_ms)
            if (
                diagnostics.operation == "qq_assert"
                and diagnostics.subcode == QQ_ASSERT_RATE_LIMITED_SUBCODE
            ):
                message_key = "qq_auth.error.retry"
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key=message_key,
            message_kwargs={"retry_after_ms": remaining_ms},
            diagnostics=diagnostics,
            retry_after_ms=remaining_ms,
        )

    def _clear_retry_after(self) -> None:
        self._retry_after_deadline_ms = None
        self._retry_after_diagnostics = None

    async def close(self) -> None:
        self._closed = True
        prepare_task = self._prepare_task
        issue_task = self._issue_task
        status_refresh_tasks = tuple(self._status_refresh_tasks)
        self._prepare_task = None
        self._issue_task = None
        self._status_refresh_tasks.clear()

        current_task = asyncio.current_task()
        active_tasks = [
            task
            for task in (prepare_task, issue_task, *status_refresh_tasks)
            if task is not None and not task.done() and task is not current_task
        ]
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_result = close_client()
            if inspect.isawaitable(close_result):
                await close_result


class ManagedOpenRouterDelegateFactory(Protocol):
    def __call__(self, api_key: str) -> LLMProvider: ...


@dataclass(slots=True)
class ManagedOpenRouterUserFacingError(RuntimeError):
    message_key: str
    message_kwargs: Mapping[str, object] = field(default_factory=dict)
    diagnostics: ManagedOpenRouterReleaseDiagnostics | None = None

    def __str__(self) -> str:
        from puripuly_heart.ui.i18n import t

        try:
            return t(self.message_key, **dict(self.message_kwargs))
        except Exception:
            return self.message_key


@dataclass(slots=True)
class ManagedOpenRouterLLMProvider(LLMProvider):
    release_service: object
    delegate_factory: ManagedOpenRouterDelegateFactory
    on_delegate_ready: Callable[[], object] | None = None
    _delegate: LLMProvider | None = field(init=False, default=None, repr=False)
    _delegate_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    @property
    def model(self) -> object | None:
        if self._delegate is not None:
            return getattr(self._delegate, "model", None)
        settings = getattr(self.release_service, "settings", None)
        openrouter_settings = getattr(settings, "openrouter", None)
        return getattr(openrouter_settings, "llm_model", None)

    @property
    def selected_source(self) -> object | None:
        if self._delegate is not None:
            return getattr(self._delegate, "selected_source", None)
        settings = getattr(self.release_service, "settings", None)
        openrouter_settings = getattr(settings, "openrouter", None)
        return getattr(openrouter_settings, "selected_source", None)

    async def _ensure_delegate(self) -> LLMProvider:
        if self._delegate is not None:
            return self._delegate

        async with self._delegate_lock:
            if self._delegate is not None:
                return self._delegate
            ensure_key = getattr(self.release_service, "ensure_key_for_llm_start")
            try:
                result = await ensure_key()
            except ManagedOpenRouterUserFacingError:
                raise
            except Exception as exc:
                raise ManagedOpenRouterUserFacingError(
                    message_key="managed_release.retry",
                    diagnostics=ManagedOpenRouterReleaseDiagnostics(message=str(exc)),
                ) from exc
            if not isinstance(result, ManagedOpenRouterReleaseResult):
                raise ManagedOpenRouterUserFacingError(
                    message_key="managed_release.retry",
                    diagnostics=ManagedOpenRouterReleaseDiagnostics(
                        message="managed release service returned an unsupported result"
                    ),
                )
            if result.behavior != ManagedOpenRouterReleaseBehavior.READY or not result.api_key:
                raise ManagedOpenRouterUserFacingError(
                    message_key=result.message_key or "managed_release.restart",
                    message_kwargs=result.message_kwargs,
                    diagnostics=result.diagnostics,
                )
            try:
                self._delegate = self.delegate_factory(result.api_key)
            except Exception as exc:
                raise ManagedOpenRouterUserFacingError(
                    message_key="managed_release.retry",
                    diagnostics=ManagedOpenRouterReleaseDiagnostics(message=str(exc)),
                ) from exc
            if self.on_delegate_ready is not None:
                callback_result = self.on_delegate_ready()
                if inspect.isawaitable(callback_result):
                    await callback_result
            return self._delegate

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        delegate = await self._ensure_delegate()
        return await delegate.translate(
            utterance_id=utterance_id,
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

    async def close(self) -> None:
        if self._delegate is not None:
            await self._delegate.close()
            self._delegate = None


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_delivery_ack_source(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in {"discord", "qq"} else None


def _delivery_ack_token_secret_key(source: str) -> str:
    normalized_source = _normalize_delivery_ack_source(source)
    if normalized_source == "qq":
        return OPENROUTER_MANAGED_QQ_DELIVERY_ACK_TOKEN_SECRET
    if normalized_source == "discord":
        return OPENROUTER_MANAGED_DELIVERY_ACK_TOKEN_SECRET
    raise ValueError("delivery ACK source must be discord or qq")


def _is_managed_china_connection(settings: AppSettings) -> bool:
    return (
        getattr(getattr(settings, "translation", None), "connection", None)
        == TranslationConnection.MANAGED_CHINA
    )


def _managed_china_key_unavailable_result() -> ManagedOpenRouterReleaseResult:
    return ManagedOpenRouterReleaseResult(
        behavior=ManagedOpenRouterReleaseBehavior.STOP,
        message_key="qq_auth.error.key_unavailable",
    )


def _discord_listener_release_error(error: Exception) -> ManagedOpenRouterReleaseError:
    message = _exception_message(
        error,
        default="Discord OAuth loopback listener unavailable",
    )
    return ManagedOpenRouterReleaseError(
        code="discord_loopback_unavailable",
        error_class="retryable",
        message=f"Discord OAuth loopback listener unavailable: {message}",
        operation="discord_start",
    )


def _discord_callback_release_error(error: Exception) -> ManagedOpenRouterReleaseError:
    if isinstance(error, DiscordOAuthCallbackError):
        return ManagedOpenRouterReleaseError(
            code="discord_oauth_callback_error",
            error_class="retryable",
            message=f"Discord OAuth callback failed: {error.error}",
            operation="discord_callback",
            subcode=error.error,
        )
    if isinstance(error, TimeoutError):
        return ManagedOpenRouterReleaseError(
            code="discord_oauth_timeout",
            error_class="retryable",
            message=_exception_message(
                error,
                default="timed out waiting for Discord OAuth callback",
            ),
            operation="discord_callback",
        )
    return ManagedOpenRouterReleaseError(
        code="discord_oauth_callback_closed",
        error_class="retryable",
        message=_exception_message(
            error,
            default="Discord OAuth callback listener closed before completion",
        ),
        operation="discord_callback",
    )


def _sanitize_qq_assert_release_error(
    error: ManagedOpenRouterReleaseError,
) -> ManagedOpenRouterReleaseError:
    return ManagedOpenRouterReleaseError(
        operation=error.operation or "qq_assert",
        code=error.code,
        error_class=error.error_class,
        subcode=error.subcode,
        retry_after_ms=_normalize_retry_after_ms(error.retry_after_ms),
        message=_qq_assert_safe_error_message(error.subcode),
        managed_lifecycle=error.managed_lifecycle,
    )


def _qq_assert_safe_error_message(subcode: str | None) -> str:
    if subcode == "ip_rate_limited":
        return "QQ credential verification is rate limited"
    if subcode == "qq_lifetime_used":
        return "QQ credential has already been used"
    return "QQ credential verification failed"


def _exception_message(error: Exception, *, default: str) -> str:
    message = str(error).strip()
    return message or default


def _normalize_retry_after_ms(value: int | None) -> int | None:
    if value is None:
        return None
    return max(0, int(value))


async def _resolve_maybe_awaitable(value: str | Awaitable[str]) -> str:
    if inspect.isawaitable(value):
        return await value
    return value


async def _resolve_provider_without_blocking_event_loop(
    provider: HardwareFingerprintProvider,
) -> str:
    if inspect.iscoroutinefunction(provider):
        return await _resolve_maybe_awaitable(provider())
    return await _resolve_maybe_awaitable(await asyncio.to_thread(provider))


def _resolve_managed_issue_model(settings: AppSettings) -> str:
    selection_alias = settings.openrouter.selection_alias
    if selection_alias is None:
        selection_alias = openrouter_alias_for_fields(
            model=settings.openrouter.llm_model.value,
            source=settings.openrouter.selected_source.value,
        )
    profile = get_openrouter_llm_profile(
        selection_alias.value if hasattr(selection_alias, "value") else selection_alias
    )
    if profile is not None and profile.openrouter_model is not None:
        return profile.openrouter_model
    return settings.openrouter.llm_model.value
