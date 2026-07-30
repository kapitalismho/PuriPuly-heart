from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol
from uuid import UUID

from puripuly_heart.app.ports.broker_client import ManagedKeyDeliveryAckRequest
from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.config.llm_profiles import (
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.settings import (
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
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
from puripuly_heart.core.managed_identity import (
    ensure_managed_identity_bundle,
    load_existing_managed_identity_bundle,
    regenerate_managed_identity_bundle,
)
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.domain.models import Translation

from .openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OpenRouterCredentialRuntimeConfig,
    best_effort_store_managed_openrouter_user_identifier,
    clear_temporary_managed_release_state,
    resolve_openrouter_credentials,
)
from .openrouter_handoff import store_managed_entitlement_snapshot

MANAGED_OPENROUTER_TRIAL_BUDGET_USD = 0.07
DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET = "openrouter_managed_delivery_ack_token"
QQ_MANAGED_DELIVERY_ACK_TOKEN_SECRET = "openrouter_managed_qq_delivery_ack_token"
BINDING_MISMATCH_SUBCODES = {
    "device_public_key_registered",
    "installation_binding_mismatch",
}
HardwareFingerprintProvider = Callable[[], str | Awaitable[str]]
DiscordOAuthListenerFactory = Callable[[], DiscordOAuthLoopbackListener]
DiscordOAuthCallbackRunner = Callable[
    [DiscordOAuthLoopbackListener, str, str],
    Awaitable[tuple[str, str]],
]


@dataclass(frozen=True, slots=True)
class OpenRouterReleaseRuntimeConfig:
    """Narrow read-only runtime DTO for managed OpenRouter release flows.

    Carries only the OpenRouter selection fields the release service reads:
    the resolved model, credential source, and selection alias. The service
    never imports ``AppSettings``; the wiring boundary builds this DTO.
    """

    llm_model: OpenRouterLLMModel
    selected_source: OpenRouterCredentialSource
    selection_alias: OpenRouterSelectionAlias | None
    managed_credential_kind: str = "standard"


def _default_signed_at() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def _managed_release_auth_task_name(attr_name: str) -> str:
    if attr_name == "_prepare_task":
        return "managed-openrouter-prepare"
    if attr_name == "_issue_task":
        return "managed-openrouter-issue"
    return f"managed-openrouter-{attr_name.strip('_').replace('_', '-')}"


def _record_close_task_result(
    failures: list[BaseException],
    result: object,
) -> None:
    if isinstance(result, asyncio.CancelledError):
        return
    if isinstance(result, BaseException):
        failures.append(result)


def _raise_close_failures(message: str, failures: list[BaseException]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    exception_failures = [failure for failure in failures if isinstance(failure, Exception)]
    if len(exception_failures) == len(failures):
        raise ExceptionGroup(message, exception_failures)
    raise BaseExceptionGroup(message, failures)


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
    for attr_name, rendered_name in (
        ("operation", "operation"),
        ("code", "code"),
        ("error_class", "class"),
        ("subcode", "subcode"),
        ("retry_after_ms", "retry_after_ms"),
    ):
        safe_value = _safe_diagnostic_scalar(getattr(diagnostics, attr_name))
        if safe_value is not None:
            parts.append(f"{rendered_name}={safe_value}")
    if diagnostics.message is not None:
        parts.append("message=<redacted>")
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
    api_key: str | None = None
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
class ManagedOpenRouterIssueSuccess:
    openrouter_api_key: str
    managed_credential_ref: str | None = None
    expires_at: str | None = None
    openrouter_user_id: str | None = None
    referral_bonus_applied: bool = False
    referral_id: str | None = None
    pass_status: TalkTogetherPassStatus | None = None
    delivery_ack_required: bool = False
    delivery_id: str | None = None
    delivery_ack_token: str | None = field(default=None, repr=False)
    delivery_ack_expires_at: str | None = None


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

    async def get_trial_status(
        self,
        *,
        installation_id: str,
        timestamp: str,
        signature: str,
    ) -> ManagedOpenRouterTrialStatusSuccess: ...

    async def acknowledge_managed_key_delivery(
        self,
        request: ManagedKeyDeliveryAckRequest,
    ) -> object: ...


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

    async def acknowledge_managed_key_delivery(
        self,
        request: ManagedKeyDeliveryAckRequest,
    ) -> object:
        _ = request
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


@dataclass(slots=True)
class ManagedOpenRouterReleaseService:
    openrouter_config: OpenRouterReleaseRuntimeConfig
    managed_state: ManagedIdentityStatePort
    secrets: SecretStore
    client: ManagedOpenRouterReleaseClient
    app_version: str
    raw_hardware_fingerprint_provider: HardwareFingerprintProvider | None = None
    hardware_hash_provider: InitVar[HardwareFingerprintProvider | None] = None
    signed_at_provider: Callable[[], str] = _default_signed_at
    monotonic_ms_provider: Callable[[], int] = _default_monotonic_ms
    discord_oauth_listener_factory: DiscordOAuthListenerFactory = bind_first_available
    discord_oauth_callback_runner: DiscordOAuthCallbackRunner = run_discord_oauth_callback_flow
    on_discord_callback_received: Callable[[], None] | None = None
    oauth_runtime: OAuthRuntime = field(default_factory=OAuthRuntime)
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

    @property
    def model(self) -> object | None:
        return self.openrouter_config.llm_model

    @property
    def selected_source(self) -> object | None:
        return self.openrouter_config.selected_source

    def _credential_runtime_config(self) -> OpenRouterCredentialRuntimeConfig:
        return OpenRouterCredentialRuntimeConfig(
            selected_source=self.openrouter_config.selected_source,
            installation_id=self.managed_state.installation_id,
            managed_credential_kind=self.openrouter_config.managed_credential_kind,
            active_managed_credential_ref=self.managed_state.active_managed_credential_ref,
            active_managed_expires_at=self.managed_state.active_managed_expires_at,
        )

    def _uses_qq_managed_credentials(self) -> bool:
        return self.openrouter_config.managed_credential_kind == "qq"

    def _qq_managed_auth_required_result(self) -> ManagedOpenRouterReleaseResult:
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RESTART,
            message_key="qq_managed_auth.required",
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="resolve_openrouter_credentials",
                code="qq_managed_key_required",
                error_class="terminal",
            ),
        )

    def _start_shared_task(
        self,
        attr_name: str,
        coro: Awaitable[ManagedOpenRouterReleaseResult],
    ) -> asyncio.Task[ManagedOpenRouterReleaseResult]:
        task = self.oauth_runtime.create_auth_task(
            coro,
            task_name=_managed_release_auth_task_name(attr_name),
        )
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
        if self._issue_task is not None and not self._issue_task.done():
            return await self._await_shared_task(self._issue_task, single_flight_reused=True)
        if self._prepare_task is not None and not self._prepare_task.done():
            return await self._await_shared_task(self._prepare_task, single_flight_reused=True)

        task = self._start_shared_task(
            "_prepare_task",
            self._run_prepare_flow(referral_id=referral_id),
        )
        return await self._await_shared_task(task, single_flight_reused=False)

    async def ensure_key_for_llm_start(self) -> ManagedOpenRouterReleaseResult:
        resolution = resolve_openrouter_credentials(
            self._credential_runtime_config(), secrets=self.secrets
        )
        if resolution.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_if_needed()
            if pending_ack_result is not None:
                return pending_ack_result
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.READY,
                message_key="managed_release.ready",
                api_key=resolution.api_key,
                local_key_available=True,
            )
        if self._uses_qq_managed_credentials():
            return self._qq_managed_auth_required_result()

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

        if _normalize_optional_text(self.managed_state.release_token) is None:
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

        observed_referral_id = normalize_owned_referral_id(self.managed_state.referral_id)
        if self._closed:
            return ManagedOpenRouterStatusRefreshResult(
                referral_id=observed_referral_id,
                succeeded=False,
            )
        if self.openrouter_config.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterStatusRefreshResult(
                referral_id=observed_referral_id,
                succeeded=False,
            )

        current_task = asyncio.current_task()
        if current_task is not None:
            self._status_refresh_tasks.add(current_task)
        try:
            try:
                bundle = load_existing_managed_identity_bundle(self.managed_state, self.secrets)
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
                    referral_id=normalize_owned_referral_id(self.managed_state.referral_id),
                    succeeded=False,
                )

            if self._closed:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=normalize_owned_referral_id(self.managed_state.referral_id),
                    succeeded=False,
                )
            returned_referral_id = normalize_owned_referral_id(
                getattr(status_response, "referral_id", None)
            )
            latest_referral_id = normalize_owned_referral_id(self.managed_state.referral_id)
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
                previous_referral_id = self.managed_state.referral_id
                self.managed_state.referral_id = returned_referral_id
                try:
                    self.managed_state.persist()
                except Exception:
                    self.managed_state.referral_id = previous_referral_id
                    return ManagedOpenRouterStatusRefreshResult(
                        referral_id=latest_referral_id,
                        succeeded=False,
                    )

            final_referral_id = normalize_owned_referral_id(self.managed_state.referral_id)
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

    async def _retry_pending_delivery_ack_if_needed(
        self,
    ) -> ManagedOpenRouterReleaseResult | None:
        source = _normalize_optional_text(
            getattr(self.managed_state, "pending_delivery_ack_source", None)
        )
        delivery_id = _normalize_optional_text(
            getattr(self.managed_state, "pending_delivery_ack_delivery_id", None)
        )
        managed_credential_ref = _normalize_optional_text(
            getattr(self.managed_state, "pending_delivery_ack_managed_credential_ref", None)
        )
        expires_at = _normalize_optional_text(
            getattr(self.managed_state, "pending_delivery_ack_expires_at", None)
        )
        if source is None and delivery_id is None and managed_credential_ref is None:
            return None
        if source not in {"discord", "qq"} or delivery_id is None or managed_credential_ref is None:
            return self._delivery_ack_retry_result("pending_delivery_ack_metadata_incomplete")
        token_key = _delivery_ack_token_secret_key(source)
        try:
            token = self.secrets.get(token_key)
        except Exception:
            return self._delivery_ack_retry_result("pending_delivery_ack_token_read_failed")
        if _normalize_optional_text(token) is None:
            return self._delivery_ack_retry_result("pending_delivery_ack_token_missing")
        try:
            ack_result = await self.client.acknowledge_managed_key_delivery(
                ManagedKeyDeliveryAckRequest(
                    delivery_id=delivery_id,
                    managed_credential_ref=managed_credential_ref,
                    delivery_ack_token=token,
                )
            )
        except ManagedOpenRouterReleaseError as exc:
            return self._handle_release_error(exc, operation="managed_key_delivery_ack")
        except Exception:
            return self._delivery_ack_retry_result("pending_delivery_ack_request_failed")
        if not (
            getattr(ack_result, "succeeded", False) is True
            and getattr(ack_result, "status", None) in {"acknowledged", "already_acknowledged"}
        ):
            return self._delivery_ack_retry_result(
                _safe_diagnostic_scalar(getattr(ack_result, "status", None))
                or "pending_delivery_ack_failed"
            )
        try:
            self.secrets.delete(token_key)
        except Exception:
            return self._delivery_ack_retry_result("pending_delivery_ack_token_clear_failed")
        self.managed_state.pending_delivery_ack_source = None
        self.managed_state.pending_delivery_ack_delivery_id = None
        self.managed_state.pending_delivery_ack_managed_credential_ref = None
        self.managed_state.pending_delivery_ack_expires_at = None
        try:
            self.managed_state.persist()
        except Exception:
            self.managed_state.pending_delivery_ack_source = source
            self.managed_state.pending_delivery_ack_delivery_id = delivery_id
            self.managed_state.pending_delivery_ack_managed_credential_ref = managed_credential_ref
            self.managed_state.pending_delivery_ack_expires_at = expires_at
            try:
                self.secrets.set(token_key, token)
            except Exception:
                return self._delivery_ack_retry_result("pending_delivery_ack_token_restore_failed")
            return self._delivery_ack_retry_result("pending_delivery_ack_clear_persist_failed")
        return None

    def _delivery_ack_retry_result(self, code: str) -> ManagedOpenRouterReleaseResult:
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="managed_release.retry",
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="managed_key_delivery_ack",
                code=code,
                error_class="retryable",
            ),
        )

    async def refresh_owned_referral_id_from_status(self) -> str | None:
        """Best-effort signed status refresh for the persisted owned Referral ID."""

        return (await self.refresh_managed_status()).referral_id

    async def _run_prepare_flow(
        self,
        *,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        resolution = resolve_openrouter_credentials(
            self._credential_runtime_config(),
            secrets=self.secrets,
            request_intent="TRANS",
        )
        if resolution.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_if_needed()
            if pending_ack_result is not None:
                return pending_ack_result
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.READY,
                message_key="managed_release.ready",
                api_key=resolution.api_key,
                local_key_available=True,
            )
        if self._uses_qq_managed_credentials():
            return self._qq_managed_auth_required_result()

        bundle = ensure_managed_identity_bundle(
            self.managed_state,
            self.secrets,
        )
        retry_result = self._result_for_retry_after_window()
        if retry_result is not None:
            return retry_result
        if _normalize_optional_text(self.managed_state.release_token) is not None:
            if self._verified_snapshot() is None:
                clear_temporary_managed_release_state(self.managed_state)
                self.managed_state.persist()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                    message_key="managed_release.restart",
                )
            return await self._await_or_start_issue_flow()

        listener: DiscordOAuthLoopbackListener | None = None
        try:
            try:
                listener = self.discord_oauth_listener_factory()
                self.oauth_runtime.attach_loopback_listener(
                    listener,
                    listener_name="discord-loopback",
                )
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
                model=_resolve_managed_issue_model(self.openrouter_config),
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
                await self.oauth_runtime.close_loopback_listener(
                    listener,
                    listener_name="discord-loopback",
                )

    def _notify_discord_callback_received(self) -> None:
        if self.on_discord_callback_received is None:
            return
        with contextlib.suppress(Exception):
            self.on_discord_callback_received()

    async def _await_or_start_issue_flow(self) -> ManagedOpenRouterReleaseResult:
        resolution = resolve_openrouter_credentials(
            self._credential_runtime_config(),
            secrets=self.secrets,
            request_intent="TRANS",
        )
        if resolution.api_key is not None:
            pending_ack_result = await self._retry_pending_delivery_ack_if_needed()
            if pending_ack_result is not None:
                return pending_ack_result
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.READY,
                message_key="managed_release.ready",
                api_key=resolution.api_key,
                local_key_available=True,
            )
        if self._uses_qq_managed_credentials():
            return self._qq_managed_auth_required_result()

        if self._issue_task is not None and not self._issue_task.done():
            return await self._await_shared_task(self._issue_task, single_flight_reused=True)

        retry_result = self._result_for_retry_after_window()
        if retry_result is not None:
            return retry_result

        task = self._start_shared_task("_issue_task", self._run_issue_flow())
        return await self._await_shared_task(task, single_flight_reused=False)

    async def _run_issue_flow(self) -> ManagedOpenRouterReleaseResult:
        if self._uses_qq_managed_credentials():
            return self._qq_managed_auth_required_result()
        bundle = ensure_managed_identity_bundle(
            self.managed_state,
            self.secrets,
        )
        release_token = _normalize_optional_text(self.managed_state.release_token)
        if release_token is None:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
            )
        verified_snapshot = self._verified_snapshot()
        if verified_snapshot is None:
            clear_temporary_managed_release_state(self.managed_state)
            self.managed_state.persist()
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
            model=_resolve_managed_issue_model(self.openrouter_config),
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
    ) -> ManagedOpenRouterReleaseResult:
        delivery_id = _normalize_optional_text(issue_response.delivery_id)
        managed_credential_ref = _normalize_optional_text(issue_response.managed_credential_ref)
        delivery_ack_token = _normalize_optional_text(issue_response.delivery_ack_token)
        delivery_ack_expires_at = _normalize_optional_text(issue_response.delivery_ack_expires_at)
        if issue_response.delivery_ack_required:
            if delivery_id is None or managed_credential_ref is None or delivery_ack_token is None:
                self._clear_retry_after()
                return self._delivery_ack_retry_result("delivery_ack_metadata_malformed")
            try:
                self.secrets.set(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET, delivery_ack_token)
            except Exception:
                self._clear_retry_after()
                return self._delivery_ack_retry_result(
                    "delivery_ack_token_store_failed_before_local_key_write"
                )
            best_effort_store_managed_openrouter_user_identifier(
                self._credential_runtime_config(),
                secrets=self.secrets,
                openrouter_user_id=issue_response.openrouter_user_id,
            )
            store_managed_entitlement_snapshot(
                self.managed_state,
                managed_credential_ref=issue_response.managed_credential_ref,
                expires_at=issue_response.expires_at,
            )
            returned_referral_id = normalize_owned_referral_id(issue_response.referral_id)
            if returned_referral_id is not None:
                self.managed_state.referral_id = returned_referral_id
            final_referral_id = normalize_owned_referral_id(self.managed_state.referral_id)
            pass_status = issue_response.pass_status
            if pass_status is not None and pass_status.pass_id != final_referral_id:
                pass_status = None
            clear_temporary_managed_release_state(self.managed_state)
            self.managed_state.pending_delivery_ack_source = "discord"
            self.managed_state.pending_delivery_ack_delivery_id = delivery_id
            self.managed_state.pending_delivery_ack_managed_credential_ref = managed_credential_ref
            self.managed_state.pending_delivery_ack_expires_at = delivery_ack_expires_at
            try:
                self.managed_state.persist()
            except Exception:
                with contextlib.suppress(Exception):
                    self.secrets.delete(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET)
                self._clear_retry_after()
                return self._delivery_ack_retry_result("delivery_ack_pending_persist_failed")
            try:
                self.secrets.set(
                    OPENROUTER_MANAGED_API_KEY_SECRET, issue_response.openrouter_api_key
                )
            except Exception:
                self._clear_retry_after()
                return self._delivery_ack_retry_result("delivery_ack_local_key_store_failed")
            try:
                ack_result = await self.client.acknowledge_managed_key_delivery(
                    ManagedKeyDeliveryAckRequest(
                        delivery_id=delivery_id,
                        managed_credential_ref=managed_credential_ref,
                        delivery_ack_token=delivery_ack_token,
                    )
                )
            except ManagedOpenRouterReleaseError as exc:
                return self._handle_release_error(exc, operation="managed_key_delivery_ack")
            except Exception:
                return self._delivery_ack_retry_result("delivery_ack_request_failed")
            if not (
                getattr(ack_result, "succeeded", False) is True
                and getattr(ack_result, "status", None) in {"acknowledged", "already_acknowledged"}
            ):
                return self._delivery_ack_retry_result(
                    _safe_diagnostic_scalar(getattr(ack_result, "status", None))
                    or "delivery_ack_failed"
                )
            try:
                self.secrets.delete(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET)
            except Exception:
                return self._delivery_ack_retry_result("delivery_ack_token_clear_failed")
            self.managed_state.pending_delivery_ack_source = None
            self.managed_state.pending_delivery_ack_delivery_id = None
            self.managed_state.pending_delivery_ack_managed_credential_ref = None
            self.managed_state.pending_delivery_ack_expires_at = None
            try:
                self.managed_state.persist()
            except Exception:
                with contextlib.suppress(Exception):
                    self.secrets.set(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET, delivery_ack_token)
                self.managed_state.pending_delivery_ack_source = "discord"
                self.managed_state.pending_delivery_ack_delivery_id = delivery_id
                self.managed_state.pending_delivery_ack_managed_credential_ref = (
                    managed_credential_ref
                )
                self.managed_state.pending_delivery_ack_expires_at = delivery_ack_expires_at
                return self._delivery_ack_retry_result("delivery_ack_clear_persist_failed")
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
        try:
            self.secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, issue_response.openrouter_api_key)
        except Exception:
            if issue_response.delivery_ack_required:
                with contextlib.suppress(Exception):
                    self.secrets.delete(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET)
            previous_release_token = self.managed_state.release_token
            previous_release_token_expires_at = self.managed_state.release_token_expires_at
            previous_verified_hardware_hash = self.managed_state.verified_hardware_hash
            previous_verified_hardware_hash_salt_version = (
                self.managed_state.verified_hardware_hash_salt_version
            )
            try:
                self.secrets.delete(OPENROUTER_MANAGED_API_KEY_SECRET)
            except Exception:
                pass
            clear_temporary_managed_release_state(self.managed_state)
            try:
                self.managed_state.persist()
            except Exception:
                self.managed_state.release_token = previous_release_token
                self.managed_state.release_token_expires_at = previous_release_token_expires_at
                self.managed_state.verified_hardware_hash = previous_verified_hardware_hash
                self.managed_state.verified_hardware_hash_salt_version = (
                    previous_verified_hardware_hash_salt_version
                )
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
            )
        best_effort_store_managed_openrouter_user_identifier(
            self._credential_runtime_config(),
            secrets=self.secrets,
            openrouter_user_id=issue_response.openrouter_user_id,
        )
        store_managed_entitlement_snapshot(
            self.managed_state,
            managed_credential_ref=issue_response.managed_credential_ref,
            expires_at=issue_response.expires_at,
        )
        returned_referral_id = normalize_owned_referral_id(issue_response.referral_id)
        if returned_referral_id is not None:
            self.managed_state.referral_id = returned_referral_id
        final_referral_id = normalize_owned_referral_id(self.managed_state.referral_id)
        pass_status = issue_response.pass_status
        if pass_status is not None and pass_status.pass_id != final_referral_id:
            pass_status = None
        clear_temporary_managed_release_state(self.managed_state)
        if issue_response.delivery_ack_required:
            self.managed_state.pending_delivery_ack_source = "discord"
            self.managed_state.pending_delivery_ack_delivery_id = delivery_id
            self.managed_state.pending_delivery_ack_managed_credential_ref = managed_credential_ref
            self.managed_state.pending_delivery_ack_expires_at = delivery_ack_expires_at
            try:
                self.managed_state.persist()
            except Exception:
                with contextlib.suppress(Exception):
                    self.secrets.delete(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET)
                    self.secrets.delete(OPENROUTER_MANAGED_API_KEY_SECRET)
                self._clear_retry_after()
                return self._delivery_ack_retry_result("delivery_ack_pending_persist_failed")
            try:
                ack_result = await self.client.acknowledge_managed_key_delivery(
                    ManagedKeyDeliveryAckRequest(
                        delivery_id=delivery_id,
                        managed_credential_ref=managed_credential_ref,
                        delivery_ack_token=delivery_ack_token,
                    )
                )
            except ManagedOpenRouterReleaseError as exc:
                return self._handle_release_error(exc, operation="managed_key_delivery_ack")
            except Exception:
                return self._delivery_ack_retry_result("delivery_ack_request_failed")
            if not (
                getattr(ack_result, "succeeded", False) is True
                and getattr(ack_result, "status", None) in {"acknowledged", "already_acknowledged"}
            ):
                return self._delivery_ack_retry_result(
                    _safe_diagnostic_scalar(getattr(ack_result, "status", None))
                    or "delivery_ack_failed"
                )
            try:
                self.secrets.delete(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET)
            except Exception:
                return self._delivery_ack_retry_result("delivery_ack_token_clear_failed")
            self.managed_state.pending_delivery_ack_source = None
            self.managed_state.pending_delivery_ack_delivery_id = None
            self.managed_state.pending_delivery_ack_managed_credential_ref = None
            self.managed_state.pending_delivery_ack_expires_at = None
            try:
                self.managed_state.persist()
            except Exception:
                with contextlib.suppress(Exception):
                    self.secrets.set(DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET, delivery_ack_token)
                self.managed_state.pending_delivery_ack_source = "discord"
                self.managed_state.pending_delivery_ack_delivery_id = delivery_id
                self.managed_state.pending_delivery_ack_managed_credential_ref = (
                    managed_credential_ref
                )
                self.managed_state.pending_delivery_ack_expires_at = delivery_ack_expires_at
                return self._delivery_ack_retry_result("delivery_ack_clear_persist_failed")
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
        self.managed_state.persist()
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
                    self.managed_state,
                    self.secrets,
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
            clear_temporary_managed_release_state(self.managed_state)
            self.managed_state.persist()
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
            clear_temporary_managed_release_state(self.managed_state)
            self.managed_state.persist()
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="managed_release.restart",
                diagnostics=diagnostics,
            )

        if error.error_class == "terminal":
            clear_temporary_managed_release_state(self.managed_state)
            self.managed_state.persist()
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
        verified_hardware_hash = _normalize_optional_text(self.managed_state.verified_hardware_hash)
        verified_hardware_hash_salt_version = self.managed_state.verified_hardware_hash_salt_version
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
        if diagnostics is not None:
            diagnostics = replace(diagnostics, retry_after_ms=remaining_ms)
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="managed_release.retry_after_ms",
            message_kwargs={"retry_after_ms": remaining_ms},
            diagnostics=diagnostics,
            retry_after_ms=remaining_ms,
        )

    def _clear_retry_after(self) -> None:
        self._retry_after_deadline_ms = None
        self._retry_after_diagnostics = None

    async def close(self) -> None:
        self._closed = True
        status_refresh_tasks = tuple(self._status_refresh_tasks)
        self._prepare_task = None
        self._issue_task = None
        self._status_refresh_tasks.clear()

        failures: list[BaseException] = []
        try:
            await self.oauth_runtime.close()
        except Exception as exc:
            failures.append(exc)

        current_task = asyncio.current_task()
        active_tasks = [
            task
            for task in status_refresh_tasks
            if task is not None and not task.done() and task is not current_task
        ]
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            results = await asyncio.gather(*active_tasks, return_exceptions=True)
            for result in results:
                _record_close_task_result(failures, result)

        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            try:
                close_result = close_client()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as exc:
                failures.append(exc)

        _raise_close_failures("ManagedOpenRouterReleaseService close failed", failures)


class ManagedOpenRouterDelegateFactory(Protocol):
    def __call__(self, api_key: str) -> LLMProvider: ...


@dataclass(slots=True)
class ManagedOpenRouterUserFacingError(RuntimeError):
    message_key: str
    message_kwargs: Mapping[str, object] = field(default_factory=dict)
    diagnostics: ManagedOpenRouterReleaseDiagnostics | None = None

    def __str__(self) -> str:
        parts = [
            "ManagedOpenRouterUserFacingError",
            f"message_key={_safe_diagnostic_label(self.message_key) or 'unknown'}",
        ]
        for key, value in sorted(self.message_kwargs.items()):
            safe_key = _safe_diagnostic_label(key)
            if not safe_key:
                continue
            if isinstance(value, str):
                continue
            safe_value = _safe_diagnostic_scalar(value)
            if safe_value is not None:
                parts.append(f"{safe_key}={safe_value}")

        diagnostics = self.diagnostics
        if diagnostics is not None:
            for name in ("operation", "code", "error_class", "subcode", "retry_after_ms"):
                safe_value = _safe_diagnostic_scalar(getattr(diagnostics, name, None))
                if safe_value is not None:
                    parts.append(f"{name}={safe_value}")
        return " ".join(parts)


def _safe_diagnostic_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 64:
        return None
    if not all(char.isalnum() or char in "._:-" for char in text):
        return None
    return text


def _safe_diagnostic_scalar(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value) if value == value and value not in (float("inf"), float("-inf")) else None
    return _safe_diagnostic_label(value)


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
        return getattr(self.release_service, "model", None)

    @property
    def selected_source(self) -> object | None:
        if self._delegate is not None:
            return getattr(self._delegate, "selected_source", None)
        return getattr(self.release_service, "selected_source", None)

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


def _normalize_optional_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _delivery_ack_token_secret_key(source: str) -> str:
    if source == "qq":
        return QQ_MANAGED_DELIVERY_ACK_TOKEN_SECRET
    return DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET


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
