from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from functools import partial
from typing import Protocol
from uuid import UUID

from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckRequest,
    ManagedOperationResumeRequest,
    ManagedOperationStatusRequest,
    ManagedOperationStatusResult,
)
from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.app.services.managed.managed_operation import (
    MANAGED_OPERATION_RESUME_TOKEN_SECRET,
    MANAGED_OPERATION_SOURCE_DISCORD,
    is_valid_resume_token,
    new_managed_operation_id,
    new_managed_operation_resume_token,
)
from puripuly_heart.config.llm_profiles import (
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
)
from puripuly_heart.config.provider_values import (
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.core.discord.discord_managed_oauth import (
    DiscordOAuthBrowserLaunchError,
    DiscordOAuthStateError,
    register_discord_oauth_expected_state,
    run_discord_oauth_callback_flow,
)
from puripuly_heart.core.discord_oauth_loopback import (
    DiscordOAuthCallbackError,
    DiscordOAuthLoopbackClosedError,
    DiscordOAuthLoopbackListener,
    bind_first_available,
)
from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.hardware_fingerprint import compute_hardware_hash
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.managed_credential import (
    ManagedCredentialBinding,
    credential_binding_secret,
    delivery_expired,
    managed_key_secret,
)
from puripuly_heart.core.managed_identity import (
    ensure_managed_identity_bundle,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.openrouter.managed_operation_identity import is_valid_operation_id
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.domain.models import Translation

from .openrouter_credentials import (
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
    failure_diagnostics: ErrorDiagnostics | None = None


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
class ManagedOpenRouterReleaseResult:
    behavior: ManagedOpenRouterReleaseBehavior
    message_key: str | None = None
    message_kwargs: Mapping[str, object] = field(default_factory=dict)
    diagnostics: ManagedOpenRouterReleaseDiagnostics | None = None
    retry_after_ms: int | None = None
    api_key: str | None = None
    local_key_available: bool = False
    pending_issue: bool = False
    single_flight_reused: bool = False


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
    delivery_ack_required: bool = False
    delivery_id: str | None = None
    delivery_ack_token: str | None = field(default=None, repr=False)
    delivery_ack_expires_at: str | None = None


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
    http_status: int | None = field(default=None, compare=False)
    failure_diagnostics: ErrorDiagnostics | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_diagnostics is None:
            object.__setattr__(
                self,
                "failure_diagnostics",
                provider_failure_report(
                    self, provider="openrouter", operation="provider_build"
                ).diagnostics,
            )

    def __str__(self) -> str:
        return self.message or self.code

    def to_diagnostics(self) -> ManagedOpenRouterReleaseDiagnostics:
        diagnostics = self.failure_diagnostics
        return ManagedOpenRouterReleaseDiagnostics(
            operation=self.operation,
            code=self.code,
            error_class=self.error_class,
            subcode=self.subcode,
            retry_after_ms=self.retry_after_ms,
            failure_diagnostics=diagnostics,
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
    ) -> ManagedOpenRouterDiscordStartSuccess: ...

    async def issue_discord_managed_key(
        self,
        request: dict[str, object],
    ) -> ManagedOpenRouterIssueSuccess: ...

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
    ) -> ManagedOpenRouterDiscordStartSuccess:
        _ = installation_id, device_public_key, redirect_uri, app_version
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
        allow_authorization: bool = True,
    ) -> ManagedOpenRouterReleaseResult:
        if (
            not allow_authorization
            and self._prepare_task is not None
            and not self._prepare_task.done()
        ):
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="discord_auth.error.action_required",
            )
        if self._issue_task is not None and not self._issue_task.done():
            return await self._await_shared_task(self._issue_task, single_flight_reused=True)
        if self._prepare_task is not None and not self._prepare_task.done():
            return await self._await_shared_task(self._prepare_task, single_flight_reused=True)

        task = self._start_shared_task(
            "_prepare_task",
            self._run_prepare_flow(
                allow_authorization=allow_authorization,
            ),
        )
        return await self._await_shared_task(task, single_flight_reused=False)

    async def ensure_key_for_llm_start(self) -> ManagedOpenRouterReleaseResult:
        for task in (self._issue_task, self._prepare_task):
            if task is not None and not task.done():
                return await self._await_shared_task(task, single_flight_reused=True)
        return await self.prepare_for_translation(allow_authorization=False)

    def _managed_source(self) -> str:
        return "qq" if self._uses_qq_managed_credentials() else "discord"

    async def _reconcile_pending_delivery(self) -> ManagedOpenRouterReleaseResult:
        if self.managed_state.pending_managed_operation_source == self._managed_source():
            recovered = await self._recover_pending_managed_operation()
            if recovered is not None:
                return recovered
        return self._managed_operation_action_required_result(
            code="managed_delivery_reconciliation_required",
            message_key=f"{'qq_managed_auth' if self._uses_qq_managed_credentials() else 'discord_auth'}.error.action_required",
            operation_status=None,
        )

    async def _retry_pending_delivery_ack_if_needed(
        self,
    ) -> ManagedOpenRouterReleaseResult | None:
        state = self.managed_state
        source = state.pending_delivery_ack_source
        if source is None:
            return None
        if source != self._managed_source():
            return self._managed_operation_action_required_result(
                code="managed_other_source_pending",
                message_key="discord_auth.error.other_source_pending",
                operation_status=None,
            )
        delivery_id = state.pending_delivery_ack_delivery_id
        ref = state.pending_delivery_ack_managed_credential_ref
        try:
            binding, credential, token = await run_owned_thread_call(
                lambda: (
                    ManagedCredentialBinding.decode(
                        self.secrets.get(credential_binding_secret(source))
                    ),
                    self.secrets.get(managed_key_secret(source)),
                    self.secrets.get(_delivery_ack_token_secret_key(source)),
                )
            )
        except Exception:
            return self._delivery_ack_retry_result("pending_delivery_observation_failed")
        operation_source = state.pending_managed_operation_source
        operation_id = state.pending_managed_operation_id
        if (
            not delivery_id
            or not ref
            or not token
            or binding is None
            or delivery_expired(state.pending_delivery_ack_expires_at)
            or not binding.matches(
                credential=credential,
                source=source,
                installation_id=state.installation_id,
                operation_id=operation_id if operation_source == source else None,
                delivery_id=delivery_id,
                managed_credential_ref=ref,
            )
        ):
            return await self._reconcile_pending_delivery()
        try:
            ack = await self.client.acknowledge_managed_key_delivery(
                ManagedKeyDeliveryAckRequest(
                    delivery_id=delivery_id, managed_credential_ref=ref, delivery_ack_token=token
                )
            )
        except ManagedOpenRouterReleaseError as exc:
            return self._handle_release_error(exc, operation="managed_key_delivery_ack")
        except Exception:
            return self._delivery_ack_retry_result("pending_delivery_ack_request_failed")
        if not ack.succeeded:
            if ack.status in {"expired", "invalid", "mismatched", "failed"}:
                return await self._reconcile_pending_delivery()
            return self._delivery_ack_retry_result(ack.status, diagnostics=ack.diagnostics)
        snapshot = state.snapshot()
        clear_operation = operation_source == source and operation_id == binding.operation_id
        state.active_managed_credential_ref = binding.managed_credential_ref
        state.pending_delivery_ack_source = None
        state.pending_delivery_ack_delivery_id = None
        state.pending_delivery_ack_managed_credential_ref = None
        state.pending_delivery_ack_expires_at = None
        if clear_operation:
            state.pending_managed_operation_id = None
            state.pending_managed_operation_source = None
            state.pending_managed_operation_installation_id = None
            state.pending_managed_operation_state = None
        try:
            await run_owned_thread_call(state.persist)
        except Exception:
            state.restore(snapshot)
            return self._delivery_ack_retry_result("pending_delivery_ack_clear_persist_failed")

        def clear_tokens() -> None:
            self.secrets.delete(_delivery_ack_token_secret_key(source))
            if clear_operation:
                self.secrets.delete(MANAGED_OPERATION_RESUME_TOKEN_SECRET)

        try:
            await run_owned_thread_call(clear_tokens)
        except Exception:
            return self._delivery_ack_retry_result("pending_delivery_ack_token_clear_failed")
        return None

    def _delivery_ack_retry_result(
        self, code: str, *, diagnostics: ErrorDiagnostics | None = None
    ) -> ManagedOpenRouterReleaseResult:
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="managed_release.retry",
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="managed_key_delivery_ack",
                code=code,
                error_class="retryable",
                failure_diagnostics=diagnostics,
            ),
        )

    async def _run_prepare_flow(
        self,
        *,
        allow_authorization: bool = True,
    ) -> ManagedOpenRouterReleaseResult:
        if self.selected_source != OpenRouterCredentialSource.MANAGED:
            return ManagedOpenRouterReleaseResult(behavior=ManagedOpenRouterReleaseBehavior.STOP)
        pending_source = self.managed_state.pending_managed_operation_source
        if (
            pending_source is not None
            and pending_source != self._managed_source()
            and (self.managed_state.pending_delivery_ack_source != self._managed_source())
        ):
            return self._managed_operation_action_required_result(
                code="managed_other_source_pending",
                message_key="discord_auth.error.other_source_pending",
                operation_status=None,
            )
        pending_ack_result = await self._retry_pending_delivery_ack_if_needed()
        if pending_ack_result is not None:
            return pending_ack_result
        pending_operation_result = await self._recover_pending_managed_operation()
        if pending_operation_result is not None:
            return pending_operation_result
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
                api_key=resolution.api_key,
                local_key_available=True,
            )
        if self._uses_qq_managed_credentials():
            return self._qq_managed_auth_required_result()

        if (
            not allow_authorization
            and _normalize_optional_text(self.managed_state.release_token) is None
        ):
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="discord_auth.error.action_required",
            )

        bundle = await run_owned_thread_call(
            partial(ensure_managed_identity_bundle, self.managed_state, self.secrets)
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

        if not allow_authorization:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="discord_auth.error.action_required",
            )

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
                register_discord_oauth_expected_state(listener, start_response.authorization_url)
            except DiscordOAuthStateError:
                return self._handle_release_error(
                    ManagedOpenRouterReleaseError(
                        code="discord_oauth_state_invalid",
                        error_class="terminal",
                        message="Discord OAuth authorization state is invalid",
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
            except DiscordOAuthBrowserLaunchError:
                return self._handle_release_error(
                    ManagedOpenRouterReleaseError(
                        code="discord_browser_launch_failed",
                        error_class="terminal",
                        message="Discord authorization browser could not open",
                        operation="discord_callback",
                    ),
                    operation="discord_callback",
                )
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
            issue_request["delivery_ack_supported"] = True
            assured_operation = await run_owned_thread_call(
                partial(self._ensure_pending_managed_operation, bundle.installation_id)
            )
            if assured_operation is None:
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="discord_auth.error.retry",
                )
            operation_id, resume_token = assured_operation
            issue_request["operation_id"] = operation_id
            issue_request["resume_token"] = resume_token
            try:
                issue_response = await self.client.issue_discord_managed_key(issue_request)
            except ManagedOpenRouterReleaseError as exc:
                if (
                    exc.error_class == "retryable"
                    and exc.subcode
                    not in {"oauth_session_expired", "discord_oauth_session_expired"}
                    and not (
                        exc.failure_diagnostics is not None
                        and exc.failure_diagnostics.fields.get("dispatch_certainty")
                        == "not_dispatched"
                    )
                ):
                    return self._managed_operation_recovering_result(
                        operation="discord_issue",
                        code="discord_issue_transport_failure",
                    )
                await run_owned_thread_call(self._clear_pending_managed_operation)
                if exc.subcode in {"oauth_session_expired", "discord_oauth_session_expired"}:
                    clear_temporary_managed_release_state(self.managed_state)
                    await run_owned_thread_call(self.managed_state.persist)
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

    def _ensure_pending_managed_operation(self, installation_id: str) -> tuple[str, str] | None:
        pending_id = _normalize_optional_text(
            getattr(self.managed_state, "pending_managed_operation_id", None)
        )
        pending_source = _normalize_optional_text(
            getattr(self.managed_state, "pending_managed_operation_source", None)
        )
        pending_installation = _normalize_optional_text(
            getattr(self.managed_state, "pending_managed_operation_installation_id", None)
        )
        if (
            pending_id is not None
            and is_valid_operation_id(pending_id)
            and pending_source == MANAGED_OPERATION_SOURCE_DISCORD
            and pending_installation == installation_id
        ):
            try:
                stored_token = self.secrets.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
            except Exception:
                return None
            if is_valid_resume_token(stored_token):
                return (pending_id, stored_token)
        if pending_id is not None:
            return None
        operation_id = new_managed_operation_id()
        resume_token = new_managed_operation_resume_token()
        try:
            self.secrets.set(MANAGED_OPERATION_RESUME_TOKEN_SECRET, resume_token)
        except Exception:
            self._clear_pending_managed_operation()
            return None
        self.managed_state.pending_managed_operation_id = operation_id
        self.managed_state.pending_managed_operation_source = MANAGED_OPERATION_SOURCE_DISCORD
        self.managed_state.pending_managed_operation_installation_id = installation_id
        self.managed_state.pending_managed_operation_state = None
        try:
            self.managed_state.persist()
        except Exception:
            self._clear_pending_managed_operation()
            return None
        return (operation_id, resume_token)

    def _clear_pending_managed_operation(self) -> None:
        source = self._managed_source()
        if self.managed_state.pending_managed_operation_source not in {None, source}:
            return
        previous = self.managed_state.snapshot()
        try:
            self.managed_state.pending_managed_operation_id = None
            self.managed_state.pending_managed_operation_source = None
            self.managed_state.pending_managed_operation_installation_id = None
            self.managed_state.pending_managed_operation_state = None
            self.managed_state.persist()
            if self.managed_state.pending_delivery_ack_source == source:
                self.managed_state.pending_delivery_ack_source = None
                self.managed_state.pending_delivery_ack_delivery_id = None
                self.managed_state.pending_delivery_ack_managed_credential_ref = None
                self.managed_state.pending_delivery_ack_expires_at = None
                self.managed_state.persist()
        except Exception:
            self.managed_state.restore(previous)
            return
        try:
            self.secrets.delete(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
        except Exception:
            pass

    def _record_operation_state(self, operation_status: str | None) -> None:
        if operation_status not in {
            "AUTHENTICATED",
            "ISSUE_READY",
            "CREATING",
            "CREATE_UNKNOWN",
            "RECONCILING",
            "CLEANUP_REQUIRED",
            "CLEAN",
            "RETRY_READY",
            "DELIVERY_PENDING",
            "ACTIVE",
            "FAILED",
        }:
            return
        try:
            self.managed_state.pending_managed_operation_state = operation_status
            self.managed_state.persist()
        except Exception:
            pass

    async def _fetch_operation_status(
        self, operation_id: str, resume_token: str, installation_id: str
    ) -> ManagedOperationStatusResult | None:
        status_method = getattr(self.client, "get_managed_operation_status", None)
        if not callable(status_method):
            return None
        try:
            return await status_method(
                ManagedOperationStatusRequest(
                    operation_id=operation_id,
                    installation_id=installation_id,
                    resume_token=resume_token,
                    source=self._managed_source(),
                )
            )
        except Exception as exc:
            return ManagedOperationStatusResult(
                succeeded=False,
                operation_status="CREATE_UNKNOWN",
                client_action="wait",
                diagnostics=getattr(exc, "failure_diagnostics", None)
                or provider_failure_report(
                    exc, provider="openrouter", operation="provider_build"
                ).diagnostics,
            )

    async def _resume_managed_operation(
        self, operation_id: str, resume_token: str, installation_id: str
    ) -> ManagedOperationStatusResult | None:
        resume_method = getattr(self.client, "resume_managed_operation", None)
        if not callable(resume_method):
            return None
        try:
            return await resume_method(
                ManagedOperationResumeRequest(
                    operation_id=operation_id,
                    installation_id=installation_id,
                    resume_token=resume_token,
                    source=self._managed_source(),
                )
            )
        except Exception as exc:
            return ManagedOperationStatusResult(
                succeeded=False,
                operation_status="CREATE_UNKNOWN",
                client_action="wait",
                diagnostics=getattr(exc, "failure_diagnostics", None)
                or provider_failure_report(
                    exc, provider="openrouter", operation="provider_build"
                ).diagnostics,
            )

    async def _recover_pending_managed_operation(
        self,
    ) -> ManagedOpenRouterReleaseResult | None:
        state = self.managed_state
        operation_id = state.pending_managed_operation_id
        source = state.pending_managed_operation_source
        installation_id = state.pending_managed_operation_installation_id
        if not is_valid_operation_id(operation_id) or source != self._managed_source():
            return None
        message_prefix = "qq_managed_auth" if source == "qq" else "discord_auth"
        if not installation_id or installation_id != state.installation_id:
            try:
                await run_owned_thread_call(
                    partial(ensure_managed_identity_bundle, state, self.secrets)
                )
            except Exception:
                return self._managed_operation_action_required_result(
                    code="managed_identity_recovery_required",
                    message_key=f"{message_prefix}.error.action_required",
                    operation_status=None,
                )
            if installation_id != state.installation_id:
                return self._managed_operation_action_required_result(
                    code="managed_operation_installation_mismatch",
                    message_key=f"{message_prefix}.error.action_required",
                    operation_status=None,
                )
        try:
            token = await run_owned_thread_call(
                lambda: self.secrets.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
            )
        except Exception:
            return self._managed_operation_recovering_result(
                operation="secret_observation", code="managed_operation_token_read_failed"
            )
        if not is_valid_resume_token(token):
            return self._managed_operation_action_required_result(
                code="managed_operation_resume_token_missing",
                message_key=f"{message_prefix}.error.action_required",
                operation_status=None,
            )
        status = await self._fetch_operation_status(operation_id, token, installation_id)
        if status is None:
            return self._managed_operation_recovering_result(
                operation="managed_operation_status", code="managed_operation_status_unavailable"
            )
        if (
            status.succeeded
            and status.operation_status == "DELIVERY_PENDING"
            and status.client_action == "acknowledge_delivery"
            and state.pending_delivery_ack_source is None
        ):
            delivery = status.delivery
            ref = status.managed_credential_ref or (
                status.attempt.managed_credential_ref if status.attempt is not None else None
            )
            if (
                delivery is not None
                and delivery.status == "pending"
                and delivery.delivery_id
                and delivery.expires_at
                and not delivery_expired(delivery.expires_at)
                and ref
            ):
                try:
                    binding, credential, ack_token = await run_owned_thread_call(
                        lambda: (
                            ManagedCredentialBinding.decode(
                                self.secrets.get(credential_binding_secret(source))
                            ),
                            self.secrets.get(managed_key_secret(source)),
                            self.secrets.get(_delivery_ack_token_secret_key(source)),
                        )
                    )
                except Exception:
                    return self._delivery_ack_retry_result("managed_secret_read_failed")
                if (
                    ack_token
                    and binding is not None
                    and binding.matches(
                        credential=credential,
                        source=source,
                        installation_id=installation_id,
                        operation_id=operation_id,
                        delivery_id=delivery.delivery_id,
                        managed_credential_ref=ref,
                    )
                ):
                    snapshot = state.snapshot()
                    state.pending_delivery_ack_source = source
                    state.pending_delivery_ack_delivery_id = delivery.delivery_id
                    state.pending_delivery_ack_managed_credential_ref = ref
                    state.pending_delivery_ack_expires_at = delivery.expires_at
                    try:
                        await run_owned_thread_call(state.persist)
                    except Exception:
                        state.restore(snapshot)
                        return self._delivery_ack_retry_result("pending_metadata_persist_failed")
                    pending = await self._retry_pending_delivery_ack_if_needed()
                    if pending is not None:
                        return pending
                    return ManagedOpenRouterReleaseResult(
                        behavior=ManagedOpenRouterReleaseBehavior.READY,
                        api_key=credential,
                        local_key_available=True,
                    )
        if (
            status.client_action in {"retry_authorized", "acknowledge_delivery"}
            and status.succeeded
        ):
            converted = _operation_status_to_issue_success(status, source=source)
            if converted is not None:
                return await self._persist_managed_issue_success(converted)
            resumed = await self._resume_managed_operation(operation_id, token, installation_id)
            if resumed is not None:
                status = resumed
        if status.operation_status == "FAILED" or status.client_action == "action_required":
            if status.operation_status == "FAILED":
                await run_owned_thread_call(self._clear_pending_managed_operation)
            result = self._managed_operation_action_required_result(
                code=status.failed_reason or "managed_operation_action_required",
                message_key=f"{message_prefix}.error."
                + (
                    "authorization_expired"
                    if status.failed_reason == "authorization_expired"
                    else "action_required"
                ),
                operation_status=status.operation_status,
            )
            return replace(
                result,
                diagnostics=replace(result.diagnostics, failure_diagnostics=status.diagnostics),
            )
        if not status.succeeded:
            if status.operation_status == "UNKNOWN_OPERATION":
                await run_owned_thread_call(self._clear_pending_managed_operation)
                return None
            result = self._managed_operation_recovering_result(
                operation="managed_operation_status", code="managed_operation_status_failed"
            )
            return replace(
                result,
                diagnostics=replace(result.diagnostics, failure_diagnostics=status.diagnostics),
            )
        self._record_operation_state(status.operation_status)
        if status.operation_status == "ACTIVE":
            try:
                binding, credential = await run_owned_thread_call(
                    lambda: (
                        ManagedCredentialBinding.decode(
                            self.secrets.get(credential_binding_secret(source))
                        ),
                        self.secrets.get(managed_key_secret(source)),
                    )
                )
            except Exception:
                return self._delivery_ack_retry_result("managed_secret_read_failed")
            ref = status.managed_credential_ref or (
                status.attempt.managed_credential_ref if status.attempt is not None else None
            )
            if (
                not ref
                or binding is None
                or not binding.matches(
                    credential=credential,
                    source=source,
                    installation_id=installation_id,
                    operation_id=operation_id,
                    managed_credential_ref=ref,
                )
            ):
                return self._managed_operation_action_required_result(
                    code="managed_operation_active_key_unbound",
                    message_key=f"{message_prefix}.error.action_required",
                    operation_status=status.operation_status,
                )
            state.active_managed_credential_ref = binding.managed_credential_ref
            await run_owned_thread_call(state.persist)
            await run_owned_thread_call(self._clear_pending_managed_operation)
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.READY,
                api_key=credential,
                local_key_available=True,
            )
        converted = _operation_status_to_issue_success(status, source=source)
        if converted is not None:
            return await self._persist_managed_issue_success(converted)
        return self._managed_operation_recovering_result(
            operation="managed_operation_status", code="managed_operation_recovery_pending"
        )

    def _managed_operation_recovering_result(
        self, *, operation: str, code: str
    ) -> ManagedOpenRouterReleaseResult:
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="discord_auth.error.recovery_pending",
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation=operation,
                code=code,
                error_class="retryable",
            ),
        )

    def _managed_operation_action_required_result(
        self, *, code: str, message_key: str, operation_status: str | None
    ) -> ManagedOpenRouterReleaseResult:
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.STOP,
            message_key=message_key,
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="managed_operation_recovery",
                code=code,
                error_class="terminal",
            ),
        )

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
        bundle = await run_owned_thread_call(
            partial(ensure_managed_identity_bundle, self.managed_state, self.secrets)
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
            if exc.code == "managed_delivery_protocol_mismatch":
                clear_temporary_managed_release_state(self.managed_state)
                await run_owned_thread_call(self.managed_state.persist)
                self._clear_retry_after()
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.STOP,
                    message_key="discord_auth.error.action_required",
                    diagnostics=exc.to_diagnostics(),
                )
            return self._handle_release_error(exc, operation="issue")

        return await self._persist_managed_issue_success(issue_response)

    async def _persist_legacy_issue_success(
        self, issue_response: ManagedOpenRouterIssueSuccess
    ) -> ManagedOpenRouterReleaseResult:
        state = self.managed_state
        source = self._managed_source()
        snapshot = state.snapshot()
        keys = (
            managed_key_secret(source),
            credential_binding_secret(source),
            "openrouter_managed_user_id",
            "openrouter_managed_user_installation_id",
        )

        def store_legacy_delivery() -> None:
            previous = {key: self.secrets.get(key) for key in keys}
            try:
                self.secrets.set(managed_key_secret(source), issue_response.openrouter_api_key)
                best_effort_store_managed_openrouter_user_identifier(
                    self._credential_runtime_config(),
                    secrets=self.secrets,
                    openrouter_user_id=issue_response.openrouter_user_id,
                )
                store_managed_entitlement_snapshot(
                    state,
                    managed_credential_ref=issue_response.managed_credential_ref,
                    expires_at=issue_response.expires_at,
                )
                clear_temporary_managed_release_state(state)
                self.secrets.delete(credential_binding_secret(source))
                state.persist()
            except Exception:
                try:
                    for key, value in previous.items():
                        if value is None:
                            self.secrets.delete(key)
                        else:
                            self.secrets.set(key, value)
                finally:
                    state.restore(snapshot)
                    clear_temporary_managed_release_state(state)
                    try:
                        state.persist()
                    except Exception:
                        state.restore(snapshot)
                raise

        try:
            await run_owned_thread_call(store_legacy_delivery)
        except Exception as exc:
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="managed_release.stop",
                diagnostics=ManagedOpenRouterReleaseDiagnostics(
                    operation="issue",
                    code="managed_legacy_delivery_store_failed",
                    error_class="terminal",
                    failure_diagnostics=provider_failure_report(
                        exc, provider="openrouter", operation="provider_build"
                    ).diagnostics,
                ),
            )
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            api_key=issue_response.openrouter_api_key,
            local_key_available=True,
        )

    async def _persist_managed_issue_success(
        self,
        issue_response: ManagedOpenRouterIssueSuccess,
    ) -> ManagedOpenRouterReleaseResult:
        if not issue_response.delivery_ack_required:
            if self.managed_state.pending_managed_operation_id is not None:
                return self._managed_operation_action_required_result(
                    code="managed_delivery_contract_missing",
                    message_key="discord_auth.error.action_required",
                    operation_status=None,
                )
            return await self._persist_legacy_issue_success(issue_response)
        source = self._managed_source()
        state = self.managed_state
        binding = None
        if issue_response.delivery_ack_required:
            try:
                if (
                    state.pending_managed_operation_source != source
                    or not is_valid_operation_id(state.pending_managed_operation_id)
                    or not issue_response.delivery_ack_token
                ):
                    raise ValueError("managed delivery has no operation authority")
                binding = ManagedCredentialBinding.create(
                    source=source,
                    installation_id=state.installation_id,
                    operation_id=state.pending_managed_operation_id,
                    delivery_id=issue_response.delivery_id or "",
                    managed_credential_ref=issue_response.managed_credential_ref or "",
                    credential=issue_response.openrouter_api_key,
                )
            except ValueError:
                clear_temporary_managed_release_state(state)
                await run_owned_thread_call(state.persist)
                return self._managed_operation_action_required_result(
                    code="managed_delivery_protocol_mismatch",
                    message_key="discord_auth.error.action_required",
                    operation_status=None,
                )

        snapshot = state.snapshot()

        def store_delivery() -> None:
            self.secrets.set(managed_key_secret(source), issue_response.openrouter_api_key)
            if binding is not None:
                observed = self.secrets.get(managed_key_secret(source))
                if observed != issue_response.openrouter_api_key:
                    raise ValueError("managed credential persistence is unconfirmed")
                self.secrets.set(credential_binding_secret(source), binding.encode())
                self.secrets.set(
                    _delivery_ack_token_secret_key(source), issue_response.delivery_ack_token or ""
                )
                state.pending_delivery_ack_source = source
                state.pending_delivery_ack_delivery_id = issue_response.delivery_id
                state.pending_delivery_ack_managed_credential_ref = (
                    issue_response.managed_credential_ref
                )
                state.pending_delivery_ack_expires_at = issue_response.delivery_ack_expires_at
            if source == "discord":
                best_effort_store_managed_openrouter_user_identifier(
                    self._credential_runtime_config(),
                    secrets=self.secrets,
                    openrouter_user_id=issue_response.openrouter_user_id,
                )
            store_managed_entitlement_snapshot(
                state,
                managed_credential_ref=issue_response.managed_credential_ref,
                expires_at=issue_response.expires_at,
            )
            clear_temporary_managed_release_state(state)
            state.persist()

        try:
            await run_owned_thread_call(store_delivery)
        except Exception:
            state.restore(snapshot)
            return self._delivery_ack_retry_result("managed_delivery_store_failed")
        if issue_response.delivery_ack_required:
            pending = await self._retry_pending_delivery_ack_if_needed()
            if pending is not None:
                return pending
        else:
            await run_owned_thread_call(self._clear_pending_managed_operation)
        self._clear_retry_after()
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            api_key=issue_response.openrouter_api_key,
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
        if error.subcode in {"oauth_session_expired", "discord_oauth_session_expired"}:
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RESTART,
                message_key="discord_auth.error.expired",
                diagnostics=diagnostics,
            )
        if error.code == "discord_browser_launch_failed" or (
            error.failure_diagnostics is not None
            and error.failure_diagnostics.fields.get("configuration_error")
            in {
                "unsupported_proxy_configuration",
                "invalid_proxy_configuration",
                "invalid_tls_configuration",
            }
        ):
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key=(
                    "discord_auth.error.browser_launch_failed"
                    if error.code == "discord_browser_launch_failed"
                    else "discord_auth.error.network_configuration"
                ),
                diagnostics=diagnostics,
            )
        if error.error_class == "security_fail" and error.subcode in BINDING_MISMATCH_SUBCODES:
            self._clear_retry_after()
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.STOP,
                message_key="discord_auth.error.action_required",
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
        self._prepare_task = None
        self._issue_task = None

        failures: list[BaseException] = []
        try:
            await self.oauth_runtime.close()
        except Exception as exc:
            failures.append(exc)

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
    failure_diagnostics: ErrorDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.failure_diagnostics is None and self.diagnostics is not None:
            self.failure_diagnostics = self.diagnostics.failure_diagnostics

    def __str__(self) -> str:
        diagnostics = self.failure_diagnostics
        if diagnostics is None:
            return "ManagedOpenRouterUserFacingError cause=unknown"
        return (
            f"ManagedOpenRouterUserFacingError code={diagnostics.code or 'unknown'} "
            f"category={diagnostics.category} status={diagnostics.status_code}"
        )


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
                    failure_diagnostics=provider_failure_report(
                        exc, provider="openrouter", operation="provider_build"
                    ).diagnostics,
                ) from exc
            if not isinstance(result, ManagedOpenRouterReleaseResult):
                raise ManagedOpenRouterUserFacingError(
                    message_key="managed_release.retry",
                    failure_diagnostics=replace(
                        provider_failure_report(
                            None, provider="openrouter", operation="provider_build"
                        ).diagnostics,
                        category="invalid_response",
                        code="provider.invalid_response",
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
                    failure_diagnostics=provider_failure_report(
                        exc, provider="openrouter", operation="provider_build"
                    ).diagnostics,
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
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        delegate = await self._ensure_delegate()
        kwargs = {
            "utterance_id": utterance_id,
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return await delegate.translate(**kwargs)  # type: ignore[arg-type]

    async def close(self) -> None:
        if self._delegate is not None:
            await self._delegate.close()
            self._delegate = None


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _operation_status_to_issue_success(
    status: ManagedOperationStatusResult,
    *,
    source: str,
) -> ManagedOpenRouterIssueSuccess | None:
    if status.delivery_ack is not None and (
        status.delivery_ack.source != source
        or status.delivery_ack.managed_credential_ref != status.managed_credential_ref
    ):
        return None
    if not status.succeeded or not status.managed_secret_key:
        return None
    delivery_ack = status.delivery_ack
    return ManagedOpenRouterIssueSuccess(
        openrouter_api_key=status.managed_secret_key,
        managed_credential_ref=status.managed_credential_ref,
        expires_at=status.expires_at,
        openrouter_user_id=status.openrouter_user_id,
        delivery_ack_required=delivery_ack is not None,
        delivery_id=delivery_ack.delivery_id if delivery_ack is not None else None,
        delivery_ack_token=delivery_ack.delivery_ack_token if delivery_ack is not None else None,
        delivery_ack_expires_at=delivery_ack.expires_at if delivery_ack is not None else None,
    )


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
        failure_diagnostics=provider_failure_report(
            error, provider="openrouter", operation="provider_build"
        ).diagnostics,
    )


def _discord_callback_release_error(error: Exception) -> ManagedOpenRouterReleaseError:
    if isinstance(error, DiscordOAuthCallbackError):
        return ManagedOpenRouterReleaseError(
            code="discord_oauth_callback_error",
            error_class="retryable",
            message=f"Discord OAuth callback failed: {error.error}",
            operation="discord_callback",
            subcode=error.error,
            failure_diagnostics=provider_failure_report(
                error, provider="openrouter", operation="provider_build"
            ).diagnostics,
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
            failure_diagnostics=provider_failure_report(
                error, provider="openrouter", operation="provider_build"
            ).diagnostics,
        )
    return ManagedOpenRouterReleaseError(
        code="discord_oauth_callback_closed",
        error_class="retryable",
        message=_exception_message(
            error,
            default="Discord OAuth callback listener closed before completion",
        ),
        operation="discord_callback",
        failure_diagnostics=provider_failure_report(
            error, provider="openrouter", operation="provider_build"
        ).diagnostics,
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
