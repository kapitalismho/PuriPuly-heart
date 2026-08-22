from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import sys
import threading
import types
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from puripuly_heart.core.discord_oauth_loopback import DiscordOAuthCallbackError
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterLLMProvider,
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterReleaseError,
    ManagedOpenRouterReleaseResult,
    ManagedOpenRouterReleaseService,
    ManagedOpenRouterStatusRefreshResult,
    ManagedOpenRouterTrialStatusSuccess,
    ManagedOpenRouterUserFacingError,
    TalkTogetherPassStatus,
    UnavailableManagedOpenRouterReleaseClient,
    format_managed_openrouter_diagnostics,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    load_managed_openrouter_user_identifier,
)

from puripuly_heart.app.ports.broker_client import ManagedKeyDeliveryAckResult
from puripuly_heart.app.wiring import (
    ManagedIdentityStateAdapter,
    build_openrouter_credential_runtime_config,
    build_openrouter_release_runtime_config,
)
from puripuly_heart.config.settings import (
    AppSettings,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    TranslationConnection,
    TranslationModel,
)
from puripuly_heart.core.managed_identity import ensure_managed_identity_bundle
from puripuly_heart.core.runtime import OAuthRuntime
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.domain.models import Translation


@dataclass
class FakeManagedReleaseClient:
    challenge_result: object | None = None
    verify_result: object | None = None
    issue_result: object | None = None
    discord_start_result: object | None = None
    discord_issue_result: object | None = None
    delivery_ack_result: object | None = None
    trial_status_result: object | None = None
    challenge_gate: asyncio.Event | None = None
    discord_start_gate: asyncio.Event | None = None
    issue_gate: asyncio.Event | None = None
    issue_started: asyncio.Event | None = None
    discord_issue_gate: asyncio.Event | None = None
    discord_issue_started: asyncio.Event | None = None
    trial_status_gate: asyncio.Event | None = None
    trial_status_started: asyncio.Event | None = None
    calls: list[tuple[str, dict[str, object]]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    async def challenge(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        app_version: str,
    ):
        self.calls.append(
            (
                "challenge",
                {
                    "installation_id": installation_id,
                    "device_public_key": device_public_key,
                    "app_version": app_version,
                },
            )
        )
        if self.challenge_gate is not None:
            await self.challenge_gate.wait()
        result = self.challenge_result
        if isinstance(result, Exception):
            raise result
        return result

    async def verify(self, request: dict[str, str]):
        self.calls.append(("verify", dict(request)))
        result = self.verify_result
        if isinstance(result, Exception):
            raise result
        return result

    async def issue(self, request: dict[str, object]):
        self.calls.append(("issue", dict(request)))
        if self.issue_started is not None:
            self.issue_started.set()
        if self.issue_gate is not None:
            await self.issue_gate.wait()
        result = self.issue_result
        if isinstance(result, Exception):
            raise result
        return result

    async def start_discord_oauth(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        redirect_uri: str,
        app_version: str,
        referral_id: str | None = None,
    ):
        payload = {
            "installation_id": installation_id,
            "device_public_key": device_public_key,
            "redirect_uri": redirect_uri,
            "app_version": app_version,
        }
        if referral_id is not None:
            payload["referral_id"] = referral_id
        self.calls.append(("discord_start", payload))
        if self.discord_start_gate is not None:
            await self.discord_start_gate.wait()
        result = self.discord_start_result
        if isinstance(result, Exception):
            raise result
        return result

    async def issue_discord_managed_key(self, request: dict[str, object]):
        self.calls.append(("discord_issue", dict(request)))
        if self.discord_issue_started is not None:
            self.discord_issue_started.set()
        if self.discord_issue_gate is not None:
            await self.discord_issue_gate.wait()
        result = self.discord_issue_result
        if isinstance(result, Exception):
            raise result
        return result

    async def get_trial_status(
        self,
        *,
        installation_id: str,
        timestamp: str,
        signature: str,
    ):
        self.calls.append(
            (
                "trial_status",
                {
                    "installation_id": installation_id,
                    "timestamp": timestamp,
                    "signature": signature,
                },
            )
        )
        if self.trial_status_started is not None:
            self.trial_status_started.set()
        if self.trial_status_gate is not None:
            await self.trial_status_gate.wait()
        result = self.trial_status_result
        if isinstance(result, Exception):
            raise result
        return result

    async def acknowledge_managed_key_delivery(self, request: object):
        self.calls.append(("delivery_ack", {"request": request}))
        result = self.delivery_ack_result
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class ClosableFakeManagedReleaseClient(FakeManagedReleaseClient):
    close_calls: int = 0

    async def close(self) -> None:
        self.close_calls += 1


@dataclass
class FailingCloseManagedReleaseClient(ClosableFakeManagedReleaseClient):
    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("client close failed")


class FailingManagedKeySecretStore(InMemorySecretStore):
    def __init__(self, *, fail_on_key: str = OPENROUTER_MANAGED_API_KEY_SECRET) -> None:
        super().__init__()
        self.fail_on_key = fail_on_key
        self.set_attempts: list[tuple[str, str]] = []

    def set(self, key: str, value: str) -> None:
        self.set_attempts.append((key, value))
        super().set(key, value)
        if key == self.fail_on_key:
            raise RuntimeError("managed key persistence failed")


class FailingDeleteSecretStore(InMemorySecretStore):
    def __init__(self, *, fail_on_key: str) -> None:
        super().__init__()
        self.fail_on_key = fail_on_key

    def delete(self, key: str) -> None:
        if key == self.fail_on_key:
            raise RuntimeError("secret delete failed")
        super().delete(key)


class FailingAckTokenSetAndManagedKeyDeleteSecretStore(InMemorySecretStore):
    fail_managed_key_delete: bool = False

    def set(self, key: str, value: str) -> None:
        if key == "openrouter_managed_delivery_ack_token":
            raise RuntimeError("ack token set failed")
        super().set(key, value)

    def delete(self, key: str) -> None:
        if self.fail_managed_key_delete and key == OPENROUTER_MANAGED_API_KEY_SECRET:
            raise RuntimeError("managed key delete failed")
        super().delete(key)


def _make_service(
    *,
    client: FakeManagedReleaseClient,
    settings: AppSettings | None = None,
    secrets: InMemorySecretStore | None = None,
    persist_calls: list[tuple[str | None, str | None]] | None = None,
    raw_hardware_fingerprint_provider: Any | None = None,
    discord_oauth_listener_factory: Any | None = None,
    discord_oauth_callback_runner: Any | None = None,
    on_discord_callback_received: Any | None = None,
    oauth_runtime: OAuthRuntime | None = None,
) -> tuple[ManagedOpenRouterReleaseService, AppSettings, InMemorySecretStore]:
    resolved_settings = settings or AppSettings()
    resolved_settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    resolved_secrets = secrets or InMemorySecretStore()
    tracked_persist_calls = persist_calls if persist_calls is not None else []

    def persist(updated: AppSettings) -> None:
        tracked_persist_calls.append(
            (
                updated.managed_identity.installation_id,
                updated.managed_identity.release_token,
            )
        )

    service_kwargs: dict[str, Any] = {
        "openrouter_config": build_openrouter_release_runtime_config(resolved_settings),
        "managed_state": ManagedIdentityStateAdapter(resolved_settings, persist),
        "secrets": resolved_secrets,
        "client": client,
        "app_version": "2.0.0",
        "raw_hardware_fingerprint_provider": (
            raw_hardware_fingerprint_provider
            if raw_hardware_fingerprint_provider is not None
            else (lambda: "raw-hardware-fingerprint-test")
        ),
        "signed_at_provider": lambda: "2026-04-08T06:00:45.000Z",
        "monotonic_ms_provider": lambda: 1_000,
    }
    if discord_oauth_listener_factory is not None:
        service_kwargs["discord_oauth_listener_factory"] = discord_oauth_listener_factory
    if discord_oauth_callback_runner is not None:
        service_kwargs["discord_oauth_callback_runner"] = discord_oauth_callback_runner
    if on_discord_callback_received is not None:
        service_kwargs["on_discord_callback_received"] = on_discord_callback_received
    if oauth_runtime is not None:
        service_kwargs["oauth_runtime"] = oauth_runtime
    service = ManagedOpenRouterReleaseService(**service_kwargs)
    return service, resolved_settings, resolved_secrets


def _make_fingerprint_salt() -> ManagedOpenRouterFingerprintSalt:
    return ManagedOpenRouterFingerprintSalt(version=7, salt="fingerprint-salt-test")


def test_format_managed_openrouter_diagnostics_redacts_raw_message() -> None:
    diagnostics = ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode="broker_backoff",
        retry_after_ms=9000,
        message="raw broker eligibility message token=broker-secret-123",
    )

    rendered = format_managed_openrouter_diagnostics(diagnostics)

    assert (
        rendered == "operation=issue code=trial_unavailable class=retryable "
        "subcode=broker_backoff retry_after_ms=9000 message=<redacted>"
    )
    assert "raw broker eligibility message" not in rendered
    assert "broker-secret-123" not in rendered
    assert "token=" not in rendered


def test_user_facing_error_str_is_diagnostic_safe_without_ui_i18n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i18n_calls: list[tuple[str, dict[str, object]]] = []
    fake_i18n = types.ModuleType("puripuly_heart.ui.i18n")

    def fake_t(key: str, **params: object) -> str:
        i18n_calls.append((key, dict(params)))
        return "localized broker raw payload token=provider-secret-123"

    fake_i18n.t = fake_t  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.i18n", fake_i18n)
    error = ManagedOpenRouterUserFacingError(
        message_key="managed_release.retry_after_ms",
        message_kwargs={"retry_after_ms": 9000, "detail": "provider-secret-123"},
        diagnostics=ManagedOpenRouterReleaseDiagnostics(
            operation="issue",
            code="trial_unavailable",
            error_class="retryable",
            subcode="broker_backoff",
            retry_after_ms=9000,
            message="broker raw payload token=provider-secret-123",
        ),
    )

    rendered = str(error)

    assert i18n_calls == []
    assert "managed_release.retry_after_ms" in rendered
    assert "retry_after_ms=9000" in rendered
    assert "operation=issue" in rendered
    assert "trial_unavailable" in rendered
    assert "broker raw payload" not in rendered
    assert "provider-secret-123" not in rendered
    assert "localized" not in rendered


@dataclass
class FakeDiscordOAuthListener:
    redirect_uri: str = "http://127.0.0.1:62187/discord/callback"
    closed: bool = False
    close_calls: int = 0

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


@dataclass
class FakeDiscordOAuthHarness:
    redirect_uri: str = "http://127.0.0.1:62187/discord/callback"
    callback_result: tuple[str, str] = ("discord-code-1", "discord-state-1")
    callback_error: BaseException | None = None

    def __post_init__(self) -> None:
        self.listeners: list[FakeDiscordOAuthListener] = []
        self.callback_calls: list[tuple[FakeDiscordOAuthListener, str, str]] = []

    def bind_listener(self) -> FakeDiscordOAuthListener:
        listener = FakeDiscordOAuthListener(redirect_uri=self.redirect_uri)
        self.listeners.append(listener)
        return listener

    async def run_callback_flow(
        self,
        listener: FakeDiscordOAuthListener,
        authorization_url: str,
        expires_at: str,
    ) -> tuple[str, str]:
        self.callback_calls.append((listener, authorization_url, expires_at))
        if self.callback_error is not None:
            raise self.callback_error
        return self.callback_result


def _make_discord_start_success(
    *,
    redirect_uri: str = "http://127.0.0.1:62187/discord/callback",
    authorization_url: str = "https://discord.com/oauth2/authorize?client_id=client-1",
    expires_at: str = "2026-04-08T06:05:00.000Z",
    issue_nonce: str = "issue-nonce-1",
) -> ManagedOpenRouterDiscordStartSuccess:
    return ManagedOpenRouterDiscordStartSuccess(
        authorization_url=authorization_url,
        redirect_uri=redirect_uri,
        oauth_session_expires_at=expires_at,
        issue_nonce=issue_nonce,
        fingerprint_salt=_make_fingerprint_salt(),
        fingerprint_salt_version=7,
    )


def _make_discord_service(
    *,
    client: FakeManagedReleaseClient | None = None,
    settings: AppSettings | None = None,
    secrets: InMemorySecretStore | None = None,
    harness: FakeDiscordOAuthHarness | None = None,
    persist_calls: list[tuple[str | None, str | None]] | None = None,
    raw_hardware_fingerprint_provider: Any | None = None,
    on_discord_callback_received: Any | None = None,
    oauth_runtime: OAuthRuntime | None = None,
) -> tuple[
    ManagedOpenRouterReleaseService,
    AppSettings,
    InMemorySecretStore,
    FakeManagedReleaseClient,
    FakeDiscordOAuthHarness,
]:
    resolved_harness = harness or FakeDiscordOAuthHarness()
    resolved_client = client or FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(
            redirect_uri=resolved_harness.redirect_uri,
        ),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    service, resolved_settings, resolved_secrets = _make_service(
        client=resolved_client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
        raw_hardware_fingerprint_provider=raw_hardware_fingerprint_provider,
        discord_oauth_listener_factory=resolved_harness.bind_listener,
        discord_oauth_callback_runner=resolved_harness.run_callback_flow,
        on_discord_callback_received=on_discord_callback_received,
        oauth_runtime=oauth_runtime,
    )
    return service, resolved_settings, resolved_secrets, resolved_client, resolved_harness


class RecordingOAuthRuntime(OAuthRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started_task_names: list[str] = []
        self.attached_listener_names: list[str] = []

    def create_auth_task(self, coro, *, task_name: str):  # type: ignore[no-untyped-def]
        self.started_task_names.append(task_name)
        return super().create_auth_task(coro, task_name=task_name)

    def attach_loopback_listener(self, listener, *, listener_name: str):  # type: ignore[no-untyped-def]
        self.attached_listener_names.append(listener_name)
        super().attach_loopback_listener(listener, listener_name=listener_name)


class FailingCloseOAuthRuntime(OAuthRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("oauth close failed")


def _expected_hardware_hash(*, fingerprint_salt: str, raw_hardware_fingerprint: str) -> str:
    digest = hashlib.sha256(
        f"{fingerprint_salt}{raw_hardware_fingerprint}".encode("utf-8")
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _set_verified_snapshot(
    settings: AppSettings,
    *,
    hardware_hash: str = "verified-hardware-hash-1",
    salt_version: int = 7,
) -> None:
    settings.managed_identity.verified_hardware_hash = hardware_hash
    settings.managed_identity.verified_hardware_hash_salt_version = salt_version


@pytest.mark.asyncio
async def test_prepare_for_translation_short_circuits_when_managed_key_exists() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")
    client = FakeManagedReleaseClient()
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert result.local_key_available is True
    assert result.pending_issue is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_managed_china_prepare_uses_only_qq_key_with_active_entitlement_state() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    settings.managed_identity.active_managed_credential_ref = "managed-ref-qq"
    settings.managed_identity.active_managed_expires_at = "2026-08-03T06:00:00.000Z"
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")
    client = FakeManagedReleaseClient()
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "qq-managed-key"
    assert result.local_key_available is True
    assert client.calls == []


@pytest.mark.asyncio
async def test_managed_china_prepare_survives_missing_active_entitlement_state_with_local_qq_key() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")
    client = FakeManagedReleaseClient()
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "qq-managed-key"
    assert result.local_key_available is True
    assert client.calls == []


@pytest.mark.asyncio
async def test_managed_china_prepare_does_not_start_standard_discord_challenge_when_qq_key_missing() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    settings.managed_identity.active_managed_credential_ref = "managed-ref-qq"
    secrets = InMemorySecretStore()
    client = FakeManagedReleaseClient()
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert result.message_key == "qq_managed_auth.required"
    assert client.calls == []


@pytest.mark.asyncio
async def test_managed_china_llm_start_requires_qq_auth_without_standard_fallback_or_issue() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    settings.managed_identity.active_managed_credential_ref = "managed-ref-qq"
    secrets = InMemorySecretStore()
    client = FakeManagedReleaseClient(issue_result=ManagedOpenRouterIssueSuccess("issued-key"))
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert result.message_key == "qq_managed_auth.required"
    assert result.api_key is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_llm_start_retries_pending_delivery_ack_before_ready() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    settings.managed_identity.active_managed_credential_ref = "managed-ref-discord"
    settings.managed_identity.pending_delivery_ack_source = "discord"
    settings.managed_identity.pending_delivery_ack_delivery_id = "delivery-discord"
    settings.managed_identity.pending_delivery_ack_managed_credential_ref = "managed-ref-discord"
    settings.managed_identity.pending_delivery_ack_expires_at = "2026-07-07T00:15:00.000Z"
    persist_calls: list[tuple[str | None, str | None]] = []
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "standard-managed-key")
    secrets.set("openrouter_managed_delivery_ack_token", "delivery-ack-token")
    client = FakeManagedReleaseClient(
        delivery_ack_result=ManagedKeyDeliveryAckResult(
            succeeded=True,
            status="acknowledged",
        )
    )
    service, _, _ = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert client.calls[0][0] == "delivery_ack"
    assert settings.managed_identity.pending_delivery_ack_source is None
    assert secrets.get("openrouter_managed_delivery_ack_token") is None
    assert persist_calls


@pytest.mark.asyncio
async def test_llm_start_with_pending_delivery_ack_failure_does_not_return_ready() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    settings.managed_identity.active_managed_credential_ref = "managed-ref-discord"
    settings.managed_identity.pending_delivery_ack_source = "discord"
    settings.managed_identity.pending_delivery_ack_delivery_id = "delivery-discord"
    settings.managed_identity.pending_delivery_ack_managed_credential_ref = "managed-ref-discord"
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "standard-managed-key")
    secrets.set("openrouter_managed_delivery_ack_token", "delivery-ack-token")
    client = FakeManagedReleaseClient(
        delivery_ack_result=ManagedKeyDeliveryAckResult(
            succeeded=False,
            status="retryable",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert client.calls[0][0] == "delivery_ack"
    assert settings.managed_identity.pending_delivery_ack_source == "discord"
    assert secrets.get("openrouter_managed_delivery_ack_token") == "delivery-ack-token"


@pytest.mark.asyncio
async def test_standard_managed_prepare_preserves_discord_flow_without_reading_qq_key() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    secrets = InMemorySecretStore()
    service, _settings, _secrets, client, harness = _make_discord_service(
        settings=settings,
        secrets=secrets,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert client.calls[0][0] == "discord_start"
    assert client.calls[-1][0] == "discord_issue"
    assert harness.listeners[0].closed is True
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    assert secrets.get(OPENROUTER_MANAGED_QQ_API_KEY_SECRET) is None


@pytest.mark.asyncio
async def test_prepare_for_translation_uses_oauth_runtime_for_shared_auth_task() -> None:
    oauth_runtime = RecordingOAuthRuntime()
    service, _settings, _secrets, _client, _harness = _make_discord_service(
        oauth_runtime=oauth_runtime,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert oauth_runtime.started_task_names == ["managed-openrouter-prepare"]
    assert oauth_runtime.attached_listener_names == ["discord-loopback"]


@pytest.mark.asyncio
async def test_discord_oauth_short_circuits_local_key_without_listener_or_broker() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")
    client = FakeManagedReleaseClient()
    bind_calls: list[str] = []
    callback_calls: list[str] = []

    def bind_listener() -> FakeDiscordOAuthListener:
        bind_calls.append("bind")
        raise AssertionError("listener should not be bound when a managed key exists")

    async def run_callback_flow(
        _listener: FakeDiscordOAuthListener,
        _authorization_url: str,
        _expires_at: str,
    ) -> tuple[str, str]:
        callback_calls.append("callback")
        raise AssertionError("browser/callback flow should not start when a managed key exists")

    service, _, _ = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        discord_oauth_listener_factory=bind_listener,
        discord_oauth_callback_runner=run_callback_flow,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert result.local_key_available is True
    assert client.calls == []
    assert bind_calls == []
    assert callback_calls == []


@pytest.mark.asyncio
async def test_discord_oauth_flow_persists_managed_key() -> None:
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, secrets, client, harness = _make_discord_service(
        persist_calls=persist_calls,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert result.local_key_available is True
    assert result.pending_issue is False
    assert [name for name, _payload in client.calls] == ["discord_start", "discord_issue"]
    start_payload = client.calls[0][1]
    assert start_payload["redirect_uri"] == harness.listeners[0].redirect_uri
    assert start_payload["app_version"] == "2.0.0"
    assert harness.callback_calls == [
        (
            harness.listeners[0],
            "https://discord.com/oauth2/authorize?client_id=client-1",
            "2026-04-08T06:05:00.000Z",
        )
    ]
    issue_payload = client.calls[1][1]
    assert issue_payload["code"] == "discord-code-1"
    assert issue_payload["state"] == "discord-state-1"
    assert issue_payload["redirect_uri"] == harness.listeners[0].redirect_uri
    assert issue_payload["issue_nonce"] == "issue-nonce-1"
    assert issue_payload["app_version"] == "2.0.0"
    assert issue_payload["reason"] == "llm_start"
    assert issue_payload["budget_usd"] == 0.07
    assert issue_payload["hardware_hash"] == _expected_hardware_hash(
        fingerprint_salt="fingerprint-salt-test",
        raw_hardware_fingerprint="raw-hardware-fingerprint-test",
    )
    assert issue_payload["hardware_hash_salt_version"] == 7
    assert issue_payload["signed_at"] == "2026-04-08T06:00:45.000Z"
    assert settings.managed_identity.installation_id
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    assert harness.listeners[0].closed is True
    assert harness.listeners[0].close_calls == 1
    assert len(persist_calls) >= 2


@pytest.mark.asyncio
async def test_discord_callback_received_hook_runs_after_callback_before_issue() -> None:
    observed_calls_at_hook: list[list[str]] = []
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )

    def on_callback_received() -> None:
        observed_calls_at_hook.append([name for name, _payload in client.calls])

    service, _settings, _secrets, client, _harness = _make_discord_service(
        client=client,
        on_discord_callback_received=on_callback_received,
    )

    await service.prepare_for_translation()

    assert observed_calls_at_hook == [["discord_start"]]
    assert [name for name, _payload in client.calls] == ["discord_start", "discord_issue"]


@pytest.mark.asyncio
async def test_discord_listener_bind_failure_maps_to_retry_result_without_broker_call() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    callback_calls: list[str] = []

    def bind_listener() -> FakeDiscordOAuthListener:
        raise OSError("no Discord OAuth loopback port is available")

    async def run_callback_flow(
        _listener: FakeDiscordOAuthListener,
        _authorization_url: str,
        _expires_at: str,
    ) -> tuple[str, str]:
        callback_calls.append("callback")
        raise AssertionError("callback flow should not start when listener binding fails")

    service, _settings, _secrets = _make_service(
        client=client,
        discord_oauth_listener_factory=bind_listener,
        discord_oauth_callback_runner=run_callback_flow,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert result.message_key == "managed_release.retry"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="discord_start",
        code="discord_loopback_unavailable",
        error_class="retryable",
        message="Discord OAuth loopback listener unavailable: no Discord OAuth loopback port is available",
    )
    assert client.calls == []
    assert callback_calls == []


@pytest.mark.asyncio
async def test_discord_redirect_mismatch_closes_listener_and_maps_terminal_result() -> None:
    harness = FakeDiscordOAuthHarness()
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(
            redirect_uri="http://127.0.0.1:62188/discord/callback",
        ),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    service, _settings, _secrets, client, harness = _make_discord_service(
        client=client,
        harness=harness,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.stop"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="discord_start",
        code="discord_redirect_mismatch",
        error_class="terminal",
        message="Discord OAuth broker returned a different redirect URI",
    )
    assert harness.callback_calls == []
    assert harness.listeners[0].closed is True
    assert [name for name, _payload in client.calls] == ["discord_start"]


@pytest.mark.asyncio
async def test_discord_callback_error_maps_to_retry_result_and_closes_listener_without_issue() -> (
    None
):
    harness = FakeDiscordOAuthHarness(
        callback_error=DiscordOAuthCallbackError("access_denied", "discord-state-1"),
    )
    service, _settings, _secrets, client, harness = _make_discord_service(harness=harness)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert result.message_key == "managed_release.retry"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="discord_callback",
        code="discord_oauth_callback_error",
        error_class="retryable",
        subcode="access_denied",
        message="Discord OAuth callback failed: access_denied",
    )
    assert harness.listeners[0].closed is True
    assert [name for name, _payload in client.calls] == ["discord_start"]


@pytest.mark.asyncio
async def test_discord_callback_timeout_maps_to_retry_result_and_closes_listener_without_issue() -> (
    None
):
    harness = FakeDiscordOAuthHarness(
        callback_error=TimeoutError("timed out waiting for Discord OAuth callback"),
    )
    service, _settings, _secrets, client, harness = _make_discord_service(harness=harness)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert result.message_key == "managed_release.retry"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="discord_callback",
        code="discord_oauth_timeout",
        error_class="retryable",
        message="timed out waiting for Discord OAuth callback",
    )
    assert harness.listeners[0].closed is True
    assert [name for name, _payload in client.calls] == ["discord_start"]


@pytest.mark.asyncio
async def test_prepare_for_translation_persists_managed_entitlement_snapshot() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="hash_abc123",
            expires_at="2026-10-17T12:34:56Z",
        ),
    )
    service, settings, _secrets, _client, _harness = _make_discord_service(client=client)

    await service.prepare_for_translation()

    assert settings.managed_identity.active_managed_credential_ref == "hash_abc123"
    assert settings.managed_identity.active_managed_expires_at == "2026-10-17T12:34:56Z"


@pytest.mark.asyncio
async def test_prepare_for_translation_passes_friend_referral_id_without_persisting_it() -> None:
    service, settings, _secrets, client, _harness = _make_discord_service()

    result = await service.prepare_for_translation(referral_id=" 7kq9m2 ")

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    discord_start_payload = client.calls[0][1]
    assert discord_start_payload["referral_id"] == " 7kq9m2 "
    assert settings.managed_identity.referral_id is None


@pytest.mark.asyncio
async def test_discord_issue_success_persists_owned_referral_id_and_exposes_result_hints() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            referral_bonus_applied=True,
            referral_id=" 7kq9m2 ",
        ),
    )
    service, settings, _secrets, _client, _harness = _make_discord_service(client=client)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.referral_bonus_applied is True
    assert result.referral_id == "7KQ9M2"
    assert settings.managed_identity.referral_id == "7KQ9M2"


@pytest.mark.asyncio
async def test_discord_issue_success_exposes_talk_together_pass_status() -> None:
    pass_status = TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=1,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            referral_id="7KQ9M2",
            pass_status=pass_status,
        ),
    )
    service, settings, _secrets, _client, _harness = _make_discord_service(client=client)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.referral_id == "7KQ9M2"
    assert result.pass_status == pass_status
    assert settings.managed_identity.referral_id == "7KQ9M2"
    assert not hasattr(settings.managed_identity, "invite_count")


@pytest.mark.asyncio
async def test_discord_issue_success_defaults_referral_bonus_and_preserves_known_owned_id() -> None:
    settings = AppSettings()
    settings.managed_identity.referral_id = "8H3J4N"
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    service, settings, _secrets, _client, _harness = _make_discord_service(
        client=client,
        settings=settings,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.referral_bonus_applied is False
    assert result.referral_id == "8H3J4N"
    assert settings.managed_identity.referral_id == "8H3J4N"


@pytest.mark.asyncio
async def test_discord_issue_success_ignores_malformed_referral_hints_without_clearing_known_id() -> (
    None
):
    settings = AppSettings()
    settings.managed_identity.referral_id = "8H3J4N"
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            referral_bonus_applied="true",  # type: ignore[arg-type]
            referral_id="ABC120",
        ),
    )
    service, settings, _secrets, _client, _harness = _make_discord_service(
        client=client,
        settings=settings,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.referral_bonus_applied is False
    assert result.referral_id == "8H3J4N"
    assert settings.managed_identity.referral_id == "8H3J4N"


@pytest.mark.asyncio
async def test_status_refresh_signs_existing_identity_request_and_persists_owned_referral_id() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    bundle = ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(referral_id=" 7kq9m2 "),
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, _secrets = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.refresh_owned_referral_id_from_status()

    expected_signature = bundle.sign_status_request(
        timestamp="2026-04-08T06:00:45.000Z",
    )["signature"]
    assert client.calls == [
        (
            "trial_status",
            {
                "installation_id": bundle.installation_id,
                "timestamp": "2026-04-08T06:00:45.000Z",
                "signature": expected_signature,
            },
        )
    ]
    assert result == "7KQ9M2"
    assert settings.managed_identity.referral_id == "7KQ9M2"
    assert persist_calls == [(bundle.installation_id, None)]


@pytest.mark.asyncio
async def test_managed_status_refresh_returns_pass_status_without_persisting_it() -> None:
    from puripuly_heart.config.settings import to_dict

    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    bundle = ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    pass_status = TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=2,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(
            referral_id="7KQ9M2",
            pass_status=pass_status,
        ),
    )
    service, settings, _secrets = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.refresh_managed_status()

    assert isinstance(result, ManagedOpenRouterStatusRefreshResult)
    assert result.succeeded is True
    assert result.referral_id == "7KQ9M2"
    assert result.pass_status == pass_status
    assert settings.managed_identity.referral_id == "7KQ9M2"
    assert "invite_count" not in to_dict(settings)["managed_identity"]
    assert client.calls[0][1]["installation_id"] == bundle.installation_id


@pytest.mark.asyncio
async def test_managed_status_refresh_failure_is_distinguishable_from_absent_pass_status() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.referral_id = "8H3J4N"
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="status failed",
        ),
    )
    service, _settings, _secrets = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.refresh_managed_status()

    assert result.succeeded is False
    assert result.referral_id == "8H3J4N"
    assert result.pass_status is None


@pytest.mark.asyncio
async def test_status_refresh_preserves_known_owned_referral_id_when_old_broker_omits_field() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.referral_id = "8H3J4N"
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(referral_id=None),
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, _secrets = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.refresh_owned_referral_id_from_status()

    assert [name for name, _payload in client.calls] == ["trial_status"]
    assert result == "8H3J4N"
    assert settings.managed_identity.referral_id == "8H3J4N"
    assert persist_calls == []


@pytest.mark.asyncio
async def test_status_refresh_preserves_known_owned_referral_id_when_broker_status_fails() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.referral_id = "8H3J4N"
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="broker request timed out",
            operation="trial_status",
        ),
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, _secrets = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.refresh_owned_referral_id_from_status()

    assert [name for name, _payload in client.calls] == ["trial_status"]
    assert result == "8H3J4N"
    assert settings.managed_identity.referral_id == "8H3J4N"
    assert persist_calls == []


@pytest.mark.asyncio
async def test_status_refresh_preserves_newer_owned_referral_id_persisted_while_in_flight() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    status_gate = asyncio.Event()
    status_started = asyncio.Event()
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            referral_id=" 8h3j4n ",
        ),
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(referral_id=" 7kq9m2 "),
        trial_status_gate=status_gate,
        trial_status_started=status_started,
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, _secrets, client, _harness = _make_discord_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    refresh_task = asyncio.create_task(service.refresh_owned_referral_id_from_status())
    try:
        await asyncio.wait_for(status_started.wait(), timeout=1.0)

        issue_result = await service.prepare_for_translation()
        persist_count_after_issue = len(persist_calls)

        assert issue_result.referral_id == "8H3J4N"
        assert settings.managed_identity.referral_id == "8H3J4N"

        status_gate.set()
        refresh_result = await asyncio.wait_for(refresh_task, timeout=1.0)

        assert refresh_result == "8H3J4N"
        assert settings.managed_identity.referral_id == "8H3J4N"
        assert len(persist_calls) == persist_count_after_issue
    finally:
        status_gate.set()
        if not refresh_task.done():
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh_task


@pytest.mark.asyncio
async def test_status_refresh_skips_without_identity_mutation_when_existing_bundle_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.installation_id = "not-a-valid-managed-installation-id"
    settings.managed_identity.release_token = "release-token-kept"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    settings.managed_identity.referral_id = "8H3J4N"
    secrets = InMemorySecretStore()
    secrets.set("unrelated-secret", "kept")
    before_secret_items = dict(secrets._items)
    before_managed_identity = (
        settings.managed_identity.installation_id,
        settings.managed_identity.release_token,
        settings.managed_identity.release_token_expires_at,
        settings.managed_identity.verified_hardware_hash,
        settings.managed_identity.verified_hardware_hash_salt_version,
        settings.managed_identity.referral_id,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("status refresh must not create or regenerate identity bundles")

    monkeypatch.setattr(
        "puripuly_heart.core.managed_openrouter_release.ensure_managed_identity_bundle",
        fail_if_called,
    )
    monkeypatch.setattr(
        "puripuly_heart.core.managed_openrouter_release.regenerate_managed_identity_bundle",
        fail_if_called,
    )
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(referral_id="7KQ9M2"),
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, secrets = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.refresh_owned_referral_id_from_status()

    assert result == "8H3J4N"
    assert client.calls == []
    assert persist_calls == []
    assert secrets._items == before_secret_items
    assert (
        settings.managed_identity.installation_id,
        settings.managed_identity.release_token,
        settings.managed_identity.release_token_expires_at,
        settings.managed_identity.verified_hardware_hash,
        settings.managed_identity.verified_hardware_hash_salt_version,
        settings.managed_identity.referral_id,
    ) == before_managed_identity


@pytest.mark.asyncio
async def test_status_refresh_close_cancels_in_flight_status_without_persisting_stale_settings() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.referral_id = "8H3J4N"
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    status_gate = asyncio.Event()
    status_started = asyncio.Event()
    client = FakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(referral_id="7KQ9M2"),
        trial_status_gate=status_gate,
        trial_status_started=status_started,
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, _secrets = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )
    refresh_task = asyncio.create_task(service.refresh_owned_referral_id_from_status())
    try:
        await asyncio.wait_for(status_started.wait(), timeout=1.0)

        await service.close()

        assert refresh_task.done()
        assert settings.managed_identity.referral_id == "8H3J4N"
        assert persist_calls == []
    finally:
        status_gate.set()
        if not refresh_task.done():
            refresh_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await refresh_task


@pytest.mark.asyncio
async def test_issue_persists_managed_user_identifier_after_managed_key_success() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            openrouter_user_id="  user-123  ",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    assert secrets.get(OPENROUTER_MANAGED_USER_ID_SECRET) == "user-123"
    assert (
        secrets.get(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET)
        == settings.managed_identity.installation_id
    )
    assert (
        load_managed_openrouter_user_identifier(
            build_openrouter_credential_runtime_config(settings), secrets=secrets
        )
        == "user-123"
    )


@pytest.mark.asyncio
async def test_direct_issue_delivery_ack_required_stores_acks_and_clears_before_ready() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="managed-ref-discord",
            delivery_ack_required=True,
            delivery_id="delivery-discord",
            delivery_ack_token="delivery-ack-token",
            delivery_ack_expires_at="2026-07-07T00:15:00.000Z",
        ),
        delivery_ack_result=ManagedKeyDeliveryAckResult(
            succeeded=True,
            status="acknowledged",
        ),
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert client.calls[-1][0] == "delivery_ack"
    assert settings.managed_identity.pending_delivery_ack_source is None
    assert secrets.get("openrouter_managed_delivery_ack_token") is None


@pytest.mark.asyncio
async def test_direct_issue_delivery_ack_failure_keeps_pending_and_returns_retry() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="managed-ref-discord",
            delivery_ack_required=True,
            delivery_id="delivery-discord",
            delivery_ack_token="delivery-ack-token",
        ),
        delivery_ack_result=ManagedKeyDeliveryAckResult(
            succeeded=False,
            status="retryable",
        ),
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert settings.managed_identity.pending_delivery_ack_source == "discord"
    assert settings.managed_identity.pending_delivery_ack_delivery_id == "delivery-discord"
    assert secrets.get("openrouter_managed_delivery_ack_token") == "delivery-ack-token"
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"


@pytest.mark.asyncio
async def test_direct_issue_delivery_ack_malformed_metadata_does_not_return_ready() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="managed-ref-discord",
            delivery_ack_required=True,
            delivery_id=None,
            delivery_ack_token="delivery-ack-token",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None
    assert settings.managed_identity.pending_delivery_ack_source is None


@pytest.mark.asyncio
async def test_direct_issue_pending_ack_persist_failure_happens_before_local_key_write() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="managed-ref-discord",
            delivery_ack_required=True,
            delivery_id="delivery-discord",
            delivery_ack_token="delivery-ack-token",
        )
    )

    def fail_persist(_updated: AppSettings) -> None:
        raise RuntimeError("pending persist failed")

    service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, fail_persist),
        secrets=secrets,
        client=client,
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
        signed_at_provider=lambda: "2026-04-08T06:00:45.000Z",
        monotonic_ms_provider=lambda: 1_000,
    )

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None


@pytest.mark.asyncio
async def test_direct_issue_ack_token_store_and_key_rollback_failure_forces_future_retry() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = FailingAckTokenSetAndManagedKeyDeleteSecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="managed-ref-discord",
            delivery_ack_required=True,
            delivery_id="delivery-discord",
            delivery_ack_token="delivery-ack-token",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    first = await service.ensure_key_for_llm_start()
    second = await service.ensure_key_for_llm_start()

    assert first.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert second.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert settings.managed_identity.pending_delivery_ack_source is None
    assert settings.managed_identity.pending_delivery_ack_delivery_id is None
    assert secrets.get("openrouter_managed_delivery_ack_token") is None


@pytest.mark.asyncio
async def test_direct_issue_ack_success_token_clear_failure_keeps_pending_and_future_retry() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = FailingDeleteSecretStore(fail_on_key="openrouter_managed_delivery_ack_token")
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            managed_credential_ref="managed-ref-discord",
            delivery_ack_required=True,
            delivery_id="delivery-discord",
            delivery_ack_token="delivery-ack-token",
        ),
        delivery_ack_result=ManagedKeyDeliveryAckResult(
            succeeded=True,
            status="acknowledged",
        ),
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    first = await service.ensure_key_for_llm_start()
    second = await service.ensure_key_for_llm_start()

    assert first.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert second.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert settings.managed_identity.pending_delivery_ack_source == "discord"


@pytest.mark.asyncio
async def test_issue_keeps_ready_and_cleans_managed_user_identifier_cache_on_second_write_failure() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = FailingManagedKeySecretStore(
        fail_on_key=OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET
    )
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    secrets.set_attempts.clear()
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            openrouter_user_id="user-123",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert result.local_key_available is True
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    assert secrets.get(OPENROUTER_MANAGED_USER_ID_SECRET) is None
    assert secrets.get(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET) is None
    assert (
        load_managed_openrouter_user_identifier(
            build_openrouter_credential_runtime_config(settings), secrets=secrets
        )
        is None
    )
    assert secrets.set_attempts == [
        (OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key"),
        (OPENROUTER_MANAGED_USER_ID_SECRET, "user-123"),
        (
            OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
            settings.managed_identity.installation_id,
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("openrouter_user_id", [None, "   "])
async def test_issue_omission_or_invalid_user_id_preserves_existing_managed_user_identifier_cache(
    openrouter_user_id: str | None,
) -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    secrets.set(OPENROUTER_MANAGED_USER_ID_SECRET, "cached-user-1")
    secrets.set(
        OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
        settings.managed_identity.installation_id,
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(
            openrouter_api_key="managed-key",
            openrouter_user_id=openrouter_user_id,
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    assert secrets.get(OPENROUTER_MANAGED_USER_ID_SECRET) == "cached-user-1"
    assert (
        secrets.get(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET)
        == settings.managed_identity.installation_id
    )
    assert (
        load_managed_openrouter_user_identifier(
            build_openrouter_credential_runtime_config(settings), secrets=secrets
        )
        == "cached-user-1"
    )


@pytest.mark.asyncio
async def test_prepare_for_translation_reuses_verified_pending_release_state_and_issues_key() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    expected_hardware_hash = _expected_hardware_hash(
        fingerprint_salt="fingerprint-salt-test",
        raw_hardware_fingerprint="raw-hardware-fingerprint-test",
    )
    settings.managed_identity.verified_hardware_hash = expected_hardware_hash
    settings.managed_identity.verified_hardware_hash_salt_version = 7
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key")
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    prepare_result = await service.prepare_for_translation()

    assert prepare_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert prepare_result.api_key == "managed-key"
    assert prepare_result.local_key_available is True
    assert prepare_result.pending_issue is False
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    issue_payload = client.calls[0][1]
    assert issue_payload["budget_usd"] == 0.07
    assert issue_payload["hardware_hash"] == expected_hardware_hash
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_prepare_for_translation_and_ensure_key_for_llm_start_share_single_issue_task() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    issue_started = asyncio.Event()
    issue_gate = asyncio.Event()
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
        issue_gate=issue_gate,
        issue_started=issue_started,
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    prepare_task = asyncio.create_task(service.prepare_for_translation())
    await issue_started.wait()
    ensure_task = asyncio.create_task(service.ensure_key_for_llm_start())
    await asyncio.sleep(0)
    issue_gate.set()

    prepare_result, ensure_result = await asyncio.gather(prepare_task, ensure_task)

    assert [name for name, _payload in client.calls] == ["issue"]
    assert prepare_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert ensure_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert prepare_result.local_key_available is True
    assert ensure_result.local_key_available is True
    assert sorted([prepare_result.single_flight_reused, ensure_result.single_flight_reused]) == [
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_issue_uses_qwen_managed_model_from_selection_alias() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key")
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.api_key == "managed-key"
    issue_payload = client.calls[0][1]
    assert issue_payload["model"] == OpenRouterLLMModel.QWEN_35_FLASH_02_23.value


@pytest.mark.asyncio
async def test_prepare_for_translation_restarts_when_legacy_release_token_lacks_verified_snapshot() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    client = FakeManagedReleaseClient()
    persist_calls: list[tuple[str | None, str | None]] = []
    service, _, _ = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert result.message_key == "managed_release.restart"
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert persist_calls[-1][1] is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_prepare_for_translation_preserves_legacy_hardware_hash_provider_semantics() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    harness = FakeDiscordOAuthHarness()
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()

    service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets=secrets,
        client=client,
        app_version="2.0.0",
        hardware_hash_provider=lambda: "precomputed-hardware-hash-123",
        signed_at_provider=lambda: "2026-04-08T06:00:45.000Z",
        monotonic_ms_provider=lambda: 1_000,
        discord_oauth_listener_factory=harness.bind_listener,
        discord_oauth_callback_runner=harness.run_callback_flow,
    )

    await service.prepare_for_translation()

    issue_payload = client.calls[1][1]
    assert issue_payload["hardware_hash"] == "precomputed-hardware-hash-123"


@pytest.mark.asyncio
async def test_prepare_for_translation_collects_sync_raw_hardware_fingerprint_off_thread() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    event_loop_thread_id = threading.get_ident()
    provider_thread_ids: list[int] = []

    def raw_provider() -> str:
        provider_thread_ids.append(threading.get_ident())
        return "raw-hardware-fingerprint-test"

    service, _, _, _client, _harness = _make_discord_service(
        client=client,
        raw_hardware_fingerprint_provider=raw_provider,
    )

    await service.prepare_for_translation()

    assert len(provider_thread_ids) == 1
    assert provider_thread_ids[0] != event_loop_thread_id


@pytest.mark.asyncio
async def test_prepare_for_translation_stops_early_on_terminal_discord_start_error() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=ManagedOpenRouterReleaseError(
            code="trial_not_eligible",
            error_class="terminal",
            message="trial is not eligible",
            operation="discord_start",
        )
    )
    service, settings, secrets, client, harness = _make_discord_service(client=client)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.not_eligible"
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None
    assert harness.listeners[0].closed is True
    assert [name for name, _payload in client.calls] == ["discord_start"]


@pytest.mark.asyncio
async def test_prepare_for_translation_reuses_single_flight_for_repeated_trans_attempts() -> None:
    gate = asyncio.Event()
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
        discord_start_gate=gate,
    )
    service, _, _, client, _harness = _make_discord_service(client=client)

    first_task = asyncio.create_task(service.prepare_for_translation())
    await asyncio.sleep(0)
    second_task = asyncio.create_task(service.prepare_for_translation())
    await asyncio.sleep(0)
    gate.set()

    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert [name for name, _payload in client.calls] == ["discord_start", "discord_issue"]
    assert first_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert second_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert sorted([first_result.single_flight_reused, second_result.single_flight_reused]) == [
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_close_closes_underlying_client_transport_when_available() -> None:
    client = ClosableFakeManagedReleaseClient()
    service, _, _ = _make_service(client=client)

    await service.close()

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_close_continues_status_and_client_cleanup_when_oauth_close_fails() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.referral_id = "8H3J4N"
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )
    status_gate = asyncio.Event()
    status_started = asyncio.Event()
    client = ClosableFakeManagedReleaseClient(
        trial_status_result=ManagedOpenRouterTrialStatusSuccess(referral_id="7KQ9M2"),
        trial_status_gate=status_gate,
        trial_status_started=status_started,
    )
    oauth_runtime = FailingCloseOAuthRuntime()
    persist_calls: list[tuple[str | None, str | None]] = []
    service, settings, _secrets = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
        oauth_runtime=oauth_runtime,
    )
    refresh_task = asyncio.create_task(service.refresh_owned_referral_id_from_status())
    try:
        await asyncio.wait_for(status_started.wait(), timeout=1.0)

        with pytest.raises(RuntimeError, match="oauth close failed"):
            await service.close()

        assert oauth_runtime.close_calls == 1
        assert client.close_calls == 1
        assert refresh_task.done()
        assert refresh_task.cancelled()
        assert settings.managed_identity.referral_id == "8H3J4N"
        assert persist_calls == []
    finally:
        status_gate.set()
        if not refresh_task.done():
            refresh_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await refresh_task


@pytest.mark.asyncio
async def test_close_aggregates_oauth_status_and_client_cleanup_failures() -> None:
    client = FailingCloseManagedReleaseClient()
    oauth_runtime = FailingCloseOAuthRuntime()
    service, _, _ = _make_service(
        client=client,
        oauth_runtime=oauth_runtime,
    )
    status_started = asyncio.Event()

    async def failing_status_task() -> None:
        status_started.set()
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError as exc:
            raise RuntimeError("status cleanup failed") from exc

    status_task = asyncio.create_task(failing_status_task())
    service._status_refresh_tasks.add(status_task)
    try:
        await asyncio.wait_for(status_started.wait(), timeout=1.0)

        with pytest.raises(ExceptionGroup) as exc_info:
            await service.close()

        messages = sorted(str(exc) for exc in exc_info.value.exceptions)
        assert messages == [
            "client close failed",
            "oauth close failed",
            "status cleanup failed",
        ]
        assert oauth_runtime.close_calls == 1
        assert client.close_calls == 1
        assert status_task.done()
    finally:
        if not status_task.done():
            status_task.cancel()
            with contextlib.suppress(Exception):
                await status_task


@pytest.mark.asyncio
async def test_unavailable_client_discord_methods_raise_release_error_not_attribute_error() -> None:
    client = UnavailableManagedOpenRouterReleaseClient()

    with pytest.raises(ManagedOpenRouterReleaseError) as start_exc:
        await client.start_discord_oauth(
            installation_id="install-discord-123",
            device_public_key="device-public-key-123",
            redirect_uri="http://127.0.0.1:62187/discord/callback",
            app_version="2.0.0",
        )
    assert start_exc.value.code == "trial_unavailable"
    assert start_exc.value.error_class == "retryable"
    assert start_exc.value.message == "managed OpenRouter release is unavailable"

    with pytest.raises(ManagedOpenRouterReleaseError) as issue_exc:
        await client.issue_discord_managed_key({"code": "discord-oauth-code"})
    assert issue_exc.value.code == "trial_unavailable"
    assert issue_exc.value.error_class == "retryable"
    assert issue_exc.value.message == "managed OpenRouter release is unavailable"


@pytest.mark.asyncio
async def test_issue_honors_retry_after_without_starting_parallel_retries() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
            retry_after_ms=9_000,
        )
    )
    monotonic_now = {"value": 1_000}

    service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets=secrets,
        client=client,
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
        signed_at_provider=lambda: "2026-04-08T06:00:45.000Z",
        monotonic_ms_provider=lambda: monotonic_now["value"],
    )

    first = await service.ensure_key_for_llm_start()
    second = await service.ensure_key_for_llm_start()

    assert first.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert first.retry_after_ms == 9_000
    assert first.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode=None,
        retry_after_ms=9_000,
        message="managed OpenRouter release is unavailable",
    )
    assert second.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert second.retry_after_ms == 9_000
    assert second.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode=None,
        retry_after_ms=9_000,
        message="managed OpenRouter release is unavailable",
    )
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_prepare_for_translation_honors_retry_after_while_pending_release_exists() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="trial_unavailable",
            error_class="retryable",
            message="managed OpenRouter release is unavailable",
            retry_after_ms=9_000,
        )
    )
    service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets=secrets,
        client=client,
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
        signed_at_provider=lambda: "2026-04-08T06:00:45.000Z",
        monotonic_ms_provider=lambda: 1_000,
    )

    issue_result = await service.ensure_key_for_llm_start()
    prepare_result = await service.prepare_for_translation()

    assert issue_result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert issue_result.retry_after_ms == 9_000
    assert issue_result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode=None,
        retry_after_ms=9_000,
        message="managed OpenRouter release is unavailable",
    )
    assert prepare_result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert prepare_result.retry_after_ms == 9_000
    assert prepare_result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode=None,
        retry_after_ms=9_000,
        message="managed OpenRouter release is unavailable",
    )
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_restart_clears_release_state_without_switching_sources() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="challenge_invalid",
            error_class="security_fail",
            subcode="signature_mismatch",
            message="signature mismatch",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None


@pytest.mark.asyncio
async def test_issue_challenge_expired_subcode_restarts_and_clears_state() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="challenge_expired",
            error_class="retryable",
            subcode="release_token_expired",
            retry_after_ms=0,
            message="release_token has expired and must be reissued",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None


@pytest.mark.asyncio
async def test_issue_trial_not_eligible_managed_key_unrecoverable_stops_as_not_eligible() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="trial_not_eligible",
            error_class="terminal",
            subcode="managed_key_unrecoverable",
            retry_after_ms=None,
            message="managed key was already issued and cannot be recovered",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.not_eligible"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_not_eligible",
        error_class="terminal",
        subcode="managed_key_unrecoverable",
        retry_after_ms=None,
        message="managed key was already issued and cannot be recovered",
    )
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_issuance_suspended_retries_with_brake_copy() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="issuance_suspended",
            error_class="retryable",
            subcode="asn_fast_path",
            retry_after_ms=5_000,
            message="new entitlement issuance is temporarily suspended",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert result.message_key == "managed_release.brake"
    assert result.message_kwargs == {"retry_after_ms": 5_000}
    assert result.retry_after_ms == 5_000
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="issuance_suspended",
        error_class="retryable",
        subcode="asn_fast_path",
        retry_after_ms=5_000,
        message="new entitlement issuance is temporarily suspended",
    )
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token == "release-token-1"
    assert settings.managed_identity.release_token_expires_at == "2026-04-08T06:15:00.000Z"
    assert settings.managed_identity.verified_hardware_hash == "verified-hardware-hash-1"
    assert settings.managed_identity.verified_hardware_hash_salt_version == 7
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_issuance_suspended_with_revoked_lifecycle_stops_with_contact_copy() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="issuance_suspended",
            error_class="retryable",
            subcode="asn_fast_path",
            retry_after_ms=5_000,
            message="revoked by policy",
            managed_lifecycle="revoked",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.revoked_contact"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="issuance_suspended",
        error_class="retryable",
        subcode="asn_fast_path",
        retry_after_ms=5_000,
        message="revoked by policy",
    )
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_trial_not_eligible_with_revoked_lifecycle_stops_with_contact_copy() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="trial_not_eligible",
            error_class="terminal",
            subcode=None,
            retry_after_ms=None,
            message="revoked by policy",
            managed_lifecycle="revoked",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.revoked_contact"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_not_eligible",
        error_class="terminal",
        subcode=None,
        retry_after_ms=None,
        message="revoked by policy",
    )
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_non_trial_code_with_revoked_lifecycle_stops_with_contact_copy() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterReleaseError(
            code="internal_error",
            error_class="terminal",
            subcode=None,
            retry_after_ms=None,
            message="revoked by policy",
            managed_lifecycle="revoked",
        )
    )
    service, _, _ = _make_service(client=client, settings=settings, secrets=secrets)

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.revoked_contact"
    assert result.diagnostics == ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="internal_error",
        error_class="terminal",
        subcode=None,
        retry_after_ms=None,
        message="revoked by policy",
    )
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_restarts_when_identity_bundle_regenerates_before_issue() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.installation_id = "018f1f56-9f2d-7abc-9def-1234567890ab"
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key")
    )
    service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets=InMemorySecretStore(),
        client=client,
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
        signed_at_provider=lambda: "2026-04-08T06:00:45.000Z",
        monotonic_ms_provider=lambda: 1_000,
    )

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert client.calls == []


@pytest.mark.asyncio
async def test_issue_stops_cleanly_when_managed_key_persistence_fails_after_successful_issue() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = FailingManagedKeySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    secrets.set_attempts.clear()
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key")
    )
    persist_calls: list[tuple[str | None, str | None]] = []
    service, _, _ = _make_service(
        client=client,
        settings=settings,
        secrets=secrets,
        persist_calls=persist_calls,
    )

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.stop"
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None
    assert secrets.set_attempts == [(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")]
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert persist_calls[-1][1] is None
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_issue_stops_and_restores_pending_release_state_when_cleanup_persist_fails() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = FailingManagedKeySecretStore()
    ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None), secrets
    )
    secrets.set_attempts.clear()
    settings.managed_identity.release_token = "release-token-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:15:00.000Z"
    _set_verified_snapshot(settings)
    client = FakeManagedReleaseClient(
        issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key")
    )
    persist_calls: list[tuple[str | None, str | None]] = []

    def persist_and_fail(updated: AppSettings) -> None:
        persist_calls.append(
            (
                updated.managed_identity.release_token,
                updated.managed_identity.verified_hardware_hash,
            )
        )
        raise RuntimeError("settings persistence failed")

    service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, persist_and_fail),
        secrets=secrets,
        client=client,
        app_version="2.0.0",
        raw_hardware_fingerprint_provider=lambda: "raw-hardware-fingerprint-test",
        signed_at_provider=lambda: "2026-04-08T06:00:45.000Z",
        monotonic_ms_provider=lambda: 1_000,
    )

    result = await service.ensure_key_for_llm_start()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.stop"
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None
    assert secrets.set_attempts == [(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")]
    assert persist_calls == [(None, None)]
    assert settings.managed_identity.release_token == "release-token-1"
    assert settings.managed_identity.release_token_expires_at == "2026-04-08T06:15:00.000Z"
    assert settings.managed_identity.verified_hardware_hash == "verified-hardware-hash-1"
    assert settings.managed_identity.verified_hardware_hash_salt_version == 7
    assert [name for name, _payload in client.calls] == ["issue"]


@pytest.mark.asyncio
async def test_prepare_single_flight_survives_waiter_cancellation() -> None:
    gate = asyncio.Event()
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
        discord_start_gate=gate,
    )
    service, _, _, client, _harness = _make_discord_service(client=client)

    first_task = asyncio.create_task(service.prepare_for_translation())
    await asyncio.sleep(0)
    second_task = asyncio.create_task(service.prepare_for_translation())
    await asyncio.sleep(0)
    first_task.cancel()
    await asyncio.sleep(0)
    gate.set()

    with pytest.raises(asyncio.CancelledError):
        await first_task
    second_result = await second_task

    assert second_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert [name for name, _payload in client.calls] == ["discord_start", "discord_issue"]


@pytest.mark.asyncio
async def test_close_cancels_in_flight_prepare_task() -> None:
    gate = asyncio.Event()
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
        discord_start_gate=gate,
    )
    service, _, _, _client, harness = _make_discord_service(client=client)

    task = asyncio.create_task(service.prepare_for_translation())
    for _ in range(10):
        if harness.listeners:
            break
        await asyncio.sleep(0)
    assert harness.listeners
    await service.close()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.listeners[0].closed is True


@pytest.mark.asyncio
async def test_prepare_for_translation_stops_when_hardware_fingerprint_lookup_fails() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    service, settings, secrets, client, harness = _make_discord_service(
        client=client,
        raw_hardware_fingerprint_provider=lambda: (_ for _ in ()).throw(
            RuntimeError("fingerprint unavailable")
        ),
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "managed_release.stop"
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.managed_identity.release_token is None
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None
    assert harness.listeners[0].closed is True
    assert [name for name, _payload in client.calls] == ["discord_start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "subcode"),
    [
        ("discord_start", "device_public_key_registered"),
        ("discord_issue", "installation_binding_mismatch"),
    ],
)
async def test_prepare_for_translation_regenerates_identity_on_binding_mismatch_security_fail(
    stage: str,
    subcode: str,
) -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    secrets = InMemorySecretStore()
    first_bundle = ensure_managed_identity_bundle(
        ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets,
    )

    if stage == "discord_start":
        client = FakeManagedReleaseClient(
            discord_start_result=ManagedOpenRouterReleaseError(
                code="trial_not_eligible",
                error_class="security_fail",
                subcode=subcode,
                message="device_public_key is already registered to a different installation_id",
            )
        )
    else:
        client = FakeManagedReleaseClient(
            discord_start_result=_make_discord_start_success(),
            discord_issue_result=ManagedOpenRouterReleaseError(
                code="trial_not_eligible",
                error_class="security_fail",
                subcode=subcode,
                message="issue must use the registered device_public_key for installation_id",
            ),
        )

    service, _, _, _client, _harness = _make_discord_service(
        client=client,
        settings=settings,
        secrets=secrets,
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RESTART
    assert result.message_key == "managed_release.restart"
    assert settings.managed_identity.installation_id != first_bundle.installation_id
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert secrets.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None


@dataclass
class FakeIssueService:
    results: list[ManagedOpenRouterReleaseResult]
    ensure_calls: list[str]

    def __init__(self, *results: ManagedOpenRouterReleaseResult) -> None:
        self.results = list(results)
        self.ensure_calls = []

    async def ensure_key_for_llm_start(self) -> ManagedOpenRouterReleaseResult:
        self.ensure_calls.append("llm_start")
        return self.results.pop(0)


@dataclass
class RaisingIssueService:
    exc: Exception

    async def ensure_key_for_llm_start(self) -> ManagedOpenRouterReleaseResult:
        raise self.exc


@dataclass
class FakeDelegateProvider:
    translate_calls: list[dict[str, object]]

    def __init__(self) -> None:
        self.translate_calls = []

    async def translate(self, **kwargs: Any) -> Translation:
        self.translate_calls.append(dict(kwargs))
        return Translation(kwargs["utterance_id"], text="translated")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_managed_openrouter_provider_issues_on_first_llm_start_only() -> None:
    service = FakeIssueService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            pending_issue=False,
        )
    )
    delegate = FakeDelegateProvider()
    created_keys: list[str] = []
    provider = ManagedOpenRouterLLMProvider(
        release_service=service,
        delegate_factory=lambda api_key: created_keys.append(api_key) or delegate,
    )

    first = await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="prompt",
        source_language="ko",
        target_language="en",
    )
    second = await provider.translate(
        utterance_id=uuid4(),
        text="again",
        system_prompt="prompt",
        source_language="ko",
        target_language="en",
    )

    assert first.text == "translated"
    assert second.text == "translated"
    assert service.ensure_calls == ["llm_start"]
    assert created_keys == ["managed-key"]
    assert len(delegate.translate_calls) == 2


@pytest.mark.asyncio
async def test_managed_openrouter_provider_translate_after_preissue_does_not_issue_again() -> None:
    client = FakeManagedReleaseClient(
        discord_start_result=_make_discord_start_success(),
        discord_issue_result=ManagedOpenRouterIssueSuccess(openrouter_api_key="managed-key"),
    )
    service, _, _, client, _harness = _make_discord_service(client=client)
    delegate = FakeDelegateProvider()
    created_keys: list[str] = []
    provider = ManagedOpenRouterLLMProvider(
        release_service=service,
        delegate_factory=lambda api_key: created_keys.append(api_key) or delegate,
    )

    prepare_result = await service.prepare_for_translation()
    translate_result = await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="prompt",
        source_language="ko",
        target_language="en",
    )

    assert prepare_result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert prepare_result.api_key == "managed-key"
    assert translate_result.text == "translated"
    assert created_keys == ["managed-key"]
    assert [name for name, _payload in client.calls] == ["discord_start", "discord_issue"]


@pytest.mark.asyncio
async def test_managed_openrouter_provider_notifies_when_delegate_becomes_ready() -> None:
    service = FakeIssueService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            pending_issue=False,
        )
    )
    delegate = FakeDelegateProvider()
    ready_calls: list[str] = []
    provider = ManagedOpenRouterLLMProvider(
        release_service=service,
        delegate_factory=lambda _api_key: delegate,
        on_delegate_ready=lambda: ready_calls.append("ready"),
    )

    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="prompt",
        source_language="ko",
        target_language="en",
    )
    await provider.translate(
        utterance_id=uuid4(),
        text="again",
        system_prompt="prompt",
        source_language="ko",
        target_language="en",
    )

    assert ready_calls == ["ready"]


@pytest.mark.asyncio
async def test_managed_openrouter_provider_preserves_diagnostics_in_user_facing_error() -> None:
    diagnostics = ManagedOpenRouterReleaseDiagnostics(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode="broker_backoff",
        retry_after_ms=9_000,
        message="broker is temporarily unavailable",
    )
    service = FakeIssueService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="managed_release.retry_after_ms",
            message_kwargs={"retry_after_ms": 9_000},
            retry_after_ms=9_000,
            diagnostics=diagnostics,
        )
    )
    provider = ManagedOpenRouterLLMProvider(
        release_service=service,
        delegate_factory=lambda _api_key: FakeDelegateProvider(),
    )

    with pytest.raises(ManagedOpenRouterUserFacingError) as exc_info:
        await provider.translate(
            utterance_id=uuid4(),
            text="hello",
            system_prompt="prompt",
            source_language="ko",
            target_language="en",
        )

    assert exc_info.value.diagnostics == diagnostics


@pytest.mark.asyncio
async def test_managed_openrouter_provider_wraps_unexpected_issue_start_error_as_user_facing_error() -> (
    None
):
    provider = ManagedOpenRouterLLMProvider(
        release_service=RaisingIssueService(RuntimeError("issue boom")),
        delegate_factory=lambda _api_key: FakeDelegateProvider(),
    )

    with pytest.raises(ManagedOpenRouterUserFacingError) as exc_info:
        await provider.translate(
            utterance_id=uuid4(),
            text="hello",
            system_prompt="prompt",
            source_language="ko",
            target_language="en",
        )

    assert exc_info.value.message_key == "managed_release.retry"
    assert exc_info.value.diagnostics == ManagedOpenRouterReleaseDiagnostics(message="issue boom")
