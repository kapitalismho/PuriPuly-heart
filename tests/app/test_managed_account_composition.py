from __future__ import annotations

from typing import cast

import pytest
from puripuly_heart.app.services.managed_auth import ManagedAuthOwner
from puripuly_heart.app.services.managed_usage import ManagedUsageOwner
from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeOwner
from puripuly_heart.app.services.provider_settings import ProviderSettingsOwner
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.app.wiring_managed_account import (
    ManagedOpenRouterReleaseRuntime,
    ManagedTranslationRuntimeAccess,
    compose_managed_account,
)
from puripuly_heart.core.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseService,
    UnavailableManagedOpenRouterReleaseClient,
)
from puripuly_heart.core.openrouter_credentials import OPENROUTER_BYOK_API_KEY_SECRET

from puripuly_heart.app import wiring_managed_account as managed_account_module
from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
)
from puripuly_heart.app.services.openrouter_pkce_flow import (
    OpenRouterPkceApplicationOwner,
    OpenRouterPkceFlowOwner,
)
from puripuly_heart.app.services.translation_enable import TranslationEnableOwner
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    TranslationConnection,
    TranslationModel,
    materialize_translation_settings,
)
from puripuly_heart.core.hardware_fingerprint import get_raw_hardware_fingerprint
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)


class SecretStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


def _managed_settings() -> AppSettings:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    return settings


@pytest.mark.asyncio
async def test_release_runtime_owns_selection_broker_fallback_and_telemetry(
    tmp_path,
) -> None:
    owner = compose_settings_owner(tmp_path / "settings.json")
    owner.current = _managed_settings()
    runtime = ManagedOpenRouterReleaseRuntime(
        settings=owner,
        config_path=owner.path,
        callback_received=lambda: None,
    )

    owner.current.openrouter.broker_base_url = "https://broker.example.test/"
    service = await runtime.rebuild(secrets=SecretStore())

    assert runtime.selected() is True
    assert isinstance(service, ManagedOpenRouterReleaseService)
    assert isinstance(service.client, HttpManagedOpenRouterBrokerClient)
    assert service.client.base_url == "https://broker.example.test"
    assert runtime.telemetry_client is service.client
    assert service.raw_hardware_fingerprint_provider is get_raw_hardware_fingerprint

    owner.current.openrouter.broker_base_url = "https://broker.example.test/prefix"
    service = await runtime.rebuild(secrets=SecretStore())

    assert isinstance(service, ManagedOpenRouterReleaseService)
    assert isinstance(service.client, UnavailableManagedOpenRouterReleaseClient)
    assert runtime.telemetry_client is None

    owner.current.translation.connection = TranslationConnection.OPENROUTER
    assert await runtime.rebuild(secrets=SecretStore()) is None
    assert runtime.selected() is False


@pytest.mark.asyncio
async def test_release_runtime_awaits_replacement_and_shutdown(tmp_path) -> None:
    closed: list[str] = []

    class Service:
        async def close(self) -> None:
            closed.append("closed")

    runtime = ManagedOpenRouterReleaseRuntime(
        settings=compose_settings_owner(tmp_path / "settings.json"),
        config_path=tmp_path / "settings.json",
        callback_received=lambda: None,
        service=cast(ManagedOpenRouterReleaseService, Service()),
    )

    await runtime.close()

    assert closed == ["closed"]
    assert runtime.service is None
    assert runtime.telemetry_client is None


@pytest.mark.asyncio
async def test_translation_runtime_access_owns_rebuild_state_and_context() -> None:
    class Runtime:
        provider = None

        def __init__(self) -> None:
            self.clear_calls = 0
            self.translation_runtime_configuration = TranslationRuntimeConfigurationOwner(
                TranslationRuntimeConfig(translation_enabled=False)
            )

        def clear_context(self) -> None:
            self.clear_calls += 1

    current = Runtime()
    rebuilds: list[str] = []

    async def rebuild() -> bool:
        rebuilds.append("rebuild")
        current.provider = object()
        return True

    access = ManagedTranslationRuntimeAccess(
        llm_runtime_provider=lambda: current,
        context_provider=lambda: current,
        translation_runtime_configuration_provider=(
            lambda: current.translation_runtime_configuration
        ),
        rebuild_llm=rebuild,
    )

    assert access.presence() == (True, False)
    assert await access.ensure("if_missing") is True
    access.set_enabled(True)
    access.clear_context()

    assert rebuilds == ["rebuild"]
    assert access.snapshot() == (True, True, current.provider)
    assert current.clear_calls == 1


@pytest.mark.asyncio
async def test_managed_account_composition_wires_all_owners_and_secret_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    owner = compose_settings_owner(tmp_path / "settings.json")
    owner.current = _managed_settings()
    owner.current.managed_identity.referral_id = " 7kq9m2 "
    store = SecretStore()

    secret_store_calls: list[tuple[object, object]] = []

    def create_store(secrets: object, *, config_path: object) -> SecretStore:
        secret_store_calls.append((secrets, config_path))
        return store

    monkeypatch.setattr(
        managed_account_module,
        "create_secret_store",
        create_store,
    )
    runtime = ManagedTranslationRuntimeAccess(
        llm_runtime_provider=lambda: None,
        context_provider=lambda: None,
        translation_runtime_configuration_provider=lambda: None,
        rebuild_llm=lambda: pytest.fail("runtime rebuild must not run"),
    )
    results = SettingsTransactionResultOwner()
    runtime_state_changes: list[str] = []
    components = compose_managed_account(
        config_path=owner.path,
        settings=owner,
        provider_settings=cast(ProviderSettingsOwner, object()),
        provider_runtime=cast(ProviderRuntimeOwner, object()),
        verifier=cast(ProviderVerifierPort, object()),
        results=results,
        runtime=runtime,
        ingress_provider=lambda: False,
        pending_sink=lambda _pending: None,
        usage_view_sink=lambda _state: None,
        dashboard_sink=lambda _enabled: None,
        message_sink=lambda _key, _kwargs: None,
        qq_dialog_sink=lambda: None,
        founder_dialog=lambda: False,
        failure_route=lambda _source: None,
        log_basic=lambda _message: None,
        log_detailed=lambda _message: None,
        log_error=lambda _message: None,
        basic_warning_sink=lambda _message: None,
        detailed_warning_sink=lambda _message, _exception: None,
        runtime_state_changed=lambda: runtime_state_changes.append("changed"),
    )

    assert isinstance(components.auth, ManagedAuthOwner)
    assert isinstance(components.usage, ManagedUsageOwner)
    assert isinstance(components.translation, TranslationEnableOwner)
    assert isinstance(components.pkce_flow, OpenRouterPkceFlowOwner)
    assert isinstance(components.pkce, OpenRouterPkceApplicationOwner)
    assert components.pkce.results is results
    assert components.auth.dashboard_action() == "prompt"
    assert components.usage.current_referral_id == "7KQ9M2"
    translation_state = components.translation.state_provider()
    assert translation_state.managed_selected is True
    assert translation_state.runtime_available is False
    components.translation.disable_for_managed_exhaustion(reopen_founder_letter=False)
    assert runtime_state_changes == ["changed"]

    callback_events: list[str] = []
    components.auth.callback_received_hook = lambda: callback_events.append("callback")
    components.release.callback_received()
    assert callback_events == ["callback"]

    secret_adapter = components.pkce.secret_store_factory(owner.current)
    assert isinstance(secret_adapter, SyncSecretStoreAdapter)
    await secret_adapter.set_secret(OPENROUTER_BYOK_API_KEY_SECRET, "secret")
    assert OPENROUTER_BYOK_API_KEY_SECRET == "openrouter_api_key"
    assert store.get(OPENROUTER_BYOK_API_KEY_SECRET) == "secret"
    assert secret_store_calls == [
        (owner.current.secrets, owner.path),
        (owner.current.secrets, owner.path),
        (owner.current.secrets, owner.path),
    ]


@pytest.mark.asyncio
async def test_managed_account_warmup_and_teardown_own_gemma_lifecycle(
    tmp_path,
) -> None:
    owner = compose_settings_owner(tmp_path / "settings.json")
    settings = AppSettings()
    settings.translation.model = TranslationModel.MANAGED_GEMMA
    settings.translation.connection = TranslationConnection.GPU
    owner.current = materialize_translation_settings(settings)
    events: list[str] = []

    class ManagedGemma:
        async def prepare(self, _selection: object) -> object:
            events.append("prepare")
            return object()

        async def deactivate(self, *, linger: bool = False) -> None:
            events.append(("deactivate", linger))

    components = compose_managed_account(
        config_path=owner.path,
        settings=owner,
        provider_settings=cast(ProviderSettingsOwner, object()),
        provider_runtime=cast(ProviderRuntimeOwner, object()),
        verifier=cast(ProviderVerifierPort, object()),
        results=SettingsTransactionResultOwner(),
        runtime=ManagedTranslationRuntimeAccess(
            llm_runtime_provider=lambda: None,
            context_provider=lambda: None,
            translation_runtime_configuration_provider=lambda: None,
            rebuild_llm=lambda: pytest.fail("runtime rebuild must not run"),
        ),
        ingress_provider=lambda: False,
        pending_sink=lambda _pending: None,
        usage_view_sink=lambda _state: None,
        dashboard_sink=lambda _enabled: None,
        message_sink=lambda _key, _kwargs: None,
        qq_dialog_sink=lambda: None,
        founder_dialog=lambda: False,
        failure_route=lambda _source: None,
        log_basic=lambda _message: None,
        log_detailed=lambda _message: None,
        log_error=lambda _message: None,
        basic_warning_sink=lambda _message: None,
        detailed_warning_sink=lambda _message, _exception: None,
        managed_gemma=ManagedGemma(),
    )

    await components.translation.warmup()
    await components.translation.teardown()

    assert events == ["prepare", ("deactivate", True)]


@pytest.mark.asyncio
async def test_release_runtime_retains_failed_replacement_cleanup_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    diagnostics: list[tuple[str, BaseException]] = []

    class Service:
        def __init__(self, *, fail_first_close: bool) -> None:
            self.fail_first_close = fail_first_close
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.fail_first_close and self.close_calls == 1:
                raise RuntimeError("retired close failed")

    previous = Service(fail_first_close=True)
    replacement = Service(fail_first_close=False)
    runtime = ManagedOpenRouterReleaseRuntime(
        settings=compose_settings_owner(tmp_path / "settings.json"),
        config_path=tmp_path / "settings.json",
        callback_received=lambda: None,
        diagnostic_sink=lambda message, exception: diagnostics.append((message, exception)),
        service=cast(ManagedOpenRouterReleaseService, previous),
    )

    def create_replacement(
        _runtime: ManagedOpenRouterReleaseRuntime,
        *,
        secrets: object,
    ) -> tuple[ManagedOpenRouterReleaseService, None]:
        _ = secrets
        return cast(ManagedOpenRouterReleaseService, replacement), None

    monkeypatch.setattr(
        ManagedOpenRouterReleaseRuntime,
        "_create",
        create_replacement,
    )

    with pytest.raises(RuntimeError, match="retired close failed"):
        await runtime.rebuild(secrets=SecretStore())

    assert runtime.service is replacement
    assert previous.close_calls == 1
    assert diagnostics[0][0].endswith("close failed")

    await runtime.retry_retired()
    await runtime.close()

    assert previous.close_calls == 2
    assert replacement.close_calls == 1
    assert runtime.service is None
