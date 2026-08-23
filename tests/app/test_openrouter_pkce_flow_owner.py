from __future__ import annotations

import asyncio
import copy
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from puripuly_heart.app.services.provider_runtime_apply import (
    ProviderRuntimeApplyPlan,
)
from puripuly_heart.app.services.provider_settings import (
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.settings_application import settings_view_surface_snapshots
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.settings_view import (
    LocalLlmBaseUrlEdit,
    OpenRouterPkceTarget,
    ProviderApplyIntent,
    TranslationSelectionEdit,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.openrouter_pkce_flow import (
    OpenRouterPkceApplicationOwner,
    OpenRouterPkceFlowOwner,
)
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
)
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY


class MemorySecretStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


class AppliedProviderRuntime:
    def __init__(self, settings: SettingsOwner) -> None:
        self.settings = settings
        self.applied: list[AppSettings] = []
        self.cancel = False

    def build_plan(
        self,
        _settings: AppSettings,
        *,
        force_rebuild_llm: bool,
    ) -> ProviderRuntimeApplyPlan:
        assert force_rebuild_llm is True
        return ProviderRuntimeApplyPlan(
            should_rebuild_llm=True,
            should_refresh_peer=False,
            should_refresh_self_stt=False,
        )

    async def apply(
        self,
        settings: AppSettings,
        _plan: ProviderRuntimeApplyPlan,
    ) -> None:
        if self.cancel:
            raise asyncio.CancelledError
        self.settings.current = settings
        self.applied.append(copy.deepcopy(settings))

    def unavailable_result(
        self,
        _settings: AppSettings,
        _plan: ProviderRuntimeApplyPlan,
        **_kwargs: object,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_owner_runs_tracks_reopens_and_clears_active_pkce_flow() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.reopen_calls = 0

        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            entered.set()
            await release.wait()
            return OpenRouterPKCEExchangeResult(api_key="key", user_id="user")

        def reopen_authorization_url(self) -> bool:
            self.reopen_calls += 1
            return True

    client = Client()
    owner = OpenRouterPkceFlowOwner(client_factory=lambda: client)

    task = asyncio.create_task(owner.run_flow())
    await entered.wait()

    assert owner.active_client is client
    assert owner.reopen_authorization_url() is True
    assert client.reopen_calls == 1

    release.set()
    result = await task

    assert result == OpenRouterPKCEExchangeResult(api_key="key", user_id="user")
    assert owner.active_client is None
    assert owner.get_runtime().active_task_names == ()


def test_owner_reopens_compatibility_client_without_runtime() -> None:
    reopen_calls = 0

    def reopen() -> bool:
        nonlocal reopen_calls
        reopen_calls += 1
        return True

    owner = OpenRouterPkceFlowOwner(client_factory=lambda: object())
    owner.active_client = type("Client", (), {"reopen_authorization_url": staticmethod(reopen)})()

    assert owner.reopen_authorization_url() is True
    assert reopen_calls == 1


@pytest.mark.asyncio
async def test_owner_close_cancels_active_flow_and_clears_client() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class Client:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    owner = OpenRouterPkceFlowOwner(client_factory=Client)
    task = asyncio.create_task(owner.run_flow())
    await entered.wait()

    await owner.close()
    await asyncio.gather(task, return_exceptions=True)

    assert cancelled.is_set()
    assert owner.active_client is None
    assert owner.get_runtime().is_closed is True
    assert task.cancelled()


@pytest.mark.asyncio
async def test_owner_close_clears_client_when_runtime_close_fails() -> None:
    class FailingRuntime:
        async def close(self) -> None:
            raise RuntimeError("close failed")

    owner = OpenRouterPkceFlowOwner(client_factory=lambda: object())
    owner.runtime = cast(OAuthRuntime, FailingRuntime())
    owner.active_client = object()

    with pytest.raises(RuntimeError, match="close failed"):
        await owner.close()

    assert owner.active_client is None


@pytest.mark.asyncio
async def test_application_owner_commits_verified_pkce_secret_settings_and_runtime(
    tmp_path: Path,
) -> None:
    current = AppSettings()
    persistence = SettingsVNextCanonicalPersistenceAdapter()
    settings = SettingsOwner(
        path=tmp_path / "settings.json",
        persistence=persistence,
        canonical=AppSettingsVNext(),
        current=current,
        authoritative=True,
        projection_snapshot=copy.deepcopy(current),
    )
    store = MemorySecretStore()
    provider_settings = ProviderSettingsOwner(
        settings=settings,
        binding=ProviderVerificationBindingOwner(
            context_provider=lambda provider: provider_verification_context(
                settings.current,
                provider,
                low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            )
        ),
        secret_store_factory=lambda _settings: SyncSecretStoreAdapter(store),
        active_secret_provider=lambda _settings, key: store.get(key),
    )
    runtime = AppliedProviderRuntime(settings)
    provider_snapshot, _general, _prompt, _overlay = settings_view_surface_snapshots(current)
    staged_translation = replace(
        provider_snapshot.translation,
        model=TranslationModel.GEMMA4,
        connection=TranslationConnection.OPENROUTER,
    )
    target = OpenRouterPkceTarget(
        selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
        provider_intent=ProviderApplyIntent(
            (
                TranslationSelectionEdit(staged_translation),
                LocalLlmBaseUrlEdit("http://staged.local:11434"),
            )
        ),
        system_prompt="PKCE prompt draft",
    )
    settings.current.ui.locale = "ko"
    results = SettingsTransactionResultOwner()

    class Flow:
        api_key = "sk-or-v1-user"

        async def run_flow(self) -> OpenRouterPKCEExchangeResult:
            return OpenRouterPKCEExchangeResult(
                api_key=self.api_key,
                user_id="user_123",
            )

    class Verifier:
        async def verify_api_key(self, provider: str, api_key: str) -> bool:
            return provider == "openrouter" and api_key.startswith("sk-or-v1-")

    flow = Flow()
    owner = OpenRouterPkceApplicationOwner(
        flow=cast(OpenRouterPkceFlowOwner, flow),
        verifier=Verifier(),
        settings=settings,
        provider_settings=provider_settings,
        provider_runtime=runtime,
        secret_store_factory=lambda _settings: SyncSecretStoreAdapter(store),
        failure_message_sink=lambda _message: None,
        failure_diagnostics_sink=lambda _message: None,
        failure_route=lambda _source: None,
        results=results,
    )

    assert await owner.connect(target=target, launch_source="settings") is True
    assert store.values["openrouter_api_key"] == "sk-or-v1-user"
    assert settings.current is not None
    assert settings.current.provider.llm == LLMProviderName.OPENROUTER
    assert settings.current.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_BYOK
    assert settings.current.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert settings.current.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert settings.current.api_key_verified.openrouter is True
    assert settings.current.system_prompt == "PKCE prompt draft"
    assert settings.current.translation.model == TranslationModel.GEMMA4
    assert settings.current.translation.connection == TranslationConnection.OPENROUTER
    assert settings.current.local_llm.base_url == "http://staged.local:11434"
    assert settings.current.ui.locale == "ko"
    assert len(runtime.applied) == 1
    assert results.current is not None
    assert results.current.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert settings.mutation_depth == 0
    assert settings.rollback_pending is False

    runtime.cancel = True
    with pytest.raises(asyncio.CancelledError):
        await owner.connect(target=target, launch_source="settings")

    assert settings.mutation_depth == 0
    assert settings.rollback_pending is False
    assert settings.current is not None
    assert settings.current.provider.llm == LLMProviderName.OPENROUTER

    runtime.cancel = False
    results.current = None
    flow.api_key = "sk-or-v1-replaced"
    persist_entered = threading.Event()
    persist_release = threading.Event()
    persist = persistence.persist

    def blocked_persist(path: Path, canonical: AppSettingsVNext) -> None:
        persist_entered.set()
        if not persist_release.wait(timeout=5):
            raise TimeoutError("settings persistence was not released")
        persist(path, canonical)

    persistence.persist = blocked_persist
    task = asyncio.create_task(owner.connect(target=target, launch_source="settings"))
    assert await asyncio.to_thread(persist_entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    persist_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.values["openrouter_api_key"] == "sk-or-v1-replaced"
    assert settings.mutation_depth == 0
    assert settings.rollback_pending is False
    assert results.current is not None
    assert results.current.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert settings.canonical is not None
    binding = provider_settings.verification_binding(
        "openrouter",
        "sk-or-v1-replaced",
        flow="openrouter_pkce",
        context_values={"launch_source": "settings"},
    )
    assert (
        settings.canonical.state.provider_verification.openrouter.secret_fingerprint
        == binding.secret_fingerprint
    )
    assert len(runtime.applied) == 2
