from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.app.ports.settings_view import OpenRouterPkceTarget
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.provider_runtime_apply import (
    ProviderRuntimeApplyAdapter,
    ProviderRuntimeOwner,
    _runtime_apply_result_as_degraded_transaction,
)
from puripuly_heart.app.services.provider_settings import ProviderSettingsOwner
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSetRequest,
    SecretSettingsTransaction,
)
from puripuly_heart.app.services.settings_application import materialize_provider_apply_intent
from puripuly_heart.app.services.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.config.llm_profiles import profile_for_alias
from puripuly_heart.config.provider_values import (
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TransactionResult,
)
from puripuly_heart.core.openrouter_credentials import OPENROUTER_BYOK_API_KEY_SECRET
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult
from puripuly_heart.core.runtime.oauth import OAuthRuntime


@dataclass(slots=True)
class OpenRouterPkceFlowOwner:
    client_factory: Callable[[], object]
    runtime_factory: Callable[[], OAuthRuntime] = OAuthRuntime
    _runtime: OAuthRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _active_client: object | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def runtime(self) -> OAuthRuntime | None:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: OAuthRuntime | None) -> None:
        self._runtime = runtime

    @property
    def active_client(self) -> object | None:
        return self._active_client

    @active_client.setter
    def active_client(self, client: object | None) -> None:
        self._active_client = client

    def get_runtime(self) -> OAuthRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory()
        return self._runtime

    async def run_flow(self) -> OpenRouterPKCEExchangeResult:
        client = self.client_factory()
        self._active_client = client
        try:
            return await self.get_runtime().run_openrouter_pkce_flow(client)
        finally:
            if self._active_client is client:
                self._active_client = None

    def reopen_authorization_url(self) -> bool:
        runtime = self._runtime
        if runtime is not None and runtime.reopen_openrouter_pkce_authorization_url():
            return True
        client = self._active_client
        if client is None:
            return False
        reopen = getattr(client, "reopen_authorization_url", None)
        return bool(reopen()) if callable(reopen) else False

    async def close(self) -> None:
        runtime = self._runtime
        try:
            if runtime is not None:
                await runtime.close()
        finally:
            self._active_client = None


@dataclass(slots=True)
class OpenRouterPkceApplicationOwner:
    flow: OpenRouterPkceFlowOwner
    verifier: ProviderVerifierPort
    settings: SettingsOwner
    provider_settings: ProviderSettingsOwner
    provider_runtime: ProviderRuntimeOwner
    secret_store_factory: Callable[[object], SecretStorePort]
    failure_message_sink: Callable[[str], None]
    failure_diagnostics_sink: Callable[[str], None]
    failure_route: Callable[[str], None]
    results: SettingsTransactionResultOwner

    async def connect(
        self,
        *,
        target: OpenRouterPkceTarget,
        launch_source: str,
    ) -> bool:
        current = self.settings.current
        if current is None:
            return False
        selection_alias = target.selection_alias
        profile = profile_for_alias(selection_alias.value)
        if profile.openrouter_source != OpenRouterCredentialSource.BYOK.value:
            raise ValueError("PKCE connection requires a BYOK OpenRouter alias")
        if profile.openrouter_model is None:
            raise ValueError("PKCE connection requires a BYOK OpenRouter model")

        try:
            result = await self.flow.run_flow()
        except Exception:
            self._fail(
                launch_source,
                "OpenRouter PKCE flow failed",
            )
            return False

        try:
            verified = await self.verifier.verify_api_key("openrouter", result.api_key)
        except Exception:
            verified = False
        if not verified:
            self._fail(
                launch_source,
                "OpenRouter PKCE key verification failed",
            )
            return False

        current = self.settings.current
        if current is None:
            return False
        updated = materialize_provider_apply_intent(
            current,
            target.provider_intent,
            materialize_translation=self.settings.materialize_translation,
        )
        updated.provider.llm = LLMProviderName.OPENROUTER
        updated.openrouter.selection_alias = OpenRouterSelectionAlias(profile.alias)
        updated.openrouter.selected_source = OpenRouterCredentialSource.BYOK
        updated.openrouter.llm_model = OpenRouterLLMModel(profile.openrouter_model)
        updated.api_key_verified.openrouter = True
        if target.system_prompt is not None:
            updated.system_prompt = target.system_prompt
            updated.system_prompts = {}
        plan = self.provider_runtime.build_plan(
            updated,
            force_rebuild_llm=True,
        )
        settings_repository = self.settings.create_legacy_patch_repository(
            base_settings=current,
            committed_settings=updated,
            surface="openrouter_pkce",
            provider_verification_binding=self.provider_settings.verification_binding(
                "openrouter",
                result.api_key,
                flow="openrouter_pkce",
                context_values={"launch_source": launch_source},
            ),
            save_failure_sink=self.failure_diagnostics_sink,
        )
        transaction = SecretSettingsTransaction(
            secret_store=self.secret_store_factory(current),
            settings_repository=settings_repository,
        )
        runtime_apply = ProviderRuntimeApplyAdapter(
            owner=self.provider_runtime,
            settings=updated,
            plan=plan,
            surface="openrouter_pkce",
            operation="openrouter_pkce_runtime_apply",
        )
        values = self.settings.legacy_snapshot_values(updated)
        scope = LifecycleScope("openrouter-pkce-commit")

        async def commit_and_apply() -> bool:
            try:
                commit_result = await transaction.set_provider_secret(
                    SecretSetRequest(
                        secret_key=OPENROUTER_BYOK_API_KEY_SECRET,
                        secret_value=result.api_key,
                        settings_values=values,
                        expected_settings_revision=None,
                        reason="openrouter_pkce",
                        correlation_id=None,
                    )
                )
                if (
                    commit_result.status
                    != TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
                ):
                    self.results.set(commit_result)
                    self._fail(
                        launch_source,
                        "OpenRouter PKCE settings commit failed",
                    )
                    return False

                runtime_result = await runtime_apply.apply_runtime(
                    RuntimeApplyRequest(
                        settings_values=values,
                        reason="openrouter_pkce",
                        correlation_id=None,
                    )
                )
                if runtime_result.status == RUNTIME_APPLY_STATUS_APPLIED:
                    result_value = TransactionResult(
                        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                        message=runtime_result.message,
                        diagnostics=runtime_result.diagnostics,
                    )
                else:
                    result_value = _runtime_apply_result_as_degraded_transaction(runtime_result)
                self.results.set(result_value)
                return True
            finally:
                if settings_repository.commit_succeeded:
                    self.settings.complete()
                else:
                    self.settings.rollback()

        operation = start_lifecycle_task(
            scope,
            commit_and_apply(),
            name="transaction",
        )
        cancelled = False
        try:
            try:
                succeeded = await asyncio.shield(operation)
            except asyncio.CancelledError:
                cancelled = True
                succeeded = await operation
        finally:
            await scope.close()
        if cancelled:
            raise asyncio.CancelledError
        return succeeded

    def _fail(self, launch_source: str, diagnostics: str) -> None:
        self.failure_message_sink("openrouter.pkce.failed")
        self.failure_diagnostics_sink(diagnostics)
        self.failure_route(launch_source)


__all__ = [
    "OpenRouterPkceApplicationOwner",
    "OpenRouterPkceFlowOwner",
]
