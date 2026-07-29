from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.provider_secret_change import (
    ProviderSecretChangeExecution,
    ProviderSecretChangeOwner,
    ProviderSecretChangeRequest,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSettingsTransaction,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.messages import TransactionResult

ProviderSecretStoreFactory = Callable[[AppSettings], SecretStorePort]
ProviderActiveSecretProvider = Callable[[AppSettings, str], str | None]
ProviderSettingsSaveFailureSink = Callable[[str], None]


def provider_verification_context(
    settings: AppSettings | None,
    provider: str,
    *,
    low_latency: bool,
) -> dict[str, object]:
    if settings is None:
        return {}
    if provider == "google":
        return {"model": settings.gemini.llm_model.value}
    if provider == "cerebras":
        return {"model": settings.cerebras.llm_model.value}
    if provider in {"alibaba_beijing", "alibaba_singapore"}:
        return {
            "base_url": (
                "https://dashscope.aliyuncs.com/api/v1"
                if provider == "alibaba_beijing"
                else "https://dashscope-intl.aliyuncs.com/api/v1"
            ),
            "model": settings.qwen.llm_model.value,
            "low_latency": low_latency,
        }
    return {}


@dataclass(slots=True)
class ProviderSettingsOwner:
    settings: SettingsOwner
    binding: ProviderVerificationBindingOwner
    secret_store_factory: ProviderSecretStoreFactory
    active_secret_provider: ProviderActiveSecretProvider
    save_failure_sink: ProviderSettingsSaveFailureSink | None = None
    secret_change: ProviderSecretChangeOwner = field(default_factory=ProviderSecretChangeOwner)
    last_result: TransactionResult | None = None

    def verification_binding(
        self,
        provider: str,
        key: str,
        *,
        flow: str,
        context_values: dict[str, object] | None = None,
    ) -> ProviderVerificationBinding:
        return self.binding.binding(
            provider,
            key,
            flow=flow,
            context_values=context_values,
        )

    def persist_verification(self, provider: str, key: str, success: bool) -> None:
        current = self._current()
        binding = (
            self.verification_binding(
                provider,
                key,
                flow="settings_api_key_verification",
            )
            if success
            else None
        )
        active_secret = (
            self.active_secret_provider(current, binding.secret_key)
            if binding is not None
            else None
        )
        self.settings.persist_provider_verification(
            provider=provider,
            key=key,
            success=success,
            binding=binding,
            active_secret=active_secret,
        )

    async def change_secret(self, secret_key: str, value: str) -> bool:
        return await self.secret_change.change(
            lambda: self._secret_change_execution(secret_key, value)
        )

    def _secret_change_execution(
        self,
        secret_key: str,
        value: str,
    ) -> ProviderSecretChangeExecution:
        current = self._current()
        provider = self.binding.provider_for_secret_key(secret_key)
        updated = copy.deepcopy(current)
        setattr(updated.api_key_verified, provider, False)
        repository = self.settings.create_legacy_patch_repository(
            base_settings=current,
            committed_settings=updated,
            surface="provider_secret_change",
            save_failure_sink=self.save_failure_sink,
        )
        transaction = SecretSettingsTransaction(
            secret_store=self.secret_store_factory(current),
            settings_repository=repository,
        )
        return ProviderSecretChangeExecution(
            transaction=transaction,
            request=ProviderSecretChangeRequest(
                provider=provider,
                secret_key=secret_key,
                secret_value=value,
                settings_values=self.settings.legacy_snapshot_values(updated),
            ),
            result_handler=lambda result, succeeded: self._apply_secret_change_result(
                repository.committed_settings,
                result,
                succeeded,
            ),
        )

    def _apply_secret_change_result(
        self,
        committed_settings: AppSettings,
        result: TransactionResult,
        succeeded: bool,
    ) -> None:
        self.last_result = result
        if not succeeded:
            return
        self.settings.current = committed_settings
        self.settings.remember_projection(committed_settings)
        self.settings.complete()

    def _current(self) -> AppSettings:
        if self.settings.current is None:
            raise RuntimeError("settings owner has no compatibility settings")
        return self.settings.current


__all__ = [
    "ProviderActiveSecretProvider",
    "ProviderSecretStoreFactory",
    "ProviderSettingsSaveFailureSink",
    "ProviderSettingsOwner",
    "provider_verification_context",
]
