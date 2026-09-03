from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)

ProviderVerificationContextProvider = Callable[[str], Mapping[str, object]]

_SECRET_KEY_BY_PROVIDER = {
    "google": "google_api_key",
    "openrouter": "openrouter_api_key",
    "deepseek": "deepseek_api_key",
    "cerebras": "cerebras_api_key",
    "alibaba_beijing": "alibaba_api_key_beijing",
    "alibaba_singapore": "alibaba_api_key_singapore",
    "deepgram": "deepgram_api_key",
    "gemini_transcribe": "gemini_transcribe_api_key",
    "elevenlabs_scribe": "elevenlabs_scribe_api_key",
    "soniox": "soniox_api_key",
}
_PROVIDER_BY_SECRET_KEY = {
    secret_key: provider for provider, secret_key in _SECRET_KEY_BY_PROVIDER.items()
}


@dataclass(slots=True)
class ProviderVerificationBindingOwner:
    context_provider: ProviderVerificationContextProvider

    def binding(
        self,
        provider: str,
        key: str,
        *,
        flow: str,
        context_values: Mapping[str, object] | None = None,
    ) -> ProviderVerificationBinding:
        secret_key = _SECRET_KEY_BY_PROVIDER.get(provider)
        if secret_key is None:
            raise ValueError(f"unsupported provider verification binding: {provider}")
        context = {"flow": flow, **self.context(provider)}
        if context_values is not None:
            context.update(context_values)
        return ProviderVerificationBinding(
            provider=provider,
            secret_key=secret_key,
            secret_revision=None,
            secret_fingerprint=f"sha256:{hashlib.sha256(key.encode('utf-8')).hexdigest()}",
            verifier_context=context,
            verifier_evidence={"source": "provider_verifier"},
        )

    def selected_model(self, provider: str) -> str | None:
        model = self.context_provider(provider).get("model")
        return model if isinstance(model, str) else None

    def context(self, provider: str) -> Mapping[str, object]:
        return self.context_provider(provider)

    @staticmethod
    def provider_for_secret_key(secret_key: str) -> str:
        provider = _PROVIDER_BY_SECRET_KEY.get(secret_key)
        if provider is None:
            raise ValueError(f"unsupported provider secret key: {secret_key}")
        return provider


__all__ = ["ProviderVerificationBindingOwner"]
