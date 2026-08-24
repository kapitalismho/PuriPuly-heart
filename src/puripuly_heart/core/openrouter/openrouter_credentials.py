from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.config.settings import OpenRouterCredentialSource
from puripuly_heart.core.storage.secrets import SecretStore

OPENROUTER_BYOK_API_KEY_SECRET = "openrouter_api_key"
OPENROUTER_MANAGED_API_KEY_SECRET = "openrouter_managed_api_key"
OPENROUTER_MANAGED_QQ_API_KEY_SECRET = "openrouter_managed_qq_api_key"
OPENROUTER_MANAGED_USER_ID_SECRET = "openrouter_managed_user_id"
OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET = "openrouter_managed_user_installation_id"
OPENROUTER_MANAGED_USER_ID_MAX_LENGTH = 256
OPENROUTER_BYOK_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True, slots=True)
class OpenRouterCredentialRuntimeConfig:
    """Narrow read-only runtime DTO for OpenRouter credential resolution.

    ``selected_source`` is the OpenRouter credential source and
    ``installation_id`` is the persisted managed-identity installation id used
    to key cached managed user identifiers.
    """

    selected_source: OpenRouterCredentialSource
    installation_id: str | None
    managed_credential_kind: str = "standard"
    active_managed_credential_ref: str | None = None
    active_managed_expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class OpenRouterCredentialResolution:
    selected_source: OpenRouterCredentialSource
    api_key: str | None
    requires_managed_challenge: bool = False


class OpenRouterManagedRecoveryAction(str, Enum):
    STOP = "stop"
    RESTART_CHALLENGE = "restart_challenge"


@dataclass(frozen=True, slots=True)
class OpenRouterManagedRecoveryResult:
    action: OpenRouterManagedRecoveryAction
    reason: str
    selected_source: OpenRouterCredentialSource
    managed_availability: str | None = None


def resolve_openrouter_credentials(
    config: OpenRouterCredentialRuntimeConfig,
    *,
    secrets: SecretStore,
    request_intent: str | None = None,
) -> OpenRouterCredentialResolution:
    selected_source = config.selected_source
    if selected_source == OpenRouterCredentialSource.NONE:
        return OpenRouterCredentialResolution(selected_source=selected_source, api_key=None)

    if selected_source == OpenRouterCredentialSource.BYOK:
        return OpenRouterCredentialResolution(
            selected_source=selected_source,
            api_key=_get_byok_api_key(secrets),
        )

    managed_credential_kind = _normalize_managed_credential_kind(config.managed_credential_kind)
    managed_secret_key = _managed_api_key_secret_for_kind(managed_credential_kind)
    opposite_managed_secret_key = _opposite_managed_api_key_secret_for_kind(managed_credential_kind)
    managed_api_key = _normalize_secret(secrets.get(managed_secret_key))
    opposite_managed_api_key = _normalize_secret(secrets.get(opposite_managed_secret_key))
    if opposite_managed_api_key is not None:
        raise ValueError("OpenRouter managed local claim conflict")
    if managed_api_key is not None:
        return OpenRouterCredentialResolution(
            selected_source=selected_source,
            api_key=managed_api_key,
        )

    return OpenRouterCredentialResolution(
        selected_source=selected_source,
        api_key=None,
        requires_managed_challenge=(
            _is_trans_intent(request_intent)
            and managed_secret_key == OPENROUTER_MANAGED_API_KEY_SECRET
        ),
    )


def require_openrouter_execution_api_key(
    config: OpenRouterCredentialRuntimeConfig,
    *,
    secrets: SecretStore,
) -> str:
    resolution = resolve_openrouter_credentials(config, secrets=secrets)
    if resolution.api_key is not None:
        return resolution.api_key
    if resolution.selected_source == OpenRouterCredentialSource.NONE:
        raise ValueError("OpenRouter selected source must not be `none` for execution")
    if resolution.selected_source == OpenRouterCredentialSource.MANAGED:
        raise ValueError("OpenRouter managed key is unavailable for the selected source")
    raise ValueError(
        f"Missing secret `{OPENROUTER_BYOK_API_KEY_SECRET}` (or env var {OPENROUTER_BYOK_API_KEY_ENV})"
    )


def normalize_managed_openrouter_user_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > OPENROUTER_MANAGED_USER_ID_MAX_LENGTH:
        return None
    return normalized


def load_managed_openrouter_user_identifier(
    config: OpenRouterCredentialRuntimeConfig,
    *,
    secrets: SecretStore,
) -> str | None:
    current_installation_id = _normalize_secret(config.installation_id)
    if current_installation_id is None:
        return None

    try:
        cached_user_id = normalize_managed_openrouter_user_identifier(
            secrets.get(OPENROUTER_MANAGED_USER_ID_SECRET)
        )
        cached_installation_id = _normalize_secret(
            secrets.get(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET)
        )
    except Exception:
        return None

    if cached_user_id is None or cached_installation_id != current_installation_id:
        return None
    return cached_user_id


def best_effort_store_managed_openrouter_user_identifier(
    config: OpenRouterCredentialRuntimeConfig,
    *,
    secrets: SecretStore,
    openrouter_user_id: object,
) -> None:
    normalized_user_id = normalize_managed_openrouter_user_identifier(openrouter_user_id)
    current_installation_id = _normalize_secret(config.installation_id)
    if normalized_user_id is None or current_installation_id is None:
        return

    try:
        secrets.set(OPENROUTER_MANAGED_USER_ID_SECRET, normalized_user_id)
        secrets.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, current_installation_id)
    except Exception:
        best_effort_clear_managed_openrouter_user_identifier(secrets)


def best_effort_clear_managed_openrouter_user_identifier(secrets: SecretStore) -> None:
    for key in (
        OPENROUTER_MANAGED_USER_ID_SECRET,
        OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    ):
        try:
            secrets.delete(key)
        except Exception:
            pass


def clear_temporary_managed_release_state(state: ManagedIdentityStatePort) -> None:
    state.release_token = None
    state.release_token_expires_at = None
    state.verified_hardware_hash = None
    state.verified_hardware_hash_salt_version = None


def handle_managed_availability(
    state: ManagedIdentityStatePort,
    *,
    managed_availability: str,
    selected_source: OpenRouterCredentialSource,
) -> OpenRouterManagedRecoveryResult:
    normalized_managed_availability = _normalize_managed_availability(managed_availability)
    if normalized_managed_availability not in {"not_eligible", "unavailable"}:
        raise ValueError("unsupported managed availability")
    clear_temporary_managed_release_state(state)
    return OpenRouterManagedRecoveryResult(
        action=OpenRouterManagedRecoveryAction.STOP,
        reason=normalized_managed_availability,
        selected_source=selected_source,
        managed_availability=normalized_managed_availability,
    )


def handle_managed_release_error(
    state: ManagedIdentityStatePort,
    *,
    error_code: str,
    selected_source: OpenRouterCredentialSource,
) -> OpenRouterManagedRecoveryResult:
    normalized_error_code = _normalize_required_text(error_code)
    if normalized_error_code not in {"challenge_expired", "security_fail"}:
        raise ValueError("unsupported managed release error")
    clear_temporary_managed_release_state(state)
    return OpenRouterManagedRecoveryResult(
        action=OpenRouterManagedRecoveryAction.RESTART_CHALLENGE,
        reason=normalized_error_code,
        selected_source=selected_source,
    )


def _get_byok_api_key(secrets: SecretStore) -> str | None:
    stored_key = _normalize_secret(secrets.get(OPENROUTER_BYOK_API_KEY_SECRET))
    if stored_key is not None:
        return stored_key
    return _normalize_secret(os.getenv(OPENROUTER_BYOK_API_KEY_ENV))


def _managed_api_key_secret_for_kind(kind: object) -> str:
    if _normalize_managed_credential_kind(kind) == "qq":
        return OPENROUTER_MANAGED_QQ_API_KEY_SECRET
    return OPENROUTER_MANAGED_API_KEY_SECRET


def _opposite_managed_api_key_secret_for_kind(kind: object) -> str:
    if _normalize_managed_credential_kind(kind) == "qq":
        return OPENROUTER_MANAGED_API_KEY_SECRET
    return OPENROUTER_MANAGED_QQ_API_KEY_SECRET


def _normalize_managed_credential_kind(kind: object) -> str:
    if isinstance(kind, str) and kind.strip() == "qq":
        return "qq"
    return "standard"


def _normalize_managed_availability(value: str) -> str:
    return _normalize_required_text(value)


def _normalize_required_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must be non-empty")
    return normalized


def _normalize_secret(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _is_trans_intent(request_intent: str | None) -> bool:
    return isinstance(request_intent, str) and request_intent.strip().upper() == "TRANS"
