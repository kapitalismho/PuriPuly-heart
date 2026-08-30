from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from puripuly_heart.config.llm_profiles import get_openrouter_llm_profile
from puripuly_heart.config.provider_values import OpenRouterCredentialSource
from puripuly_heart.config.settings_vnext import compat, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel


@dataclass
class OpenRouterBag:
    selected_source: object = OpenRouterCredentialSource.MANAGED
    selection_alias: object | None = None
    llm_model: object | None = None


@dataclass
class TranslationBag:
    model: object = TranslationModel.GEMMA4_26B_31B
    connection: object = TranslationConnection.MANAGED


@dataclass
class ManagedIdentityBag:
    installation_id: str = ""
    release_token: str | None = None
    release_token_expires_at: str | None = None
    verified_hardware_hash: str | None = None
    verified_hardware_hash_salt_version: int | None = None
    active_managed_credential_ref: str | None = None
    active_managed_expires_at: str | None = None
    founder_letter_seen_credential_ref: str | None = None
    referral_id: str | None = None
    local_managed_claim_sources: tuple[str, ...] = ()
    pending_delivery_ack_source: str | None = None
    pending_delivery_ack_delivery_id: str | None = None
    pending_delivery_ack_managed_credential_ref: str | None = None
    pending_delivery_ack_expires_at: str | None = None


@dataclass
class ReleaseSettings:
    openrouter: OpenRouterBag = field(default_factory=OpenRouterBag)
    translation: TranslationBag = field(default_factory=TranslationBag)
    managed_identity: ManagedIdentityBag = field(default_factory=ManagedIdentityBag)


def _value(item: object) -> str | None:
    if item is None:
        return None
    return str(getattr(item, "value", item))


def canonical_settings(settings: ReleaseSettings) -> AppSettingsVNext:
    current = AppSettingsVNext()
    source_value = _value(settings.openrouter.selected_source) or "managed"
    model_value = _value(settings.translation.model) or current.intent.translation.model
    connection_value = (
        _value(settings.translation.connection) or current.intent.translation.connection
    )
    translation = replace(
        current.intent.translation,
        model=model_value,
        connection=connection_value,
        openrouter_selected_source=source_value,
    )
    alias_value = _value(settings.openrouter.selection_alias)
    if alias_value and alias_value != "none":
        profile = get_openrouter_llm_profile(alias_value)
        if profile is not None:
            translation = replace(
                translation,
                openrouter_selection_alias=alias_value,
                openrouter_model=profile.openrouter_model,
            )
    identity = settings.managed_identity
    return replace(
        current,
        intent=replace(current.intent, translation=translation),
        state=replace(
            current.state,
            managed_connection=replace(
                current.state.managed_connection,
                **asdict(identity),
            ),
        ),
    )


def managed_identity_payload(settings: ReleaseSettings) -> dict[str, object]:
    return asdict(settings.managed_identity)


def persist_release_settings(path: Path, settings: ReleaseSettings) -> None:
    result = compat.save_vnext_settings(path, canonical_settings(settings))
    if not result.ok:
        raise RuntimeError(result.error.message if result.error is not None else result.status)


def load_release_settings(path: Path) -> ReleaseSettings:
    result = compat.load_vnext_settings(path)
    if result.settings is None:
        raise RuntimeError(result.error.message if result.error is not None else result.status)
    identity = result.settings.state.managed_connection
    translation = result.settings.intent.translation
    return ReleaseSettings(
        openrouter=OpenRouterBag(
            selected_source=OpenRouterCredentialSource(translation.openrouter_selected_source),
            selection_alias=translation.openrouter_selection_alias,
            llm_model=translation.openrouter_model,
        ),
        translation=TranslationBag(
            model=translation.model,
            connection=translation.connection,
        ),
        managed_identity=ManagedIdentityBag(**asdict(identity)),
    )


def serialize_release_settings(settings: ReleaseSettings) -> dict[str, object]:
    return serialization.to_dict(canonical_settings(settings))
