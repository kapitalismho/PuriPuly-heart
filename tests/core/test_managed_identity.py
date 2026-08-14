from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_SECRET,
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    OpenRouterCredentialRuntimeConfig,
    load_managed_openrouter_user_identifier,
)

from puripuly_heart.app.wiring import ManagedIdentityStateAdapter
from puripuly_heart.config.settings import (
    AppSettings,
    load_settings,
    save_settings,
    to_dict,
)
from puripuly_heart.core.managed_identity import (
    MANAGED_DEVICE_PRIVATE_KEY_SECRET,
    MANAGED_DEVICE_PUBLIC_KEY_SECRET,
    canonical_discord_issue_payload,
    canonical_issue_payload,
    canonical_status_payload,
    canonical_verify_payload,
    decode_base64url,
    ensure_managed_identity_bundle,
    regenerate_managed_identity_bundle,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore


def _persisted_settings_writer(path: Path):
    def persist(settings: AppSettings) -> None:
        save_settings(path, settings)

    return persist


def _raise_persist_failed(_: AppSettings) -> None:
    raise RuntimeError("persist failed")


def _managed_state(
    settings: AppSettings,
    persist=None,
) -> ManagedIdentityStateAdapter:
    return ManagedIdentityStateAdapter(settings, persist if persist is not None else lambda _: None)


def _credential_config(settings: AppSettings) -> OpenRouterCredentialRuntimeConfig:
    return OpenRouterCredentialRuntimeConfig(
        selected_source=settings.openrouter.selected_source,
        installation_id=settings.managed_identity.installation_id,
    )


def test_ensure_managed_identity_bundle_generates_uuid7_and_keeps_secret_boundary() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()
    persisted_snapshots: list[dict[str, object]] = []

    def persist(updated: AppSettings) -> None:
        persisted_snapshots.append(to_dict(updated))

    bundle = ensure_managed_identity_bundle(_managed_state(settings, persist), store)
    persisted = to_dict(settings)

    assert uuid.UUID(bundle.installation_id).version == 7
    assert settings.managed_identity.installation_id == bundle.installation_id
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    private_key = store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET)
    assert private_key is not None
    assert len(decode_base64url(private_key)) == 32
    assert bundle.device_public_key == base64.urlsafe_b64encode(
        Ed25519PublicKey.from_public_bytes(decode_base64url(bundle.device_public_key)).public_bytes(
            Encoding.Raw,
            PublicFormat.Raw,
        )
    ).decode("ascii").rstrip("=")
    assert persisted_snapshots == [persisted]
    assert persisted["managed_identity"] == {
        "active_managed_credential_ref": None,
        "active_managed_expires_at": None,
        "founder_letter_seen_credential_ref": None,
        "installation_id": bundle.installation_id,
        "local_managed_claim_sources": [],
        "pending_delivery_ack_delivery_id": None,
        "pending_delivery_ack_expires_at": None,
        "pending_delivery_ack_managed_credential_ref": None,
        "pending_delivery_ack_source": None,
        "referral_id": None,
        "release_token": None,
        "release_token_expires_at": None,
        "verified_hardware_hash": None,
        "verified_hardware_hash_salt_version": None,
    }


def test_ensure_managed_identity_bundle_reuses_existing_valid_bundle() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    private_before = store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET)

    second = ensure_managed_identity_bundle(_managed_state(settings), store)

    assert second.installation_id == first.installation_id
    assert second.device_public_key == first.device_public_key
    assert store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET) == private_before


def test_ensure_managed_identity_bundle_generates_when_port_persistence_is_available() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()
    persist_calls: list[AppSettings] = []

    bundle = ensure_managed_identity_bundle(_managed_state(settings, persist_calls.append), store)

    assert uuid.UUID(bundle.installation_id).version == 7
    assert settings.managed_identity.installation_id == bundle.installation_id
    assert len(persist_calls) == 1
    assert persist_calls[0] is settings


def test_missing_installation_id_regenerates_bundle_and_clears_release_state() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    private_before = store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET)
    settings.managed_identity.installation_id = ""
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    settings.managed_identity.verified_hardware_hash = "hardware-hash-1"
    settings.managed_identity.verified_hardware_hash_salt_version = 7

    second = ensure_managed_identity_bundle(_managed_state(settings), store)

    assert second.installation_id != first.installation_id
    assert second.device_public_key != first.device_public_key
    assert store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET) != private_before
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None


def test_regenerate_managed_identity_bundle_rotates_bundle_and_clears_managed_release_state() -> (
    None
):
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    settings.managed_identity.verified_hardware_hash = "hardware-hash-1"
    settings.managed_identity.verified_hardware_hash_salt_version = 7
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")
    store.set(OPENROUTER_MANAGED_USER_ID_SECRET, "user-123")
    store.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, first.installation_id)

    second = regenerate_managed_identity_bundle(_managed_state(settings), store)

    assert second.installation_id != first.installation_id
    assert second.device_public_key != first.device_public_key
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None
    assert store.get(OPENROUTER_BYOK_API_KEY_SECRET) == "byok-key"
    assert store.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None
    assert store.get(OPENROUTER_MANAGED_USER_ID_SECRET) is None
    assert store.get(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET) is None
    assert (
        load_managed_openrouter_user_identifier(_credential_config(settings), secrets=store) is None
    )


def test_corrupted_secret_material_regenerates_bundle_and_clears_release_state() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")
    store.set(
        MANAGED_DEVICE_PRIVATE_KEY_SECRET,
        base64.urlsafe_b64encode(b"\x01" * 32).decode("ascii").rstrip("="),
    )

    second = ensure_managed_identity_bundle(_managed_state(settings), store)

    assert second.installation_id != first.installation_id
    assert second.device_public_key != first.device_public_key
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert store.get(OPENROUTER_BYOK_API_KEY_SECRET) == "byok-key"
    assert store.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None


def test_broker_public_key_mismatch_regenerates_bundle_atomically() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    second = ensure_managed_identity_bundle(
        _managed_state(settings),
        store,
        broker_device_public_key="broker-mismatch-public-key",
    )

    assert second.installation_id != first.installation_id
    assert second.device_public_key != first.device_public_key
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert store.get(OPENROUTER_BYOK_API_KEY_SECRET) == "byok-key"
    assert store.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None


def test_broker_installation_id_mismatch_regenerates_bundle_atomically() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    second = ensure_managed_identity_bundle(
        _managed_state(settings),
        store,
        broker_installation_id="01961ad7-a7c1-7000-8000-aaaaaaaaaaaa",
    )

    assert second.installation_id != first.installation_id
    assert second.device_public_key != first.device_public_key
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert store.get(OPENROUTER_BYOK_API_KEY_SECRET) == "byok-key"
    assert store.get(OPENROUTER_MANAGED_API_KEY_SECRET) is None


def test_mixed_state_secret_overwrite_does_not_reuse_old_installation_id() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(_managed_state(settings), store)
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"

    overwritten_private_key = Ed25519PrivateKey.generate()
    overwritten_private_value = (
        base64.urlsafe_b64encode(
            overwritten_private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        )
        .decode("ascii")
        .rstrip("=")
    )
    overwritten_public_value = (
        base64.urlsafe_b64encode(
            overwritten_private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        )
        .decode("ascii")
        .rstrip("=")
    )
    store.set(MANAGED_DEVICE_PRIVATE_KEY_SECRET, overwritten_private_value)
    store.set(MANAGED_DEVICE_PUBLIC_KEY_SECRET, overwritten_public_value)

    second = ensure_managed_identity_bundle(_managed_state(settings), store)

    assert second.installation_id != first.installation_id
    assert second.device_public_key != overwritten_public_value
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None


def test_regeneration_rolls_back_secret_and_settings_when_persist_fails(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    store = InMemorySecretStore()

    first = ensure_managed_identity_bundle(
        _managed_state(settings, _persisted_settings_writer(path)),
        store,
    )
    old_private_key = store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET)
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")
    store.set(OPENROUTER_MANAGED_USER_ID_SECRET, "user-123")
    store.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, first.installation_id)
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    save_settings(path, settings)
    persisted_before = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="persist failed"):
        ensure_managed_identity_bundle(
            _managed_state(settings, _raise_persist_failed),
            store,
            broker_device_public_key="broker-mismatch-public-key",
        )

    assert settings.managed_identity.installation_id == first.installation_id
    assert settings.managed_identity.release_token == "release-1"
    assert settings.managed_identity.release_token_expires_at == "2026-04-08T06:00:45.000Z"
    assert store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET) == old_private_key
    assert store.get(OPENROUTER_MANAGED_API_KEY_SECRET) == "managed-key"
    assert store.get(OPENROUTER_MANAGED_USER_ID_SECRET) == "user-123"
    assert store.get(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET) == first.installation_id
    assert (
        load_managed_openrouter_user_identifier(_credential_config(settings), secrets=store)
        == "user-123"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == persisted_before
    restored = load_settings(path)
    restored_bundle = ensure_managed_identity_bundle(_managed_state(restored), store)
    assert restored_bundle.installation_id == first.installation_id
    assert restored_bundle.device_public_key == first.device_public_key
    assert (
        load_managed_openrouter_user_identifier(_credential_config(restored), secrets=store)
        == "user-123"
    )


@pytest.mark.parametrize(
    ("budget_usd", "expected_budget_line"),
    [
        (1e21, "1e+21"),
        (1.23e21, "1.23e+21"),
        (1e20, "100000000000000000000"),
        (1e-7, "1e-7"),
        (1e-6, "0.000001"),
    ],
)
def test_issue_payload_budget_encoding_matches_js_number_stringification(
    budget_usd: float,
    expected_budget_line: str,
) -> None:
    payload = canonical_issue_payload(
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        device_public_key="device-public-key",
        release_token="release-1",
        reason="llm_start",
        hardware_hash="hardware-hash",
        budget_usd=budget_usd,
        model="google/gemma-4-26b-a4b-it",
        signed_at="2026-04-08T06:00:45.000Z",
    )

    assert payload.splitlines()[5] == expected_budget_line.encode("utf-8")


def test_issue_payload_matches_landed_broker_field_order() -> None:
    payload = canonical_issue_payload(
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        device_public_key="device-public-key",
        release_token="release-1",
        hardware_hash="hardware-hash",
        reason="llm_start",
        budget_usd=1.0,
        model="google/gemma-4-26b-a4b-it",
        signed_at="2026-04-08T06:00:45.000Z",
    )

    assert payload == (
        "\n".join(
            [
                "01961ad7-a7c1-7000-8000-0123456789ab",
                "device-public-key",
                "release-1",
                "hardware-hash",
                "llm_start",
                "1",
                "google/gemma-4-26b-a4b-it",
                "2026-04-08T06:00:45.000Z",
            ]
        )
    ).encode("utf-8")


def test_discord_issue_payload_matches_landed_broker_field_order_and_hashes_code() -> None:
    code_hash = (
        base64.urlsafe_b64encode(hashlib.sha256("discord-oauth-code".encode("utf-8")).digest())
        .decode("ascii")
        .rstrip("=")
    )

    payload = canonical_discord_issue_payload(
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        device_public_key="device-public-key",
        state="discord-state-1",
        code="discord-oauth-code",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        hardware_hash="hardware-hash",
        hardware_hash_salt_version=7,
        app_version="2.0.0",
        reason="llm_start",
        budget_usd=0.07,
        model="google/gemma-4-26b-a4b-it",
        issue_nonce="issue-nonce-1",
        signed_at="2026-04-30T06:00:30.000Z",
    )

    assert payload == (
        "\n".join(
            [
                "POST",
                "/v1/providers/openrouter/discord/issue",
                "01961ad7-a7c1-7000-8000-0123456789ab",
                "device-public-key",
                "discord-state-1",
                code_hash,
                "http://127.0.0.1:62187/discord/callback",
                "hardware-hash",
                "7",
                "2.0.0",
                "llm_start",
                "0.07",
                "google/gemma-4-26b-a4b-it",
                "issue-nonce-1",
                "2026-04-30T06:00:30.000Z",
            ]
        )
    ).encode("utf-8")


def test_bundle_signing_matches_canonical_payload_contracts() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()
    bundle = ensure_managed_identity_bundle(_managed_state(settings), store)
    public_key = Ed25519PublicKey.from_public_bytes(decode_base64url(bundle.device_public_key))

    verify_payload = canonical_verify_payload(
        installation_id=bundle.installation_id,
        device_public_key=bundle.device_public_key,
        challenge="challenge-123",
        challenge_expires_at="2026-04-08T06:00:00.000Z",
        hardware_hash="hardware-hash",
        app_version="2.0.0",
        signed_at="2026-04-08T06:00:30.000Z",
    )
    assert verify_payload == (
        "\n".join(
            [
                bundle.installation_id,
                bundle.device_public_key,
                "challenge-123",
                "2026-04-08T06:00:00.000Z",
                "hardware-hash",
                "2.0.0",
                "2026-04-08T06:00:30.000Z",
            ]
        )
    ).encode("utf-8")
    signed_verify = bundle.sign_verify_request(
        challenge="challenge-123",
        challenge_expires_at="2026-04-08T06:00:00.000Z",
        hardware_hash="hardware-hash",
        app_version="2.0.0",
        signed_at="2026-04-08T06:00:30.000Z",
    )
    assert re.fullmatch(r"[A-Za-z0-9_-]+", signed_verify["signature"])
    public_key.verify(decode_base64url(signed_verify["signature"]), verify_payload)

    status_payload = canonical_status_payload(
        installation_id=bundle.installation_id,
        timestamp="2026-04-08T06:01:00.000Z",
    )
    assert status_payload == (f"{bundle.installation_id}\n2026-04-08T06:01:00.000Z").encode("utf-8")
    signed_status = bundle.sign_status_request(timestamp="2026-04-08T06:01:00.000Z")
    public_key.verify(decode_base64url(signed_status["signature"]), status_payload)

    issue_payload = canonical_issue_payload(
        installation_id=bundle.installation_id,
        device_public_key=bundle.device_public_key,
        release_token="release-1",
        reason="llm_start",
        hardware_hash="hardware-hash",
        budget_usd=1.0,
        model="google/gemma-4-26b-a4b-it",
        signed_at="2026-04-08T06:00:45.000Z",
    )
    assert issue_payload == (
        "\n".join(
            [
                bundle.installation_id,
                bundle.device_public_key,
                "release-1",
                "hardware-hash",
                "llm_start",
                "1",
                "google/gemma-4-26b-a4b-it",
                "2026-04-08T06:00:45.000Z",
            ]
        )
    ).encode("utf-8")
    signed_issue = bundle.sign_issue_request(
        release_token="release-1",
        reason="llm_start",
        hardware_hash="hardware-hash",
        budget_usd=1.0,
        model="google/gemma-4-26b-a4b-it",
        signed_at="2026-04-08T06:00:45.000Z",
    )
    assert signed_issue["hardware_hash"] == "hardware-hash"
    public_key.verify(decode_base64url(signed_issue["signature"]), issue_payload)


def test_bundle_signs_discord_issue_request_without_sending_code_hash_field() -> None:
    settings = AppSettings()
    store = InMemorySecretStore()
    bundle = ensure_managed_identity_bundle(_managed_state(settings), store)
    public_key = Ed25519PublicKey.from_public_bytes(decode_base64url(bundle.device_public_key))

    payload = canonical_discord_issue_payload(
        installation_id=bundle.installation_id,
        device_public_key=bundle.device_public_key,
        state="discord-state-1",
        code="discord-oauth-code",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        hardware_hash="hardware-hash",
        hardware_hash_salt_version=7,
        app_version="2.0.0",
        reason="llm_start",
        budget_usd=0.07,
        model="google/gemma-4-26b-a4b-it",
        issue_nonce="issue-nonce-1",
        signed_at="2026-04-30T06:00:30.000Z",
    )
    signed_request = bundle.sign_discord_issue_request(
        code="discord-oauth-code",
        state="discord-state-1",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        hardware_hash="hardware-hash",
        hardware_hash_salt_version=7,
        app_version="2.0.0",
        reason="llm_start",
        budget_usd=0.07,
        model="google/gemma-4-26b-a4b-it",
        issue_nonce="issue-nonce-1",
        signed_at="2026-04-30T06:00:30.000Z",
    )

    assert signed_request == {
        "code": "discord-oauth-code",
        "state": "discord-state-1",
        "installation_id": bundle.installation_id,
        "device_public_key": bundle.device_public_key,
        "redirect_uri": "http://127.0.0.1:62187/discord/callback",
        "hardware_hash": "hardware-hash",
        "hardware_hash_salt_version": 7,
        "app_version": "2.0.0",
        "reason": "llm_start",
        "budget_usd": 0.07,
        "model": "google/gemma-4-26b-a4b-it",
        "issue_nonce": "issue-nonce-1",
        "signed_at": "2026-04-30T06:00:30.000Z",
        "signature_alg": "ed25519",
        "signature": signed_request["signature"],
    }
    assert "code_hash" not in signed_request
    public_key.verify(decode_base64url(signed_request["signature"]), payload)
