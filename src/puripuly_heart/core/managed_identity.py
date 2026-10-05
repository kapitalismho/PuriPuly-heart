from __future__ import annotations

import base64
import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
)
from puripuly_heart.core.storage.secrets import SecretStore

MANAGED_DEVICE_PRIVATE_KEY_SECRET = "managed_device_private_key"
MANAGED_DEVICE_PUBLIC_KEY_SECRET = "managed_device_public_key"
MANAGED_IDENTITY_BINDING_SECRET = "managed_identity_binding"
DISCORD_OPENROUTER_ISSUE_METHOD = "POST"
DISCORD_OPENROUTER_ISSUE_PATH = "/v1/providers/openrouter/discord/issue"


def encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def decode_base64url(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("base64url value must be a non-empty string")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except Exception as exc:  # pragma: no cover - base64 errors vary by runtime
        raise ValueError("invalid base64url value") from exc


@dataclass(frozen=True, slots=True)
class ManagedIdentityBundle:
    installation_id: str
    device_public_key: str
    _private_key: Ed25519PrivateKey = field(repr=False, compare=False)

    def sign_verify_request(
        self,
        *,
        challenge: str,
        challenge_expires_at: str,
        hardware_hash: str,
        app_version: str,
        signed_at: str,
    ) -> dict[str, str]:
        payload = canonical_verify_payload(
            installation_id=self.installation_id,
            device_public_key=self.device_public_key,
            challenge=challenge,
            challenge_expires_at=challenge_expires_at,
            hardware_hash=hardware_hash,
            app_version=app_version,
            signed_at=signed_at,
        )
        return {
            "installation_id": self.installation_id,
            "device_public_key": self.device_public_key,
            "challenge": challenge,
            "challenge_expires_at": challenge_expires_at,
            "hardware_hash": hardware_hash,
            "app_version": app_version,
            "signed_at": signed_at,
            "signature": _sign_payload(self._private_key, payload),
        }

    def sign_status_request(self, *, timestamp: str) -> dict[str, str]:
        payload = canonical_status_payload(
            installation_id=self.installation_id,
            timestamp=timestamp,
        )
        return {
            "installation_id": self.installation_id,
            "timestamp": timestamp,
            "signature": _sign_payload(self._private_key, payload),
        }

    def sign_issue_request(
        self,
        *,
        release_token: str,
        hardware_hash: str,
        reason: str,
        budget_usd: int | float,
        model: str,
        signed_at: str,
    ) -> dict[str, str | int | float]:
        payload = canonical_issue_payload(
            installation_id=self.installation_id,
            device_public_key=self.device_public_key,
            release_token=release_token,
            hardware_hash=hardware_hash,
            reason=reason,
            budget_usd=budget_usd,
            model=model,
            signed_at=signed_at,
        )
        return {
            "installation_id": self.installation_id,
            "device_public_key": self.device_public_key,
            "release_token": release_token,
            "hardware_hash": hardware_hash,
            "reason": reason,
            "budget_usd": budget_usd,
            "model": model,
            "signed_at": signed_at,
            "signature": _sign_payload(self._private_key, payload),
        }

    def sign_discord_issue_request(
        self,
        *,
        code: str,
        state: str,
        redirect_uri: str,
        hardware_hash: str,
        hardware_hash_salt_version: int,
        app_version: str,
        reason: str,
        budget_usd: int | float,
        model: str,
        issue_nonce: str,
        signed_at: str,
    ) -> dict[str, str | int | float]:
        payload = canonical_discord_issue_payload(
            installation_id=self.installation_id,
            device_public_key=self.device_public_key,
            state=state,
            code=code,
            redirect_uri=redirect_uri,
            hardware_hash=hardware_hash,
            hardware_hash_salt_version=hardware_hash_salt_version,
            app_version=app_version,
            reason=reason,
            budget_usd=budget_usd,
            model=model,
            issue_nonce=issue_nonce,
            signed_at=signed_at,
        )
        return {
            "code": code,
            "state": state,
            "installation_id": self.installation_id,
            "device_public_key": self.device_public_key,
            "redirect_uri": redirect_uri,
            "hardware_hash": hardware_hash,
            "hardware_hash_salt_version": hardware_hash_salt_version,
            "app_version": app_version,
            "reason": reason,
            "budget_usd": budget_usd,
            "model": model,
            "issue_nonce": issue_nonce,
            "signed_at": signed_at,
            "signature_alg": "ed25519",
            "signature": _sign_payload(self._private_key, payload),
        }


def canonical_verify_payload(
    *,
    installation_id: str,
    device_public_key: str,
    challenge: str,
    challenge_expires_at: str,
    hardware_hash: str,
    app_version: str,
    signed_at: str,
) -> bytes:
    return _canonical_payload(
        installation_id,
        device_public_key,
        challenge,
        challenge_expires_at,
        hardware_hash,
        app_version,
        signed_at,
    )


def canonical_status_payload(*, installation_id: str, timestamp: str) -> bytes:
    return _canonical_payload(installation_id, timestamp)


def canonical_issue_payload(
    *,
    installation_id: str,
    device_public_key: str,
    release_token: str,
    hardware_hash: str,
    reason: str,
    budget_usd: int | float,
    model: str,
    signed_at: str,
) -> bytes:
    return _canonical_payload(
        installation_id,
        device_public_key,
        release_token,
        hardware_hash,
        reason,
        _canonical_number_string(budget_usd),
        model,
        signed_at,
    )


def canonical_discord_issue_payload(
    *,
    installation_id: str,
    device_public_key: str,
    state: str,
    code: str,
    redirect_uri: str,
    hardware_hash: str,
    hardware_hash_salt_version: int,
    app_version: str,
    reason: str,
    budget_usd: int | float,
    model: str,
    issue_nonce: str,
    signed_at: str,
) -> bytes:
    code_hash = encode_base64url(hashlib.sha256(code.encode("utf-8")).digest())
    return _canonical_payload(
        DISCORD_OPENROUTER_ISSUE_METHOD,
        DISCORD_OPENROUTER_ISSUE_PATH,
        installation_id,
        device_public_key,
        state,
        code_hash,
        redirect_uri,
        hardware_hash,
        str(hardware_hash_salt_version),
        app_version,
        reason,
        _canonical_number_string(budget_usd),
        model,
        issue_nonce,
        signed_at,
    )


def ensure_managed_identity_bundle(
    state: ManagedIdentityStatePort,
    secret_store: SecretStore,
    *,
    broker_installation_id: str | None = None,
    broker_device_public_key: str | None = None,
) -> ManagedIdentityBundle:
    existing = _load_existing_bundle(state, secret_store)
    if existing is not None and _bundle_matches_broker(
        existing,
        broker_installation_id=broker_installation_id,
        broker_device_public_key=broker_device_public_key,
    ):
        return existing
    return _replace_managed_identity_bundle(state, secret_store)


def load_existing_managed_identity_bundle(
    state: ManagedIdentityStatePort,
    secret_store: SecretStore,
) -> ManagedIdentityBundle | None:
    """Load a valid persisted managed identity bundle without creating or repairing it."""

    return _load_existing_bundle(state, secret_store)


def regenerate_managed_identity_bundle(
    state: ManagedIdentityStatePort,
    secret_store: SecretStore,
) -> ManagedIdentityBundle:
    return _replace_managed_identity_bundle(state, secret_store)


def _bundle_matches_broker(
    bundle: ManagedIdentityBundle,
    *,
    broker_installation_id: str | None,
    broker_device_public_key: str | None,
) -> bool:
    normalized_broker_installation_id = _normalize_optional_string(broker_installation_id)
    if (
        normalized_broker_installation_id is not None
        and normalized_broker_installation_id != bundle.installation_id
    ):
        return False

    normalized_broker_device_public_key = _normalize_optional_string(broker_device_public_key)
    if (
        normalized_broker_device_public_key is not None
        and normalized_broker_device_public_key != bundle.device_public_key
    ):
        return False
    return True


def _load_existing_bundle(
    state: ManagedIdentityStatePort,
    secret_store: SecretStore,
) -> ManagedIdentityBundle | None:
    installation_id = state.installation_id.strip()
    if not _is_uuid7(installation_id):
        return None

    private_key_value = secret_store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET)
    public_key_value = secret_store.get(MANAGED_DEVICE_PUBLIC_KEY_SECRET)
    binding_value = secret_store.get(MANAGED_IDENTITY_BINDING_SECRET)
    if private_key_value is None or public_key_value is None or binding_value is None:
        return None

    try:
        private_key = Ed25519PrivateKey.from_private_bytes(decode_base64url(private_key_value))
    except ValueError:
        return None

    derived_public_key = encode_base64url(private_key.public_key().public_bytes_raw())
    if derived_public_key != public_key_value:
        return None
    if binding_value != _managed_identity_binding_value(installation_id, derived_public_key):
        return None

    return ManagedIdentityBundle(
        installation_id=installation_id,
        device_public_key=public_key_value,
        _private_key=private_key,
    )


def _replace_managed_identity_bundle(
    state: ManagedIdentityStatePort,
    secret_store: SecretStore,
) -> ManagedIdentityBundle:
    previous = state.snapshot()
    previous_managed_api_key = secret_store.get(OPENROUTER_MANAGED_API_KEY_SECRET)
    previous_managed_user_id = secret_store.get(OPENROUTER_MANAGED_USER_ID_SECRET)
    previous_managed_user_installation_id = secret_store.get(
        OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET
    )
    previous_private_key = secret_store.get(MANAGED_DEVICE_PRIVATE_KEY_SECRET)
    previous_public_key = secret_store.get(MANAGED_DEVICE_PUBLIC_KEY_SECRET)
    previous_binding_value = secret_store.get(MANAGED_IDENTITY_BINDING_SECRET)

    installation_id = _generate_uuid7()
    private_key = Ed25519PrivateKey.generate()
    private_key_value = encode_base64url(private_key.private_bytes_raw())
    public_key_value = encode_base64url(private_key.public_key().public_bytes_raw())
    binding_value = _managed_identity_binding_value(installation_id, public_key_value)

    try:
        secret_store.delete(OPENROUTER_MANAGED_API_KEY_SECRET)
        secret_store.delete(OPENROUTER_MANAGED_USER_ID_SECRET)
        secret_store.delete(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET)
        secret_store.set(MANAGED_DEVICE_PRIVATE_KEY_SECRET, private_key_value)
        secret_store.set(MANAGED_DEVICE_PUBLIC_KEY_SECRET, public_key_value)
        secret_store.set(MANAGED_IDENTITY_BINDING_SECRET, binding_value)
        state.installation_id = installation_id
        state.release_token = None
        state.release_token_expires_at = None
        state.verified_hardware_hash = None
        state.verified_hardware_hash_salt_version = None
        state.persist()
    except Exception as exc:
        state.restore(previous)
        rollback_error = _restore_secret_state(
            secret_store,
            managed_api_key=previous_managed_api_key,
            managed_user_id=previous_managed_user_id,
            managed_user_installation_id=previous_managed_user_installation_id,
            private_key=previous_private_key,
            public_key=previous_public_key,
            binding_value=previous_binding_value,
        )
        if rollback_error is not None:
            exc.add_note(f"secret rollback failed: {rollback_error}")
        raise

    return ManagedIdentityBundle(
        installation_id=installation_id,
        device_public_key=public_key_value,
        _private_key=private_key,
    )


def _canonical_payload(*lines: str) -> bytes:
    return "\n".join(lines).encode("utf-8")


def _canonical_number_string(value: int | float) -> str:
    if isinstance(value, bool):
        raise TypeError("canonical numeric values must not be bool")
    try:
        number = float(value)
    except OverflowError:
        return "Infinity" if value > 0 else "-Infinity"
    except (TypeError, ValueError) as exc:
        raise TypeError("canonical numeric values must be real numbers") from exc

    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    if number == 0:
        return "0"

    rendered = repr(number).lower()
    if "e" not in rendered:
        return _trim_decimal_string(rendered)

    mantissa, exponent_text = rendered.split("e", maxsplit=1)
    exponent = int(exponent_text)
    absolute_number = abs(number)

    if absolute_number >= 1e21 or absolute_number < 1e-6:
        exponent_sign = "+" if exponent >= 0 else ""
        return f"{_trim_decimal_string(mantissa)}e{exponent_sign}{exponent}"

    return _expand_scientific_notation(
        mantissa=_trim_decimal_string(mantissa),
        exponent=exponent,
    )


def _generate_uuid7() -> str:
    timestamp_ms = time.time_ns() // 1_000_000
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    value = (
        ((timestamp_ms & ((1 << 48) - 1)) << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return str(uuid.UUID(int=value))


def _is_uuid7(value: str) -> bool:
    try:
        return uuid.UUID(value).version == 7
    except ValueError:
        return False


def _normalize_optional_string(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _managed_identity_binding_value(installation_id: str, device_public_key: str) -> str:
    digest = hashlib.sha256(f"{installation_id}\n{device_public_key}".encode("utf-8")).digest()
    return encode_base64url(digest)


def _restore_secret_state(
    secret_store: SecretStore,
    *,
    managed_api_key: str | None,
    managed_user_id: str | None,
    managed_user_installation_id: str | None,
    private_key: str | None,
    public_key: str | None,
    binding_value: str | None,
) -> Exception | None:
    try:
        _restore_secret_value(secret_store, OPENROUTER_MANAGED_API_KEY_SECRET, managed_api_key)
        _restore_secret_value(secret_store, OPENROUTER_MANAGED_USER_ID_SECRET, managed_user_id)
        _restore_secret_value(
            secret_store,
            OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
            managed_user_installation_id,
        )
        _restore_secret_value(secret_store, MANAGED_DEVICE_PRIVATE_KEY_SECRET, private_key)
        _restore_secret_value(secret_store, MANAGED_DEVICE_PUBLIC_KEY_SECRET, public_key)
        _restore_secret_value(secret_store, MANAGED_IDENTITY_BINDING_SECRET, binding_value)
    except Exception as exc:
        return exc
    return None


def _restore_secret_value(secret_store: SecretStore, key: str, value: str | None) -> None:
    if value is None:
        secret_store.delete(key)
    else:
        secret_store.set(key, value)


def _trim_decimal_string(value: str) -> str:
    trimmed = value
    if "." in trimmed:
        trimmed = trimmed.rstrip("0").rstrip(".")
    return trimmed


def _expand_scientific_notation(*, mantissa: str, exponent: int) -> str:
    sign = ""
    digits = mantissa
    if digits.startswith("-"):
        sign = "-"
        digits = digits[1:]

    integer_part, dot, fraction_part = digits.partition(".")
    coefficient = integer_part + fraction_part
    decimal_index = len(integer_part) + exponent

    if decimal_index <= 0:
        expanded = f"0.{('0' * (-decimal_index))}{coefficient}"
    elif decimal_index >= len(coefficient):
        expanded = coefficient + ("0" * (decimal_index - len(coefficient)))
    else:
        expanded = f"{coefficient[:decimal_index]}.{coefficient[decimal_index:]}"

    return sign + _trim_decimal_string(expanded)


def _sign_payload(private_key: Ed25519PrivateKey, payload: bytes) -> str:
    return encode_base64url(private_key.sign(payload))
