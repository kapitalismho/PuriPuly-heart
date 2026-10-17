from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


def credential_binding_secret(source: str) -> str:
    if source not in {"discord", "qq"}:
        raise ValueError("unsupported managed credential source")
    return f"openrouter_managed_{source}_credential_binding"


def managed_key_secret(source: str) -> str:
    if source == "qq":
        return "openrouter_managed_qq_api_key"
    if source == "discord":
        return "openrouter_managed_api_key"
    raise ValueError("unsupported managed credential source")


@dataclass(frozen=True, slots=True)
class ManagedCredentialBinding:
    source: str
    installation_id: str
    operation_id: str
    delivery_id: str
    managed_credential_ref: str
    credential_digest: str

    @classmethod
    def create(
        cls,
        *,
        source: str,
        installation_id: str,
        operation_id: str,
        delivery_id: str,
        managed_credential_ref: str,
        credential: str,
    ) -> ManagedCredentialBinding:
        values = (installation_id, operation_id, delivery_id, managed_credential_ref, credential)
        if source not in {"discord", "qq"} or not all(values):
            raise ValueError("incomplete managed credential binding")
        return cls(
            source,
            installation_id,
            operation_id,
            delivery_id,
            managed_credential_ref,
            hashlib.sha256(credential.encode()).hexdigest(),
        )

    def encode(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def decode(cls, value: str | None) -> ManagedCredentialBinding | None:
        if not value:
            return None
        try:
            data = json.loads(value)
            binding = cls(**data)
            if binding.source not in {"discord", "qq"} or not all(
                isinstance(item, str) and item for item in asdict(binding).values()
            ):
                return None
            return binding
        except TypeError, ValueError:
            return None

    def matches(
        self,
        *,
        credential: str | None,
        source: str,
        installation_id: str,
        operation_id: str | None = None,
        delivery_id: str | None = None,
        managed_credential_ref: str | None = None,
    ) -> bool:
        return bool(
            credential
            and self.source == source
            and self.installation_id == installation_id
            and (operation_id is None or self.operation_id == operation_id)
            and (delivery_id is None or self.delivery_id == delivery_id)
            and (
                managed_credential_ref is None
                or self.managed_credential_ref == managed_credential_ref
            )
            and hmac.compare_digest(
                self.credential_digest, hashlib.sha256(credential.encode()).hexdigest()
            )
        )


def delivery_expired(expires_at: str | None) -> bool:
    if expires_at is None:
        return False
    try:
        return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= datetime.now(UTC)
    except ValueError, TypeError:
        return True
