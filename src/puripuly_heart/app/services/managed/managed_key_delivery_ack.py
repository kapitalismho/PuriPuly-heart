from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from puripuly_heart.app.ports.broker_client import (
    BrokerClientPort,
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckRequest,
    ManagedKeyDeliveryAckResult,
)
from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.app.services.managed.managed_operation import (
    clear_resume_token,
    other_source_pending_operation,
)
from puripuly_heart.config.provider_values import normalize_owned_referral_id
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    DiagnosticFieldValue,
    ErrorDiagnostics,
)

ACK_SOURCE_DISCORD = "discord"
ACK_SOURCE_QQ = "qq"
DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET = "openrouter_managed_delivery_ack_token"
QQ_MANAGED_DELIVERY_ACK_TOKEN_SECRET = "openrouter_managed_qq_delivery_ack_token"


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAckServiceResult:
    succeeded: bool
    status: str
    diagnostics: ErrorDiagnostics | None = None
    ack_result: ManagedKeyDeliveryAckResult | None = None


class ManagedKeyDeliveryAckTokenStoreError(Exception):
    pass


class ManagedKeyDeliveryAckTokenClearError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ManagedKeyDeliveryAckService:
    broker_client: BrokerClientPort
    secret_store: SecretStorePort
    managed_state: ManagedIdentityStatePort

    async def store_pending(self, metadata: ManagedKeyDeliveryAckMetadata) -> None:
        key = secret_key_for_ack_source(metadata.source)
        try:
            result = await self.secret_store.set_secret(key, metadata.delivery_ack_token)
        except Exception as exc:
            raise ManagedKeyDeliveryAckTokenStoreError("delivery ACK token store failed") from exc
        if not result.succeeded:
            raise ManagedKeyDeliveryAckTokenStoreError("delivery ACK token store failed")
        self.managed_state.pending_delivery_ack_source = metadata.source
        self.managed_state.pending_delivery_ack_delivery_id = metadata.delivery_id
        self.managed_state.pending_delivery_ack_managed_credential_ref = (
            metadata.managed_credential_ref
        )
        self.managed_state.pending_delivery_ack_expires_at = metadata.expires_at

    async def recover_pending(
        self,
        *,
        source: str,
        managed_secret_key: str,
    ) -> ManagedKeyDeliveryAckServiceResult:
        if self.managed_state.pending_delivery_ack_source != source:
            return ManagedKeyDeliveryAckServiceResult(succeeded=True, status="none")
        if (
            not self.managed_state.pending_delivery_ack_delivery_id
            or not self.managed_state.pending_delivery_ack_managed_credential_ref
        ):
            return ManagedKeyDeliveryAckServiceResult(
                succeeded=False,
                status="pending_metadata_incomplete",
                diagnostics=_diagnostics(
                    operation="recover_managed_key_delivery",
                    code="delivery_ack_pending_metadata_incomplete",
                    fields={"source": source},
                ),
            )
        try:
            managed_secret = await self.secret_store.get_secret(managed_secret_key)
        except Exception:
            managed_secret = None
        if managed_secret is None or not managed_secret.value:
            return ManagedKeyDeliveryAckServiceResult(
                succeeded=False,
                status="managed_secret_missing",
                diagnostics=_diagnostics(
                    operation="recover_managed_key_delivery",
                    code="delivery_ack_managed_secret_missing",
                    fields={"source": source, "managed_secret_present": False},
                ),
            )
        return await self.retry_pending()

    async def retry_pending(self) -> ManagedKeyDeliveryAckServiceResult:
        source = self.managed_state.pending_delivery_ack_source
        delivery_id = self.managed_state.pending_delivery_ack_delivery_id
        managed_credential_ref = self.managed_state.pending_delivery_ack_managed_credential_ref
        expires_at = self.managed_state.pending_delivery_ack_expires_at
        if not source or not delivery_id or not managed_credential_ref:
            return ManagedKeyDeliveryAckServiceResult(succeeded=True, status="none")
        try:
            token_read = await self.secret_store.get_secret(secret_key_for_ack_source(source))
        except Exception:
            return ManagedKeyDeliveryAckServiceResult(
                succeeded=False,
                status="token_read_failed",
                diagnostics=_diagnostics(
                    operation="read_delivery_ack_token",
                    code="delivery_ack_token_read_failed",
                    fields={"source": source, "token_present": False},
                ),
            )
        if not token_read.value:
            return ManagedKeyDeliveryAckServiceResult(
                succeeded=False,
                status="missing_token",
                diagnostics=_diagnostics(
                    operation="read_delivery_ack_token",
                    code="delivery_ack_token_missing",
                    fields={"source": source, "token_present": False},
                ),
            )
        try:
            ack_result = await self.broker_client.acknowledge_managed_key_delivery(
                ManagedKeyDeliveryAckRequest(
                    delivery_id=delivery_id,
                    managed_credential_ref=managed_credential_ref,
                    delivery_ack_token=token_read.value,
                )
            )
        except Exception:
            return ManagedKeyDeliveryAckServiceResult(
                succeeded=False,
                status="ack_exception",
                diagnostics=_diagnostics(
                    operation="acknowledge_managed_key_delivery",
                    code="delivery_ack_exception",
                    fields={"source": source},
                ),
            )
        if ack_result.succeeded and ack_result.status in {"acknowledged", "already_acknowledged"}:
            previous_referral_id = self.managed_state.referral_id
            previous_referral_source = self.managed_state.referral_source
            previous_operation_id = self.managed_state.pending_managed_operation_id
            previous_operation_source = self.managed_state.pending_managed_operation_source
            previous_operation_installation_id = (
                self.managed_state.pending_managed_operation_installation_id
            )
            previous_operation_state = self.managed_state.pending_managed_operation_state
            try:
                await self.clear_pending(source)
            except ManagedKeyDeliveryAckTokenClearError:
                return ManagedKeyDeliveryAckServiceResult(
                    succeeded=False,
                    status="token_clear_failed",
                    diagnostics=_diagnostics(
                        operation="clear_delivery_ack_token",
                        code="delivery_ack_token_clear_failed",
                        fields={"source": source},
                    ),
                    ack_result=ack_result,
                )
            apply_ack_referral_to_managed_state(self.managed_state, ack_result, source)
            preserve_operation = (
                other_source_pending_operation(self.managed_state, source=source) is not None
            )
            if not preserve_operation:
                self.managed_state.pending_managed_operation_id = None
                self.managed_state.pending_managed_operation_source = None
                self.managed_state.pending_managed_operation_installation_id = None
                self.managed_state.pending_managed_operation_state = None
            try:
                self.managed_state.persist()
            except Exception:
                try:
                    restore_result = await self.secret_store.set_secret(
                        secret_key_for_ack_source(source),
                        token_read.value,
                    )
                except Exception:
                    restore_result = None
                self.managed_state.pending_delivery_ack_source = source
                self.managed_state.pending_delivery_ack_delivery_id = delivery_id
                self.managed_state.pending_delivery_ack_managed_credential_ref = (
                    managed_credential_ref
                )
                self.managed_state.pending_delivery_ack_expires_at = expires_at
                self.managed_state.referral_id = previous_referral_id
                self.managed_state.referral_source = previous_referral_source
                self.managed_state.pending_managed_operation_id = previous_operation_id
                self.managed_state.pending_managed_operation_source = previous_operation_source
                self.managed_state.pending_managed_operation_installation_id = (
                    previous_operation_installation_id
                )
                self.managed_state.pending_managed_operation_state = previous_operation_state
                if restore_result is None or not restore_result.succeeded:
                    return ManagedKeyDeliveryAckServiceResult(
                        succeeded=False,
                        status="token_restore_failed",
                        diagnostics=_diagnostics(
                            operation="restore_delivery_ack_token",
                            code="delivery_ack_token_restore_failed",
                            fields={"source": source},
                        ),
                        ack_result=ack_result,
                    )
                return ManagedKeyDeliveryAckServiceResult(
                    succeeded=False,
                    status="pending_clear_persist_failed",
                    diagnostics=_diagnostics(
                        operation="clear_delivery_ack_metadata",
                        code="delivery_ack_clear_persist_failed",
                        fields={"source": source},
                    ),
                    ack_result=ack_result,
                )
            if not preserve_operation:
                try:
                    await clear_resume_token(self.secret_store)
                except Exception:
                    pass
            return ManagedKeyDeliveryAckServiceResult(
                succeeded=True,
                status=ack_result.status,
                ack_result=ack_result,
            )
        return ManagedKeyDeliveryAckServiceResult(
            succeeded=False,
            status=ack_result.status,
            diagnostics=ack_result.diagnostics
            or _diagnostics(
                operation="acknowledge_managed_key_delivery",
                code="delivery_ack_failed",
                fields={"source": source, "ack_status": ack_result.status},
            ),
            ack_result=ack_result,
        )

    async def clear_pending(self, source: str | None = None) -> None:
        pending_source = source or self.managed_state.pending_delivery_ack_source
        if pending_source:
            try:
                result = await self.secret_store.clear_secret(
                    secret_key_for_ack_source(pending_source)
                )
            except Exception as exc:
                raise ManagedKeyDeliveryAckTokenClearError(
                    "delivery ACK token clear failed"
                ) from exc
            if not result.succeeded:
                raise ManagedKeyDeliveryAckTokenClearError("delivery ACK token clear failed")
        self.managed_state.pending_delivery_ack_source = None
        self.managed_state.pending_delivery_ack_delivery_id = None
        self.managed_state.pending_delivery_ack_managed_credential_ref = None
        self.managed_state.pending_delivery_ack_expires_at = None


async def store_pending_ack_in_settings_values(
    *,
    settings_values: Mapping[str, object],
    secret_store: SecretStorePort,
    metadata: ManagedKeyDeliveryAckMetadata,
) -> dict[str, object]:
    try:
        result = await secret_store.set_secret(
            secret_key_for_ack_source(metadata.source),
            metadata.delivery_ack_token,
        )
    except Exception as exc:
        raise ManagedKeyDeliveryAckTokenStoreError("delivery ACK token store failed") from exc
    if not result.succeeded:
        raise ManagedKeyDeliveryAckTokenStoreError("delivery ACK token store failed")
    values = _copy_mapping(settings_values)
    state = values.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        values["state"] = state
    managed = state.setdefault("managed_connection", {})
    if not isinstance(managed, dict):
        managed = {}
        state["managed_connection"] = managed
    managed.update(
        {
            "pending_delivery_ack_source": metadata.source,
            "pending_delivery_ack_delivery_id": metadata.delivery_id,
            "pending_delivery_ack_managed_credential_ref": metadata.managed_credential_ref,
            "pending_delivery_ack_expires_at": metadata.expires_at,
        }
    )
    return values


def clear_pending_ack_in_settings_values(
    settings_values: Mapping[str, object],
) -> dict[str, object]:
    values = _copy_mapping(settings_values)
    state = values.setdefault("state", {})
    if isinstance(state, dict):
        managed = state.setdefault("managed_connection", {})
        if isinstance(managed, dict):
            managed.update(
                {
                    "pending_delivery_ack_source": None,
                    "pending_delivery_ack_delivery_id": None,
                    "pending_delivery_ack_managed_credential_ref": None,
                    "pending_delivery_ack_expires_at": None,
                }
            )
    return values


def pending_ack_metadata_settings_values(
    *,
    settings_values: Mapping[str, object],
    metadata: ManagedKeyDeliveryAckMetadata,
) -> dict[str, object]:
    return _with_pending_ack_metadata(_copy_mapping(settings_values), metadata)


def _copy_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {key: _copy_value(value) for key, value in values.items()}


def _copy_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if isinstance(value, list | tuple):
        return [_copy_value(item) for item in value]
    return value


def _with_pending_ack_metadata(
    values: dict[str, object],
    metadata: ManagedKeyDeliveryAckMetadata,
) -> dict[str, object]:
    state = values.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        values["state"] = state
    managed = state.setdefault("managed_connection", {})
    if not isinstance(managed, dict):
        managed = {}
        state["managed_connection"] = managed
    managed.update(
        {
            "pending_delivery_ack_source": metadata.source,
            "pending_delivery_ack_delivery_id": metadata.delivery_id,
            "pending_delivery_ack_managed_credential_ref": metadata.managed_credential_ref,
            "pending_delivery_ack_expires_at": metadata.expires_at,
        }
    )
    return values


def secret_key_for_ack_source(source: str) -> str:
    if source == ACK_SOURCE_QQ:
        return QQ_MANAGED_DELIVERY_ACK_TOKEN_SECRET
    return DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET


def apply_ack_referral_to_managed_state(
    managed_state: ManagedIdentityStatePort,
    ack_result: ManagedKeyDeliveryAckResult,
    source: str,
) -> None:
    if source not in {ACK_SOURCE_DISCORD, ACK_SOURCE_QQ}:
        return
    current_source = managed_state.referral_source
    if current_source not in {ACK_SOURCE_DISCORD, ACK_SOURCE_QQ}:
        current_source = (
            ACK_SOURCE_DISCORD
            if normalize_owned_referral_id(managed_state.referral_id) is not None
            else None
        )
    if current_source is not None and current_source != source:
        managed_state.referral_id = None
    managed_state.referral_source = source
    acknowledged_referral_id = normalize_owned_referral_id(ack_result.referral_id)
    if acknowledged_referral_id is not None:
        managed_state.referral_id = acknowledged_referral_id


@dataclass(frozen=True, slots=True)
class PendingDeliveryAckMetadata:
    source: str
    delivery_id: str
    managed_credential_ref: str
    expires_at: str | None = None


def read_any_pending_delivery_ack(
    managed_state: ManagedIdentityStatePort,
) -> PendingDeliveryAckMetadata | None:
    source = managed_state.pending_delivery_ack_source
    delivery_id = managed_state.pending_delivery_ack_delivery_id
    managed_credential_ref = managed_state.pending_delivery_ack_managed_credential_ref
    if (
        source not in (ACK_SOURCE_DISCORD, ACK_SOURCE_QQ)
        or not delivery_id
        or not managed_credential_ref
    ):
        return None
    return PendingDeliveryAckMetadata(
        source=source,
        delivery_id=delivery_id,
        managed_credential_ref=managed_credential_ref,
        expires_at=managed_state.pending_delivery_ack_expires_at,
    )


def other_source_pending_delivery_ack(
    managed_state: ManagedIdentityStatePort,
    *,
    source: str,
) -> PendingDeliveryAckMetadata | None:
    pending = read_any_pending_delivery_ack(managed_state)
    if pending is None or pending.source == source:
        return None
    return pending


def _diagnostics(
    *,
    operation: str,
    code: str,
    fields: Mapping[str, DiagnosticFieldValue],
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="managed_key_delivery_ack",
        operation=operation,
        code=code,
        category=DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields=fields,
    )


__all__ = [
    "ACK_SOURCE_DISCORD",
    "ACK_SOURCE_QQ",
    "DISCORD_MANAGED_DELIVERY_ACK_TOKEN_SECRET",
    "ManagedKeyDeliveryAckService",
    "ManagedKeyDeliveryAckServiceResult",
    "ManagedKeyDeliveryAckTokenClearError",
    "ManagedKeyDeliveryAckTokenStoreError",
    "PendingDeliveryAckMetadata",
    "QQ_MANAGED_DELIVERY_ACK_TOKEN_SECRET",
    "clear_pending_ack_in_settings_values",
    "other_source_pending_delivery_ack",
    "pending_ack_metadata_settings_values",
    "read_any_pending_delivery_ack",
    "secret_key_for_ack_source",
    "store_pending_ack_in_settings_values",
]
