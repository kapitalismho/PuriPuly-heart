from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckRequest,
    ManagedKeyDeliveryAckResult,
    ManagedOperationAttemptSnapshot,
    ManagedOperationDeliverySnapshot,
    ManagedOperationResumeRequest,
    ManagedOperationStatusRequest,
    ManagedOperationStatusResult,
    QqManagedAssertionFailureSubcode,
    QqManagedAssertionRequest,
    QqManagedAssertionResult,
    QqManagedEntitlementSnapshot,
)
from puripuly_heart.core import messages, network_clients
from puripuly_heart.core.external_network import NetworkConfigurationError
from puripuly_heart.core.qq_auth_diagnostics import exception_fields, safe_fields

from .managed_openrouter_release import (
    ManagedOpenRouterChallengeSuccess,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterPreflightStop,
    ManagedOpenRouterReleaseError,
    ManagedOpenRouterVerifySuccess,
)
from .openrouter_credentials import (
    normalize_managed_openrouter_user_identifier,
)

RETRYABLE_ERROR_CODE = "trial_unavailable"
RETRYABLE_ERROR_CLASS = "retryable"
PUBLIC_ERROR_CODES = frozenset(
    {
        "invalid_request",
        "rate_limited",
        "challenge_expired",
        "challenge_invalid",
        "issuance_suspended",
        "trial_unavailable",
        "trial_not_eligible",
        "internal_error",
        "authorization_expired",
        "resume_issuance_unavailable",
    }
)
PUBLIC_ERROR_CLASSES = frozenset({"retryable", "terminal", "security_fail"})


class _BrokerPayload(dict[str, object]):
    def __init__(self, payload: Mapping[str, object], response: httpx.Response) -> None:
        super().__init__(payload)
        self.response = response


@dataclass(slots=True)
class HttpManagedOpenRouterBrokerClient:
    base_url: str
    timeout: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)

    async def challenge(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        app_version: str,
    ) -> ManagedOpenRouterChallengeSuccess | ManagedOpenRouterPreflightStop:
        payload = await self._post_json(
            path="/v1/trial/challenge",
            request_body={
                "installation_id": installation_id,
                "device_public_key": device_public_key,
                "app_version": app_version,
            },
            operation="challenge",
        )
        try:
            return ManagedOpenRouterChallengeSuccess(
                challenge=_require_text(payload, "challenge"),
                challenge_expires_at=_require_text(payload, "challenge_expires_at"),
                fingerprint_salt=_parse_fingerprint_salt(payload, operation="challenge"),
            )
        except ValueError as exc:
            raise _retryable_error(
                "challenge", f"broker returned malformed payload: {exc}", payload=payload
            ) from exc

    async def start_discord_oauth(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        redirect_uri: str,
        app_version: str,
    ) -> ManagedOpenRouterDiscordStartSuccess:
        request_body = {
            "installation_id": installation_id,
            "device_public_key": device_public_key,
            "redirect_uri": redirect_uri,
            "app_version": app_version,
        }

        payload = await self._post_json(
            path="/v1/auth/discord/start",
            request_body=request_body,
            operation="discord_start",
        )
        try:
            return ManagedOpenRouterDiscordStartSuccess(
                authorization_url=_require_text(payload, "authorization_url"),
                redirect_uri=_require_text(payload, "redirect_uri"),
                oauth_session_expires_at=_require_text(payload, "oauth_session_expires_at"),
                issue_nonce=_require_text(payload, "issue_nonce"),
                fingerprint_salt=_parse_fingerprint_salt(payload, operation="discord_start"),
                fingerprint_salt_version=_require_int(payload, "fingerprint_salt_version"),
            )
        except ValueError as exc:
            raise _retryable_error(
                "discord_start", f"broker returned malformed payload: {exc}", payload=payload
            ) from exc

    async def verify(self, request: dict[str, str]) -> ManagedOpenRouterVerifySuccess:
        payload = await self._post_json(
            path="/v1/trial/challenge/verify",
            request_body=request,
            operation="verify",
        )
        try:
            return ManagedOpenRouterVerifySuccess(
                release_token=_require_text(payload, "release_token"),
                release_token_expires_at=_require_text(payload, "release_token_expires_at"),
            )
        except ValueError as exc:
            raise _retryable_error(
                "verify", f"broker returned malformed payload: {exc}", payload=payload
            ) from exc

    async def issue(self, request: dict[str, object]) -> ManagedOpenRouterIssueSuccess:
        payload = await self._post_json(
            path="/v1/providers/openrouter/issue",
            request_body=request,
            operation="issue",
        )
        if _delivery_ack_required(payload):
            error = _broker_error(
                operation="issue",
                code="managed_delivery_protocol_mismatch",
                error_class="terminal",
                message="Legacy managed delivery cannot require acknowledgement.",
            )
            if isinstance(payload, _BrokerPayload):
                error = _with_response(error, payload.response, shape="malformed_success")
            raise error
        try:
            return ManagedOpenRouterIssueSuccess(
                openrouter_api_key=_require_text(payload, "openrouter_api_key"),
                managed_credential_ref=_require_optional_text(payload, "managed_credential_ref"),
                expires_at=_require_optional_text(payload, "expires_at"),
                openrouter_user_id=normalize_managed_openrouter_user_identifier(
                    payload.get("openrouter_user_id")
                ),
            )
        except ValueError as exc:
            raise _retryable_error(
                "issue", f"broker returned malformed payload: {exc}", payload=payload
            ) from exc

    async def issue_discord_managed_key(
        self,
        request: dict[str, object],
    ) -> ManagedOpenRouterIssueSuccess:
        payload = await self._post_json(
            path="/v1/providers/openrouter/discord/issue",
            request_body={**dict(request), "delivery_ack_supported": True},
            operation="discord_issue",
        )
        try:
            _validate_operation_response_binding(
                payload,
                operation_id=request.get("operation_id"),
                installation_id=request.get("installation_id"),
                source="discord",
            )
            return ManagedOpenRouterIssueSuccess(
                openrouter_api_key=_require_text(payload, "openrouter_api_key"),
                managed_credential_ref=_require_optional_text(payload, "managed_credential_ref"),
                expires_at=_require_optional_text(payload, "expires_at"),
                openrouter_user_id=normalize_managed_openrouter_user_identifier(
                    payload.get("openrouter_user_id")
                ),
                delivery_ack_required=_delivery_ack_required(payload),
                delivery_id=_delivery_ack_text(payload, "delivery_id"),
                delivery_ack_token=_delivery_ack_text(payload, "delivery_ack_token"),
                delivery_ack_expires_at=_require_optional_text(payload, "delivery_ack_expires_at"),
            )
        except ValueError as exc:
            raise _retryable_error(
                "discord_issue", f"broker returned malformed payload: {exc}", payload=payload
            ) from exc

    async def assert_qq_managed_identity(
        self,
        request: QqManagedAssertionRequest,
    ) -> QqManagedAssertionResult:
        request_body: dict[str, object] = {
            "qq_identity": request.qq_identity,
            "credential": request.credential,
            "asserted_at": request.asserted_at,
            "delivery_ack_supported": True,
        }
        if request.installation_id:
            request_body["installation_id"] = request.installation_id
        if request.operation_id:
            request_body["operation_id"] = request.operation_id
        if request.resume_token:
            request_body["resume_token"] = request.resume_token
        try:
            payload = await self._post_json(
                path="/v1/auth/qq/assert",
                request_body=request_body,
                operation="qq_assert",
            )
        except ManagedOpenRouterReleaseError as exc:
            return _qq_assertion_failure_from_error(exc)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return _qq_assertion_failure(
                failure_subcode="broker_unavailable",
                code="broker_exception",
                operation="qq_assert",
                category=messages.DIAGNOSTIC_CATEGORY_NETWORK,
                retry_after_ms=None,
                fields=safe_fields(exception_fields(exc)),
            )

        status = payload.get("status")
        if not isinstance(status, str) or status not in {"issued", "delivery_pending"}:
            return _qq_assertion_failure(
                failure_subcode="key_unavailable",
                code="qq_key_unavailable",
                operation="qq_assert",
                category=messages.DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
                retry_after_ms=_header_retry_after(payload.response),
                fields={
                    **_response_fields(payload.response, shape="malformed_success"),
                    **safe_fields({"status": status}),
                },
            )
        try:
            _validate_operation_response_binding(
                payload,
                operation_id=request.operation_id,
                installation_id=request.installation_id,
                source="qq",
            )
            entitlement = QqManagedEntitlementSnapshot(
                qq_subject_ref=_require_text(payload, "qq_subject_ref"),
                managed_credential_ref=_require_optional_text(
                    payload,
                    "managed_credential_ref",
                ),
                expires_at=_require_optional_text(payload, "expires_at"),
                openrouter_user_id=normalize_managed_openrouter_user_identifier(
                    payload.get("openrouter_user_id")
                ),
            )
            return QqManagedAssertionResult(
                succeeded=True,
                managed_secret_key=_require_text(payload, "openrouter_api_key"),
                entitlement=entitlement,
                failure_subcode=None,
                retry_after_ms=None,
                message=None,
                diagnostics=None,
                delivery_ack=_parse_delivery_ack_metadata(payload, source="qq"),
            )
        except ValueError:
            return _qq_assertion_failure(
                failure_subcode="key_unavailable",
                code="qq_malformed_success_payload",
                operation="qq_assert",
                category=messages.DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
                retry_after_ms=_header_retry_after(payload.response),
                fields=_response_fields(payload.response, shape="malformed_success"),
            )

    async def acknowledge_managed_key_delivery(
        self,
        request: ManagedKeyDeliveryAckRequest,
    ) -> ManagedKeyDeliveryAckResult:
        try:
            payload = await self._post_json(
                path="/v1/providers/openrouter/managed-key-delivery/ack",
                request_body={
                    "delivery_id": request.delivery_id,
                    "managed_credential_ref": request.managed_credential_ref,
                    "delivery_ack_token": request.delivery_ack_token,
                },
                operation="managed_key_delivery_ack",
            )
        except ManagedOpenRouterReleaseError as exc:
            return _ack_failure_from_error(exc)

        status = payload.get("status")
        succeeded = (
            payload.get("ok") is True
            and isinstance(status, str)
            and status
            in {
                "acknowledged",
                "already_acknowledged",
            }
        )
        if not isinstance(status, str):
            status = "malformed"
        return ManagedKeyDeliveryAckResult(
            succeeded=succeeded,
            status=status,
            diagnostics=(
                None
                if succeeded
                else _ack_diagnostics(
                    code="managed_key_delivery_ack_failed",
                    status=status,
                    fields=_response_fields(payload.response, shape="malformed_success"),
                    retry_after_ms=_header_retry_after(payload.response),
                )
            ),
        )

    async def get_managed_operation_status(
        self,
        request: ManagedOperationStatusRequest,
    ) -> ManagedOperationStatusResult:
        return await self._managed_operation_request(
            path="/v1/providers/openrouter/managed-operation/status",
            request=request,
            operation="managed_operation_status",
        )

    async def resume_managed_operation(
        self,
        request: ManagedOperationResumeRequest,
    ) -> ManagedOperationStatusResult:
        return await self._managed_operation_request(
            path="/v1/providers/openrouter/managed-operation/resume",
            request=request,
            operation="managed_operation_resume",
        )

    async def _managed_operation_request(
        self,
        *,
        path: str,
        request: ManagedOperationStatusRequest | ManagedOperationResumeRequest,
        operation: str,
    ) -> ManagedOperationStatusResult:
        try:
            payload, status_code = await self._post_operation_json(
                path=path,
                request_body=_managed_operation_request_body(request),
                operation=operation,
            )
        except ManagedOpenRouterReleaseError as exc:
            return _managed_operation_failure_from_error(exc)
        if status_code == 404 and payload.get("code") == "invalid_request":
            return ManagedOperationStatusResult(
                succeeded=False,
                operation_status="UNKNOWN_OPERATION",
                client_action="wait",
                message=None,
                diagnostics=_parse_error_response(
                    payload.response, operation=operation
                ).failure_diagnostics,
                failed_reason=None,
            )
        try:
            _validate_operation_response_binding(
                payload,
                operation_id=request.operation_id,
                installation_id=request.installation_id,
                source=request.source,
            )
        except ValueError:
            return _managed_operation_failure_from_error(
                _retryable_error(operation, "broker returned mismatched operation", payload=payload)
            )
        if payload.get("code") == "authorization_expired":
            return _managed_operation_failure_from_error(
                _parse_error_response(payload.response, operation=operation)
            )
        if payload.get("code") == "resume_issuance_unavailable":
            try:
                result = _parse_managed_operation_status(
                    payload, operation=operation, expected_source=request.source
                )
            except ValueError:
                return _managed_operation_failure_from_error(
                    _parse_error_response(payload.response, operation=operation)
                )
            error = _parse_error_response(payload.response, operation=operation)
            return replace(
                result, client_action="action_required", diagnostics=error.failure_diagnostics
            )
        try:
            return _parse_managed_operation_status(
                payload, operation=operation, expected_source=request.source
            )
        except ValueError:
            return _managed_operation_failure_from_error(
                _retryable_error(operation, "broker returned malformed operation", payload=payload)
            )

    async def _post_operation_json(
        self,
        *,
        path: str,
        request_body: Mapping[str, object],
        operation: str,
    ) -> tuple[Mapping[str, object], int]:
        response = await self._request(
            "POST", path=path, operation=operation, json=dict(request_body)
        )
        if response.is_error:
            try:
                payload = _parse_json_mapping(response, operation=operation)
            except ManagedOpenRouterReleaseError:
                raise _parse_error_response(response, operation=operation) from None
            if response.status_code == 404 and payload.get("code") == "invalid_request":
                return payload, response.status_code
            if payload.get("state") is not None and payload.get("client_action") is not None:
                return payload, response.status_code
            if payload.get("code") == "authorization_expired":
                return payload, response.status_code
            raise _parse_error_response(response, operation=operation)
        return _parse_json_mapping(response, operation=operation), response.status_code

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.aclose()

    async def _post_json(
        self,
        *,
        path: str,
        request_body: Mapping[str, object],
        operation: str,
    ) -> Mapping[str, object]:
        response = await self._request(
            "POST", path=path, operation=operation, json=dict(request_body)
        )
        if response.is_error:
            raise _parse_error_response(response, operation=operation)
        return _parse_json_mapping(response, operation=operation)

    async def _request(self, method: str, *, path: str, operation: str, **kwargs):
        try:
            client = await self._get_http_client()
            return await client.request(method, path, **kwargs)
        except NetworkConfigurationError as exc:
            raise _broker_error(
                operation=operation,
                code="invalid_request",
                error_class="terminal",
                subcode=exc.code,
                message="Correct the configured proxy or CA trust settings.",
                fields=safe_fields(exception_fields(exc)),
            ) from exc
        except httpx.HTTPError as exc:
            fields = safe_fields(exception_fields(exc))
            fields.setdefault("dispatch_certainty", "possible")
            raise _broker_error(
                operation=operation,
                code=RETRYABLE_ERROR_CODE,
                error_class=RETRYABLE_ERROR_CLASS,
                message="Broker request failed.",
                http_status=fields.get("http_status"),
                fields=fields,
            ) from exc

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is None:
                normalized_base_url = self.base_url.strip().rstrip("/")
                self._client = network_clients.external_async_client(
                    base_url=normalized_base_url,
                    timeout=self.timeout,
                    transport=self.transport,
                )
            return self._client


def _parse_delivery_ack_metadata(
    payload: Mapping[str, object],
    *,
    source: str,
) -> ManagedKeyDeliveryAckMetadata | None:
    if payload.get("delivery_ack_required") is not True:
        return None
    try:
        delivery_id = _require_text(payload, "delivery_id")
        managed_credential_ref = _require_text(payload, "managed_credential_ref")
        delivery_ack_token = _require_text(payload, "delivery_ack_token")
        expires_at = _require_optional_text(payload, "delivery_ack_expires_at")
    except ValueError as exc:
        raise ValueError(
            f"broker returned malformed delivery ACK metadata: {_safe_field_label(str(exc))}"
        ) from exc
    return ManagedKeyDeliveryAckMetadata(
        source=source,
        delivery_id=delivery_id,
        managed_credential_ref=managed_credential_ref,
        expires_at=expires_at,
        delivery_ack_token=delivery_ack_token,
    )


def _delivery_ack_required(payload: Mapping[str, object]) -> bool:
    return payload.get("delivery_ack_required") is True


def _delivery_ack_text(payload: Mapping[str, object], key: str) -> str | None:
    if _delivery_ack_required(payload):
        return _require_text(payload, key)
    return _require_optional_text(payload, key)


def _ack_failure_from_error(error: ManagedOpenRouterReleaseError) -> ManagedKeyDeliveryAckResult:
    retryable = error.error_class == RETRYABLE_ERROR_CLASS or error.code == RETRYABLE_ERROR_CODE
    subcode = error.subcode or error.code
    status = (
        "retryable"
        if retryable
        else {
            "delivery_ack_invalid": "invalid",
            "delivery_ack_expired": "expired",
            "delivery_ack_mismatched": "mismatched",
        }.get(subcode, subcode)
    )
    return ManagedKeyDeliveryAckResult(
        succeeded=False,
        status=status,
        message=None,
        diagnostics=replace(
            error.failure_diagnostics,
            code="managed_key_delivery_ack_error",
            fields={
                **error.failure_diagnostics.fields,
                **safe_fields(exception_fields(error)),
                **safe_fields({"ack_status": status}),
            },
        ),
    )


def _ack_diagnostics(
    *,
    code: str,
    status: str,
    fields: Mapping[str, messages.DiagnosticFieldValue],
    retry_after_ms: int | None = None,
) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="managed_openrouter_broker_client",
        operation="managed_key_delivery_ack",
        code=code,
        category=messages.DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=fields.get("http_status"),
        retry_after_ms=retry_after_ms,
        fields={"ack_status": status, **{k: v for k, v in fields.items() if v is not None}},
    )


def _managed_operation_request_body(
    request: ManagedOperationStatusRequest | ManagedOperationResumeRequest,
) -> dict[str, object]:
    body: dict[str, object] = {"operation_id": request.operation_id}
    if request.installation_id:
        body["installation_id"] = request.installation_id
    if request.resume_token:
        body["resume_token"] = request.resume_token
    return body


def _managed_operation_error_is_unknown_operation(error: ManagedOpenRouterReleaseError) -> bool:
    candidates = (error.code or "", error.subcode or "")
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if not normalized:
            continue
        if normalized in {"operation_not_found", "unknown_operation", "unknown_operation_id"}:
            return True
        if "not_found" in normalized or "unknown_operation" in normalized:
            return True
    return False


def _validate_operation_response_binding(payload, *, operation_id, installation_id, source) -> None:
    for name, expected in (
        ("operation_id", operation_id),
        ("installation_id", installation_id),
        ("issue_source", source),
    ):
        echoed = payload.get(name)
        if echoed is not None and expected is not None and echoed != expected:
            raise ValueError("broker returned mismatched operation binding")
    echoed_source = payload.get("issue_source")
    if echoed_source is not None and echoed_source not in ("discord", "qq"):
        raise ValueError("broker returned unknown operation source")
    delivery = payload.get("delivery")
    if isinstance(delivery, Mapping) and payload.get("delivery_id") is not None:
        if delivery.get("delivery_id") != payload["delivery_id"]:
            raise ValueError("broker returned mismatched delivery")
    attempt = _parse_managed_operation_attempts(payload.get("attempts"))
    credential_ref = payload.get("managed_credential_ref")
    if attempt is not None and attempt.managed_credential_ref and credential_ref is not None:
        if attempt.managed_credential_ref != credential_ref:
            raise ValueError("broker returned mismatched credential reference")


def _managed_operation_failure_from_error(
    error: ManagedOpenRouterReleaseError,
) -> ManagedOperationStatusResult:
    retryable = error.error_class == RETRYABLE_ERROR_CLASS
    if _managed_operation_error_is_unknown_operation(error):
        return ManagedOperationStatusResult(
            succeeded=False,
            operation_status="UNKNOWN_OPERATION",
            client_action="wait",
            message=None,
            diagnostics=None,
            failed_reason=None,
        )
    if error.code == "authorization_expired" or error.subcode == "authorization_expired":
        return ManagedOperationStatusResult(
            succeeded=True,
            operation_status="FAILED",
            client_action="action_required",
            message=None,
            diagnostics=error.failure_diagnostics,
            failed_reason="authorization_expired",
        )
    return ManagedOperationStatusResult(
        succeeded=False,
        operation_status="CREATE_UNKNOWN",
        client_action="wait" if retryable else "action_required",
        message=None,
        diagnostics=error.failure_diagnostics,
        failed_reason=None,
    )


def _parse_managed_operation_credential(
    payload: Mapping[str, object],
    *,
    operation: str,
    expected_source: str | None,
) -> ManagedOperationStatusResult:
    try:
        managed_secret_key = _require_text(payload, "openrouter_api_key")
        delivery_ack = _parse_managed_operation_delivery_ack(
            payload, expected_source=expected_source
        )
    except ValueError as exc:
        raise ValueError(
            f"broker returned malformed managed operation payload: {_safe_field_label(str(exc))}"
        ) from exc
    if delivery_ack is None:
        raise ValueError("broker returned managed operation credential without delivery metadata")
    return ManagedOperationStatusResult(
        succeeded=True,
        operation_status="DELIVERY_PENDING",
        client_action="acknowledge_delivery",
        message=None,
        diagnostics=None,
        attempt=None,
        delivery=None,
        failed_reason=None,
        qq_subject_ref=_require_optional_text(payload, "qq_subject_ref"),
        managed_secret_key=managed_secret_key,
        managed_credential_ref=_require_optional_text(payload, "managed_credential_ref"),
        expires_at=_require_optional_text(payload, "expires_at"),
        openrouter_user_id=normalize_managed_openrouter_user_identifier(
            payload.get("openrouter_user_id")
        ),
        delivery_ack=delivery_ack,
    )


def _parse_managed_operation_status(
    payload: Mapping[str, object],
    *,
    operation: str,
    expected_source: str | None,
) -> ManagedOperationStatusResult:
    if "state" not in payload or "client_action" not in payload:
        return _parse_managed_operation_credential(
            payload, operation=operation, expected_source=expected_source
        )
    try:
        operation_status = _require_text(payload, "state")
        client_action = _require_text(payload, "client_action")
    except ValueError as exc:
        raise ValueError(
            f"broker returned malformed managed operation payload: {_safe_field_label(str(exc))}"
        ) from exc
    if operation_status not in {
        "AUTHENTICATED",
        "ISSUE_READY",
        "CREATING",
        "CREATE_UNKNOWN",
        "RECONCILING",
        "CLEANUP_REQUIRED",
        "CLEAN",
        "RETRY_READY",
        "DELIVERY_PENDING",
        "ACTIVE",
        "FAILED",
    }:
        raise ValueError(
            f"broker returned unknown operation status: {_safe_field_label(operation_status)}"
        )
    if client_action not in {"wait", "retry_authorized", "acknowledge_delivery", "action_required"}:
        raise ValueError(
            f"broker returned unknown client action: {_safe_field_label(client_action)}"
        )
    attempt = _parse_managed_operation_attempts(payload.get("attempts"))
    delivery = _parse_managed_operation_delivery(payload.get("delivery"))
    failed_reason = payload.get("failure_reason")
    if not isinstance(failed_reason, str) or failed_reason not in {
        "authorization_expired",
        "terminal_provider_failure",
        "cleanup_failed_terminal",
    }:
        failed_reason = None
    if operation_status != "FAILED":
        failed_reason = None
    return ManagedOperationStatusResult(
        succeeded=True,
        operation_status=operation_status,
        client_action=client_action,
        message=None,
        diagnostics=(
            _broker_error(
                operation=operation,
                code=(
                    "authorization_expired"
                    if failed_reason == "authorization_expired"
                    else (
                        "resume_issuance_unavailable"
                        if payload.get("code") == "resume_issuance_unavailable"
                        else "invalid_request"
                    )
                ),
                error_class="terminal",
                message="Managed operation requires user action.",
                http_status=payload.response.status_code,
                fields=_response_fields(payload.response, shape="rich_operation"),
            ).failure_diagnostics
            if client_action == "action_required" and isinstance(payload, _BrokerPayload)
            else None
        ),
        attempt=attempt,
        delivery=delivery,
        failed_reason=failed_reason if isinstance(failed_reason, str) else None,
        managed_secret_key=_require_optional_text(payload, "openrouter_api_key"),
        managed_credential_ref=_require_optional_text(payload, "managed_credential_ref"),
        expires_at=_require_optional_text(payload, "expires_at"),
        openrouter_user_id=normalize_managed_openrouter_user_identifier(
            payload.get("openrouter_user_id")
        ),
        delivery_ack=_parse_managed_operation_delivery_ack(
            payload, expected_source=expected_source
        ),
    )


def _parse_managed_operation_attempts(value: object) -> ManagedOperationAttemptSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("broker returned malformed managed operation attempts")
    if not value:
        return None
    return _parse_managed_operation_attempt(value[-1])


def _parse_managed_operation_attempt(value: object) -> ManagedOperationAttemptSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("broker returned malformed managed operation attempt")
    attempt_index = value.get("attempt_index")
    if attempt_index is not None and (
        isinstance(attempt_index, bool) or not isinstance(attempt_index, int)
    ):
        raise ValueError("broker returned malformed managed operation attempt index")
    provider_key_name = value.get("provider_key_name")
    if provider_key_name is not None and not isinstance(provider_key_name, str):
        raise ValueError("broker returned malformed managed operation provider key name")
    managed_credential_ref = value.get("managed_credential_ref")
    if managed_credential_ref is not None and not isinstance(managed_credential_ref, str):
        raise ValueError("broker returned malformed managed operation credential reference")
    outcome = value.get("outcome")
    if outcome is not None and outcome not in ("created", "unknown", "cleaned"):
        raise ValueError("broker returned malformed managed operation attempt outcome")
    return ManagedOperationAttemptSnapshot(
        attempt_index=attempt_index,
        provider_key_name=provider_key_name,
        managed_credential_ref=managed_credential_ref,
        outcome=outcome,
    )


def _parse_managed_operation_delivery(value: object) -> ManagedOperationDeliverySnapshot | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("broker returned malformed managed operation delivery")
    delivery_id = value.get("delivery_id")
    if delivery_id is not None and not isinstance(delivery_id, str):
        raise ValueError("broker returned malformed managed operation delivery id")
    status = value.get("status")
    if status is not None and status not in (
        "pending",
        "acknowledged",
        "expired",
        "cleanup_required",
    ):
        raise ValueError("broker returned malformed managed operation delivery status")
    expires_at = value.get("expires_at")
    if expires_at is not None and not isinstance(expires_at, str):
        raise ValueError("broker returned malformed managed operation delivery expiry")
    return ManagedOperationDeliverySnapshot(
        delivery_id=delivery_id,
        status=status,
        expires_at=expires_at,
    )


def _parse_managed_operation_delivery_ack(
    payload: Mapping[str, object],
    *,
    expected_source: str | None,
) -> ManagedKeyDeliveryAckMetadata | None:
    if payload.get("delivery_ack_required") is not True:
        return None
    issue_source = payload.get("issue_source")
    if not isinstance(issue_source, str) or not issue_source:
        issue_source = expected_source
    if issue_source not in ("discord", "qq"):
        raise ValueError("broker returned managed operation delivery without a known issue source")
    if expected_source is not None and issue_source != expected_source:
        raise ValueError(
            "broker returned managed operation delivery for an unexpected issue source"
        )
    try:
        return ManagedKeyDeliveryAckMetadata(
            source=issue_source,
            delivery_id=_require_text(payload, "delivery_id"),
            managed_credential_ref=_require_text(payload, "managed_credential_ref"),
            expires_at=_require_optional_text(payload, "delivery_ack_expires_at"),
            delivery_ack_token=_require_text(payload, "delivery_ack_token"),
        )
    except ValueError as exc:
        raise ValueError(
            f"broker returned malformed delivery ACK metadata: {_safe_field_label(str(exc))}"
        ) from exc


def _qq_assertion_failure_from_error(
    error: ManagedOpenRouterReleaseError,
) -> QqManagedAssertionResult:
    transport_fields = safe_fields(exception_fields(error))
    failure_subcode = (
        "broker_unavailable"
        if transport_fields.get("network_error_kind") != "unknown"
        else _qq_safe_failure_subcode(error)
    )
    result = _qq_assertion_failure(
        failure_subcode=failure_subcode,
        code=error.code,
        operation=error.operation or "qq_assert",
        category=_qq_failure_category(failure_subcode),
        retry_after_ms=error.retry_after_ms,
        fields=transport_fields,
    )
    return replace(
        result,
        diagnostics=replace(
            error.failure_diagnostics,
            fields={**error.failure_diagnostics.fields, **result.diagnostics.fields},
        ),
    )


def _qq_assertion_failure(
    *,
    failure_subcode: QqManagedAssertionFailureSubcode,
    code: str,
    operation: str,
    category: messages.DiagnosticCategory,
    retry_after_ms: int | None,
    fields: Mapping[str, messages.DiagnosticFieldValue],
) -> QqManagedAssertionResult:
    diagnostics = messages.ErrorDiagnostics(
        component="managed_openrouter_broker_client",
        operation=operation,
        code=code,
        category=category,
        visibility=messages.DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=fields.get("http_status") if type(fields.get("http_status")) is int else None,
        retry_after_ms=retry_after_ms,
        fields={
            "qq_failure_subcode": failure_subcode,
            **{key: value for key, value in fields.items() if value is not None},
        },
    )
    return QqManagedAssertionResult(
        succeeded=False,
        managed_secret_key=None,
        entitlement=None,
        failure_subcode=failure_subcode,
        retry_after_ms=retry_after_ms,
        message=None,
        diagnostics=diagnostics,
    )


def _qq_safe_failure_subcode(
    error: ManagedOpenRouterReleaseError,
) -> QqManagedAssertionFailureSubcode:
    if error.subcode == "qq_credential_invalid":
        return "invalid_credential"
    if error.subcode in {
        "qq_identity_mismatch",
        "qq_subject_mismatch",
        "installation_binding_mismatch",
        "device_public_key_registered",
    }:
        return "mismatch"
    if error.subcode == "qq_lifetime_used":
        return "lifetime_used"
    if error.code == "rate_limited" or error.retry_after_ms is not None:
        return "rate_limited"
    if error.code in {"internal_error", "trial_unavailable"}:
        return "key_unavailable"
    return "broker_unavailable"


def _qq_failure_category(
    subcode: QqManagedAssertionFailureSubcode,
) -> messages.DiagnosticCategory:
    if subcode in {"invalid_credential", "mismatch"}:
        return messages.DIAGNOSTIC_CATEGORY_AUTH
    if subcode == "rate_limited":
        return messages.DIAGNOSTIC_CATEGORY_RATE_LIMIT
    if subcode == "lifetime_used":
        return messages.DIAGNOSTIC_CATEGORY_QUOTA
    return messages.DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE


def _safe_field_label(value: str) -> str:
    stripped = value.strip()
    safe = "".join(char if char.isalnum() or char in "._:-" else "_" for char in stripped)
    return safe[:64] or "invalid"


def _parse_error_response(
    response: httpx.Response, *, operation: str
) -> ManagedOpenRouterReleaseError:
    try:
        payload = _parse_json_mapping(response, operation=operation)
    except ManagedOpenRouterReleaseError as exc:
        fields = exc.failure_diagnostics.fields
        return _http_rejection(response, operation=operation, fields=fields)
    raw_error = payload.get("error")
    nested = isinstance(raw_error, Mapping)
    raw_error = raw_error if nested else payload
    shape = "public_error_envelope" if nested else "flat_error"
    fields = _response_fields(response, shape=shape)
    retry_after = _bounded_retry_after(raw_error.get("retry_after_ms"))
    if retry_after is None:
        retry_after = _header_retry_after(response)
    managed_state = payload.get("managed_state")
    lifecycle = managed_state.get("lifecycle") if isinstance(managed_state, Mapping) else None
    try:
        code = _require_public_error_code(raw_error, "code")
        error_class = (
            _require_public_error_class(raw_error, "class")
            if nested
            else _http_error_class(response.status_code)
        )
        if code in {"authorization_expired", "resume_issuance_unavailable"}:
            error_class = "terminal"
        raw_message = raw_error.get("message")
        return _broker_error(
            operation=operation,
            code=code,
            error_class=error_class,
            subcode=_require_optional_text(raw_error, "subcode"),
            retry_after_ms=retry_after,
            http_status=response.status_code,
            message=(
                raw_message
                if isinstance(raw_message, str) and raw_message
                else "Broker request failed."
            ),
            managed_lifecycle=lifecycle if isinstance(lifecycle, str) else None,
            fields=fields,
        )
    except ValueError:
        return _with_response(
            _retryable_error(operation, "broker returned malformed error payload"),
            response,
            shape=shape,
        )


def _parse_json_mapping(response: httpx.Response, *, operation: str) -> Mapping[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise _with_response(
            _retryable_error(operation, "broker returned malformed JSON"),
            response,
            shape="non_json",
        ) from exc
    if not isinstance(payload, Mapping):
        raise _with_response(
            _retryable_error(operation, "broker returned a non-object JSON payload"),
            response,
            shape="non_object",
        )
    return _BrokerPayload(payload, response)


def _parse_fingerprint_salt(
    payload: Mapping[str, object],
    *,
    operation: str,
) -> ManagedOpenRouterFingerprintSalt:
    raw_fingerprint_salt = payload.get("fingerprint_salt")
    if not isinstance(raw_fingerprint_salt, Mapping):
        raise _retryable_error(
            operation, "broker returned malformed fingerprint_salt payload", payload=payload
        )
    return ManagedOpenRouterFingerprintSalt(
        version=_require_int(raw_fingerprint_salt, "version"),
        salt=_require_text(raw_fingerprint_salt, "salt"),
    )


def _require_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_public_error_code(payload: Mapping[str, object], key: str) -> str:
    value = _require_text(payload, key)
    if value not in PUBLIC_ERROR_CODES:
        raise ValueError(f"{key} must be a supported public error code")
    return value


def _require_public_error_class(payload: Mapping[str, object], key: str) -> str:
    value = _require_text(payload, key)
    if value not in PUBLIC_ERROR_CLASSES:
        raise ValueError(f"{key} must be a supported public error class")
    return value


def _require_optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _retryable_error(
    operation: str, detail: str, *, payload: Mapping[str, object] | None = None
) -> ManagedOpenRouterReleaseError:
    error = _broker_error(
        operation=operation,
        code=RETRYABLE_ERROR_CODE,
        error_class=RETRYABLE_ERROR_CLASS,
        message=f"managed OpenRouter broker {operation} failed: {detail}",
    )
    if isinstance(payload, _BrokerPayload):
        return _with_response(error, payload.response, shape="malformed_success")
    return error


def _broker_error(
    *,
    operation,
    code,
    error_class,
    message,
    subcode=None,
    http_status=None,
    retry_after_ms=None,
    managed_lifecycle=None,
    fields=None,
) -> ManagedOpenRouterReleaseError:
    fields = safe_fields(
        {
            **(fields or {}),
            "broker_code": code,
            "broker_subcode": subcode,
            "broker_class": error_class,
        }
    )
    category = (
        messages.DIAGNOSTIC_CATEGORY_RATE_LIMIT
        if code == "rate_limited"
        else (
            messages.DIAGNOSTIC_CATEGORY_LIFECYCLE
            if fields.get("dispatch_certainty") == "not_dispatched"
            else (
                messages.DIAGNOSTIC_CATEGORY_TIMEOUT
                if fields.get("network_error_kind") == "timeout"
                else (
                    messages.DIAGNOSTIC_CATEGORY_NETWORK
                    if fields.get("network_error_kind") in {"proxy", "tls", "dns", "network"}
                    else (
                        messages.DIAGNOSTIC_CATEGORY_AUTH
                        if error_class != "retryable"
                        else (
                            messages.DIAGNOSTIC_CATEGORY_INVALID_RESPONSE
                            if fields.get("stage") == "protocol_parse"
                            else messages.DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE
                        )
                    )
                )
            )
        )
    )
    diagnostics = messages.ErrorDiagnostics(
        component="managed_openrouter_broker_client",
        operation=operation,
        code=code,
        category=category,
        visibility=messages.DIAGNOSTIC_VISIBILITY_DIAGNOSTIC_ONLY,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=http_status,
        retry_after_ms=retry_after_ms,
        fields=fields,
    )
    return ManagedOpenRouterReleaseError(
        operation=operation,
        code=code,
        error_class=error_class,
        message=message,
        subcode=subcode,
        http_status=http_status,
        retry_after_ms=retry_after_ms,
        managed_lifecycle=managed_lifecycle,
        failure_diagnostics=diagnostics,
    )


def _response_fields(response, *, shape):
    return safe_fields(
        {
            **response.extensions.get("connection_diagnostics", {}),
            "stage": "protocol_parse",
            "dispatch_certainty": "response_received",
            "http_status": response.status_code,
            "response_shape": shape,
        }
    )


def _with_response(error, response, *, shape):
    diagnostics = replace(
        error.failure_diagnostics,
        status_code=response.status_code,
        category=messages.DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
        retry_after_ms=(
            error.retry_after_ms
            if error.retry_after_ms is not None
            else _header_retry_after(response)
        ),
        fields={**error.failure_diagnostics.fields, **_response_fields(response, shape=shape)},
    )
    return replace(
        error,
        http_status=response.status_code,
        retry_after_ms=diagnostics.retry_after_ms,
        failure_diagnostics=diagnostics,
    )


def _http_error_class(status):
    return "terminal" if status in {400, 401, 403, 404, 410, 422} else "retryable"


def _http_rejection(response, *, operation, fields):
    code = (
        "rate_limited"
        if response.status_code == 429
        else (
            "invalid_request"
            if _http_error_class(response.status_code) == "terminal"
            else RETRYABLE_ERROR_CODE
        )
    )
    return _broker_error(
        operation=operation,
        code=code,
        error_class=_http_error_class(response.status_code),
        subcode="access_rejected" if response.status_code in {401, 403} else None,
        message=(
            "Broker rejected the request."
            if code == "invalid_request"
            else "Broker request is temporarily unavailable."
        ),
        http_status=response.status_code,
        retry_after_ms=_header_retry_after(response),
        fields=fields,
    )


def _bounded_retry_after(value):
    return min(value, 86400000) if type(value) is int and value >= 0 else None


def _header_retry_after(response):
    value = response.headers.get("Retry-After", "").strip()
    if value.isascii() and value.isdecimal() and len(value) <= 12:
        return min(int(value) * 1000, 86400000)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        return min(max(0, int((date - datetime.now(UTC)).total_seconds() * 1000)), 86400000)
    except ValueError, TypeError, OverflowError:
        return None


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("broker base_url must be a non-empty string")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.path not in {"", "/"}:
        raise ValueError("broker base_url must not include a path prefix")
    return normalized
