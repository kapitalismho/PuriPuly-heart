from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckRequest,
    ManagedKeyDeliveryAckResult,
    QqManagedAssertionFailureSubcode,
    QqManagedAssertionRequest,
    QqManagedAssertionResult,
    QqManagedEntitlementSnapshot,
)
from puripuly_heart.config.provider_values import normalize_owned_referral_id
from puripuly_heart.core import messages

from .managed_openrouter_release import (
    ManagedOpenRouterChallengeSuccess,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterPreflightStop,
    ManagedOpenRouterReleaseError,
    ManagedOpenRouterTrialStatusSuccess,
    ManagedOpenRouterVerifySuccess,
    TalkTogetherPassStatus,
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
    }
)
PUBLIC_ERROR_CLASSES = frozenset({"retryable", "terminal", "security_fail"})


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
                "challenge", f"broker returned malformed payload: {exc}"
            ) from exc

    async def start_discord_oauth(
        self,
        *,
        installation_id: str,
        device_public_key: str,
        redirect_uri: str,
        app_version: str,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterDiscordStartSuccess:
        request_body = {
            "installation_id": installation_id,
            "device_public_key": device_public_key,
            "redirect_uri": redirect_uri,
            "app_version": app_version,
        }
        normalized_referral_id = _normalize_friend_referral_id(referral_id)
        if normalized_referral_id is not None:
            request_body["referral_id"] = normalized_referral_id

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
                "discord_start", f"broker returned malformed payload: {exc}"
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
            raise _retryable_error("verify", f"broker returned malformed payload: {exc}") from exc

    async def issue(self, request: dict[str, object]) -> ManagedOpenRouterIssueSuccess:
        payload = await self._post_json(
            path="/v1/providers/openrouter/issue",
            request_body=request,
            operation="issue",
        )
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
            raise _retryable_error("issue", f"broker returned malformed payload: {exc}") from exc

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
            return ManagedOpenRouterIssueSuccess(
                openrouter_api_key=_require_text(payload, "openrouter_api_key"),
                managed_credential_ref=_require_optional_text(payload, "managed_credential_ref"),
                expires_at=_require_optional_text(payload, "expires_at"),
                openrouter_user_id=normalize_managed_openrouter_user_identifier(
                    payload.get("openrouter_user_id")
                ),
                referral_bonus_applied=_parse_referral_bonus_applied(payload),
                referral_id=_parse_owned_referral_id(payload),
                pass_status=_parse_talk_together_pass_status(payload),
                delivery_ack_required=_delivery_ack_required(payload),
                delivery_id=_delivery_ack_text(payload, "delivery_id"),
                delivery_ack_token=_delivery_ack_text(payload, "delivery_ack_token"),
                delivery_ack_expires_at=_require_optional_text(payload, "delivery_ack_expires_at"),
            )
        except ValueError as exc:
            raise _retryable_error(
                "discord_issue", f"broker returned malformed payload: {exc}"
            ) from exc

    async def assert_qq_managed_identity(
        self,
        request: QqManagedAssertionRequest,
    ) -> QqManagedAssertionResult:
        try:
            payload = await self._post_json(
                path="/v1/auth/qq/assert",
                request_body={
                    "qq_identity": request.qq_identity,
                    "credential": request.credential,
                    "asserted_at": request.asserted_at,
                    "delivery_ack_supported": True,
                },
                operation="qq_assert",
            )
        except ManagedOpenRouterReleaseError as exc:
            return _qq_assertion_failure_from_error(exc)

        status = payload.get("status")
        if status not in {"issued", "delivery_pending"}:
            return _qq_assertion_failure(
                failure_subcode="key_unavailable",
                code="qq_key_unavailable",
                operation="qq_assert",
                category=messages.DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
                retry_after_ms=None,
                fields={"status": status if isinstance(status, str) else "missing"},
            )
        try:
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
        except ValueError as exc:
            return _qq_assertion_failure(
                failure_subcode="key_unavailable",
                code="qq_malformed_success_payload",
                operation="qq_assert",
                category=messages.DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
                retry_after_ms=None,
                fields={"payload_valid": False, "reason": _safe_field_label(str(exc))},
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
        succeeded = payload.get("ok") is True and status in {
            "acknowledged",
            "already_acknowledged",
        }
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
                    fields={"broker_ok": payload.get("ok") is True},
                )
            ),
            referral_bonus_applied=_parse_referral_bonus_applied(payload),
            referral_id=_parse_owned_referral_id(payload),
            pass_status=_parse_talk_together_pass_status(payload),
        )

    async def get_trial_status(
        self,
        *,
        installation_id: str,
        timestamp: str,
        signature: str,
    ) -> ManagedOpenRouterTrialStatusSuccess:
        payload = await self._get_json(
            path="/v1/trial/status",
            params={"installation_id": installation_id},
            headers={
                "X-Puripuly-Timestamp": timestamp,
                "X-Puripuly-Signature": signature,
            },
            operation="trial_status",
        )
        return ManagedOpenRouterTrialStatusSuccess(
            referral_id=_parse_owned_referral_id(payload),
            pass_status=_parse_talk_together_pass_status(payload),
        )

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
        client = await self._get_http_client()
        try:
            response = await client.post(path, json=dict(request_body))
        except httpx.TimeoutException as exc:
            raise _retryable_error(operation, f"broker request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise _retryable_error(operation, f"broker transport failure: {exc}") from exc
        except httpx.HTTPError as exc:
            raise _retryable_error(operation, f"broker request failed: {exc}") from exc

        if response.is_error:
            raise _parse_error_response(response, operation=operation)

        return _parse_json_mapping(response, operation=operation)

    async def _get_json(
        self,
        *,
        path: str,
        params: Mapping[str, object],
        headers: Mapping[str, str],
        operation: str,
    ) -> Mapping[str, object]:
        client = await self._get_http_client()
        try:
            response = await client.get(path, params=dict(params), headers=dict(headers))
        except httpx.TimeoutException as exc:
            raise _retryable_error(operation, f"broker request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise _retryable_error(operation, f"broker transport failure: {exc}") from exc
        except httpx.HTTPError as exc:
            raise _retryable_error(operation, f"broker request failed: {exc}") from exc

        if response.is_error:
            raise _parse_error_response(response, operation=operation)

        return _parse_json_mapping(response, operation=operation)

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is None:
                normalized_base_url = self.base_url.strip().rstrip("/")
                self._client = httpx.AsyncClient(
                    base_url=normalized_base_url,
                    timeout=self.timeout,
                    transport=self.transport,
                )
            return self._client


def _normalize_friend_referral_id(referral_id: str | None) -> str | None:
    if not isinstance(referral_id, str):
        return None
    normalized = referral_id.strip().upper()
    return normalized or None


def _parse_referral_bonus_applied(payload: Mapping[str, object]) -> bool:
    return payload.get("referral_bonus_applied") is True


def _parse_owned_referral_id(payload: Mapping[str, object]) -> str | None:
    return normalize_owned_referral_id(payload.get("referral_id"))


_MAX_SAFE_JSON_INTEGER = 2**53 - 1
_DEFAULT_TALK_TOGETHER_PASS_BONUS_TRANSLATIONS = 200


def _parse_talk_together_pass_status(
    payload: Mapping[str, object],
) -> TalkTogetherPassStatus | None:
    owned_referral_id = _parse_owned_referral_id(payload)
    if owned_referral_id is None:
        return None

    raw_status = payload.get("talk_together_pass")
    if not isinstance(raw_status, Mapping):
        return None

    pass_id = normalize_owned_referral_id(raw_status.get("pass_id"))
    if pass_id != owned_referral_id:
        return None

    invite_count = _parse_json_int(raw_status.get("invite_count"))
    invite_limit = _parse_json_int(raw_status.get("invite_limit"))
    if invite_count is None or invite_count < 0:
        return None
    if invite_limit is None or invite_limit <= 0:
        return None

    bonus = _parse_json_int(raw_status.get("bonus_translations_per_friend"))
    if bonus is None or bonus <= 0:
        bonus = _DEFAULT_TALK_TOGETHER_PASS_BONUS_TRANSLATIONS

    return TalkTogetherPassStatus(
        pass_id=owned_referral_id,
        invite_count=invite_count,
        invite_limit=invite_limit,
        bonus_translations_per_friend=bonus,
    )


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
    status = "retryable" if retryable else subcode
    return ManagedKeyDeliveryAckResult(
        succeeded=False,
        status=status,
        message=None,
        diagnostics=_ack_diagnostics(
            code="managed_key_delivery_ack_error",
            status=status,
            retry_after_ms=error.retry_after_ms,
            fields={
                "broker_code": error.code,
                "broker_class": error.error_class,
                "broker_subcode": error.subcode,
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
        status_code=None,
        retry_after_ms=retry_after_ms,
        fields={"ack_status": status, **{k: v for k, v in fields.items() if v is not None}},
    )


def _qq_assertion_failure_from_error(
    error: ManagedOpenRouterReleaseError,
) -> QqManagedAssertionResult:
    failure_subcode = _qq_safe_failure_subcode(error)
    return _qq_assertion_failure(
        failure_subcode=failure_subcode,
        code=error.code,
        operation=error.operation or "qq_assert",
        category=_qq_failure_category(failure_subcode),
        retry_after_ms=error.retry_after_ms,
        fields={
            "broker_code": error.code,
            "broker_class": error.error_class,
            "broker_subcode": error.subcode,
        },
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
        status_code=None,
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


def _parse_json_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if abs(value) > _MAX_SAFE_JSON_INTEGER:
        return None
    return value


def _parse_error_response(
    response: httpx.Response, *, operation: str
) -> ManagedOpenRouterReleaseError:
    payload = _parse_json_mapping(response, operation=operation)
    raw_error = payload.get("error")
    if not isinstance(raw_error, Mapping):
        return _retryable_error(
            operation,
            f"broker returned an unexpected error payload (status={response.status_code})",
        )

    managed_lifecycle = None
    raw_managed_state = payload.get("managed_state")
    if isinstance(raw_managed_state, Mapping):
        lifecycle = raw_managed_state.get("lifecycle")
        if isinstance(lifecycle, str) and lifecycle:
            managed_lifecycle = lifecycle

    try:
        return ManagedOpenRouterReleaseError(
            operation=operation,
            code=_require_public_error_code(raw_error, "code"),
            error_class=_require_public_error_class(raw_error, "class"),
            subcode=_require_optional_text(raw_error, "subcode"),
            retry_after_ms=_require_optional_int(raw_error, "retry_after_ms"),
            message=_require_text(raw_error, "message"),
            managed_lifecycle=managed_lifecycle,
        )
    except ValueError as exc:
        return _retryable_error(operation, f"broker returned malformed error payload: {exc}")


def _parse_json_mapping(response: httpx.Response, *, operation: str) -> Mapping[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise _retryable_error(operation, "broker returned malformed JSON") from exc
    if not isinstance(payload, Mapping):
        raise _retryable_error(operation, "broker returned a non-object JSON payload")
    return payload


def _parse_fingerprint_salt(
    payload: Mapping[str, object],
    *,
    operation: str,
) -> ManagedOpenRouterFingerprintSalt:
    raw_fingerprint_salt = payload.get("fingerprint_salt")
    if not isinstance(raw_fingerprint_salt, Mapping):
        raise _retryable_error(operation, "broker returned malformed fingerprint_salt payload")
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


def _require_optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null")
    return value


def _retryable_error(operation: str, detail: str) -> ManagedOpenRouterReleaseError:
    return ManagedOpenRouterReleaseError(
        operation=operation,
        code=RETRYABLE_ERROR_CODE,
        error_class=RETRYABLE_ERROR_CLASS,
        message=f"managed OpenRouter broker {operation} failed: {detail}",
    )


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("broker base_url must be a non-empty string")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.path not in {"", "/"}:
        raise ValueError("broker base_url must not include a path prefix")
    return normalized
