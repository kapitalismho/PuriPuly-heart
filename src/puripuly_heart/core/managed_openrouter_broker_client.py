from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

from puripuly_heart.config.settings import normalize_owned_referral_id
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterChallengeSuccess,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterPreflightStop,
    ManagedOpenRouterQqAssertSuccess,
    ManagedOpenRouterReleaseError,
    ManagedOpenRouterTrialStatusSuccess,
    ManagedOpenRouterVerifySuccess,
    TalkTogetherPassStatus,
)
from puripuly_heart.core.openrouter_credentials import (
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
QQ_ASSERT_VERIFIED_STATUSES = frozenset({"verified", "already_verified"})
QQ_ASSERT_WIRE_STATUSES = QQ_ASSERT_VERIFIED_STATUSES | {"issued"}
QQ_ASSERT_PUBLIC_SUBCODES = frozenset(
    {
        "qq_credential_invalid",
        "qq_credential_mismatch",
        "qq_lifetime_used",
        "ip_rate_limited",
    }
)
QQ_ASSERT_CREDENTIAL_SUBCODES = frozenset({"qq_credential_invalid", "qq_credential_mismatch"})


@dataclass(slots=True)
class HttpManagedOpenRouterBrokerClient:
    base_url: str
    timeout: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _client_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(
            self.base_url,
            allow_insecure_scheme=self.transport is not None,
        )

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
            request_body=request,
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
            )
        except ValueError as exc:
            raise _retryable_error(
                "discord_issue", f"broker returned malformed payload: {exc}"
            ) from exc

    async def assert_qq_credential(
        self,
        *,
        qq_identity: str,
        credential: str,
        asserted_at: str,
    ) -> ManagedOpenRouterQqAssertSuccess:
        logger.info("[QQAuth] assert_qq_credential: posting QQ assertion request")
        payload = await self._post_json(
            path="/v1/auth/qq/assert",
            request_body={
                "qq_identity": qq_identity,
                "credential": credential,
                "asserted_at": asserted_at,
            },
            operation="qq_assert",
        )
        try:
            result = _parse_qq_assert_success(payload)
            logger.info(
                "[QQAuth] assert_qq_credential: broker accepted assertion status=%s issued=%s",
                result.status,
                result.issue is not None,
            )
            return result
        except ValueError as exc:
            logger.warning("[QQAuth] assert_qq_credential: malformed broker payload")
            raise _retryable_error(
                "qq_assert", "broker returned malformed QQ assertion payload"
            ) from exc

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


def _parse_json_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if abs(value) > _MAX_SAFE_JSON_INTEGER:
        return None
    return value


def _parse_qq_assert_success(payload: Mapping[str, object]) -> ManagedOpenRouterQqAssertSuccess:
    if payload.get("ok") is not True:
        raise ValueError("ok must be true")

    wire_status = _require_text(payload, "status")
    if wire_status not in QQ_ASSERT_WIRE_STATUSES:
        raise ValueError("status must be verified, already_verified, or issued")

    qq_subject_ref = _require_text(payload, "qq_subject_ref")
    openrouter_api_key = _normalize_qq_assert_openrouter_api_key(payload.get("openrouter_api_key"))

    if openrouter_api_key is None:
        if wire_status == "issued":
            raise ValueError("issued status requires openrouter_api_key")
        return ManagedOpenRouterQqAssertSuccess(
            status=wire_status,
            qq_subject_ref=qq_subject_ref,
            issue=None,
        )

    return ManagedOpenRouterQqAssertSuccess(
        status="issued",
        qq_subject_ref=qq_subject_ref,
        issue=ManagedOpenRouterIssueSuccess(
            openrouter_api_key=openrouter_api_key,
            managed_credential_ref=_require_optional_stripped_text(
                payload,
                "managed_credential_ref",
            ),
            expires_at=_require_optional_text(payload, "expires_at"),
            openrouter_user_id=normalize_managed_openrouter_user_identifier(
                payload.get("openrouter_user_id")
            ),
            referral_bonus_applied=_parse_referral_bonus_applied(payload),
            referral_id=_parse_owned_referral_id(payload),
            pass_status=_parse_talk_together_pass_status(payload),
        ),
    )


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

    if operation == "qq_assert":
        return _parse_qq_assert_error_response(
            raw_error,
            operation=operation,
            managed_lifecycle=managed_lifecycle,
        )

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


def _parse_qq_assert_error_response(
    raw_error: Mapping[str, object],
    *,
    operation: str,
    managed_lifecycle: str | None,
) -> ManagedOpenRouterReleaseError:
    try:
        subcode = _normalize_qq_assert_subcode(raw_error.get("subcode"))
        return ManagedOpenRouterReleaseError(
            operation=operation,
            code=_require_public_error_code(raw_error, "code"),
            error_class=_require_public_error_class(raw_error, "class"),
            subcode=subcode,
            retry_after_ms=_normalize_qq_assert_retry_after_ms(raw_error.get("retry_after_ms")),
            message=_qq_assert_error_message(subcode),
            managed_lifecycle=managed_lifecycle,
        )
    except ValueError as exc:
        return _retryable_error(
            operation,
            f"broker returned malformed QQ assertion error payload: {exc}",
        )


def _normalize_qq_assert_subcode(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value not in QQ_ASSERT_PUBLIC_SUBCODES:
        return None
    return value


def _normalize_qq_assert_retry_after_ms(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(0, min(value, _MAX_SAFE_JSON_INTEGER))


def _normalize_qq_assert_openrouter_api_key(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("openrouter_api_key must be a string or null")
    return value.strip() or None


def _qq_assert_error_message(subcode: str | None) -> str:
    if subcode in QQ_ASSERT_CREDENTIAL_SUBCODES:
        return "QQ credential verification failed"
    if subcode == "qq_lifetime_used":
        return "QQ credential has already been used"
    if subcode == "ip_rate_limited":
        return "QQ credential verification is rate limited"
    return "QQ credential verification failed"


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


def _require_optional_stripped_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    normalized = _normalize_optional_stripped_text(value)
    if normalized is None:
        raise ValueError(f"{key} must be a non-empty string or null")
    return normalized


def _normalize_optional_stripped_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


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


def _normalize_base_url(base_url: str, *, allow_insecure_scheme: bool = False) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("broker base_url must be a non-empty string")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("broker base_url must not include userinfo")
    if parsed.query:
        raise ValueError("broker base_url must not include a query string")
    if parsed.path not in {"", "/"}:
        raise ValueError("broker base_url must not include a path prefix")
    if parsed.scheme != "https" and not (allow_insecure_scheme and parsed.scheme == "http"):
        raise ValueError("broker base_url must use HTTPS")
    return normalized
