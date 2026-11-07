from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.ports.provider_verifier import (
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    ProviderVerificationRequest,
    ProviderVerificationResult,
    ProviderVerifierPort,
)
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort, RuntimeApplyRequest
from puripuly_heart.app.services.operational_state_payloads import (
    ProviderVerificationEvidence,
    provider_verification_state_values,
)
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSetRequest,
    SecretSettingsTransaction,
    _is_sensitive_evidence_key,
    _sanitize_verification_fields,
    _secret_fingerprint,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    DiagnosticFieldValue,
    ErrorDiagnostics,
    TransactionResult,
)


@dataclass(frozen=True, slots=True)
class OpenRouterPkceHandoffRequest:
    provider: str
    secret_key: str
    transient_api_key: str = field(repr=False)
    settings_values: Mapping[str, object] = field(repr=False)
    expected_settings_revision: str | None
    reason: str | None
    correlation_id: str | None
    verifier_context: Mapping[str, DiagnosticFieldValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings_values", freeze_settings_values(self.settings_values))
        object.__setattr__(
            self,
            "verifier_context",
            MappingProxyType(
                _sanitize_verification_fields(
                    self.verifier_context,
                    secret_value=self.transient_api_key,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class OpenRouterPkceHandoffService:
    provider_verifier: ProviderVerifierPort
    secret_transaction: SecretSettingsTransaction
    runtime_apply: RuntimeApplyPort

    async def complete_handoff(
        self,
        request: OpenRouterPkceHandoffRequest,
    ) -> TransactionResult:
        if _caller_settings_values_are_unsafe(
            request.settings_values,
            secret_value=request.transient_api_key,
        ):
            return _unsafe_settings_values_result(request)

        verification_result = await self._verify_transient_key(request)
        if isinstance(verification_result, TransactionResult):
            return verification_result

        settings_values = _settings_values_with_verification_evidence(
            request=request,
            verification_result=verification_result,
        )
        local_commit_result = await self.secret_transaction.set_provider_secret(
            SecretSetRequest(
                secret_key=request.secret_key,
                secret_value=request.transient_api_key,
                settings_values=settings_values,
                expected_settings_revision=request.expected_settings_revision,
                reason=request.reason,
                correlation_id=request.correlation_id,
            )
        )
        if local_commit_result.status != TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
            return local_commit_result

        try:
            runtime_result = await self.runtime_apply.apply_runtime(
                RuntimeApplyRequest(
                    settings_values=settings_values,
                    reason=request.reason,
                    correlation_id=request.correlation_id,
                )
            )
        except Exception:
            return _runtime_apply_exception_result(request)

        if runtime_result.status == RUNTIME_APPLY_STATUS_APPLIED:
            return TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=runtime_result.message or local_commit_result.message,
                diagnostics=runtime_result.diagnostics or local_commit_result.diagnostics,
            )

        return TransactionResult(
            status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
            message=runtime_result.message or local_commit_result.message,
            diagnostics=_runtime_apply_degraded_diagnostics(
                request=request,
                runtime_status=runtime_result.status,
                runtime_diagnostics_present=runtime_result.diagnostics is not None,
            ),
        )

    async def _verify_transient_key(
        self,
        request: OpenRouterPkceHandoffRequest,
    ) -> ProviderVerificationResult | TransactionResult:
        if not request.verifier_context:
            return _provider_verification_failed_result(
                request=request,
                code="provider_verification_context_missing",
                verifier_status=None,
            )

        try:
            verification_result = await self.provider_verifier.verify_provider_secret(
                ProviderVerificationRequest(
                    provider=request.provider,
                    secret_key=request.secret_key,
                    secret_value=request.transient_api_key,
                    secret_revision=None,
                    context=request.verifier_context,
                )
            )
        except Exception:
            return _provider_verification_failed_result(
                request=request,
                code="provider_verifier_exception",
                verifier_status=None,
            )

        if (
            verification_result.provider != request.provider
            or verification_result.secret_key != request.secret_key
        ):
            return _provider_verification_failed_result(
                request=request,
                code="provider_verifier_result_mismatch",
                verifier_status=verification_result.status,
            )

        if verification_result.status != PROVIDER_VERIFICATION_STATUS_VERIFIED:
            return _provider_verification_failed_result(
                request=request,
                code="provider_verification_failed",
                verifier_status=verification_result.status,
            )

        return verification_result


def _settings_values_with_verification_evidence(
    *,
    request: OpenRouterPkceHandoffRequest,
    verification_result: ProviderVerificationResult,
) -> Mapping[str, object]:
    values = _merge_nested_settings_values(
        request.settings_values,
        provider_verification_state_values(
            ProviderVerificationEvidence(
                status=verification_result.status,
                provider=request.provider,
                secret_key=request.secret_key,
                secret_revision=verification_result.secret_revision,
                secret_fingerprint=_secret_fingerprint(request.transient_api_key),
                verifier_context=dict(request.verifier_context),
                verifier_evidence=_sanitize_verification_fields(
                    verification_result.evidence,
                    secret_value=request.transient_api_key,
                ),
            )
        ),
    )
    return freeze_settings_values(values)


def _merge_nested_settings_values(
    base: Mapping[str, object],
    overlay: Mapping[str, object],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, object]:
    merged = {str(key): _copy_settings_value(value) for key, value in base.items()}
    for raw_key, overlay_value in overlay.items():
        key = str(raw_key)
        existing_value = merged.get(key)
        next_path = (*path, key)
        if path == ("state", "provider_verification"):
            merged[key] = _copy_settings_value(overlay_value)
        elif isinstance(existing_value, Mapping) and isinstance(overlay_value, Mapping):
            merged[key] = _merge_nested_settings_values(
                existing_value,
                overlay_value,
                path=next_path,
            )
        else:
            merged[key] = _copy_settings_value(overlay_value)
    return merged


def _copy_settings_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_settings_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_copy_settings_value(item) for item in value]
    if isinstance(value, list):
        return [_copy_settings_value(item) for item in value]
    return value


def _caller_settings_values_are_unsafe(
    value: object,
    *,
    secret_value: str,
) -> bool:
    if isinstance(value, Mapping):
        return any(
            _caller_settings_key_is_unsafe(key, secret_value=secret_value)
            or _caller_settings_values_are_unsafe(nested_value, secret_value=secret_value)
            for key, nested_value in value.items()
        )

    if isinstance(value, list | tuple):
        return any(
            _caller_settings_values_are_unsafe(item, secret_value=secret_value) for item in value
        )

    return isinstance(value, str) and bool(secret_value and secret_value in value)


def _caller_settings_key_is_unsafe(
    key: object,
    *,
    secret_value: str,
) -> bool:
    if not isinstance(key, str):
        return True
    if secret_value and secret_value in key:
        return True
    return _is_sensitive_evidence_key(key)


def _provider_verification_failed_result(
    *,
    request: OpenRouterPkceHandoffRequest,
    code: str,
    verifier_status: str | None,
) -> TransactionResult:
    fields: dict[str, DiagnosticFieldValue] = {
        "provider": request.provider,
        "secret_key": request.secret_key,
        "phase": "verify_transient_key",
        "verification_succeeded": False,
    }
    if verifier_status is not None:
        fields["verifier_status"] = verifier_status
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=None,
        diagnostics=ErrorDiagnostics(
            component="openrouter_pkce_handoff",
            operation="verify_transient_key",
            code=code,
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields=fields,
        ),
    )


def _unsafe_settings_values_result(
    request: OpenRouterPkceHandoffRequest,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=None,
        diagnostics=ErrorDiagnostics(
            component="openrouter_pkce_handoff",
            operation="validate_settings_values",
            code="unsafe_settings_values",
            category=DIAGNOSTIC_CATEGORY_TRANSACTION,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={
                "provider": request.provider,
                "phase": "validate_settings_values",
                "settings_values_accepted": False,
            },
        ),
    )


def _runtime_apply_degraded_diagnostics(
    *,
    request: OpenRouterPkceHandoffRequest,
    runtime_status: str,
    runtime_diagnostics_present: bool,
) -> ErrorDiagnostics:
    return ErrorDiagnostics(
        component="openrouter_pkce_handoff",
        operation="runtime_apply",
        code="runtime_apply_degraded",
        category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={
            "provider": request.provider,
            "secret_key": request.secret_key,
            "phase": "runtime_apply",
            "runtime_status": runtime_status,
            "runtime_diagnostics_present": runtime_diagnostics_present,
        },
    )


def _runtime_apply_exception_result(
    request: OpenRouterPkceHandoffRequest,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        message=None,
        diagnostics=ErrorDiagnostics(
            component="openrouter_pkce_handoff",
            operation="runtime_apply",
            code="runtime_apply_exception",
            category=DIAGNOSTIC_CATEGORY_LIFECYCLE,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={
                "provider": request.provider,
                "secret_key": request.secret_key,
                "phase": "runtime_apply",
            },
        ),
    )


__all__ = [
    "OpenRouterPkceHandoffRequest",
    "OpenRouterPkceHandoffService",
]
