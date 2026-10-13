from __future__ import annotations

import logging
from dataclasses import replace

from puripuly_heart.core.diagnostics import current_diagnostic_operation, emit_failure
from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.messages import DiagnosticCategory, ErrorDiagnostics


class LLMResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        diagnostic_category: DiagnosticCategory = "invalid_response",
        operation: str = "translate",
    ) -> None:
        self.diagnostic_provider = provider
        self.status_code = status_code
        self.diagnostic_category = diagnostic_category
        super().__init__(message)
        self.failure_diagnostics = provider_failure_report(
            self, provider=provider, operation=operation
        ).diagnostics


def record_probe_failure(
    exception: BaseException | None, *, provider: str, status_code: int | None = None
) -> ErrorDiagnostics:
    if exception is None and status_code is not None:
        exception = LLMResponseError(
            "Provider key probe failed",
            provider=provider,
            status_code=status_code,
            diagnostic_category="unknown",
            operation="probe",
        )
    diagnostics = provider_failure_report(
        exception, provider=provider, operation="probe"
    ).diagnostics
    diagnostics = replace(
        diagnostics,
        operation="probe",
        status_code=status_code if status_code is not None else diagnostics.status_code,
        **current_diagnostic_operation(),
        fields={
            **diagnostics.fields,
            "verification_scope": "key_probe",
            "recognition_readiness": "unobserved",
        },
    )
    emit_failure(logging.getLogger(__name__), diagnostics, stage="probe", exception=exception)
    return diagnostics


def record_probe_cleanup_failure(
    exception: BaseException,
    *,
    provider: str,
    primary_failure: ErrorDiagnostics | None,
) -> None:
    diagnostics = provider_failure_report(
        exception, provider=provider, operation="cleanup"
    ).diagnostics
    emit_failure(
        logging.getLogger(__name__),
        replace(
            diagnostics,
            **current_diagnostic_operation(),
            cause_id=primary_failure.diagnostic_id if primary_failure else None,
            fields={**diagnostics.fields, "cleanup_confirmed": False},
        ),
        severity="warning",
        disposition="degraded",
        stage="cleanup",
    )


__all__ = ["LLMResponseError", "record_probe_failure", "record_probe_cleanup_failure"]
