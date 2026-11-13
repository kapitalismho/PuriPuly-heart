from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from puripuly_heart.app.ports.ui_presentation import SettingsRenderResult
from puripuly_heart.core.diagnostics import emit_failure
from puripuly_heart.core.messages import (
    ErrorDiagnostics,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
    TransactionResult,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ResultScope:
    transaction: TransactionResult | None = None
    revision: Callable[[], int] | None = None
    committed_revision: int | None = None
    reverted: bool = False
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0

    @property
    def current(self) -> TransactionResult | None:
        result = self.transaction
        if result is None or not (self.secondary_diagnostics or self.secondary_omitted_count):
            return result
        retained_ids = {cause.diagnostic_id for cause in result.secondary_diagnostics}
        return replace(
            result,
            secondary_diagnostics=(
                *result.secondary_diagnostics,
                *(
                    replace(
                        cause,
                        cause_id=(
                            result.diagnostics.diagnostic_id
                            if result.diagnostics is not None else cause.cause_id
                        ),
                    )
                    for cause in self.secondary_diagnostics
                    if cause.diagnostic_id not in retained_ids
                ),
            ),
            secondary_omitted_count=result.secondary_omitted_count + self.secondary_omitted_count,
        )


class SettingsTransactionResultOwner:
    def __init__(self) -> None:
        self._latest: TransactionResult | None = None
        self._scope: ContextVar[_ResultScope | None] = ContextVar(
            "settings_transaction_result_scope",
            default=None,
        )
        self._pending_diagnostics: tuple[ErrorDiagnostics, ...] = ()
        self._pending_omitted_count = 0

    @property
    def current(self) -> TransactionResult | None:
        scope = self._scope.get()
        return scope.transaction if scope is not None else self._latest

    @contextmanager
    def capture(self, *, revision: Callable[[], int] | None = None):
        scope = _ResultScope(
            revision=revision,
            secondary_diagnostics=self._pending_diagnostics,
            secondary_omitted_count=self._pending_omitted_count,
        )
        self._pending_diagnostics = ()
        self._pending_omitted_count = 0
        token = self._scope.set(scope)
        try:
            yield scope
        finally:
            result = scope.current
            self._scope.reset(token)
            if result is not None:
                self._latest = result
            elif scope.secondary_diagnostics or scope.secondary_omitted_count:
                parent = self._scope.get()
                if parent is not None:
                    combined = (*parent.secondary_diagnostics, *scope.secondary_diagnostics)
                    parent.secondary_diagnostics = combined[:4]
                    parent.secondary_omitted_count += (
                        scope.secondary_omitted_count + max(0, len(combined) - 4)
                    )
                else:
                    combined = (*scope.secondary_diagnostics, *self._pending_diagnostics)
                    self._pending_diagnostics = combined[:4]
                    self._pending_omitted_count += (
                        scope.secondary_omitted_count + max(0, len(combined) - 4)
                    )

    def set(self, result: TransactionResult) -> None:
        scope = self._scope.get()
        if scope is None and (self._pending_diagnostics or self._pending_omitted_count):
            result = replace(
                result,
                secondary_diagnostics=(*result.secondary_diagnostics, *self._pending_diagnostics),
                secondary_omitted_count=result.secondary_omitted_count + self._pending_omitted_count,
            )
            self._pending_diagnostics = ()
            self._pending_omitted_count = 0
        self._latest = result
        scope = self._scope.get()
        if scope is not None:
            scope.transaction = result
        disposition = (
            "degraded"
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            else (
                "cancelled"
                if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED
                else (
                    "recovered"
                    if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
                    else "failed"
                )
            )
        )
        if result.diagnostics is not None:
            emit_failure(
                logger,
                result.diagnostics,
                disposition=disposition,
                severity=(
                    "info"
                    if disposition == "recovered"
                    else "warning" if disposition != "failed" else "error"
                ),
            )
        for secondary in result.secondary_diagnostics:
            emit_failure(logger, secondary, disposition="degraded", severity="warning")

    def add_presentation_result(self, result: SettingsRenderResult) -> None:
        if result.diagnostics is not None:
            self.add_secondary(result.diagnostics)
        for diagnostics in result.secondary_diagnostics:
            self.add_secondary(diagnostics)
        if result.secondary_omitted_count:
            scope = self._scope.get()
            if scope is not None:
                scope.secondary_omitted_count += result.secondary_omitted_count
            elif self._latest is not None:
                self._latest = replace(
                    self._latest,
                    secondary_omitted_count=(
                        self._latest.secondary_omitted_count + result.secondary_omitted_count
                    ),
                )
            else:
                self._pending_omitted_count += result.secondary_omitted_count

    def add_secondary(self, diagnostics: ErrorDiagnostics) -> None:
        scope = self._scope.get()
        if scope is not None:
            if len(scope.secondary_diagnostics) < 4:
                scope.secondary_diagnostics = (*scope.secondary_diagnostics, diagnostics)
            else:
                scope.secondary_omitted_count += 1
            emit_failure(logger, diagnostics, disposition="degraded", severity="warning")
            return
        current = self.current
        if current is not None:
            self.set(
                replace(
                    current,
                    secondary_diagnostics=(
                        *current.secondary_diagnostics,
                        replace(
                            diagnostics,
                            cause_id=(
                                current.diagnostics.diagnostic_id
                                if current.diagnostics is not None else diagnostics.cause_id
                            ),
                        ),
                    ),
                )
            )
        elif len(self._pending_diagnostics) < 4:
            self._pending_diagnostics = (*self._pending_diagnostics, diagnostics)
        else:
            self._pending_omitted_count += 1

    def mark_settings_committed(self) -> None:
        scope = self._scope.get()
        if scope is not None and scope.revision is not None:
            scope.transaction = None
            scope.committed_revision = scope.revision()

    def committed(self) -> bool:
        current = self.current
        return current is not None and current.status in {
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
        }

    def mark_settings_reverted(self) -> None:
        scope = self._scope.get()
        if scope is not None:
            scope.reverted = True


__all__ = ["SettingsTransactionResultOwner"]
