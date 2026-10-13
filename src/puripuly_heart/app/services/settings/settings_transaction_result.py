from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from puripuly_heart.core.diagnostics import emit_failure
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
    TransactionResult,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ResultScope:
    current: TransactionResult | None = None
    revision: Callable[[], int] | None = None
    committed_revision: int | None = None


class SettingsTransactionResultOwner:
    def __init__(self) -> None:
        self._latest: TransactionResult | None = None
        self._scope: ContextVar[_ResultScope | None] = ContextVar(
            "settings_transaction_result_scope",
            default=None,
        )

    @property
    def current(self) -> TransactionResult | None:
        scope = self._scope.get()
        return scope.current if scope is not None else self._latest

    @contextmanager
    def capture(self, *, revision: Callable[[], int] | None = None):
        scope = _ResultScope(revision=revision)
        token = self._scope.set(scope)
        try:
            yield scope
        finally:
            self._scope.reset(token)

    def set(self, result: TransactionResult) -> None:
        self._latest = result
        scope = self._scope.get()
        if scope is not None:
            scope.current = result
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

    def mark_settings_committed(self) -> None:
        scope = self._scope.get()
        if scope is not None and scope.revision is not None:
            scope.current = None
            scope.committed_revision = scope.revision()

    def committed(self) -> bool:
        current = self.current
        return current is not None and current.status in {
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_INTERRUPTED,
        }


__all__ = ["SettingsTransactionResultOwner"]
