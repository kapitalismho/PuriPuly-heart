from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TransactionResult,
)


@dataclass(slots=True)
class SettingsTransactionResultOwner:
    current: TransactionResult | None = None

    def set(self, result: TransactionResult) -> None:
        self.current = result

    def committed(self) -> bool:
        return self.current is not None and self.current.status in {
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        }


__all__ = ["SettingsTransactionResultOwner"]
