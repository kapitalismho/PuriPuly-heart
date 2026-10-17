from __future__ import annotations

from typing import Literal

from puripuly_heart.core.llm.fallback_racing import LLMProviderRaceError
from puripuly_heart.core.messages import DiagnosticCategory


class IndexTranslateError(RuntimeError):
    diagnostic_provider = "index_translate"

    def __init__(
        self,
        reason: str,
        *,
        status_code: int | None = None,
        diagnostic_category: DiagnosticCategory = "invalid_response",
    ) -> None:
        self.reason = reason
        self.status_code = status_code
        self.diagnostic_category = diagnostic_category
        super().__init__(f"Index translation failed: {reason}")


class IndexServiceUnavailableError(IndexTranslateError):
    def __init__(
        self,
        reason: Literal["timeout", "transport", "http_status", "invalid_response"],
        *,
        status_code: int | None = None,
    ) -> None:
        category: DiagnosticCategory = {
            "timeout": "timeout",
            "transport": "network",
            "http_status": "unknown",
            "invalid_response": "invalid_response",
        }[reason]
        super().__init__(reason, status_code=status_code, diagnostic_category=category)


class IndexRequestLimitError(IndexTranslateError):
    def __init__(
        self,
        reason: Literal["input_limit", "output_limit"],
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(reason, status_code=status_code)


def is_index_service_failure(exception: BaseException) -> bool:
    if isinstance(exception, IndexServiceUnavailableError):
        return True
    if isinstance(exception, LLMProviderRaceError):
        return bool(exception.errors) and all(
            is_index_service_failure(error) for error in exception.errors
        )
    return False
