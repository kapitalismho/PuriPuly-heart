from __future__ import annotations

import logging

from puripuly_heart.core.runtime_logging import emit_basic_log

logger = logging.getLogger(__name__)


def log_feature(
    module: str,
    stage: str,
    code: str,
    *,
    error: BaseException | None = None,
    http_status: int | None = None,
    level: int = logging.ERROR,
) -> None:
    if module not in {"ModelDownload", "QuickTranslate", "AudioTest"}:
        raise ValueError("Unknown diagnostic module")
    if not all(value.isascii() and value.replace("_", "").isalnum() for value in (stage, code)):
        raise ValueError("Diagnostic codes must be static identifiers")
    fields = [
        f"[{module}] failure" if level >= logging.WARNING else f"[{module}] lifecycle",
        f"stage={stage}",
        f"code={code}",
    ]
    if error is not None:
        fields.append(f"exception_type={type(error).__name__}")
        os_code = getattr(error, "winerror", None) or getattr(error, "errno", None)
        if isinstance(os_code, int):
            fields.append(f"os_code={os_code}")
    if isinstance(http_status, int) and 100 <= http_status <= 599:
        fields.append(f"http_status={http_status}")
    emit_basic_log(logger, " ".join(fields), level=level)
