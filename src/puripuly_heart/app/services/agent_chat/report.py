from __future__ import annotations

import gzip
import io
import json
import platform
import re
import tarfile
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.agent_chat import AgentReportOffer, AgentReportStatus
from puripuly_heart.app.ports.application_control import ApplicationControl

MAX_SUPPORT_BYTES = 25_000_000
MAX_REPORT_SUMMARY = 2000


@dataclass(slots=True)
class SupportReport:
    id: str
    generation: int
    summary: str
    message_index: int | None = None
    status: AgentReportStatus = "pending"
    ticket: str | None = None
    archive: tuple[dict, ...] | None = None
    diagnostic_codes: tuple[str, ...] | None = None
    bundle: bytes | None = None
    attempt: int = 0
    response_index: int | None = None
    displayed: bool = False
    clicked: bool = False

    @property
    def snapshot(self) -> AgentReportOffer:
        return AgentReportOffer(
            self.id,
            self.message_index,
            self.status,
            self.ticket,
            self.generation,
            self.response_index,
            self.displayed,
            self.clicked,
        )


async def recent_diagnostic_codes(control: ApplicationControl) -> tuple[str, ...]:
    try:
        history = await control.query("diagnostics.history", {"limit": 10})
        if history.get("truncated") and type(history.get("latest_sequence")) is int:
            history = await control.query(
                "diagnostics.history",
                {"limit": 10, "after": max(0, history["latest_sequence"] - 10)},
            )
        codes = []
        for record in reversed(history.get("records", [])):
            code = record.get("code") if isinstance(record, dict) else None
            if (
                isinstance(code, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", code)
                and code not in codes
            ):
                codes.append(code)
                if len(codes) == 10:
                    break
        return tuple(codes)
    except Exception:
        return ()


def build_support_bundle(
    *,
    ticket: str,
    app_version: str,
    summary: str,
    diagnostic_codes: tuple[str, ...],
    archive: tuple[dict, ...],
    log_snapshot: Callable[[], bytes],
) -> bytes:
    log = log_snapshot()
    meta = {
        "ticket": ticket,
        "app_version": app_version,
        "os": platform.system(),
        "agent_protocol": 3,
        "summary": summary,
        "diagnostic_codes": list(diagnostic_codes),
    }
    members = (
        ("meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8")),
        ("puri_chat.json", json.dumps(archive, ensure_ascii=False).encode("utf-8")),
        ("runtime.log", log),
    )
    target = io.BytesIO()
    with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, content in members:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o600
                tar.addfile(member, io.BytesIO(content))
    return target.getvalue()
