from __future__ import annotations

import gzip
import io
import json
import platform
import tarfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from puripuly_heart.app.ports.agent_chat import AgentReportOffer, AgentReportStatus
from puripuly_heart.app.ports.application_control import ApplicationControl
from puripuly_heart.core.diagnostics import project_diagnostic_record

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
    diagnostics: dict | None = None
    bundle: bytes | None = None
    attempt: int = 0
    response_index: int | None = None
    displayed: bool = False
    clicked: bool = False

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        return diagnostic_codes(self.diagnostics)

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


async def recent_diagnostics(control: ApplicationControl) -> dict:
    try:
        history = await control.query("diagnostics.history", {"limit": 10})
        if history.get("status") == "failed":
            raise ValueError("diagnostics_unavailable")
        if history.get("truncated") and type(history.get("latest_sequence")) is int:
            history = await control.query(
                "diagnostics.history",
                {"limit": 10, "after": max(0, history["latest_sequence"] - 10)},
            )
        if history.get("status") == "failed":
            raise ValueError("diagnostics_unavailable")
        return {
            "availability": history.get("availability", "unknown"),
            "coverage": {
                key: value
                for key, value in history.items()
                if key in {
                    "instance_id", "lookup", "oldest_sequence", "latest_sequence",
                    "next_after", "gap", "evicted_records", "truncated",
                }
                and (value is None or type(value) in (str, int, bool))
            },
            "records": [
                project_diagnostic_record(record)
                for record in reversed(history.get("records", []))
                if isinstance(record, Mapping)
            ][:10],
        }
    except Exception:
        return {"availability": "query_failed", "records": []}


def diagnostic_codes(diagnostics: dict | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(record["code"] for record in (diagnostics or {}).get("records", []))
    )


def build_support_bundle(
    *,
    ticket: str,
    app_version: str,
    summary: str,
    diagnostics: dict,
    archive: tuple[dict, ...],
    log_snapshot: Callable[[], bytes] | None,
) -> bytes:
    diagnostics = {
        **diagnostics,
        "records": [
            project_diagnostic_record(record)
            for record in diagnostics.get("records", [])[:10]
            if isinstance(record, Mapping)
        ],
    }
    missing = {}
    log = None
    if log_snapshot is None:
        missing["runtime.log"] = "collection_unavailable"
    else:
        try:
            log = log_snapshot()
            if not isinstance(log, bytes):
                raise TypeError("invalid_log_snapshot")
        except Exception:
            missing["runtime.log"] = "collection_failed"
    if diagnostics.get("availability") != "retained_current_run":
        missing["diagnostics"] = diagnostics.get("availability", "unknown")
    meta = {
        "ticket": ticket,
        "app_version": app_version,
        "os": platform.system(),
        "agent_protocol": 3,
        "summary": summary,
        "diagnostic_codes": list(diagnostic_codes(diagnostics)),
        "diagnostics": diagnostics,
        "missing_material": missing,
    }
    members = [
        ("meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8")),
        ("puri_chat.json", json.dumps(archive, ensure_ascii=False).encode("utf-8")),
    ]
    if log is not None:
        members.append(("runtime.log", log))
    target = io.BytesIO()
    with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, content in members:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o600
                tar.addfile(member, io.BytesIO(content))
    return target.getvalue()
