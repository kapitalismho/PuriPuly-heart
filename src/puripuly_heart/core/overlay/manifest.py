from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.core.runtime_logging import SessionLoggingMode

OVERLAY_CONTRACT_VERSION = 7
OVERLAY_EXECUTION_CONTRACT = {"version": 1, "revision": "r1"}
OVERLAY_NATIVE_RETRY_CONTRACT = {"version": 1, "ownership": "exclusive"}
_MANIFEST_FIELDS = {
    "contract_version",
    "app_version",
    "overlay_instance_id",
    "bridge_url",
    "session_token",
    "parent_pid",
    "startup_deadline_ms",
    "log_dir",
    "log_level",
    "locale",
    "logging_mode",
    "diagnostics_enabled",
}


def normalize_overlay_logging_mode(
    mode: SessionLoggingMode | str | bool | object,
) -> str:
    if isinstance(mode, bool):
        return SessionLoggingMode.DETAILED.value if mode else SessionLoggingMode.BASIC.value
    return SessionLoggingMode(mode).value


@dataclass(frozen=True, slots=True)
class OverlayLaunchManifest:
    contract_version: int
    app_version: str
    overlay_instance_id: str
    bridge_url: str
    session_token: str
    parent_pid: int
    startup_deadline_ms: int
    log_dir: str
    log_level: str
    locale: str
    logging_mode: str = SessionLoggingMode.BASIC.value

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "app_version": self.app_version,
            "overlay_instance_id": self.overlay_instance_id,
            "bridge_url": self.bridge_url,
            "session_token": self.session_token,
            "parent_pid": self.parent_pid,
            "startup_deadline_ms": self.startup_deadline_ms,
            "log_dir": self.log_dir,
            "log_level": self.log_level,
            "locale": self.locale,
            "logging_mode": normalize_overlay_logging_mode(self.logging_mode),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OverlayLaunchManifest":
        extra_fields = set(data) - _MANIFEST_FIELDS
        if extra_fields:
            joined = ", ".join(sorted(extra_fields))
            raise ValueError(f"overlay manifest contains unsupported runtime fields: {joined}")

        required_fields = _MANIFEST_FIELDS - {
            "logging_mode",
            "diagnostics_enabled",
        }
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            joined = ", ".join(sorted(missing_fields))
            raise ValueError(f"overlay manifest is missing required fields: {joined}")

        if "logging_mode" not in data and "diagnostics_enabled" not in data:
            raise ValueError("overlay manifest is missing required fields: logging_mode")

        logging_mode = data.get("logging_mode")
        if logging_mode is None:
            logging_mode = data.get("diagnostics_enabled", False)

        return cls(
            contract_version=int(data["contract_version"]),
            app_version=str(data["app_version"]),
            overlay_instance_id=str(data["overlay_instance_id"]),
            bridge_url=str(data["bridge_url"]),
            session_token=str(data["session_token"]),
            parent_pid=int(data["parent_pid"]),
            startup_deadline_ms=int(data["startup_deadline_ms"]),
            log_dir=str(data["log_dir"]),
            log_level=str(data["log_level"]),
            locale=str(data["locale"]),
            logging_mode=normalize_overlay_logging_mode(logging_mode),
        )
