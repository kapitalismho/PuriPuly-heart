from __future__ import annotations

from dataclasses import dataclass

OVERLAY_CONTRACT_VERSION = 17
OVERLAY_EXECUTION_CONTRACT = {"version": 1, "revision": "r2"}
OVERLAY_NATIVE_RETRY_CONTRACT = {"version": 2, "ownership": "exclusive"}
OVERLAY_SPEAKER_IDENTITY_CONTRACT = {
    "version": 2,
    "policy": "immutable_first_readable_style",
}
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
}
_VR_COMPATIBILITY_MODE = "vr_compatibility_mode"


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
    vr_compatibility_mode: bool = False

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
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
        }
        if self.vr_compatibility_mode:
            data[_VR_COMPATIBILITY_MODE] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OverlayLaunchManifest":
        extra_fields = set(data) - _MANIFEST_FIELDS - {_VR_COMPATIBILITY_MODE}
        if extra_fields:
            joined = ", ".join(sorted(extra_fields))
            raise ValueError(f"overlay manifest contains unsupported runtime fields: {joined}")

        required_fields = _MANIFEST_FIELDS
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            joined = ", ".join(sorted(missing_fields))
            raise ValueError(f"overlay manifest is missing required fields: {joined}")
        vr_compatibility_mode = data.get(_VR_COMPATIBILITY_MODE, False)
        if not isinstance(vr_compatibility_mode, bool):
            raise ValueError("overlay manifest vr_compatibility_mode must be a boolean")

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
            vr_compatibility_mode=vr_compatibility_mode,
        )
