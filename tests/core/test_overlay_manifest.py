from __future__ import annotations

import pytest

from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest


def test_overlay_manifest_uses_structured_block_contract_version() -> None:
    assert OVERLAY_CONTRACT_VERSION == 9


def test_overlay_manifest_round_trips_contract_fields() -> None:
    manifest = OverlayLaunchManifest(
        contract_version=OVERLAY_CONTRACT_VERSION,
        app_version="1.2.3",
        overlay_instance_id="overlay-1",
        bridge_url="ws://127.0.0.1:8765",
        session_token="token",
        parent_pid=1234,
        startup_deadline_ms=3000,
        log_dir="logs",
        log_level="INFO",
        locale="en",
    )

    restored = OverlayLaunchManifest.from_dict(manifest.to_dict())

    assert restored.contract_version == OVERLAY_CONTRACT_VERSION
    assert restored.app_version == "1.2.3"
    assert restored.overlay_instance_id == "overlay-1"
    assert restored.bridge_url == "ws://127.0.0.1:8765"
    assert restored.log_dir == "logs"
    assert restored.log_level == "INFO"


def test_overlay_manifest_rejects_obsolete_diagnostics_field() -> None:
    with pytest.raises(ValueError):
        OverlayLaunchManifest.from_dict(
            {
                "contract_version": OVERLAY_CONTRACT_VERSION,
                "app_version": "1.2.3",
                "overlay_instance_id": "overlay-1",
                "bridge_url": "ws://127.0.0.1:8765",
                "session_token": "token",
                "parent_pid": 1234,
                "startup_deadline_ms": 3000,
                "log_dir": "logs",
                "log_level": "INFO",
                "locale": "en",
                "diagnostics_enabled": True,
            }
        )


def test_overlay_manifest_rejects_live_runtime_state_fields() -> None:
    with pytest.raises(ValueError):
        OverlayLaunchManifest.from_dict(
            {
                "contract_version": OVERLAY_CONTRACT_VERSION,
                "app_version": "1.2.3",
                "overlay_instance_id": "overlay-1",
                "bridge_url": "ws://127.0.0.1:8765",
                "session_token": "token",
                "parent_pid": 1234,
                "startup_deadline_ms": 3000,
                "log_dir": "logs",
                "log_level": "INFO",
                "locale": "en",
                "presentation_snapshot": {"blocks": []},
            }
        )
