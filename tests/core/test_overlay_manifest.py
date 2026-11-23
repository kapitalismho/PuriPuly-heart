from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest


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
    assert not restored.vr_compatibility_mode
    assert "vr_compatibility_mode" not in manifest.to_dict()

    compatibility = replace(manifest, vr_compatibility_mode=True)
    compatibility_wire = compatibility.to_dict()
    assert compatibility_wire["vr_compatibility_mode"] is True
    assert OverlayLaunchManifest.from_dict(compatibility_wire) == compatibility
    compatibility_wire["vr_compatibility_mode"] = "true"
    with pytest.raises(ValueError, match="vr_compatibility_mode must be a boolean"):
        OverlayLaunchManifest.from_dict(compatibility_wire)


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
