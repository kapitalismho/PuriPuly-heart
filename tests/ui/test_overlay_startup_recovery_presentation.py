from __future__ import annotations

import json
from pathlib import Path

from puripuly_heart.ui.overlay_peer_contract import (
    build_overlay_peer_consumer_contract,
)
from puripuly_heart.ui.views.settings import SettingsView

I18N_DIR = Path(__file__).resolve().parents[2] / "src" / "puripuly_heart" / "data" / "i18n"


def _action_kind(reason: str | None) -> str:
    view = SettingsView.__new__(SettingsView)
    view._overlay_failure_reason = reason
    return view._desktop_overlay_failure_action_kind()


def test_recovering_presents_intent_on_without_warning() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="recovering",
        overlay_failure_reason=None,
        peer_intent_enabled=False,
        peer_effective_enabled=False,
    )

    assert contract.overlay.state == "on"
    assert contract.overlay.warning_reason is None


def test_recovering_keeps_peer_in_starting() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="recovering",
        overlay_failure_reason=None,
        peer_intent_enabled=True,
        peer_effective_enabled=False,
    )

    assert contract.peer.state == "starting"
    assert contract.peer.warning_reason == "overlay_starting"


def test_recovering_status_text_uses_localized_copy() -> None:
    contract = build_overlay_peer_consumer_contract(
        overlay_intent_enabled=True,
        overlay_state="recovering",
        overlay_failure_reason=None,
        peer_intent_enabled=False,
        peer_effective_enabled=False,
    )

    bundle = json.loads((I18N_DIR / "en.json").read_text(encoding="utf-8"))
    assert bundle["settings.overlay.status.recovering"] not in ("", "recovering")
    assert contract.overlay.status_text == bundle["settings.overlay.status.recovering"]


def test_reveal_loss_reuses_reopen_action() -> None:
    assert _action_kind("window_reveal_lost") == "reopen"
    assert _action_kind("window_visibility_unstable") == "reopen"
    assert _action_kind("window_configuration_failed") == "reopen"


def test_identity_and_bounds_failures_reuse_retry_action() -> None:
    assert _action_kind("window_identity_failed") == "retry"
    assert _action_kind("window_observation_failed") == "retry"
    assert _action_kind("window_bounds_failed") == "retry"
    assert _action_kind("window_native_ready_failed") == "retry"


def test_visibility_vs_bounds_identity_copy_is_distinct_and_localized() -> None:
    bundles = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(I18N_DIR.glob("*.json"))
    }
    for locale in ("en", "ko", "ja", "zh-CN", "ru"):
        bundle = bundles[locale]
        reveal = bundle["settings.overlay.failure.window_reveal_lost"]
        unstable = bundle["settings.overlay.failure.window_visibility_unstable"]
        bounds = bundle["settings.overlay.failure.window_bounds_failed"]
        identity = bundle["settings.overlay.failure.window_identity_failed"]
        observation = bundle["settings.overlay.failure.window_observation_failed"]
        native_ready = bundle["settings.overlay.failure.window_native_ready_failed"]
        for copy in (reveal, unstable, bounds, identity, observation, native_ready):
            assert copy and "settings." not in copy
        assert reveal != bounds
        assert unstable != bounds
        assert identity != bounds
        assert identity != reveal

    en = bundles["en"]
    assert "hidden" in en["settings.overlay.failure.window_reveal_lost"]
    assert "position" in en["settings.overlay.failure.window_bounds_failed"]
    assert "unexpected window" in en["settings.overlay.failure.window_identity_failed"]
