from __future__ import annotations

from dataclasses import replace

from puripuly_heart.app.ports.ui_models import (
    OscControlPresentationName,
    OscControlPresentationState,
)
from puripuly_heart.app.services.osc.presentation_state import (
    presentation_state_from_settings,
)
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState, state_from_settings
from puripuly_heart.config.settings import AppSettings


def osc_control_presentation_state(
    changed_control: OscControlPresentationName,
    *,
    settings: AppSettings | None = None,
    canonical_state: OscCanonicalState | None = None,
    **canonical_changes: object,
) -> OscControlPresentationState:
    value = settings or AppSettings()
    canonical = canonical_state or state_from_settings(value)
    if canonical_changes:
        canonical = replace(canonical, **canonical_changes)
    return presentation_state_from_settings(
        value,
        canonical_state=canonical,
        changed_control=changed_control,
    )


__all__ = ["osc_control_presentation_state"]
