from __future__ import annotations

import pytest

pytest.importorskip("flet")
import flet as ft

from puripuly_heart.ui.components.managed_trial_usage_bar import ManagedTrialUsageBar
from puripuly_heart.ui.i18n import get_locale, set_locale, t


def test_managed_trial_usage_bar_renders_inline_placeholder_when_percent_unknown() -> None:
    set_locale("en")
    bar = ManagedTrialUsageBar()

    assert isinstance(bar, ft.Row)
    assert bar.percent is None
    assert len(bar.controls) == 1
    assert bar.spacing == 0
    assert bar.controls[0] is bar._track
    assert not hasattr(bar, "_status_icon")
    assert bar._remaining_text.value == t("settings.managed_trial_usage.remaining_placeholder")


@pytest.mark.parametrize(
    ("percent", "expected_percent"),
    [
        (42, 42),
        (71, 71),
        (-5, 0),
        (135, 100),
    ],
)
def test_managed_trial_usage_bar_formats_and_clamps_percent(
    percent: int,
    expected_percent: int,
) -> None:
    set_locale("en")
    bar = ManagedTrialUsageBar(percent=percent)

    assert bar.percent == expected_percent
    assert bar._remaining_text.value == t(
        "settings.managed_trial_usage.remaining", percent=expected_percent
    )


def test_managed_trial_usage_bar_distributes_fill_horizontally() -> None:
    bar = ManagedTrialUsageBar(percent=42)

    assert bar._fill_segment.expand == 42
    assert bar._empty_segment.expand == 58


def test_managed_trial_usage_bar_handles_empty_and_full_states() -> None:
    empty_bar = ManagedTrialUsageBar(percent=0)
    full_bar = ManagedTrialUsageBar(percent=100)

    assert empty_bar._fill_segments.controls == [empty_bar._empty_segment]
    assert full_bar._fill_segments.controls == [full_bar._fill_segment]


def test_managed_trial_usage_bar_apply_locale_refreshes_overlay_text() -> None:
    previous_locale = get_locale()
    try:
        set_locale("en")
        bar = ManagedTrialUsageBar(percent=71)

        set_locale("ko")
        bar.apply_locale()
        assert bar._remaining_text.value == t(
            "settings.managed_trial_usage.remaining", percent=71
        )

        bar.set_percent(None)
        bar.apply_locale()
        assert bar._remaining_text.value == t(
            "settings.managed_trial_usage.remaining_placeholder"
        )
    finally:
        set_locale(previous_locale)


def test_managed_trial_usage_bar_never_renders_status_icon() -> None:
    unknown_bar = ManagedTrialUsageBar()
    known_bar = ManagedTrialUsageBar(percent=71)

    assert not hasattr(unknown_bar, "_status_icon")
    assert not hasattr(known_bar, "_status_icon")
