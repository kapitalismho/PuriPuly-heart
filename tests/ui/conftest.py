from __future__ import annotations

import pytest

from tests.helpers.settings_view import SettingsIntentRecorder


@pytest.fixture
def settings_events() -> SettingsIntentRecorder:
    return SettingsIntentRecorder()
