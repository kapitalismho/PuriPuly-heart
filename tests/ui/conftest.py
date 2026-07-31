from __future__ import annotations

import pytest

from puripuly_heart.ui.i18n import get_locale, set_locale


@pytest.fixture(autouse=True)
def restore_locale_after_test():
    previous_locale = get_locale()
    try:
        yield
    finally:
        set_locale(previous_locale)
