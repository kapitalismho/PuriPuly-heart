from __future__ import annotations

import pytest

from puripuly_heart.config.prompts import _reset_prompt_cache_for_tests


@pytest.fixture(autouse=True)
def reset_prompt_cache():
    _reset_prompt_cache_for_tests()
    yield
    _reset_prompt_cache_for_tests()
