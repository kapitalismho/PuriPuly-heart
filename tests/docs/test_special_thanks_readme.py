from __future__ import annotations

import pytest

from tests.helpers.paths import REPO_ROOT as ROOT

README_FILES = ["README.md", "README.ko.md", "README.ja.md", "README.zh-CN.md"]
SPECIAL_THANKS_TEXT = (
    "SUI\\_32C, Nagikokoro, motoka96, \\_Ykol魚, kascr\\_, "
    "Just Monika V, FLUVIA, Han โชเล่ย์, EA\\_PE, Ephedrine, ~ eri ~"
)


@pytest.mark.parametrize("readme_name", README_FILES)
def test_readme_special_thanks_lists_expected_names_in_order(readme_name: str) -> None:
    readme_text = (ROOT / readme_name).read_text(encoding="utf-8")

    assert SPECIAL_THANKS_TEXT in readme_text
