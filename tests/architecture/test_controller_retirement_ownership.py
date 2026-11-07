from __future__ import annotations

import pytest

from tests.helpers.ast_sources import imported_modules
from tests.helpers.paths import SOURCE_ROOT

CASES = [
    {
        "id": "managed_auth",
        "owner_paths": (
            "app/services/managed/managed_auth.py",
            "app/wiring/wiring_managed_auth_factory.py",
            "app/wiring/wiring_managed_account.py",
        ),
        "forbidden_prefixes": ("puripuly_heart.ui",),
        "forbidden_exact": (),
    },
    {
        "id": "managed_usage",
        "owner_paths": ("app/services/managed/managed_usage.py",),
        "forbidden_prefixes": ("puripuly_heart.ui",),
        "forbidden_exact": ("puripuly_heart.config.settings",),
    },
    {
        "id": "translation_enable",
        "owner_paths": (
            "app/services/translation_enable.py",
            "app/wiring/wiring_managed_auth_factory.py",
        ),
        "forbidden_prefixes": ("puripuly_heart.ui",),
        "forbidden_exact": (),
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_owner_does_not_import_ui_or_controller(case) -> None:
    for relative_path in case["owner_paths"]:
        imports = imported_modules(SOURCE_ROOT / relative_path)
        assert not any(
            name.startswith(prefix) for name in imports for prefix in case["forbidden_prefixes"]
        )
        assert not set(case["forbidden_exact"]) & imports
