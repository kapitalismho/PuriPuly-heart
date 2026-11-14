from __future__ import annotations

import json
from pathlib import Path

from puripuly_heart.app.services.agent_chat.tools import NAVIGATION_SECTIONS
from puripuly_heart.app.services.application_control import SETTINGS_FIELDS
from puripuly_heart.config.settings_vnext.schema import PHOTO_ROUTES
from puripuly_heart.ui.views.settings import _SETTINGS_SUBTAB_ORDER

ROOT = Path(__file__).resolve().parents[2]


def _guide_screens() -> str:
    guide = (ROOT / "agent-relay/prompts/agent/guide.md").read_text(encoding="utf-8")
    start = guide.index("\n## Screens\n")
    end = guide.index("\n## ", start + 1)
    return guide[start:end]


def _relay_tool(name: str) -> dict:
    source = (ROOT / "agent-relay/src/app-tools.ts").read_text(encoding="utf-8")
    tools = json.loads(source.removeprefix("export const appTools = ").strip().removesuffix(";"))
    return next(tool for tool in tools if tool["name"] == name)


def _relay_open_app_sections() -> tuple[str, ...]:
    return tuple(_relay_tool("open_app")["parameters"]["properties"]["section"]["enum"])


def test_puri_guide_names_settings_tabs_as_the_english_ui_does_in_order() -> None:
    labels = json.loads((ROOT / "src/puripuly_heart/data/i18n/en.json").read_text(encoding="utf-8"))
    screens = _guide_screens()
    positions = [
        screens.index(f"**{labels[f'settings.subtab.{key}']}**") for key in _SETTINGS_SUBTAB_ORDER
    ]

    assert positions == sorted(positions)


def test_relay_open_app_sections_match_the_app_and_the_guide() -> None:
    screens = _guide_screens()

    assert _relay_open_app_sections() == NAVIGATION_SECTIONS
    for section in NAVIGATION_SECTIONS:
        assert f"{section} " in screens.split("Settings navigation sections map to tabs:", 1)[1]


def test_relay_settings_proposals_accept_every_photo_setting() -> None:
    changes = _relay_tool("manage_proposal")["parameters"]["properties"]["changes"]["properties"]

    assert {name for name in SETTINGS_FIELDS if name.startswith("photo.")} == {
        name for name in changes if name.startswith("photo.")
    }
    assert changes["photo.route"]["enum"] == list(PHOTO_ROUTES)
