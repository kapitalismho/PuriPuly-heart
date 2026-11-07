from functools import partial

import pytest

from puripuly_heart.app.control_presentation import (
    SETTINGS_PRESENTATION,
    setting_catalog,
    setting_value_text,
)
from puripuly_heart.app.services.application_control import SETTINGS_FIELDS
from puripuly_heart.ui.agent_chat_model import change_blocks
from puripuly_heart.ui.i18n import get_locale, t, t_for_locale
from puripuly_heart.ui.views.settings import SettingsView


@pytest.mark.parametrize(("alpha", "expected"), [(0, "100%"), (1, "0%"), (0.6, "40%")])
def test_transparency_matches_settings_and_approval(alpha, expected):
    assert setting_value_text("overlay.background_alpha", alpha, t) == expected
    view = object.__new__(SettingsView)
    assert view._desktop_overlay_background_alpha_label_for(alpha) == expected
    assert change_blocks({"changes": {"overlay.background_alpha": alpha}})[0].after == expected


def test_all_cli_fields_have_metadata_and_private_values_never_leave_formatter():
    assert SETTINGS_FIELDS <= SETTINGS_PRESENTATION.keys()
    for field, descriptor in SETTINGS_PRESENTATION.items():
        if descriptor.local_only:
            assert "PRIVATE_SENTINEL" not in setting_value_text(field, "PRIVATE_SENTINEL", t)
            assert "PRIVATE_SENTINEL" not in repr(
                change_blocks(
                    {
                        "changes": {field: "PRIVATE_SENTINEL"},
                        "previous": {field: "PRIVATE_SENTINEL"},
                    }
                )
            )


@pytest.mark.parametrize("locale", ["en", "ko", "ja", "zh-CN", "ru"])
def test_catalog_localizes_without_mutating_locale_or_copying_private_state(locale):
    localize = partial(t_for_locale, locale)
    previous_locale = get_locale()
    catalog = {
        "fields": sorted(SETTINGS_FIELDS),
        "field_schemas": {
            "overlay.background_alpha": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "choices": {
            "translation.connection": {"gemma_4": ["openrouter"]},
            "languages": {"codes": ["en", "ja"], "peer_source_modes": ["manual", "auto"]},
        },
        "current": {"prompts.value": "PRIVATE_SENTINEL"},
    }
    presented = setting_catalog(catalog, localize)
    assert "PRIVATE_SENTINEL" not in repr(presented)
    assert get_locale() == previous_locale


def test_expanded_fields_are_not_dropped_by_group_prefix():
    changes = {
        "translation.gpu_device_id": "auto",
        "audio.output_device": "Speakers",
        "translation.qwen.region": "beijing",
        "chatbox.activation_notice.enabled": True,
        "stt.soniox.enable_speaker_diarization": True,
    }
    blocks = change_blocks({"changes": changes})
    assert len(blocks) == len(changes)


def test_unknown_value_is_not_reported_as_applied_content():
    assert "PRIVATE_SENTINEL" not in setting_value_text("not.a.setting", "PRIVATE_SENTINEL", t)


def test_dynamic_gpu_choices_are_captured_and_unknown_old_ids_are_unavailable():
    field = "stt.gpu_device_id"
    localize = partial(t_for_locale, "en")
    presentation = setting_catalog(
        {
            "fields": [field],
            "choices": {field: [{"value": "gpu-1", "display_name": "Test GPU", "available": True}]},
        },
        localize,
    )
    assert presentation[field]["choices"][0]["label"] == "Test GPU"
    blocks = change_blocks(
        {"changes": {field: "gpu-1"}, "previous": {field: "old-gpu"}, "presentation": presentation}
    )
    assert blocks[0].after == "Test GPU"
    assert blocks[0].before == t("settings.gpu_device.unavailable", device="old-gpu")
    masked = setting_value_text(
        "prompts.value", "SECRET", localize, choice_labels={"SECRET": "LEAK"}
    )
    assert "SECRET" not in masked
    assert "LEAK" not in masked
