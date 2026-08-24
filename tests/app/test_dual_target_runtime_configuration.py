from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.wiring.wiring_provider_runtime import (
    project_translation_runtime_settings,
)
from puripuly_heart.app.wiring.wiring_translation_runtime_configuration import (
    build_translation_runtime_config,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigCategory,
    TranslationRuntimeConfigurationOwner,
)


def test_settings_projection_resolves_one_or_two_unique_targets_in_configured_order() -> None:
    settings = AppSettings()
    settings.languages.target_language = "zh-CN"
    settings.languages.secondary_target_language = "ja"
    settings.validate()

    values = project_translation_runtime_settings(settings)
    configuration = build_translation_runtime_config(values)

    assert values.self_target_languages == ("zh-CN", "ja")
    assert configuration.target_language == "zh-CN"
    assert configuration.self_target_languages == ("zh-CN", "ja")


def test_secondary_target_change_is_a_self_language_runtime_change() -> None:
    initial = TranslationRuntimeConfig(
        target_language="en",
        self_target_languages=("en", "ja"),
    )
    owner = TranslationRuntimeConfigurationOwner(initial)

    change = owner.replace(
        replace(initial, self_target_languages=("en", "fr"))
    )

    assert change.self_language_changed is True
    assert change.changed_fields == {"self_target_languages"}
    assert change.categories == {TranslationRuntimeConfigCategory.LANGUAGES}


def test_runtime_configuration_rejects_more_than_two_targets() -> None:
    with pytest.raises(ValueError, match="at most two targets"):
        TranslationRuntimeConfig(
            target_language="en",
            self_target_languages=("en", "ja", "fr"),
        )
