from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from puripuly_heart.app.ports.settings_view import (
    ImmediateSettingsIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.settings.settings_application import (
    materialize_immediate_settings_intent,
    materialize_prompt_apply_intent,
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.ui.views.settings import SettingsView


def apply_provider_intent(
    settings: AppSettingsVNext, intent: ProviderApplyIntent
) -> AppSettingsVNext:
    return materialize_provider_apply_intent(
        settings,
        intent,
        materialize_translation=materialize_canonical_translation_settings,
    )


class SettingsIntentRecorder:
    def __init__(self) -> None:
        self.settings = AppSettingsVNext()
        self.immediate_intents: list[ImmediateSettingsIntent] = []
        self.prompt_intents: list[PromptApplyIntent] = []
        self.on_change: Callable[[AppSettingsVNext], None] | None = None

    def record_immediate(self, intent: ImmediateSettingsIntent) -> None:
        self.immediate_intents.append(intent)
        self.settings = materialize_immediate_settings_intent(self.settings, intent)
        if self.on_change is not None:
            self.on_change(self.settings)

    def record_prompt(self, intent: PromptApplyIntent) -> None:
        self.prompt_intents.append(intent)
        self.settings = materialize_prompt_apply_intent(self.settings, intent)
        if self.on_change is not None:
            self.on_change(self.settings)

    def load(
        self,
        view: SettingsView,
        settings: AppSettingsVNext,
        *,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> None:
        self.settings = settings
        view.on_settings_changed = self.record_immediate
        view.on_prompt_apply_settings = self.record_prompt
        provider, general, prompt, overlay = settings_view_surface_snapshots(settings)
        view.load_from_settings(
            provider=provider,
            general=general,
            prompt=prompt,
            overlay=overlay,
            config_path=config_path,
            preserve_custom_vocab_draft=preserve_custom_vocab_draft,
        )
