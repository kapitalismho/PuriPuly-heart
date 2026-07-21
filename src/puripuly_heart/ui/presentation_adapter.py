from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.ui_presentation import UiPresentationPort


@dataclass(slots=True)
class FletUiPresentationAdapter:
    _app: UiPresentationPort

    @property
    def view_dashboard(self) -> object | None:
        return getattr(self._app, "view_dashboard", None)

    @property
    def view_settings(self) -> object | None:
        return getattr(self._app, "view_settings", None)

    @property
    def view_logs(self) -> object | None:
        return getattr(self._app, "view_logs", None)

    @property
    def debug_ui_preview(self) -> bool:
        return bool(getattr(self._app, "debug_ui_preview", False))

    def refresh_overlay_peer_contract(self) -> None:
        self._app.refresh_overlay_peer_contract()

    def apply_locale(self) -> None:
        self._app.apply_locale()

    def add_history_entry(self, *args, **kwargs) -> None:
        self._app.add_history_entry(*args, **kwargs)

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        return self._app.get_event_language_codes()

    def is_event_translation_enabled(self) -> bool:
        return self._app.is_event_translation_enabled()

    def get_event_stt_state(self) -> object | None:
        return self._app.get_event_stt_state()

    def clear_managed_auth_pending_state(self) -> None:
        self._app.clear_managed_auth_pending_state()

    def show_snackbar(self, *args, **kwargs) -> None:
        self._app.show_snackbar(*args, **kwargs)

    def on_github_star_translation_success(self) -> None:
        self._app.on_github_star_translation_success()

    def on_telemetry_translation_success(self) -> None:
        self._app.on_telemetry_translation_success()

    def on_overlay_state_changed(self, **kwargs) -> None:
        self._app.on_overlay_state_changed(**kwargs)

    def on_desktop_overlay_state_changed(self, *args, **kwargs) -> None:
        self._app.on_desktop_overlay_state_changed(*args, **kwargs)

    def show_qq_managed_auth_dialog(self) -> None:
        self._app.show_qq_managed_auth_dialog()

    def show_founder_letter_dialog(self) -> None:
        self._app.show_founder_letter_dialog()

    def show_local_qwen_hallucination_dialog(self) -> None:
        self._app.show_local_qwen_hallucination_dialog()

    async def close_after_launch_tasks(self) -> None:
        await self._app.close_after_launch_tasks()

    async def close_github_star_prompt_runtime(self) -> None:
        await self._app.close_github_star_prompt_runtime()

    async def close_oauth_runtime(self) -> None:
        await self._app.close_oauth_runtime()


__all__ = ["FletUiPresentationAdapter"]
