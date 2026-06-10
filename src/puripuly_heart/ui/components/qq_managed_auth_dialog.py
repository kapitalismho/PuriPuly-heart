from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.components.warm_document_dialog import (
    WarmDocumentDialogAction,
    WarmDocumentDialogResult,
    join_body_paragraphs,
    open_warm_document_dialog,
    split_body_paragraphs,
)
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_DIVIDER, COLOR_ON_BACKGROUND, COLOR_PRIMARY

_QQ_CREDENTIAL_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_RECOVERABLE_ERROR_KEYS = frozenset(
    {
        "qq_auth.error.invalid_input",
        "qq_auth.error.credential_mismatch",
        "qq_auth.error.lifetime_used",
        "qq_auth.error.retry",
    }
)
_MAX_DISPLAY_RETRY_AFTER_MS = 24 * 60 * 60 * 1000


class QqManagedAuthDialog:
    action_labels = [
        "qq_auth.close",
        "qq_auth.submit",
    ]
    waiting_action_labels = ["qq_auth.cancel"]

    def __init__(
        self,
        page: ft.Page,
        *,
        on_submit: Callable[[], None],
        on_close: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
        on_completion_message: Callable[[str], None] | None = None,
    ) -> None:
        self._page = page
        self._on_submit = on_submit
        self._on_close = on_close
        self._on_cancel = on_cancel
        self._on_completion_message = on_completion_message
        self._dialog: ft.AlertDialog | None = None
        self._is_open = False
        self._is_waiting = False
        self._auth_generation = 0
        self._pending_qq_identity = ""
        self._pending_credential = ""

        self._dialog_result: WarmDocumentDialogResult | None = None
        self._body_text: ft.Text | None = None
        self._actions: ft.Row | None = None
        self._submit_button: ft.TextButton | None = None
        self._close_button: ft.TextButton | None = None
        self._cancel_button: ft.TextButton | None = None
        self._qq_identity_field: ft.TextField | None = None
        self._credential_field: ft.TextField | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_waiting(self) -> bool:
        return self._is_waiting

    @property
    def auth_generation(self) -> int:
        return self._auth_generation

    @property
    def qq_identity(self) -> str:
        if self._qq_identity_field is None:
            return ""
        value = self._qq_identity_field.value
        return value.strip() if isinstance(value, str) else ""

    @property
    def credential(self) -> str:
        if self._credential_field is None:
            return ""
        value = self._credential_field.value
        return value if isinstance(value, str) else ""

    def open(self) -> None:
        if self._dialog is not None and self._is_open:
            return

        self._is_waiting = False
        self._pending_qq_identity = ""
        self._pending_credential = ""
        self._auth_generation += 1
        self._qq_identity_field = self._build_qq_identity_field()
        self._credential_field = self._build_credential_field()
        self._dialog_result = open_warm_document_dialog(
            self._page,
            body_paragraphs=split_body_paragraphs(t("qq_auth.body")),
            extra_body_controls=[self._qq_identity_field, self._credential_field],
            body_spacing=44,
            action_top_margin=24,
            actions=self._build_initial_actions(),
            glow_factory=create_glow_stack,
        )
        self._dialog = self._dialog_result.dialog
        self._body_text = self._dialog_result.body_text
        self._actions = self._dialog_result.action_row
        (
            self._close_button,
            self._submit_button,
        ) = self._dialog_result.initial_action_buttons[0:2]
        self._cancel_button = None
        self._is_open = True

    def _build_qq_identity_field(self, value: str = "") -> ft.TextField:
        return ft.TextField(
            label=t("qq_auth.qq_identity.label"),
            value=value,
            helper_text=t("qq_auth.qq_identity.helper"),
            dense=False,
            border_radius=14,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=20),
            text_size=22,
            color=COLOR_ON_BACKGROUND,
            on_submit=lambda _: self._submit(),
        )

    def _build_credential_field(self, value: str = "") -> ft.TextField:
        return ft.TextField(
            label=t("qq_auth.credential.label"),
            value=value,
            helper_text=t("qq_auth.credential.helper"),
            dense=False,
            border_radius=14,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=20),
            text_size=22,
            color=COLOR_ON_BACKGROUND,
            password=True,
            multiline=False,
            max_lines=1,
            on_submit=lambda _: self._submit(),
        )

    def set_waiting(self) -> None:
        if not self._is_open:
            return
        if not self._is_waiting:
            self._auth_generation += 1
        self._is_waiting = True
        if self._dialog_result is None or self._body_text is None:
            return

        self._capture_pending_values()
        self._remove_editable_fields()

        self._body_text.value = join_body_paragraphs(
            split_body_paragraphs(t("qq_auth.waiting_body"))
        )
        waiting_buttons = self._dialog_result.set_actions(self._build_waiting_actions())
        self._close_button = None
        self._submit_button = None
        self._cancel_button = waiting_buttons[0]
        self._update_page_if_possible()

    def set_recoverable_error(
        self,
        message_key: str,
        *,
        clear_credential: bool = False,
        message_kwargs: Mapping[str, object] | None = None,
        generation: int | None = None,
    ) -> bool:
        if self._is_stale_generation(generation) or not self._is_open:
            return False
        if self._dialog_result is None or self._body_text is None:
            return False

        self._capture_pending_values()
        safe_message_key = (
            message_key if message_key in _RECOVERABLE_ERROR_KEYS else "qq_auth.error.retry"
        )
        safe_message_kwargs = self._safe_message_kwargs(message_kwargs or {})
        error_text = t(safe_message_key, **safe_message_kwargs)
        qq_identity = self._pending_qq_identity
        credential = "" if clear_credential else self._pending_credential
        self._pending_qq_identity = qq_identity
        self._pending_credential = credential
        self._is_waiting = False

        self._body_text.value = join_body_paragraphs(
            [error_text, *split_body_paragraphs(t("qq_auth.body"))]
        )
        self._restore_editable_fields(
            qq_identity=qq_identity,
            credential=credential,
        )
        initial_buttons = self._dialog_result.set_actions(self._build_initial_actions())
        self._close_button, self._submit_button = initial_buttons[0:2]
        self._cancel_button = None
        self._update_page_if_possible()
        return True

    def complete_success(self, *, generation: int | None = None) -> bool:
        if self._is_stale_generation(generation) or not self._is_open:
            return False
        self._emit_completion_message("qq_auth.success")
        self.close()
        return True

    def complete_key_unavailable(self, *, generation: int | None = None) -> bool:
        if self._is_stale_generation(generation) or not self._is_open:
            return False
        self._emit_completion_message("qq_auth.error.key_unavailable")
        self.close()
        return True

    def complete_translation_enable_failed(
        self,
        message_key: str = "qq_auth.error.retry",
        *,
        generation: int | None = None,
    ) -> bool:
        if self._is_stale_generation(generation) or not self._is_open:
            return False
        safe_message_key = (
            message_key if message_key in _RECOVERABLE_ERROR_KEYS else "qq_auth.error.retry"
        )
        self._emit_completion_message(safe_message_key)
        self.close()
        return True

    def close(self) -> None:
        dialog = self._dialog
        if dialog is None or not self._is_open:
            self._is_open = False
            self._is_waiting = False
            self._clear_sensitive_state(clear_dialog_refs=True)
            return
        self._page.close(dialog)
        self._is_open = False
        self._is_waiting = False
        self._auth_generation += 1
        self._clear_sensitive_state(clear_dialog_refs=True)

    def _build_initial_actions(self) -> list[WarmDocumentDialogAction]:
        return [
            WarmDocumentDialogAction(
                label=t("qq_auth.close"),
                on_select=lambda: self._close_then(self._on_close),
                close_before_action=False,
            ),
            WarmDocumentDialogAction(
                label=t("qq_auth.submit"),
                on_select=self._submit,
                close_before_action=False,
            ),
        ]

    def _build_waiting_actions(self) -> list[WarmDocumentDialogAction]:
        return [
            WarmDocumentDialogAction(
                label=t("qq_auth.cancel"),
                on_select=self._cancel_waiting,
                close_before_action=False,
            )
        ]

    def _submit(self) -> None:
        if self._is_waiting:
            return
        validated = self._validated_inputs()
        if validated is None:
            self.set_recoverable_error("qq_auth.error.invalid_input")
            return

        qq_identity, credential = validated
        self._pending_qq_identity = qq_identity
        self._pending_credential = credential
        if self._qq_identity_field is not None:
            self._qq_identity_field.value = qq_identity
        if self._credential_field is not None:
            self._credential_field.value = credential
        self._is_waiting = True
        self._auth_generation += 1
        if self._submit_button is not None:
            self._submit_button.disabled = True
        self._update_page_if_possible()
        self._on_submit()

    def _validated_inputs(self) -> tuple[str, str] | None:
        qq_identity = self._validate_qq_identity(self.qq_identity)
        credential = self._validate_credential(self.credential)
        if qq_identity is None or credential is None:
            return None
        return qq_identity, credential

    def _validate_qq_identity(self, value: str) -> str | None:
        qq_identity = value.strip()
        if not 1 <= len(qq_identity) <= 128:
            return None
        if any(character.isspace() for character in qq_identity):
            return None
        if any(unicodedata.category(character).startswith("C") for character in qq_identity):
            return None
        return qq_identity

    def _validate_credential(self, value: str) -> str | None:
        if _QQ_CREDENTIAL_PATTERN.fullmatch(value) is None:
            return None
        return value

    def _capture_pending_values(self) -> None:
        if self._qq_identity_field is not None:
            self._pending_qq_identity = self.qq_identity
        if self._credential_field is not None:
            self._pending_credential = self.credential

    def _remove_editable_fields(self) -> None:
        if self._dialog_result is not None:
            body_controls = self._dialog_result.body_column.controls
            for field in (self._qq_identity_field, self._credential_field):
                if field is not None and field in body_controls:
                    body_controls.remove(field)
        if self._qq_identity_field is not None:
            self._qq_identity_field.value = ""
        if self._credential_field is not None:
            self._credential_field.value = ""
        self._qq_identity_field = None
        self._credential_field = None

    def _restore_editable_fields(self, *, qq_identity: str, credential: str) -> None:
        if self._dialog_result is None or self._body_text is None:
            return
        self._qq_identity_field = self._build_qq_identity_field(qq_identity)
        self._credential_field = self._build_credential_field(credential)
        self._dialog_result.body_column.controls = [
            self._body_text,
            self._qq_identity_field,
            self._credential_field,
        ]

    def _clear_sensitive_state(self, *, clear_dialog_refs: bool) -> None:
        if self._qq_identity_field is not None:
            self._qq_identity_field.value = ""
        if self._credential_field is not None:
            self._credential_field.value = ""
        self._pending_qq_identity = ""
        self._pending_credential = ""
        self._qq_identity_field = None
        self._credential_field = None
        self._close_button = None
        self._submit_button = None
        self._cancel_button = None
        if clear_dialog_refs:
            self._dialog = None
            self._dialog_result = None
            self._body_text = None
            self._actions = None

    def _safe_message_kwargs(
        self,
        message_kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        retry_after_ms = self._sanitize_retry_after_ms(message_kwargs.get("retry_after_ms"))
        if retry_after_ms is None:
            return {}
        return {"retry_after_ms": retry_after_ms}

    def _sanitize_retry_after_ms(self, value: object) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            retry_after_ms = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, min(retry_after_ms, _MAX_DISPLAY_RETRY_AFTER_MS))

    def _is_stale_generation(self, generation: int | None) -> bool:
        return generation is not None and generation != self._auth_generation

    def _emit_completion_message(self, message_key: str) -> None:
        if self._on_completion_message is not None:
            self._on_completion_message(message_key)

    def _cancel_waiting(self) -> None:
        self.close()
        if self._on_cancel is not None:
            self._on_cancel()
        else:
            self._on_close()

    def _close_then(self, action: Callable[[], None]) -> None:
        self.close()
        action()

    def _update_page_if_possible(self) -> None:
        update = getattr(self._page, "update", None)
        if callable(update):
            update()
