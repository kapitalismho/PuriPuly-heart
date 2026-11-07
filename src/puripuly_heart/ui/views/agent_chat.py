from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import flet as ft

from puripuly_heart.app.ports.agent_chat import AgentChatPort, AgentChatSnapshot
from puripuly_heart.ui.agent_chat_model import (
    ChatItem,
    LeftView,
    applied_label,
    change_blocks,
    chat_items,
    done_status,
    history_rows,
    last_reply_index,
    select_left,
)
from puripuly_heart.ui.components.power_button import PowerButton
from puripuly_heart.ui.flet_runtime import is_hover_active, update_control_if_mounted
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import provider_label, t
from puripuly_heart.ui.motion import press
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_DISPLAY_SOURCE,
    COLOR_DIVIDER,
    COLOR_ERROR,
    COLOR_NEUTRAL,
    COLOR_ON_BACKGROUND,
    COLOR_ON_PRIMARY_CONTAINER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_TRANS_ON,
    COLOR_TRANS_TONAL,
)

LEFT_WIDTH = 491
BUBBLE_GAP = 72
SCROLL_END_SLACK = 24.0
SCROLL_RESTORE_DELAY_S = 0.05
HISTORY_BUBBLE_WIDTH = 380
FADE = ft.Animation(200, ft.AnimationCurve.EASE_OUT)
HOVER = ft.Animation(200, ft.AnimationCurve.EASE_OUT)
DICTATION_LIMIT_S = 60.0
HISTORY_REFRESH_S = 0.5
X_CONTACT_URL = "https://x.com/kapitalismho"


class DictationSource(Protocol):
    def observe(self, callback: Callable[[str, str, bool], None]) -> Callable[[], None]: ...

    def ensure_capture(self) -> Callable[[], None]: ...


@dataclass(slots=True)
class AgentChatHooks:
    open_folder: Callable[[], None] | None = None
    show_snackbar: Callable[[str, str], None] | None = None
    conversation_records: Callable[[], Sequence[object]] | None = None
    dictation: DictationSource | None = None
    run_task: Callable[[Callable[[], object]], object] | None = None


class _Dictation:
    def __init__(self) -> None:
        self.active = False
        self.order: list[str] = []
        self.texts: dict[str, str] = {}
        self.final: set[str] = set()
        self.base = ""
        self.unsubscribe: Callable[[], None] | None = None
        self.restore: Callable[[], None] | None = None
        self.timer: asyncio.Task | None = None

    def text(self) -> str:
        committed = [self.texts[key] for key in self.order if key in self.final and self.texts[key]]
        return " ".join(part for part in [self.base, *committed] if part)


class _ScrollMemory:
    """Keeps a list at its end while the reader is there, otherwise at the last read offset."""

    def __init__(self, listing: ft.ListView, run: Callable[[Callable[[], object]], object]) -> None:
        self.listing = listing
        self.offset = 0.0
        self.at_end = True
        self._run = run
        listing.on_scroll = self._scrolled
        listing.scroll_interval = 100

    def _scrolled(self, event) -> None:
        self.offset = float(event.pixels)
        self.at_end = event.pixels >= event.max_scroll_extent - SCROLL_END_SLACK

    def follow(self) -> None:
        if self.at_end:
            self._run(partial(self._scroll, -1))

    def restore(self) -> None:
        self._run(partial(self._scroll, -1 if self.at_end else self.offset, wait=True))

    async def _scroll(self, offset: float, *, wait: bool = False) -> None:
        if wait:
            await asyncio.sleep(SCROLL_RESTORE_DELAY_S)
        try:
            await self.listing.scroll_to(offset=offset)
        except Exception:
            return


class AgentChatView(ft.Row):
    def __init__(self, owner: AgentChatPort, hooks: AgentChatHooks | None = None) -> None:
        self._owner = owner
        self._hooks = hooks or AgentChatHooks()
        self._unsubscribe: Callable[[], None] | None = None
        self._snapshot = owner.snapshot
        self._history_open = False
        self._key_entry_step: set[str] = set()
        self._dismissed: set[str] = set()
        self._turn_lines: dict[int, str] = {}
        self._pending_lines: list[str] = []
        self._left_key: str | None = None
        self._displayed: set[str] = set()
        self._action_pending = False
        self._key_failed: set[str] = set()
        self._dictation = _Dictation()
        self._chat_keys: tuple[str, ...] = ()
        self._chat_rows: dict[str, ft.Control] = {}
        self._input_edits = 0

        self._history_tile = PowerButton("HISTORY", ft.Icons.FORUM, self._open_history)
        self._folder_tile = PowerButton("FOLDER", ft.Icons.FOLDER, self._open_folder)
        self._new_tile = PowerButton("NEW", ft.Icons.REFRESH, self._new_conversation)
        self._voice_tile = PowerButton("VOICE", ft.Icons.MIC, self._toggle_voice)
        self._home = ft.Column(
            [
                ft.Row(
                    [self._history_tile, self._folder_tile],
                    expand=True,
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
                ft.Row(
                    [self._new_tile, self._voice_tile],
                    expand=True,
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ],
            expand=True,
            spacing=12,
            key="home",
        )
        self._switcher = ft.AnimatedSwitcher(
            self._home,
            duration=200,
            reverse_duration=200,
            switch_in_curve=ft.AnimationCurve.EASE_OUT,
            switch_out_curve=ft.AnimationCurve.EASE_OUT,
            transition=ft.AnimatedSwitcherTransition.FADE,
            expand=True,
        )
        self._messages = ft.Column(
            spacing=18,
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            auto_scroll_animation=0,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True,
        )
        self._history_list = ft.ListView(expand=True, spacing=14)
        self._history_scroll = _ScrollMemory(self._history_list, self._run)
        self._history_keys: tuple = ()
        self._history_rows: dict[str, tuple[object, ft.Control]] = {}
        self._history_panel: ft.Container | None = None
        self._history_back: ft.Container | None = None
        self._history_polling = False
        self._mounted = False
        self._input = ft.TextField(
            border_radius=16,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            bgcolor=COLOR_BACKGROUND,
            hover_color=COLOR_BACKGROUND,
            text_size=20,
            content_padding=ft.Padding.symmetric(horizontal=18, vertical=18),
            expand=True,
            on_change=self._input_changed,
            on_submit=self._submit,
        )
        chat = ft.Container(
            content=ft.Column(
                [
                    ft.Container(content=self._messages, padding=24, expand=True),
                    ft.Container(
                        content=ft.Row([self._input]),
                        padding=ft.Padding.only(left=24, right=24, bottom=24, top=12),
                    ),
                ],
                expand=True,
                spacing=0,
            ),
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            expand=True,
        )
        super().__init__(
            controls=[ft.Container(content=self._switcher, width=LEFT_WIDTH), chat],
            expand=True,
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self._render(self._snapshot)

    def did_mount(self) -> None:
        self._mounted = True
        self._unsubscribe = self._owner.observe(self._render)
        self._render(self._owner.snapshot)
        if self._history_open:
            self._history_card()
            self._history_scroll.restore()
            self._start_history_polling()

    def will_unmount(self) -> None:
        self._mounted = False
        if self._unsubscribe is not None:
            self._unsubscribe()
        self._unsubscribe = None
        self._stop_dictation(send=False)

    def use_owner(self, owner: AgentChatPort) -> None:
        if owner is self._owner:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._owner = owner
        if self._mounted:
            self._unsubscribe = owner.observe(self._render)
        self._render(owner.snapshot)

    # Rendering

    def _render(self, snapshot: AgentChatSnapshot) -> None:
        self._snapshot = snapshot
        self._record_turn_line(snapshot)
        self._render_chat(snapshot)
        self._render_left(snapshot)
        unavailable = not snapshot.enabled or snapshot.phase == "closed"
        self._input.disabled = unavailable
        self._voice_tile.disabled = unavailable
        self._new_tile.disabled = unavailable
        update_control_if_mounted(self)

    def _record_turn_line(self, snapshot: AgentChatSnapshot) -> None:
        if snapshot.busy:
            return
        index = last_reply_index(snapshot)
        if index is None or index in self._turn_lines:
            return
        parts = list(self._pending_lines)
        done = done_status(snapshot.tool_results)
        if done:
            parts.append(done)
        if parts:
            self._turn_lines[index] = " · ".join(parts)
        self._pending_lines.clear()

    def _render_chat(self, snapshot: AgentChatSnapshot) -> None:
        items = chat_items(snapshot, turn_lines=self._turn_lines)
        keys = tuple(
            f"{item.key}:{len(item.lines)}:{item.archived}:{item.busy}:{item.lines}"
            for item in items
        )
        if keys == self._chat_keys:
            return
        self._chat_keys = keys
        cached = self._chat_rows
        self._chat_rows = {
            key: cached.get(key) or self._chat_control(item) for key, item in zip(keys, items)
        }
        self._messages.controls = list(self._chat_rows.values())

    def _chat_control(self, item: ChatItem) -> ft.Control:
        opacity = 0.35 if item.archived else 1.0
        if item.kind == "user":
            return ft.Row(
                [
                    ft.Container(width=BUBBLE_GAP),
                    ft.Container(
                        content=ft.Text(
                            item.lines[0],
                            size=19,
                            color=COLOR_ON_PRIMARY_CONTAINER,
                            selectable=True,
                        ),
                        bgcolor=COLOR_PRIMARY_CONTAINER,
                        padding=ft.Padding.symmetric(horizontal=18, vertical=13),
                        border_radius=ft.BorderRadius.only(
                            top_left=22, top_right=22, bottom_left=22, bottom_right=8
                        ),
                        expand=True,
                        expand_loose=True,
                    ),
                ],
                alignment=ft.MainAxisAlignment.END,
                opacity=opacity,
                animate_opacity=FADE,
                key=item.key,
            )
        if item.kind in {"assistant", "greeting"}:
            count = len(item.lines)
            bubbles = []
            for position, line in enumerate(item.lines):
                bubbles.append(
                    ft.Container(
                        content=ft.Text(line, size=19, color=COLOR_ON_BACKGROUND, selectable=True),
                        bgcolor=COLOR_BACKGROUND,
                        padding=ft.Padding.symmetric(horizontal=18, vertical=13),
                        border_radius=ft.BorderRadius.only(
                            top_left=22 if position == 0 else 8,
                            top_right=22,
                            bottom_right=22,
                            bottom_left=22 if position == count - 1 else 8,
                        ),
                        border=ft.Border.only(bottom=ft.BorderSide(1, COLOR_DIVIDER)),
                        opacity=1.0,
                        animate_opacity=FADE,
                    )
                )
            return ft.Container(
                content=ft.Column(
                    bubbles,
                    spacing=5,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
                padding=ft.Padding.only(right=BUBBLE_GAP),
                opacity=opacity,
                animate_opacity=FADE,
                key=item.key,
            )
        if item.kind == "status":
            row: list[ft.Control] = []
            if item.busy:
                row.append(
                    ft.ProgressRing(width=16, height=16, stroke_width=2.5, color=COLOR_PRIMARY)
                )
            row.append(ft.Text(item.lines[0], size=16, color=COLOR_NEUTRAL, expand=True))
            return ft.Row(row, spacing=8, opacity=opacity, key=item.key)
        return ft.Row(
            [
                ft.Text(
                    item.lines[0],
                    size=16,
                    color=COLOR_NEUTRAL,
                    text_align=ft.TextAlign.CENTER,
                    expand=True,
                )
            ],
            opacity=opacity,
            key=item.key,
        )

    def _render_left(self, snapshot: AgentChatSnapshot) -> None:
        view = select_left(
            snapshot,
            history_open=self._history_open,
            key_entry_step=self._key_entry_step,
            dismissed=self._dismissed,
        )
        key = view.key
        if view.kind == "change":
            key = f"{key}:{self._action_pending}"
        if view.kind == "key_entry" and view.secure_request is not None:
            key = f"{key}:{view.secure_request.status}:{view.secure_request.id in self._key_failed}"
        if key == self._left_key:
            return
        self._left_key = key
        content = self._left_control(view, snapshot)
        self._switcher.content = ft.Container(content=content, key=key, expand=True)
        self._acknowledge_display(view)

    def _acknowledge_display(self, view: LeftView) -> None:
        if view.kind in {"key_create", "account"} and view.setup_action is not None:
            identity = f"setup:{view.setup_action.id}"
            if identity not in self._displayed and not view.setup_action.displayed:
                self._displayed.add(identity)
                self._owner.mark_setup_action_displayed(view.setup_action.id)
        if view.kind in {"key_create", "key_entry"} and view.secure_request is not None:
            identity = f"secure:{view.secure_request.id}"
            if identity not in self._displayed and not view.secure_request.displayed:
                self._displayed.add(identity)
                self._owner.mark_secure_entry_displayed(view.secure_request.id)
        if view.kind == "report" and view.report is not None:
            identity = f"report:{view.report.id}"
            if identity not in self._displayed and not view.report.displayed:
                self._displayed.add(identity)
                self._owner.mark_report_displayed(view.report.id)

    def _left_control(self, view: LeftView, snapshot: AgentChatSnapshot) -> ft.Control:
        if view.kind == "home":
            return self._home
        if view.kind == "history":
            return self._history_card()
        if view.kind == "change" and view.confirmation is not None:
            card = {**view.confirmation, "previous": snapshot.confirmation_previous or {}}
            return self._change_card(card)
        if view.kind == "key_create" and view.setup_action is not None:
            action = view.setup_action
            return self._tap_card(
                [self._value_block(provider_label(action.provider), t("agent_chat.key.create"))],
                on_tap=lambda: self._open_key_console(view),
                on_back=lambda: self._leave_key_create(view),
            )
        if view.kind == "key_entry" and view.secure_request is not None:
            return self._key_entry_card(view)
        if view.kind == "report" and view.report is not None:
            return self._report_card(view)
        if view.kind == "account" and view.setup_action is not None:
            action = view.setup_action
            value = t(
                "agent_chat.account.usage"
                if action.kind == "public_usage"
                else "agent_chat.account.login"
            )
            return self._tap_card(
                [self._value_block(provider_label(action.provider), value)],
                on_tap=lambda: self._activate_account(action.id),
                on_back=lambda: self._dismiss(f"account:{action.id}"),
            )
        return self._home

    # Card building blocks

    def _back_button(self, on_back: Callable[[], None]) -> ft.Control:
        text = ft.Text(t("agent_chat.back"), size=20, color=COLOR_SECONDARY)

        def hover(event) -> None:
            text.color = COLOR_PRIMARY if is_hover_active(event) else COLOR_SECONDARY
            update_control_if_mounted(text)

        return ft.Container(
            content=text,
            height=52,
            padding=ft.Padding.symmetric(horizontal=8),
            alignment=ft.Alignment.CENTER_LEFT,
            on_click=lambda _event: on_back(),
            on_hover=hover,
        )

    def _option(self, text: str, *, size: int = 28) -> ft.Text:
        return ft.Text(text, size=size, color=COLOR_SECONDARY, text_align=ft.TextAlign.CENTER)

    def _value(self, text: str, *, size: int = 48, max_lines: int | None = None) -> ft.Text:
        return ft.Text(
            text,
            size=size,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
            max_lines=max_lines,
            overflow=ft.TextOverflow.ELLIPSIS if max_lines else None,
        )

    def _value_block(self, option: str, value: str) -> ft.Control:
        return ft.Column(
            [self._option(option), ft.Container(height=20), self._value(value)],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _panel(self, controls: list[ft.Control]) -> ft.Container:
        return ft.Container(
            content=ft.Column(controls, spacing=24, expand=True),
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            padding=24,
            expand=True,
            scale=1.0,
            animate_scale=HOVER,
        )

    def _tap_card(
        self,
        body: list[ft.Control],
        *,
        on_tap: Callable[[], None] | None,
        on_back: Callable[[], None],
        busy: bool = False,
        gap: int = 56,
    ) -> ft.Control:
        values = [
            control
            for block in body
            for control in _texts(block)
            if control.color == COLOR_ON_BACKGROUND
        ]
        arrows = [control for block in body for control in _icons(block)]
        zone = ft.Container(
            content=ft.Column(
                body,
                spacing=gap,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            ),
            expand=True,
            alignment=ft.Alignment.CENTER,
        )
        panel = self._panel([self._back_button(on_back), zone])
        if on_tap is None or busy:
            return panel

        hovered = [False]

        def rest() -> float:
            return 1.02 if hovered[0] else 1.0

        def hover(event) -> None:
            active = is_hover_active(event)
            hovered[0] = active
            panel.scale = rest()
            for value in values:
                value.color = COLOR_PRIMARY if active else COLOR_ON_BACKGROUND
            for arrow in arrows:
                arrow.color = COLOR_PRIMARY if active else COLOR_SECONDARY
            update_control_if_mounted(panel)

        zone.on_hover = hover
        zone.on_tap_down = lambda _event: press(panel, rest=rest, animation=HOVER)
        zone.on_click = lambda _event: on_tap()
        return panel

    def _arrow(self, size: int = 44) -> ft.Icon:
        return ft.Icon(ft.Icons.ARROW_DOWNWARD, size=size, color=COLOR_SECONDARY)

    # Cards

    def _change_card(self, card: dict) -> ft.Control:
        blocks = change_blocks(card)
        card_id = card["id"]
        if len(blocks) <= 1:
            block = blocks[0] if blocks else None
            body: list[ft.Control] = []
            if block is not None:
                body = [
                    ft.Column(
                        [
                            self._option(block.label),
                            ft.Container(height=20),
                            self._value(block.before or "—"),
                        ],
                        spacing=0,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._arrow(),
                    self._value(
                        block.after,
                        size=24 if block.long_text else 48,
                        max_lines=2 if block.long_text else None,
                    ),
                ]
            return self._tap_card(
                body,
                on_tap=lambda: self._confirm(card_id, True),
                on_back=lambda: self._confirm(card_id, False),
                busy=self._action_pending,
            )
        stacked: list[ft.Control] = []
        for position, block in enumerate(blocks):
            column = ft.Column(
                [
                    self._option(block.label),
                    self._value(block.before or "—", size=36),
                    self._arrow(36),
                    self._value(
                        block.after,
                        size=24 if block.long_text else 36,
                        max_lines=2 if block.long_text else None,
                    ),
                ],
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
            if position:
                column = ft.Container(
                    content=column,
                    border=ft.Border.only(top=ft.BorderSide(1, COLOR_DIVIDER)),
                    padding=ft.Padding.only(top=40),
                )
            stacked.append(column)
        return self._tap_card(
            stacked,
            on_tap=lambda: self._confirm(card_id, True),
            on_back=lambda: self._confirm(card_id, False),
            busy=self._action_pending,
            gap=0,
        )

    def _key_entry_card(self, view: LeftView) -> ft.Control:
        request = view.secure_request
        assert request is not None
        failed = request.id in self._key_failed or request.status == "failed"
        field = ft.TextField(
            password=True,
            width=LEFT_WIDTH - 48,
            border_radius=16,
            border_width=2,
            border_color=COLOR_ERROR if failed else COLOR_DIVIDER,
            focused_border_color=COLOR_ERROR if failed else COLOR_PRIMARY,
            bgcolor=COLOR_BACKGROUND,
            text_size=24,
            height=80,
            content_padding=ft.Padding.symmetric(horizontal=20, vertical=22),
            disabled=request.status == "verifying",
        )

        async def blur(_event) -> None:
            value = field.value or ""
            if value.strip():
                await self._submit_key(request.id, request.provider, value)

        field.on_blur = blur
        field.on_submit = blur
        body = ft.Column(
            [self._option(t("agent_chat.key.label")), ft.Container(height=20), field],
            spacing=0,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True,
        )
        panel = self._panel(
            [
                self._back_button(lambda: self._leave_key_entry(view)),
                ft.Container(content=body, expand=True, alignment=ft.Alignment.CENTER),
            ]
        )
        return panel

    def _report_card(self, view: LeftView) -> ft.Control:
        offer = view.report
        assert offer is not None
        dismiss = lambda: self._dismiss(f"report:{offer.id}")  # noqa: E731
        if offer.status == "sending":
            return self._tap_card(
                [self._value_block(t("agent_chat.report.option"), t("agent_chat.report.sending"))],
                on_tap=None,
                on_back=dismiss,
                busy=True,
            )
        if offer.status == "sent":
            ticket = ft.Text(
                offer.ticket or "", size=24, color=COLOR_SECONDARY, text_align=ft.TextAlign.CENTER
            )
            return self._tap_card(
                [
                    ft.Column(
                        [
                            ticket,
                            ft.Container(height=20),
                            self._value(t("agent_chat.report.contact")),
                        ],
                        spacing=0,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    )
                ],
                on_tap=lambda: webbrowser.open(X_CONTACT_URL),
                on_back=dismiss,
            )
        if offer.status == "failed":
            return self._tap_card(
                [self._value_block(t("agent_chat.report.failed"), t("agent_chat.report.retry"))],
                on_tap=lambda: self._send_report(offer.id),
                on_back=dismiss,
            )
        return self._tap_card(
            [self._value_block(t("agent_chat.report.option"), t("agent_chat.report.send"))],
            on_tap=lambda: self._send_report(offer.id),
            on_back=lambda: self._reject_report(offer.id),
        )

    def _history_card(self) -> ft.Control:
        if self._history_panel is None or self._history_back is None:
            self._history_back = self._back_button(self._close_history)
            self._history_panel = self._panel(
                [self._history_back, ft.SelectionArea(content=self._history_list, expand=True)]
            )
        back_text = self._history_back.content
        if isinstance(back_text, ft.Text):
            back_text.value = t("agent_chat.back")
            back_text.color = COLOR_SECONDARY
        self._refresh_history()
        return self._history_panel

    def _refresh_history(self) -> bool:
        history = history_rows(self._records())
        keys = tuple(history)
        if keys == self._history_keys:
            return False
        self._history_keys = keys
        rows: dict[str, tuple[object, ft.Control]] = {}
        controls: list[ft.Control] = []
        for row in history:
            cached = self._history_rows.get(row.key)
            control = (
                cached[1] if cached is not None and cached[0] == row else self._history_row(row)
            )
            if control is None:
                continue
            rows[row.key] = (row, control)
            controls.append(control)
        self._history_rows = rows
        self._history_list.controls = controls
        self._history_scroll.follow()
        return True

    def _history_row(self, row) -> ft.Control | None:
        texts: list[ft.Control] = []
        if row.source:
            texts.append(
                ft.Text(
                    row.source,
                    size=16,
                    color=COLOR_DISPLAY_SOURCE,
                    font_family=font_for_language(row.source_language),
                )
            )
        languages = row.translation_languages or (None,) * len(row.translations)
        for translation, language in zip(row.translations, languages, strict=False):
            texts.append(
                ft.Text(
                    translation,
                    size=20,
                    color=COLOR_TRANS_ON if row.mine else COLOR_ON_BACKGROUND,
                    font_family=font_for_language(language),
                )
            )
        if row.failed:
            texts.append(
                ft.Text(t("logs.conversation.target.failed"), size=17, color=COLOR_TRANS_ON)
            )
        if not texts:
            return None
        bubble = ft.Container(
            content=ft.Column(texts, spacing=4, tight=True),
            bgcolor=COLOR_TRANS_TONAL if row.mine else COLOR_BACKGROUND,
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            border_radius=ft.BorderRadius.only(
                top_left=20,
                top_right=20,
                bottom_left=20 if row.mine else 8,
                bottom_right=8 if row.mine else 20,
            ),
            border=None if row.mine else ft.Border.only(bottom=ft.BorderSide(1, COLOR_DIVIDER)),
        )
        side = ft.CrossAxisAlignment.END if row.mine else ft.CrossAxisAlignment.START
        entry = ft.Column(
            [
                bubble,
                ft.Container(
                    ft.Text(row.time_label, size=14, color=COLOR_NEUTRAL, selectable=True),
                    padding=ft.Padding.symmetric(horizontal=6),
                ),
            ],
            spacing=4,
            tight=True,
            horizontal_alignment=side,
            width=HISTORY_BUBBLE_WIDTH,
        )
        return ft.Row(
            [entry],
            alignment=ft.MainAxisAlignment.END if row.mine else ft.MainAxisAlignment.START,
            key=row.key,
        )

    def _start_history_polling(self) -> None:
        if self._history_polling or not self._mounted:
            return
        self._history_polling = True
        if self._run(self._poll_history) is None:
            self._history_polling = False

    async def _poll_history(self) -> None:
        try:
            while self._history_open and self._mounted:
                await asyncio.sleep(HISTORY_REFRESH_S)
                if self._history_open and self._mounted and self._refresh_history():
                    update_control_if_mounted(self._history_list)
        finally:
            self._history_polling = False

    def _records(self) -> Sequence[object]:
        provider = self._hooks.conversation_records
        return tuple(provider()) if provider is not None else ()

    # Actions

    def _open_history(self) -> None:
        self._history_open = True
        self._render(self._owner.snapshot)
        self._history_scroll.restore()
        self._start_history_polling()

    def _close_history(self) -> None:
        self._history_open = False
        self._render(self._owner.snapshot)

    def _open_folder(self) -> None:
        if self._hooks.open_folder is not None:
            self._hooks.open_folder()

    def _new_conversation(self) -> None:
        self._owner.new_conversation()
        self._history_open = False
        self._render(self._owner.snapshot)

    def _dismiss(self, identity: str) -> None:
        self._dismissed.add(identity)
        self._render(self._owner.snapshot)

    def _confirm(self, card_id: str, approved: bool) -> None:
        card = self._owner.snapshot.confirmation
        if self._action_pending or card is None or card.get("id") != card_id:
            return
        self._action_pending = True
        self._render(self._owner.snapshot)

        async def run() -> None:
            try:
                result = await self._owner.confirm(card_id, approved)
                if approved and result.get("status") in {"applied", "degraded"}:
                    changes = result.get("changes")
                    label = applied_label(changes) if isinstance(changes, dict) else None
                    if label:
                        self._pending_lines.append(label)
            finally:
                self._action_pending = False
                self._render(self._owner.snapshot)

        self._run(run)

    def _open_key_console(self, view: LeftView) -> None:
        action = view.setup_action
        if action is None:
            return
        result = self._owner.activate_setup_action(action.id)
        if view.secure_request is not None and result.get("status") == "opened":
            self._key_entry_step.add(view.secure_request.id)
        self._render(self._owner.snapshot)

    def _leave_key_create(self, view: LeftView) -> None:
        if view.secure_request is not None:
            self._owner.cancel_secure_entry(view.secure_request.id)
            self._dismissed.add(f"secure:{view.secure_request.id}")
        if view.setup_action is not None:
            self._dismissed.add(f"setup:{view.setup_action.id}")
        self._render(self._owner.snapshot)

    def _leave_key_entry(self, view: LeftView) -> None:
        request = view.secure_request
        if request is None:
            return
        if view.setup_action is not None and view.setup_action.available:
            self._key_entry_step.discard(request.id)
        else:
            self._owner.cancel_secure_entry(request.id)
            self._dismissed.add(f"secure:{request.id}")
        self._render(self._owner.snapshot)

    async def _submit_key(self, request_id: str, provider: str, value: str) -> None:
        result = await self._owner.submit_secure_entry(request_id, value)
        status = result.get("status")
        if status == "verified":
            self._key_failed.discard(request_id)
            self._snackbar(
                t("snackbar.verification_ok", provider=provider_label(provider)),
                ft.Colors.GREEN_400,
            )
        elif status == "failed":
            self._key_failed.add(request_id)
            request = next(
                (
                    item
                    for item in self._owner.snapshot.secure_entry_requests
                    if item.id == request_id
                ),
                None,
            )
            code = request.code if request is not None and request.code else "verification_failed"
            message = t(code, default=code) if code.startswith("error.") else code
            self._snackbar(t("snackbar.verification_failed", message=message), ft.Colors.RED_400)
        self._render(self._owner.snapshot)

    def _snackbar(self, message: str, color: str) -> None:
        if self._hooks.show_snackbar is not None:
            self._hooks.show_snackbar(message, color)

    def _send_report(self, offer_id: str) -> None:

        async def run() -> None:
            await self._owner.send_report(offer_id)
            self._render(self._owner.snapshot)

        self._run(run)

    def _reject_report(self, offer_id: str) -> None:
        self._owner.reject_report(offer_id)
        self._dismissed.add(f"report:{offer_id}")
        self._render(self._owner.snapshot)

    def _activate_account(self, action_id: str) -> None:
        self._owner.activate_setup_action(action_id)
        self._render(self._owner.snapshot)

    async def _submit(self, _event=None) -> None:
        if self._dictation.active:
            self._stop_dictation(send=False)
        text = self._input.value or ""
        if not text.strip():
            return
        self._input.value = ""
        self._input_edits += 1
        edits = self._input_edits
        update_control_if_mounted(self._input)
        result = await self._owner.send(text)
        if result.status in {"rejected", "unavailable"} and self._input_edits == edits:
            self._input.value = text
        self._render(self._owner.snapshot)

    def _input_changed(self, _event=None) -> None:
        self._input_edits += 1

    def ask_about_error(self, error_text: str, diagnostic_id: str) -> None:
        self._history_open = False

        async def run() -> None:
            await self._owner.send(
                t("agent_chat.error_entry", error=error_text), diagnostic_id=diagnostic_id
            )
            self._render(self._owner.snapshot)

        self._render(self._owner.snapshot)
        self._run(run)

    def _run(self, coroutine_factory: Callable[[], object]) -> object | None:
        if self._hooks.run_task is not None:
            return self._hooks.run_task(coroutine_factory)
        try:
            page = self.page
        except RuntimeError:
            return None
        return _run(page, coroutine_factory)

    # Dictation

    def _toggle_voice(self) -> None:
        if self._dictation.active:
            self._stop_dictation(send=True)
        else:
            self._start_dictation()

    def _start_dictation(self) -> None:
        source = self._hooks.dictation
        if source is None or self._dictation.active:
            return
        state = _Dictation()
        state.active = True
        state.base = (self._input.value or "").strip()
        self._dictation = state

        def heard(utterance_id: str, text: str, final: bool) -> None:
            if not state.active:
                return
            if utterance_id not in state.texts:
                state.order.append(utterance_id)
            state.texts[utterance_id] = text.strip()
            if final:
                state.final.add(utterance_id)
                self._show_dictation()

        state.unsubscribe = source.observe(heard)
        state.restore = source.ensure_capture()

        async def limit() -> None:
            await asyncio.sleep(DICTATION_LIMIT_S)
            if self._dictation is state and state.active:
                self._stop_dictation(send=False)

        state.timer = self._run(limit)
        self._voice_tile.set_label("STOP")
        self._voice_tile.set_icon(ft.Icons.STOP)
        self._voice_tile.set_state(True)
        self._input.border_color = COLOR_PRIMARY
        self._input.focused_border_color = COLOR_PRIMARY
        self._show_dictation()

    def _show_dictation(self) -> None:
        self._input.value = self._dictation.text()
        self._input_edits += 1
        update_control_if_mounted(self._input)

    def _stop_dictation(self, *, send: bool) -> None:
        state = self._dictation
        if not state.active:
            return
        state.active = False
        final_text = state.text()
        if state.unsubscribe is not None:
            state.unsubscribe()
        if state.restore is not None:
            state.restore()
        timer = state.timer
        if timer is not None and timer is not asyncio.current_task() and not timer.done():
            timer.cancel()
        self._input.value = final_text
        self._input_edits += 1
        self._voice_tile.set_label("VOICE")
        self._voice_tile.set_icon(ft.Icons.MIC)
        self._voice_tile.set_state(False)
        self._input.border_color = COLOR_DIVIDER
        self._input.focused_border_color = COLOR_PRIMARY
        update_control_if_mounted(self)
        if send and final_text.strip():
            self._run(self._submit)


def _run(page: object | None, coroutine_factory: Callable[[], object]) -> object | None:
    runner = getattr(page, "run_task", None)
    return runner(coroutine_factory) if callable(runner) else None


def _texts(control: ft.Control) -> list[ft.Text]:
    if isinstance(control, ft.Text):
        return [control]
    found: list[ft.Text] = []
    for child in _children(control):
        found.extend(_texts(child))
    return found


def _icons(control: ft.Control) -> list[ft.Icon]:
    if isinstance(control, ft.Icon):
        return [control]
    found: list[ft.Icon] = []
    for child in _children(control):
        found.extend(_icons(child))
    return found


def _children(control: ft.Control) -> list[ft.Control]:
    children: list[ft.Control] = []
    content = getattr(control, "content", None)
    if isinstance(content, ft.Control):
        children.append(content)
    controls = getattr(control, "controls", None)
    if isinstance(controls, list):
        children.extend(item for item in controls if isinstance(item, ft.Control))
    return children
