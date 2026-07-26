from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from puripuly_heart.ui.foundation.primitives import (
    FoundationActionButton,
    FoundationCard,
    FoundationSectionTitle,
    FoundationStatusPill,
)
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS
from puripuly_heart.ui.i18n import t_for_locale


@dataclass(frozen=True, slots=True)
class FoundationPreviewCopy:
    title: str
    body: str
    ready: str
    action: str
    unavailable: str


def foundation_preview_copy(locale: str) -> FoundationPreviewCopy:
    return FoundationPreviewCopy(
        title=t_for_locale(locale, "foundation.preview.title"),
        body=t_for_locale(locale, "foundation.preview.body"),
        ready=t_for_locale(locale, "foundation.preview.ready"),
        action=t_for_locale(locale, "foundation.preview.action"),
        unavailable=t_for_locale(locale, "foundation.preview.unavailable"),
    )


class FoundationPreviewSurface(ft.Column):
    def __init__(self, locale: str) -> None:
        copy = foundation_preview_copy(locale)
        self.copy = copy
        self.preview_card = FoundationCard(
            ft.Column(
                controls=[
                    FoundationSectionTitle(copy.title),
                    ft.Text(
                        copy.body,
                        size=FOUNDATION_DESIGN_TOKENS.typography.body,
                        color=FOUNDATION_DESIGN_TOKENS.palette.on_background,
                    ),
                    ft.Row(
                        controls=[
                            FoundationStatusPill(copy.ready),
                            FoundationActionButton(copy.action),
                            FoundationActionButton(copy.unavailable, disabled=True),
                        ],
                        spacing=FOUNDATION_DESIGN_TOKENS.spacing.compact,
                    ),
                ],
                spacing=FOUNDATION_DESIGN_TOKENS.spacing.inline,
                tight=True,
            ),
            width=560,
        )
        super().__init__(
            controls=[self.preview_card],
            spacing=0,
            tight=True,
        )


__all__ = [
    "FoundationPreviewCopy",
    "FoundationPreviewSurface",
    "foundation_preview_copy",
]
