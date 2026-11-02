from __future__ import annotations

import unicodedata

PEER_PAGE_INTERVAL_SECONDS = 3.0
PEER_LINE_WIDTH_EM = (4096 - 2 * (48 + 24 + 5)) / 132 * 0.95
_SOURCE_TARGET_LENGTH_RATIOS = {("en", "ko"): 0.856, ("ko", "en"): 1.617}


def _caption_char_width_em(char: str) -> float:
    codepoint = ord(char)
    if char.isspace():
        return 0.32
    if 0x1F000 <= codepoint <= 0x1FAFF:
        return 1.15
    if (
        0x1100 <= codepoint <= 0x11FF
        or 0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xF900 <= codepoint <= 0xFAFF
    ):
        return 1.0
    if char in ".,;:!?'\"-–—()[]{}·…":
        return 0.38
    if char in "ilI|":
        return 0.42
    return 0.62 if char.isascii() else 1.0


def estimate_peer_translation_lines(
    source_text: str, source_language: str, target_language: str
) -> float:
    source = source_language.lower().replace("_", "-").split("-", 1)[0]
    target = target_language.lower().replace("_", "-").split("-", 1)[0]
    ratio = 1.0 if source == target else _SOURCE_TARGET_LENGTH_RATIOS.get((source, target), 1.617)
    return sum(_caption_char_width_em(char) for char in source_text) / PEER_LINE_WIDTH_EM * ratio


def peer_translation_pages(text: str, *, text_scale: float = 1.0) -> tuple[str, ...]:
    width_limit = PEER_LINE_WIDTH_EM / max(1.0, text_scale)
    lines: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = cursor
        width = 0.0
        last_space: int | None = None
        while end < len(text):
            char = text[end]
            if char == "\n":
                end += 1
                break
            next_width = width + _caption_char_width_em(char)
            if next_width > width_limit and end > cursor:
                if last_space is not None:
                    end = last_space
                break
            width = next_width
            end += 1
            if char.isspace():
                last_space = end
        lines.append(text[cursor:end].rstrip("\r\n"))
        cursor = end
    return tuple("\n".join(lines[index : index + 2]) for index in range(0, len(lines), 2))


def peer_page_reading_budget(text: str) -> float:
    wide = sum(unicodedata.east_asian_width(char) in {"W", "F"} for char in text)
    narrow = len(text) - wide
    return min(3.0, max(1.0, 0.3 + wide / 9 + narrow / 17))
