from __future__ import annotations

import unicodedata

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
    source_text: str, source_language: str, target_language: str, *, text_scale: float = 1.0
) -> float:
    source = source_language.lower().replace("_", "-").split("-", 1)[0]
    target = target_language.lower().replace("_", "-").split("-", 1)[0]
    ratio = 1.0 if source == target else _SOURCE_TARGET_LENGTH_RATIOS.get((source, target), 1.617)
    width_limit = PEER_LINE_WIDTH_EM / max(1.0, text_scale)
    return sum(_caption_char_width_em(char) for char in source_text) / width_limit * ratio


def _wrap_bounds(text: str, width_limit: float) -> tuple[list[float], list[int], list[int]]:
    size = len(text)
    widths = [0.0]
    spaces = [0]
    newlines = [size] * (size + 1)
    for index, char in enumerate(text):
        widths.append(widths[-1] + (0.0 if char == "\n" else _caption_char_width_em(char)))
        spaces.append(index + 1 if char.isspace() else spaces[-1])
    for index in range(size - 1, -1, -1):
        newlines[index] = index if text[index] == "\n" else newlines[index + 1]
    raw_ends = [size] * (size + 1)
    line_ends = [size] * (size + 1)
    end = 0
    for start in range(size):
        end = max(start, end)
        while end < size and widths[end + 1] - widths[start] <= width_limit:
            end += 1
        raw_end = min(max(start + 1, end), newlines[start] + 1, size)
        raw_ends[start] = raw_end
        if raw_end == size or raw_end == newlines[start] + 1:
            line_ends[start] = raw_end
        else:
            line_ends[start] = spaces[raw_end] if spaces[raw_end] > start else raw_end
    return widths, raw_ends, line_ends


def _boundary_ranks(text: str) -> list[int]:
    closing = "\"')]}”’」』】）》〉"
    ranks = [0]
    for index, char in enumerate(text):
        following = text[index + 1 : index + 2]
        if char == "\n":
            rank = 3
        elif char.isspace():
            rank = max(1, ranks[-1])
        elif char in closing:
            rank = ranks[-1]
        elif char in ".!?。！？…;:,，、；：":
            external = not following or following.isspace() or following in closing
            if external or char in "。！？":
                rank = 3 if char in ".!?。！？…" else 2
            else:
                rank = 0
        else:
            rank = 0
        ranks.append(rank)
    return ranks


def peer_translation_fragments(text: str, *, text_scale: float = 1.0) -> tuple[str, ...]:
    if not text:
        return ()
    width_limit = PEER_LINE_WIDTH_EM / max(1.0, text_scale)
    widths, raw_ends, line_ends = _wrap_bounds(text, width_limit)
    size = len(text)
    line_counts = [0] * (size + 1)
    for start in range(size - 1, -1, -1):
        line_counts[start] = 1 + line_counts[line_ends[start]]
    remaining = (line_counts[0] + 1) // 2
    ranks = _boundary_ranks(text) if remaining > 1 else []
    fragments: list[str] = []
    start = 0
    while remaining:
        cut = size
        if remaining > 1:
            target = (widths[size] - widths[start]) / remaining
            limit = min(raw_ends[line_ends[start]], size - remaining + 1)
            feasible = [
                end
                for end in range(start + 1, limit + 1)
                if line_counts[end] <= 2 * (remaining - 1)
            ]
            attached = [
                end
                for end in feasible
                if text[end : end + 1] not in ".,;:!?…。。，、；：！？\"')]}”’」』】）》〉"
            ]
            candidates = attached or feasible
            preferred = [
                end
                for end in candidates
                if 0.8 * target <= widths[end] - widths[start] <= 1.2 * target and ranks[end]
            ]
            if preferred:
                cut = min(
                    preferred,
                    key=lambda end: (-ranks[end], abs(widths[end] - widths[start] - target)),
                )
            else:
                cut = min(candidates, key=lambda end: abs(widths[end] - widths[start] - target))
        lines: list[str] = []
        cursor = start
        while cursor < cut:
            end = cut if cut <= raw_ends[cursor] else line_ends[cursor]
            lines.append(text[cursor:end].rstrip("\r\n"))
            cursor = end
        fragments.append("\n".join(lines))
        start = cut
        remaining -= 1
    return tuple(fragments)


def peer_fragment_reading_budget(text: str) -> float:
    wide = sum(unicodedata.east_asian_width(char) in {"W", "F"} for char in text)
    narrow = len(text) - wide
    return min(3.0, max(1.0, 0.3 + wide / 9 + narrow / 17))
