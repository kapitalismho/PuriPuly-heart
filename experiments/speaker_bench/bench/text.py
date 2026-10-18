from __future__ import annotations

import re
import unicodedata

CHARACTER_LANGUAGES = frozenset({"ko", "ja", "zh", "zh-CN", "zh-TW", "yue"})
_WORD = re.compile(r"[\w']+", re.UNICODE)


def character_language(language: str | None) -> bool:
    return (language or "").split("-")[0] in {code.split("-")[0] for code in CHARACTER_LANGUAGES}


def tokens(text: str, language: str | None) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    if character_language(language):
        return [
            ch
            for ch in normalized
            if not ch.isspace() and unicodedata.category(ch)[0] not in {"P", "S", "Z", "C"}
        ]
    out = []
    for match in _WORD.finditer(normalized):
        word = match.group(0).strip("'")
        if word:
            out.append(word)
    return out
