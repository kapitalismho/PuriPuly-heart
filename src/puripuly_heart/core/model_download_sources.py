from __future__ import annotations

from typing import Literal

ModelDownloadSource = Literal["huggingface", "modelscope", "hf_mirror"]
MODEL_DOWNLOAD_SOURCES: tuple[ModelDownloadSource, ...] = ("huggingface", "modelscope", "hf_mirror")
HF_MIRROR_ENDPOINT = "https://hf-mirror.net"
_CHINESE_LOCALES = frozenset({"zh-cn", "zh-hk", "zh-hant-hk"})


def model_download_source_order(locale: str | None) -> tuple[ModelDownloadSource, ...]:
    normalized = (locale or "").strip().replace("_", "-").lower()
    if normalized in _CHINESE_LOCALES:
        return ("modelscope", "hf_mirror", "huggingface")
    return ("huggingface", "modelscope", "hf_mirror")


def model_download_source_candidates(
    registered: tuple[str, ...] | list[str],
    *,
    locale: str | None,
    explicit: str | None = None,
) -> tuple[ModelDownloadSource, ...]:
    available = set(registered)
    if "huggingface" in available:
        available.add("hf_mirror")
    if explicit is not None:
        if explicit not in MODEL_DOWNLOAD_SOURCES or explicit not in available:
            return ()
        return (explicit,)
    return tuple(name for name in model_download_source_order(locale) if name in available)


__all__ = [
    "HF_MIRROR_ENDPOINT",
    "MODEL_DOWNLOAD_SOURCES",
    "ModelDownloadSource",
    "model_download_source_candidates",
    "model_download_source_order",
]
