from __future__ import annotations

import pytest

from puripuly_heart.core.overlay.peer_caption_layout import (
    PEER_LINE_WIDTH_EM,
    estimate_peer_translation_lines,
    peer_page_reading_budget,
    peer_translation_pages,
)


def test_pages_keep_complete_text_in_two_line_chunks() -> None:
    text = "가" * 145
    pages = peer_translation_pages(text)
    assert len(pages) == 3
    assert all(len(page.splitlines()) <= 2 for page in pages)
    assert "".join(pages).replace("\n", "") == text
    assert pages[0] == "가" * 28 + "\n" + "가" * 28


def test_pages_preserve_words_and_explicit_line_breaks() -> None:
    assert peer_translation_pages("first\nsecond\nthird\nfourth\nfifth") == (
        "first\nsecond",
        "third\nfourth",
        "fifth",
    )
    text = "A repeated word with punctuation, " * 10
    pages = peer_translation_pages(text)
    assert " ".join(" ".join(page.split()) for page in pages) == " ".join(text.split())
    assert peer_translation_pages("") == ()


def test_join_estimate_reuses_harness_geometry_and_expansion_without_rounding() -> None:
    assert PEER_LINE_WIDTH_EM == pytest.approx((4096 - 2 * (48 + 24 + 5)) / 132 * 0.95)
    assert estimate_peer_translation_lines("가" * 30, "ko-KR", "en-US") == pytest.approx(
        30 / PEER_LINE_WIDTH_EM * 1.617
    )
    assert estimate_peer_translation_lines("a" * 100, "en", "ko") == pytest.approx(
        62 / PEER_LINE_WIDTH_EM * 0.856
    )
    assert estimate_peer_translation_lines("日" * 30, "ja", "en") == pytest.approx(
        30 / PEER_LINE_WIDTH_EM * 1.617
    )
    assert estimate_peer_translation_lines("same", "en", "en") < 1


def test_reading_budget_uses_wide_narrow_rates_and_bounds() -> None:
    assert peer_page_reading_budget("") == 1.0
    assert peer_page_reading_budget("가" * 9 + "a" * 17) == pytest.approx(2.3)
    assert peer_page_reading_budget("a" * 1000) == 3.0


def test_larger_calibration_keeps_two_line_pages_with_complete_coverage() -> None:
    text = "가" * 100
    pages = peer_translation_pages(text, text_scale=2.0)
    assert pages[0] == "가" * 14 + "\n" + "가" * 14
    assert all(len(line) <= 14 for page in pages for line in page.splitlines())
    assert "".join(pages).replace("\n", "") == text


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2, 2.0])
def test_join_estimate_uses_same_effective_scale_as_page_width(scale: float) -> None:
    source = "가" * 50
    estimate = estimate_peer_translation_lines(source, "ko", "en", text_scale=scale)
    assert estimate == pytest.approx(50 / PEER_LINE_WIDTH_EM * 1.617 * max(1.0, scale))
    assert peer_translation_pages(source, text_scale=scale)[0].splitlines()[0] == (
        "가" * int(PEER_LINE_WIDTH_EM / max(1.0, scale))
    )
