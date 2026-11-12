from __future__ import annotations

import pytest

from puripuly_heart.core.overlay.peer_caption_layout import (
    PEER_LINE_WIDTH_EM,
    estimate_peer_translation_lines,
    peer_fragment_reading_budget,
    peer_translation_fragments,
)


def _assert_capacity_and_preservation(
    text: str, fragments: tuple[str, ...], scale: float = 1.0
) -> None:
    assert "".join("".join(fragment.split()) for fragment in fragments) == "".join(text.split())
    for fragment in fragments:
        assert len(fragment.splitlines()) <= 2
        for line in fragment.splitlines():
            assert estimate_peer_translation_lines(line, "ko", "ko", text_scale=scale) <= 1.000001


def test_fragments_balance_complete_text_without_adding_pieces() -> None:
    text = "가" * 145
    fragments = peer_translation_fragments(text)
    assert len(fragments) == 3
    _assert_capacity_and_preservation(text, fragments)
    assert (
        max(len(fragment.replace("\n", "")) for fragment in fragments)
        - min(len(fragment.replace("\n", "")) for fragment in fragments)
        <= 1
    )


def test_fragments_preserve_words_and_explicit_line_breaks() -> None:
    assert peer_translation_fragments("first\nsecond") == ("first\nsecond",)
    text = "first\nsecond\nthird\nfourth\nfifth"
    fragments = peer_translation_fragments(text)
    assert len(fragments) == 3
    _assert_capacity_and_preservation(text, fragments)
    text = "A repeated word with punctuation, " * 10
    fragments = peer_translation_fragments(text)
    assert " ".join(" ".join(fragment.split()) for fragment in fragments) == " ".join(text.split())
    assert peer_translation_fragments("") == ()


def test_fitting_translation_keeps_existing_greedy_wrap() -> None:
    assert peer_translation_fragments("가" * 50) == ("가" * 28 + "\n" + "가" * 22,)
    assert peer_translation_fragments("We can meet at the station.") == (
        "We can meet at the station.",
    )


def test_fragment_count_comes_from_valid_word_wrap_not_total_width() -> None:
    text = " ".join(["a" * 30] * 3)
    assert estimate_peer_translation_lines(text, "en", "en") < 2
    fragments = peer_translation_fragments(text)
    assert len(fragments) == 2
    _assert_capacity_and_preservation(text, fragments)


def test_attractive_sentence_boundary_cannot_overflow_remaining_fragment() -> None:
    text = "가" * 43 + "。" + "나" * 65
    fragments = peer_translation_fragments(text)
    assert len(fragments) == 2
    _assert_capacity_and_preservation(text, fragments)
    assert "나" in fragments[0]


@pytest.mark.parametrize("length, count", [(64, 2), (128, 3), (24, 1)])
def test_full_revision_partitions_statelessly(length: int, count: int) -> None:
    text = "가" * length
    fragments = peer_translation_fragments(text)
    assert len(fragments) == count
    _assert_capacity_and_preservation(text, fragments)


@pytest.mark.parametrize(
    "text, following",
    [
        (
            "오늘은 역 앞에서 만나서 공원까지 함께 걸어가기로 했어요. "
            "비가 오면 산책은 취소하고 근처 도서관에서 책을 읽다가 점심을 먹으려고 해요.",
            "비가 오면",
        ),
        (
            "We can meet at the station and walk to the park together. "
            "If it starts raining, we will go to the library instead and have lunch nearby.",
            "If it starts",
        ),
    ],
)
def test_sentence_boundary_keeps_following_condition_together(text: str, following: str) -> None:
    fragments = peer_translation_fragments(text)
    assert len(fragments) == 2
    _assert_capacity_and_preservation(text, fragments)
    assert fragments[0].rstrip().endswith(".")
    assert fragments[1].startswith(following)


@pytest.mark.parametrize(
    "text",
    [
        "At the next meeting we will discuss the new project with Dr. Kim and review the revised "
        "delivery schedule before assigning the remaining tasks.",
        "오늘 meeting starts at 10:30 and we will review the deployment plan together. "
        "이후에는 테스트 결과와 다음 작업 일정을 확인합니다.",
        'We can meet at the station and walk to the park together.") '
        "If it starts raining, we will go to the library instead and have lunch nearby.",
        "Review version 2.5 at 10:30 via https://example.test/path before the meeting, "
        "then discuss the remaining deployment tasks and confirm the delivery schedule.",
    ],
)
def test_lightweight_punctuation_preserves_capacity_and_closing_groups(text: str) -> None:
    fragments = peer_translation_fragments(text)
    _assert_capacity_and_preservation(text, fragments)
    assert all(not fragment.startswith(('"', ")")) for fragment in fragments[1:])


def test_explicit_source_newline_wins_over_nearer_whitespace() -> None:
    text = "가" * 44 + "\n" + "나" * 52
    fragments = peer_translation_fragments(text)
    assert len(fragments) == 2
    _assert_capacity_and_preservation(text, fragments)
    assert fragments[0].replace("\n", "") == "가" * 44
    assert fragments[1].replace("\n", "") == "나" * 52


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
    assert peer_fragment_reading_budget("") == 1.0
    assert peer_fragment_reading_budget("가" * 9 + "a" * 17) == pytest.approx(2.3)
    assert peer_fragment_reading_budget("a" * 1000) == 3.0


def test_larger_calibration_keeps_two_line_fragments_with_complete_coverage() -> None:
    text = "가" * 100
    fragments = peer_translation_fragments(text, text_scale=2.0)
    assert len(fragments) == 4
    _assert_capacity_and_preservation(text, fragments, 2.0)


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2, 2.0])
def test_join_estimate_uses_same_effective_scale_as_fragment_width(scale: float) -> None:
    source = "가" * 50
    estimate = estimate_peer_translation_lines(source, "ko", "en", text_scale=scale)
    assert estimate == pytest.approx(50 / PEER_LINE_WIDTH_EM * 1.617 * max(1.0, scale))
    fragments = peer_translation_fragments(source, text_scale=scale)
    _assert_capacity_and_preservation(source, fragments, scale)
