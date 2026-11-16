from __future__ import annotations

import json
from itertools import permutations

import pytest

from puripuly_heart.core.photo import (
    BboxReady,
    BlockComplete,
    PhotoBlock,
    PhotoBlockParser,
    PhotoParseError,
    SourceReady,
    render_photo_prompt,
)


def _response(*blocks: dict[str, object]) -> str:
    return json.dumps({"blocks": list(blocks)}, ensure_ascii=False)


def _block(
    source: str = "Open\n24 hours",
    translation: str = "24시간 영업",
    *,
    google: bool = False,
) -> dict[str, object]:
    return {
        "box_2d" if google else "bbox": [10, 20, 100, 120],
        "source": source,
        "translation": translation,
    }


def test_stages_arrive_at_validated_boundaries_before_final_close() -> None:
    parser = PhotoBlockParser()
    assert parser.feed('{"blocks":[{"bbox":[10,20,100,120') == []
    assert parser.feed("]") == [BboxReady(0, (10, 20, 100, 120))]
    assert parser.feed(',"source":"Open\\n24') == []
    assert parser.feed(' hours"') == [SourceReady(0, (10, 20, 100, 120), "Open\n24 hours", 2)]
    assert parser.feed(',"translation":"24시간 영업"') == []
    assert parser.feed("}") == [
        BlockComplete(0, (10, 20, 100, 120), "Open\n24 hours", "24시간 영업")
    ]
    assert parser.feed("]}") == []
    assert parser.finish() == [PhotoBlock((10, 20, 100, 120), "Open\n24 hours", "24시간 영업")]


def test_multiple_blocks_accumulate_in_one_delta_without_repetition() -> None:
    parser = PhotoBlockParser()
    events = parser.feed(_response(_block(), _block("Exit", "출구")))
    assert [event.kind for event in events] == [
        "bbox_ready",
        "source_ready",
        "block_complete",
        "bbox_ready",
        "source_ready",
        "block_complete",
    ]
    assert [event.index for event in events] == [0, 0, 0, 1, 1, 1]
    assert parser.feed("") == []
    assert parser.feed("\n") == []
    blocks = parser.finish()
    assert [block.source for block in blocks] == ["Open\n24 hours", "Exit"]


def test_every_two_chunk_split_preserves_escaped_text_and_stage_order() -> None:
    source = 'quoted "text" \\ {braces} [array]\nsecond\n'
    translation = '번역 "문자" \\ } ]'
    text = _response(_block(source, translation))
    expected_events = [
        BboxReady(0, (10, 20, 100, 120)),
        SourceReady(0, (10, 20, 100, 120), source, 3),
        BlockComplete(0, (10, 20, 100, 120), source, translation),
    ]
    for split in range(len(text) + 1):
        parser = PhotoBlockParser()
        events = parser.feed(text[:split]) + parser.feed(text[split:])
        assert events == expected_events
        assert parser.finish() == [PhotoBlock((10, 20, 100, 120), source, translation)]


def test_single_character_chunks_decode_unicode_escapes_and_escaped_keys() -> None:
    text = (
        '{"bl\\u006fcks":[{"bb\\u006fx":[0,0,1280,720],'
        '"source":"\\u0041\\n\\uD55C\\uAE00\\\\\\"{}",'
        '"translation":"\\ud83d\\ude00"}]}'
    )
    parser = PhotoBlockParser()
    events = [event for char in text for event in parser.feed(char)]
    assert events == [
        BboxReady(0, (0, 0, 1280, 720)),
        SourceReady(0, (0, 0, 1280, 720), 'A\n한글\\"{}', 2),
        BlockComplete(0, (0, 0, 1280, 720), 'A\n한글\\"{}', "😀"),
    ]
    assert parser.finish()[0].translation == "😀"


@pytest.mark.parametrize("source, count", [("", 1), ("a", 1), ("a\nb", 2), ("a\r\nb\n", 3)])
def test_source_line_count_only_uses_complete_decoded_physical_lines(
    source: str, count: int
) -> None:
    parser = PhotoBlockParser()
    events = parser.feed(_response(_block(source)))
    assert events[1] == SourceReady(0, (10, 20, 100, 120), source, count)
    parser.finish()


@pytest.mark.parametrize("google", [False, True])
def test_tolerates_fence_preamble_and_uses_first_complete_object(google: bool) -> None:
    parser = PhotoBlockParser(google=google)
    prefix = "Here is the result:\n```json\n"
    text = prefix + _response(_block(google=google)) + '\n```\n{"ignored":"suffix"}'
    events = [event for char in text for event in parser.feed(char)]
    assert len(events) == 3
    assert events[0].bbox == (10, 20, 100, 120)
    assert len(parser.finish()) == 1


@pytest.mark.parametrize("google", [False, True])
def test_empty_blocks_is_a_valid_completed_result(google: bool) -> None:
    parser = PhotoBlockParser(google=google)
    assert parser.feed('{"blocks": []}') == []
    assert parser.finish() == []


def test_google_preserves_provider_yx_coordinates_without_mapping() -> None:
    parser = PhotoBlockParser(google=True)
    block = {"box_2d": [0, 50, 1000, 900], "source": "One", "translation": "하나"}
    events = parser.feed(_response(block))
    assert events[0] == BboxReady(0, (0, 50, 1000, 900))
    assert parser.finish() == [PhotoBlock((0, 50, 1000, 900), "One", "하나")]


@pytest.mark.parametrize("google", [False, True])
@pytest.mark.parametrize(
    "coordinates",
    [
        [],
        [0, 0, 1],
        [0, 0, 1, 1, 2],
        [True, 0, 1, 1],
        [0, 0, 1.0, 1],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [0, 0, "1", 1],
        [0, 0, None, 1],
    ],
)
def test_invalid_coordinate_structure_never_emits_a_bbox_stage(
    google: bool, coordinates: object
) -> None:
    parser = PhotoBlockParser(google=google)
    block = _block(google=google)
    block["box_2d" if google else "bbox"] = coordinates
    assert parser.feed(_response(block)) == []
    with pytest.raises(PhotoParseError):
        parser.finish()


@pytest.mark.parametrize("coordinates", ["[00,0,1,1]", "[+0,0,1,1]", "[0,0,1e2,1]", "[0,0,1,1,]"])
def test_non_json_or_non_integer_coordinates_are_rejected(coordinates: str) -> None:
    parser = PhotoBlockParser()
    assert (
        parser.feed('{"blocks":[{"bbox":' + coordinates + ',"source":"a","translation":"b"}]}')
        == []
    )
    with pytest.raises(PhotoParseError):
        parser.finish()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "```json\n",
        "{}",
        "[]",
        '[{"blocks":[]}]',
        '{"other":[]}',
        '{"blocks":{}}',
        '{"blocks":[null]}',
        '{"blocks":[[]]}',
        '{"blocks":[],"other":0}',
        '{"blocks":[],"blocks":[]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a"}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":"b","extra":0}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"bbox":[0,0,1,1],"source":"a","translation":"b"}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":null,"translation":"b"}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":1,"translation":"b"}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":false}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":{}}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"bad\\q","translation":"b"}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"bad\nline","translation":"b"}]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":"b"},]}',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":"b"},',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":"unterminated',
        '{"blocks":[{"bbox":[0,0,1,1],"source":"a","translation":"b"}]',
    ],
)
def test_malformed_or_truncated_response_never_succeeds_at_finish(text: str) -> None:
    parser = PhotoBlockParser()
    parser.feed(text)
    with pytest.raises(PhotoParseError):
        parser.finish()


@pytest.mark.parametrize("google", [False, True])
def test_wrong_provider_coordinate_field_is_rejected(google: bool) -> None:
    parser = PhotoBlockParser(google=google)
    assert parser.feed(_response(_block(google=not google))) == []
    with pytest.raises(PhotoParseError):
        parser.finish()


def test_valid_blocks_are_delivered_even_when_same_delta_later_becomes_invalid() -> None:
    parser = PhotoBlockParser()
    text = _response(_block())[:-2] + ',{"bbox":[0,0,false,1]}]}'
    events = parser.feed(text)
    assert len(events) == 3
    assert isinstance(events[-1], BlockComplete)
    assert parser.feed(_response(_block())) == []
    with pytest.raises(PhotoParseError):
        parser.finish()


def test_invalid_source_never_emits_source_or_completion_stages() -> None:
    parser = PhotoBlockParser()
    events = parser.feed('{"blocks":[{"bbox":[0,0,1,1],"source":"bad\\u12xz","translation":"b"}]}')
    assert events == [BboxReady(0, (0, 0, 1, 1))]
    with pytest.raises(PhotoParseError):
        parser.finish()


@pytest.mark.parametrize("google", [False, True])
@pytest.mark.parametrize("order", list(permutations(("coordinates", "source", "translation"))))
def test_complete_blocks_accept_schema_fields_in_any_order(
    google: bool, order: tuple[str, str, str]
) -> None:
    parser = PhotoBlockParser(google=google)
    coordinate_field = "box_2d" if google else "bbox"
    fields = {"coordinates": [0, 0, 100, 200], "source": "a\nb", "translation": "번역"}
    block = {coordinate_field if name == "coordinates" else name: fields[name] for name in order}
    events = parser.feed(_response(block))
    assert events == [
        BboxReady(0, (0, 0, 100, 200)),
        SourceReady(0, (0, 0, 100, 200), "a\nb", 2),
        BlockComplete(0, (0, 0, 100, 200), "a\nb", "번역"),
    ]
    assert parser.finish() == [PhotoBlock((0, 0, 100, 200), "a\nb", "번역")]


def test_reordered_source_waits_for_bbox_before_emitting_geometry_stages() -> None:
    parser = PhotoBlockParser()
    assert parser.feed('{"blocks":[{"source":"a\\nb"') == []
    assert parser.feed(',"bbox":[0,0,100,200') == []
    assert parser.feed("]") == [
        BboxReady(0, (0, 0, 100, 200)),
        SourceReady(0, (0, 0, 100, 200), "a\nb", 2),
    ]
    assert parser.feed(',"translation":"번역"') == []
    assert parser.feed("}]}") == [BlockComplete(0, (0, 0, 100, 200), "a\nb", "번역")]
    parser.finish()


def test_feed_after_successful_finish_is_explicitly_rejected() -> None:
    parser = PhotoBlockParser()
    parser.feed('{"blocks":[]}')
    parser.finish()
    with pytest.raises(PhotoParseError):
        parser.feed(" ")


@pytest.mark.parametrize(
    "language, name",
    [
        ("ko", "Korean"),
        ("ko-KR", "Korean"),
        ("ja", "Japanese"),
        ("zh-TW", "Chinese (Traditional)"),
        ("xx", "English"),
    ],
)
def test_prompt_renders_target_consistently_and_has_ordered_strict_schema(
    language: str, name: str
) -> None:
    prompt = render_photo_prompt(language, 1920, 1080)
    assert name in prompt.instructions
    assert name in prompt.user_message
    assert "${" not in prompt.instructions
    if name != "Korean":
        assert "Korean" not in prompt.instructions
        assert "Korean" not in prompt.user_message
    assert "1920x1080" in prompt.user_message
    schema = prompt.schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["blocks"]
    assert list(schema["properties"]) == ["blocks"]
    block_schema = schema["properties"]["blocks"]["items"]
    assert block_schema["additionalProperties"] is False
    assert block_schema["required"] == ["bbox", "source", "translation"]
    assert list(block_schema["properties"]) == ["bbox", "source", "translation"]
    assert block_schema["properties"]["bbox"] == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 4,
        "maxItems": 4,
    }
    assert block_schema["properties"]["source"] == {"type": "string"}
    assert block_schema["properties"]["translation"] == {"type": "string"}


def test_google_adapter_requests_provider_coordinate_field_in_strict_schema() -> None:
    google = render_photo_prompt("ko", google=True)
    schema = google.schema
    block_schema = schema["properties"]["blocks"]["items"]
    assert block_schema["required"] == ["box_2d", "source", "translation"]
    assert list(block_schema["properties"]) == ["box_2d", "source", "translation"]


@pytest.mark.parametrize("width, height", [(0, 720), (1280, -1), (True, 720), (1280, 720.5)])
def test_invalid_image_dimensions_are_explicitly_rejected(width: object, height: object) -> None:
    with pytest.raises(ValueError):
        render_photo_prompt("ko", width, height)
