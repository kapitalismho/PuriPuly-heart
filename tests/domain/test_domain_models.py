from __future__ import annotations

import uuid

import pytest

from puripuly_heart.domain.events import STTFinalEvent, STTPartialEvent
from puripuly_heart.domain.models import Transcript, Translation, UtteranceBundle


@pytest.mark.parametrize("translation_first", [True, False])
def test_utterance_bundle_preserves_content_in_either_arrival_order(translation_first):
    utterance_id = uuid.uuid4()
    bundle = UtteranceBundle(utterance_id=utterance_id)

    transcript = Transcript(utterance_id=utterance_id, text="안녕", is_final=True)
    translation = Translation(utterance_id=utterance_id, text="hello")
    updates = (
        (lambda: bundle.with_translation(translation), lambda: bundle.with_transcript(transcript))
        if translation_first
        else (
            lambda: bundle.with_transcript(transcript),
            lambda: bundle.with_translation(translation),
        )
    )
    for update in updates:
        update()

    assert bundle.translation is not None
    assert bundle.final is not None
    assert bundle.utterance_id == utterance_id
    assert (bundle.final.utterance_id, bundle.final.text) == (utterance_id, "안녕")
    assert (bundle.translation.utterance_id, bundle.translation.text) == (utterance_id, "hello")
    assert bundle.partial is None


def test_partial_ignored_after_final():
    utterance_id = uuid.uuid4()
    bundle = UtteranceBundle(utterance_id=utterance_id)

    bundle.with_transcript(Transcript(utterance_id=utterance_id, text="hi", is_final=True))
    bundle.with_transcript(Transcript(utterance_id=utterance_id, text="h", is_final=False))

    assert bundle.final is not None
    assert bundle.final.text == "hi"
    assert bundle.partial is None


def test_event_type_validation():
    utterance_id = uuid.uuid4()

    partial = Transcript(utterance_id=utterance_id, text="p", is_final=False)
    final = Transcript(utterance_id=utterance_id, text="f", is_final=True)

    STTPartialEvent(utterance_id, partial)
    STTFinalEvent(utterance_id, final)

    with pytest.raises(ValueError):
        STTPartialEvent(utterance_id, final)
    with pytest.raises(ValueError):
        STTFinalEvent(utterance_id, partial)
