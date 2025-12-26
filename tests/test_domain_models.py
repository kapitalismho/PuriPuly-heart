from __future__ import annotations

import uuid

import pytest

from ajebal_daera_translator.domain.events import STTFinalEvent, STTPartialEvent
from ajebal_daera_translator.domain.models import Transcript, Translation, UtteranceBundle


def test_utterance_bundle_merges_out_of_order():
    utterance_id = uuid.uuid4()
    bundle = UtteranceBundle(utterance_id=utterance_id)

    bundle.with_translation(Translation(utterance_id=utterance_id, text="hello"))
    bundle.with_transcript(Transcript(utterance_id=utterance_id, text="안녕", is_final=True))

    assert bundle.translation is not None
    assert bundle.final is not None
    assert bundle.partial is None


def test_partial_ignored_after_final():
    utterance_id = uuid.uuid4()
    bundle = UtteranceBundle(utterance_id=utterance_id)

    bundle.with_transcript(Transcript(utterance_id=utterance_id, text="hi", is_final=True))
    bundle.with_transcript(Transcript(utterance_id=utterance_id, text="h", is_final=False))

    assert bundle.final is not None
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
