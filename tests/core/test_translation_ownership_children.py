from __future__ import annotations

from uuid import uuid4, uuid5

import pytest

from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipUnit
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnChild,
    TranslationTurnLifecycleOwner,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.domain.models import FinalLanguageRun, Transcript


async def _noop(*_args, **_kwargs):
    return None


def _owner(process) -> TranslationTurnLifecycleOwner:
    return TranslationTurnLifecycleOwner(
        on_child_created=_noop,
        on_child_started=_noop,
        process_child=process,
        on_child_terminal=_noop,
        on_parent_closed=_noop,
        on_parent_rejected=_noop,
    )


def _request(parent_id, *, units=(), runs=(), targets=("ko",)):
    text = (
        "".join(unit.text for unit in units)
        if units
        else "".join(run.text for run in runs) or "hello"
    )
    return TranslationTurnRequest(
        transcript=Transcript(
            utterance_id=parent_id,
            text=text,
            is_final=True,
            channel="peer",
            final_language_runs=runs or (FinalLanguageRun(text, "en"),),
            publication_generation=1,
            source_order=1,
        ),
        source="Peer",
        turn_kind="peer",
        target_languages=targets,
        config_snapshot=TranslationRuntimeConfigSnapshot(
            revision=0, value=TranslationRuntimeConfig()
        ),
        ownership_units=units,
    )


@pytest.mark.asyncio
async def test_ownership_groups_are_ordered_children_of_one_parent() -> None:
    parent_id = uuid4()
    observed: list[TranslationTurnChild] = []

    async def process(child: TranslationTurnChild, _cancel):
        observed.append(child)
        return TranslationTurnProcessResult("source_only")

    units = (
        PretranslationOwnershipUnit(
            group_id="CURRENT-0",
            relation="CURRENT",
            text="Hello ",
            language_runs=(FinalLanguageRun("Hello ", "en"),),
            token_indexes=(0,),
        ),
        PretranslationOwnershipUnit(
            group_id="OTHER-1",
            relation="OTHER",
            text="Hello ",
            language_runs=(FinalLanguageRun("Hello ", "en"),),
            token_indexes=(1,),
        ),
        PretranslationOwnershipUnit(
            group_id="OTHER-2",
            relation="OTHER",
            text="there",
            language_runs=(FinalLanguageRun("there", "en"),),
            token_indexes=(2,),
        ),
    )
    owner = _owner(process)
    try:
        child_ids = await owner.submit(_request(parent_id, units=units))
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert len(child_ids) == 3
    assert parent_id not in child_ids
    assert len(set(child_ids)) == 3
    assert [child.ownership_group_id for child in observed] == [
        "CURRENT-0",
        "OTHER-1",
        "OTHER-2",
    ]
    assert observed[0].transcript.text == observed[1].transcript.text == "Hello "
    assert child_ids[0] == uuid5(parent_id, "peer:CURRENT-0:0:0:en:ko")


@pytest.mark.asyncio
async def test_disabled_path_keeps_existing_peer_child_identity() -> None:
    parent_id = uuid4()

    async def process(_child, _cancel):
        return TranslationTurnProcessResult("source_only")

    owner = _owner(process)
    try:
        child_ids = await owner.submit(
            _request(parent_id, runs=(FinalLanguageRun("hello", "en"),))
        )
        await owner.wait_for_idle()
    finally:
        await owner.close()
    assert child_ids == (uuid5(parent_id, "peer:0:0:en:ko"),)
