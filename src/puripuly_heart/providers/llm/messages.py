from __future__ import annotations


def build_translation_user_message(
    *, text: str, context: str, scene_participant_count: int | None = None
) -> str:
    input_block = f"<input>\n{text}\n</input>"
    if (
        isinstance(scene_participant_count, int)
        and not isinstance(scene_participant_count, bool)
        and scene_participant_count >= 1
    ):
        input_block = f"<scene>\nPeople: {scene_participant_count}\n</scene>\n\n{input_block}"
    if context:
        return f"<context>\n{context}\n</context>\n\n{input_block}"
    return input_block
