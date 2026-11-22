from __future__ import annotations


def build_translation_user_message(
    *, text: str, context: str, scene_participant_count: int | None = None
) -> str:
    scene = ""
    if (
        isinstance(scene_participant_count, int)
        and not isinstance(scene_participant_count, bool)
        and scene_participant_count >= 1
    ):
        scene = f"<scene>\nPeople: {scene_participant_count}\n</scene>\n\n"
    if context:
        return f"<context>\n{context}\n</context>\n\n{scene}<input>\n{text}\n</input>"
    return f"{scene}<input>\n{text}\n</input>"
