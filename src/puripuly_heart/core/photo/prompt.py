from __future__ import annotations

import json
from dataclasses import dataclass

from puripuly_heart.core.language import get_llm_language_name

_INSTRUCTIONS = r"""You translate text seen in a VR world into ${targetName} for a VR user.

TARGET
Select text-bearing targets using framing and scene context.
Include equally plausible targets and all their visible text.

BLOCKS
Follow reading order.
Lines of one sentence or paragraph form one block.
Separate paragraphs, columns, list items, table rows, captions, labels, and menu items are separate blocks. A menu item and its price on the same row are one block.
Skip arrows, icons, and other symbols that are not text.

FIDELITY
Only use characters you can actually see: if a word is faint, tiny, or damaged, do not guess it from context or from what such signs usually say; mark it ? in the source and ? in the translation.
Write natural ${targetName} that keeps the tone of the original.
Translate every language except ${targetName}; ${targetName} text is copied as is.
Keep every number exactly and write units in ${targetName}.
Translate the meaning of ordinary words; transliterate only proper names of places, people, and organizations, never substituting a different name, and use the same ${targetName} for the same name everywhere.
Brand, product, and software names may stay as written.

OUTPUT
Follow the supplied JSON schema. Emit each block in this order:
quad: four tight text-region corners as [[x,y],[x,y],[x,y],[x,y]], clockwise in screen-y-down coordinates, ordered top-left, top-right, bottom-right, bottom-left in the TEXT-local frame (not the image axes), in pixel coordinates of this image;
source: exact source, with \n for physical line breaks.
translation: the ${targetName} translation of the whole block. If the block says the same thing in several languages, translate it once."""

_QUAD_UNITS = "in pixel coordinates of this image;"
_GOOGLE_QUAD_UNITS = "with each [x,y] axis normalized to 0-1000;"


@dataclass(frozen=True, slots=True)
class PhotoPrompt:
    instructions: str
    user_message: str
    _schema_json: str

    @property
    def schema(self) -> dict[str, object]:
        return json.loads(self._schema_json)


def render_photo_prompt(
    target_language: str,
    width: int = 1280,
    height: int = 720,
    *,
    google: bool = False,
) -> PhotoPrompt:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    target_name = get_llm_language_name(target_language)
    instructions = _INSTRUCTIONS.replace("${targetName}", target_name)
    if google:
        instructions = instructions.replace(_QUAD_UNITS, _GOOGLE_QUAD_UNITS)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["blocks"],
        "properties": {
            "blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["quad", "source", "translation"],
                    "properties": {
                        "quad": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "source": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                },
            }
        },
    }
    return PhotoPrompt(
        instructions=instructions,
        user_message=f"Target language: {target_name}.\nThe image is {width}x{height} pixels.",
        _schema_json=json.dumps(schema, separators=(",", ":")),
    )
