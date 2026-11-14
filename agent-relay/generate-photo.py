from __future__ import annotations

import json
from pathlib import Path

from puripuly_heart.core.language import SUPPORTED_LANGUAGES
from puripuly_heart.core.photo.prompt import _INSTRUCTIONS, render_photo_prompt

if __name__ == "__main__":
    contract = {
        "instructions": _INSTRUCTIONS,
        "schema": render_photo_prompt("en").schema,
        "languages": {code: info.name for code, info in SUPPORTED_LANGUAGES.items()},
    }
    Path(__file__).with_name("src").joinpath("photo-contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
