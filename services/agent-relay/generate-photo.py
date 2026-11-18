from __future__ import annotations

import argparse
import json
from pathlib import Path

from puripuly_heart.core.language import SUPPORTED_LANGUAGES
from puripuly_heart.core.photo.prompt import _INSTRUCTIONS, render_photo_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "src/photo-contract.json"
    )
    args = parser.parse_args()
    contract = {
        "instructions": _INSTRUCTIONS,
        "schema": render_photo_prompt("en").schema,
        "languages": {code: info.name for code, info in SUPPORTED_LANGUAGES.items()},
    }
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
