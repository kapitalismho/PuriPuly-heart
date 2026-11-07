from __future__ import annotations

import argparse
import json
from pathlib import Path

from puripuly_heart.app.services.agent_chat.modules import MODULE_DESCRIPTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("agent-relay/src/context-modules.ts"))
    args = parser.parse_args()
    modules = [
        {"name": module, "description": description}
        for module, description in MODULE_DESCRIPTIONS.items()
    ]
    args.output.write_text(
        "export const contextModules = "
        + json.dumps(modules, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
