from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from puripuly_heart.app.services.agent_chat.execution import (
    APP_TOOLS,
    app_tool_descriptions,
    app_tool_schemas,
)

DEFINITION_NAMES = {
    "source": "language",
    "secondary_target": "optional_language",
    "peer_stt.provider": "stt_provider",
}


def _openai_schema(value: object) -> object:
    if isinstance(value, dict):
        return {key: _openai_schema(item) for key, item in value.items() if key != "uniqueItems"}
    if isinstance(value, list):
        return [_openai_schema(item) for item in value]
    return value


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _children(node: dict):
    for field, value in node.get("properties", {}).items():
        yield node["properties"], field, field
    if isinstance(node.get("items"), dict):
        yield node, "items", None


def _share_definitions(schema: dict) -> dict:
    schema = deepcopy(schema)
    definitions = {}
    while True:
        found = {}

        def visit(node: dict, key: str | None) -> None:
            for parent, slot, field in _children(node):
                child = parent[slot]
                name = field or key
                if "$ref" not in child:
                    identity = _compact(child)
                    found.setdefault(identity, [name, child, 0])[2] += 1
                visit(child, name)

        visit(schema, None)
        best = None
        for identity, (key, child, count) in found.items():
            name = DEFINITION_NAMES.get(key, key.replace(".", "_"))
            while name in definitions:
                name += "_"
            reference = _compact({"$ref": f"#/$defs/{name}"})
            saving = (count - 1) * len(identity) - count * len(reference) - len(name) - 3
            if saving > 0 and (best is None or saving > best[0]):
                best = (saving, identity, name, child)
        if best is None:
            break
        _, identity, name, child = best
        definitions[name] = deepcopy(child)

        def replace(node: dict) -> None:
            for parent, slot, _ in _children(node):
                if _compact(parent[slot]) == identity:
                    parent[slot] = {"$ref": f"#/$defs/{name}"}
                else:
                    replace(parent[slot])

        replace(schema)
    if definitions:
        schema["$defs"] = definitions
    return schema


def app_tools(catalog: dict) -> list[dict]:
    schemas = app_tool_schemas(catalog)
    descriptions = app_tool_descriptions()
    return [
        {
            "type": "function",
            "name": name,
            "description": descriptions[name],
            "strict": False,
            "parameters": _share_definitions(_openai_schema(schemas[name])),
        }
        for name in APP_TOOLS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("choices", type=Path)
    parser.add_argument("--output", type=Path, default=Path("agent-relay/src/app-tools.ts"))
    args = parser.parse_args()
    catalog = json.loads(args.choices.read_text(encoding="utf-8"))
    args.output.write_text(
        "export const appTools = "
        + json.dumps(app_tools(catalog), ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
