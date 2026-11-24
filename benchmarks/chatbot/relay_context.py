from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath

from puripuly_heart.app.ports.agent_chat import AGENT_CHAT_PROTOCOL

FORMAT = "puri-agent-relay-context-1"
CONTEXT_IDS = frozenset({"classify", "work", "social", "out_of_scope"})
STATIC_FIELDS = frozenset(
    {
        "model",
        "input_prefix",
        "instructions",
        "tools",
        "questions",
        "reasoning",
        "prompt_cache_options",
        "store",
        "stream",
        "parallel_tool_calls",
        "max_output_tokens",
        "include",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(f"Invalid relay context manifest: {description}")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _prefix_item(item: object) -> bool:
    if not isinstance(item, dict) or item.get("role") not in ("developer", "system"):
        return False
    content = item.get("content")
    return _text(content) or (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(part, dict) and part.get("type") == "input_text" and _text(part.get("text"))
            for part in content
        )
    )


def validate_relay_context(value: object, *, relay_build: str | None) -> dict:
    _require(isinstance(value, dict), "expected an object")
    _require(value.get("format") == FORMAT, f"format must be {FORMAT}")
    _require(
        value.get("agent_protocol") == AGENT_CHAT_PROTOCOL,
        f"agent_protocol must be {AGENT_CHAT_PROTOCOL}",
    )
    _require(_text(relay_build), "the measured relay build is required")
    _require(
        value.get("relay_build") == relay_build, "relay_build does not match the measured relay"
    )
    hashes = value.get("bundle_hashes")
    _require(
        isinstance(hashes, dict) and bool(hashes), "bundle_hashes must contain emitted modules"
    )
    for name, digest in hashes.items():
        _require(
            isinstance(name, str)
            and bool(name)
            and "\\" not in name
            and ":" not in name
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            and isinstance(digest, str)
            and _SHA256.fullmatch(digest) is not None,
            "bundle_hashes must map relative module paths to SHA256 digests",
        )
    _require(
        relay_build in hashes.values(), "relay_build must identify a hashed emitted entry module"
    )
    contexts = value.get("contexts")
    _require(
        isinstance(contexts, dict) and set(contexts) == CONTEXT_IDS,
        "all four exact context IDs are required",
    )
    for identity, context in contexts.items():
        _require(isinstance(context, dict), f"{identity} must be an object")
        _require(set(context) <= STATIC_FIELDS, f"{identity} contains unknown or per-call fields")
        _require(_text(context.get("model")), f"{identity}.model is required")
        prefix = context.get("input_prefix")
        _require(
            isinstance(prefix, list) and all(_prefix_item(item) for item in prefix),
            f"{identity}.input_prefix must contain trusted messages only",
        )
        if "instructions" in context:
            _require(
                _text(context["instructions"]), f"{identity}.instructions must be nonempty text"
            )
        if "tools" in context:
            _require(
                isinstance(context["tools"], list)
                and all(
                    isinstance(tool, dict) and _text(tool.get("type")) for tool in context["tools"]
                ),
                f"{identity}.tools must contain tool objects",
            )
        if "questions" in context:
            questions = context["questions"]
            _require(
                isinstance(questions, list)
                and bool(questions)
                and all(
                    isinstance(question, dict)
                    and question.get("type") == "choice"
                    and _text(question.get("name"))
                    and _text(question.get("instructions"))
                    and isinstance(question.get("choices"), list)
                    and bool(question["choices"])
                    and all(
                        isinstance(choice, dict)
                        and _text(choice.get("value"))
                        and ("description" not in choice or _text(choice["description"]))
                        for choice in question["choices"]
                    )
                    for question in questions
                ),
                f"{identity}.questions must contain the emitted choice questions",
            )
        if "reasoning" in context:
            _require(
                isinstance(context["reasoning"], dict)
                and context["reasoning"].get("effort") in ("low", "medium", "high"),
                f"{identity}.reasoning must declare an effort",
            )
        if "prompt_cache_options" in context:
            cache = context["prompt_cache_options"]
            _require(
                isinstance(cache, dict)
                and _text(cache.get("mode"))
                and ("ttl" not in cache or _text(cache["ttl"])),
                f"{identity}.prompt_cache_options must contain emitted cache options",
            )
        for option in ("store", "stream", "parallel_tool_calls"):
            if option in context:
                _require(isinstance(context[option], bool), f"{identity}.{option} must be boolean")
        if "max_output_tokens" in context:
            limit = context["max_output_tokens"]
            _require(
                type(limit) is int and limit > 0, f"{identity}.max_output_tokens must be positive"
            )
        if "include" in context:
            _require(
                isinstance(context["include"], list)
                and all(_text(item) for item in context["include"]),
                f"{identity}.include must be a string list",
            )
        if identity == "classify":
            _require("questions" in context, "classify.questions are required")
        else:
            _require(
                "tools" in context and "reasoning" in context,
                f"{identity} tools and reasoning are required",
            )
            _require(
                bool(prefix) or _text(context.get("instructions")),
                f"{identity} trusted instructions are required",
            )
            if identity != "work":
                _require(context["tools"] == [], f"{identity} cannot expose work tools")
    return deepcopy(value)


def load_relay_context(path: str | Path | None, *, relay_build: str | None) -> dict:
    if not path:
        raise ValueError(
            "Live simulation requires --relay-context from the actual emitted relay bundle"
        )
    return validate_relay_context(
        json.loads(Path(path).read_text(encoding="utf-8")), relay_build=relay_build
    )


def upstream_context_id(method: str, arguments: dict) -> str:
    if method == "classify":
        return "classify"
    route = arguments.get("route")
    if method != "respond" or route not in ("work", "unclear", "social", "out_of_scope"):
        raise ValueError("A recorded model call requires its actual supported relay route")
    return "work" if route == "unclear" else route
