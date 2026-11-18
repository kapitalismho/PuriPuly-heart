from __future__ import annotations

import pytest

from benchmarks.chatbot.app import choice_catalog
from puripuly_heart.app.ports.agent_chat import AGENT_CHAT_PROTOCOL
from puripuly_heart.app.services.agent_chat.execution import (
    app_tool_descriptions,
    app_tool_schemas,
)


@pytest.fixture
async def relay_context_manifest() -> dict:
    marker = "TRUSTED_CONTEXT_ONLY_NOT_FOR_TERRA"
    schemas = app_tool_schemas(await choice_catalog())
    descriptions = app_tool_descriptions()
    return {
        "format": "puri-agent-relay-context-1",
        "agent_protocol": AGENT_CHAT_PROTOCOL,
        "relay_build": "a" * 64,
        "bundle_hashes": {"index.js": "a" * 64, "guide.md": "b" * 64},
        "contexts": {
            "classify": {
                "model": "synthetic-model",
                "input_prefix": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": marker,
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            }
                        ],
                    }
                ],
                "questions": [
                    {
                        "type": "choice",
                        "name": "route",
                        "instructions": "Classify current_message by the Route section.",
                        "choices": [{"value": "work"}, {"value": "unclear"}],
                    }
                ],
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            },
            "work": {
                "model": "synthetic-model",
                "input_prefix": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": marker,
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            }
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": name,
                        "description": descriptions[name],
                        "strict": False,
                        "parameters": schema,
                    }
                    for name, schema in schemas.items()
                ],
                "reasoning": {"effort": "high"},
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "store": False,
                "stream": True,
                "parallel_tool_calls": False,
                "max_output_tokens": 25000,
                "include": ["reasoning.encrypted_content"],
            },
            **{
                route: {
                    "model": "synthetic-model",
                    "input_prefix": [],
                    "instructions": f"{marker} {route}",
                    "tools": [],
                    "reasoning": {"effort": "low"},
                }
                for route in ("social", "out_of_scope")
            },
        },
        "export_provenance": {"mode": "controlled_test_fixture", "external_model_calls": 0},
    }
