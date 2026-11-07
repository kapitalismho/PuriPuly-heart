from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

from issue373 import credentials, decode, save
from issue373_score import plain


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--env", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    from puripuly_heart.core.translation_backend import LlmTranslationBackend
    from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider

    provider = OpenRouterLLMProvider(api_key=credentials(args.env)("OPENROUTER_API_KEY"))
    backend = LlmTranslationBackend(provider)
    client = provider._get_client()
    cache, examples, calls = {}, [], []
    responses = []

    async def response_receipt(response):
        await response.aread()
        try:
            payload = response.json()
        except ValueError:
            return
        responses.append(
            {key: payload[key] for key in ("id", "model", "provider", "usage") if key in payload}
        )

    http = await client._get_http_client()
    http.event_hooks["response"].append(response_receipt)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    try:
        for item in selection:
            arm = json.loads(Path(item["file"]).read_text(encoding="utf-8"))
            request = decode(arm["requests"][item["request_index"]]["request"])
            kwargs = dataclasses.asdict(request)
            kwargs.pop("utterance_id")
            body = client._build_request_body(**kwargs)
            digest = hashlib.sha256(
                json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            cached = digest in cache
            if not cached:
                started = time.monotonic()
                try:
                    translation = await backend.translate(request)
                    cache[digest] = {
                        "translation": translation.text,
                        "service_s": time.monotonic() - started,
                    }
                    calls.append(
                        {
                            "request_sha256": digest,
                            "status": "completed",
                            "service_s": time.monotonic() - started,
                            "body": body,
                            "response": responses[-1] if responses else None,
                        }
                    )
                except Exception as exc:
                    cache[digest] = {
                        "error": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    }
                    calls.append(
                        {
                            "request_sha256": digest,
                            "status": "failed",
                            "error": type(exc).__name__,
                            "status_code": getattr(exc, "status_code", None),
                            "service_s": time.monotonic() - started,
                            "body": body,
                        }
                    )
            request_id = str(request.utterance_id)
            final_events = [
                plain(row["event"])
                for row in arm["events"]
                if plain(row["event"]).get("utterance_id") == request_id
                and plain(row["event"]).get("is_final")
            ]
            assignment = final_events[-1].get("speaker_assignment") if final_events else None
            examples.append(
                {
                    **item,
                    "arm": arm["arm"],
                    "language": arm["language"],
                    "source": request.text,
                    "speaker_assignment": assignment,
                    "request_sha256": digest,
                    "cache_hit": cached,
                    **cache[digest],
                }
            )
            save(
                args.output,
                {
                    "model": provider.model,
                    "models": list(provider.models),
                    "route": "OpenRouterLLMProvider -> LlmTranslationBackend",
                    "examples": examples,
                    "calls": calls,
                },
            )
    finally:
        await backend.close()
    print(
        json.dumps(
            {
                "examples": len(examples),
                "actual_calls": len(calls),
                "completed": sum(c["status"] == "completed" for c in calls),
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
