from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from puripuly_heart.core import network_clients

from puripuly_heart.core.photo.image import PreparedPhoto
from puripuly_heart.core.photo.parsing import BlockComplete, PhotoBlockParser, SourceReady
from puripuly_heart.core.photo.protocol import PhotoRequestSent, PhotoStreamCompleted, PhotoTextDelta
from puripuly_heart.core.storage.secrets import KeyringSecretStore
from puripuly_heart.providers.photo import OpenAIPhotoConnection

ARMS = ("jpeg85", "jpeg95", "png")
QUAD_INSTRUCTIONS = """Locate the four corners of each supplied text block in the image.
Return geometry only, never transcription or translation. Match every supplied block_id exactly once,
in the supplied reading order. Use the existing source text and bbox to identify the target, but inspect
its actual visible text orientation and extent: do not merely convert the bbox into four corners.
quad is exactly four [x,y] points ordered top-left, top-right, bottom-right, bottom-left in the text's
local reading frame, enclosing all physical lines, punctuation, and visible glyph outlines tightly.
Coordinates are SEND IMAGE pixels, not normalized coordinates. Do not include neighboring blocks.
Use genuine convex ordered corners. Return only the supplied JSON schema."""
QUAD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["quads"],
    "properties": {"quads": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["block_id", "quad"],
        "properties": {
            "block_id": {"type": "integer"},
            "quad": {"type": "array", "minItems": 4, "maxItems": 4, "items": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "number"},
            }},
        },
    }}},
}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepared(meta: dict, arm: str) -> PreparedPhoto:
    return PreparedPhoto(
        jpeg_base64=base64.b64encode(Path(meta["encodings"][arm]["path"]).read_bytes()).decode("ascii"),
        source_path=Path(meta["source_path"]),
        source_width=meta["source_dimensions"][0], source_height=meta["source_dimensions"][1],
        scale=meta["scale"], offset_x=meta["offset_x"], offset_y=meta["offset_y"],
        preparation_ms=meta["encodings"][arm]["preparation_ms"],
    )


def quad_context(blocks: list[dict], dimensions: list[int]) -> str:
    return json.dumps({
        "send_dimensions": dimensions,
        "blocks": [{"block_id": b["block_id"], "bbox": b["bbox_send"], "source": b["source"],
                    "source_line_count": b["source_line_count"]} for b in blocks],
    }, ensure_ascii=False)


class ExperimentConnection(OpenAIPhotoConnection):
    def __init__(self, key: str, *, mime: str, directory: Path, blocks=None, **kwargs):
        super().__init__(key, **kwargs)
        self.mime = mime
        self.directory = directory
        self.blocks = blocks
        self.body_bytes: bytes | None = None

    def _body(self, image: PreparedPhoto, target_language: str) -> dict:
        body = super()._body(image, target_language)
        content = body["input"][0]["content"]
        content[0]["image_url"] = f"data:{self.mime};base64,{image.jpeg_base64}"
        if self.blocks is not None:
            body["instructions"] = QUAD_INSTRUCTIONS
            content[1]["text"] = quad_context(self.blocks, list(image.send_size))
            body["text"]["format"].update(name="quads", schema=QUAD_SCHEMA)
        self.body_bytes = json.dumps(body, separators=(",", ":")).encode()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "body.json").write_bytes(self.body_bytes)
        return body


class CaptureStream(httpx.AsyncByteStream):
    def __init__(self, original, path: Path):
        self.original = original
        self.path = path

    async def __aiter__(self):
        with self.path.open("wb") as output:
            async for chunk in self.original:
                output.write(chunk)
                output.flush()
                yield chunk

    async def aclose(self):
        await self.original.aclose()


def select_final_answer(events: list[dict]) -> tuple[str, dict]:
    responses = [event.get("response") for event in events if event.get("type") == "response.completed"]
    if len(responses) != 1 or not isinstance(responses[0], dict):
        raise ValueError("expected exactly one completed terminal response")
    messages = [item for item in responses[0].get("output", []) if item.get("type") == "message"]
    final = [message for message in messages if message.get("phase") == "final_answer"]
    if len(final) != 1:
        raise ValueError(f"expected exactly one explicit final_answer message, got {len(final)}")
    parts = [part.get("text") for part in final[0].get("content", []) if part.get("type") == "output_text"]
    if not parts or any(not isinstance(part, str) for part in parts):
        raise ValueError("final_answer must contain output_text")
    return "".join(parts), {
        "message_id": final[0].get("id"), "message_phases": [message.get("phase") for message in messages],
        "source": "completed terminal response; explicit message phase=final_answer",
    }


def validate_quads(raw: str, blocks: list[dict], image: PreparedPhoto) -> dict:
    result = {"raw_predictions": None, "errors": [], "per_block": {}, "sidecar": None}
    expected = [b["block_id"] for b in blocks]
    try:
        predictions = json.loads(raw)["quads"]
        if not isinstance(predictions, list):
            raise ValueError("quads is not an array")
        result["raw_predictions"] = predictions
    except (ValueError, KeyError, TypeError) as error:
        result["errors"].append(str(error))
        predictions = []
    counts = Counter(p.get("block_id") for p in predictions if isinstance(p, dict)
                     and type(p.get("block_id")) is int)
    actual_order = [p.get("block_id") for p in predictions if isinstance(p, dict)]
    if actual_order != expected:
        result["errors"].append({"expected_order": expected, "actual_order": actual_order})
    sidecar = {}
    for block_id in expected:
        entry = {"status": "unavailable", "errors": [], "quad_send": None, "quad_original": None}
        result["per_block"][str(block_id)] = entry
        if counts[block_id] != 1:
            entry["errors"].append(f"expected one prediction, got {counts[block_id]}")
            continue
        prediction = next(p for p in predictions if isinstance(p, dict) and p.get("block_id") == block_id)
        q = prediction.get("quad")
        entry["quad_send"] = q
        if not (isinstance(q, list) and len(q) == 4 and all(
            isinstance(p, list) and len(p) == 2 and all(
                type(v) in (int, float) and math.isfinite(v) for v in p) for p in q)):
            entry["errors"].append("quad must have four finite numeric xy pairs")
            continue
        cross = []
        for i in range(4):
            a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
            cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not all(v > 1e-6 for v in cross):
            entry["errors"].append("nonconvex, reversed, crossing, or degenerate corner order")
        width, height = image.send_size
        if any(not (image.offset_x <= x <= min(width, image.offset_x + image.source_width * image.scale)
                    and image.offset_y <= y <= min(height, image.offset_y + image.source_height * image.scale))
               for x, y in q):
            entry["errors"].append("corner outside source-content rectangle; not clipped")
        original = [[(x - image.offset_x) / image.scale, (y - image.offset_y) / image.scale] for x, y in q]
        entry["quad_original"] = original
        if not entry["errors"]:
            entry["status"] = "available"
            sidecar[str(block_id)] = original
    extras = [p for p in predictions if not isinstance(p, dict) or type(p.get("block_id")) is not int
              or p["block_id"] not in expected]
    if extras:
        result["errors"].append({"extra_or_invalid_predictions": extras})
    if not result["errors"] and all(p["status"] == "available" for p in result["per_block"].values()):
        result["sidecar"] = sidecar
    return result


def quad_overlay(path: Path, quads: dict, image: PreparedPhoto) -> None:
    width, height = image.send_size
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{image.jpeg_base64}"/>']
    for index, prediction in enumerate(quads["raw_predictions"] or []):
        if not isinstance(prediction, dict):
            continue
        q = prediction.get("quad")
        if not (isinstance(q, list) and len(q) == 4 and all(
            isinstance(p, list) and len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p)
            for p in q)):
            continue
        block_id = prediction.get("block_id")
        label = str(block_id) if type(block_id) is int else f"invalid-{index}"
        color = "lime" if quads["per_block"].get(label, {}).get("status") == "available" else "red"
        points = " ".join(f"{x},{y}" for x, y in q)
        parts.append(f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for corner, (x, y) in enumerate(q):
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            parts.append(f'<text x="{x + 4}" y="{y - 4}" fill="{color}" stroke="black" stroke-width=".3" font-size="12">{label}:{corner}</text>')
    path.write_text("".join(parts) + "</svg>", encoding="utf-8")


def extract_response(directory: Path, photo: PreparedPhoto, arm: str, canonical, record: dict) -> None:
    record.update(parse_errors=[], structured_parse_status="failed", phase_selection_error=None, phase_selection=None, block_count=None)
    record["first_text_timing_boundary"] = "first raw provider output text; may include commentary"
    try:
        text, selection = select_final_answer(load(directory / "terminal-events.json"))
    except (ValueError, OSError) as error:
        record["phase_selection_error"] = str(error)
        record["parse_errors"].append(str(error))
        record["timing_ms"]["request_start_to_first_block"] = None
        record["first_block_unavailable_reason"] = "explicit final_answer selection failed"
        for name in ("blocks.json", "quad-original.json"):
            (directory / name).unlink(missing_ok=True)
        if arm == "quad":
            record.update(geometry_available=False, unavailable_block_ids=[str(block["block_id"]) for block in canonical],
                          geometry_errors=[str(error)], block_count=len(canonical))
            save(directory / "quads.json", {
                "raw_predictions": None, "errors": [str(error)], "sidecar": None,
                "per_block": {str(block["block_id"]): {"status": "unavailable", "errors": [str(error)],
                              "quad_send": None, "quad_original": None} for block in canonical},
            })
            (directory / "quad-overlay.svg").unlink(missing_ok=True)
        return
    record["phase_selection"] = selection
    (directory / "final-answer.txt").write_text(text, encoding="utf-8")
    raw_text = (directory / "response.txt").read_text(encoding="utf-8")
    if selection["message_phases"] != ["final_answer"] or raw_text != text:
        record["timing_ms"]["request_start_to_first_block"] = None
        record["first_block_unavailable_reason"] = "raw stream spans other output phases; final block timing not observed"
    else:
        record["first_block_timing_boundary"] = "completed block streamed in sole explicit final_answer message"
    if arm == "quad":
        quads = validate_quads(text, canonical, photo)
        save(directory / "quads.json", quads)
        quad_overlay(directory / "quad-overlay.svg", quads, photo)
        record["geometry_available"] = quads["sidecar"] is not None
        record["unavailable_block_ids"] = [key for key, value in quads["per_block"].items() if value["status"] != "available"]
        record["geometry_errors"] = quads["errors"]
        record["first_block_unavailable_reason"] = "quad extraction is not the photo block streaming protocol"
        if quads["sidecar"] is not None:
            save(directory / "quad-original.json", quads["sidecar"])
            record["structured_parse_status"] = "completed"
        else:
            (directory / "quad-original.json").unlink(missing_ok=True)
        record["block_count"] = len(canonical)
        return
    parser = PhotoBlockParser()
    events = parser.feed(text)
    lines = {event.index: event.source_line_count for event in events if isinstance(event, SourceReady)}
    blocks = []
    try:
        for index, block in enumerate(parser.finish()):
            item = {"block_id": index, "bbox_send": block.bbox, "source": block.source,
                    "translation": block.translation, "source_line_count": lines[index], "bbox": None}
            try:
                mapped = photo.map_box(block.bbox, google=False)
                item.update(bbox=list(mapped.bbox), bbox_corrected=mapped.corrected)
            except ValueError as error:
                item["mapping_error"] = str(error)
                record["parse_errors"].append({"block_id": index, "mapping_error": str(error)})
            blocks.append(item)
        record["block_count"] = len(blocks)
        record["structured_parse_status"] = "failed" if record["parse_errors"] else "completed"
        save(directory / "blocks.json", {"blocks": blocks, "parse_events": [asdict(event) for event in events]})
    except ValueError as error:
        record["parse_errors"].append(str(error))
        (directory / "blocks.json").unlink(missing_ok=True)
        save(directory / "partial-parse-events.json", [asdict(event) for event in events])


async def request(root: Path, image_info: dict, meta: dict, arm: str, order: int, key: str):
    directory = root / image_info["id"] / "requests" / arm / "attempt-01"
    measurement_path = directory / "measurement.json"
    if measurement_path.exists():
        return load(measurement_path)
    directory.mkdir(parents=True, exist_ok=True)
    encoding = "jpeg85" if arm == "quad" else arm
    photo = prepared(meta, encoding)
    canonical = None
    if arm == "quad":
        baseline_path = root / image_info["id"] / "requests/jpeg85/attempt-01/blocks.json"
        if not baseline_path.exists():
            raise RuntimeError(f"{image_info['id']}: canonical baseline unavailable")
        canonical = load(baseline_path)["blocks"]
    record = {
        "image_id": image_info["id"], "experiment": "B" if arm == "quad" else "A", "arm": arm,
        "request_order": order, "attempt": 1, "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "encoded_bytes": meta["encodings"][encoding]["bytes"], "request_json_bytes": None,
        "request_size_boundary": "exact outgoing JSON body bytes; excludes headers, HTTP framing and TLS",
        "encode_ms": meta["encodings"][encoding]["encode_ms"],
        "preparation_ms": meta["encodings"][encoding]["preparation_ms"],
        "timing_ms": {"request_start_to_first_block": None, "request_start_to_completion": None,
                      "request_start_to_first_text": None, "request_start_to_headers_sent": None},
        "usage_raw": None, "model": None, "error": None, "parse_errors": [], "block_count": None,
    }
    save(measurement_path, record)
    raw = []
    parser = PhotoBlockParser()
    completed = False
    start = time.monotonic()

    async def capture_response(response):
        record["http_status"] = response.status_code
        response.stream = CaptureStream(response.stream, directory / "response.sse")

    async def capture_request(outgoing):
        if outgoing.content != connection.body_bytes:
            raise RuntimeError("captured body differs from outgoing body")

    async with network_clients.external_async_client(timeout=httpx.Timeout(240, connect=20), follow_redirects=False,
                                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                                event_hooks={"response": [capture_response], "request": [capture_request]}) as client:
        connection = ExperimentConnection(key, mime=meta["encodings"][encoding]["mime"],
                                          directory=directory, blocks=canonical, client=client)
        start = time.monotonic()
        try:
            async for event in connection.stream(photo, "ko"):
                elapsed = (time.monotonic() - start) * 1000
                if isinstance(event, PhotoRequestSent):
                    record["timing_ms"]["request_start_to_headers_sent"] = (event.monotonic_time - start) * 1000
                elif isinstance(event, PhotoTextDelta):
                    if record["timing_ms"]["request_start_to_first_text"] is None:
                        record["timing_ms"]["request_start_to_first_text"] = elapsed
                    raw.append(event.text)
                    if arm != "quad":
                        for parsed_event in parser.feed(event.text):
                            if isinstance(parsed_event, BlockComplete) and record["timing_ms"]["request_start_to_first_block"] is None:
                                record["timing_ms"]["request_start_to_first_block"] = elapsed
                elif isinstance(event, PhotoStreamCompleted):
                    completed = True
                    record["timing_ms"]["request_start_to_completion"] = elapsed
                    record["usage_raw"] = event.usage or None
                    record["model"] = event.model
            if not completed:
                raise RuntimeError("stream exited without completed event")
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
                               "retryable": getattr(error, "retryable", None)}
        finally:
            record["timing_ms"]["request_start_to_terminal"] = (time.monotonic() - start) * 1000
            await connection.close()
    text = "".join(raw)
    (directory / "response.txt").write_text(text, encoding="utf-8")
    record["request_json_bytes"] = len(connection.body_bytes) if connection.body_bytes is not None else None
    sse_path = directory / "response.sse"
    if sse_path.exists():
        terminal_events = []
        for frame in sse_path.read_text(encoding="utf-8", errors="replace").split("\n\n"):
            data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") in ("response.completed", "response.failed", "response.incomplete", "error"):
                terminal_events.append(event)
                response = event.get("response") or {}
                if record["usage_raw"] is None and response.get("usage") is not None:
                    record["usage_raw"] = response["usage"]
                if record["model"] is None:
                    record["model"] = response.get("model")
        save(directory / "terminal-events.json", terminal_events)
    extract_response(directory, photo, arm, canonical, record)
    save(measurement_path, record)
    with (root / "attempts.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"image_id": image_info["id"], "arm": arm, "status": record["status"],
                      "block_count": record["block_count"], "error": record["error"],
                      "geometry_available": record.get("geometry_available")}), flush=True)
    return record


async def native_process(exe: Path, flag: str, job: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    save(job_path, job)
    start = time.monotonic()
    process = await asyncio.create_subprocess_exec(str(exe), "--role", "photo", flag, str(job_path),
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    result = {"exit_code": process.returncode, "process_elapsed_ms": (time.monotonic() - start) * 1000,
              "timing_boundary": "subprocess creation through process exit; includes startup and artifact writes"}
    (directory / "stdout.txt").write_bytes(stdout)
    (directory / "stderr.txt").write_bytes(stderr)
    save(directory / "process.json", result)
    return result


def summarize(root: Path):
    records = [load(p) for p in sorted(root.glob("image-*/requests/*/attempt-*/measurement.json"))]
    aggregate = {}
    for arm in (*ARMS, "quad"):
        rows = [r for r in records if r["arm"] == arm]
        usage = {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [r["usage_raw"].get(field) for r in rows if r["usage_raw"] is not None]
            available = [v for v in values if type(v) is int]
            usage[field] = {"reported_sum": sum(available) if available else None,
                            "available_requests": len(available), "unavailable_requests": len(rows) - len(available)}
        timings = {}
        for field in ("request_start_to_first_block", "request_start_to_completion", "request_start_to_terminal"):
            values = [r["timing_ms"].get(field) for r in rows]
            present = [v for v in values if v is not None]
            timings[field] = {"mean": statistics.mean(present) if present else None,
                              "median": statistics.median(present) if present else None, "available": len(present)}
        aggregate[arm] = {"attempts": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
                          "failed": sum(r["status"] == "failed" for r in rows), "usage": usage, "timing_ms": timings,
                          "encoded_bytes_sum": sum(r["encoded_bytes"] for r in rows),
                          "request_json_bytes_sum": sum(r["request_json_bytes"] or 0 for r in rows)}
    save(root / "measurements.json", {"attempts": records, "aggregate": aggregate,
         "usage_detail_policy": "usage_raw retains reported subsets; cached/reasoning/image details are not added to totals"})
    return aggregate


def reparse(root: Path, images: list[dict]) -> None:
    rows = []
    for image in images:
        meta = load(root / image["id"] / "preparation/preparation.json")
        for arm in (*ARMS, "quad"):
            directory = root / image["id"] / "requests" / arm / "attempt-01"
            record = load(directory / "measurement.json")
            old_usage = record["usage_raw"]
            old_blocks = load(directory / "blocks.json") if (directory / "blocks.json").exists() else None
            old_quads = load(directory / "quads.json") if (directory / "quads.json").exists() else None
            superseded = directory / "superseded-concatenated-adapter"
            superseded.mkdir(exist_ok=True)
            for name in ("measurement.json", "blocks.json", "quads.json", "quad-overlay.svg", "quad-original.json", "partial-parse-events.json"):
                source = directory / name
                if source.exists() and not (superseded / name).exists():
                    shutil.copy2(source, superseded / name)
            canonical = load(root / image["id"] / "requests/jpeg85/attempt-01/blocks.json")["blocks"] if arm == "quad" else None
            extract_response(directory, prepared(meta, "jpeg85" if arm == "quad" else arm), arm, canonical, record)
            record["reextracted_from_captured_terminal"] = True
            save(directory / "measurement.json", record)
            rows.append({
                "image_id": image["id"], "arm": arm, "structured_parse_status": record["structured_parse_status"],
                "blocks_unchanged": load(directory / "blocks.json") == old_blocks if arm != "quad" and (directory / "blocks.json").exists() else None,
                "quads_unchanged": load(directory / "quads.json") == old_quads if arm == "quad" else None,
                "usage_unchanged": record["usage_raw"] == old_usage,
                "geometry_available": record.get("geometry_available"), "phase_selection_error": record["phase_selection_error"],
            })
    save(root / "phase-reextraction.json", {"new_provider_calls": 0, "secret_access": False, "rows": rows})
    summarize(root)


async def run(args):
    root = args.root.resolve()
    manifest = load(root / "corpus.json")
    images = manifest["images"]
    if args.image:
        images = [image for image in images if image["id"] in args.image]
    if args.phase == "prepare":
        for image in images:
            if hashlib.sha256(Path(image["source_path"]).read_bytes()).hexdigest() != image["sha256"]:
                raise RuntimeError(f"source checksum mismatch: {image['id']}")
            directory = root / image["id"] / "preparation"
            if (directory / "preparation.json").exists():
                continue
            result = await native_process(args.exe.resolve(), "--prepare-experiment",
                                          {"source_path": image["source_path"], "output_dir": str(directory)}, directory)
            if result["exit_code"]:
                raise RuntimeError(f"preparation failed: {image['id']}")
    elif args.phase == "provider":
        configuration_path = root / "provider-configuration.json"
        if not configuration_path.exists():
            save(configuration_path, {
                "baseline": manifest["baseline"], "route": "openai-api", "model": "gpt-6-luna",
                "endpoint": OpenAIPhotoConnection.endpoint, "target_language": "ko",
                "reasoning": {"effort": "none"}, "store": False, "stream": True,
                "max_output_tokens": 8192, "automatic_retries": 0, "provider_concurrency_limit": 3,
                "credential_owner": "KeyringSecretStore; existing openai_api_key; credential never retained",
                "product_settings": "not read or modified; managed selection unchanged",
                "order": "rotate jpeg85/jpeg95/png by manifest image index; quad after compression arms",
                "geometry_encoding": "jpeg85", "quad_instructions": QUAD_INSTRUCTIONS,
                "quad_schema": QUAD_SCHEMA, "quad_context": "canonical JPEG85 block_id/bbox_send/source/source_line_count only",
                "python": sys.version, "interpreter": sys.executable,
                "checkout": str(Path(__file__).resolve().parents[1]),
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timing_boundary": "request monotonic start before stream/body preparation, no preopen; completion measured separately from close",
                "perspective_warp": "not implemented; unevaluated",
            })
        key = KeyringSecretStore().get("openai_api_key")
        if not key:
            raise RuntimeError("OpenAI API credential unavailable through SecretStore")
        with (root / "provider-executions.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "started_utc": datetime.now(timezone.utc).isoformat(), "concurrency": args.concurrency,
                "max_calls": args.max_calls, "image_ids": [image["id"] for image in images],
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }) + "\n")
        semaphore = asyncio.Semaphore(args.concurrency)
        budget = args.max_calls
        admitted = 0

        async def per_image(image):
            nonlocal admitted
            async with semaphore:
                meta = load(root / image["id"] / "preparation/preparation.json")
                index = manifest["images"].index(image)
                order = list(ARMS[index % 3:] + ARMS[:index % 3])
                save(root / image["id"] / "request-order.json", order + ["quad"])
                for position, arm in enumerate(order + ["quad"]):
                    if (root / image["id"] / "requests" / arm / "attempt-01/measurement.json").exists():
                        continue
                    if budget is not None and admitted >= budget:
                        return
                    admitted += 1
                    await request(root, image, meta, arm, position, key)
        await asyncio.gather(*(per_image(image) for image in images))
        summarize(root)
    elif args.phase == "reparse":
        reparse(root, images)
    elif args.phase == "replay":
        for image in images:
            for arm in (*ARMS, "quad"):
                directory = root / image["id"] / "replay" / arm
                if (directory / "process.json").exists():
                    continue
                source_arm = "jpeg85" if arm == "quad" else arm
                measurement = load(root / image["id"] / "requests" / source_arm / "attempt-01/measurement.json")
                if measurement.get("structured_parse_status") != "completed":
                    save(directory / "unavailable.json", {"reason": "explicit final_answer structured parse unavailable"})
                    continue
                blocks_path = root / image["id"] / "requests" / source_arm / "attempt-01/blocks.json"
                if not blocks_path.exists():
                    save(directory / "unavailable.json", {"reason": "parsed blocks unavailable"})
                    continue
                blocks = load(blocks_path)["blocks"]
                if any(b["bbox"] is None for b in blocks):
                    save(directory / "unavailable.json", {"reason": "invalid baseline mapping retained; no block dropped"})
                    continue
                job = {"source_path": image["source_path"], "output_dir": str(directory),
                       "model_path": str(args.model.resolve()), "runtime_path": str(args.runtime.resolve()),
                       "blocks": [{key: b[key] for key in ("block_id", "bbox", "source", "translation")} for b in blocks],
                       "source_line_counts": {str(b["block_id"]): b["source_line_count"] for b in blocks}}
                if arm == "quad":
                    quad_path = root / image["id"] / "requests/quad/attempt-01/quad-original.json"
                    if not quad_path.exists():
                        save(directory / "unavailable.json", {"reason": "quad invalid/missing; Director disposition required; no AABB fallback"})
                        print(f"UNAVAILABLE {image['id']} quad: Director disposition required", flush=True)
                        continue
                    job["quad_path"] = str(quad_path)
                result = await native_process(args.exe.resolve(), "--offline", job, directory)
                print(json.dumps({"image_id": image["id"], "arm": arm, **result}), flush=True)
        summarize(root)
    else:
        print(json.dumps(summarize(root), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "provider", "reparse", "replay", "summary"), required=True)
    parser.add_argument("--root", type=Path, default=Path("build/issue-438"))
    parser.add_argument("--exe", type=Path, default=Path("build/issue-438/native/PuriPulyHeartOverlay.exe"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--image", action="append")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    if args.phase == "replay" and (args.model is None or args.runtime is None):
        parser.error("replay requires --model and --runtime")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
