from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import io
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, features

import bench_photo_438 as original

BASELINE = "fd47717576448a44d15b42dc1e8857c8a5fa9e32"
EXE_SHA256 = "8e4dd8add4c07d7743cc83538096528ab2973f8de7161a772bb2083fbcb75012"
OPTIONS = {"lossless": True, "method": 4, "quality": 100, "exact": True}
ARMS = ("jpeg85", "jpeg95", "png", "webp")
LIMITS = [
    "WebP is a later, noninterleaved batch with one generation per image; cache and service conditions may differ.",
    "COMPOSED preparation estimate reuses prior native source-read/decode/resize plus measured Pillow/libwebp encoding; cached PNG load is separate, not added.",
    "Composed serial-stage estimates are not continuous production end-to-end timings; no Rust encoder performance claim.",
    "Cached/reasoning token details are subsets, never added to input/output/total; missing usage is unavailable, not zero.",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stats(values):
    values = [value for value in values if value is not None]
    return {"available": len(values), "sum": sum(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None}


def manifest(old):
    result = original.load(old / "corpus.json")
    if len(result["images"]) != 25 or len({image["id"] for image in result["images"]}) != 25:
        raise RuntimeError("expected exactly 25 distinct owner-selected images")
    return result


def prepare(root, old, corpus):
    if not features.check("webp"):
        raise RuntimeError("Pillow WebP support unavailable; no dependencies installed")
    encoder = {"implementation": "Pillow/libwebp RGB lossless", "pillow": PIL.__version__,
               "libwebp": features.version("webp"), "options": OPTIONS,
               "quality_meaning": "lossless compression effort, NOT JPEG quality", "python": sys.version}
    rows = []
    for image in corpus["images"]:
        directory = root / image["id"] / "preparation"
        if (directory / "preparation.json").exists():
            rows.append(original.load(directory / "preparation.json"))
            continue
        meta = original.load(old / image["id"] / "preparation/preparation.json")
        png = old / image["id"] / "preparation/send-png.png"
        start = time.perf_counter()
        with Image.open(png) as source:
            source.load()
            if source.mode != "RGB" or list(source.size) != meta["send_dimensions"]:
                raise RuntimeError(f"unexpected shared RGB canvas: {image['id']}")
            canvas = source.copy()
        png_load_ms = (time.perf_counter() - start) * 1000
        rgb = canvas.tobytes()
        output = io.BytesIO()
        start = time.perf_counter()
        canvas.save(output, format="WEBP", **OPTIONS)
        encode_ms = (time.perf_counter() - start) * 1000
        data = output.getvalue()
        start = time.perf_counter()
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            if decoded.mode != "RGB" or decoded.size != canvas.size or decoded.tobytes() != rgb:
                raise RuntimeError(f"lossless pixel parity failed: {image['id']}")
        parity_ms = (time.perf_counter() - start) * 1000
        directory.mkdir(parents=True, exist_ok=True)
        encoded_path = directory / "send-webp.webp"
        encoded_path.write_bytes(data)
        composed = sum(meta[field] for field in ("source_read_ms", "decode_ms", "resize_ms")) + encode_ms
        meta["encodings"] = {"webp": {"path": str(encoded_path), "mime": "image/webp", "bytes": len(data),
                                      "encode_ms": encode_ms, "preparation_ms": composed}}
        meta.update(image_id=image["id"], encoder=encoder, png_source=str(png),
                    png_sha256=digest(png.read_bytes()), rgb_sha256=digest(rgb), encoded_sha256=digest(data),
                    pixel_parity=True, png_load_ms=png_load_ms, parity_decode_check_ms=parity_ms,
                    preparation_boundary=LIMITS[1])
        photo = original.prepared(meta, "webp")
        old_meta = original.load(old / image["id"] / "preparation/preparation.json")
        old_photo = original.prepared(old_meta, "png")
        if photo.send_size != old_photo.send_size or photo.map_box((10, 10, 100, 100), google=False) != old_photo.map_box((10, 10, 100, 100), google=False):
            raise RuntimeError("prepared mapping differs from shared PNG")
        webp_connection = original.ExperimentConnection("local-proof-no-credential", mime="image/webp", directory=directory / "body-proof")
        png_connection = original.ExperimentConnection("local-proof-no-credential", mime="image/png", directory=directory / "png-body-proof")
        webp_body = webp_connection._body(photo, "ko")
        png_body = png_connection._body(old_photo, "ko")
        url = webp_body["input"][0]["content"][0]["image_url"]
        if not url.startswith("data:image/webp;base64,") or base64.b64decode(url.split(",", 1)[1]) != data:
            raise RuntimeError("WebP MIME/body proof failed")
        webp_body["input"][0]["content"][0]["image_url"] = png_body["input"][0]["content"][0]["image_url"]
        if webp_body != png_body:
            raise RuntimeError("request settings/prompt differ beyond image encoding")
        meta["local_proof"] = {"mime_and_bytes": True, "same_prompt_settings": True, "same_prepared_mapping": True}
        original.save(directory / "preparation.json", meta)
        rows.append(meta)
    summary = {"images": len(rows), "pixel_parity_count": sum(row["pixel_parity"] for row in rows),
               "encoder": encoder, "encoded_bytes": stats([row["encodings"]["webp"]["bytes"] for row in rows]),
               "encode_ms": stats([row["encodings"]["webp"]["encode_ms"] for row in rows]),
               "png_load_ms": stats([row["png_load_ms"] for row in rows]),
               "composed_preparation_ms": stats([row["encodings"]["webp"]["preparation_ms"] for row in rows]),
               "old_encoded_bytes": {arm: stats([original.load(old / image["id"] / "preparation/preparation.json")["encodings"][arm]["bytes"] for image in corpus["images"]]) for arm in ARMS[:-1]},
               "provider_calls": 0, "secrets_loaded": False, "limits": LIMITS}
    original.save(root / "preparation-summary.json", summary)
    original.save(root / "corpus.json", corpus)
    print(json.dumps(summary, indent=2))


def usage_fields(value, prefix=""):
    result = {}
    for key, item in (value or {}).items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(usage_fields(item, name))
        elif type(item) is int:
            result[name] = item
    return result


def summarize(root, old, corpus):
    rows = []
    for image in corpus["images"]:
        for arm in ARMS:
            origin = root if arm == "webp" else old
            path = origin / image["id"] / "requests" / arm / "attempt-01/measurement.json"
            if not path.exists():
                continue
            row = original.load(path)
            process_path = origin / image["id"] / "replay" / arm / "process.json"
            process = original.load(process_path) if process_path.exists() else None
            row["replay"] = process
            row["composed_serial_stage_estimate_ms"] = (
                row["preparation_ms"] + row["timing_ms"]["request_start_to_completion"] + process["process_elapsed_ms"]
                if process and row["timing_ms"]["request_start_to_completion"] is not None else None)
            rows.append(row)
    fields = sorted({key for row in rows for key in usage_fields(row["usage_raw"])})
    aggregate = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        aggregate[arm] = {"attempts": len(selected), "completed": sum(row["status"] == "completed" for row in selected),
                          "failed": sum(row["status"] != "completed" for row in selected),
                          "parsed": sum(row.get("structured_parse_status") == "completed" for row in selected),
                          "replay_count": sum(row["replay"] is not None for row in selected),
                          "replay_exit_zero": sum(row["replay"] is not None and row["replay"]["exit_code"] == 0 for row in selected),
                          "usage": {field: {**stats([usage_fields(row["usage_raw"]).get(field) for row in selected]),
                                             "unavailable": sum(field not in usage_fields(row["usage_raw"]) for row in selected)} for field in fields}}
        for field in ("encoded_bytes", "request_json_bytes", "encode_ms", "preparation_ms", "composed_serial_stage_estimate_ms"):
            aggregate[arm][field] = stats([row.get(field) for row in selected])
        aggregate[arm]["timing_ms"] = {field: stats([row["timing_ms"].get(field) for row in selected]) for field in ("request_start_to_first_block", "request_start_to_completion", "request_start_to_terminal")}
        aggregate[arm]["replay_ms"] = stats([row["replay"]["process_elapsed_ms"] for row in selected if row["replay"]])
    original.save(root / "measurements.json", {"attempts": rows, "aggregate": aggregate, "limits": LIMITS})
    csv_fields = ["image_id", "arm", "status", "structured_parse_status", "block_count", "encoded_bytes", "request_json_bytes", "encode_ms", "preparation_ms", "request_ms", "replay_ms", "replay_exit", "composed_serial_stage_estimate_ms", *fields]
    with (root / "sanitized-measurements.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            values = {key: row.get(key) for key in csv_fields[:9]}
            values.update(request_ms=row["timing_ms"].get("request_start_to_completion"), replay_ms=row["replay"]["process_elapsed_ms"] if row["replay"] else None,
                          replay_exit=row["replay"]["exit_code"] if row["replay"] else None,
                          composed_serial_stage_estimate_ms=row["composed_serial_stage_estimate_ms"])
            values.update(usage_fields(row["usage_raw"]))
            writer.writerow(values)
    print(json.dumps(aggregate, indent=2))
    return aggregate


async def run(args):
    root, old = args.root.resolve(), args.old_root.resolve()
    if root == old or old in root.parents:
        raise RuntimeError("WebP artifacts must use a separate root")
    corpus = manifest(old)
    if args.phase == "prepare":
        prepare(root, old, corpus)
        return
    if args.phase == "provider":
        if not args.director_go:
            raise RuntimeError("provider phase requires explicit Director GO; no secrets loaded")
        configuration = original.load(old / "provider-configuration.json")
        if configuration["model"] != "gpt-6-luna" or configuration["target_language"] != "ko":
            raise RuntimeError("unexpected baseline request configuration")
        for image in corpus["images"]:
            meta = original.load(root / image["id"] / "preparation/preparation.json")
            if not meta["pixel_parity"] or digest(Path(meta["encodings"]["webp"]["path"]).read_bytes()) != meta["encoded_sha256"]:
                raise RuntimeError("prepared WebP evidence mismatch")
        original.save(root / "provider-configuration.json", {**{key: configuration[key] for key in ("route", "model", "endpoint", "target_language", "reasoning", "store", "stream", "max_output_tokens", "credential_owner")},
                      "baseline": BASELINE, "old_evidence_root": str(old), "automatic_retries": 0, "provider_concurrency_limit": args.concurrency,
                      "maximum_actual_calls": 25, "encoding": "webp", "encoder": OPTIONS, "started_utc": datetime.now(timezone.utc).isoformat(),
                      "adapter_sha256": digest(Path(__file__).read_bytes()), "reused_adapter_sha256": digest(Path(original.__file__).read_bytes()), "limits": LIMITS})
        key = original.KeyringSecretStore().get("openai_api_key")
        if not key:
            raise RuntimeError("OpenAI API credential unavailable through SecretStore")
        semaphore = asyncio.Semaphore(args.concurrency)

        async def one(image):
            async with semaphore:
                meta = original.load(root / image["id"] / "preparation/preparation.json")
                return await original.request(root, image, meta, "webp", corpus["images"].index(image), key)

        first = await one(corpus["images"][0])
        if first["status"] != "completed" or first["structured_parse_status"] != "completed":
            summarize(root, old, corpus)
            raise RuntimeError("first WebP usability request failed; retained, no fallback or retry")
        await asyncio.gather(*(one(image) for image in corpus["images"][1:]))
    elif args.phase == "replay":
        exe = old / "native/PuriPulyHeartOverlay.exe"
        if digest(exe.read_bytes()) != EXE_SHA256:
            raise RuntimeError("frozen native executable checksum mismatch")
        for image in corpus["images"]:
            directory = root / image["id"] / "replay/webp"
            if (directory / "process.json").exists():
                continue
            request_dir = root / image["id"] / "requests/webp/attempt-01"
            measurement = original.load(request_dir / "measurement.json")
            if measurement["structured_parse_status"] != "completed":
                original.save(directory / "unavailable.json", {"reason": "final_answer parsing failed; no fallback"})
                continue
            blocks = original.load(request_dir / "blocks.json")["blocks"]
            job = original.load(old / image["id"] / "replay/jpeg85/job.json")
            job.update(output_dir=str(directory), blocks=[{key: block[key] for key in ("block_id", "bbox", "source", "translation")} for block in blocks],
                       source_line_counts={str(block["block_id"]): block["source_line_count"] for block in blocks})
            result = await original.native_process(exe, "--offline", job, directory)
            print(json.dumps({"image_id": image["id"], **result}), flush=True)
    summarize(root, old, corpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "provider", "replay", "summary"), required=True)
    parser.add_argument("--root", type=Path, default=Path("build/issue-438-webp"))
    parser.add_argument("--old-root", type=Path, default=Path("build/issue-438"))
    parser.add_argument("--director-go", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
