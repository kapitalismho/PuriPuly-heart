from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from uuid import UUID
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    home = str(Path.home())
    text = text.replace(home.replace("\\", "\\\\"), "<USER_HOME>")
    text = text.replace(home, "<USER_HOME>").replace(home.replace("\\", "/"), "<USER_HOME>")
    path.write_text(text + "\n", encoding="utf-8")


def archived_jobs(values, capture_pcm):
    jobs = {}
    archive = ROOT / "native/nemotron/issue218_evidence.zip"
    with zipfile.ZipFile(archive) as stream:
        for member in ("cpu_initial/trace.json", "cpu_rotated_readback/trace.json"):
            raw = stream.read(member)
            report = json.loads(raw)
            for case in report["cases"]:
                if case["case"] != "distinct_repeated_A_B_A":
                    continue
                if hashlib.sha256(values.tobytes()).hexdigest() != case["input_pcm_f32_sha256"]:
                    raise ValueError("Assembled float32 PCM differs from archived exact input")
                results = {
                    event["value"]["identity"]["provider_turn_id"]: event["value"]
                    for event in case["trace"] if event["event"] == "native_result"
                }
                begins = {
                    event["request"]["identity"]["provider_turn_id"]: event["request"]
                    for event in case["trace"] if event["event"] == "native_begin"
                }
                for index, event in enumerate(e for e in case["trace"] if e["event"] == "native_submit"):
                    pieces = []
                    ranges = []
                    for payload in event["payloads"]:
                        piece = b"".join(
                            capture_pcm[span["normalized_start_sample"] * 2:span["normalized_end_sample"] * 2]
                            for span in payload["source_ranges"]
                        )
                        if len(piece) // 2 != payload["samples"] or hashlib.sha256(piece).hexdigest() != payload["pcm16_sha256"]:
                            raise ValueError("Archived submitted payload hash mismatch")
                        pieces.append(piece)
                        for span in payload["source_ranges"]:
                            left, right = span["normalized_start_sample"], span["normalized_end_sample"]
                            role = payload["payload_role"]
                            if ranges and ranges[-1][1] == left and ranges[-1][2] == role:
                                ranges[-1][1] = right
                            else:
                                ranges.append([left, right, role])
                    pcm = b"".join(pieces)
                    sha = hashlib.sha256(pcm).hexdigest()
                    if sha != event["pcm16_sha256"]:
                        raise ValueError("Whole submitted request hash mismatch")
                    turn = event["identity"]["provider_turn_id"]
                    reuse = {
                        "archive_member": str(archive.relative_to(ROOT)) + ":" + member,
                        "portable_member_sha256": hashlib.sha256(raw).hexdigest(),
                        "arm": case["profile"] if case["mode"] == "new" else "baseline",
                        "request_index": index,
                        "old_runtime": report["dependencies"]["sherpa-onnx"],
                        "old_native_text": results[turn]["text"],
                        "old_native_options": begins[turn].get("native_options"),
                    }
                    if sha not in jobs:
                        jobs[sha] = {"pcm": pcm, "pcm16_sha256": sha, "samples": len(pcm) // 2, "capture_ranges": ranges, "old_evidence_reused": [], "request": begins[turn], "submission": event}
                    jobs[sha]["old_evidence_reused"].append(reuse)
    return list(jobs.values())


class ObservedRecognizer:
    def __init__(self, inner):
        self.inner = inner
        self.results = []

    def create_stream(self):
        return self.inner.create_stream()

    def decode_stream(self, stream):
        self.inner.decode_stream(stream)
        result = stream.result
        self.results.append({name: getattr(result, name, None) for name in ("text", "tokens", "timestamps", "durations", "ys_log_probs", "lang", "words")})


def frozen(value):
    if isinstance(value, list):
        return tuple(frozen(item) for item in value)
    return value


async def scoped_decode(backend, pcm, raw_request, submission):
    from puripuly_heart.core.audio.format import AudioCaptureSpan
    from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
    from puripuly_heart.core.stt.backend import STTProviderTurnIdentity, STTProviderTurnRequest, STTSessionProjection

    raw_identity = raw_request["identity"]
    raw_segment = raw_identity["segment"]
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(**{**raw_segment, "segment_id": UUID(raw_segment["segment_id"])}),
        provider_epoch_id=raw_identity["provider_epoch_id"],
        provider_turn_id=raw_identity["provider_turn_id"],
        settings_scope=frozen(raw_identity["settings_scope"]),
    )
    settings = AudioSegmentSettingsSnapshot(**{key: frozen(value) for key, value in raw_request["settings"].items()})
    session = await backend.open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=identity.provider_epoch_id))
    request = STTProviderTurnRequest(
        identity=identity,
        settings=settings,
        channel=raw_request["channel"],
        display_group_id=UUID(raw_request["display_group_id"]) if raw_request["display_group_id"] else None,
        isolate_failure=raw_request["isolate_failure"],
        native_options=session.snapshot_decode_options(),
    )
    try:
        await session.begin_turn(request)
        cursor = 0
        for payload in submission["payloads"]:
            end = cursor + payload["samples"] * 2
            await session.send_turn_audio(
                identity,
                pcm[cursor:end],
                payload_sequence=payload["payload_sequence"],
                source_ranges=tuple(AudioCaptureSpan(**span) for span in payload["source_ranges"]),
                context_only=payload["context_only"],
            )
            cursor = end
        if cursor != len(pcm):
            raise ValueError("Scoped request differs from reconstructed native PCM")
        await session.seal_turn(
            identity,
            sealed_content_ranges=tuple(AudioCaptureSpan(**span) for span in submission["sealed_content_ranges"]),
            seal_reason=submission["seal_reason"],
            observed_trailing_silence_ms=submission["observed_trailing_silence_ms"],
        )
        terminal = await asyncio.wait_for(anext(session.turn_events()), 30)
        return terminal
    finally:
        await session.close()


async def run(args):
    if args.runtime_target:
        sys.path.insert(0, str(args.runtime_target.resolve()))
    import numpy as np
    import psutil
    import issue215_smoke as prior
    from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes, pcm16le_bytes_to_float32
    from puripuly_heart.providers.stt.local_parakeet_sherpa import LocalParakeetV3SherpaSTTBackend

    inputs, cases = prior.input_cases(SimpleNamespace(corpus=args.corpus, correction_root=args.correction_root))
    values, mapping = cases["distinct_repeated_A_B_A"]
    jobs = archived_jobs(values, float32_to_pcm16le_bytes(values))
    record = {
        "schema": "issue218-parakeet-exact-native-pcm-1",
        "command": sys.argv,
        "probe_sha256": digest(Path(__file__)),
        "python": sys.version,
        "dependencies": {key: importlib.metadata.version(key) for key in ("sherpa-onnx", "sherpa-onnx-core", "numpy", "onnxruntime")},
        "model_sha256": {path.name: digest(path) for path in args.model_dir.iterdir() if path.is_file()},
        "assembly_pcm_f32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "assembly_samples": int(values.size),
        "assembly_scoring_only": mapping,
        "input_provenance": inputs,
        "native_options": {"provider": "cpu", "num_threads": 2, "sample_rate_hz": 16000, "feature_dim": 80, "decoding_method": "greedy_search", "model_type": "nemo_transducer", "language_hint": None, "hotwords": []},
        "scope": "Same archived PCM16 requests decoded through actual product provider; no speaker policy, padding, context change, translation, cloud or VR calls. Old outputs reused, not rerun.",
        "product_path": "scoped session/frozen options/PCM16 queue/native terminal" if args.scoped else "provider decode_f32/native recognizer",
        "cases": [],
    }
    backend = LocalParakeetV3SherpaSTTBackend(model_dir=args.model_dir, num_threads=2, language_hint=None)
    recognizer = await backend._ensure_recognizer()
    observed = ObservedRecognizer(recognizer)
    backend._recognizer = observed
    import sherpa_onnx
    record["sherpa_module"] = str(Path(sherpa_onnx.__file__).resolve())
    record["native_runtime"] = {name: str(getattr(sherpa_onnx, name)) for name in ("git_sha1", "git_date", "version", "onnxruntime_version")}
    native_module = importlib.import_module("sherpa_onnx.lib._sherpa_onnx")
    record["native_module"] = {"path": str(Path(native_module.__file__).resolve()), "sha256": digest(Path(native_module.__file__))}
    record["loaded_native_libraries"] = [
        {"path": entry.path, "sha256": digest(Path(entry.path))}
        for entry in psutil.Process().memory_maps()
        if any(name in Path(entry.path).name.lower() for name in ("sherpa", "onnxruntime")) and Path(entry.path).is_file()
    ]
    try:
        for job in jobs:
            pcm = job.pop("pcm")
            raw_request = job.pop("request")
            submission = job.pop("submission")
            samples = pcm16le_bytes_to_float32(pcm)
            job["native_float32_sha256"] = hashlib.sha256(samples.tobytes()).hexdigest()
            job["native_float32_is_exact_pcm16_conversion"] = bool(np.array_equal(samples, np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768))
            started = time.perf_counter()
            if args.scoped:
                terminal = await scoped_decode(backend, pcm, raw_request, submission)
                job["product_text"] = terminal.text
                job["product_terminal"] = {"outcome": terminal.outcome, "text_authority": terminal.text_authority, "failure_reason": terminal.failure_reason, "epoch_disposition": terminal.epoch_disposition}
            else:
                job["product_text"] = await backend.decode_f32(samples)
            job["decode_seconds"] = time.perf_counter() - started
            job["raw_native_result"] = observed.results[-1]
            record["cases"].append(job)
            save(args.output, record)
            print(json.dumps({key: job[key] for key in ("capture_ranges", "pcm16_sha256", "product_text")}), flush=True)
    finally:
        await backend.close()
    record["status"] = "executed"
    save(args.output, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    home = Path.home()
    parser.add_argument("--runtime-target", type=Path)
    parser.add_argument("--scoped", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=home / "AppData/Local/puripuly-heart/models/parakeet-tdt-0.6b-v3-int8-sherpa")
    parser.add_argument("--corpus", type=Path, default=home / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1")
    parser.add_argument("--correction-root", type=Path, default=ROOT.parent / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.resolve().parent != ROOT / "native/nemotron" or not arguments.output.name.startswith("issue218_parakeet_"):
        raise ValueError("Report must remain within task-owned native/nemotron/issue218_parakeet_*")
    asyncio.run(run(arguments))
