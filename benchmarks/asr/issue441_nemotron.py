from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

import numpy as np

PROFILE = (9, 4, 0, 264, 264, 222)
MODEL_SHA256 = "9b277e1dcec60b2c771a1dcb45dc8e36f7285ff7c1619d880d3926a373036b38"
SOURCE_REVISION = "b809bbb467fb2da2be0042fff5ef1f503828c9cf"
RATE = 16000


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ModelConfig(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t), ("model_path", ct.c_char_p),
        ("gpu", ct.c_int32), ("preset", ct.c_char_p),
        *((name, ct.c_int32) for name in (
            "chunk_frames", "right_context_frames", "left_context_frames",
            "fifo_frames", "spkcache_frames", "update_period_frames")),
    ]


class Session:
    def __init__(self, runtime, model, device):
        self.directory = os.add_dll_directory(str((runtime / "bin").resolve()))
        self.dll = ct.CDLL(str((runtime / "bin/nemo_speech_asr_c.dll").resolve()), winmode=0x1100)
        signatures = {
            "nemo_speech_diar_create": ([ct.POINTER(ModelConfig), ct.POINTER(ct.c_void_p)], ct.c_int),
            "nemo_speech_diar_destroy": ([ct.c_void_p], None),
            "nemo_speech_diar_stream_open": ([ct.c_void_p, ct.POINTER(ct.c_void_p)], ct.c_int),
            "nemo_speech_diar_stream_close": ([ct.c_void_p], None),
            "nemo_speech_diar_stream_push_f32": ([ct.c_void_p, ct.POINTER(ct.c_float), ct.c_size_t, ct.c_int32], ct.c_int),
            "nemo_speech_diar_num_speakers": ([ct.c_void_p], ct.c_int32),
            "nemo_speech_diar_seconds_per_frame": ([ct.c_void_p], ct.c_double),
            "puripuly_nemotron_consumed_samples": ([ct.c_void_p], ct.c_int64),
            "puripuly_nemotron_immutable_frames": ([ct.c_void_p], ct.c_int64),
            "puripuly_nemotron_drain": ([ct.c_void_p, ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.c_int64), ct.POINTER(ct.c_size_t)], ct.c_int),
            "nemo_speech_asr_last_error": ([], ct.c_char_p),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = args, result
        self.model, self.stream = ct.c_void_p(), ct.c_void_p()
        self.buffer = np.empty((1024, 8), dtype=np.float32)
        config = ModelConfig(ct.sizeof(ModelConfig), str(model.resolve()).encode(), device, None, *PROFILE)
        try:
            self.check(self.dll.nemo_speech_diar_create(ct.byref(config), ct.byref(self.model)))
            if self.dll.nemo_speech_diar_num_speakers(self.model) != 8:
                raise ValueError("unexpected speaker capacity")
            if abs(self.dll.nemo_speech_diar_seconds_per_frame(self.model) - 0.01) > 1e-9:
                raise ValueError("unexpected probability cadence")
            self.check(self.dll.nemo_speech_diar_stream_open(self.model, ct.byref(self.stream)))
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            error = self.dll.nemo_speech_asr_last_error()
            raise RuntimeError(error.decode(errors="replace") if error else f"native status {status}")

    def push(self, pcm):
        self.check(self.dll.nemo_speech_diar_stream_push_f32(
            self.stream, pcm.ctypes.data_as(ct.POINTER(ct.c_float)), pcm.size, RATE))
        first, count = ct.c_int64(), ct.c_size_t()
        self.check(self.dll.puripuly_nemotron_drain(
            self.stream, self.buffer.ctypes.data_as(ct.POINTER(ct.c_float)), 1024,
            ct.byref(first), ct.byref(count)))
        if count.value > 1024:
            raise ValueError("oversized native drain")
        values = self.buffer[:count.value]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("invalid probabilities")
        return dict(first_frame=first.value, count=count.value,
                    processed_samples=self.dll.puripuly_nemotron_consumed_samples(self.stream),
                    immutable_frames=self.dll.puripuly_nemotron_immutable_frames(self.stream),
                    probabilities=values.tolist())

    def close(self):
        if self.stream:
            self.dll.nemo_speech_diar_stream_close(self.stream)
            self.stream = ct.c_void_p()
        if self.model:
            self.dll.nemo_speech_diar_destroy(self.model)
            self.model = ct.c_void_p()
        self.directory.close()


def save(path, record):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_record(args, spec):
    output = args.output / (spec["id"] + ".json")
    if output.exists():
        raise FileExistsError(f"refusing to replace previous execution: {output}")
    args.output.mkdir(parents=True, exist_ok=True)
    connection_start = None
    record = dict(schema_version=1, arm="N", recording_id=spec["id"],
                  metadata={}, source_origin_mono_s=None, events=[], sends=[], observations=[],
                  status="starting", connected=False, errors=[], sent_audio_s=0, connection_duration_s=0)
    session = None
    origin = time.monotonic()
    journal_path = output.with_suffix(".journal.jsonl")

    def journal(kind, raw):
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(kind=kind, raw=raw), allow_nan=False) + "\n")


    def event(kind, raw, arrival=None):
        if arrival is None:
            arrival = time.monotonic() - origin
        record["events"].append(dict(seq=len(record["events"]), arrival_s=arrival, kind=kind, raw=raw))
        journal("event", record["events"][-1])
        return arrival

    try:
        model_hash = digest(args.model)
        if args.model.stat().st_size != 201832928 or model_hash != MODEL_SHA256:
            raise ValueError("model differs from pinned artifact")
        manifest_path = args.runtime / "runtime-manifest.json"
        runtime = json.loads(manifest_path.read_text(encoding="utf-8"))
        if runtime["source"]["revision"] != SOURCE_REVISION:
            raise ValueError("runtime source revision mismatch")
        for name, expected in runtime["files"].items():
            if digest(args.runtime / name) != expected:
                raise ValueError(f"runtime inventory mismatch: {name}")
        if args.smoke:
            pcm = np.zeros(RATE * 3, dtype=np.float32)
            input_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
        else:
            audio = Path(spec["audio_path"])
            if not audio.is_absolute():
                audio = args.root / audio
            input_hash = digest(audio)
            if input_hash != spec["sha256"]:
                raise ValueError("input differs from frozen manifest")
            with wave.open(str(audio), "rb") as source:
                if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
                    raise ValueError("16 kHz mono PCM16 required")
                pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
            if abs(len(pcm) / RATE - spec["duration_s"]) > 1 / RATE:
                raise ValueError("manifest duration mismatch")
        record["metadata"] = dict(
            checkout=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip(),
            runner_sha256=digest(Path(__file__)), model_sha256=model_hash,
            model_revision="25cf0d6b0c91b679376df0f6ef3087b4fd96af1f",
            runtime=runtime, runtime_sha256=digest(manifest_path),
            profile=dict(name="1.04", geometry=list(PROFILE), input_buffer_delay_s=1.04),
            input_sha256=input_hash, manifest_sha256=None if args.smoke else digest(args.inputs),
            language=spec.get("language"), config=dict(device=args.device, sample_rate=RATE,
                send_samples=1600, activity_threshold=0.5, language_hint=None,
                pacing="source block end at realtime; EOF padding unpaced",
                eof_padding_samples=17152, preserve_silence=True,
                environment={key: os.environ.get(key) for key in (
                    "HF_HUB_OFFLINE", "GGML_VK_DISABLE_GRAPH_OPTIMIZE", "NEMO_SPEECH_SCHEDULE_CAPTURE")}))
        record["metadata"]["failure_journal"] = journal_path.name
        save(output, record)
        startup_start = time.monotonic()
        connection_start = startup_start
        session = Session(args.runtime, args.model, args.device)
        record["connected"] = True
        origin = time.monotonic()
        record["source_origin_mono_s"] = origin
        record["status"] = "running"
        event("startup", dict(start_mono_s=startup_start, ready_mono_s=origin,
                              duration_s=origin-startup_start, speakers=8, seconds_per_frame=0.01))
        transmitted = 0
        next_frame = 0
        save(output, record)

        def send(block, start, kind):
            nonlocal transmitted, next_frame
            before = time.monotonic() - origin
            entry = dict(source_start_s=start / RATE if kind == "source" else None,
                         source_end_s=(start+len(block)) / RATE if kind == "source" else None,
                         transmitted_start_s=transmitted/RATE,
                         transmitted_end_s=(transmitted+len(block))/RATE,
                         start_s=before, completed_s=None, status="attempted", kind=kind,
                         sample_count=len(block))
            if kind == "source":
                entry["scheduled_s"] = (start + len(block)) / RATE
                entry["pacing_lateness_s"] = before - entry["scheduled_s"]
            record["sends"].append(entry)
            journal("send_attempt", entry)
            processing_start = time.monotonic() - origin
            entry["processing_started_s"] = processing_start
            raw = session.push(block)
            received = time.monotonic() - origin
            entry.update(completed_s=received, status="completed", processing_delay_s=received-processing_start)
            journal("send_complete", entry)
            transmitted += len(block)
            if kind == "source":
                record["sent_audio_s"] += len(block)/RATE
            if raw["first_frame"] != next_frame or raw["immutable_frames"] != next_frame + raw["count"]:
                raise ValueError("noncontiguous or nonimmutable probability output")
            if not 0 <= raw["immutable_frames"]*160 <= raw["processed_samples"] <= transmitted:
                raise ValueError("invalid native frontier")
            raw.update(processing_delay_s=received-processing_start, transmitted_samples=transmitted,
                       source_received_s=min(transmitted,len(pcm))/RATE,
                       input_buffer_delay_s=1.04)
            arrival = event("probabilities" if kind == "source" else "eof_probabilities", raw, received)
            for index, row in enumerate(raw["probabilities"]):
                frame = next_frame + index
                if frame * 160 >= len(pcm):
                    break
                for speaker, probability in enumerate(row):
                    if probability >= 0.5:
                        record["observations"].append(dict(arrival_s=arrival, start_s=frame*0.01,
                            end_s=min((frame+1)*0.01,len(pcm)/RATE), speaker=str(speaker),
                            text=None, settled=True, kind="activity", probability=probability,
                            event_seq=len(record["events"])-1, eof=kind != "source"))
            next_frame += raw["count"]

        for start in range(0, len(pcm), 1600):
            block = pcm[start:start+1600]
            time.sleep(max(0, origin+(start+len(block))/RATE-time.monotonic()))
            send(block, start, "source")
            if start % RATE == 0:
                save(output, record)
        event("source_eof", dict(source_samples=len(pcm), transmitted_samples=transmitted))
        padding = np.zeros(17152, dtype=np.float32)
        for start in range(0, len(padding), 1600):
            send(padding[start:start+1600], len(pcm), "eof_padding")
        record["eof"] = dict(received=True, method="explicit zero padding then probability drain",
                            padding_samples=len(padding), emitted_frames=next_frame,
                            source_samples=len(pcm), covered_source_samples=min(next_frame*160,len(pcm)),
                            arrival_s=time.monotonic()-origin)
        if next_frame * 160 < len(pcm):
            raise RuntimeError("EOF drain did not cover source tail")
        record["status"] = "completed"
        event("drain_complete", record["eof"])
    except BaseException as error:
        record["status"] = "failed"
        record["errors"].append(dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        if session is not None:
            session.close()
        record["connection_duration_s"] = 0 if connection_start is None else time.monotonic() - connection_start
        record["sent_source_samples"] = sum(entry["sample_count"] for entry in record["sends"]
                                             if entry["kind"] == "source" and entry["status"] == "completed")
        record["transmitted_samples"] = sum(entry["sample_count"] for entry in record["sends"]
                                              if entry["status"] == "completed")
        save(output, record)
    print(json.dumps(dict(recording_id=spec["id"],status=record["status"],sent_audio_s=record["sent_audio_s"],
                          connection_duration_s=record["connection_duration_s"], errors=record["errors"])), flush=True)
    return record["status"] == "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--inputs", type=Path, default=Path("benchmarks/asr/issue441-inputs.json"))
    parser.add_argument("--runtime", type=Path, default=Path(".tmp/issue441/nemotron-runtime/runtime"))
    parser.add_argument("--model", type=Path, default=Path(".tmp/issue441/nemotron-runtime/model.gguf"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/issue441/runs/N"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--recording")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--frozen-sha256")
    args = parser.parse_args()
    args.root = args.root.resolve()
    for key in ("inputs", "runtime", "model", "output"):
        value = getattr(args, key)
        if not value.is_absolute():
            setattr(args, key, args.root/value)
    for key, value in {"HF_HUB_OFFLINE":"1", "GGML_VK_DISABLE_GRAPH_OPTIMIZE":"1", "NEMO_SPEECH_SCHEDULE_CAPTURE":"1"}.items():
        os.environ[key] = value
    if args.device < 0:
        parser.error("explicit nonnegative Vulkan device ordinal required")
    if args.smoke:
        specs = [dict(id="synthetic-silence", language=None)]
    else:
        if not args.frozen_sha256 or digest(args.inputs) != args.frozen_sha256:
            parser.error("Director-confirmed frozen manifest SHA-256 required")
        manifest = json.loads(args.inputs.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            parser.error("unsupported input schema")
        specs = manifest["recordings"]
        if args.recording:
            specs = [spec for spec in specs if spec["id"] == args.recording]
            if not specs:
                parser.error("unknown recording")
    if args.recording or args.smoke:
        return 0 if run_record(args, specs[0]) else 1
    failed = False
    for spec in specs:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/(spec["id"]+".json")
        if path.exists():
            raise FileExistsError(f"refusing to replace previous execution: {path}")
        with (args.output/(spec["id"]+".runtime.log")).open("w",encoding="utf-8") as log:
            try:
                result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                    *sys.argv[1:], "--recording", spec["id"]], stdout=log, stderr=subprocess.STDOUT,
                    timeout=spec["duration_s"]+180)
                failure = None if result.returncode == 0 else f"worker exit {result.returncode}"
            except subprocess.TimeoutExpired as error:
                failure = str(error)
        if failure:
            failed = True
            record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(
                schema_version=1,arm="N",recording_id=spec["id"],metadata={},
                source_origin_mono_s=None,events=[],sends=[],observations=[],errors=[],sent_audio_s=0,
                connection_duration_s=None)
            record["status"] = "failed"
            record["errors"].append(dict(type="ProcessFailure",message=failure))
            save(path,record)
        print(json.dumps(dict(recording_id=spec["id"],failure=failure)),flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
