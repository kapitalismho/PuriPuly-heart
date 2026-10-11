from __future__ import annotations

import ctypes as ct
import json
import os
import sys
from pathlib import Path

import numpy as np

from .artifacts import inspect_artifact, validate_artifact_provenance, validate_runtime
from .protocol import PROFILES, encode, read_message

MAX_FRAMES = 1024


class ModelConfig(ct.Structure):
    _fields_ = [
        ("size", ct.c_size_t),
        ("model_path", ct.c_char_p),
        ("gpu", ct.c_int32),
        ("preset", ct.c_char_p),
        ("chunk_frames", ct.c_int32),
        ("right_context_frames", ct.c_int32),
        ("left_context_frames", ct.c_int32),
        ("fifo_frames", ct.c_int32),
        ("spkcache_frames", ct.c_int32),
        ("update_period_frames", ct.c_int32),
    ]


class NativeSession:
    def __init__(self, artifact_path: Path, runtime_dir: Path, device: int, profile: str):
        if device < 0 or profile not in PROFILES:
            raise ValueError("explicit Vulkan device and official profile required")
        runtime = validate_runtime(runtime_dir)
        artifact = inspect_artifact(artifact_path)
        validate_artifact_provenance(artifact_path, artifact)
        self.directory = os.add_dll_directory(str((runtime_dir / "bin").resolve()))
        self.dll = ct.CDLL(
            str((runtime_dir / "bin/nemo_speech_asr_c.dll").resolve()), winmode=0x1100
        )
        signatures = {
            "nemo_speech_diar_create": (
                [ct.POINTER(ModelConfig), ct.POINTER(ct.c_void_p)],
                ct.c_int,
            ),
            "nemo_speech_diar_destroy": ([ct.c_void_p], None),
            "nemo_speech_diar_stream_open": ([ct.c_void_p, ct.POINTER(ct.c_void_p)], ct.c_int),
            "nemo_speech_diar_stream_close": ([ct.c_void_p], None),
            "nemo_speech_diar_stream_push_f32": (
                [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_size_t, ct.c_int32],
                ct.c_int,
            ),
            "nemo_speech_diar_num_speakers": ([ct.c_void_p], ct.c_int32),
            "nemo_speech_diar_seconds_per_frame": ([ct.c_void_p], ct.c_double),
            "puripuly_nemotron_consumed_samples": ([ct.c_void_p], ct.c_int64),
            "puripuly_nemotron_immutable_frames": ([ct.c_void_p], ct.c_int64),
            "puripuly_nemotron_drain": (
                [
                    ct.c_void_p,
                    ct.POINTER(ct.c_float),
                    ct.c_size_t,
                    ct.POINTER(ct.c_int64),
                    ct.POINTER(ct.c_size_t),
                ],
                ct.c_int,
            ),
            "nemo_speech_asr_last_error": ([], ct.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = arguments, result
        self.model = ct.c_void_p()
        self.stream = ct.c_void_p()
        self.buffer = np.empty((MAX_FRAMES, 8), dtype=np.float32)
        self.cursor = 0
        self.epoch = 0
        config = ModelConfig(
            ct.sizeof(ModelConfig),
            str(artifact_path.resolve()).encode("utf-8"),
            device,
            None,
            *PROFILES[profile],
        )
        try:
            self.check(self.dll.nemo_speech_diar_create(ct.byref(config), ct.byref(self.model)))
            if (
                self.dll.nemo_speech_diar_num_speakers(self.model) != 8
                or abs(self.dll.nemo_speech_diar_seconds_per_frame(self.model) - 0.01) > 1e-9
            ):
                raise ValueError("native artifact capacity/cadence mismatch")
            self.reset(0)
        except BaseException:
            self.close()
            raise
        self.ready = {
            "tail_padding_samples": (PROFILES[profile][0] + PROFILES[profile][1]) * 1280 + 512,
            "max_revision_samples": (PROFILES[profile][0] + PROFILES[profile][1]) * 1280 + 512,
            "diagnostics": [
                ["backend", "Vulkan"],
                ["device_ordinal", str(device)],
                ["profile", profile],
                ["geometry", "/".join(str(v) for v in PROFILES[profile])],
                ["artifact_sha256", artifact["sha256"]],
                ["tensor_formats", json.dumps(artifact["tensor_formats"], sort_keys=True)],
                ["source_revision", runtime["source"]["revision"]],
                ["neural_cpu_fallback", "forbidden"],
                ["probability_frontier", "immutable emitted probabilities, not identity certainty"],
                ["processed_cursor", "consumed mel hop prefix; excludes submitted lookahead"],
                [
                    "accumulation",
                    "Vulkan shader selected; F32 auxiliary tensors/activations; not all-F16",
                ],
            ],
        }

    def check(self, status: int) -> None:
        if status != 0:
            raw = self.dll.nemo_speech_asr_last_error()
            raise RuntimeError(
                raw.decode("utf-8", errors="replace") if raw else f"native error {status}"
            )

    def reset(self, epoch: int) -> None:
        if self.stream:
            self.dll.nemo_speech_diar_stream_close(self.stream)
            self.stream = ct.c_void_p()
        self.check(self.dll.nemo_speech_diar_stream_open(self.model, ct.byref(self.stream)))
        self.cursor, self.epoch = 0, epoch

    def feed(self, payload: bytes, start: int):
        if start != self.cursor or len(payload) % 4 or len(payload) > 64000 or not payload:
            raise ValueError("noncontiguous/oversized Nemotron input")
        pcm = np.frombuffer(payload, dtype="<f4")
        if not np.isfinite(pcm).all():
            raise ValueError("nonfinite input PCM")
        self.check(
            self.dll.nemo_speech_diar_stream_push_f32(
                self.stream, pcm.ctypes.data_as(ct.POINTER(ct.c_float)), pcm.size, 16000
            )
        )
        self.cursor += pcm.size
        return self.drain()

    def drain(self):
        first, count = ct.c_int64(), ct.c_size_t()
        self.check(
            self.dll.puripuly_nemotron_drain(
                self.stream,
                self.buffer.ctypes.data_as(ct.POINTER(ct.c_float)),
                MAX_FRAMES,
                ct.byref(first),
                ct.byref(count),
            )
        )
        processed = self.dll.puripuly_nemotron_consumed_samples(self.stream)
        immutable = self.dll.puripuly_nemotron_immutable_frames(self.stream) * 160
        if count.value > MAX_FRAMES or not 0 <= immutable <= processed <= self.cursor:
            raise RuntimeError("invalid native probability frontier")
        selected = self.buffer[: count.value]
        if not np.isfinite(selected).all() or (selected < 0).any() or (selected > 1).any():
            raise RuntimeError("invalid native probabilities")
        return {
            "first_frame": first.value,
            "count": count.value,
            "processed": processed,
            "immutable": immutable,
            "epoch": self.epoch,
        }, selected.tobytes()

    def close(self):
        if self.stream:
            self.dll.nemo_speech_diar_stream_close(self.stream)
            self.stream = ct.c_void_p()
        if self.model:
            self.dll.nemo_speech_diar_destroy(self.model)
            self.model = ct.c_void_p()
        self.directory.close()


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("Nemotron native worker requires Windows Vulkan")
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    session = None
    try:
        while True:
            request = read_message(sys.stdin.buffer)
            if request is None:
                break
            metadata, payload = request
            ident = metadata.get("id")
            try:
                operation = metadata["op"]
                if operation == "start":
                    if session is not None:
                        raise ValueError("worker already started")
                    session = NativeSession(
                        Path(metadata["artifact_path"]),
                        Path(metadata["runtime_dir"]),
                        metadata["device"],
                        metadata["profile"],
                    )
                    response, data = session.ready, b""
                elif operation == "stop":
                    output.write(encode({"id": ident, "ok": True}))
                    break
                elif session is None:
                    raise ValueError("worker not started")
                elif operation == "reset":
                    session.reset(metadata["epoch"])
                    response, data = {}, b""
                elif metadata.get("epoch") != session.epoch:
                    raise ValueError("stale Nemotron epoch")
                elif operation == "feed":
                    response, data = session.feed(payload, metadata["start"])
                elif operation == "pause":
                    response, data = session.drain()
                else:
                    raise ValueError("unknown Nemotron operation")
                output.write(encode({"id": ident, "ok": True, **response}, data))
            except Exception as error:
                output.write(encode({"id": ident, "ok": False, "error": str(error)[:2048]}))
                break
    finally:
        if session is not None:
            session.close()
        output.close()


if __name__ == "__main__":
    main()
