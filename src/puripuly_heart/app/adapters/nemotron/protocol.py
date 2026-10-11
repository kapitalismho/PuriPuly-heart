from __future__ import annotations

import json
import struct

MAX_METADATA = 8192
MAX_PAYLOAD = 65536
HEADER = struct.Struct("<II")
PROFILES = {"1.04": (9, 4, 0, 264, 264, 222), "0.64": (6, 2, 0, 264, 264, 222)}


def encode(metadata: dict, payload: bytes = b"") -> bytes:
    encoded = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_METADATA or len(payload) > MAX_PAYLOAD:
        raise ValueError("Nemotron IPC message exceeds capacity")
    return HEADER.pack(len(encoded), len(payload)) + encoded + payload


def decode_header(header: bytes) -> tuple[int, int]:
    if len(header) != HEADER.size:
        raise ValueError("truncated Nemotron IPC header")
    sizes = HEADER.unpack(header)
    if sizes[0] > MAX_METADATA or sizes[1] > MAX_PAYLOAD:
        raise ValueError("Nemotron IPC response exceeds capacity")
    return sizes


def read_message(stream) -> tuple[dict, bytes] | None:
    header = stream.read(HEADER.size)
    if not header:
        return None
    metadata_size, payload_size = decode_header(header)
    metadata = stream.read(metadata_size)
    payload = stream.read(payload_size)
    if len(metadata) != metadata_size or len(payload) != payload_size:
        raise ValueError("truncated Nemotron IPC message")
    return json.loads(metadata), payload
