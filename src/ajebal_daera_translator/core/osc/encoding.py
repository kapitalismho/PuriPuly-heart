from __future__ import annotations

import struct
from typing import Iterable, TypeAlias

OscArg: TypeAlias = str | int | float | bool


def _pad4(data: bytes) -> bytes:
    padding = (-len(data)) % 4
    if not padding:
        return data
    return data + (b"\0" * padding)


def encode_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\0"
    return _pad4(raw)


def encode_message(address: str, args: Iterable[OscArg]) -> bytes:
    if not address or not address.startswith("/"):
        raise ValueError("OSC address must start with '/'")

    args_list = list(args)
    type_tags = [","]
    encoded_args: list[bytes] = []

    for arg in args_list:
        if isinstance(arg, bool):
            type_tags.append("T" if arg else "F")
            continue

        if isinstance(arg, int):
            type_tags.append("i")
            encoded_args.append(struct.pack(">i", arg))
            continue

        if isinstance(arg, float):
            type_tags.append("f")
            encoded_args.append(struct.pack(">f", arg))
            continue

        if isinstance(arg, str):
            type_tags.append("s")
            encoded_args.append(encode_string(arg))
            continue

        raise TypeError(f"Unsupported OSC arg type: {type(arg)!r}")

    header = encode_string(address) + encode_string("".join(type_tags))
    return header + b"".join(encoded_args)
