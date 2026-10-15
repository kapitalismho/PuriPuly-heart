from __future__ import annotations

import struct
from pathlib import Path


def _version_node(key: str, value: bytes, children: bytes = b"", *, text: bool = False) -> bytes:
    payload = bytearray(struct.pack("<HHH", 0, len(value) // 2 if text else len(value), int(text)))
    payload.extend((key + "\0").encode("utf-16le"))
    payload.extend(b"\0" * (-len(payload) % 4))
    payload.extend(value)
    if children:
        payload.extend(b"\0" * (-len(payload) % 4))
        payload.extend(children)
    struct.pack_into("<H", payload, 0, len(payload))
    return bytes(payload)


def write_pe(
    path: Path,
    imports: tuple[str, ...] = (),
    delay_imports: tuple[str, ...] = (),
    *,
    version: str = "14.44.35211.0",
    company: str = "Microsoft Corporation",
    debug: bool = False,
    machine: int = 0x8664,
    metadata: dict[str, str] | None = None,
) -> Path:
    payload = bytearray(0x2200)
    payload[:2] = b"MZ"
    struct.pack_into("<I", payload, 0x3C, 0x80)
    payload[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", payload, 0x84, machine, 1, 0, 0, 0, 240, 0x2022)
    optional = 0x98
    struct.pack_into("<H", payload, optional, 0x20B)
    struct.pack_into("<Q", payload, optional + 24, 0x180000000)
    struct.pack_into("<II", payload, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", payload, optional + 56, 0x3000, 0x200)
    struct.pack_into("<H", payload, optional + 68, 3)
    struct.pack_into("<I", payload, optional + 108, 16)
    struct.pack_into(
        "<8sIIIIIIHHI",
        payload,
        optional + 240,
        b".rdata\0\0",
        0x2000,
        0x1000,
        0x2000,
        0x200,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    cursor = 0x200

    def allocate(data: bytes) -> tuple[int, int]:
        nonlocal cursor
        cursor = (cursor + 7) & ~7
        offset = cursor
        payload[offset : offset + len(data)] = data
        cursor += len(data)
        return offset, offset - 0x200 + 0x1000

    for names, directory, width in ((imports, 1, 20), (delay_imports, 13, 32)):
        if not names:
            continue
        offset, rva = allocate(bytes(width * (len(names) + 1)))
        struct.pack_into(
            "<II", payload, optional + 112 + directory * 8, rva, width * (len(names) + 1)
        )
        for index, name in enumerate(names):
            _, name_rva = allocate(name.encode("ascii") + b"\0")
            _, thunk_rva = allocate(struct.pack("<QQ", 0x8000000000000001, 0))
            if directory == 1:
                struct.pack_into(
                    "<IIIII", payload, offset + index * width, thunk_rva, 0, 0, name_rva, thunk_rva
                )
            else:
                struct.pack_into(
                    "<IIIIIIII",
                    payload,
                    offset + index * width,
                    1,
                    name_rva,
                    0,
                    thunk_rva,
                    thunk_rva,
                    0,
                    0,
                    0,
                )
    major, minor, build, revision = (int(part) for part in version.split("."))
    fixed = struct.pack(
        "<13I",
        0xFEEF04BD,
        0x10000,
        major << 16 | minor,
        build << 16 | revision,
        major << 16 | minor,
        build << 16 | revision,
        0x3F,
        int(debug),
        0x40004,
        2,
        0,
        0,
        0,
    )
    fields = {"CompanyName": company, **(metadata or {})}
    entries = bytearray()
    for key, value in fields.items():
        entries.extend(_version_node(key, (value + "\0").encode("utf-16le"), text=True))
        entries.extend(b"\0" * (-len(entries) % 4))
    table = _version_node("040904b0", b"", entries, text=True)
    strings = _version_node("StringFileInfo", b"", table, text=True)
    version_info = _version_node("VS_VERSION_INFO", fixed, strings)
    resource_offset, resource_rva = allocate(bytes(88))
    _, value_rva = allocate(version_info)
    for index in range(3):
        struct.pack_into("<IIHHHH", payload, resource_offset + index * 24, 0, 0, 0, 0, 0, 1)
    struct.pack_into("<II", payload, resource_offset + 16, 16, 0x80000000 | 24)
    struct.pack_into("<II", payload, resource_offset + 40, 1, 0x80000000 | 48)
    struct.pack_into("<II", payload, resource_offset + 64, 1033, 72)
    struct.pack_into("<IIII", payload, resource_offset + 72, value_rva, len(version_info), 0, 0)
    struct.pack_into("<II", payload, optional + 112 + 2 * 8, resource_rva, cursor - resource_offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path
