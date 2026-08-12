from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

DETAIL_SHARD_LIMIT_BYTES = 20 * 1024**2
AGGREGATE_JSON_LIMIT_BYTES = 10 * 1024**2
MAX_SINGLE_ROW_PLAIN_BYTES = 8 * 1024 * 1024


class Phase5StorageError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def framed_digest(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = canonical_json(row).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def rows_sha256(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def deterministic_gzip(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def encode_row(row: Any) -> bytes:
    encoded = canonical_json(row).encode("utf-8") + b"\n"
    if len(encoded) > MAX_SINGLE_ROW_PLAIN_BYTES:
        raise Phase5StorageError("single row exceeds plain byte guard")
    return encoded


def _split_rows(
    encoded_rows: list[bytes],
    shard_limit: int = DETAIL_SHARD_LIMIT_BYTES,
) -> list[list[bytes]]:
    if not encoded_rows:
        return []
    chunk_count = max(1, len(encoded_rows))
    pending: list[list[bytes]] = [[]]
    plain_size = 0
    for row in encoded_rows:
        if plain_size + len(row) > shard_limit * 4:
            pending.append([])
            plain_size = 0
        pending[-1].append(row)
        plain_size += len(row)
    result: list[list[bytes]] = []
    while pending:
        chunk = pending.pop(0)
        compressed = deterministic_gzip(b"".join(chunk))
        if len(compressed) > shard_limit and len(chunk) > 1:
            middle = len(chunk) // 2
            pending.insert(0, chunk[middle:])
            pending.insert(0, chunk[:middle])
            continue
        if len(compressed) > shard_limit:
            raise Phase5StorageError("single row exceeds compressed shard limit")
        result.append(chunk)
    return result


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RepresentationWriter:
    def __init__(
        self,
        directory: Path,
        representation_name: str,
        field_order: list[str] | None = None,
        shard_limit: int = DETAIL_SHARD_LIMIT_BYTES,
    ) -> None:
        self.directory = Path(directory)
        self.representation_name = representation_name
        self.field_order = list(field_order) if field_order is not None else None
        self.shard_limit = shard_limit
        self._keys: list[str] = []
        self._encoded: list[bytes] = []

    def add_rows(self, keyed_rows: Iterable[tuple[str, Any]]) -> None:
        for key, row in keyed_rows:
            if self._keys and key < self._keys[-1]:
                raise Phase5StorageError(
                    f"{self.representation_name}: key order drift ({self._keys[-1]} > {key})"
                )
            self._keys.append(key)
            self._encoded.append(encode_row(row))

    @property
    def row_count(self) -> int:
        return len(self._keys)

    def write(self) -> dict[str, Any]:
        chunks = _split_rows(self._encoded, self.shard_limit)
        self.directory.mkdir(parents=True, exist_ok=True)
        receipts: list[dict[str, Any]] = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            plain = b"".join(chunk)
            compressed = deterministic_gzip(plain)
            filename = f"{self.representation_name}.{index:04d}.jsonl.gz"
            final_path = self.directory / filename
            temporary_path = self.directory / (filename + ".tmp")
            with temporary_path.open("wb") as handle:
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
            if len(compressed) > self.shard_limit:
                temporary_path.unlink(missing_ok=True)
                raise Phase5StorageError(
                    f"{self.representation_name} shard {index} exceeds limit"
                )
            stored = temporary_path.read_bytes()
            if stored != compressed:
                temporary_path.unlink(missing_ok=True)
                raise Phase5StorageError(
                    f"{self.representation_name} shard {index} readback mismatch"
                )
            os.replace(temporary_path, final_path)
            first_key = self._keys[cursor]
            last_key = self._keys[cursor + len(chunk) - 1]
            rolling = hashlib.sha256()
            for row in chunk:
                rolling.update(row)
            receipt = {
                "shard_index": index,
                "filename": filename,
                "row_count": len(chunk),
                "first_key": first_key,
                "last_key": last_key,
                "rolling_content_sha256": rolling.hexdigest(),
                "compressed_byte_sha256": sha256_bytes(compressed),
                "compressed_byte_count": len(compressed),
                "size_bytes": len(compressed),
            }
            receipts.append(receipt)
            cursor += len(chunk)
        if cursor != len(self._keys):
            raise Phase5StorageError(
                f"{self.representation_name} shard row accounting drift"
            )
        total_compressed = sum(int(row["compressed_byte_count"]) for row in receipts)
        result = {
            "representation": self.representation_name,
            "field_order": self.field_order,
            "row_count": len(self._keys),
            "shard_count": len(receipts),
            "shards": receipts,
            "rows_sha256": sha256_bytes(
                b"".join(self._encoded) if self._encoded else b""
            ),
            "total_compressed_bytes": total_compressed,
        }
        self._keys = []
        self._encoded = []
        return result


def read_shard_rows(
    shard_path: Path,
    field_order: list[str] | None = None,
) -> list[Any]:
    compressed = shard_path.read_bytes()
    plain = gzip.decompress(compressed)
    rows: list[Any] = []
    for line in plain.splitlines():
        if not line:
            continue
        parsed = json.loads(line)
        if field_order is not None and isinstance(parsed, list):
            if len(parsed) != len(field_order):
                raise Phase5StorageError(
                    f"{shard_path.name}: field count drift {len(parsed)} != {len(field_order)}"
                )
            parsed = dict(zip(field_order, parsed))
        rows.append(parsed)
    return rows


def read_representation(
    directory: Path,
    receipt: dict[str, Any],
) -> list[Any]:
    rows: list[Any] = []
    for shard in receipt["shards"]:
        path = directory / str(shard["filename"])
        if not path.is_file():
            raise Phase5StorageError(f"shard file missing: {path}")
        if sha256_file(path) != shard["compressed_byte_sha256"]:
            raise Phase5StorageError(f"shard byte hash drift: {path}")
        shard_rows = read_shard_rows(path, receipt.get("field_order"))
        if len(shard_rows) != int(shard["row_count"]):
            raise Phase5StorageError(f"shard row count drift: {path}")
        rolling = hashlib.sha256()
        for row in shard_rows:
            rolling.update(encode_row(row))
        if rolling.hexdigest() != shard["rolling_content_sha256"]:
            raise Phase5StorageError(f"shard rolling hash drift: {path}")
        rows.extend(shard_rows)
    if len(rows) != int(receipt["row_count"]):
        raise Phase5StorageError(f"representation row count drift: {receipt['representation']}")
    return rows


def verify_shard_receipts(directory: Path, receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    total_bytes = 0
    for receipt in receipts:
        path = directory / str(receipt["filename"])
        if not path.is_file():
            raise Phase5StorageError(f"shard file missing: {path}")
        if path.stat().st_size != int(receipt["size_bytes"]):
            raise Phase5StorageError(f"shard size drift: {path}")
        if sha256_file(path) != receipt["compressed_byte_sha256"]:
            raise Phase5StorageError(f"shard byte hash drift: {path}")
        total_rows += int(receipt["row_count"])
        total_bytes += int(receipt["compressed_byte_count"])
    return {"row_count": total_rows, "compressed_byte_count": total_bytes}


def default_temp_root() -> Path:
    return Path(os.environ.get("TEMP") or tempfile.gettempdir())


def phase5_cache_root() -> Path:
    return default_temp_root() / "puripuly_stb_phase5" / "turn_episode_v1"
