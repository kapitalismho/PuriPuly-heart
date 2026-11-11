from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_SCHEMA = "puripuly-heart/legacy-pyinstaller-ownership/v1"
NATIVE_SCHEMA = "puripuly-heart/native-artifact-manifest/v1"
PLAN_SCHEMA = "puripuly-heart/native-installer-cleanup/v2"
PRUNING_SCHEMA = "puripuly-heart/native-packaging-pruning/v1"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_RESERVED = re.compile(
    r"(?:CON|CONIN\$|CONOUT\$|CLOCK\$|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])(?: *\.| *$)",
    re.IGNORECASE,
)


def normalize_windows_relative(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ownership paths must be nonempty strings")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or part[-1] in " ."
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or _RESERVED.match(part)
        ):
            raise ValueError(f"unsafe Windows relative ownership path: {value!r}")
    return normalized


def preserved_user_path(value: str) -> bool:
    root = normalize_windows_relative(value).split("/", 1)[0].casefold()
    return (
        root.split(".", 1)[0] in {"settings", "secrets", "prompts", "extensions", "models"}
        or root == ".env"
        or root.startswith(".env.")
    )


@dataclass(frozen=True)
class OwnedFile:
    path: str
    bytes: int
    sha256: str

    @property
    def key(self) -> str:
        return self.path.casefold()

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


def read_inventory(payload: dict[str, Any], schema: str) -> dict[str, OwnedFile]:
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected manifest schema {schema}")
    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("manifest inventory must be a nonempty list")
    return _read_owned_files(inventory)


def _read_owned_files(inventory: list[Any]) -> dict[str, OwnedFile]:
    result: dict[str, OwnedFile] = {}
    for row in inventory:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("inventory entries must contain exactly path, bytes and sha256")
        path = normalize_windows_relative(row["path"])
        size = row["bytes"]
        digest = row["sha256"]
        if type(size) is not int or not 0 <= size <= 2**63 - 1:
            raise ValueError(f"invalid inventory size for {path}")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid SHA256 for {path}")
        owned = OwnedFile(path, size, digest.lower())
        if owned.key in result:
            raise ValueError(f"case-insensitive duplicate inventory path: {path}")
        result[owned.key] = owned
    for key in result:
        parts = key.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in result:
                raise ValueError(f"inventory file is also a directory ancestor: {key}")
    return result


def obsolete_owned_files(legacy: dict[str, Any], native: dict[str, Any]) -> tuple[OwnedFile, ...]:
    old = read_inventory(legacy, LEGACY_SCHEMA)
    new = read_inventory(native, NATIVE_SCHEMA)
    obsolete = {
        key: row for key, row in old.items() if key not in new and not preserved_user_path(row.path)
    }
    for key in new:
        parts = key.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in obsolete:
                raise ValueError(f"obsolete file conflicts with a new directory: {key}")
    for key in obsolete:
        parts = key.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in new:
                raise ValueError(f"new file conflicts with an obsolete directory: {key}")
    return tuple(obsolete[key] for key in sorted(obsolete))


def _packaging_obsolete_files(
    payload: dict[str, Any], native: dict[str, OwnedFile]
) -> dict[str, OwnedFile]:
    if not isinstance(payload, dict) or payload.get("schema") != PRUNING_SCHEMA:
        raise ValueError(f"expected packaging evidence schema {PRUNING_SCHEMA}")
    inventories = {}
    for field in ("removed_files", "obsolete_files"):
        rows = payload.get(field)
        if not isinstance(rows, list):
            raise ValueError(f"packaging evidence {field} must be a list")
        inventories[field] = _read_owned_files(rows)
    removed = inventories["removed_files"]
    obsolete = inventories["obsolete_files"]
    for key, row in removed.items():
        counterpart = obsolete.get(key)
        if counterpart is None or (counterpart.bytes, counterpart.sha256) != (
            row.bytes,
            row.sha256,
        ):
            raise ValueError(f"removed file lacks matching obsolete ownership: {row.path}")
    scripts_dlls = {
        f"site-packages/scripts/{name}"
        for name in ("onnxruntime.dll", "sherpa-onnx-c-api.dll", "sherpa-onnx-cxx-api.dll")
    }
    old_preloads = {
        f"_runtime/local_qwen/{name}": f"site-packages/onnxruntime/capi/{name}"
        for name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll")
    }
    for key, row in obsolete.items():
        if key in native:
            raise ValueError(f"packaging obsolete file is in the current artifact: {row.path}")
        if key in removed:
            parts = key.split("/")
            dependency_test = (
                len(parts) >= 4
                and parts[0] == "site-packages"
                and parts[1] in {"numpy", "scipy"}
                and "tests" in parts[2:-1]
            )
            if key not in scripts_dlls and not dependency_test:
                raise ValueError(f"unsupported packaging removal path: {row.path}")
            if key in scripts_dlls:
                counterpart = native.get(f"site-packages/sherpa_onnx/lib/{parts[-1]}")
                if counterpart is None or (row.bytes, row.sha256) != (
                    counterpart.bytes,
                    counterpart.sha256,
                ):
                    raise ValueError(f"unproven retained sherpa DLL identity: {row.path}")
        else:
            counterpart = native.get(old_preloads.get(key, ""))
            if counterpart is None or (row.bytes, row.sha256) != (
                counterpart.bytes,
                counterpart.sha256,
            ):
                raise ValueError(f"unproven obsolete preload identity: {row.path}")
    return obsolete


def cleanup_plan(
    legacy_bytes: bytes, native_bytes: bytes, packaging_pruning_bytes: bytes | None = None
) -> dict[str, Any]:
    legacy = json.loads(legacy_bytes)
    native = json.loads(native_bytes)
    old = read_inventory(legacy, LEGACY_SCHEMA)
    new = read_inventory(native, NATIVE_SCHEMA)
    alternatives: dict[str, OwnedFile] = {}
    if packaging_pruning_bytes is not None:
        packaging = _packaging_obsolete_files(json.loads(packaging_pruning_bytes), new)
        for key, row in packaging.items():
            if key in old and (old[key].bytes, old[key].sha256) != (row.bytes, row.sha256):
                alternatives[key] = row
            elif key not in old:
                old[key] = row
    merged = {"schema": LEGACY_SCHEMA, "inventory": [row.as_dict() for row in old.values()]}
    owned = obsolete_owned_files(merged, native)
    preserved = [
        old[key].path
        for key in sorted(old)
        if key not in new and preserved_user_path(old[key].path)
    ]
    directories: set[str] = set()
    for row in owned:
        parts = row.path.split("/")
        directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    inventory = []
    for row in owned:
        item = row.as_dict()
        if row.key in alternatives:
            alternative = alternatives[row.key]
            item["alternative_identities"] = [
                {"bytes": alternative.bytes, "sha256": alternative.sha256}
            ]
        inventory.append(item)
    plan = {
        "schema": PLAN_SCHEMA,
        "legacy_manifest_sha256": hashlib.sha256(legacy_bytes).hexdigest(),
        "native_manifest_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "inventory": inventory,
        "directories": sorted(directories, key=lambda path: (-path.count("/"), path.casefold())),
        "preserved_paths": preserved,
    }
    if packaging_pruning_bytes is not None:
        plan["packaging_pruning_sha256"] = hashlib.sha256(packaging_pruning_bytes).hexdigest()
    return plan


def render_include(plan: dict[str, Any]) -> str:
    serialized = json.dumps(plan, ensure_ascii=True, separators=(",", ":"))
    chunks = [serialized[index : index + 200] for index in range(0, len(serialized), 200)]
    lines = [
        '#if GetSHA256OfFile(MyPackagedAppDir + "\\native-artifact-manifest.json") != "'
        + plan["native_manifest_sha256"]
        + '"',
        "  #error Native cleanup include does not match the final packaged native artifact manifest. Regenerate it after finalizing the payload.",
        "#endif",
        "procedure WriteNativeCleanupPlan(const FileName: String);",
        "var",
        "  Plan: String;",
        "begin",
        "  Plan := '';",
    ]
    lines.extend("  Plan := Plan + '" + chunk.replace("'", "''") + "';" for chunk in chunks)
    lines.extend(
        [
            "  if not SaveStringToFile(FileName, Plan, False) then begin",
            "    RaiseException('Cannot write native cleanup ownership plan.');",
            "  end;",
            "end;",
            "",
        ]
    )
    return "\n".join(lines)


def generate_include(
    legacy_manifest: Path,
    native_manifest: Path,
    output: Path,
    packaging_pruning: Path | None = None,
) -> dict[str, Any]:
    inputs = {legacy_manifest.resolve(), native_manifest.resolve()}
    if packaging_pruning is not None:
        inputs.add(packaging_pruning.resolve())
    if output.resolve() in inputs:
        raise ValueError("cleanup output must not overwrite an input manifest or evidence")
    plan = cleanup_plan(
        legacy_manifest.read_bytes(),
        native_manifest.read_bytes(),
        packaging_pruning.read_bytes() if packaging_pruning is not None else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_include(plan), encoding="utf-8")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", required=True, type=Path)
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--packaging-pruning", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = generate_include(
        args.legacy_manifest, args.native_manifest, args.output, args.packaging_pruning
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "obsolete_files": len(plan["inventory"]),
                "preserved_paths": plan["preserved_paths"],
                "native_manifest_sha256": plan["native_manifest_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
