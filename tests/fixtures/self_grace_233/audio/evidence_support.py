from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def plain(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, (UUID, Path)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(
        json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class EvidenceState:
    def __init__(self, report, *, model_root=None):
        if Path(report).name != report or not report.endswith(".json"):
            raise ValueError("Evidence report must be a task-local JSON filename")
        if (HERE / report).exists():
            raise FileExistsError("Retained reports are immutable; choose a new report filename")
        self.report = report
        self.directory = ROOT / ".venv" / "self-stream-233" / Path(report).stem
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model_root = None if model_root is None else Path(model_root).resolve()
        self.original_home = Path.home()
        self.environment = dict(os.environ)
        for key, child in (
            ("LOCALAPPDATA", "local"),
            ("APPDATA", "roaming"),
            ("USERPROFILE", "profile"),
            ("HOME", "profile"),
            ("TEMP", "temp"),
            ("TMP", "temp"),
        ):
            directory = self.directory / child
            directory.mkdir(exist_ok=True)
            self.environment[key] = str(directory)
        self.environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "PURIPULY_HEART_SECRETS_PASSPHRASE": "invented-task-local-not-a-user-secret-233",
            }
        )

    @contextmanager
    def activate(self):
        with patch.dict(os.environ, self.environment, clear=True):
            yield

    def sanitize(self, value):
        replacements = [(self.directory, "<isolated-state>")]
        if self.model_root is not None:
            replacements.append((self.model_root, "<model-root>"))
        replacements.extend(((ROOT, "<checkout>"), (self.original_home, "<user-home>")))

        def transform(item):
            if isinstance(item, dict):
                return {key: transform(val) for key, val in item.items()}
            if isinstance(item, list):
                return [transform(val) for val in item]
            if isinstance(item, str):
                for source, replacement in replacements:
                    for spelling in (str(source), str(source).replace("\\", "/")):
                        item = item.replace(spelling, replacement)
                return item
            return item

        return transform(plain(value))

    def export(self, filename, value):
        target = HERE / filename
        if target.exists():
            raise FileExistsError("Retained exports are immutable; choose a new report filename")
        raw = self.directory / filename
        save(raw, value)
        exported = self.sanitize(value)
        exported["trace_export"] = {
            "kind": "sanitized export, not original local bytes",
            "original_local_sha256": digest(raw),
            "redactions": "Personal checkout/model/state/home paths only; executed source/input/model/profile hashes and observation values retained",
            "original_local_location": "<isolated-state>/" + filename,
        }
        save(target, exported)
        return exported


def install_network_guard():
    def audit(event, args):
        if event not in {"socket.connect", "socket.sendto", "socket.getaddrinfo"}:
            return
        address = (
            args[-1]
            if event == "socket.sendto"
            else args[1] if event == "socket.connect" else (args[0], args[1])
        )
        if not isinstance(address, tuple) or len(address) < 2:
            raise PermissionError("Task probe permits loopback network addresses only")
        host, port = address[:2]
        if host not in ("localhost", None):
            try:
                allowed = ipaddress.ip_address(host).is_loopback
            except ValueError:
                allowed = False
            if not allowed:
                raise PermissionError("Task probe forbids non-loopback network access")
        if port in (9000, 9001, "9000", "9001"):
            raise PermissionError("Task probe forbids default VRChat OSC ports")

    sys.addaudithook(audit)
