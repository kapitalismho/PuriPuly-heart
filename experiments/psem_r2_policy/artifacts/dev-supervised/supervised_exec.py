"""Supervised single-attempt executor for the canonical PSEM R2 DEV cases.

Prep artifact owned by the supervised-execution owner (artifacts/dev-supervised/).

Invariants:
- Exactly one child process per invocation. Never retries, never re-launches.
- The child's stdout is redirected by the OS straight into a unique, exclusively
  created case file; payloads larger than 600 MB never pass through this process.
- stderr is read as it arrives, preserved byte-for-byte in its own file, and
  echoed to this driver's console so fatal startup/import/auth errors are
  visible immediately.
- No deadline of any kind is imposed on the child: one Popen + one wait() with
  no timeout, no alarm, no signal, no kill path except an explicit operator
  interrupt of the supervising process.
- The canonical journal is only ever read (lock-free, atomic-replace safe). It
  is never written, never locked, never reset.
- Existing artifacts are never overwritten: case dirs are created with
  exist_ok=False and case files with exclusive mode "xb".
- Calling the paid canonical path requires --ack-go DIRECTOR_GO.

Modes:
  canonical --meeting M --ack-go DIRECTOR_GO   preflight gate, canonical paid
                                               launch, per-case preservation,
                                               objective pacing report
  preflight                                    read-only gate + capsule reuse
                                               validation for the five DEV
                                               meetings (no providers)
  standin --label L [...]                      zero-cost exercise of the same
                                               preservation/record path

Usage (from the target worktree root):
  .venv/Scripts/python.exe experiments/psem_r2_policy/artifacts/dev-supervised/supervised_exec.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

TARGET = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls"
)
PACKAGE = TARGET / "experiments/psem_r2_policy"
ARTIFACTS = PACKAGE / "artifacts"
HOME_DIR = ARTIFACTS / "dev-supervised"
CASES_DIR = HOME_DIR / "cases"
READONLY_WRAPPERS = ARTIFACTS / "excluded-launcher-shadowing/wrappers"
GATE = READONLY_WRAPPERS / "prepaid_gate.py"
CASE_TOOLS = READONLY_WRAPPERS / "case_tools.py"
PACING_PROBE = ARTIFACTS / "pacing-probe/pacing_probe.py"
PY = TARGET / ".venv/Scripts/python.exe"
LEDGER = ARTIFACTS / "budget_ledger.json"
LOCK = HOME_DIR / "supervised_exec.lock"
DEV_MEETINGS = ("ES2009a", "ES2009c", "ES2009d", "ES2002b", "EN2009d")
CANONICAL_LAUNCH = "experiments/psem_r2_policy/launch.py"
ACK_GO = "DIRECTOR_GO"
WINDOW_MIN_SPAN_S = 3.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Console:
    """Append-only driver console mirrored to a durable log file."""

    def __init__(self, log_path: Path | None) -> None:
        self._handle = log_path.open("x", encoding="utf-8") if log_path is not None else None

    def __call__(self, message: str) -> None:
        line = f"[{_utc()} mono={time.monotonic():.3f}] {message}"
        print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def ledger_boundary() -> dict[str, Any]:
    """Lock-free journal snapshot: atomic os.replace means readers see whole JSON."""
    try:
        raw = LEDGER.read_bytes()
        state = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return {"path": str(LEDGER), "error": f"{type(exc).__name__}: {exc}", "monotonic_s": time.monotonic()}
    entries = list(state.get("entries") or ())
    kinds = Counter(str((entry.get("meta") or {}).get("kind") or "") for entry in entries)
    reserved_by_kind: dict[str, float] = {}
    for entry in entries:
        if entry.get("state") != "reserved":
            continue
        kind = str((entry.get("meta") or {}).get("kind") or "")
        reserved_by_kind[kind] = reserved_by_kind.get(kind, 0.0) + float(entry.get("reserved_usd") or 0.0)
    return {
        "path": str(LEDGER),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entries": len(entries),
        "entry_kinds": dict(sorted(kinds.items())),
        "reserved_usd_by_kind": {key: round(value, 12) for key, value in sorted(reserved_by_kind.items())},
        "cap_usd": state.get("cap_usd"),
        "phase_caps_usd": state.get("phase_caps_usd"),
        "read_at_utc": _utc(),
        "read_mode": "lock-free direct read (no budget lock, no write)",
    }


CHILD_STDIO = {"PYTHONIOENCODING": "utf-8"}


def _child_env() -> dict[str, str]:
    """Environment for every supervised child process.

    PYTHONIOENCODING is pinned to utf-8: the canonical launcher prints its full
    payload through stdout, and with stdout redirected to a file Python would
    otherwise use the locale codec (cp949 on this host), producing a dump that is
    not valid UTF-8 and can raise UnicodeEncodeError for non-locale characters.
    The pin is recorded in every prelaunch.json.
    """
    env = dict(os.environ)
    env.update(CHILD_STDIO)
    return env


class Attempt:
    """One child process, one attempt, full preservation."""

    def __init__(
        self,
        argv: Sequence[str],
        case_dir: Path,
        name: str,
        poll_s: float,
        console: Console,
        command_label: str = "",
    ) -> None:
        self.argv = [str(item) for item in argv]
        self.command_label = command_label
        self.case_dir = case_dir
        self.name = name
        self.poll_s = max(1.0, float(poll_s))
        self.stdout_path = case_dir / "stdout.json"
        self.stderr_path = case_dir / "stderr.log"
        self.progress_path = case_dir / "progress.jsonl"
        self.console = console
        self._stdout_handle = None
        self._stderr_handle = None
        self._progress_handle = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _heartbeat(self, proc: subprocess.Popen) -> None:
        while not self._stop.wait(self.poll_s):
            ledger = ledger_boundary()
            record = {
                "at_utc": _utc(),
                "monotonic_s": time.monotonic(),
                "child_alive": proc.poll() is None,
                "stdout_bytes": self.stdout_path.stat().st_size if self.stdout_path.exists() else 0,
                "ledger_bytes": ledger.get("bytes"),
                "ledger_entries": ledger.get("entries"),
                "ledger_reserved_usd_by_kind": ledger.get("reserved_usd_by_kind"),
            }
            self._progress_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._progress_handle.flush()
            self.console(
                f"heartbeat stdout={record['stdout_bytes']}B ledger_entries={record['ledger_entries']} "
                f"reserved={record['ledger_reserved_usd_by_kind']}"
            )

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        stream = proc.stderr
        assert stream is not None
        for chunk in iter(lambda: stream.readline(), b""):
            self._stderr_handle.write(chunk)
            self._stderr_handle.flush()
            text = chunk.decode("utf-8", errors="replace").rstrip("\r\n")
            self.console(f"stderr {len(chunk)}B: {text[:400]}")

    def run(self) -> dict[str, Any]:
        dry = {
            "argv": self.argv,
            "cwd": str(TARGET),
            "name": self.name,
            "case_dir": str(self.case_dir),
            "canonical_command": self.command_label,
            "argv_resolution_note": "argv uses absolute interpreter/entry paths; the canonical_command field is the brief-form relative command from this cwd",
            "child_stdio_encoding": {
                **CHILD_STDIO,
                "applies_to": ["canonical launcher child", "prepaid gate", "case_tools", "post_case adapter"],
                "reason": "stdout is redirected to a file; without the pin Python uses the locale codec (cp949), which produced a non-UTF-8 dump on 2026-09-11",
            },
        }
        prelaunch = {
            **dry,
            "mode": "single attempt; no retry; no deadline",
            "implicit_deadline": None,
            "interpreter": str(PY),
            "interpreter_exists": PY.is_file(),
            "ack_required": ACK_GO,
            "started_utc": _utc(),
            "started_epoch_s": time.time(),
            "started_monotonic_s": time.monotonic(),
            "ledger_before": ledger_boundary(),
            "driver_python": sys.executable,
            "driver_script_sha256": _sha256_file(Path(__file__).resolve()),
        }
        _write_json(self.case_dir / "prelaunch.json", prelaunch)
        self._stdout_handle = self.stdout_path.open("xb")
        self._stderr_handle = self.stderr_path.open("xb")
        self._progress_handle = self.progress_path.open("x", encoding="utf-8")
        self.console(f"start name={self.name} argv={self.argv} cwd={TARGET}")
        started_mono = time.monotonic()
        try:
            proc = subprocess.Popen(  # noqa: S603 - fixed canonical argv, no shell
                self.argv,
                cwd=str(TARGET),
                env=_child_env(),
                stdout=self._stdout_handle,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001 - startup failure is recorded
            self.console(f"spawn failed: {type(exc).__name__}: {exc}")
            outcome = {
                **dry,
                "status": "SPAWN_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_utc": _utc(),
                "wall_s": round(time.monotonic() - started_mono, 3),
            }
            _write_json(self.case_dir / "outcome.json", outcome)
            self._close()
            return outcome
        self.console(f"child pid={proc.pid}")
        for target in (self._pump_stderr, self._heartbeat):
            thread = threading.Thread(target=target, args=(proc,), daemon=True)
            thread.start()
            self._threads.append(thread)
        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            self._stop.set()
            for thread in self._threads:
                thread.join(timeout=5.0)
            outcome = {
                **dry,
                "status": "INTERRUPTED",
                "child_pid": proc.pid,
                "child_alive_at_interrupt": proc.poll() is None,
                "interrupted_utc": _utc(),
                "wall_s": round(time.monotonic() - started_mono, 3),
                "retried": False,
                "note": "operator interrupt of the supervising process; the child was not killed by this driver and no retry was made",
            }
            _write_json(self.case_dir / "outcome.json", outcome)
            self.console("INTERRUPTED: recorded, no retry")
            self._close()
            return outcome
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=30.0)
        wall_s = time.monotonic() - started_mono
        self.console(f"child exited exit_code={exit_code} wall_s={wall_s:.3f}")
        self._close()
        stdout_stat = self.stdout_path.stat()
        stderr_stat = self.stderr_path.stat()
        outcome = {
            **dry,
            "status": "COMPLETED",
            "exit_code": int(exit_code),
            "started_utc": prelaunch["started_utc"],
            "started_epoch_s": prelaunch["started_epoch_s"],
            "finished_utc": _utc(),
            "finished_epoch_s": time.time(),
            "finished_monotonic_s": time.monotonic(),
            "wall_s": round(wall_s, 3),
            "interrupted_by_deadline": False,
            "retried": False,
            "stdout": {
                "path": str(self.stdout_path),
                "bytes": stdout_stat.st_size,
                "sha256": _sha256_file(self.stdout_path),
            },
            "stderr": {
                "path": str(self.stderr_path),
                "bytes": stderr_stat.st_size,
                "sha256": _sha256_file(self.stderr_path),
            },
            "ledger_after": ledger_boundary(),
        }
        _write_json(self.case_dir / "outcome.json", outcome)
        return outcome

    def _close(self) -> None:
        for handle in (self._stdout_handle, self._stderr_handle, self._progress_handle):
            if handle is not None:
                handle.close()
        self._stdout_handle = self._stderr_handle = self._progress_handle = None


def run_gate(meeting: str, case_dir: Path) -> dict[str, Any]:
    gate_path = case_dir / "gate.json"
    command = [str(PY), str(GATE), "--meeting", meeting]
    completed = subprocess.run(
        command, cwd=str(TARGET), env=_child_env(), capture_output=True, text=False, check=False
    )
    raw = completed.stdout
    with gate_path.open("xb") as handle:
        handle.write(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 - recorded as-is
        payload = None
    return {
        "command": command,
        "cwd": str(TARGET),
        "exit_code": int(completed.returncode),
        "ok": bool(payload and payload.get("ok")),
        "payload": payload,
        "stderr": completed.stderr.decode("utf-8", errors="replace")[-4000:],
        "path": str(gate_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run_case_tools(meeting: str, case_dir: Path) -> dict[str, Any]:
    """Canonical post-case step, canonical-only and loud on failure.

    The supervised child pins PYTHONIOENCODING=utf-8, so the launcher dump is
    valid UTF-8 for the read-only canonical wrapper. If the wrapper still fails
    (for example a non-UTF-8 dump), the failure is recorded as-is and is not
    reinterpreted: no decoder fallback, no replacement characters, no swallowing.
    The archived one-shot tool dev-supervised/post_case.py exists only for the
    2026-09-11 historical cp949 dump and is never invoked automatically.
    """
    command = [
        str(PY),
        str(CASE_TOOLS),
        "case",
        "--meeting",
        meeting,
        "--stdout",
        str(case_dir / "stdout.json"),
        "--case-dir",
        str(case_dir),
        "--launch-meta",
        str(case_dir / "stderr.log"),
    ]
    completed = subprocess.run(
        command, cwd=str(TARGET), env=_child_env(), capture_output=True, text=False, check=False
    )
    (case_dir / "summary-print.json").write_bytes(completed.stdout)
    (case_dir / "case-tools.stderr.log").write_bytes(completed.stderr)
    record: dict[str, Any] = {
        "path": "canonical" if completed.returncode == 0 else "failed",
        "command": command,
        "exit_code": int(completed.returncode),
        "stdout_bytes": len(completed.stdout),
        "stdout_encoding": CHILD_STDIO["PYTHONIOENCODING"],
        "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-2000:],
    }
    return record


def _load_window_rates():
    spec = importlib.util.spec_from_file_location("psem_pacing_probe", PACING_PROBE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.window_rates


def pacing_report(meeting: str, summary_path: Path, out_path: Path) -> dict[str, Any]:
    """Objective pacing numbers from capture_timing.feed_progress; no threshold verdicts."""
    if not summary_path.is_file():
        return {"status": "SKIPPED", "reason": f"no summary: {summary_path}"}
    summary = _read_json(summary_path)
    capture = summary.get("capture_timing") or {}
    progress = capture.get("feed_progress") or {}
    samples = [[float(row[0]), float(row[1])] for row in (progress.get("samples") or ())]
    window_rates = _load_window_rates()
    rates = window_rates(samples, min_span_s=WINDOW_MIN_SPAN_S)
    ordered = sorted(rates)
    median = ordered[len(ordered) // 2] if ordered else None
    source_span = samples[-1][1] - samples[0][1] if samples else None
    wall_span = samples[-1][0] - samples[0][0] if samples else None
    payload = {
        "meeting": meeting,
        "source": "capture_timing.feed_progress (actual source position vs monotonic clock)",
        "summary_path": str(summary_path),
        "interval_s": progress.get("interval_s"),
        "n_samples": len(samples),
        "source_span_s": source_span,
        "wall_span_s": wall_span,
        "overall_rate": (source_span / wall_span) if (source_span and wall_span) else None,
        "window_min_span_s": WINDOW_MIN_SPAN_S,
        "window_rates": [round(rate, 6) for rate in rates],
        "window_rate_min": min(rates) if rates else None,
        "window_rate_median": median,
        "window_rate_max": max(rates) if rates else None,
        "first_sample": samples[0] if samples else None,
        "last_sample": samples[-1] if samples else None,
        "source_eof_monotonic_s": capture.get("last_arrival_monotonic_s"),
        "native_chunk_arrival_span_s": capture.get("native_chunk_arrival_span_s"),
        "native_chunk_arrival_distinct_stamps": capture.get("native_chunk_arrival_distinct_stamps"),
        "native_chunk_arrival_batched": capture.get("native_chunk_arrival_batched"),
        "fed_source_samples": capture.get("fed_source_samples"),
        "unprocessed_source_samples": capture.get("unprocessed_source_samples"),
        "capture_frame_seconds": capture.get("capture_frame_seconds"),
        "run_wall_s": summary.get("wall_s"),
        "verdict_from_this_report": "none: these are objective windowed rates for Director judgment",
    }
    _write_json(out_path, payload)
    return payload


def _acquire_lock(console: Console) -> None:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = LOCK.read_text(encoding="utf-8", errors="replace") if LOCK.exists() else "<unreadable>"
        console(f"REFUSED: {LOCK} exists; a supervised attempt may be live. Holder pid: {holder}")
        raise SystemExit(2)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} started_utc={_utc()}\n")


def _attempt_dir(meeting: str) -> Path:
    meeting_dir = CASES_DIR / meeting
    meeting_dir.mkdir(parents=True, exist_ok=True)
    used = sorted(int(path.name.split("-")[-1]) for path in meeting_dir.glob("attempt-*") if path.is_dir())
    return meeting_dir / f"attempt-{(used[-1] + 1) if used else 1}"


def canonical(args: argparse.Namespace) -> int:
    if args.ack_go != ACK_GO:
        print(f"REFUSED: paid canonical execution needs --ack-go {ACK_GO}", file=sys.stderr)
        return 2
    if args.meeting not in DEV_MEETINGS:
        print(f"REFUSED: {args.meeting} is not one of the five declared DEV meetings", file=sys.stderr)
        return 2
    if not PY.is_file() or not GATE.is_file() or not CASE_TOOLS.is_file():
        print("REFUSED: canonical interpreter or read-only wrappers are missing", file=sys.stderr)
        return 2
    lock_dir = HOME_DIR / "lock-logs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_log = lock_dir / f"canonical-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.log"
    lock_console = Console(lock_log)
    _acquire_lock(lock_console)
    try:
        case_dir = _attempt_dir(args.meeting)
        case_dir.mkdir(parents=False, exist_ok=False)
    except BaseException:
        LOCK.unlink(missing_ok=True)
        lock_console.close()
        raise
    lock_console.close()
    lock_log.unlink(missing_ok=True)
    console = Console(case_dir / "driver.log")
    try:
        console(f"case dir {case_dir}")
        console("preflight: read-only prepaid gate (no provider calls)")
        gate = run_gate(args.meeting, case_dir)
        console(f"gate ok={gate['ok']} exit={gate['exit_code']} sha256={gate['sha256']}")
        if not gate["ok"]:
            _write_json(
                case_dir / "outcome.json",
                {"meeting": args.meeting, "status": "GATE_FAILED", "gate": gate, "child_started": False},
            )
            return 2

        argv = [
            str(PY),
            CANONICAL_LAUNCH,
            "--paid",
            "--phase",
            "dev",
            "--meeting",
            args.meeting,
        ]
        attempt = Attempt(
            argv,
            case_dir,
            name=f"canonical:{args.meeting}",
            poll_s=args.poll_s,
            console=console,
            command_label=f".venv/Scripts/python.exe {CANONICAL_LAUNCH} --paid --phase dev --meeting {args.meeting}",
        )
        outcome = attempt.run()
        if outcome["status"] == "COMPLETED":
            console("post-case: canonical case output via existing case_tools (UTF-8 dump)")
            post = run_case_tools(args.meeting, case_dir)
            console(f"post-case path={post['path']} exit={post['exit_code']} stdout={post['stdout_bytes']}B")
            outcome["case_tools"] = post
            if post["path"] == "canonical":
                outcome["pacing"] = pacing_report(args.meeting, case_dir / "summary.json", case_dir / "pacing.json")
                console(f"pacing written: {case_dir / 'pacing.json'}")
            _write_json(case_dir / "outcome.json", outcome)
        console(f"done status={outcome['status']} exit_code={outcome.get('exit_code')}")
        return int(outcome.get("exit_code") or 0)
    finally:
        LOCK.unlink(missing_ok=True)
        console.close()


def standin(args: argparse.Namespace) -> int:
    exercise_dir = HOME_DIR / "exercise"
    exercise_dir.mkdir(parents=True, exist_ok=True)
    lock_console = Console(exercise_dir / f"lock-{args.label}.log")
    _acquire_lock(lock_console)
    case_dir = HOME_DIR / "exercise" / args.label
    try:
        case_dir.mkdir(parents=True, exist_ok=False)
    except BaseException:
        LOCK.unlink(missing_ok=True)
        lock_console.close()
        raise
    lock_console.close()
    (exercise_dir / f"lock-{args.label}.log").unlink(missing_ok=True)
    console = Console(case_dir / "driver.log")
    try:
        argv = [
            str(PY),
            str(HOME_DIR / "_standin.py"),
            "--bytes",
            str(args.bytes),
            "--exit-code",
            str(args.exit_code),
            "--sleep-s",
            str(args.sleep_s),
            "--stderr-lines",
            str(args.stderr_lines),
            "--stderr-period-s",
            str(args.stderr_period_s),
        ]
        if args.unicode_sample:
            argv.append("--unicode-sample")
        attempt = Attempt(
            argv,
            case_dir,
            name=f"standin:{args.label}",
            poll_s=args.poll_s,
            console=console,
            command_label=" ".join(argv),
        )
        outcome = attempt.run()
        markers = {}
        with (case_dir / "stdout.json").open("rb") as handle:
            tail = handle.read()[-4096:]
        for line in reversed(tail.decode("utf-8", errors="replace").splitlines()):
            if line.startswith("{"):
                try:
                    markers = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        payload_bytes = int(markers.get("payload_bytes") or 0)
        digest = hashlib.sha256()
        with (case_dir / "stdout.json").open("rb") as handle:
            remaining = payload_bytes
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
        outcome["standin_marker"] = markers
        outcome["preservation"] = {
            "payload_bytes_declared": payload_bytes,
            "payload_bytes_hashed_from_file": payload_bytes - remaining,
            "prefix_sha256_from_file": digest.hexdigest(),
            "prefix_sha256_declared": markers.get("sha256"),
            "prefix_hash_match": digest.hexdigest() == markers.get("sha256"),
            "file_bytes": (case_dir / "stdout.json").stat().st_size,
            "truncated": remaining != 0,
        }
        if markers.get("marker") == "standin-text":
            raw = (case_dir / "stdout.json").read_bytes()
            try:
                text = raw.decode("utf-8")
                decode_error = None
            except UnicodeDecodeError as exc:
                text = ""
                decode_error = f"{type(exc).__name__}: {exc}"
            sample = str(markers.get("unicode_sample") or "")
            outcome["text_payload_check"] = {
                "strict_utf8_decode": decode_error is None,
                "decode_error": decode_error,
                "bytes": len(raw),
                "sample_in_dump": bool(sample) and sample in text,
                "sample_sha256": hashlib.sha256(sample.encode("utf-8")).hexdigest(),
                "replacement_characters": text.count("\ufffd") if decode_error is None else None,
                "chars": markers.get("chars"),
                "repeats": markers.get("repeats"),
            }
            console(f"standin text check {outcome['text_payload_check']}")
        _write_json(case_dir / "outcome.json", outcome)
        console(f"standin preservation {outcome['preservation']}")
        return int(outcome.get("exit_code") or 0)
    finally:
        LOCK.unlink(missing_ok=True)
        console.close()


def preflight(args: argparse.Namespace) -> int:
    case_dir = HOME_DIR / "preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case_dir.mkdir(parents=True, exist_ok=False)
    console = Console(case_dir / "driver.log")
    records: list[dict[str, Any]] = []
    for meeting in DEV_MEETINGS:
        gate_path = case_dir / f"gate-{meeting}.json"
        command = [str(PY), str(GATE), "--meeting", meeting]
        completed = subprocess.run(
            command, cwd=str(TARGET), env=_child_env(), capture_output=True, text=False, check=False
        )
        with gate_path.open("wb") as handle:
            handle.write(completed.stdout)
        payload = json.loads(completed.stdout.decode("utf-8"))
        record = {
            "meeting": meeting,
            "exit_code": int(completed.returncode),
            "ok": bool(payload.get("ok")),
            "audio_sha256": payload.get("audio_sha256"),
            "gt_words": payload.get("gt_words"),
            "duration_s": payload.get("duration_s"),
            "failures": payload.get("failures"),
            "gate_path": str(gate_path),
            "gate_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        }
        records.append(record)
        console(f"gate {meeting}: ok={record['ok']} exit={record['exit_code']} failures={record['failures']}")
    reuse = capsule_reuse_check()
    console(f"capsule reuse: {json.dumps(reuse, ensure_ascii=False)[:400]}")
    payload = {
        "mode": "preflight",
        "at_utc": _utc(),
        "manifest": {
            "path": r"C:/tmp/psem-u8-inputs/input-integrity-manifest.json",
            "sha256": hashlib.sha256(
                Path(r"C:/tmp/psem-u8-inputs/input-integrity-manifest.json").read_bytes()
            ).hexdigest(),
        },
        "gates": records,
        "gates_all_ok": all(record["ok"] for record in records),
        "capsule_reuse": reuse,
        "ledger": ledger_boundary(),
        "providers_started": False,
        "paid_calls": 0,
    }
    out = case_dir / "preflight.json"
    _write_json(out, payload)
    console(f"preflight written: {out}")
    console.close()
    return 0 if payload["gates_all_ok"] else 2


def capsule_reuse_check() -> dict[str, Any]:
    """Read-only: does the current tree still match the newest built capsule?"""
    pin = json.loads((PACKAGE / "RUNTIME_PIN.json").read_text(encoding="utf-8"))
    archive = PACKAGE / str(pin["runtime_archive"]["file"])
    manifests = sorted(
        (PACKAGE / str(pin["capsule"]["dir"])).glob(f"*/{pin['capsule']['manifest']}"),
        key=lambda path: path.stat().st_mtime,
    )
    payload: dict[str, Any] = {
        "archive": str(archive),
        "archive_exists": archive.is_file(),
        "archive_sha256": _sha256_file(archive) if archive.is_file() else None,
        "archive_sha256_pinned": pin["runtime_archive"]["sha256"],
        "newest_manifest": str(manifests[-1]) if manifests else None,
    }
    payload["archive_match"] = payload["archive_sha256"] == payload["archive_sha256_pinned"]
    if not manifests:
        payload["reuse_expected"] = False
        payload["reason"] = "no built capsule manifest"
        return payload
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    mismatches = []
    for item in manifest["overlay"]:
        source = TARGET / str(item["source"])
        observed = _sha256_file(source) if source.is_file() else None
        if observed != item["sha256"]:
            mismatches.append({"capsule": item["capsule"], "source": item["source"], "observed": observed})
    payload.update(
        {
            "capsule_fingerprint": manifest.get("fingerprint"),
            "capsule_root": str(manifests[-1].parent),
            "overlay_files": len(manifest["overlay"]),
            "overlay_mismatches": mismatches,
            "reuse_expected": bool(payload["archive_match"] and not mismatches),
            "expected_tests": manifest.get("expected_tests"),
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supervised_exec.py", allow_abbrev=False)
    sub = parser.add_subparsers(dest="mode", required=True)
    canon = sub.add_parser("canonical")
    canon.add_argument("--meeting", required=True)
    canon.add_argument("--ack-go", required=True)
    canon.add_argument("--poll-s", type=float, default=20.0)
    pre = sub.add_parser("preflight")
    pace = sub.add_parser("pacing")
    pace.add_argument("--meeting", required=True)
    pace.add_argument("--summary", required=True)
    pace.add_argument("--out", required=True)
    stand = sub.add_parser("standin")
    stand.add_argument("--label", required=True)
    stand.add_argument("--bytes", type=int, default=1 << 20)
    stand.add_argument("--exit-code", type=int, default=0)
    stand.add_argument("--sleep-s", type=float, default=0.0)
    stand.add_argument("--stderr-lines", type=int, default=3)
    stand.add_argument("--stderr-period-s", type=float, default=0.5)
    stand.add_argument("--unicode-sample", action="store_true")
    stand.add_argument("--poll-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.mode == "canonical":
        return canonical(args)
    if args.mode == "pacing":
        payload = pacing_report(args.meeting, Path(args.summary), Path(args.out))
        print(json.dumps({k: payload.get(k) for k in ("meeting", "n_samples", "source_span_s", "wall_span_s", "overall_rate", "window_rate_min", "window_rate_median", "window_rate_max", "source_eof_monotonic_s")}, indent=1, ensure_ascii=False))
        return 0 if payload.get("n_samples") else 2
    if args.mode == "standin":
        return standin(args)
    return preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
