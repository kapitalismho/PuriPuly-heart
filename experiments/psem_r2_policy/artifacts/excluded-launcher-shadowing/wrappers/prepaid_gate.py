"""MANDATORY external pre-paid gate for each DEV case.

Verifies the immutable input manifest (raw-byte sha256), the meeting's audio and
four A-D word XML raw hashes, audio geometry, and usable GT via the canonical
harness loader. Read-only; exits 0 on PASS, 2 on FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import wave
from pathlib import Path

TARGET = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls"
)
MANIFEST = Path(r"C:/tmp/psem-u8-inputs/input-integrity-manifest.json")
PINNED_MANIFEST_SHA256 = "4734623b165f89cfa89a47ce5a6d23f1756efc9e178c82c366a60e1e9e3a9030"
HARNESS_LIVE_RUNNER = TARGET / "experiments/psem_r2_policy/live_runner.py"

sys.path.insert(0, str(TARGET / "src"))
sys.path.insert(0, str(TARGET))

from experiments.psem_r2_policy.metrics import AMI_WORDS, ROLES, load_ami_words  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def harness_ami_audio() -> str:
    text = HARNESS_LIVE_RUNNER.read_text(encoding="utf-8")
    match = re.search(r'^AMI_AUDIO = Path\(r"([^"]+)"\)', text, re.MULTILINE)
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", required=True)
    args = parser.parse_args()
    meeting = args.meeting
    checks: list[dict] = []
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    manifest_raw = MANIFEST.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    check("manifest_raw_sha256", manifest_sha == PINNED_MANIFEST_SHA256, manifest_sha)
    manifest = json.loads(manifest_raw.decode("utf-8"))
    check("manifest_ok", bool(manifest.get("ok")), manifest.get("blockers"))
    record = next((row for row in manifest["dev"] if row["meeting"] == meeting), None)
    check("meeting_declared_in_manifest", record is not None, meeting)
    if record is None:
        print(json.dumps({"ok": False, "meeting": meeting, "failures": failures, "checks": checks}, indent=1))
        return 2

    audio = record["audio"]
    wav = Path(audio["path"])
    check("audio_exists", wav.is_file(), str(wav))
    check(
        "audio_path_form",
        wav.name in {f"{meeting}.Mix-Headset.wav", f"{meeting}.wav"} and wav.parent.name == meeting,
        str(wav),
    )
    harness_root = Path(harness_ami_audio())
    check("harness_ami_audio_root", harness_root == wav.parent.parent, f"{harness_root} vs {wav.parent.parent}")
    observed_sha = sha256_file(wav)
    check("audio_raw_sha256", observed_sha == audio["sha256"], observed_sha)
    with wave.open(str(wav), "rb") as handle:
        frames, rate = handle.getnframes(), handle.getframerate()
        channels, width = handle.getnchannels(), handle.getsampwidth()
    check("audio_frames", frames == audio["nframes"], frames)
    check("audio_rate", rate == audio["framerate_hz"], rate)
    check("audio_channels", channels == audio["channels"], channels)
    check("audio_sampwidth", width == audio["sampwidth_bytes"], width)
    check("audio_duration", abs(frames / rate - audio["duration_s"]) < 1e-6, frames / rate)

    for role in ROLES:
        entry = record["words"][role]
        path = Path(entry["path"])
        check(f"xml_exists_{role}", path.is_file(), str(path))
        check(f"xml_parse_ok_{role}", bool(entry.get("parse_ok")), entry.get("parse_error"))
        observed = sha256_file(path)
        check(f"xml_raw_sha256_{role}", observed == entry["sha256"], observed)

    gt = load_ami_words(meeting)
    check("gt_usable_nonempty", len(gt) > 0, len(gt))
    check("gt_word_count", len(gt) == record["gt"]["n_words"], len(gt))
    max_end = max((row["end_src"] for row in gt), default=0)
    check("gt_end_within_audio", max_end <= frames + 16000, max_end)
    check("gt_words_dir", Path(str(AMI_WORDS)) == Path(str(manifest["paths"]["ami_words_dir"])), str(AMI_WORDS))

    payload = {
        "ok": not failures,
        "meeting": meeting,
        "manifest_sha256": manifest_sha,
        "audio_sha256": observed_sha,
        "gt_words": len(gt),
        "duration_s": frames / rate,
        "failures": failures,
        "checks": checks,
    }
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
