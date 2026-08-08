"""Materialize the Phase 1 addendum AMI audio additions (approved plan, rev 2).

- Recomputes the frozen selection at runtime (eligibility + union-find component graph +
  hash order) and fails closed on mismatch with the frozen lists.
- Downloads each wav to `<name>.part` with Range resume and atomically renames after
  full validation (mono 16 kHz PCM16 decode; duration within the frozen tolerance;
  per-file SHA-256).
- Emits a canonical, self-hashed materialization manifest with the ordered selected list
  and group (development|reserved) per meeting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.ami import (
    _load_meetings_xml,
    ami_mirror_url,
)
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

PLAN_BLOB = "24340f488f1bb46c666a5fc15eef2fc87ef1f826"
TOUCHED_SERIES = {"ES2003", "ES2004", "IS1008", "IS1009"}
TOUCHED_MEETINGS = {"ES2003a", "ES2004a", "IS1008a", "IS1009a"}
CANONICAL_RATE_HZ = 16000
DURATION_TOLERANCE_S = 2.0
TARGET_DEVELOPMENT = 8
TARGET_RESERVED = 8

DEVELOPMENT = (
    "EN2002c",
    "TS3006a",
    "EN2001d",
    "TS3009b",
    "ES2015d",
    "TS3007a",
    "TS3012c",
    "TS3005b",
)
RESERVED = (
    "TS3003b",
    "ES2014a",
    "TS3004a",
    "EN2006a",
    "EN2009d",
    "TS3008b",
    "ES2016a",
    "ES2002b",
)


class MaterializationError(RuntimeError):
    pass


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        if node not in self.parent:
            self.parent[node] = node
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


@dataclass(frozen=True, slots=True)
class MeetingInfo:
    meeting_id: str
    duration_s: float
    files: tuple[Path, ...]


def _gather_meetings(
    annotations_dir: Path,
) -> tuple[dict[str, MeetingInfo], dict[str, dict[str, str]]]:
    meta = _load_meetings_xml(annotations_dir)
    words_dir = annotations_dir / "words"
    files_by_meeting: dict[str, list[Path]] = defaultdict(list)
    for p in words_dir.glob("*.words.xml"):
        files_by_meeting[p.name.split(".")[0]].append(p)
    infos: dict[str, MeetingInfo] = {}
    for meeting_id, files in files_by_meeting.items():
        duration_s = (meta.get(meeting_id) or {}).get("duration_s")
        if not duration_s:
            continue
        infos[meeting_id] = MeetingInfo(
            meeting_id=meeting_id,
            duration_s=float(duration_s),
            files=tuple(sorted(files)),
        )
    return infos, meta


def _component_graph(
    infos: dict[str, MeetingInfo],
    meta: dict[str, dict[str, str]],
) -> dict[str, str]:
    uf = _UnionFind()
    for meeting_id in infos:
        uf.find(meeting_id)
    by_participant: dict[str, list[str]] = defaultdict(list)
    for meeting_id, info in infos.items():
        agents = dict((meta.get(meeting_id) or {}).get("agents") or {})
        for path in info.files:
            letter = path.name.split(".")[1]
            global_id = agents.get(letter)
            if global_id:
                by_participant[global_id].append(meeting_id)
    for ids in by_participant.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
    by_series: dict[str, list[str]] = defaultdict(list)
    for meeting_id in infos:
        by_series[meeting_id.rstrip("abcdefghijklmnopqrstuvwxyz")].append(meeting_id)
    for ids in by_series.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
    return {meeting_id: uf.find(meeting_id) for meeting_id in infos}


def _preflight_pass(
    meeting_id: str,
    url: str,
    annotation_duration_s: float,
) -> bool:
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "stb-preflight/1.0", "Range": "bytes=0-63"}
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            header = resp.read(64)
        length = int(resp.headers.get("Content-Range", "").split("/")[-1]) or 0
    except Exception:
        return False
    if len(header) < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return False
    format_tag = int.from_bytes(header[20:22], "little")
    channels = int.from_bytes(header[22:24], "little")
    sample_rate = int.from_bytes(header[24:28], "little")
    bits = int.from_bytes(header[34:36], "little")
    if format_tag != 1 or channels != 1 or sample_rate != CANONICAL_RATE_HZ or bits != 16:
        return False
    decoded_s = length / 32000.0
    return abs(decoded_s - annotation_duration_s) <= DURATION_TOLERANCE_S


def compute_frozen_selection(
    infos: dict[str, MeetingInfo],
    meta: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    component_of = _component_graph(infos, meta)
    touched_roots = {component_of[m] for m in TOUCHED_MEETINGS if m in component_of}
    passing: list[str] = []
    for meeting_id, info in infos.items():
        series = meeting_id.rstrip("abcdefghijklmnopqrstuvwxyz")
        if meeting_id in TOUCHED_MEETINGS or series in TOUCHED_SERIES:
            continue
        if _preflight_pass(meeting_id, ami_mirror_url(meeting_id), info.duration_s):
            passing.append(meeting_id)
    ordered = sorted(passing, key=lambda m: (hashlib.sha256(m.encode()).hexdigest(), m))
    development: list[str] = []
    reserved: list[str] = []
    seen = set(touched_roots)
    for meeting_id in ordered:
        component = component_of[meeting_id]
        if component in seen:
            continue
        seen.add(component)
        if len(development) < TARGET_DEVELOPMENT:
            development.append(meeting_id)
        elif len(reserved) < TARGET_RESERVED:
            reserved.append(meeting_id)
        else:
            break
    if len(development) != TARGET_DEVELOPMENT or len(reserved) != TARGET_RESERVED:
        raise MaterializationError(
            f"selection incomplete: dev={len(development)} reserved={len(reserved)}"
        )
    return development, reserved


def download_with_resume(url: str, part: Path) -> None:
    part.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "stb-turn-episode/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset > 0:
            part.unlink()
            offset = 0
            request = urllib.request.Request(url, headers={"User-Agent": "stb-turn-episode/1.0"})
            response = urllib.request.urlopen(request, timeout=120)
        else:
            raise MaterializationError(f"download failed for {part.name}: {exc}") from exc
    if offset > 0:
        if response.status != 206:
            part.unlink()
            offset = 0
            request = urllib.request.Request(url, headers={"User-Agent": "stb-turn-episode/1.0"})
            response = urllib.request.urlopen(request, timeout=120)
        else:
            content_range = response.headers.get("Content-Range", "")
            expected = f"bytes {offset}-"
            if not content_range.startswith(expected):
                part.unlink()
                offset = 0
                request = urllib.request.Request(
                    url, headers={"User-Agent": "stb-turn-episode/1.0"}
                )
                response = urllib.request.urlopen(request, timeout=120)
    with response, part.open("ab") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize approved AMI audio additions")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="corpus root (default: STB_PHASE2_CORPORA_ROOT or TEMP/opencode/stb_phase2_corpora)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="manifest output directory (default: results/turn_episode_v1)",
    )
    args = parser.parse_args()

    corpus_root = args.corpus_root or external.corpus_root()
    if not corpus_root.is_dir():
        raise MaterializationError(f"corpus root not found: {corpus_root}")
    if args.out is None:
        args.out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    args.out.mkdir(parents=True, exist_ok=True)

    annotations_dir = corpus_root / "ami" / "annotations"
    infos, meta = _gather_meetings(annotations_dir)

    # Runtime recomputation of the frozen selection; fails closed on mismatch.
    development, reserved = compute_frozen_selection(infos, meta)
    expected_dev = list(DEVELOPMENT)
    expected_res = list(RESERVED)
    if development != expected_dev or reserved != expected_res:
        raise MaterializationError(
            "selection mismatch: recomputed "
            f"dev={development} reserved={reserved}; frozen dev={expected_dev} "
            f"reserved={expected_res}"
        )

    entries: dict[str, dict[str, Any]] = {}
    for meeting_id in expected_dev + expected_res:
        info = infos[meeting_id]
        group = "development" if meeting_id in expected_dev else "reserved"
        destination = corpus_root / "ami" / "audio" / meeting_id / f"{meeting_id}.Mix-Headset.wav"
        part = destination.with_name(destination.name + ".part")
        if not destination.is_file():
            download_with_resume(ami_mirror_url(meeting_id), part)
            try:
                samples = load_canonical_wav(part)
            except Exception as exc:
                raise MaterializationError(
                    f"{meeting_id}: invalid canonical wav after download: {exc}"
                ) from exc
            decoded_duration_s = samples.size / CANONICAL_RATE_HZ
            if abs(decoded_duration_s - info.duration_s) > DURATION_TOLERANCE_S:
                raise MaterializationError(
                    f"{meeting_id}: decoded {decoded_duration_s:.1f}s vs annotation "
                    f"{info.duration_s:.1f}s"
                )
            part.replace(destination)
        try:
            samples = load_canonical_wav(destination)
        except Exception as exc:
            raise MaterializationError(f"{meeting_id}: invalid canonical wav: {exc}") from exc
        decoded_duration_s = samples.size / CANONICAL_RATE_HZ
        if abs(decoded_duration_s - info.duration_s) > DURATION_TOLERANCE_S:
            raise MaterializationError(
                f"{meeting_id}: decoded {decoded_duration_s:.1f}s vs annotation "
                f"{info.duration_s:.1f}s"
            )
        entries[meeting_id] = {
            "group": group,
            "url": ami_mirror_url(meeting_id),
            "destination": str(destination),
            "decoded_duration_s": round(decoded_duration_s, 3),
            "annotation_duration_s": info.duration_s,
            "sha256": external.sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
        print(f"materialized {meeting_id} [{group}] {entries[meeting_id]['sha256'][:12]}")

    payload: dict[str, Any] = {
        "schema_version": "turn_episode_v1.ami_materialization",
        "plan_blob": PLAN_BLOB,
        "selection_rule": (
            "preflight-passing (mono 16k PCM16, duration within 2s of annotation); "
            "component graph over global participant ids and series; exclude touched "
            "components; order by sha256(meeting_id); new component only; 8 development "
            "then 8 reserved"
        ),
        "selected_meetings": [
            {"meeting_id": m, "group": g}
            for m, g in [(m, "development") for m in expected_dev]
            + [(m, "reserved") for m in expected_res]
        ],
        "meetings": entries,
    }
    manifest: dict[str, Any] = {
        **payload,
        "content_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    out_path = args.out / "ami_materialization_manifest.json"
    out_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
