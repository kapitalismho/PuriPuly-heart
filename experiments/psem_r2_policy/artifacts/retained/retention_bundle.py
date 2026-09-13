from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
RETAINED = Path(__file__).resolve().parent
POLICY = ROOT / "experiments/psem_r2_policy"
ARTIFACTS = POLICY / "artifacts"

RAW_SOURCES = [
    ("old", "ES2009a", "dev/ES2009a/20260911T234330502940Z.json", "db19f0d70605bda2ba3717bbc23347bbf7a8f47c8a9a33e35f902d0a17dbef00"),
    ("old", "ES2009c", "dev/ES2009c/20260912T073421006732Z.json", "c5ac0c27bbb0572406a038b50ae0fe86dbe3fdb374978d4e7a44581e16684c2d"),
    ("old", "ES2009d", "dev/ES2009d/20260912T084839024097Z.json", "288b38a25c03e24978974117f674710a03e2f59fb2c1b1f796bdfeb108ca183e"),
    ("current", "ES2009a", "dev/ES2009a/20260912T110908853831Z.json", "9698e43ffa19335b2e9acaf0b49fe1bed323aecbec00ba902c490a32395f2156"),
    ("current", "ES2009c", "dev/ES2009c/20260912T115833024065Z.json", "f07c7464c61639c186776fb1a69052eff42c265986f90e99cb7a81eae83c17c7"),
    ("current", "ES2009d", "dev/ES2009d/20260912T130134673943Z.json", "766a572513d697bcb2652080475c71ba79e344b6fe4d03540cf25e57a11398d5"),
    ("current", "ES2002b", "dev/ES2002b/20260912T154147758028Z.json", "9acc74523a1db314e9bd6df8fc49a474b1426d03b11fd60dd3ebe454a125817c"),
    ("current", "EN2009d", "dev/EN2009d/20260913T003834479506Z.json", "bf674dd2eb542bb496c70f5ae4c98a384d2cf9243fcc8261f57f72497ffa07b7"),
]


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BundleWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=9)
        self.digest = hashlib.sha256()
        self.rows = 0
        self.uncompressed_bytes = 0

    def write(self, value: Any) -> None:
        data = canonical(value)
        self.compressed.write(data)
        self.digest.update(data)
        self.rows += 1
        self.uncompressed_bytes += len(data)

    def close(self) -> dict[str, Any]:
        self.compressed.close()
        self.raw.close()
        return {
            "path": self.path.relative_to(ROOT).as_posix(),
            "sha256": sha256(self.path),
            "content_sha256": self.digest.hexdigest(),
            "rows": self.rows,
            "bytes": self.path.stat().st_size,
            "uncompressed_bytes": self.uncompressed_bytes,
        }


def build_policy() -> dict[str, Any]:
    import ijson

    output = RETAINED / "historical_policy_inputs.jsonl.gz"
    writer = BundleWriter(output)
    writer.write({"type": "header", "schema": "PSEM-R2-RETAINED-POLICY-INPUTS-1", "sources": [{"cohort": c, "meeting": m, "path": p, "sha256": h} for c, m, p, h in RAW_SOURCES]})
    counts: dict[str, dict[str, int]] = {}
    for cohort, meeting, relative, expected in RAW_SOURCES:
        path = ARTIFACTS / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {relative}: {actual}")
        key = f"{cohort}:{meeting}"
        counts[key] = {"parents": 0, "native_chunks": 0}
        with path.open("rb") as source:
            for parent in ijson.items(source, "parents.item", use_float=True):
                parent.pop("evidence", None)
                writer.write({"type": "parent", "cohort": cohort, "meeting": meeting, "source_sha256": expected, "value": parent})
                counts[key]["parents"] += 1
        with path.open("rb") as source:
            for chunk in ijson.items(source, "native_chunks.item", use_float=True):
                writer.write({"type": "native_chunk", "cohort": cohort, "meeting": meeting, "source_sha256": expected, "value": chunk})
                counts[key]["native_chunks"] += 1
    metadata = writer.close()
    metadata["counts"] = counts
    return metadata


def load_journal_completions(path: Path, acquisition_id: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            event = json.loads(line)
            if event.get("event") != "completed":
                continue
            child_id = event["original_child_id"]
            completed[child_id] = {
                "acquisition_id": acquisition_id,
                "journal_line": number,
                "original_request_id": event["original_request_id"],
                "original_child_id": child_id,
                "new_utterance_id": event["new_utterance_id"],
                "parent_id": event["parent_id"],
                "meeting": event["meeting"],
                "status": event["status"],
                "response": event["new_response"] if event["status"] == "translated" else None,
                "error": event.get("error"),
                "request_body_sha256": event["request_body_sha256"] if "request_body_sha256" in event else None,
            }
    return completed


def build_translations() -> dict[str, Any]:
    import ijson

    base = ARTIFACTS / "dev-text-reacquisition"
    completions = {}
    completions.update(load_journal_completions(base / "acquisition.jsonl", "DEV-R2-TEXT-REACQUISITION-1"))
    completions.update(load_journal_completions(base / "acquisition-continuation-1.jsonl", "DEV-R2-TEXT-REACQUISITION-1-CONTINUATION-1"))
    output = RETAINED / "translated_texts.jsonl.gz"
    writer = BundleWriter(output)
    writer.write({"type": "header", "schema": "PSEM-R2-RETAINED-TRANSLATIONS-1", "prepared_input_sha256": "5fce260381d61992fb1c42ce2061ed539fcb4f2753bd11cd06e7561f529e27b6", "failed_outputs_are_never_promoted": True})
    counts = {"parents": 0, "r0_translated": 0, "r0_unavailable": 0, "r2_translated": 0, "r2_failed": 0, "r2_source_only": 0}
    linked_completions = set()
    prepared = base / "prepared-input-v3.json"
    if sha256(prepared) != "5fce260381d61992fb1c42ce2061ed539fcb4f2753bd11cd06e7561f529e27b6":
        raise RuntimeError("prepared-input-v3 hash mismatch")
    with prepared.open("rb") as source:
        for parent in ijson.items(source, "parents.item", use_float=True):
            retained_r0 = parent.get("retained_r0", [])
            r2_rows = []
            for child in parent.get("r2_children", []):
                child_id = child["original_child_id"]
                completion = completions.get(child_id)
                row = dict(child)
                if completion is None:
                    status = child.get("original_status")
                    if status == "source_only":
                        counts["r2_source_only"] += 1
                    else:
                        counts["r2_failed"] += 1
                else:
                    if completion["parent_id"] != parent["parent_id"]:
                        raise RuntimeError(f"parent identity mismatch: {child_id}")
                    row["actual_acquisition"] = completion
                    linked_completions.add(child_id)
                    counts[f"r2_{completion['status']}"] += 1
                r2_rows.append(row)
            for row in retained_r0:
                counts["r0_translated" if row.get("status") == "translated" else "r0_unavailable"] += 1
            writer.write({"type": "parent", "ordered_index": parent["ordered_index"], "meeting": parent["meeting"], "parent_id": parent["parent_id"], "accepted_text": parent["accepted_text"], "operational_status": parent["operational_status"], "terminal_outcome": parent["terminal_outcome"], "failure_reason": parent.get("failure_reason"), "retained_r0": retained_r0, "r2_children": r2_rows})
            counts["parents"] += 1
    if set(completions) - linked_completions:
        raise RuntimeError("unlinked completed acquisition")
    metadata = writer.close()
    metadata["counts"] = counts
    return metadata


def build_arm_key() -> dict[str, Any]:
    import ijson

    source_path = ARTIFACTS / "dev-text-reacquisition/inspection-combined-continuation-arm-key-1.json"
    expected = "5629ef7707e9025ae2c1635a5ad26b135ae8b769a709b6773a2890b85b900895"
    if sha256(source_path) != expected:
        raise RuntimeError("arm key hash mismatch")
    writer = BundleWriter(RETAINED / "private_arm_key.jsonl.gz")
    writer.write({"type": "header", "schema": "PSEM-R2-RETAINED-PRIVATE-ARM-KEY-1", "private": True, "source_sha256": expected})
    with source_path.open("rb") as source:
        for mapping in ijson.items(source, "mapping.item", use_float=True):
            writer.write({"type": "mapping", **mapping})
    return writer.close()


def build_probe() -> dict[str, Any]:
    base = ARTIFACTS / "sortformer-low-latency-probe"
    writer = BundleWriter(RETAINED / "local_native_probe.jsonl.gz")
    writer.write({"type": "header", "schema": "PSEM-R2-RETAINED-LOCAL-NATIVE-PROBE-1", "clock_limit": "The old ASR feed-start/native pacing origin was not retained; historical alignment remains sensitivity analysis."})
    for relative in ["PLAN.json", "selection.json", "result.json", "conclusion.json", "recorded-default/run.json", "official-low-latency/run.json"]:
        path = base / relative
        writer.write({"type": "json_artifact", "path": relative, "source_sha256": sha256(path), "value": json.loads(path.read_text(encoding="utf-8"))})
    for profile in ["recorded-default", "official-low-latency"]:
        events = base / profile / "native-events.jsonl"
        with events.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                writer.write({"type": "native_event", "profile": profile, "source_line": number, "value": json.loads(line)})
        trace = base / profile / "dump/diar.trace.json"
        writer.write({"type": "native_trace", "profile": profile, "source_sha256": sha256(trace), "value": json.loads(trace.read_text(encoding="utf-8"))})
    waveform = base / "input-prefix-plus-tail.wav"
    writer.write({"type": "input_identity", "path": "input-prefix-plus-tail.wav", "sha256": sha256(waveform), "bytes": waveform.stat().st_size, "retained_content": False, "reconstruction": "Original AMI audio and annotations remain outside the generated-artifact deletion scope; selection.json preserves the exact prefix/tail selection."})
    return writer.close()


def build_capsule_index() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((POLICY / ".capsule").glob("*/capsule_manifest.json")):
        rows.append({"capsule": path.parent.name, "manifest_sha256": sha256(path), "manifest": json.loads(path.read_text(encoding="utf-8"))})
    return rows


def iter_bundle(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def iter_translation_children(path: Path) -> Iterable[dict[str, Any]]:
    for row in iter_bundle(path):
        if row.get("type") == "parent":
            yield from row["r2_children"]


def bundle_content_identity(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    rows = 0
    size = 0
    with gzip.open(path, "rb") as source:
        for line in source:
            json.loads(line)
            digest.update(line)
            rows += 1
            size += len(line)
    return digest.hexdigest(), rows, size

def recount_policy(path: Path) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "old_parents": 0,
        "old_nonempty": 0,
        "old_r2_units": 0,
        "current_parents": 0,
        "current_nonempty": 0,
        "current_r2_units": 0,
        "native_chunks": 0,
        "admissions": 0,
    }
    digest = hashlib.sha256()
    for row in iter_bundle(path):
        if row.get("type") == "native_chunk":
            counts["native_chunks"] += 1
            continue
        if row.get("type") != "parent":
            continue
        cohort = row["cohort"]
        parent = row["value"]
        counts[f"{cohort}_parents"] += 1
        if parent.get("text"):
            counts[f"{cohort}_nonempty"] += 1
            counts[f"{cohort}_r2_units"] += len(parent.get("r2", {}).get("child_ids", []))
        if parent.get("marks", {}).get("translation_admission") is not None:
            counts["admissions"] += 1
        digest.update(canonical([cohort, row["meeting"], parent.get("parent_id"), parent.get("text"), parent.get("tokens")]))
    counts["accepted_text_token_content_sha256"] = digest.hexdigest()
    return counts


def recount_translations(path: Path) -> dict[str, int]:
    counts = {"parents": 0, "r0_translated": 0, "r0_unavailable": 0, "r2_translated": 0, "r2_failed": 0, "r2_source_only": 0}
    for row in iter_bundle(path):
        if row.get("type") != "parent":
            continue
        counts["parents"] += 1
        for r0 in row["retained_r0"]:
            counts["r0_translated" if r0.get("status") == "translated" else "r0_unavailable"] += 1
        for child in row["r2_children"]:
            actual = child.get("actual_acquisition")
            if actual:
                counts[f"r2_{actual['status']}"] += 1
            elif child.get("original_status") == "source_only":
                counts["r2_source_only"] += 1
            else:
                counts["r2_failed"] += 1
    return counts


def verify() -> dict[str, Any]:
    manifest = json.loads((RETAINED / "RETENTION_MANIFEST.json").read_text(encoding="utf-8"))
    checked = {}
    for name, expected in manifest["bundles"].items():
        path = ROOT / expected["path"]
        actual_content, rows, size = bundle_content_identity(path)
        actual = {"sha256": sha256(path), "content_sha256": actual_content, "rows": rows, "uncompressed_bytes": size, "bytes": path.stat().st_size}
        for key, value in actual.items():
            if value != expected[key]:
                raise RuntimeError(f"{name} {key}: expected {expected[key]}, got {value}")
        checked[name] = actual
    translations = recount_translations(ROOT / manifest["bundles"]["translations"]["path"])
    expected_translations = {"parents": 2482, "r0_translated": 2363, "r0_unavailable": 4, "r2_translated": 2456, "r2_failed": 1, "r2_source_only": 2}
    if translations != expected_translations or translations != manifest["bundles"]["translations"]["counts"]:
        raise RuntimeError(f"translation census mismatch: {translations}")
    policy = recount_policy(ROOT / manifest["bundles"]["policy"]["path"])
    expected_policy = manifest["bundles"]["policy"]["recount"]
    if policy != expected_policy:
        raise RuntimeError(f"policy census mismatch: {policy}")
    core = json.loads((RETAINED / "CORE_RESULTS.json").read_text(encoding="utf-8"))
    if core["full_dev"]["translation_units"] != {"r0": 2367, "r2": 2459}:
        raise RuntimeError("core DEV unit census mismatch")
    return {"ok": True, "schema": manifest["schema"], "bundles": checked, "critical_censuses": {"policy_inputs": policy, "old_new_units": [1808, 1057], "dev_units": [2367, 2459], "translations": translations}}


def build() -> None:
    RETAINED.mkdir(parents=True, exist_ok=True)
    result = {
        "policy": build_policy(),
        "translations": build_translations(),
        "private_arm_key": build_arm_key(),
        "local_probe": build_probe(),
    }
    result["capsules"] = build_capsule_index()
    (RETAINED / "BUILD_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify", "summary", "extract"])
    parser.add_argument("--bundle", choices=["policy", "translations", "private_arm_key", "local_probe"], default="policy")
    parser.add_argument("--type")
    parser.add_argument("--cohort")
    parser.add_argument("--meeting")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "verify":
        print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))
    elif args.command == "summary":
        core = json.loads((RETAINED / "CORE_RESULTS.json").read_text(encoding="utf-8"))
        print(json.dumps(core, ensure_ascii=False, indent=2))
    else:
        manifest = json.loads((RETAINED / "RETENTION_MANIFEST.json").read_text(encoding="utf-8"))
        path = ROOT / manifest["bundles"][args.bundle]["path"]
        for row in iter_bundle(path):
            if args.type and row.get("type") != args.type:
                continue
            if args.cohort and row.get("cohort") != args.cohort:
                continue
            if args.meeting and row.get("meeting") != args.meeting:
                continue
            print(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
