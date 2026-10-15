from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

from prepare import HERE, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    candidate = subprocess.run(["git", "rev-parse", args.candidate + "^{commit}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    index_path = HERE / "results/evidence-index.json"
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw)
    entries = index["entries"]
    prefix = (HERE / "results").relative_to(ROOT).as_posix()
    query = "".join(f"{candidate}:{prefix}/{entry['path']}\n" for entry in entries)
    blobs = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input=query.encode("utf-8"), capture_output=True, check=True).stdout
    cursor = 0
    failures = []
    matched = 0
    publication_total = 0
    for entry in entries:
        path = HERE / "results" / entry["path"]
        raw = path.read_bytes()
        raw.decode("utf-8")
        publication_total += len(raw)
        if b"\r" in raw:
            failures.append({"path": entry["path"], "reason": "worktree result contains CR bytes"})
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            failures.append({"path": entry["path"], "reason": "index differs from publication bytes"})
        end = blobs.index(b"\n", cursor)
        header = blobs[cursor:end].split()
        cursor = end + 1
        if len(header) != 3 or header[1] != b"blob":
            failures.append({"path": entry["path"], "reason": "candidate result blob unavailable", "header": b" ".join(header).decode("utf-8", errors="replace")})
            continue
        size = int(header[2])
        blob = memoryview(blobs)[cursor:cursor + size]
        cursor += size + 1
        if size != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            failures.append({"path": entry["path"], "reason": "index differs from candidate Git blob", "candidate_bytes": size, "candidate_sha256": hashlib.sha256(blob).hexdigest()})
        else:
            matched += 1
    if b"\r" in index_raw or index.get("encoding") != "UTF-8" or index.get("line_endings") != "LF":
        failures.append({"path": "evidence-index.json", "reason": "index encoding/newline declaration or bytes are not canonical"})
    if publication_total != index["total_bytes"]:
        failures.append({"path": "evidence-index.json", "reason": "total byte count differs"})
    report = {"reference_candidate": candidate, "scientific_product_baseline": "7a8151be6114a950f062e2430bd36f899d57e4de", "scope": "Compare the repaired working publication index and actual UTF-8 LF result bytes against read-only Git result blobs at the reference candidate. The repaired index/helper/docs are not claimed committed there; the index excludes itself. No Git mutations or inference.", "indexed_files": len(entries), "matching_candidate_result_blobs": matched, "total_publication_bytes": publication_total, "publication_index_sha256": hashlib.sha256(index_raw).hexdigest(), "producer_compaction_metadata_matches_candidate": not any(item["path"] == "producer-compaction.json" for item in failures), "failures": failures, "outcome": "pass" if not failures else "fail"}
    (HERE / "PUBLICATION_AUDIT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
