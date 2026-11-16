from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from benchmarks.asr.issue441_score import Journal, normalized, number, source_lexical_unit
except ModuleNotFoundError:
    from issue441_score import Journal, normalized, number, source_lexical_unit


ARMS = ("S0", "S1", "E0")
MANIFEST_SHA256 = "dc9b9e08ddc55c2f1e501de007ceaf8d7c1b24cd529c9c5e0c2f0dcc81acf294"
SCORE_SHA256 = "5ec5f84b42eb23c65be277defb8648c1173126679ee8433fdb1df5e03731b78c"
ORIGINAL_SCORE_SHA256 = "07cc4333869bee13b2265d6d95902b112ae16d6d36392d807a5452ebd5c07c1f"
NS = "{http://nite.sourceforge.net/}"
PROTOCOL = {
    "status": "Post-hoc diagnostic supplement; not a preregistered pilot or replacement score.",
    "inventory": "All 120 original anchors; EN independently timed intact AMI word nodes, KO explicit item[5] pWord.ortho. Hangul, JA complete untagged EAF phrases, ZH ineligible. Original parent flags retained; no nearest-word fallback.",
    "source_identity": "Recording + annotation hash + original word/interval/annotation ID. Keep duplicates, annotated positive-duration different-speaker overlap, and repeated source text visible. Units sharing recordings/speakers/context are not independent trials.",
    "cutoff": "Original parent span end + 1 second. Independent online/eventual Journal.snapshot; only exported capture-time source mapping and cutoff-canonical IDs. Never remap with the eventual send ledger.",
    "surface": "Soniox final append order then current partial. E0 indexed segments in segment_index/word_index order then current partial; missing segment indices are barriers. Unindexed timestamp families stay separate text/timing diagnostics, never supply indexed identity or adjacency.",
    "matching": "Whole consecutive native-piece spans; normalize concatenated raw text with NFKC, casefold, Unicode alphanumeric only. Retain punctuation/spacing, untimed/unmapped lexical barriers and combining extensions. EN/KO reject raw adjacent letter/digit/mark boundaries. No fuzzy matching, substring within pieces, tokenization invention, collars, point-coverage requirement or time widening.",
    "localization": "Every contributing lexical piece must have mapped positive-duration source bounds positively intersecting the independent unit interval. Preserve provider/source bounds and source_ranges. Reject discontinuous/partial/padding mapping and backward source/provider progression; silence gaps are allowed.",
    "ambiguity": "Enumerate localized occurrences without speaker/reference-ID knowledge. Distinct observation occurrences, including replay copies, remain ambiguous. Collapse only punctuation wrappers with identical lexical observation provenance.",
    "identity": "Exactly one localized occurrence; every contributing lexical piece must have one known unanimous (session,native-ID). Any unknown retains missing_speaker; multiple known IDs retain mixed_identity, including both flags together. Punctuation/control/audio-event IDs never vote. Apply original fixed arm/record/session mapping last; unanimous unmatched ID is wrong, not excluded.",
    "denominators": "C/W/U, assigned coverage, conditional accuracy and correct coverage use the same source-eligible inventory. Report EN word, KO word, JA phrase and annotated-overlap strata; pooled rows are descriptive, not a global provider ranking. The 24 source-ineligible rows are not provider failures.",
    "limits": [
        "Fixed maps were optimized for the old point metric, not these units; do not refit.",
        "KO covers only one reference speaker; no annotated overlap does not prove absence.",
        "ZH has no lexical truth. JA whole phrases are harder strict matching units than words.",
        "Normalization erases punctuation/whitespace and is not linguistic or semantic equivalence; no script conversion, stemming or number expansion.",
        "Whole-piece boundaries, raw boundary ambiguity, substitutions and timing disagreement can prevent support; lexical mismatch is not an ASR-deletion verdict.",
        "E0 unindexed timing availability is separate from indexed identity/timing availability; no text/adjacency correlation is inferred.",
        "Corpus annotations are not new listening/adjudication. No product change, N lexical score, T1/T2 work, provider/MT/GUI call or turn-prediction adoption.",
    ],
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def ami_units(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        meeting = next(
            node
            for node in ET.fromstring(archive.read("corpusResources/meetings.xml"))
            if node.get("observation") == "ES2004d"
        )
        for participant in meeting:
            agent = participant.attrib["nxt_agent"]
            member = f"words/ES2004d.{agent}.words.xml"
            words = list(ET.fromstring(archive.read(member)))
            indices = {word.attrib[NS + "id"]: i for i, word in enumerate(words)}
            for segment in ET.fromstring(archive.read(f"segments/ES2004d.{agent}.segments.xml")):
                nodes = []
                for child in segment:
                    ids = re.findall(r"id\(([^)]+)\)", child.attrib["href"])
                    nodes.extend(words[indices[ids[0]] : indices[ids[-1]] + 1])
                result[segment.attrib[NS + "id"]] = [
                    {
                        "id": node.attrib[NS + "id"],
                        "start": float(node.attrib["starttime"]),
                        "end": float(node.attrib["endtime"]),
                        "text": node.text or "",
                        "clean": node.tag == "w" and node.get("trunc") != "true",
                        "source_member": member,
                        "source_attributes": dict(node.attrib),
                    }
                    for node in nodes
                    if node.tag == "w" and "starttime" in node.attrib and "endtime" in node.attrib
                ]
    return result


def ko_units(path):
    data = Path(path).read_bytes()
    text = data.decode("utf-16" if data[:2] in (b"\xfe\xff", b"\xff\xfe") else "utf-8-sig")
    blocks = [
        block
        for block in re.finditer(r"item \[(\d+)\]:(.*?)(?=\n\s*item \[\d+\]:|\Z)", text, re.S)
        if block[1] == "5"
    ]
    checked(len(blocks) == 1, "Missing unique KO item[5]")
    block = blocks[0]
    checked(re.search(r'name = "([^"]*)"', block[2])[1] == "pWord.ortho.", "Wrong KO tier")
    return [
        {
            "id": f"item[5]/intervals[{interval[1]}]",
            "start": float(interval[2]),
            "end": float(interval[3]),
            "text": interval[4].replace('""', '"'),
            "clean": bool(re.search(r"[\uac00-\ud7a3]", interval[4]))
            and "<" not in interval[4]
            and ">" not in interval[4],
            "source_member": "item[5]:pWord.ortho.",
            "source_attributes": {"xmin": interval[2], "xmax": interval[3]},
        }
        for interval in re.finditer(
            r'intervals \[(\d+)\]:\s*xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "((?:[^"]|"")*)"',
            block[2],
        )
    ]


def ja_units(path):
    document = ET.parse(path).getroot()
    times = {
        slot.attrib["TIME_SLOT_ID"]: slot.attrib["TIME_VALUE"]
        for slot in document.findall("./TIME_ORDER/TIME_SLOT")
    }
    return {
        node.attrib["ANNOTATION_ID"]: {
            "id": node.attrib["ANNOTATION_ID"],
            "start": int(times[node.attrib["TIME_SLOT_REF1"]]) / 1000,
            "end": int(times[node.attrib["TIME_SLOT_REF2"]]) / 1000,
            "text": node.findtext("ANNOTATION_VALUE", ""),
            "clean": "<" not in node.findtext("ANNOTATION_VALUE", "")
            and ">" not in node.findtext("ANNOTATION_VALUE", ""),
            "source_member": tier.attrib["TIER_ID"],
            "source_attributes": dict(node.attrib),
        }
        for tier in document.findall("TIER")
        for node in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION")
    }


def derive_entries(manifest, root):
    entries, cache = [], {}
    for recording in manifest["recordings"]:
        source = recording["source"]
        path = root / source["annotation_path"]
        checked(digest(path) == source["annotation_sha256"], f"Source hash changed: {path}")
        language, offset = recording["language"], source["offset_s"]
        key = language, source["annotation_sha256"]
        if key not in cache:
            cache[key] = {"en": ami_units, "ko": ko_units, "ja": ja_units}.get(
                language, lambda _: {}
            )(path)
        units = cache[key]
        parents = {span["id"]: span for span in recording["spans"]}
        for anchor in recording["anchors"]:
            parent = parents[anchor["span_id"]]
            checked(anchor["span_end_s"] == parent["end_s"], "Original cutoff disagreement")
            candidates = []
            if language == "en":
                candidates = [
                    unit
                    for unit in units[parent["source_annotation_id"]]
                    if unit["start"] < anchor["time_s"] + offset < unit["end"]
                    and round((unit["start"] + unit["end"]) / 2 - offset, 6) == anchor["time_s"]
                    and unit["text"] == anchor["reference_word"]
                ]
            elif language == "ko":
                candidates = [
                    unit
                    for unit in units
                    if unit["start"] <= anchor["time_s"] + offset < unit["end"]
                    and parent["source_start_s"]
                    <= unit["start"]
                    < unit["end"]
                    <= parent["source_end_s"]
                ]
            elif language == "ja":
                unit = units[parent["source_annotation_id"]]
                checked(
                    (unit["text"], unit["start"], unit["end"])
                    == (parent["text"], parent["source_start_s"], parent["source_end_s"]),
                    "JA source/parent disagreement",
                )
                candidates = [unit]
            entry = {
                "entry_id": anchor["id"],
                "recording_id": recording["id"],
                "language": language,
                "granularity": (
                    "phrase" if language == "ja" else "word" if language in ("en", "ko") else None
                ),
                "original_anchor": anchor,
                "original_parent": parent,
                "online_cutoff_s": parent["end_s"] + 1,
                "source_annotation_path": source["annotation_path"],
                "source_annotation_sha256": source["annotation_sha256"],
                "eligible": False,
                "eligibility_reason": (
                    "no_lexical_reference" if language == "zh" else "source_unit_not_unique"
                ),
            }
            if len(candidates) == 1:
                unit = candidates[0]
                intact = (
                    parent["source_start_s"]
                    <= unit["start"]
                    < unit["end"]
                    <= parent["source_end_s"]
                )
                eligible = unit["clean"] and intact and bool(normalized(unit["text"]))
                entry.update(
                    source_unit=unit,
                    source_occurrence_id=f"{recording['id']}:{source['annotation_sha256']}:{unit['id']}",
                    reference_text=unit["text"],
                    normalized_reference=normalized(unit["text"]),
                    start_s=unit["start"] - offset,
                    end_s=unit["end"] - offset,
                    eligible=eligible,
                    eligibility_reason=(
                        "clean_intact_source_unit"
                        if eligible
                        else "tagged_or_truncated_source_unit"
                    ),
                )
                entry["overlap_span_ids"] = [
                    span["id"]
                    for span in recording["spans"]
                    if span["speaker"] != anchor["speaker"]
                    and max(unit["start"], span["source_start_s"])
                    < min(unit["end"], span["source_end_s"])
                ]
                entry["annotated_overlap"] = bool(entry["overlap_span_ids"])
                pool = (
                    [unit for segment in units.values() for unit in segment]
                    if language == "en"
                    else units.values() if language == "ja" else units
                )
                entry["same_normalized_source_text_count"] = len(
                    {
                        other["id"]
                        for other in pool
                        if other["end"] > offset
                        and other["start"] < offset + recording["duration_s"]
                        and normalized(other["text"]) == entry["normalized_reference"]
                    }
                )
            entries.append(entry)
    occurrences = Counter(entry["source_occurrence_id"] for entry in entries if entry["eligible"])
    for entry in entries:
        entry["duplicate_source_unit"] = occurrences.get(entry.get("source_occurrence_id"), 0) > 1
    return entries


def inventory(entries):
    eligible = [entry for entry in entries if entry["eligible"]]
    return {
        "inventory": len(entries),
        "eligible": len(eligible),
        "source_ineligible": len(entries) - len(eligible),
        "unique_source_units": len({entry["source_occurrence_id"] for entry in eligible}),
        "annotated_overlap": sum(entry["annotated_overlap"] for entry in eligible),
        "by_language": {
            language: {
                "eligible": sum(entry["language"] == language for entry in eligible),
                "annotated_overlap": sum(
                    entry["language"] == language and entry["annotated_overlap"]
                    for entry in eligible
                ),
            }
            for language in sorted({entry["language"] for entry in entries})
        },
        "by_recording": {
            recording_id: {
                "eligible": sum(entry["recording_id"] == recording_id for entry in eligible),
                "annotated_overlap": sum(
                    entry["recording_id"] == recording_id and entry["annotated_overlap"]
                    for entry in eligible
                ),
            }
            for recording_id in sorted({entry["recording_id"] for entry in entries})
        },
    }


def prepare(root, inputs, score, selection, original_score):
    checked(digest(inputs) == MANIFEST_SHA256, "Original manifest hash changed")
    checked(digest(score) == SCORE_SHA256, "Accepted point score hash changed")
    checked(digest(original_score) == ORIGINAL_SCORE_SHA256, "Original score hash changed")
    manifest, fixed, selected = load(inputs), load(score), load(selection)
    checked(selected["manifest_sha256"] == MANIFEST_SHA256, "Selection manifest mismatch")
    entries = derive_entries(manifest, root)
    counts = inventory(entries)
    checked(
        counts
        == {
            "inventory": 120,
            "eligible": 96,
            "source_ineligible": 24,
            "unique_source_units": 96,
            "annotated_overlap": 25,
            "by_language": {
                "en": {"eligible": 40, "annotated_overlap": 3},
                "ja": {"eligible": 36, "annotated_overlap": 22},
                "ko": {"eligible": 20, "annotated_overlap": 0},
                "zh": {"eligible": 0, "annotated_overlap": 0},
            },
            "by_recording": {
                "en": {"eligible": 20, "annotated_overlap": 1},
                "en-diagnostic": {"eligible": 20, "annotated_overlap": 2},
                "ja": {"eligible": 19, "annotated_overlap": 14},
                "ja-diagnostic": {"eligible": 17, "annotated_overlap": 8},
                "ko": {"eligible": 20, "annotated_overlap": 0},
                "zh": {"eligible": 0, "annotated_overlap": 0},
            },
        },
        f"Source-only inventory differs; obtain Director decision before scoring: {counts}",
    )
    pins = [
        {"path": path.as_posix(), "sha256": digest(path)}
        for path in (inputs, score, selection, original_score)
    ]
    runs = []
    selected_by_key = {(row["arm"], row["recording_id"]): row for row in selected["selected"]}
    for row in selected["selected"]:
        path = root / row["selected"]
        checked(digest(path) == row["sha256"], f"Selected log hash changed: {path}")
        pins.append({"path": Path(row["selected"]).as_posix(), "sha256": row["sha256"]})
    for arm in ARMS:
        recordings = fixed["arms"][arm]["recordings"]
        checked(len(recordings) == 6, "Expected six selected recordings per arm")
        for record in recordings:
            selected_row = selected_by_key[arm, record["recording_id"]]
            checked(record["log_sha256"] == selected_row["sha256"], "Fixed-map/log mismatch")
            checked(
                (root / record["evidence_path"]).resolve()
                == (root / selected_row["selected"]).resolve(),
                "Selected path mismatch",
            )
            runs.append(
                {
                    "arm": arm,
                    "recording_id": record["recording_id"],
                    "path": record["evidence_path"],
                    "sha256": record["log_sha256"],
                    "mapping": record["mapping"],
                }
            )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_sha256": canonical_hash(PROTOCOL),
        "original_manifest": inputs.as_posix(),
        "original_score": score.as_posix(),
        "preserved_files": pins,
        "inventory": counts,
        "entries": entries,
        "runs": runs,
    }


def lexical_piece(unit):
    return source_lexical_unit(unit) or (
        (unit.get("raw") or {}).get("type") not in {"spacing", "audio_event"}
        and any(unicodedata.category(char).startswith("M") for char in unit.get("text", ""))
    )


def streams(journal, cutoff):
    checked(bool(journal.run.get("snapshots")), "Exported snapshots required; no ledger fallback")
    primary, diagnostic = defaultdict(list), defaultdict(list)
    for unit in journal.snapshot(cutoff):
        key = unit["snapshot_key"]
        session = unit["session"]
        if journal.run["arm"] in {"S0", "S1"}:
            checked(key in {"soniox:final", "soniox:partial"}, "Unexpected Soniox snapshot family")
            order = (key == "soniox:partial", unit["event_seq"], unit.get("word_index") or 0)
            primary[session].append((order, unit))
        elif key == "scribe:partial" or unit.get("segment_index") is not None:
            index = unit.get("segment_index")
            checked(index is None or isinstance(index, int), "Invalid indexed segment order")
            order = (
                key == "scribe:partial",
                index if index is not None else 0,
                unit.get("word_index") or 0,
            )
            primary[session].append((order, unit))
        else:
            diagnostic[session, key].append(
                ((unit["event_seq"], unit.get("word_index") or 0), unit)
            )
    blocks = []
    for session, rows in primary.items():
        block, previous = [], None
        for _, unit in sorted(rows, key=lambda row: row[0]):
            index = unit.get("segment_index")
            if index is not None and previous is not None and index > previous + 1:
                blocks.append({"session": session, "units": block})
                block = []
            block.append(unit)
            if index is not None:
                previous = index
        if block:
            blocks.append({"session": session, "units": block})
    return blocks, [
        {
            "session": session,
            "family": key,
            "units": [unit for _, unit in sorted(rows, key=lambda row: row[0])],
        }
        for (session, key), rows in diagnostic.items()
    ]


def intersects(unit, entry):
    return (
        number(unit.get("start_s"))
        and number(unit.get("end_s"))
        and max(unit["start_s"], entry["start_s"]) < min(unit["end_s"], entry["end_s"])
    )


def mapped(unit):
    if unit.get("mapping_status") != "mapped":
        return False
    start, end = unit.get("start_s"), unit.get("end_s")
    ranges = unit.get("source_ranges") or []
    if not number(start) or not number(end) or end <= start or not ranges:
        return False
    if any(
        not number(span.get("start_s"))
        or not number(span.get("end_s"))
        or span["end_s"] <= span["start_s"]
        for span in ranges
    ):
        return False
    return (
        abs(ranges[0]["start_s"] - start) < 1e-5
        and abs(ranges[-1]["end_s"] - end) < 1e-5
        and all(abs(a["end_s"] - b["start_s"]) < 1e-5 for a, b in zip(ranges, ranges[1:]))
    )


def boundary_char(char):
    return bool(char) and (char.isalnum() or unicodedata.category(char).startswith("M"))


def boundary_supported(units, start, end, language):
    if language not in {"en", "ko"}:
        return True
    before = "".join(unit.get("text", "") for unit in units[:start])
    text = "".join(unit.get("text", "") for unit in units[start : end + 1])
    after = "".join(unit.get("text", "") for unit in units[end + 1 :])
    return not (
        (before and text and boundary_char(before[-1]) and boundary_char(text[0]))
        or (after and text and boundary_char(text[-1]) and boundary_char(after[0]))
    )


def progressing(units):
    for a, b in zip(units, units[1:]):
        for prefix in ("", "provider_"):
            if all(
                number(unit.get(prefix + edge + "_s"))
                for unit in (a, b)
                for edge in ("start", "end")
            ):
                if (
                    b[prefix + "start_s"] < a[prefix + "start_s"]
                    or b[prefix + "end_s"] < a[prefix + "end_s"]
                ):
                    return False
    return True


def evidence(unit):
    return {
        key: unit.get(key)
        for key in (
            "evidence",
            "event_seq",
            "arrival_s",
            "snapshot_key",
            "revision",
            "word_index",
            "segment_index",
            "session",
            "text",
            "start_s",
            "end_s",
            "provider_start_s",
            "provider_end_s",
            "source_ranges",
            "mapping_status",
            "speaker",
            "settled",
            "raw",
        )
    }


def occurrences(blocks, entry):
    localized, unlocalized, boundary_rejected, seen = [], [], [], set()
    for block in blocks:
        units = block["units"]
        indices = [index for index, unit in enumerate(units) if lexical_piece(unit)]
        for position, start in enumerate(indices):
            first = units[start]
            if mapped(first) and not intersects(first, entry):
                continue
            for end in indices[position:]:
                parts = [unit for unit in units[start : end + 1] if lexical_piece(unit)]
                if any(mapped(unit) and not intersects(unit, entry) for unit in parts):
                    break
                text = "".join(unit.get("text", "") for unit in units[start : end + 1])
                if normalized(text) != entry["normalized_reference"]:
                    continue
                identity = tuple(unit["evidence"] for unit in parts)
                if identity in seen:
                    continue
                seen.add(identity)
                candidate = {
                    "session": block["session"],
                    "raw_text": text,
                    "pieces": [evidence(unit) for unit in units[start : end + 1]],
                    "lexical_evidence": list(identity),
                    "left_boundary": evidence(units[start - 1]) if start else None,
                    "right_boundary": evidence(units[end + 1]) if end + 1 < len(units) else None,
                }
                if not boundary_supported(units, start, end, entry["language"]):
                    candidate["reason"] = "raw_word_boundary_unsupported"
                    boundary_rejected.append(candidate)
                elif not all(mapped(unit) for unit in parts):
                    candidate["reason"] = "exact_text_unlocalizable_timing_or_mapping"
                    unlocalized.append(candidate)
                elif not progressing(parts):
                    candidate["reason"] = "backward_source_or_provider_progression"
                    unlocalized.append(candidate)
                else:
                    localized.append(candidate)
    return {
        "localized": localized,
        "unlocalized_exact": unlocalized,
        "boundary_rejected": boundary_rejected,
    }


def observe(blocks, entry, mapping):
    found = occurrences(blocks, entry)
    result = {
        "entry_id": entry["entry_id"],
        "outcome": "unassigned",
        "reason": None,
        "occurrence_count": len(found["localized"]),
        **found,
    }
    if len(found["localized"]) > 1:
        result["reason"] = "alignment_ambiguous"
    elif len(found["localized"]) == 1:
        candidate = found["localized"][0]
        pieces = [
            piece
            for piece in candidate["pieces"]
            if piece["evidence"] in candidate["lexical_evidence"]
        ]
        identities = {
            (piece["session"], str(piece["speaker"]))
            for piece in pieces
            if piece["speaker"] not in (None, "")
        }
        missing = any(piece["speaker"] in (None, "") for piece in pieces)
        result.update(
            missing_speaker=missing,
            mixed_identity=len(identities) > 1,
            settled=all(piece["settled"] for piece in pieces),
        )
        if missing or len(identities) != 1:
            result["reason"] = "mixed_identity" if len(identities) > 1 else "missing_speaker"
        else:
            session, native = next(iter(identities))
            reference = mapping.get(session, {}).get(native)
            result.update(
                native_identity=[session, native],
                mapped_reference=reference,
                unmatched_id=reference is None,
                outcome="correct" if reference == entry["original_anchor"]["speaker"] else "wrong",
                reason=(
                    "fixed_mapping_match"
                    if reference == entry["original_anchor"]["speaker"]
                    else "fixed_mapping_wrong_or_unmatched"
                ),
            )
    elif found["unlocalized_exact"]:
        result["reason"] = "exact_text_unlocalizable_timing_or_mapping"
    elif found["boundary_rejected"]:
        result["reason"] = "raw_word_boundary_unsupported"
    else:
        pieces = [unit for block in blocks for unit in block["units"] if lexical_piece(unit)]
        local = [unit for unit in pieces if intersects(unit, entry)]
        result["reason"] = (
            "exact_lexical_mismatch"
            if local
            else (
                "unlocalizable_text_or_timing"
                if any(not mapped(unit) for unit in pieces)
                else "missing_localizable_text" if pieces else "missing_text"
            )
        )
        result["local_evidence"] = [evidence(unit) for unit in local]
    return result


def summarize(rows, entries):
    indexed = {entry["entry_id"]: entry for entry in entries}
    groups = defaultdict(list)
    for row in rows:
        entry = indexed[row["entry_id"]]
        if not entry["eligible"]:
            continue
        for label in (
            "pooled_descriptive_only",
            f"language:{entry['language']}:{entry['granularity']}",
            f"language:{entry['language']}:{entry['granularity']}:annotated_overlap:{entry['annotated_overlap']}",
            f"recording:{entry['recording_id']}",
        ):
            groups[row["arm"], row["view"], label].append(row)
    result = []
    for (arm, view, stratum), members in sorted(groups.items()):
        counts = Counter(row["outcome"] for row in members)
        correct, wrong = counts["correct"], counts["wrong"]
        assigned, eligible = correct + wrong, len(members)
        result.append(
            {
                "arm": arm,
                "view": view,
                "stratum": stratum,
                "eligible": eligible,
                "unique_source_units": len(
                    {indexed[row["entry_id"]]["source_occurrence_id"] for row in members}
                ),
                "correct": correct,
                "wrong": wrong,
                "unassigned": counts["unassigned"],
                "coverage": assigned / eligible,
                "conditional_accuracy": correct / assigned if assigned else None,
                "correct_coverage": correct / eligible,
                "reasons": dict(Counter(row["reason"] for row in members)),
            }
        )
    return result


def verify_frozen(frozen, root):
    checked(frozen["protocol"] == PROTOCOL, "Protocol differs from frozen implementation")
    checked(frozen["protocol_sha256"] == canonical_hash(PROTOCOL), "Protocol hash mismatch")
    for pin in frozen["preserved_files"]:
        checked(
            digest(root / pin["path"]) == pin["sha256"], f"Preserved file changed: {pin['path']}"
        )
    manifest, score = load(root / frozen["original_manifest"]), load(
        root / frozen["original_score"]
    )
    checked(derive_entries(manifest, root) == frozen["entries"], "Source-only entries changed")
    checked(inventory(frozen["entries"]) == frozen["inventory"], "Frozen inventory changed")
    for run in frozen["runs"]:
        record = next(
            row
            for row in score["arms"][run["arm"]]["recordings"]
            if row["recording_id"] == run["recording_id"]
        )
        checked(record["mapping"] == run["mapping"], "Fixed original mapping changed")
        checked(
            record["log_sha256"] == run["sha256"] == digest(root / run["path"]),
            "Selected log changed",
        )
    return {
        "preserved_file_count": len(frozen["preserved_files"]),
        "source_entries_equal": True,
        "fixed_maps_equal": True,
        "original_anchor_parent_cutoffs_equal": True,
    }


def compare(frozen, root):
    before = verify_frozen(frozen, root)
    rows = []
    for selected in frozen["runs"]:
        run = load(root / selected["path"])
        checked(
            (run["arm"], run["recording_id"]) == (selected["arm"], selected["recording_id"]),
            "Run identity mismatch",
        )
        journal = Journal(run, selected["path"])
        for entry in frozen["entries"]:
            if entry["recording_id"] != selected["recording_id"]:
                continue
            for view, cutoff in (("online", entry["online_cutoff_s"]), ("eventual", math.inf)):
                base = {
                    "arm": selected["arm"],
                    "view": view,
                    "entry_id": entry["entry_id"],
                    "cutoff_s": cutoff if math.isfinite(cutoff) else None,
                }
                if not entry["eligible"]:
                    rows.append(
                        {
                            **base,
                            "outcome": "source_ineligible",
                            "reason": entry["eligibility_reason"],
                        }
                    )
                    continue
                primary, diagnostic = streams(journal, cutoff)
                observation = observe(primary, entry, selected["mapping"])
                observation["unindexed_text_timing_diagnostic"] = occurrences(diagnostic, entry)
                rows.append({**base, **observation})
    after = verify_frozen(frozen, root)
    checked(before == after, "Invariance proof changed during comparison")
    return {
        "protocol_sha256": frozen["protocol_sha256"],
        "inventory": frozen["inventory"],
        "invariance_before": before,
        "invariance_after": after,
        "rows": rows,
        "summary": summarize(rows, frozen["entries"]),
        "limits": PROTOCOL["limits"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, default=Path("benchmarks/asr/issue441-inputs.json"))
    prep.add_argument(
        "--score", type=Path, default=Path(".tmp/issue441/scores/score-reasons-repaired.json")
    )
    prep.add_argument(
        "--selection", type=Path, default=Path(".tmp/issue441/selected-runs/selection-index.json")
    )
    prep.add_argument(
        "--original-score", type=Path, default=Path(".tmp/issue441/scores/score.json")
    )
    prep.add_argument(
        "--output", type=Path, default=Path("benchmarks/asr/issue441-lexical-inputs.json")
    )
    scoring = subparsers.add_parser("compare")
    scoring.add_argument(
        "--frozen", type=Path, default=Path("benchmarks/asr/issue441-lexical-inputs.json")
    )
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path.cwd()
    if args.command == "prepare":
        result = prepare(root, args.inputs, args.score, args.selection, args.original_score)
    else:
        frozen = load(args.frozen)
        result = compare(frozen, root)
        result["frozen_inputs_sha256"] = digest(args.frozen)
    save_new(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest(args.output),
                "protocol_sha256": canonical_hash(PROTOCOL),
            }
        )
    )


if __name__ == "__main__":
    main()
