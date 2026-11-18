from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:
    from benchmarks.asr import issue441_lexical as lexical
    from benchmarks.asr import issue441_score as point
except ModuleNotFoundError:
    import issue441_lexical as lexical
    import issue441_score as point

ARM = "I441"
BASELINE = "4dab89126bcee83bc501fd95f5bbd742ef861d37"
MANIFEST = "benchmarks/asr/issue441-inputs.json"
LEXICAL_INPUTS = "benchmarks/asr/issue441-lexical-inputs.json"
SELECTION = ".tmp/issue441/scores/mt-selection-frozen.json"
DISPLAY_CASES = ("mt-01", "mt-02", "mt-07")
SUPERSEDED_DISPLAY_CASES = ("mt-03", "mt-04", "mt-05", "mt-06")
OLD_LEXICAL = ".tmp/issue441/lexical/result.json"
PINNED = {
    LEXICAL_INPUTS: "926cf06540d7d0e3f59c5835ad54426c663e8897ead7ee84aaf33a244388fa77",
    SELECTION: "a252da053f364791f799500f39054448b3f8f196d35b6c086a2284b548b89710",
    OLD_LEXICAL: "ff6ccd1dee404ed7156c706ca0333bf35a0dda3188532b8a19ca776bc77d1b1b",
    "src/puripuly_heart/data/vad/silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    "src/puripuly_heart/data/models/smart-turn-v3.2-cpu.onnx": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
}
VAD = {"threshold": 0.4, "hangover_ms": 900, "pre_roll_ms": 500}
WIRE = {
    "model_id": "scribe_v3_live",
    "audio_format": "pcm_16000",
    "commit_strategy": "manual",
    "diarize": "true",
    "include_timestamps": "true",
    "filter_background_audio": "false",
    "multilang_mode": "true",
}
DRAIN_S = 3.0


def candidate_import_root(root, candidate):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    lexical.checked(head == candidate, f"Candidate pin mismatch: {head}")
    changed = subprocess.run(["git", "diff", "--quiet", head, "--", "src"], cwd=root, check=False)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src"],
        cwd=root,
        text=True,
    )
    lexical.checked(
        not changed.returncode and not untracked.strip(), "Product sources differ from pin"
    )
    sys.path[:0] = [str(root / "src"), str(root)]
    import puripuly_heart

    lexical.checked(
        Path(puripuly_heart.__file__).resolve().is_relative_to(root / "src"),
        "Product import resolved outside the pinned source tree",
    )
    return head


def make_freeze(root):
    for path, digest in PINNED.items():
        lexical.checked(lexical.digest(root / path) == digest, f"Preserved input changed: {path}")
    frozen = lexical.load(root / LEXICAL_INPUTS)
    old = lexical.verify_frozen(frozen, root)
    manifest = lexical.load(root / MANIFEST)
    inventory = [
        {key: row[key] for key in ("id", "audio_path", "duration_s", "sha256", "language")}
        for row in manifest["recordings"]
    ]
    for row in inventory:
        path = root / row["audio_path"]
        lexical.checked(lexical.digest(path) == row["sha256"], f"WAV changed: {path}")
    point_anchors = sum(len(row["anchors"]) for row in manifest["recordings"])
    edges = sum(len(row["changes"]) for row in manifest["recordings"])
    lexical.checked((len(inventory), point_anchors, edges) == (6, 120, 24), "Corpus changed")
    return {
        "schema_version": 4,
        "superseded_preparation_file_sha256": [
            "1a0a7fa743729b81d37f700ee64acf4b0b8bf10217739f511f192f5974c4ac64",
            "4a3b756f9153b38d0c3ea56288675d51f3efd5f4f71afeaf962ccd90cc690f88",
            "4edd8c340dfa97c8583b860a3a059ae07e05d5d09ab9713b0e832ee419416f86",
        ],
        "product_baseline": BASELINE,
        "preserved": {**PINNED, **{x["path"]: x["sha256"] for x in frozen["preserved_files"]}},
        "original_invariance": old,
        "original_lexical_protocol_sha256": frozen["protocol_sha256"],
        "recordings": inventory,
        "capture": {
            "arm": ARM,
            "source": "Frozen mono PCM16/16000 WAV, 512 samples paced at source time",
            "replacement": "Controlled file/device boundary and isolated capture admission/application composition; real VAD, controller, scoped engine and Scribe adapter",
            "vad": VAD,
            "requested_wire": WIRE,
            "language_mode": "auto, no language_code or secondary language query",
            "intentionally_not_overridden": {
                "include_language_detection": "Not set explicitly by product",
                "no_verbatim": "Product default only; no transcript editing option set",
                "vad_silence_threshold_secs": "Not sent by product in manual mode",
                "vad_threshold": "Not sent by product in manual mode",
                "min_speech_duration_ms": "Not sent by product in manual mode",
                "min_silence_duration_ms": "Not sent by product in manual mode",
            },
            "per_recording_sessions": "One planned initial session; the product's own bounded recovery stays enabled; retain actual socket attempts and all provider session scopes",
            "owned_text_settlement": "Use the existing product 12s final watchdog plus its ordinary event drain. Do not count the metadata observation cap against text or time-limit the product's own recovery.",
            "experiment_post_text_metadata_window_s": DRAIN_S,
            "metadata_wait_exit": "After owned text settlement, observe only still-live sessions for up to 3 seconds, exit earlier if all observed nonempty commits have indexed metadata. Missing metadata remains explicitly unavailable/partial.",
            "transmitted_input": "Peer VAD selection/replay/padding; not E0 continuous PCM",
            "clock": "Monotonic relative to source origin; subzero startup permitted",
            "send_completion": "Completed local transport write; no implication of server PCM acceptance",
        },
        "replay": {
            "input": "Immutable E0 native messages in original receipt order and clock; historical original WS frame bytes unavailable",
            "lifecycle": "Reconstructed around observed E0 manual-controller boundaries, not a historical production recognition receipt; separate from live I441",
            "mt": "Immediate deterministic structural MT; production rendered default prompt; coherent source-ordered Peer context; no actual service latency",
            "history_comparison": "Original E0 selected MT used empty prior-turn context. Original S1 used captured production history. Candidate uses its frozen production request contexts; compare MT quality only for matched request context/settings, never rank S1 vs E0 across these contexts",
            "display": "Reuse retained E0 evidence. Do not run a full historical E0 replay campaign or additional paid historical MT.",
        },
        "fresh_application_and_display": {
            "structural": "During each planned I441 capture, forward the same production raw-adapter/scoped events through actual Peer callback/ledger admission, caption, output, and logical presenter. Replace only MT with immediate deterministic local completion for structural observations. Retain exact admission order, registrations, full prepared and started requests, terminal histories and source partitions; do not make extra ASR calls.",
            "pack": "Export actual captured streams for the retained natural display cases only, with sufficient original context. Correlate provider session plus native segment index to app terminal and actual OwnedVadEvent/turn identity and completed writes. No transcript similarity or adjacent receipt infers identity.",
            "selected_cases": list(DISPLAY_CASES),
            "superseded_display_cases": list(SUPERSEDED_DISPLAY_CASES),
            "selection": "Use the preserved source-time bounds for mt-01, mt-02 and mt-07. Select a complete non-preview I441 request only where its native session/index/character ranges contain lexical word evidence positively intersecting the frozen interval using the original capture-time source mapping. Owned range overlap without localized group evidence is unavailable; do not backfill from E0. All six native recordings remain in the unchanged score.",
            "desktop": "Prioritize actual I441 MT and real Desktop. Re-deliver the typed, actually admitted I441 application input into the same production Peer caption/output, including unchanged full warm-up. Use the actual presenter and gated declared MT only for exact selected stable request key and backend body; no mid-stream idle wait, no display card injection. Raw native join was exercised upstream during the same recorded capture, not replayed as a new provider session. Count paid dispatch, settlement and destination receipt separately.",
        },
        "scoring": {
            "anchors": point_anchors,
            "edges": edges,
            "point_algorithm": point.MAPPING_METHOD,
            "view": "Online reference span end + 1.0 s, eventual retained observations independently",
            "mapping": "One fresh global map per actual provider session from eventual I441 point anchors only; reused unchanged for I441 lexical score",
            "lexical": frozen["inventory"],
            "limits": "Native and application caption surfaces separate; not WER, CER or DER. Original N/S0/S1/E0 point and lexical results are read, not rerun",
        },
    }


def validate_capture(run, recording, candidate):
    lexical.checked(
        (run["arm"], run["recording_id"], run["metadata"]["checkout"])
        == (ARM, recording["id"], candidate),
        "Capture run identity differs from requested candidate",
    )
    lexical.checked(run["metadata"]["live_provider"], "Cannot score a controlled transport as I441")
    lexical.checked(
        run["metadata"]["input_sha256"] == recording["sha256"], "Capture WAV hash differs"
    )
    rows = run.get("snapshots", [])
    words = run.get("observations", [])
    expected = {
        row["seq"]
        for row in run["events"]
        if row["kind"] == "native"
        and row["raw"].get("message_type")
        in {
            "partial_transcript",
            "committed_transcript",
            "committed_transcript_with_timestamps",
            "diarized_transcript",
        }
    }
    lexical.checked(expected == {row["event_seq"] for row in rows}, "Missing native snapshot")
    for row in rows:
        for index in row["observation_indices"]:
            word = words[index]
            lexical.checked(
                (word["event_seq"], word["arrival_s"], word["session_id"])
                == (row["event_seq"], row["arrival_s"], row["session_id"]),
                "Observation mismatches source snapshot receipt",
            )
    scopes = {row["session_id"] for row in run["session_scopes"]}
    observed = {
        row["session_id"] for row in run["transport_sessions"] if row.get("session_started_mono_s")
    }
    lexical.checked(scopes <= observed, "Uncorrelated product provider session")
    lexical.checked(
        {row["session_id"] for row in rows} <= observed, "Native snapshots lack actual socket scope"
    )


def capture_scope(run):
    return {
        "transport_attempts": len(run["transport_sessions"]),
        "transport_sessions": run["transport_sessions"],
        "session_scopes": run["session_scopes"],
        "source_to_transmitted": run["source_to_transmitted"],
        "first_nonempty_native_text_s": next(
            (
                row["arrival_s"]
                for row in run["events"]
                if row["kind"] == "native"
                and row["raw"].get("message_type") in {"partial_transcript", "committed_transcript"}
                and str(row["raw"].get("text", "")).strip()
            ),
            None,
        ),
    }


def score_lexical(selected, journal, mapping):
    rows = []
    for entry in selected:
        for view, cutoff in (("online", entry["online_cutoff_s"]), ("eventual", math.inf)):
            base = {
                "arm": ARM,
                "view": view,
                "entry_id": entry["entry_id"],
                "cutoff_s": cutoff if math.isfinite(cutoff) else None,
            }
            if not entry["eligible"]:
                rows.append(
                    {**base, "outcome": "source_ineligible", "reason": entry["eligibility_reason"]}
                )
                continue
            primary, diagnostics = (
                lexical.streams(journal, cutoff) if journal.run["snapshots"] else ([], [])
            )
            rows.append(
                {
                    **base,
                    **lexical.observe(primary, entry, mapping),
                    "unindexed_text_timing_diagnostic": lexical.occurrences(diagnostics, entry),
                }
            )
    return rows


def compare(root, freeze, captures, candidate):
    before = make_freeze(root)
    lexical.checked(before == freeze, "Source, methods or preserved references differ from freeze")
    manifest = lexical.load(root / MANIFEST)
    reference = lexical.load(root / LEXICAL_INPUTS)
    point_rows, lexical_rows = [], []
    for recording in manifest["recordings"]:
        path = captures / f"{recording['id']}.json"
        run = lexical.load(path)
        validate_capture(run, recording, candidate)
        result = point.score_recording(recording, run, str(path), include_unavailable=True)
        result["log_sha256"] = lexical.digest(path)
        result["evidence_path"] = str(path)
        result["transport_connected"] = run["connected"]
        result["session_started"] = run["session_started"]
        result["capture_scope"] = capture_scope(run)
        point_rows.append(result)
        lexical_rows.extend(
            score_lexical(
                [
                    entry
                    for entry in reference["entries"]
                    if entry["recording_id"] == recording["id"]
                ],
                point.Journal(run, str(path)),
                result["mapping"],
            )
        )
    after = make_freeze(root)
    lexical.checked(before == after, "Invariance proof changed during comparison")

    def edge_counts(rows, view):
        cases = [case for row in rows for case in row["changes"] if case.get("view") == view]
        return {
            "eligible": len(cases),
            "available": sum(case["status"] == "observed" for case in cases),
            "same_native_id": sum(case["missed_change"] for case in cases),
            "correct_pair": sum(case["correct_identity_pair"] is True for case in cases),
        }

    languages = sorted({row["language"] for row in point_rows})
    edges = {
        view: {
            "all": edge_counts(point_rows, view),
            "by_language": {
                lang: edge_counts([row for row in point_rows if row["language"] == lang], view)
                for lang in languages
            },
        }
        for view in ("online", "eventual")
    }
    lexical.checked(
        all(row["all"]["eligible"] == freeze["scoring"]["edges"] for row in edges.values()),
        "Frozen edge denominators did not survive scoring",
    )
    totals = point.aggregate(point_rows)
    totals["connected_recordings"] = sum(row["transport_connected"] for row in point_rows)
    totals["unrun_recordings"] = len(point_rows) - totals["connected_recordings"]
    original = lexical.load(root / reference["original_score"])
    original_lexical = lexical.load(root / OLD_LEXICAL)
    return {
        "arm": ARM,
        "candidate": candidate,
        "source_method_sha256": lexical.canonical_hash(freeze),
        "unchanged_original_sha256": before["preserved"],
        "capture_status": dict(Counter(row["status"] for row in point_rows)),
        "retained_baselines": {
            "point": {arm: original["arms"][arm]["summary"] for arm in ("N", "S0", "S1", "E0")},
            "lexical": original_lexical["summary"],
            "limits": "Read retained prior evidence only; changed Scribe production input does not isolate provider quality or caption postprocessing.",
        },
        "point": {"summary": totals, "edges": edges, "recordings": point_rows},
        "lexical": {
            "summary": lexical.summarize(lexical_rows, reference["entries"]),
            "rows": lexical_rows,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "compare"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        if command == "compare":
            sub.add_argument("--frozen", type=Path, required=True)
            sub.add_argument("--captures", type=Path, required=True)
            sub.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = (
        make_freeze(args.root)
        if args.command == "freeze"
        else compare(args.root, lexical.load(args.frozen), args.captures, args.candidate)
    )
    lexical.save_new(args.output, result)


if __name__ == "__main__":
    main()
