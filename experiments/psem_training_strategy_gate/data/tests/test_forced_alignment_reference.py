from __future__ import annotations

import subprocess
from decimal import getcontext
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    ForcedAlignmentReferenceError,
    acquire_reference,
    build_alimeeting_speaker_map,
    build_ami_speaker_map,
    build_reference_inventory,
    parse_rttm,
    resolve_reference_path,
    validate_reference_checkout,
)


def write_rttm(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def speaker_row(
    session: str,
    start: str,
    duration: str,
    speaker: str,
) -> str:
    return f"SPEAKER {session} 1 {start} {duration} <NA> <NA> {speaker} <NA> <NA>"


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def test_ami_rttm_maps_agents_unions_touching_rows_and_preserves_overlap(
    tmp_path: Path,
) -> None:
    path = write_rttm(
        tmp_path / "IS1009a.rttm",
        [
            speaker_row("IS1009a", "1.000", "0.500", "IS1009a.A"),
            speaker_row("IS1009a", "1.500", "0.250", "IS1009a.A"),
            speaker_row("IS1009a", "1.250", "0.500", "IS1009a.B"),
        ],
    )
    parsed = parse_rttm(
        path,
        corpus="AMI",
        session_id="IS1009a",
        speaker_map={"A": "FIE088", "B": "FIO084"},
        scored_start_sample=0,
        scored_end_sample=32000,
    )
    assert parsed.raw_row_count == 3
    assert parsed.clipped_tail_row_count == 0
    assert parsed.raw_speaker_ids == ("IS1009a.A", "IS1009a.B")
    assert [span.to_dict() for span in parsed.spans] == [
        {
            "start_sample": 16000,
            "end_sample": 28000,
            "speaker_id": "FIE088",
            "source_annotation_ids": ["IS1009a.rttm#L1", "IS1009a.rttm#L2"],
        },
        {
            "start_sample": 20000,
            "end_sample": 28000,
            "speaker_id": "FIO084",
            "source_annotation_ids": ["IS1009a.rttm#L3"],
        },
    ]


def test_alimeeting_rttm_maps_official_speaker_identity(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "R8001_M8004.rttm",
        [speaker_row("R8001_M8004", "0.00003125", "0.00009375", "N_SPK8013")],
    )
    parsed = parse_rttm(
        path,
        corpus="AliMeeting",
        session_id="R8001_M8004",
        speaker_map={"SPK8013": "SPK8013"},
        scored_start_sample=0,
        scored_end_sample=2,
    )
    assert parsed.spans[0].start_sample == 0
    assert parsed.spans[0].end_sample == 2
    assert parsed.spans[0].speaker_id == "SPK8013"


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("", "ten fields"),
        (speaker_row("other", "0", "1", "IS1009a.A"), "row identity"),
        (speaker_row("IS1009a", "nan", "1", "IS1009a.A"), "non-finite"),
        (speaker_row("IS1009a", "-0.1", "1", "IS1009a.A"), "invalid RTTM bounds"),
        (speaker_row("IS1009a", "0", "0", "IS1009a.A"), "invalid RTTM bounds"),
        (speaker_row("IS1009a", "0", "1.1", "IS1009a.A"), "scored range"),
        (speaker_row("IS1009a", "0", "1", "IS1009a.B"), "unmapped RTTM speaker"),
    ],
)
def test_rttm_fails_closed_on_malformed_rows(
    tmp_path: Path,
    row: str,
    match: str,
) -> None:
    path = write_rttm(tmp_path / "IS1009a.rttm", [row])
    with pytest.raises(ForcedAlignmentReferenceError, match=match):
        parse_rttm(
            path,
            corpus="AMI",
            session_id="IS1009a",
            speaker_map={"A": "FIE088"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_rttm_rejects_noninjective_speaker_mapping(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "IS1009a.rttm",
        [speaker_row("IS1009a", "0", "1", "IS1009a.A")],
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="not one-to-one"):
        parse_rttm(
            path,
            corpus="AMI",
            session_id="IS1009a",
            speaker_map={"A": "speaker", "B": "speaker"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_ami_rttm_clips_only_one_terminal_timestamp_quantum(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "ES2010b.rttm",
        [speaker_row("ES2010b", "0.500", "0.5003125", "ES2010b.A")],
    )
    parsed = parse_rttm(
        path,
        corpus="AMI",
        session_id="ES2010b",
        speaker_map={"A": "speaker-a"},
        scored_start_sample=0,
        scored_end_sample=16000,
    )
    assert parsed.clipped_tail_row_count == 1
    assert parsed.spans[0].end_sample == 16000
    path = write_rttm(
        tmp_path / "ES2010b.rttm",
        [speaker_row("ES2010b", "0.500", "0.5010625", "ES2010b.A")],
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="scored range"):
        parse_rttm(
            path,
            corpus="AMI",
            session_id="ES2010b",
            speaker_map={"A": "speaker-a"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_alimeeting_rttm_never_uses_ami_source_tail_rule(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "R8001_M8004.rttm",
        [speaker_row("R8001_M8004", "0.500", "0.5003125", "N_SPK8013")],
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="scored range"):
        parse_rttm(
            path,
            corpus="AliMeeting",
            session_id="R8001_M8004",
            speaker_map={"SPK8013": "SPK8013"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_rttm_checks_raw_bounds_before_sample_rounding(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "IS1009a.rttm",
        [speaker_row("IS1009a", "0.999999", "0.1", "IS1009a.A")],
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="scored range"):
        parse_rttm(
            path,
            corpus="AMI",
            session_id="IS1009a",
            speaker_map={"A": "FIE088"},
            scored_start_sample=16000,
            scored_end_sample=32000,
        )
    path = write_rttm(
        tmp_path / "IS1009a.rttm",
        [speaker_row("IS1009a", "0.5", "0.50003125", "IS1009a.A")],
    )
    parsed = parse_rttm(
        path,
        corpus="AMI",
        session_id="IS1009a",
        speaker_map={"A": "FIE088"},
        scored_start_sample=0,
        scored_end_sample=16000,
    )
    assert parsed.clipped_tail_row_count == 1
    assert parsed.spans[0].end_sample == 16000


def test_rttm_conversion_is_independent_of_decimal_context(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "IS1009a.rttm",
        [speaker_row("IS1009a", "1.00003125", "0.10003125", "IS1009a.A")],
    )
    previous_precision = getcontext().prec
    try:
        getcontext().prec = 1
        parsed = parse_rttm(
            path,
            corpus="AMI",
            session_id="IS1009a",
            speaker_map={"A": "FIE088"},
            scored_start_sample=0,
            scored_end_sample=32000,
        )
    finally:
        getcontext().prec = previous_precision
    assert parsed.spans[0].start_sample == 16000
    assert parsed.spans[0].end_sample == 17601
    path = write_rttm(
        tmp_path / "IS1009a.rttm",
        [speaker_row("IS1009a", "1e999999999", "1", "IS1009a.A")],
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="out-of-range"):
        parse_rttm(
            path,
            corpus="AMI",
            session_id="IS1009a",
            speaker_map={"A": "FIE088"},
            scored_start_sample=0,
            scored_end_sample=32000,
        )


def test_alimeeting_parser_rejects_caller_forged_identity(tmp_path: Path) -> None:
    path = write_rttm(
        tmp_path / "R8001_M8004.rttm",
        [speaker_row("R8001_M8004", "0", "1", "N_SPK8013")],
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="noncanonical"):
        parse_rttm(
            path,
            corpus="AliMeeting",
            session_id="R8001_M8004",
            speaker_map={"SPK8013": "forged"},
            scored_start_sample=0,
            scored_end_sample=16000,
        )


def test_reference_path_requires_exactly_one_allowed_partition(tmp_path: Path) -> None:
    write_rttm(tmp_path / "AMI" / "train" / "ES2005a.rttm", ["row"])
    resolved = resolve_reference_path(tmp_path, corpus="AMI", session_id="ES2005a")
    assert resolved == (tmp_path / "AMI" / "train" / "ES2005a.rttm").resolve()
    write_rttm(tmp_path / "AMI" / "dev" / "ES2005a.rttm", ["row"])
    with pytest.raises(ForcedAlignmentReferenceError, match="found 2"):
        resolve_reference_path(tmp_path, corpus="AMI", session_id="ES2005a")
    with pytest.raises(ForcedAlignmentReferenceError, match="found 0"):
        resolve_reference_path(tmp_path, corpus="AliMeeting", session_id="R0001_M0001")


def test_official_corpus_speaker_maps_are_fail_closed(tmp_path: Path) -> None:
    meetings = tmp_path / "meetings.xml"
    meetings.write_text(
        '<root><meeting observation="IS1009a">'
        '<speaker nxt_agent="A" global_name="FIE088" />'
        '<speaker nxt_agent="B" global_name="FIO084" />'
        "</meeting></root>",
        encoding="utf-8",
    )
    assert build_ami_speaker_map(meetings, "IS1009a") == {
        "A": "FIE088",
        "B": "FIO084",
    }
    assert build_alimeeting_speaker_map(["SPK8013", "SPK8014"]) == {
        "SPK8013": "SPK8013",
        "SPK8014": "SPK8014",
    }
    meetings.write_text(
        '<root><meeting observation="IS1009a">'
        '<speaker nxt_agent="A" />'
        "</meeting></root>",
        encoding="utf-8",
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="unresolved"):
        build_ami_speaker_map(meetings, "IS1009a")
    meetings.write_text(
        '<root><meeting observation="IS1009a">'
        '<speaker nxt_agent=" " global_name="FIE088" />'
        "</meeting></root>",
        encoding="utf-8",
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="unresolved"):
        build_ami_speaker_map(meetings, "IS1009a")
    with pytest.raises(ForcedAlignmentReferenceError, match="inventory is invalid"):
        build_alimeeting_speaker_map(["SPK8013", "SPK8013"])


def initialize_reference_checkout(tmp_path: Path) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init")
    git(origin, "config", "user.name", "Test")
    git(origin, "config", "user.email", "test@example.invalid")
    (origin / "LICENSE").write_text("license\n", encoding="utf-8")
    (origin / "README.md").write_text("readme\n", encoding="utf-8")
    (origin / ".gitignore").write_text("ignored.rttm\n", encoding="utf-8")
    write_rttm(
        origin / "AMI" / "train" / "ES2005a.rttm",
        [speaker_row("ES2005a", "0", "1", "ES2005a.A")],
    )
    git(origin, "add", ".")
    git(origin, "commit", "-m", "fixture")
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", str(origin), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return checkout


def test_checkout_validation_rejects_wrong_repository_commit_and_dirty_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        git(checkout, "rev-parse", "HEAD"),
    )
    with pytest.raises(ForcedAlignmentReferenceError, match="repository mismatch"):
        validate_reference_checkout(checkout)
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    validated = validate_reference_checkout(checkout)
    assert validated["commit"] == git(checkout, "rev-parse", "HEAD")
    (checkout / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ForcedAlignmentReferenceError, match="differ"):
        validate_reference_checkout(checkout)


def test_checkout_validation_rejects_ignored_skip_worktree_and_sparse_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        git(checkout, "rev-parse", "HEAD"),
    )
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    (checkout / "ignored.rttm").write_text("ignored\n", encoding="utf-8")
    with pytest.raises(ForcedAlignmentReferenceError, match="untracked or ignored"):
        validate_reference_checkout(checkout)
    (checkout / "ignored.rttm").unlink()
    rttm = checkout / "AMI" / "train" / "ES2005a.rttm"
    git(checkout, "update-index", "--skip-worktree", rttm.relative_to(checkout).as_posix())
    rttm.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ForcedAlignmentReferenceError, match="index flags"):
        validate_reference_checkout(checkout)
    git(checkout, "update-index", "--no-skip-worktree", rttm.relative_to(checkout).as_posix())
    git(checkout, "checkout", "--", rttm.relative_to(checkout).as_posix())
    git(checkout, "config", "core.sparseCheckout", "true")
    with pytest.raises(ForcedAlignmentReferenceError, match="sparse"):
        validate_reference_checkout(checkout)


def test_checkout_validation_rejects_checkout_filter_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        git(checkout, "rev-parse", "HEAD"),
    )
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    rttm = checkout / "AMI" / "train" / "ES2005a.rttm"
    rttm.write_bytes(rttm.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ForcedAlignmentReferenceError, match="tracked bytes differ"):
        validate_reference_checkout(checkout)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("remote.origin.promisor", "true"),
        ("extensions.partialClone", "origin"),
        ("remote.origin.partialCloneFilter", "blob:none"),
    ],
)
def test_checkout_validation_rejects_every_partial_clone_config_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        git(checkout, "rev-parse", "HEAD"),
    )
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    git(checkout, "config", key, value)
    with pytest.raises(ForcedAlignmentReferenceError, match="partial"):
        validate_reference_checkout(checkout)


def test_checkout_validation_rejects_promisor_pack_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        git(checkout, "rev-parse", "HEAD"),
    )
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    pack_root = checkout / ".git" / "objects" / "pack"
    (pack_root / "fixture.promisor").write_bytes(b"")
    with pytest.raises(ForcedAlignmentReferenceError, match="partial"):
        validate_reference_checkout(checkout)


def test_reference_inventory_binds_checkout_file_and_canonical_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    fixture_commit = git(checkout, "rev-parse", "HEAD")
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        fixture_commit,
    )
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    inventory = build_reference_inventory(
        checkout,
        [
            {
                "source_id": "ami_ES2005a",
                "corpus": "AMI",
                "session_id": "ES2005a",
                "speaker_map": {"A": "speaker-a"},
                "scored_start_sample": 0,
                "scored_end_sample": 16000,
            }
        ],
    )
    assert inventory["artifact_role"] == "psem_forced_alignment_reference_provenance"
    assert inventory["upstream"]["commit"] == fixture_commit
    assert inventory["source_count"] == 1
    assert inventory["sources"][0]["reference_ref"] == "AMI/train/ES2005a.rttm"
    assert inventory["sources"][0]["mapped_speaker_ids"] == ["speaker-a"]


def test_reference_inventory_rejects_duplicate_selected_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = initialize_reference_checkout(tmp_path)
    monkeypatch.setattr(
        "experiments.psem_training_strategy_gate.data.forced_alignment_reference.REFERENCE_COMMIT",
        git(checkout, "rev-parse", "HEAD"),
    )
    git(checkout, "remote", "set-url", "origin", REFERENCE_REPOSITORY)
    selected = {
        "source_id": "ami_ES2005a",
        "corpus": "AMI",
        "session_id": "ES2005a",
        "speaker_map": {"A": "speaker-a"},
        "scored_start_sample": 0,
        "scored_end_sample": 16000,
    }
    with pytest.raises(ForcedAlignmentReferenceError, match="duplicate reference selection"):
        build_reference_inventory(
            checkout,
            [selected, {**selected, "source_id": "duplicate_ES2005a"}],
        )


def test_acquisition_reuses_only_a_valid_existing_checkout(
    tmp_path: Path,
) -> None:
    target = tmp_path / "reference"
    target.mkdir()
    with pytest.raises(ForcedAlignmentReferenceError, match="not a Git checkout"):
        acquire_reference(target)
    missing_parent_target = tmp_path / "missing" / "reference"
    with pytest.raises(ForcedAlignmentReferenceError, match="parent directory"):
        acquire_reference(missing_parent_target)


def test_pinned_reference_identity_is_exact() -> None:
    assert REFERENCE_COMMIT == "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
    assert REFERENCE_REPOSITORY == "https://github.com/nttcslab-sp/diar-forced-alignment"
