from __future__ import annotations

from collections import Counter

from experiments.speaker_representation_scd.r5_data import build_grouped_split


def _row(index: int, label: str, group: str, corpus: str = "x") -> dict:
    return {
        "block_id": group,
        "candidate_id": f"candidate-{index}",
        "class": label,
        "corpus": corpus,
        "session_id": f"session-{index}",
        "synthetic_manifest": None,
        "waveform_id": f"waveform-{index}",
    }


def test_grouped_split_excludes_r4_and_keeps_groups_disjoint() -> None:
    rows = []
    for group_index in range(20):
        rows.append(_row(group_index * 2, "positive", f"group-{group_index}"))
        rows.append(_row(group_index * 2 + 1, "negative", f"group-{group_index}"))
    rows.append(
        {
            **_row(100, "positive", "excluded"),
            "session_id": "r4-session",
        }
    )
    result = build_grouped_split(
        rows,
        {"r4-session"},
        dev_fraction=0.2,
        seed=7,
        search_trials=256,
    )
    entries = result["entries"]
    assert all(row["session_id"] != "r4-session" for row in entries)
    train_groups = {row["group_id"] for row in entries if row["split"] == "train"}
    dev_groups = {row["group_id"] for row in entries if row["split"] == "dev"}
    assert train_groups.isdisjoint(dev_groups)
    for corpus in {row["corpus"] for row in entries}:
        assert any(row["corpus"] == corpus and row["split"] == "train" for row in entries)
        assert any(row["corpus"] == corpus and row["split"] == "dev" for row in entries)
    counts = Counter((row["split"], row["class"]) for row in entries)
    assert all(counts[key] > 0 for key in counts)


def test_grouped_split_is_deterministic() -> None:
    rows = [
        _row(index, "positive" if index % 2 == 0 else "negative", f"group-{index // 2}")
        for index in range(40)
    ]
    first = build_grouped_split(rows, set(), dev_fraction=0.2, seed=11, search_trials=128)
    second = build_grouped_split(rows, set(), dev_fraction=0.2, seed=11, search_trials=128)
    assert first == second
