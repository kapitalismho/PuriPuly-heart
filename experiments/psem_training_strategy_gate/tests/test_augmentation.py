from __future__ import annotations

import pytest
import torch

from experiments.psem_training_strategy_gate.augmentation import (
    AUGMENTATION_FAMILIES,
    apply_augmentation,
    augmentation_decision,
    augmentation_manifest_summary,
    validate_augmentation_decision,
)


def test_augmentation_decision_is_deterministic_and_row_key_only() -> None:
    first = augmentation_decision("epoch-01-window-0000")
    second = augmentation_decision("epoch-01-window-0000")
    different = augmentation_decision("epoch-01-window-0001")
    assert first == second
    assert first != different
    assert set(first) == {"recipe_version", "decision_key", *AUGMENTATION_FAMILIES}
    validate_augmentation_decision(first)


def test_augmentation_applies_to_complete_window_deterministically() -> None:
    decision = augmentation_decision("full-window")
    waveform = torch.linspace(-0.5, 0.5, 48000)
    first = apply_augmentation(waveform, decision)
    second = apply_augmentation(waveform, decision)
    assert first.shape == waveform.shape
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0


def test_augmentation_rejects_partial_or_multichannel_windows() -> None:
    decision = augmentation_decision("invalid-window")
    with pytest.raises(RuntimeError, match="complete mono window"):
        apply_augmentation(torch.zeros(47999), decision)
    with pytest.raises(RuntimeError, match="complete mono window"):
        apply_augmentation(torch.zeros(1, 48000), decision)


def test_augmentation_summary_has_all_families_and_no_synthetic_data() -> None:
    decisions = [augmentation_decision(f"row-{index}") for index in range(100)]
    summary = augmentation_manifest_summary(decisions)
    assert summary["families"] == list(AUGMENTATION_FAMILIES)
    assert all(summary["enabled_counts"][family] > 0 for family in AUGMENTATION_FAMILIES)
    assert summary["label_fields_consulted"] == []
    assert summary["whole_window_consistency"] is True
    assert summary["synthetic_manifest"] is None
    assert summary["synthetic_optimizer_batch_fraction"] == 0.0
