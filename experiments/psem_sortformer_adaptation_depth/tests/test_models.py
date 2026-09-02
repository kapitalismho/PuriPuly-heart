import pytest
import torch

from experiments.psem_sortformer_adaptation_depth.models import (
    NATIVE_SORTFORMER_CHECKPOINT_SHA256,
    NATIVE_SORTFORMER_LOSS_KIND,
    NATIVE_SORTFORMER_LOSS_ORIGIN,
    NativeSortformerLoss,
    PSEMHead,
    bind_native_sortformer_loss,
    build_psem_features,
    composite_loss,
    masked_balanced_bce_with_logits,
)


def _features(frames: int, reset_frame: int | None = None) -> torch.Tensor:
    hidden = torch.randn(2, frames, 192)
    logits = torch.randn(2, frames, 4)
    alive = torch.ones(2, frames, 4, dtype=torch.bool)
    anchor = torch.zeros(2, frames, 4)
    anchor[..., 1] = 1
    reset = torch.zeros(2, frames, 1)
    if reset_frame is not None:
        reset[:, reset_frame] = 1
    delay = torch.full((2, frames, 1), 1.04)
    return build_psem_features(hidden, logits, alive, anchor, reset, delay)


def test_feature_interface_is_exactly_208_dimensions() -> None:
    assert _features(7).shape == (2, 7, 208)


def test_head_is_causal_for_a_shared_prefix() -> None:
    torch.manual_seed(107)
    head = PSEMHead().eval()
    features = _features(9)
    prefix, _ = head(features[:, :5])
    complete, _ = head(features)
    assert torch.allclose(
        prefix["anchor_present"], complete["anchor_present"][:, :5], atol=1e-7, rtol=0
    )
    assert torch.allclose(
        prefix["replacement_evidence"],
        complete["replacement_evidence"][:, :5],
        atol=1e-7,
        rtol=0,
    )


def test_reset_indicator_clears_prior_gru_state() -> None:
    torch.manual_seed(107)
    head = PSEMHead().eval()
    before = _features(4)
    _, state = head(before)
    after = _features(3, reset_frame=0)
    reset_outputs, _ = head(after, state)
    fresh_outputs, _ = head(after)
    assert torch.allclose(
        reset_outputs["anchor_present"], fresh_outputs["anchor_present"], atol=1e-7, rtol=0
    )
    assert torch.allclose(
        reset_outputs["replacement_evidence"],
        fresh_outputs["replacement_evidence"],
        atol=1e-7,
        rtol=0,
    )


def test_sequence_start_reset_fast_path_matches_framewise_recurrence() -> None:
    torch.manual_seed(107)
    head = PSEMHead().eval()
    features = _features(12, reset_frame=0)
    framewise, _ = head(features)
    fast, _ = head(features, sequence_start_reset_only=True)
    assert torch.allclose(framewise["anchor_present"], fast["anchor_present"], atol=1e-6, rtol=0)
    assert torch.allclose(
        framewise["replacement_evidence"], fast["replacement_evidence"], atol=1e-6, rtol=0
    )


def test_composite_loss_uses_the_frozen_weights() -> None:
    head = PSEMHead()
    outputs, _ = head(_features(4))
    targets = torch.zeros(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    native = torch.tensor(2.0, requires_grad=True)
    roles = ("PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-TRAIN")
    bound_native = bind_native_sortformer_loss(
        native,
        sampling_roles=roles,
        kind=NATIVE_SORTFORMER_LOSS_KIND,
        origin=NATIVE_SORTFORMER_LOSS_ORIGIN,
        checkpoint_sha256=NATIVE_SORTFORMER_CHECKPOINT_SHA256,
    )
    losses = composite_loss(
        outputs,
        replacement_targets=targets,
        anchor_targets=targets,
        mask=mask,
        replacement_positive_weight=1.0,
        anchor_positive_weight=1.0,
        sampling_roles=roles,
        native_sortformer_loss=bound_native,
    )
    expected = losses["replacement"] + 0.5 * losses["anchor"] + 0.5 * native
    assert torch.equal(losses["total"], expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("alive", 0.5),
        ("anchor", 0.5),
        ("reset", 0.5),
        ("delay", 1.05),
    ),
)
def test_feature_semantics_reject_invalid_values(field: str, value: float) -> None:
    hidden = torch.randn(1, 2, 192)
    logits = torch.randn(1, 2, 4)
    alive = torch.ones(1, 2, 4)
    anchor = torch.zeros(1, 2, 4)
    anchor[..., 1] = 1
    reset = torch.zeros(1, 2, 1)
    delay = torch.full((1, 2, 1), 1.04)
    values = {"alive": alive, "anchor": anchor, "reset": reset, "delay": delay}
    values[field][..., 0] = value
    with pytest.raises(ValueError):
        build_psem_features(
            hidden,
            logits,
            values["alive"],
            values["anchor"],
            values["reset"],
            values["delay"],
        )


def test_anchor_field_must_be_one_hot() -> None:
    hidden = torch.randn(1, 2, 192)
    logits = torch.randn(1, 2, 4)
    alive = torch.ones(1, 2, 4)
    anchor = torch.zeros(1, 2, 4)
    reset = torch.zeros(1, 2, 1)
    delay = torch.full((1, 2, 1), 1.04)
    with pytest.raises(ValueError, match="one-hot"):
        build_psem_features(hidden, logits, alive, anchor, reset, delay)


def test_delay_field_rejects_integer_dtype() -> None:
    hidden = torch.randn(1, 2, 192)
    logits = torch.randn(1, 2, 4)
    alive = torch.ones(1, 2, 4)
    anchor = torch.zeros(1, 2, 4)
    anchor[..., 1] = 1
    reset = torch.zeros(1, 2, 1)
    delay = torch.ones(1, 2, 1, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating-point"):
        build_psem_features(hidden, logits, alive, anchor, reset, delay)


@pytest.mark.parametrize(("target", "mask"), ((0.5, 1.0), (0.0, -1.0)))
def test_masked_loss_rejects_nonbinary_inputs(target: float, mask: float) -> None:
    with pytest.raises(ValueError):
        masked_balanced_bce_with_logits(
            torch.zeros(1, 1),
            torch.full((1, 1), target),
            torch.full((1, 1), mask),
            1.0,
        )


def test_masked_loss_returns_differentiable_zero_without_valid_targets() -> None:
    logits = torch.zeros(1, 2, requires_grad=True)
    loss = masked_balanced_bce_with_logits(
        logits,
        torch.zeros_like(logits),
        torch.zeros_like(logits),
        1.0,
    )
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_masked_loss_is_invariant_to_microbatch_grouping() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [-2.0, 1.0, 3.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    batched = masked_balanced_bce_with_logits(logits, targets, mask, 2.0)
    individual = torch.stack(
        [
            masked_balanced_bce_with_logits(
                logits[index : index + 1],
                targets[index : index + 1],
                mask[index : index + 1],
                2.0,
            )
            for index in range(2)
        ]
    ).mean()
    assert torch.allclose(batched, individual, rtol=0, atol=1e-7)


def test_native_loss_binding_accepts_dev_but_rejects_eval_or_mixed_roles() -> None:
    bound = bind_native_sortformer_loss(
        torch.tensor(1.0),
        sampling_roles=("PSEM-STRATEGY-DEV",),
        kind=NATIVE_SORTFORMER_LOSS_KIND,
        origin=NATIVE_SORTFORMER_LOSS_ORIGIN,
        checkpoint_sha256=NATIVE_SORTFORMER_CHECKPOINT_SHA256,
    )
    assert bound.sampling_roles == ("PSEM-STRATEGY-DEV",)
    with pytest.raises(ValueError, match="TRAIN or DEV"):
        bind_native_sortformer_loss(
            torch.tensor(1.0),
            sampling_roles=("PSEM-STRATEGY-EVAL",),
            kind=NATIVE_SORTFORMER_LOSS_KIND,
            origin=NATIVE_SORTFORMER_LOSS_ORIGIN,
            checkpoint_sha256=NATIVE_SORTFORMER_CHECKPOINT_SHA256,
        )
    with pytest.raises(ValueError, match="homogeneous"):
        bind_native_sortformer_loss(
            torch.tensor(1.0),
            sampling_roles=("PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-DEV"),
            kind=NATIVE_SORTFORMER_LOSS_KIND,
            origin=NATIVE_SORTFORMER_LOSS_ORIGIN,
            checkpoint_sha256=NATIVE_SORTFORMER_CHECKPOINT_SHA256,
        )


@pytest.mark.parametrize(
    ("kind", "origin", "checkpoint"),
    (
        ("other", NATIVE_SORTFORMER_LOSS_ORIGIN, NATIVE_SORTFORMER_CHECKPOINT_SHA256),
        (NATIVE_SORTFORMER_LOSS_KIND, "other", NATIVE_SORTFORMER_CHECKPOINT_SHA256),
        (NATIVE_SORTFORMER_LOSS_KIND, NATIVE_SORTFORMER_LOSS_ORIGIN, "0" * 64),
    ),
)
def test_native_loss_binding_rejects_wrong_provenance(
    kind: str, origin: str, checkpoint: str
) -> None:
    with pytest.raises(ValueError):
        bind_native_sortformer_loss(
            torch.tensor(1.0),
            sampling_roles=("PSEM-STRATEGY-TRAIN",),
            kind=kind,
            origin=origin,
            checkpoint_sha256=checkpoint,
        )


def test_composite_loss_rejects_unbound_native_loss() -> None:
    outputs = {
        "replacement_evidence": torch.zeros(1, 2),
        "anchor_present": torch.zeros(1, 2),
    }
    forged = NativeSortformerLoss(
        value=torch.tensor(1.0),
        sampling_roles=("PSEM-STRATEGY-TRAIN",),
        _token=object(),
    )
    with pytest.raises(ValueError, match="provenance"):
        composite_loss(
            outputs,
            replacement_targets=torch.zeros(1, 2),
            anchor_targets=torch.zeros(1, 2),
            mask=torch.ones(1, 2),
            replacement_positive_weight=1.0,
            anchor_positive_weight=1.0,
            sampling_roles=("PSEM-STRATEGY-TRAIN",),
            native_sortformer_loss=forged,
        )
