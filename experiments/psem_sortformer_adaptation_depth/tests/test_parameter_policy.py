import pytest

from experiments.psem_sortformer_adaptation_depth.parameter_policy import (
    ParameterPolicyError,
    audit_parameter_graph,
)


def _names() -> list[str]:
    values = []
    for index in range(18):
        values.extend(
            (
                f"sortformer.transformer_encoder.layers.{index}.self_attention.weight",
                f"sortformer.transformer_encoder.layers.{index}.first_norm.weight",
            )
        )
    values.extend(
        (
            "sortformer.encoder.layers.0.weight",
            "sortformer.sortformer_modules.encoder_proj.weight",
            "sortformer.sortformer_modules.first_hidden_to_hidden.weight",
            "sortformer.sortformer_modules.single_hidden_to_spks.weight",
            "psem_head.input_norm.weight",
            "psem_head.gru.weight_ih_l0",
            "psem_head.anchor_present.weight",
            "psem_head.replacement_evidence.weight",
        )
    )
    return values


def test_arm_whitelists_freeze_the_acoustic_encoder() -> None:
    audit = audit_parameter_graph(_names())
    assert audit["trainable_by_arm"]["F0-FROZEN-FLOAT"] == []
    assert all(name.startswith("psem_head.") for name in audit["trainable_by_arm"]["H-HEAD"])
    top = audit["trainable_by_arm"]["T2-TOP"]
    assert any("layers.16." in name for name in top)
    assert any("layers.17." in name for name in top)
    assert not any("layers.15." in name for name in top)
    assert not any("sortformer.encoder." in name for name in top)
    temporal = audit["trainable_by_arm"]["TA-ALL-TEMPORAL"]
    assert any("layers.0." in name for name in temporal)
    assert any("layers.17." in name for name in temporal)
    assert not any("sortformer.encoder." in name for name in temporal)


def test_graph_audit_rejects_a_missing_temporal_layer() -> None:
    names = [name for name in _names() if ".layers.17." not in name]
    with pytest.raises(ParameterPolicyError):
        audit_parameter_graph(names)
