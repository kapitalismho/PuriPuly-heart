from __future__ import annotations

import numpy as np

from experiments.speaker_representation_scd.r5_lookahead import LookaheadDataset


def test_lookahead_dataset_shifts_features_but_not_logical_labels() -> None:
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    rows = [{"vector_rows": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 1, 1, 0, 0]}]
    values, labels, _ = LookaheadDataset(vectors, rows, 1)[0]
    np.testing.assert_array_equal(values.numpy(), vectors[1:])
    np.testing.assert_array_equal(labels.numpy(), np.asarray([0, 1, 1, 1, 0]))


def test_lookahead_dataset_supports_three_hops() -> None:
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    rows = [{"vector_rows": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 1, 1, 0, 0]}]
    values, labels, _ = LookaheadDataset(vectors, rows, 3)[0]
    assert values.shape == (3, 4)
    assert labels.tolist() == [0.0, 1.0, 1.0]
