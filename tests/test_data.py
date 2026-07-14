"""build_contamination_stream: rising-defect-rate stream over cached embeddings."""

import numpy as np
import pytest

from cadcms.data import build_contamination_stream


def _make_pool(n_normal: int = 50, n_defect: int = 50, dim: int = 4):
    rng = np.random.default_rng(0)
    normal_embeddings = rng.normal(loc=0.0, size=(n_normal, dim))
    defect_embeddings = rng.normal(loc=5.0, size=(n_defect, dim))
    embeddings = np.concatenate([normal_embeddings, defect_embeddings], axis=0)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_defect)])
    return embeddings, labels


def test_stream_shape_and_t_range():
    embeddings, labels = _make_pool()

    stream_embeddings, stream_labels, t = build_contamination_stream(
        embeddings, labels, length=200, start_defect_rate=0.05, end_defect_rate=0.6, seed=0
    )

    assert stream_embeddings.shape == (200, embeddings.shape[1])
    assert stream_labels.shape == (200,)
    assert t[0] == 0.0
    assert t[-1] == 1.0
    assert set(np.unique(stream_labels)) <= {0.0, 1.0}


def test_defect_rate_rises_from_start_to_end():
    embeddings, labels = _make_pool()

    _, stream_labels, t = build_contamination_stream(
        embeddings, labels, length=2000, start_defect_rate=0.05, end_defect_rate=0.6, seed=0
    )

    first_quarter = stream_labels[: len(stream_labels) // 4]
    last_quarter = stream_labels[-len(stream_labels) // 4 :]

    # With 2000 samples the empirical rate should land close to the target
    # rate at each end (each quarter still spans a range of rates, so use a
    # generous tolerance rather than pinning to exactly 0.05/0.6).
    assert first_quarter.mean() < 0.2
    assert last_quarter.mean() > 0.45
    assert first_quarter.mean() < last_quarter.mean()


def test_constant_rate_is_reproducible_given_seed():
    embeddings, labels = _make_pool()

    _, labels_a, _ = build_contamination_stream(
        embeddings, labels, length=100, start_defect_rate=0.3, end_defect_rate=0.3, seed=42
    )
    _, labels_b, _ = build_contamination_stream(
        embeddings, labels, length=100, start_defect_rate=0.3, end_defect_rate=0.3, seed=42
    )

    np.testing.assert_array_equal(labels_a, labels_b)


def test_different_seeds_give_different_streams():
    embeddings, labels = _make_pool()

    _, labels_a, _ = build_contamination_stream(
        embeddings, labels, length=100, start_defect_rate=0.3, end_defect_rate=0.3, seed=1
    )
    _, labels_b, _ = build_contamination_stream(
        embeddings, labels, length=100, start_defect_rate=0.3, end_defect_rate=0.3, seed=2
    )

    assert not np.array_equal(labels_a, labels_b)


def test_raises_without_both_classes():
    embeddings, labels = _make_pool(n_normal=10, n_defect=0)

    with pytest.raises(ValueError):
        build_contamination_stream(embeddings, labels, length=10, start_defect_rate=0.1, end_defect_rate=0.5)
