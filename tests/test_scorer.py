"""Mahalanobis scoring, score fusion, and thresholding."""

import numpy as np

from cadcms.memory import GaussianMemory
from cadcms.scorer import fuse_scores, is_confidently_normal, mahalanobis_scores, percentile_threshold


def test_mahalanobis_scores_rank_far_points_higher():
    memory = GaussianMemory(mean=np.zeros(3), covariance=np.eye(3), count=100)
    embeddings = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    scores = mahalanobis_scores(memory, embeddings)

    assert scores[0] < scores[1] < scores[2]


def test_fuse_scores_weighting():
    medium = np.array([1.0, 2.0, 3.0])
    fast = np.array([3.0, 2.0, 1.0])

    np.testing.assert_allclose(fuse_scores(medium, fast, w=1.0), medium)
    np.testing.assert_allclose(fuse_scores(medium, fast, w=0.0), fast)
    np.testing.assert_allclose(fuse_scores(medium, fast, w=0.5), np.array([2.0, 2.0, 2.0]))


def test_percentile_threshold_and_confidence_check():
    recent_scores = list(range(1, 101))  # 1..100

    threshold = percentile_threshold(recent_scores, percentile=50)

    assert threshold == 50.5
    assert is_confidently_normal(10, threshold)
    assert not is_confidently_normal(90, threshold)
