"""Mahalanobis scoring, score fusion, and thresholding."""

from collections import deque

import numpy as np

from cadcms.memory import FastMemory, GaussianMemory
from cadcms.scorer import (
    fuse_scores,
    fused_mahalanobis_scores,
    is_confidently_normal,
    mahalanobis_scores,
    percentile_threshold,
    score_stream_with_fast_memory,
)


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


def test_fused_mahalanobis_scores_extremes_match_single_memory():
    medium = GaussianMemory(mean=np.zeros(2), covariance=np.eye(2), count=100)
    fast = FastMemory(medium, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)
    fast.update(np.array([5.0, 5.0]), medium)  # nudge fast memory away from medium

    embeddings = np.array([[1.0, 1.0], [2.0, -1.0]])

    np.testing.assert_allclose(
        fused_mahalanobis_scores(medium, fast, embeddings, w=1.0), medium.mahalanobis(embeddings)
    )
    np.testing.assert_allclose(
        fused_mahalanobis_scores(medium, fast, embeddings, w=0.0), fast.mahalanobis(embeddings)
    )


def test_score_stream_with_fast_memory_adapts_only_when_confident():
    rng = np.random.default_rng(0)
    medium = GaussianMemory(mean=np.zeros(4), covariance=np.eye(4), count=100)
    fast = FastMemory(medium, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)

    # A stream of near-medium (normal-looking) samples, all essentially at the
    # mean, so once there's history every sample should read as confidently
    # normal (score ~0, at or below whatever percentile of near-zero history).
    embeddings = rng.normal(scale=0.01, size=(50, 4))
    recent_scores: deque = deque(maxlen=200)

    scores = score_stream_with_fast_memory(embeddings, medium, fast, fusion_weight=0.5, confidence_percentile=50, recent_scores=recent_scores)

    assert len(scores) == 50
    assert fast.num_updates > 0
    assert len(recent_scores) == 50


def test_score_stream_with_fast_memory_never_updates_from_far_outliers():
    medium = GaussianMemory(mean=np.zeros(4), covariance=np.eye(4), count=100)
    fast = FastMemory(medium, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)

    # Bootstrap history with normal-looking scores, then feed one huge outlier.
    recent_scores: deque = deque([0.1, 0.2, 0.15, 0.12, 0.18] * 10, maxlen=200)
    outlier = np.array([[100.0, 100.0, 100.0, 100.0]])

    score_stream_with_fast_memory(outlier, medium, fast, fusion_weight=0.5, confidence_percentile=50, recent_scores=recent_scores)

    assert fast.num_updates == 0


def test_on_sample_hook_does_not_change_scores_or_updates():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(40, 4))

    medium = GaussianMemory(mean=np.zeros(4), covariance=np.eye(4), count=100)
    fast_plain = FastMemory(medium, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)
    fast_hooked = FastMemory(medium, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)

    scores_plain = score_stream_with_fast_memory(
        embeddings, medium, fast_plain, fusion_weight=0.5, confidence_percentile=50, recent_scores=deque(maxlen=200)
    )
    scores_hooked = score_stream_with_fast_memory(
        embeddings,
        medium,
        fast_hooked,
        fusion_weight=0.5,
        confidence_percentile=50,
        recent_scores=deque(maxlen=200),
        on_sample=lambda *args: None,
    )

    np.testing.assert_allclose(scores_plain, scores_hooked)
    assert fast_plain.num_updates == fast_hooked.num_updates
    np.testing.assert_allclose(fast_plain.mean, fast_hooked.mean)
    np.testing.assert_allclose(fast_plain.covariance, fast_hooked.covariance)


def test_on_sample_hook_reports_correct_gate_decisions():
    medium = GaussianMemory(mean=np.zeros(4), covariance=np.eye(4), count=100)
    fast = FastMemory(medium, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)

    # Bootstrap history with small scores, then one huge outlier that must fail the gate.
    recent_scores: deque = deque([0.1, 0.2, 0.15, 0.12, 0.18] * 10, maxlen=200)
    embeddings = np.array([[0.0, 0.0, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]])

    records = []
    score_stream_with_fast_memory(
        embeddings,
        medium,
        fast,
        fusion_weight=0.5,
        confidence_percentile=50,
        recent_scores=recent_scores,
        on_sample=lambda i, score, threshold, passed: records.append((i, score, threshold, passed)),
    )

    assert len(records) == 2
    assert records[0][3] is True  # near-mean sample should pass the gate
    assert records[1][3] is False  # far outlier should fail the gate
    assert records[0][2] is not None and records[1][2] is not None  # history was non-empty for both
