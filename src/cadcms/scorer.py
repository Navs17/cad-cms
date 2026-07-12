"""Mahalanobis scoring, score fusion, and percentile-based thresholding."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cadcms.memory import GaussianMemory


def mahalanobis_scores(memory: GaussianMemory, embeddings: np.ndarray) -> np.ndarray:
    """Anomaly score of each embedding: Mahalanobis distance to ``memory``."""
    return memory.mahalanobis(embeddings)


def fuse_scores(medium_scores: np.ndarray, fast_scores: np.ndarray, w: float) -> np.ndarray:
    """Final anomaly score = w * medium_score + (1 - w) * fast_score."""
    return w * np.asarray(medium_scores) + (1.0 - w) * np.asarray(fast_scores)


def percentile_threshold(recent_scores: Sequence[float], percentile: float) -> float:
    """Threshold below which a score is treated as 'confidently normal'."""
    return float(np.percentile(np.asarray(recent_scores, dtype=np.float64), percentile))


def is_confidently_normal(score: float, threshold: float) -> bool:
    return score <= threshold
