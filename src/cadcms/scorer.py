"""Mahalanobis scoring, score fusion, and percentile-based thresholding."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Sequence

import numpy as np

from cadcms.memory import FastMemory, GaussianMemory


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


def fused_mahalanobis_scores(
    medium_memory: GaussianMemory,
    fast_memory: FastMemory,
    embeddings: np.ndarray,
    w: float,
) -> np.ndarray:
    """Static (non-adapting) fused score: w * medium + (1 - w) * fast."""
    return fuse_scores(medium_memory.mahalanobis(embeddings), fast_memory.mahalanobis(embeddings), w)


def score_stream_with_fast_memory(
    embeddings: np.ndarray,
    medium_memory: GaussianMemory,
    fast_memory: FastMemory,
    fusion_weight: float,
    confidence_percentile: float,
    recent_scores: deque,
    on_sample: Optional[Callable[[int, float, Optional[float], bool], None]] = None,
) -> np.ndarray:
    """Score a sequential inference stream, adapting ``fast_memory`` online.

    Each sample is scored first (fused medium + fast), then -- if it scores
    below the percentile threshold of ``recent_scores`` -- fed into
    ``fast_memory.update``. ``recent_scores`` is mutated in place (a bounded
    deque) so the notion of "recent" can persist across calls, spanning a
    whole deployment stream rather than resetting per call.

    ``on_sample``, if given, is called after each sample is processed (score
    computed, gate decision made, update applied if gated in) with
    ``(index, final_score, threshold_or_None, gate_passed)``. Purely an
    observation hook for diagnostics -- it cannot affect scores or updates,
    and is not called by default (``None``), so this is not a behavior
    change for any existing caller.
    """
    embeddings = np.asarray(embeddings)
    final_scores = np.empty(len(embeddings))

    for i, embedding in enumerate(embeddings):
        row = embedding.reshape(1, -1)
        medium_score = medium_memory.mahalanobis(row)[0]
        fast_score = fast_memory.mahalanobis(row)[0]
        final_score = fuse_scores(medium_score, fast_score, fusion_weight)
        final_scores[i] = final_score

        threshold = None
        gate_passed = False
        if recent_scores:
            threshold = percentile_threshold(recent_scores, confidence_percentile)
            if is_confidently_normal(final_score, threshold):
                gate_passed = True
                fast_memory.update(embedding, medium_memory)
        recent_scores.append(final_score)

        if on_sample is not None:
            on_sample(i, final_score, threshold, gate_passed)

    return final_scores
