"""GaussianMemory (per-task statistics) and ContinuumMemory (fusion across tasks).

MEDIUM memory level: one Gaussian (mean, shrunk covariance, sample count) is
fit per task from that task's normal training embeddings (DNE-style). At
inference, all task Gaussians are fused into a single distribution, either by
DNE's sample-and-refit or by closed-form moment matching of the mixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def shrink_covariance(covariance: np.ndarray, alpha: float) -> np.ndarray:
    """sigma_shrunk = (1 - alpha) * sigma + alpha * (trace(sigma) / D) * I

    Keeps the covariance invertible even when the sample count is smaller
    than the embedding dimension, by pulling it toward an isotropic target.
    """
    dim = covariance.shape[0]
    target = (np.trace(covariance) / dim) * np.eye(dim)
    return (1.0 - alpha) * covariance + alpha * target


class GaussianMemory:
    """A single Gaussian (mean, shrunk covariance, sample count) over embeddings."""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, count: int) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.count = int(count)
        self._inv_covariance: Optional[np.ndarray] = None

    @classmethod
    def fit(cls, embeddings: np.ndarray, shrinkage_alpha: float) -> "GaussianMemory":
        """Fit mean + shrunk covariance from a set of (normal) embeddings."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        n = embeddings.shape[0]
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        covariance = (centered.T @ centered) / max(n - 1, 1)
        covariance = shrink_covariance(covariance, shrinkage_alpha)
        return cls(mean=mean, covariance=covariance, count=n)

    @property
    def inv_covariance(self) -> np.ndarray:
        if self._inv_covariance is None:
            self._inv_covariance = np.linalg.inv(self.covariance)
        return self._inv_covariance

    def mahalanobis(self, embeddings: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each row of ``embeddings`` to this Gaussian."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        diff = embeddings - self.mean
        squared = np.einsum("ij,jk,ik->i", diff, self.inv_covariance, diff)
        return np.sqrt(np.maximum(squared, 0.0))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, covariance=self.covariance, count=self.count)

    @classmethod
    def load(cls, path: str | Path) -> "GaussianMemory":
        data = np.load(path)
        return cls(mean=data["mean"], covariance=data["covariance"], count=int(data["count"]))


def fuse_moment_matching(memories: list[GaussianMemory]) -> GaussianMemory:
    """Closed-form moment matching: collapse a Gaussian mixture into one Gaussian
    with the same overall mean and covariance as the pooled samples would have.
    """
    counts = np.array([m.count for m in memories], dtype=np.float64)
    total = counts.sum()
    weights = counts / total

    means = np.stack([m.mean for m in memories])
    fused_mean = (weights[:, None] * means).sum(axis=0)

    dim = fused_mean.shape[0]
    fused_covariance = np.zeros((dim, dim))
    for weight, memory in zip(weights, memories):
        diff = (memory.mean - fused_mean).reshape(-1, 1)
        fused_covariance += weight * (memory.covariance + diff @ diff.T)

    return GaussianMemory(mean=fused_mean, covariance=fused_covariance, count=int(total))


def fuse_sample_refit(
    memories: list[GaussianMemory],
    num_samples: int,
    shrinkage_alpha: float,
    rng: np.random.Generator,
) -> GaussianMemory:
    """DNE-style fusion: draw synthetic samples from each task Gaussian, pool
    them, and refit a single Gaussian (with shrinkage) on the pooled samples.
    """
    per_task = max(num_samples // len(memories), 1)
    pooled = np.concatenate(
        [rng.multivariate_normal(m.mean, m.covariance, size=per_task) for m in memories],
        axis=0,
    )
    return GaussianMemory.fit(pooled, shrinkage_alpha)


class ContinuumMemory:
    """Owns one GaussianMemory per task and fuses them into a single distribution."""

    def __init__(
        self,
        shrinkage_alpha: float,
        fusion_method: str = "moment_matching",
        fusion_num_samples: int = 2000,
        seed: int = 0,
    ) -> None:
        if fusion_method not in ("moment_matching", "sample_refit"):
            raise ValueError(f"unknown fusion_method {fusion_method!r}")

        self.shrinkage_alpha = shrinkage_alpha
        self.fusion_method = fusion_method
        self.fusion_num_samples = fusion_num_samples
        self.rng = np.random.default_rng(seed)
        self.task_memories: dict[str, GaussianMemory] = {}
        self.task_order: list[str] = []

    def add_task(self, task_id: str, embeddings: np.ndarray) -> GaussianMemory:
        """Fit and store the medium-memory Gaussian for one task's normal embeddings."""
        memory = GaussianMemory.fit(embeddings, self.shrinkage_alpha)
        if task_id not in self.task_memories:
            self.task_order.append(task_id)
        self.task_memories[task_id] = memory
        return memory

    def fuse(self) -> GaussianMemory:
        """Fuse all stored task Gaussians into a single distribution."""
        if not self.task_memories:
            raise RuntimeError("no task memories to fuse")

        memories = [self.task_memories[t] for t in self.task_order]
        if len(memories) == 1:
            return memories[0]

        if self.fusion_method == "moment_matching":
            return fuse_moment_matching(memories)
        return fuse_sample_refit(memories, self.fusion_num_samples, self.shrinkage_alpha, self.rng)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for task_id in self.task_order:
            self.task_memories[task_id].save(directory / f"{task_id}.npz")
        with open(directory / "task_order.json", "w", encoding="utf-8") as f:
            json.dump(self.task_order, f)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        shrinkage_alpha: float,
        fusion_method: str = "moment_matching",
        fusion_num_samples: int = 2000,
        seed: int = 0,
    ) -> "ContinuumMemory":
        directory = Path(directory)
        with open(directory / "task_order.json", "r", encoding="utf-8") as f:
            task_order = json.load(f)

        memory = cls(
            shrinkage_alpha=shrinkage_alpha,
            fusion_method=fusion_method,
            fusion_num_samples=fusion_num_samples,
            seed=seed,
        )
        memory.task_order = task_order
        memory.task_memories = {
            task_id: GaussianMemory.load(directory / f"{task_id}.npz") for task_id in task_order
        }
        return memory
