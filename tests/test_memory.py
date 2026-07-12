"""Memory math on synthetic Gaussians: shrinkage, fusion, save/load."""

import numpy as np
import pytest

from cadcms.memory import ContinuumMemory, GaussianMemory, fuse_moment_matching, shrink_covariance


def test_shrinkage_keeps_covariance_invertible():
    # 5 samples in 20 dimensions: raw sample covariance is rank-deficient (singular).
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(5, 20))

    memory = GaussianMemory.fit(embeddings, shrinkage_alpha=0.1)

    # np.linalg.inv raises LinAlgError on a singular matrix; this must not raise.
    inv = np.linalg.inv(memory.covariance)
    assert np.isfinite(inv).all()


def test_shrinkage_alpha_one_is_isotropic():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(50, 8), scale=2.0)
    covariance = np.cov(embeddings, rowvar=False)

    shrunk = shrink_covariance(covariance, alpha=1.0)
    expected = (np.trace(covariance) / covariance.shape[0]) * np.eye(covariance.shape[0])

    np.testing.assert_allclose(shrunk, expected)


def test_shrinkage_alpha_zero_is_identity():
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(50, 8))
    covariance = np.cov(embeddings, rowvar=False)

    shrunk = shrink_covariance(covariance, alpha=0.0)

    np.testing.assert_allclose(shrunk, covariance)


def test_mahalanobis_zero_at_mean():
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(200, 4))
    memory = GaussianMemory.fit(embeddings, shrinkage_alpha=0.1)

    score = memory.mahalanobis(memory.mean.reshape(1, -1))

    assert score[0] == pytest.approx(0.0, abs=1e-8)


def test_fuse_moment_matching_two_known_gaussians():
    # Two well-separated point clouds with known mean/covariance/count.
    mean_a, cov_a, n_a = np.array([0.0, 0.0]), np.eye(2) * 1.0, 100
    mean_b, cov_b, n_b = np.array([4.0, 0.0]), np.eye(2) * 1.0, 100

    mem_a = GaussianMemory(mean=mean_a, covariance=cov_a, count=n_a)
    mem_b = GaussianMemory(mean=mean_b, covariance=cov_b, count=n_b)

    fused = fuse_moment_matching([mem_a, mem_b])

    # Equal counts -> fused mean is the midpoint.
    np.testing.assert_allclose(fused.mean, np.array([2.0, 0.0]))

    # Law of total variance: within-group covariance (1.0) + between-group
    # spread of the means around the fused mean (each mean is 2.0 away on x).
    expected_cov = np.array([[1.0 + 4.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(fused.covariance, expected_cov)
    assert fused.count == n_a + n_b


def test_fuse_moment_matching_matches_pooled_sample_statistics():
    # Fusing per-task Gaussians via moment matching should reproduce the mean
    # and covariance of the pooled raw samples (same total count, no shrinkage).
    rng = np.random.default_rng(4)
    samples_a = rng.normal(loc=[0, 0], scale=1.0, size=(300, 2))
    samples_b = rng.normal(loc=[3, 1], scale=1.0, size=(300, 2))

    mem_a = GaussianMemory.fit(samples_a, shrinkage_alpha=0.0)
    mem_b = GaussianMemory.fit(samples_b, shrinkage_alpha=0.0)
    fused = fuse_moment_matching([mem_a, mem_b])

    pooled = np.concatenate([samples_a, samples_b], axis=0)
    np.testing.assert_allclose(fused.mean, pooled.mean(axis=0), atol=1e-8)
    np.testing.assert_allclose(fused.covariance, np.cov(pooled, rowvar=False), atol=0.05)


def test_continuum_memory_single_task_fuse_returns_that_task():
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(50, 4))

    cm = ContinuumMemory(shrinkage_alpha=0.1)
    added = cm.add_task("pill", embeddings)
    fused = cm.fuse()

    np.testing.assert_allclose(fused.mean, added.mean)
    np.testing.assert_allclose(fused.covariance, added.covariance)


def test_continuum_memory_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    cm = ContinuumMemory(shrinkage_alpha=0.1, fusion_method="moment_matching")
    cm.add_task("pill", rng.normal(size=(50, 4)))
    cm.add_task("capsule", rng.normal(loc=1.0, size=(50, 4)))

    cm.save(tmp_path / "memory")
    loaded = ContinuumMemory.load(tmp_path / "memory", shrinkage_alpha=0.1, fusion_method="moment_matching")

    assert loaded.task_order == cm.task_order
    for task_id in cm.task_order:
        np.testing.assert_allclose(loaded.task_memories[task_id].mean, cm.task_memories[task_id].mean)
        np.testing.assert_allclose(
            loaded.task_memories[task_id].covariance, cm.task_memories[task_id].covariance
        )


def test_fuse_sample_refit_recovers_approximate_moments():
    rng = np.random.default_rng(7)
    samples_a = rng.normal(loc=[0, 0], scale=1.0, size=(500, 2))
    samples_b = rng.normal(loc=[5, 0], scale=1.0, size=(500, 2))

    cm = ContinuumMemory(shrinkage_alpha=0.05, fusion_method="sample_refit", fusion_num_samples=20000, seed=8)
    cm.add_task("a", samples_a)
    cm.add_task("b", samples_b)
    fused = cm.fuse()

    pooled = np.concatenate([samples_a, samples_b], axis=0)
    np.testing.assert_allclose(fused.mean, pooled.mean(axis=0), atol=0.2)
