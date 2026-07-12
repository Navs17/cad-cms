"""FastMemory: EMA convergence to a stationary mean, and pullback decay."""

import numpy as np

from cadcms.memory import FastMemory, GaussianMemory


def test_ema_converges_to_stationary_mean():
    rng = np.random.default_rng(0)
    true_mean = np.array([3.0, -2.0])
    true_covariance = np.eye(2) * 0.5

    initial = GaussianMemory(mean=np.zeros(2), covariance=np.eye(2), count=10)
    fast = FastMemory(initial, ema_rate=0.01, pullback_coefficient=0.0, shrinkage_alpha=0.0)
    dummy_medium = initial  # irrelevant: pullback_coefficient=0 zeroes its contribution

    for _ in range(20000):
        sample = rng.multivariate_normal(true_mean, true_covariance)
        fast.update(sample, dummy_medium)

    np.testing.assert_allclose(fast.mean, true_mean, atol=0.2)


def test_pullback_decays_toward_medium_memory():
    medium = GaussianMemory(mean=np.array([10.0, 10.0]), covariance=np.eye(2) * 2.0, count=100)
    fast = FastMemory(
        GaussianMemory(mean=np.zeros(2), covariance=np.eye(2), count=10),
        ema_rate=0.0,  # disable adaptation from new samples
        pullback_coefficient=0.1,
        shrinkage_alpha=0.0,
    )

    for _ in range(200):
        # ema_rate=0 means this sample has no effect beyond triggering pullback.
        fast.update(np.array([999.0, -999.0]), medium)

    np.testing.assert_allclose(fast.mean, medium.mean, atol=1e-6)
    np.testing.assert_allclose(fast.covariance, medium.covariance, atol=1e-6)


def test_update_keeps_covariance_invertible():
    rng = np.random.default_rng(1)
    initial = GaussianMemory(mean=np.zeros(5), covariance=np.eye(5), count=5)
    fast = FastMemory(initial, ema_rate=0.2, pullback_coefficient=0.05, shrinkage_alpha=0.1)
    medium = initial

    for _ in range(20):
        fast.update(rng.normal(size=5), medium)
        inv = np.linalg.inv(fast.covariance)
        assert np.isfinite(inv).all()


def test_mahalanobis_and_num_updates():
    initial = GaussianMemory(mean=np.zeros(3), covariance=np.eye(3), count=10)
    fast = FastMemory(initial, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)

    assert fast.num_updates == 0
    scores_before = fast.mahalanobis(np.array([[1.0, 0.0, 0.0]]))

    fast.update(np.array([5.0, 5.0, 5.0]), initial)

    assert fast.num_updates == 1
    scores_after = fast.mahalanobis(np.array([[1.0, 0.0, 0.0]]))
    assert scores_before[0] != scores_after[0]
