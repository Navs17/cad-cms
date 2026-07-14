"""FastMemory: EMA convergence to a stationary mean, and pullback decay."""

import numpy as np
import pytest

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


def test_mode_defaults_to_gated_and_does_not_affect_update_math():
    initial = GaussianMemory(mean=np.zeros(3), covariance=np.eye(3), count=10)

    default_mode = FastMemory(initial, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1)
    explicit_gated = FastMemory(
        initial, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1, mode="gated"
    )
    gatefree = FastMemory(
        initial, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1, mode="gatefree_slow"
    )

    assert default_mode.mode == "gated"
    assert explicit_gated.mode == "gated"
    assert gatefree.mode == "gatefree_slow"

    sample = np.array([2.0, -1.0, 0.5])
    default_mode.update(sample, initial)
    explicit_gated.update(sample, initial)
    gatefree.update(sample, initial)

    # mode is a label only -- update()'s math is identical regardless.
    np.testing.assert_allclose(default_mode.mean, explicit_gated.mean)
    np.testing.assert_allclose(default_mode.mean, gatefree.mean)
    np.testing.assert_allclose(default_mode.covariance, gatefree.covariance)


def test_invalid_mode_raises():
    initial = GaussianMemory(mean=np.zeros(3), covariance=np.eye(3), count=10)
    with pytest.raises(ValueError):
        FastMemory(initial, ema_rate=0.1, pullback_coefficient=0.05, shrinkage_alpha=0.1, mode="bogus")
