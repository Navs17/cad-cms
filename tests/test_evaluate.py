"""ACC / FM formulas on a hand-constructed AUROC matrix."""

import pytest

from cadcms.evaluate import compute_acc, compute_fm


def test_acc_and_fm_no_forgetting():
    tasks = ["a", "b"]
    # task "a" AUROC stays perfect across both stages -> no forgetting.
    auroc_matrix = [
        {"a": 1.0},
        {"a": 1.0, "b": 0.9},
    ]

    assert compute_acc(auroc_matrix, tasks) == pytest.approx((1.0 + 0.9) / 2)
    assert compute_fm(auroc_matrix, tasks) == pytest.approx(0.0)


def test_fm_detects_forgetting():
    tasks = ["a", "b"]
    # task "a" scores 0.95 right after training on it, then drops to 0.6 once
    # task "b" is learned -> forgetting on "a" = 0.95 - 0.6 = 0.35.
    auroc_matrix = [
        {"a": 0.95},
        {"a": 0.60, "b": 0.85},
    ]

    acc = compute_acc(auroc_matrix, tasks)
    fm = compute_fm(auroc_matrix, tasks)

    assert acc == pytest.approx((0.60 + 0.85) / 2)
    # task "a" forgetting = 0.95 - 0.60 = 0.35; task "b" forgetting = 0.85 - 0.85 = 0.0
    assert fm == pytest.approx((0.35 + 0.0) / 2)


def test_fm_uses_best_ever_not_just_first_stage():
    tasks = ["a"]
    # AUROC on "a" improves then dips: best-ever (0.9) - final (0.7) = 0.2.
    auroc_matrix = [{"a": 0.8}, {"a": 0.9}, {"a": 0.7}]

    assert compute_fm(auroc_matrix, tasks) == pytest.approx(0.2)
