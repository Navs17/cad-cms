"""Sequential continual-learning loop, on synthetic embeddings.

Constructs three well-separated per-task Gaussians so that the qualitative
continual-learning behavior is unambiguous: "naive" (single Gaussian
overwritten each task) should catastrophically forget earlier tasks, while
"medium_only" (DNE-style fusion of all tasks seen so far) should retain much
higher AUROC on them.
"""

import numpy as np
import pytest

from cadcms.evaluate import compute_acc, compute_fm
from cadcms.train import run_sequential_from_embeddings

TASKS = ["a", "b", "c"]
DIM = 16
TASK_CENTERS = {"a": 0.0, "b": 8.0, "c": 16.0}  # far apart along one axis


def _make_task_data(rng: np.random.Generator, task_id: str):
    center = np.zeros(DIM)
    center[0] = TASK_CENTERS[task_id]

    train_normal = rng.normal(loc=center, scale=1.0, size=(150, DIM))
    test_normal = rng.normal(loc=center, scale=1.0, size=(60, DIM))
    # defects: same task center, distinct shift on a different axis.
    defect_center = center.copy()
    defect_center[1] += 5.0
    test_defect = rng.normal(loc=defect_center, scale=1.0, size=(60, DIM))

    test_embeddings = np.concatenate([test_normal, test_defect], axis=0)
    test_labels = np.concatenate([np.zeros(60), np.ones(60)])
    return train_normal, test_embeddings, test_labels


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    train_embeddings, test_embeddings, test_labels = {}, {}, {}
    for task_id in TASKS:
        train, test_emb, test_lbl = _make_task_data(rng, task_id)
        train_embeddings[task_id] = train
        test_embeddings[task_id] = test_emb
        test_labels[task_id] = test_lbl
    return train_embeddings, test_embeddings, test_labels


def test_auroc_matrix_shape_grows_per_stage(synthetic_data):
    train_embeddings, test_embeddings, test_labels = synthetic_data

    result = run_sequential_from_embeddings(
        tasks=TASKS,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        baseline="medium_only",
        shrinkage_alpha=0.1,
        fusion_method="moment_matching",
        fusion_num_samples=2000,
        seed=0,
    )

    auroc_matrix = result["auroc_matrix"]
    assert len(auroc_matrix) == 3
    assert list(auroc_matrix[0].keys()) == ["a"]
    assert list(auroc_matrix[1].keys()) == ["a", "b"]
    assert list(auroc_matrix[2].keys()) == ["a", "b", "c"]


def test_medium_only_learns_each_task_well_when_first_seen(synthetic_data):
    train_embeddings, test_embeddings, test_labels = synthetic_data

    result = run_sequential_from_embeddings(
        tasks=TASKS,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        baseline="medium_only",
        shrinkage_alpha=0.1,
        fusion_method="moment_matching",
        fusion_num_samples=2000,
        seed=0,
    )

    auroc_matrix = result["auroc_matrix"]
    for stage_idx, task_id in enumerate(TASKS):
        assert auroc_matrix[stage_idx][task_id] > 0.9


def test_naive_forgets_more_than_medium_only(synthetic_data):
    train_embeddings, test_embeddings, test_labels = synthetic_data
    common_kwargs = dict(
        tasks=TASKS,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        shrinkage_alpha=0.1,
        fusion_method="moment_matching",
        fusion_num_samples=2000,
        seed=0,
    )

    naive_result = run_sequential_from_embeddings(baseline="naive", **common_kwargs)
    medium_result = run_sequential_from_embeddings(baseline="medium_only", **common_kwargs)

    naive_fm = compute_fm(naive_result["auroc_matrix"], TASKS)
    medium_fm = compute_fm(medium_result["auroc_matrix"], TASKS)
    naive_acc = compute_acc(naive_result["auroc_matrix"], TASKS)
    medium_acc = compute_acc(medium_result["auroc_matrix"], TASKS)

    assert naive_fm > medium_fm
    assert medium_acc > naive_acc


def test_unknown_baseline_raises(synthetic_data):
    train_embeddings, test_embeddings, test_labels = synthetic_data

    with pytest.raises(NotImplementedError):
        run_sequential_from_embeddings(
            tasks=TASKS,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            baseline="medium_fast",
            shrinkage_alpha=0.1,
            fusion_method="moment_matching",
            fusion_num_samples=2000,
            seed=0,
        )
