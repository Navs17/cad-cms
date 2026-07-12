"""Sequential task loop: extract -> fit medium memory -> eval.

``run_sequential_from_embeddings`` is the pure continual-learning algorithm
(no data/backbone I/O), so it can be unit tested directly on synthetic
embeddings. ``run_sequential`` wraps it with the real data pipeline
(embedding extraction/caching via the frozen backbone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from cadcms.data import build_transform, get_dataloader
from cadcms.evaluate import AurocMatrix, compute_auroc
from cadcms.features import ResNetBackbone, get_device, get_embeddings
from cadcms.memory import ContinuumMemory


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_paths(config: dict, repo_root: str | Path) -> dict:
    """Make every entry under ``config["paths"]`` an absolute Path, relative to ``repo_root``."""
    repo_root = Path(repo_root)
    config = dict(config)
    resolved_paths = {}
    for key, value in config["paths"].items():
        path = Path(value)
        resolved_paths[key] = path if path.is_absolute() else repo_root / path
    config["paths"] = resolved_paths
    return config


def run_sequential_from_embeddings(
    tasks: list[str],
    train_embeddings: dict[str, np.ndarray],
    test_embeddings: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    baseline: str,
    shrinkage_alpha: float,
    fusion_method: str,
    fusion_num_samples: int,
    seed: int,
    memory_dir: Optional[Path] = None,
) -> dict:
    """Core sequential continual-learning loop over precomputed embeddings.

    ``baseline``:
      - "naive": each stage scores with only the current task's own Gaussian
        (previous task statistics are discarded -- catastrophic forgetting).
      - "medium_only": each stage scores with the fusion of every task's
        Gaussian seen so far (DNE-style).
    """
    if baseline not in ("naive", "medium_only"):
        raise NotImplementedError(
            f"baseline {baseline!r} not supported here (medium_fast lands in Phase 5)"
        )

    continuum = ContinuumMemory(
        shrinkage_alpha=shrinkage_alpha,
        fusion_method=fusion_method,
        fusion_num_samples=fusion_num_samples,
        seed=seed,
    )

    auroc_matrix: AurocMatrix = []

    for stage_idx, task_id in enumerate(tasks):
        task_memory = continuum.add_task(task_id, train_embeddings[task_id])
        active_memory = task_memory if baseline == "naive" else continuum.fuse()

        stage_results: dict[str, float] = {}
        for seen_task in tasks[: stage_idx + 1]:
            scores = active_memory.mahalanobis(test_embeddings[seen_task])
            stage_results[seen_task] = compute_auroc(test_labels[seen_task], scores)
        auroc_matrix.append(stage_results)

        if memory_dir is not None:
            continuum.save(Path(memory_dir) / baseline)

    return {"tasks": tasks, "auroc_matrix": auroc_matrix, "baseline": baseline}


def run_sequential(config: dict, baseline: Optional[str] = None) -> dict:
    """Full pipeline: extract/cache embeddings via the frozen backbone, then
    run the sequential continual-learning loop.

    ``config["paths"]`` must already be resolved to absolute paths (see
    ``resolve_config_paths``).
    """
    baseline = baseline or config["baseline"]
    tasks = config["tasks"]
    device = get_device()

    backbone = ResNetBackbone(
        pretrained=config["backbone"]["pretrained"],
        layer=config["backbone"]["layer"],
        freeze=config["backbone"]["freeze"],
    )
    transform = build_transform(
        config["data"]["image_size"],
        config["data"]["normalize_mean"],
        config["data"]["normalize_std"],
    )

    train_embeddings: dict[str, np.ndarray] = {}
    test_embeddings: dict[str, np.ndarray] = {}
    test_labels: dict[str, np.ndarray] = {}

    for task_id in tasks:
        for split, store in (("train", train_embeddings), ("test", test_embeddings)):
            dataloader = get_dataloader(
                config["paths"]["data_root"],
                task_id,
                split,
                transform,
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"]["num_workers"],
            )
            batch, _ = get_embeddings(
                backbone,
                dataloader,
                config["paths"]["cache_dir"],
                task_id,
                split,
                config["backbone"]["name"],
                device,
            )
            store[task_id] = batch.embeddings
            if split == "test":
                test_labels[task_id] = batch.labels

    return run_sequential_from_embeddings(
        tasks=tasks,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        baseline=baseline,
        shrinkage_alpha=config["memory"]["shrinkage_alpha"],
        fusion_method=config["memory"]["fusion_method"],
        fusion_num_samples=config["memory"]["fusion_num_samples"],
        seed=config["seed"],
        memory_dir=config["paths"]["memory_dir"],
    )
