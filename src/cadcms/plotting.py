"""Result plots: AUROC-per-task over stages, drift curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cadcms.evaluate import AurocMatrix


def plot_auroc_per_task(
    auroc_matrix: AurocMatrix,
    tasks: list[str],
    save_path: str | Path,
    title: str = "",
) -> None:
    """One line per task: its AUROC at every stage from when it first appears onward."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for task_id in tasks:
        stage_numbers = []
        aurocs = []
        for stage_idx, stage in enumerate(auroc_matrix):
            if task_id in stage:
                stage_numbers.append(stage_idx + 1)
                aurocs.append(stage[task_id])
        ax.plot(stage_numbers, aurocs, marker="o", label=task_id)

    ax.set_xlabel("Stage")
    ax.set_ylabel("AUROC")
    ax.set_xticks(range(1, len(auroc_matrix) + 1))
    ax.set_xticklabels(tasks[: len(auroc_matrix)], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_drift_curve(
    window_t: np.ndarray,
    scores_by_method: dict[str, np.ndarray],
    save_path: str | Path,
    title: str = "Drift stream",
) -> None:
    """Windowed AUROC over a drift-replayed stream, one line per scoring method."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for method_name, values in scores_by_method.items():
        ax.plot(window_t, values, marker="o", label=method_name)

    ax.set_xlabel("Drift severity (stream position, 0=start 1=end)")
    ax.set_ylabel("Windowed AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
