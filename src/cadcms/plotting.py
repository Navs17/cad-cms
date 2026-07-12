"""Result plots: AUROC-per-task over stages, drift curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

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
