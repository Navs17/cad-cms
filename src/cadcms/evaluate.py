"""AUROC, ACC, FM computation and results writing."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# auroc_matrix[stage_idx] is a dict: task_id -> AUROC, for every task seen by
# that stage (stage_idx counts from 0 after training on tasks[0], 1 after
# tasks[0:2], ...).
AurocMatrix = list[dict[str, float]]


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level AUROC (higher score = more anomalous, label 1 = defective)."""
    return float(roc_auc_score(labels, scores))


def compute_acc(auroc_matrix: AurocMatrix, tasks: list[str]) -> float:
    """ACC = mean final-stage AUROC over all tasks."""
    final_stage = auroc_matrix[-1]
    return float(np.mean([final_stage[task_id] for task_id in tasks]))


def compute_fm(auroc_matrix: AurocMatrix, tasks: list[str]) -> float:
    """FM = mean over tasks of (best AUROC ever achieved on that task - final AUROC)."""
    final_stage = auroc_matrix[-1]
    forgetting = []
    for task_id in tasks:
        scores_over_stages = [stage[task_id] for stage in auroc_matrix if task_id in stage]
        best = max(scores_over_stages)
        forgetting.append(best - final_stage[task_id])
    return float(np.mean(forgetting))


def write_auroc_csv(auroc_matrix: AurocMatrix, tasks: list[str], path: str | Path) -> None:
    """Rows = stages (after training on the stage's task), columns = per-task AUROC."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage_task"] + tasks)
        for stage_idx, stage in enumerate(auroc_matrix):
            writer.writerow([tasks[stage_idx]] + [stage.get(task_id, "") for task_id in tasks])


def write_summary_csv(summary: dict[str, dict[str, float]], path: str | Path) -> None:
    """``summary``: baseline name -> {"acc": ..., "fm": ...}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "acc", "fm"])
        for baseline, metrics in summary.items():
            writer.writerow([baseline, metrics["acc"], metrics["fm"]])


def compute_windowed_auroc(
    t: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """AUROC over non-overlapping windows of a stream-ordered sequence.

    ``t`` is the per-sample drift severity (used only to label each window
    by its mean severity). Windows with only one label class present are
    skipped, since AUROC is undefined for them.
    """
    window_t = []
    window_auroc = []
    for start in range(0, len(scores), window_size):
        end = min(start + window_size, len(scores))
        window_labels = labels[start:end]
        if len(np.unique(window_labels)) < 2:
            continue
        window_t.append(float(np.mean(t[start:end])))
        window_auroc.append(compute_auroc(window_labels, scores[start:end]))
    return np.array(window_t), np.array(window_auroc)


def write_drift_csv(window_t: np.ndarray, scores_by_method: dict[str, np.ndarray], path: str | Path) -> None:
    """``scores_by_method``: method name -> per-window AUROC, aligned to ``window_t``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(scores_by_method.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + methods)
        for i, t in enumerate(window_t):
            writer.writerow([t] + [scores_by_method[m][i] for m in methods])


def write_gate_records_csv(records: list[dict], path: str | Path) -> None:
    """One row per stream sample: index, t, label, final_score, threshold,
    gate_passed, fast_mean_dist_from_medium (see scripts/diagnose_drift_gate.py).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "t", "label", "final_score", "threshold", "gate_passed", "fast_mean_dist_from_medium"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def compute_gate_window_stats(records: list[dict], window_size: int) -> list[dict]:
    """Aggregate per-sample gate records into non-overlapping windows.

    Per window: pass_rate (fraction of samples that passed the gate),
    normal_gate_recall (fraction of TRUE normal samples that passed the gate
    -- low means the gate is rejecting the normals it needs), gate_purity
    (fraction of gated-in samples that were actually normal -- low means
    defects are leaking through the gate), and the FAST-medium mean distance
    at the end of the window.
    """
    windows = []
    for start in range(0, len(records), window_size):
        chunk = records[start : start + window_size]
        n = len(chunk)
        passed = [r for r in chunk if r["gate_passed"]]
        true_normal = [r for r in chunk if r["label"] == 0]
        passed_normal = [r for r in passed if r["label"] == 0]

        windows.append(
            {
                "t": float(np.mean([r["t"] for r in chunk])),
                "n": n,
                "num_passed": len(passed),
                "pass_rate": len(passed) / n,
                "num_true_normal": len(true_normal),
                "normal_gate_recall": (len(passed_normal) / len(true_normal)) if true_normal else float("nan"),
                "gate_purity": (len(passed_normal) / len(passed)) if passed else float("nan"),
                "fast_mean_dist_from_medium": chunk[-1]["fast_mean_dist_from_medium"],
            }
        )
    return windows


def write_gate_windows_csv(windows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(windows[0].keys()))
        writer.writeheader()
        writer.writerows(windows)
