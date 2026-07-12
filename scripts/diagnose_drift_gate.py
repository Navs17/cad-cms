"""Phase B diagnostic: instrument the FAST-memory confidence gate on the
drift stream to test the hypothesis that drifted-but-normal samples fail the
gate (so FAST never learns the new normal), as opposed to defects leaking
through the gate (so FAST adapts to the wrong samples).

No behavior change to the drift experiment itself: this reuses the exact
same scoring/gating/update path (score_stream_with_fast_memory) via its
optional on_sample observation hook, which cannot affect scores or updates.

Usage:
    python scripts/diagnose_drift_gate.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from torch.utils.data import DataLoader

from cadcms.data import DriftStreamDataset, MVTecDataset, build_transform, get_dataloader
from cadcms.evaluate import compute_gate_window_stats, write_gate_records_csv, write_gate_windows_csv
from cadcms.features import ResNetBackbone, extract_embeddings_with_t, get_device, get_embeddings
from cadcms.memory import ContinuumMemory, FastMemory
from cadcms.plotting import plot_gate_diagnostics
from cadcms.scorer import score_stream_with_fast_memory
from cadcms.train import load_config, resolve_config_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), REPO_ROOT)
    task_id = config["tasks"][0]
    device = get_device()
    print(f"drift task: {task_id}  device: {device}")

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

    train_loader = get_dataloader(
        config["paths"]["data_root"],
        task_id,
        "train",
        transform,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    train_batch, _ = get_embeddings(
        backbone,
        train_loader,
        config["paths"]["cache_dir"],
        task_id,
        "train",
        config["backbone"]["name"],
        device,
    )
    continuum = ContinuumMemory(
        shrinkage_alpha=config["memory"]["shrinkage_alpha"],
        fusion_method=config["memory"]["fusion_method"],
        fusion_num_samples=config["memory"]["fusion_num_samples"],
        seed=config["seed"],
    )
    medium_memory = continuum.add_task(task_id, train_batch.embeddings)

    raw_test_dataset = MVTecDataset(config["paths"]["data_root"], task_id, "test", transform=None)
    drift_dataset = DriftStreamDataset(
        raw_test_dataset,
        length=config["drift"]["stream_length"],
        brightness_range=tuple(config["drift"]["brightness_range"]),
        blur_sigma_range=tuple(config["drift"]["blur_sigma_range"]),
        transform=transform,
        seed=config["seed"],
    )
    drift_loader = DataLoader(drift_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    stream_batch, stream_t = extract_embeddings_with_t(backbone, drift_loader, device)

    fast_memory = FastMemory(
        medium_memory,
        ema_rate=config["fast_memory"]["ema_rate"],
        pullback_coefficient=config["fast_memory"]["pullback_coefficient"],
        shrinkage_alpha=config["memory"]["shrinkage_alpha"],
    )
    recent_scores: deque = deque(maxlen=config["fast_memory"]["recent_scores_window"])

    records: list[dict] = []

    def record(index: int, final_score: float, threshold, gate_passed: bool) -> None:
        records.append(
            {
                "index": index,
                "t": float(stream_t[index]),
                "label": int(stream_batch.labels[index]),
                "final_score": final_score,
                "threshold": threshold if threshold is not None else float("nan"),
                "gate_passed": gate_passed,
                "fast_mean_dist_from_medium": float(np.linalg.norm(fast_memory.mean - medium_memory.mean)),
            }
        )

    score_stream_with_fast_memory(
        stream_batch.embeddings,
        medium_memory,
        fast_memory,
        config["fast_memory"]["fusion_weight"],
        config["fast_memory"]["confidence_percentile"],
        recent_scores,
        on_sample=record,
    )

    results_dir = config["paths"]["results_dir"]
    figures_dir = config["paths"]["figures_dir"]
    write_gate_records_csv(records, results_dir / "drift_gate_records.csv")

    windows = compute_gate_window_stats(records, config["drift"]["window_size"])
    write_gate_windows_csv(windows, results_dir / "drift_gate_windows.csv")

    plot_gate_diagnostics(
        window_t=np.array([w["t"] for w in windows]),
        pass_rate=np.array([w["pass_rate"] for w in windows]),
        normal_gate_recall=np.array([w["normal_gate_recall"] for w in windows]),
        gate_purity=np.array([w["gate_purity"] for w in windows]),
        fast_mean_distance=np.array([w["fast_mean_dist_from_medium"] for w in windows]),
        save_path=figures_dir / "drift_gate_diagnostics.png",
        title=f"Confidence-gate diagnostics ({task_id})",
    )

    total_n = len(records)
    total_passed = sum(1 for r in records if r["gate_passed"])
    total_true_normal = sum(1 for r in records if r["label"] == 0)
    total_passed_normal = sum(1 for r in records if r["gate_passed"] and r["label"] == 0)
    total_passed_defect = sum(1 for r in records if r["gate_passed"] and r["label"] == 1)

    overall_pass_rate = total_passed / total_n
    overall_normal_recall = total_passed_normal / total_true_normal if total_true_normal else float("nan")
    overall_purity = total_passed_normal / total_passed if total_passed else float("nan")

    print(f"\nstream length: {total_n}  true normal: {total_true_normal}  true defect: {total_n - total_true_normal}")
    print(f"gate passed: {total_passed}/{total_n} ({overall_pass_rate:.1%})")
    print(f"  of which normal: {total_passed_normal}  defect: {total_passed_defect}")
    print(f"normal gate recall (fraction of TRUE normals let through): {overall_normal_recall:.1%}")
    print(f"gate purity (fraction of gated-in samples that were normal): {overall_purity:.1%}")

    print("\n--- hypothesis check ---")
    if overall_purity < 0.9 and total_passed_defect > 0:
        print(f"DEFECTS LEAKING THROUGH: {total_passed_defect} defective samples passed the gate "
              f"(purity {overall_purity:.1%}). FAST may be adapting toward the wrong samples.")
    if overall_normal_recall < 0.5:
        print(f"NORMALS REJECTED: only {overall_normal_recall:.1%} of true-normal samples passed the "
              f"gate. FAST is starved of the drifted-normal samples it needs to adapt.")
    if overall_normal_recall >= 0.5 and (overall_purity >= 0.9 or total_passed_defect == 0):
        print("Neither failure mode dominates by these thresholds -- see the full windowed breakdown.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
