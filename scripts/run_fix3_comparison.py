"""Fix 3 Phase 3: seeded comparison of medium_only vs gated FAST vs gate-free
slow FAST (Fix 3), on:

  1. the main continual-learning benchmark (ACC/FM -- confirms Fix 3 doesn't
     break retention),
  2. the monotonic color/blur drift stream (the documented gated failure),
  3. the contamination (rising-defect-rate) stream (tests whether gate-free
     adaptation quietly "normalizes" defects).

5 seeds throughout; mean +/- std reported for every metric. CSVs to
results/, plots to results/figures/.

Usage:
    python scripts/run_fix3_comparison.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from torch.utils.data import DataLoader

from cadcms.data import (
    DriftStreamDataset,
    MVTecDataset,
    build_contamination_stream,
    build_transform,
    get_dataloader,
)
from cadcms.evaluate import compute_acc, compute_fm, compute_windowed_auroc
from cadcms.features import ResNetBackbone, extract_embeddings_with_t, get_device, get_embeddings
from cadcms.memory import ContinuumMemory, FastMemory
from cadcms.plotting import plot_drift_curve
from cadcms.scorer import score_stream_gatefree, score_stream_with_fast_memory
from cadcms.train import load_config, resolve_config_paths, run_sequential

CONDITIONS = ("medium_only", "gated", "gatefree_slow")


def build_fast_memory(medium_memory, config: dict, condition: str) -> FastMemory:
    fm_cfg = config["fast_memory"]
    if condition == "gated":
        return FastMemory(
            medium_memory,
            ema_rate=fm_cfg["ema_rate"],
            pullback_coefficient=fm_cfg["pullback_coefficient"],
            shrinkage_alpha=config["memory"]["shrinkage_alpha"],
            mode="gated",
        )
    return FastMemory(
        medium_memory,
        ema_rate=fm_cfg["gatefree_ema_rate"],
        pullback_coefficient=fm_cfg["gatefree_pullback_coefficient"],
        shrinkage_alpha=config["memory"]["shrinkage_alpha"],
        mode="gatefree_slow",
    )


def score_condition(stream_embeddings, medium_memory, config: dict, condition: str) -> np.ndarray:
    if condition == "medium_only":
        return medium_memory.mahalanobis(stream_embeddings)

    fast_memory = build_fast_memory(medium_memory, config, condition)
    fm_cfg = config["fast_memory"]
    if condition == "gated":
        recent_scores: deque = deque(maxlen=fm_cfg["recent_scores_window"])
        return score_stream_with_fast_memory(
            stream_embeddings, medium_memory, fast_memory, fm_cfg["fusion_weight"], fm_cfg["confidence_percentile"], recent_scores
        )
    return score_stream_gatefree(stream_embeddings, medium_memory, fast_memory, fm_cfg["fusion_weight"])


def average_windowed_auroc_across_seeds(per_seed_results: list[tuple[np.ndarray, np.ndarray]]):
    """Align by (rounded) t rather than assuming every seed produced the same
    number of windows -- a window with only one class present is skipped by
    compute_windowed_auroc, and that can happen for a different window on a
    different seed (most likely at low defect rates in the contamination
    stream). Returns (t_values, mean_auroc, std_auroc, n_seeds_per_t).
    """
    buckets = defaultdict(list)
    for window_t, window_auroc in per_seed_results:
        for t, auroc in zip(window_t, window_auroc):
            buckets[round(float(t), 4)].append(float(auroc))

    t_values = np.array(sorted(buckets.keys()))
    mean_auroc = np.array([np.mean(buckets[t]) for t in t_values])
    std_auroc = np.array([np.std(buckets[t]) for t in t_values])
    n_seeds = np.array([len(buckets[t]) for t in t_values])
    return t_values, mean_auroc, std_auroc, n_seeds


def run_main_benchmark_comparison(config: dict, seeds: list[int]) -> tuple[list[dict], dict]:
    raw_rows = []
    per_condition = {c: {"acc": [], "fm": []} for c in CONDITIONS}

    for seed in seeds:
        cfg = dict(config)
        cfg["seed"] = seed

        for condition in CONDITIONS:
            if condition == "medium_only":
                result = run_sequential(cfg, baseline="medium_only")
            else:
                cfg_fast = dict(cfg)
                cfg_fast["fast_memory"] = {**config["fast_memory"], "mode": condition}
                result = run_sequential(cfg_fast, baseline="medium_fast")

            acc = compute_acc(result["auroc_matrix"], result["tasks"])
            fm = compute_fm(result["auroc_matrix"], result["tasks"])
            per_condition[condition]["acc"].append(acc)
            per_condition[condition]["fm"].append(fm)
            raw_rows.append({"seed": seed, "condition": condition, "acc": acc, "fm": fm})
            print(f"  [main benchmark] seed={seed} condition={condition}: ACC={acc:.4f} FM={fm:.4f}")

    summary = {
        c: {
            "acc_mean": float(np.mean(v["acc"])),
            "acc_std": float(np.std(v["acc"])),
            "fm_mean": float(np.mean(v["fm"])),
            "fm_std": float(np.std(v["fm"])),
        }
        for c, v in per_condition.items()
    }
    return raw_rows, summary


def run_stream_comparison(config: dict, seeds: list[int], stream_type: str) -> tuple[list[dict], dict, dict]:
    """stream_type: "monotonic" or "contamination". Returns (raw_rows,
    scalar_summary, windowed_summary)."""
    task_id = config["tasks"][0]
    device = get_device()
    backbone = ResNetBackbone(
        pretrained=config["backbone"]["pretrained"], layer=config["backbone"]["layer"], freeze=config["backbone"]["freeze"]
    )
    transform = build_transform(
        config["data"]["image_size"], config["data"]["normalize_mean"], config["data"]["normalize_std"]
    )

    raw_rows = []
    per_condition_means: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    per_condition_windows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in CONDITIONS}

    cached_test_batch = None
    if stream_type == "contamination":
        test_loader = get_dataloader(
            config["paths"]["data_root"], task_id, "test", transform,
            batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"],
        )
        cached_test_batch, _ = get_embeddings(
            backbone, test_loader, config["paths"]["cache_dir"], task_id, "test", config["backbone"]["name"], device
        )

    for seed in seeds:
        train_loader = get_dataloader(
            config["paths"]["data_root"], task_id, "train", transform,
            batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"],
        )
        train_batch, _ = get_embeddings(
            backbone, train_loader, config["paths"]["cache_dir"], task_id, "train", config["backbone"]["name"], device
        )
        continuum = ContinuumMemory(
            shrinkage_alpha=config["memory"]["shrinkage_alpha"],
            fusion_method=config["memory"]["fusion_method"],
            fusion_num_samples=config["memory"]["fusion_num_samples"],
            seed=seed,
        )
        medium_memory = continuum.add_task(task_id, train_batch.embeddings)

        if stream_type == "monotonic":
            raw_test_dataset = MVTecDataset(config["paths"]["data_root"], task_id, "test", transform=None)
            drift_dataset = DriftStreamDataset(
                raw_test_dataset,
                length=config["drift"]["stream_length"],
                brightness_range=tuple(config["drift"]["brightness_range"]),
                blur_sigma_range=tuple(config["drift"]["blur_sigma_range"]),
                transform=transform,
                seed=seed,
            )
            drift_loader = DataLoader(drift_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
            stream_batch, stream_t = extract_embeddings_with_t(backbone, drift_loader, device)
            stream_embeddings, stream_labels = stream_batch.embeddings, stream_batch.labels
            window_size = config["drift"]["window_size"]
        else:
            cc = config["drift"]["contamination"]
            stream_embeddings, stream_labels, stream_t = build_contamination_stream(
                cached_test_batch.embeddings,
                cached_test_batch.labels,
                length=cc["stream_length"],
                start_defect_rate=cc["start_defect_rate"],
                end_defect_rate=cc["end_defect_rate"],
                seed=seed,
            )
            window_size = cc["window_size"]

        for condition in CONDITIONS:
            scores = score_condition(stream_embeddings, medium_memory, config, condition)
            window_t, window_auroc = compute_windowed_auroc(stream_t, stream_labels, scores, window_size)
            mean_auroc = float(window_auroc.mean())
            per_condition_means[condition].append(mean_auroc)
            per_condition_windows[condition].append((window_t, window_auroc))
            raw_rows.append({"seed": seed, "condition": condition, "mean_windowed_auroc": mean_auroc})
            print(f"  [{stream_type}] seed={seed} condition={condition}: mean windowed AUROC={mean_auroc:.4f}")

    scalar_summary = {
        c: {"mean": float(np.mean(v)), "std": float(np.std(v))} for c, v in per_condition_means.items()
    }
    windowed_summary = {c: average_windowed_auroc_across_seeds(v) for c, v in per_condition_windows.items()}
    return raw_rows, scalar_summary, windowed_summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), REPO_ROOT)
    seeds = config["seeds"]
    results_dir = config["paths"]["results_dir"]
    figures_dir = config["paths"]["figures_dir"]

    print("=== main benchmark (ACC/FM retention check) ===")
    benchmark_raw, benchmark_summary = run_main_benchmark_comparison(config, seeds)
    write_csv(benchmark_raw, results_dir / "fix3_benchmark_raw.csv")
    write_csv(
        [{"condition": c, **stats} for c, stats in benchmark_summary.items()],
        results_dir / "fix3_benchmark_summary.csv",
    )
    print("\nmain benchmark summary:")
    for c, stats in benchmark_summary.items():
        print(f"  {c}: ACC={stats['acc_mean']:.4f}+/-{stats['acc_std']:.4f}  FM={stats['fm_mean']:.4f}+/-{stats['fm_std']:.4f}")

    for stream_type in ("monotonic", "contamination"):
        print(f"\n=== {stream_type} drift stream ===")
        raw_rows, scalar_summary, windowed_summary = run_stream_comparison(config, seeds, stream_type)

        write_csv(raw_rows, results_dir / f"fix3_drift_{stream_type}_raw.csv")
        write_csv(
            [{"condition": c, **stats} for c, stats in scalar_summary.items()],
            results_dir / f"fix3_drift_{stream_type}_summary.csv",
        )

        windowed_rows = []
        for condition, (t_values, mean_auroc, std_auroc, n_seeds) in windowed_summary.items():
            for t, mean_v, std_v, n in zip(t_values, mean_auroc, std_auroc, n_seeds):
                windowed_rows.append(
                    {"condition": condition, "t": t, "mean_auroc": mean_v, "std_auroc": std_v, "n_seeds": n}
                )
        write_csv(windowed_rows, results_dir / f"fix3_drift_{stream_type}_windowed.csv")

        plot_scores = {c: windowed_summary[c][1] for c in CONDITIONS}
        # All conditions share the same t grid by construction (t is purely
        # positional); use condition 0's t values for the x-axis.
        plot_drift_curve(
            windowed_summary[CONDITIONS[0]][0],
            plot_scores,
            figures_dir / f"fix3_drift_{stream_type}.png",
            title=f"Fix 3 comparison: {stream_type} stream",
        )

        print(f"\n{stream_type} summary:")
        for c, stats in scalar_summary.items():
            print(f"  {c}: mean windowed AUROC={stats['mean']:.4f}+/-{stats['std']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
