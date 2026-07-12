"""Re-run the main benchmark (naive / medium_only / medium_fast) across the
fixed seed list in config["seeds"], reporting mean +/- std ACC/FM per
baseline. No method logic changes here -- this only establishes whether the
single-seed gaps between baselines are real or noise.

Embeddings are cached per category/split/backbone (not per seed), so
re-running across seeds does not re-extract them -- only memory
fitting/fusion/scoring is repeated.

Usage:
    python scripts/run_benchmark_seeds.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from cadcms.evaluate import compute_acc, compute_fm
from cadcms.train import BASELINES, load_config, resolve_config_paths, run_sequential


def write_raw_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "baseline", "acc", "fm"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "acc_mean", "acc_std", "fm_mean", "fm_std"])
        for baseline, stats in summary.items():
            writer.writerow([baseline, stats["acc_mean"], stats["acc_std"], stats["fm_mean"], stats["fm_std"]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), REPO_ROOT)
    seeds = config["seeds"]
    results_dir = config["paths"]["results_dir"]

    raw_rows: list[dict] = []
    per_baseline_acc: dict[str, list[float]] = {b: [] for b in BASELINES}
    per_baseline_fm: dict[str, list[float]] = {b: [] for b in BASELINES}

    for seed in seeds:
        cfg = dict(config)
        cfg["seed"] = seed
        for baseline in BASELINES:
            result = run_sequential(cfg, baseline=baseline)
            acc = compute_acc(result["auroc_matrix"], result["tasks"])
            fm = compute_fm(result["auroc_matrix"], result["tasks"])
            raw_rows.append({"seed": seed, "baseline": baseline, "acc": acc, "fm": fm})
            per_baseline_acc[baseline].append(acc)
            per_baseline_fm[baseline].append(fm)
            print(f"seed={seed} baseline={baseline}: ACC={acc:.4f} FM={fm:.4f}")

    summary = {}
    for baseline in BASELINES:
        acc_vals = np.array(per_baseline_acc[baseline])
        fm_vals = np.array(per_baseline_fm[baseline])
        summary[baseline] = {
            "acc_mean": float(acc_vals.mean()),
            "acc_std": float(acc_vals.std()),
            "fm_mean": float(fm_vals.mean()),
            "fm_std": float(fm_vals.std()),
        }

    write_raw_csv(raw_rows, results_dir / "benchmark_seeds_raw.csv")
    write_summary_csv(summary, results_dir / "benchmark_seeds_summary.csv")

    print(f"\n=== mean +/- std over {len(seeds)} seeds {seeds} ===")
    print(f"{'baseline':<14} {'ACC':<18} {'FM':<18}")
    for baseline, stats in summary.items():
        acc_str = f"{stats['acc_mean']:.4f} +/- {stats['acc_std']:.4f}"
        fm_str = f"{stats['fm_mean']:.4f} +/- {stats['fm_std']:.4f}"
        print(f"{baseline:<14} {acc_str:<18} {fm_str:<18}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
