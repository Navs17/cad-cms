"""Run baselines (a) naive, (b) medium-only, and (c) medium+fast from one command.

Usage:
    python scripts/run_benchmark.py [--config configs/default.yaml] [--baselines naive medium_only medium_fast]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cadcms.evaluate import compute_acc, compute_fm, write_auroc_csv, write_summary_csv
from cadcms.plotting import plot_auroc_per_task
from cadcms.train import BASELINES as IMPLEMENTED_BASELINES
from cadcms.train import load_config, resolve_config_paths, run_sequential


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--baselines", nargs="+", default=IMPLEMENTED_BASELINES, choices=IMPLEMENTED_BASELINES)
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), REPO_ROOT)
    results_dir = config["paths"]["results_dir"]
    figures_dir = config["paths"]["figures_dir"]

    summary: dict[str, dict[str, float]] = {}
    for baseline in args.baselines:
        print(f"\n=== baseline: {baseline} ===")
        result = run_sequential(config, baseline=baseline)
        tasks, auroc_matrix = result["tasks"], result["auroc_matrix"]

        acc = compute_acc(auroc_matrix, tasks)
        fm = compute_fm(auroc_matrix, tasks)
        summary[baseline] = {"acc": acc, "fm": fm}
        print(f"ACC={acc:.4f}  FM={fm:.4f}")

        for stage_idx, stage in enumerate(auroc_matrix):
            per_task = ", ".join(f"{t}={auroc:.4f}" for t, auroc in stage.items())
            print(f"  stage {stage_idx + 1} ({tasks[stage_idx]}): {per_task}")

        write_auroc_csv(auroc_matrix, tasks, results_dir / f"auroc_{baseline}.csv")
        plot_auroc_per_task(auroc_matrix, tasks, figures_dir / f"auroc_{baseline}.png", title=baseline)

    write_summary_csv(summary, results_dir / "summary.csv")

    print("\n=== summary ===")
    for baseline, metrics in summary.items():
        print(f"{baseline}: ACC={metrics['acc']:.4f}  FM={metrics['fm']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
