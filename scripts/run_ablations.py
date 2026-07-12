"""Run ablation sweeps: shrinkage alpha, fusion method, EMA rate, fusion weight.

Usage:
    python scripts/run_ablations.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cadcms.evaluate import compute_acc, compute_fm
from cadcms.train import load_config, resolve_config_paths, run_sequential


def write_sweep_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(config: dict, baseline: str) -> tuple[float, float]:
    result = run_sequential(config, baseline=baseline)
    acc = compute_acc(result["auroc_matrix"], result["tasks"])
    fm = compute_fm(result["auroc_matrix"], result["tasks"])
    return acc, fm


def sweep_shrinkage_alpha(config: dict) -> list[dict]:
    rows = []
    for alpha in config["ablations"]["shrinkage_alpha_values"]:
        cfg = dict(config)
        cfg["memory"] = {**config["memory"], "shrinkage_alpha": alpha}
        acc, fm = _run_one(cfg, "medium_only")
        rows.append({"shrinkage_alpha": alpha, "acc": acc, "fm": fm})
        print(f"  shrinkage_alpha={alpha}: ACC={acc:.4f} FM={fm:.4f}")
    return rows


def sweep_fusion_method(config: dict) -> list[dict]:
    rows = []
    for method in config["ablations"]["fusion_methods"]:
        cfg = dict(config)
        cfg["memory"] = {**config["memory"], "fusion_method": method}
        acc, fm = _run_one(cfg, "medium_only")
        rows.append({"fusion_method": method, "acc": acc, "fm": fm})
        print(f"  fusion_method={method}: ACC={acc:.4f} FM={fm:.4f}")
    return rows


def sweep_ema_rate(config: dict) -> list[dict]:
    rows = []
    for rate in config["ablations"]["ema_rate_values"]:
        cfg = dict(config)
        cfg["fast_memory"] = {**config["fast_memory"], "ema_rate": rate}
        acc, fm = _run_one(cfg, "medium_fast")
        rows.append({"ema_rate": rate, "acc": acc, "fm": fm})
        print(f"  ema_rate={rate}: ACC={acc:.4f} FM={fm:.4f}")
    return rows


def sweep_fusion_weight(config: dict) -> list[dict]:
    rows = []
    for w in config["ablations"]["fusion_weight_values"]:
        cfg = dict(config)
        cfg["fast_memory"] = {**config["fast_memory"], "fusion_weight": w}
        acc, fm = _run_one(cfg, "medium_fast")
        rows.append({"fusion_weight": w, "acc": acc, "fm": fm})
        print(f"  fusion_weight={w}: ACC={acc:.4f} FM={fm:.4f}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), REPO_ROOT)
    results_dir = config["paths"]["results_dir"]

    print("=== shrinkage alpha sweep (medium_only) ===")
    write_sweep_csv(sweep_shrinkage_alpha(config), results_dir / "ablation_shrinkage_alpha.csv")

    print("=== fusion method sweep (medium_only) ===")
    write_sweep_csv(sweep_fusion_method(config), results_dir / "ablation_fusion_method.csv")

    print("=== EMA rate sweep (medium_fast) ===")
    write_sweep_csv(sweep_ema_rate(config), results_dir / "ablation_ema_rate.csv")

    print("=== fusion weight sweep (medium_fast) ===")
    write_sweep_csv(sweep_fusion_weight(config), results_dir / "ablation_fusion_weight.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
