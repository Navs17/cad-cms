"""Phase C: targeted, seeded sweep on the drift experiment.

Sweeps ema_rate x confidence_percentile for medium+fast against the
medium-only reference, across every seed in config["seeds"]. Reports the
full grid (mean +/- std over seeds), not a single best cell -- the point is
to see whether any region robustly beats medium-only across seeds, or
whether the whole surface sits below it.

Embeddings are re-extracted once per seed (the drift stream's composition
depends on the seed) and then reused for every (ema_rate, confidence_percentile)
cell at that seed -- those hyperparameters only affect the FAST-memory
adaptation loop, not the stream or the medium memory.

Usage:
    python scripts/run_drift_sweep.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from torch.utils.data import DataLoader

from cadcms.data import DriftStreamDataset, MVTecDataset, build_transform, get_dataloader
from cadcms.evaluate import compute_windowed_auroc
from cadcms.features import ResNetBackbone, extract_embeddings_with_t, get_device, get_embeddings
from cadcms.memory import ContinuumMemory, FastMemory
from cadcms.scorer import score_stream_with_fast_memory
from cadcms.train import load_config, resolve_config_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    config = resolve_config_paths(load_config(args.config), REPO_ROOT)
    task_id = config["tasks"][0]
    device = get_device()
    seeds = config["seeds"]
    ema_rate_values = config["drift_sweep"]["ema_rate_values"]
    confidence_percentile_values = config["drift_sweep"]["confidence_percentile_values"]
    window_size = config["drift"]["window_size"]

    backbone = ResNetBackbone(
        pretrained=config["backbone"]["pretrained"],
        layer=config["backbone"]["layer"],
        freeze=config["backbone"]["freeze"],
    )
    transform = build_transform(
        config["data"]["image_size"], config["data"]["normalize_mean"], config["data"]["normalize_std"]
    )

    medium_only_by_seed: dict[int, float] = {}
    # cms_auroc_by_cell[(ema_rate, confidence_percentile)] -> list of per-seed mean windowed AUROC
    cms_auroc_by_cell: dict[tuple[float, float], list[float]] = {
        (e, c): [] for e in ema_rate_values for c in confidence_percentile_values
    }
    raw_rows: list[dict] = []

    for seed in seeds:
        print(f"\n=== seed {seed} ===")

        train_loader = get_dataloader(
            config["paths"]["data_root"],
            task_id,
            "train",
            transform,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
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

        medium_scores = medium_memory.mahalanobis(stream_batch.embeddings)
        _, medium_auroc = compute_windowed_auroc(stream_t, stream_batch.labels, medium_scores, window_size)
        medium_only_by_seed[seed] = float(medium_auroc.mean())
        print(f"  medium_only mean windowed AUROC: {medium_only_by_seed[seed]:.4f}")

        for ema_rate in ema_rate_values:
            for confidence_percentile in confidence_percentile_values:
                fast_memory = FastMemory(
                    medium_memory,
                    ema_rate=ema_rate,
                    pullback_coefficient=config["fast_memory"]["pullback_coefficient"],
                    shrinkage_alpha=config["memory"]["shrinkage_alpha"],
                )
                recent_scores: deque = deque(maxlen=config["fast_memory"]["recent_scores_window"])
                cms_scores = score_stream_with_fast_memory(
                    stream_batch.embeddings,
                    medium_memory,
                    fast_memory,
                    config["fast_memory"]["fusion_weight"],
                    confidence_percentile,
                    recent_scores,
                )
                _, cms_auroc = compute_windowed_auroc(stream_t, stream_batch.labels, cms_scores, window_size)
                mean_auroc = float(cms_auroc.mean())
                cms_auroc_by_cell[(ema_rate, confidence_percentile)].append(mean_auroc)
                raw_rows.append(
                    {
                        "seed": seed,
                        "ema_rate": ema_rate,
                        "confidence_percentile": confidence_percentile,
                        "mean_windowed_auroc": mean_auroc,
                        "medium_only_mean_windowed_auroc": medium_only_by_seed[seed],
                    }
                )
                print(
                    f"  ema_rate={ema_rate} confidence_percentile={confidence_percentile}: "
                    f"mean windowed AUROC={mean_auroc:.4f}"
                )

    results_dir = config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "drift_sweep_raw.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    medium_only_vals = np.array(list(medium_only_by_seed.values()))
    medium_only_mean, medium_only_std = float(medium_only_vals.mean()), float(medium_only_vals.std())

    summary_rows = []
    for (ema_rate, confidence_percentile), vals in cms_auroc_by_cell.items():
        arr = np.array(vals)
        summary_rows.append(
            {
                "ema_rate": ema_rate,
                "confidence_percentile": confidence_percentile,
                "mean_windowed_auroc": float(arr.mean()),
                "std_windowed_auroc": float(arr.std()),
                "medium_only_mean_windowed_auroc": medium_only_mean,
                "medium_only_std_windowed_auroc": medium_only_std,
                "beats_medium_only_mean": bool(arr.mean() > medium_only_mean),
            }
        )

    with open(results_dir / "drift_sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n=== medium_only reference over {len(seeds)} seeds: {medium_only_mean:.4f} +/- {medium_only_std:.4f} ===")
    print("\n=== medium+fast grid: mean +/- std windowed AUROC over seeds ===")
    header = "ema_rate \\ percentile".ljust(22) + "".join(f"{p:>22}" for p in confidence_percentile_values)
    print(header)
    for ema_rate in ema_rate_values:
        row = f"{ema_rate:<22}"
        for confidence_percentile in confidence_percentile_values:
            vals = np.array(cms_auroc_by_cell[(ema_rate, confidence_percentile)])
            cell = f"{vals.mean():.4f} +/- {vals.std():.4f}"
            row += f"{cell:>22}"
        print(row)

    any_beats = any(r["beats_medium_only_mean"] for r in summary_rows)
    print(f"\nany cell beats medium_only on mean: {any_beats}")
    if any_beats:
        winners = [r for r in summary_rows if r["beats_medium_only_mean"]]
        print("winning cells (mean only, no significance claim):")
        for w in winners:
            print(
                f"  ema_rate={w['ema_rate']} confidence_percentile={w['confidence_percentile']}: "
                f"{w['mean_windowed_auroc']:.4f} +/- {w['std_windowed_auroc']:.4f} "
                f"vs medium_only {medium_only_mean:.4f} +/- {medium_only_std:.4f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
