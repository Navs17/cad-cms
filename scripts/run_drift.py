"""Run the drift-stream experiment: medium-only vs full CMS (medium+fast).

Replays task 1's test split with gradually increasing brightness and
Gaussian blur, and compares windowed AUROC over the stream for a static
medium-only memory against the full CMS (medium+fast, adapting online).
This is the headline experiment.

Usage:
    python scripts/run_drift.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch.utils.data import DataLoader

from cadcms.data import DriftStreamDataset, MVTecDataset, build_transform, get_dataloader
from cadcms.evaluate import compute_windowed_auroc, write_drift_csv
from cadcms.features import ResNetBackbone, extract_embeddings_with_t, get_device, get_embeddings
from cadcms.memory import ContinuumMemory, FastMemory
from cadcms.plotting import plot_drift_curve
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

    # Fit the medium memory for task 1 on its (undrifted) normal train images.
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

    # Build and extract the drifted stream (order preserved -- no shuffling).
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

    # medium-only: static scoring, no adaptation.
    medium_scores = medium_memory.mahalanobis(stream_batch.embeddings)

    # full CMS: sequential scoring + online adaptation.
    fast_memory = FastMemory(
        medium_memory,
        ema_rate=config["fast_memory"]["ema_rate"],
        pullback_coefficient=config["fast_memory"]["pullback_coefficient"],
        shrinkage_alpha=config["memory"]["shrinkage_alpha"],
    )
    recent_scores: deque = deque(maxlen=config["fast_memory"]["recent_scores_window"])
    cms_scores = score_stream_with_fast_memory(
        stream_batch.embeddings,
        medium_memory,
        fast_memory,
        config["fast_memory"]["fusion_weight"],
        config["fast_memory"]["confidence_percentile"],
        recent_scores,
    )

    window_t, medium_auroc = compute_windowed_auroc(
        stream_t, stream_batch.labels, medium_scores, config["drift"]["window_size"]
    )
    _, cms_auroc = compute_windowed_auroc(
        stream_t, stream_batch.labels, cms_scores, config["drift"]["window_size"]
    )

    results_dir = config["paths"]["results_dir"]
    figures_dir = config["paths"]["figures_dir"]
    scores_by_method = {"medium_only": medium_auroc, "medium_fast": cms_auroc}
    write_drift_csv(window_t, scores_by_method, results_dir / "drift_stream.csv")
    plot_drift_curve(
        window_t, scores_by_method, figures_dir / "drift_stream.png", title=f"Drift stream ({task_id})"
    )

    print(f"fast memory adapted on {fast_memory.num_updates} / {len(stream_t)} stream samples")
    print(f"medium-only mean windowed AUROC: {medium_auroc.mean():.4f}")
    print(f"medium+fast (full CMS) mean windowed AUROC: {cms_auroc.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
