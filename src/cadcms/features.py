"""Backbone wrapper, embedding extraction, and disk caching.

The backbone is the SLOW memory level: a frozen ImageNet-pretrained ResNet-18.
Features are the penultimate layer (post global-average-pool), L2-normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


def get_device() -> torch.device:
    """Auto-detect CUDA, fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetBackbone(nn.Module):
    """Frozen ResNet-18 feature extractor.

    Outputs the global-average-pooled penultimate layer (512-dim for
    ResNet-18), L2-normalized. Only ``layer="penultimate"`` is supported.
    """

    EMBEDDING_DIM = 512

    def __init__(self, pretrained: bool = True, layer: str = "penultimate", freeze: bool = True) -> None:
        super().__init__()
        if layer != "penultimate":
            raise NotImplementedError(f"only layer='penultimate' is supported, got {layer!r}")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = resnet18(weights=weights)
        # Drop the final fc layer; keep everything up to (and including) avgpool.
        self.encoder = nn.Sequential(*list(full_model.children())[:-1])

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.frozen = freeze

    def train(self, mode: bool = True):
        # Keep a frozen backbone in eval mode regardless of caller intent
        # (e.g. an enclosing training loop for a future fine-tuning phase).
        if self.frozen:
            return super().train(False)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)          # (B, 512, 1, 1)
        features = torch.flatten(features, 1)  # (B, 512)
        return F.normalize(features, p=2, dim=1)


@dataclass
class EmbeddingBatch:
    """Extracted embeddings for one category/split, plus metadata."""

    embeddings: np.ndarray  # (N, D), float32, L2-normalized
    labels: np.ndarray      # (N,), int64, 0=normal 1=anomalous
    paths: list[str]


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EmbeddingBatch:
    """Run the frozen backbone over a dataloader and collect embeddings."""
    model = model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(np.asarray(labels))
            all_paths.extend(paths)

    return EmbeddingBatch(
        embeddings=np.concatenate(all_embeddings, axis=0).astype(np.float32),
        labels=np.concatenate(all_labels, axis=0).astype(np.int64),
        paths=all_paths,
    )


def cache_path(cache_dir: str | Path, category: str, split: str, backbone_name: str) -> Path:
    return Path(cache_dir) / f"{category}_{split}_{backbone_name}.npz"


def load_cached_embeddings(path: Path) -> Optional[EmbeddingBatch]:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return EmbeddingBatch(
        embeddings=data["embeddings"],
        labels=data["labels"],
        paths=list(data["paths"]),
    )


def save_embeddings(path: Path, batch: EmbeddingBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=batch.embeddings,
        labels=batch.labels,
        paths=np.array(batch.paths),
    )


def get_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    cache_dir: str | Path,
    category: str,
    split: str,
    backbone_name: str,
    device: torch.device,
    force_recompute: bool = False,
) -> tuple[EmbeddingBatch, bool]:
    """Fetch embeddings for a category/split, using the disk cache when possible.

    Returns ``(batch, cache_hit)``.
    """
    path = cache_path(cache_dir, category, split, backbone_name)

    if not force_recompute:
        cached = load_cached_embeddings(path)
        if cached is not None:
            return cached, True

    batch = extract_embeddings(model, dataloader, device)
    save_embeddings(path, batch)
    return batch, False
