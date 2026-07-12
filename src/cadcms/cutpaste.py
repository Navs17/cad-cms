"""CutPaste pseudo-anomaly fine-tuning (Phase 6, flag-gated: ``cutpaste.enabled``).

DNE-style: a short self-supervised fine-tune (normal vs CutPaste-augmented,
i.e. a patch cut from the image and pasted elsewhere) lightly adapts the
backbone once, before task 1. The 2-class head is then discarded and the
backbone is re-frozen for the rest of the pipeline. Off by default -- the
default pipeline runs with a fully frozen backbone and no training at all.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cadcms.data import MVTecDataset
from cadcms.features import ResNetBackbone


def cutpaste_augment(
    image: Image.Image,
    rng: random.Random,
    area_ratio: tuple[float, float] = (0.02, 0.15),
    aspect_ratio: float = 0.3,
) -> Image.Image:
    """Cut a random rectangular patch from ``image`` and paste it at a
    different random location, creating a synthetic local defect.
    """
    width, height = image.size
    image_area = width * height

    patch_area = rng.uniform(*area_ratio) * image_area
    aspect = rng.uniform(aspect_ratio, 1.0 / aspect_ratio)
    patch_w = max(1, min(int(round((patch_area * aspect) ** 0.5)), width - 1))
    patch_h = max(1, min(int(round((patch_area / aspect) ** 0.5)), height - 1))

    src_x, src_y = rng.randint(0, width - patch_w), rng.randint(0, height - patch_h)
    dst_x, dst_y = rng.randint(0, width - patch_w), rng.randint(0, height - patch_h)

    patch = image.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))
    augmented = image.copy()
    augmented.paste(patch, (dst_x, dst_y))
    return augmented


class CutPasteDataset(Dataset):
    """Wraps a raw (transform=None) MVTec train split. Each item is either
    the original normal image (label 0) or a CutPaste-augmented version of
    it (label 1), chosen with 50/50 probability.
    """

    def __init__(self, base_dataset: MVTecDataset, transform: Callable, seed: int = 0) -> None:
        if base_dataset.transform is not None:
            raise ValueError("base_dataset must be built with transform=None for CutPasteDataset")
        self.base_dataset = base_dataset
        self.transform = transform
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, _, _ = self.base_dataset[index]
        label = self.rng.randint(0, 1)
        if label == 1:
            image = cutpaste_augment(image, self.rng)
        return self.transform(image), label


class CutPasteHead(nn.Module):
    """Temporary 2-class (normal vs CutPaste) head on top of backbone features."""

    def __init__(self, embedding_dim: int = ResNetBackbone.EMBEDDING_DIM) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def finetune_backbone_with_cutpaste(
    backbone: ResNetBackbone,
    train_dataset: MVTecDataset,
    transform: Callable,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> None:
    """Unfreeze ``backbone``, fine-tune it plus a temporary 2-class head to
    discriminate normal vs CutPaste-augmented images, then re-freeze the
    backbone and discard the head. Mutates ``backbone`` in place.
    """
    dataset = CutPasteDataset(train_dataset, transform, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    backbone.to(device)
    backbone.frozen = False
    for param in backbone.encoder.parameters():
        param.requires_grad = True
    backbone.train()

    head = CutPasteHead().to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)

    for epoch in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            features = backbone(images)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += images.size(0)

        print(f"  cutpaste epoch {epoch + 1}/{epochs}: loss={total_loss / n:.4f} acc={correct / n:.4f}")

    for param in backbone.encoder.parameters():
        param.requires_grad = False
    backbone.frozen = True
    backbone.eval()
