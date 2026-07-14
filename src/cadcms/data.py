"""MVTec AD datasets/loaders, the drift-transform wrapper, and the
contamination (rising-defect-rate) stream builder.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def build_transform(image_size: int, mean: list[float], std: list[float]) -> Callable:
    """Standard resize -> tensor -> normalize pipeline (ImageNet stats for a pretrained backbone)."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class MVTecDataset(Dataset):
    """A single MVTec AD category/split.

    ``split="train"`` yields only the defect-free ``train/good`` images
    (label 0). ``split="test"`` yields every subfolder under ``test/``:
    label 0 for ``good``, label 1 for any defect-type folder.
    """

    def __init__(
        self,
        data_root: str | Path,
        category: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.data_root = Path(data_root)
        self.category = category
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = self._collect_samples()

    def _collect_samples(self) -> list[tuple[Path, int, str]]:
        category_dir = self.data_root / self.category
        samples: list[tuple[Path, int, str]] = []

        if self.split == "train":
            good_dir = category_dir / "train" / "good"
            for path in sorted(good_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, 0, "good"))
        else:
            test_dir = category_dir / "test"
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == "good" else 1
                for path in sorted(defect_dir.iterdir()):
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((path, label, defect_dir.name))

        if not samples:
            raise RuntimeError(
                f"no images found for category={self.category!r} split={self.split!r} "
                f"under {category_dir}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, defect_type = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path)


def get_dataloader(
    data_root: str | Path,
    category: str,
    split: str,
    transform: Optional[Callable],
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for one MVTec category/split."""
    dataset = MVTecDataset(data_root, category, split, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _lerp(value_range: tuple[float, float], t: float) -> float:
    lo, hi = value_range
    return lo + t * (hi - lo)


class DriftStreamDataset(Dataset):
    """Replays a base MVTec split with gradually increasing brightness and blur.

    Simulates slow test-time drift (e.g. lighting change, lens softening).
    Index ``i`` in ``[0, length)`` maps to a drift severity ``t = i / (length - 1)``
    in ``[0, 1]``, linearly interpolated across ``brightness_range`` and
    ``blur_sigma_range``. Samples are drawn (with replacement, seeded) from
    ``base_dataset``, which must be constructed with ``transform=None`` so the
    raw PIL image is available for the brightness/blur ops before the final
    ``transform`` (resize/tensor/normalize) is applied.
    """

    def __init__(
        self,
        base_dataset: MVTecDataset,
        length: int,
        brightness_range: tuple[float, float],
        blur_sigma_range: tuple[float, float],
        transform: Optional[Callable],
        seed: int = 0,
    ) -> None:
        if base_dataset.transform is not None:
            raise ValueError("base_dataset must be built with transform=None for DriftStreamDataset")

        self.base_dataset = base_dataset
        self.length = length
        self.brightness_range = brightness_range
        self.blur_sigma_range = blur_sigma_range
        self.transform = transform

        rng = random.Random(seed)
        self.stream_indices = [rng.randrange(len(base_dataset)) for _ in range(length)]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image, label, path = self.base_dataset[self.stream_indices[index]]
        t = index / max(self.length - 1, 1)

        brightness = _lerp(self.brightness_range, t)
        sigma = _lerp(self.blur_sigma_range, t)

        image = TF.adjust_brightness(image, brightness)
        if sigma > 0:
            kernel_size = max(3, int(2 * round(3 * sigma) + 1))
            image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, path, t


def build_contamination_stream(
    embeddings: np.ndarray,
    labels: np.ndarray,
    length: int,
    start_defect_rate: float,
    end_defect_rate: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a rising-defect-rate stream directly from already-extracted
    embeddings (e.g. cached test-split embeddings/labels) -- no image-level
    transform is involved, since only the class mix changes over the stream,
    not any visual property, so there is nothing here that requires
    re-running the backbone.

    Samples are drawn with replacement from the normal (label 0) / defective
    (label 1) pools of ``embeddings``/``labels``. Index ``i`` maps to
    ``t = i / (length - 1)`` in ``[0, 1]``, with per-position defect
    probability linearly interpolated from ``start_defect_rate`` to
    ``end_defect_rate`` -- the same ``t`` semantics as ``DriftStreamDataset``,
    so existing windowed-AUROC code works unchanged on this stream too.

    Returns ``(stream_embeddings, stream_labels, t)``.
    """
    labels = np.asarray(labels)
    normal_indices = np.flatnonzero(labels == 0)
    defect_indices = np.flatnonzero(labels == 1)
    if len(normal_indices) == 0 or len(defect_indices) == 0:
        raise ValueError("contamination stream requires at least one normal and one defective sample")

    rng = np.random.default_rng(seed)
    t = np.array([i / max(length - 1, 1) for i in range(length)])
    defect_rate = start_defect_rate + t * (end_defect_rate - start_defect_rate)
    is_defect = rng.random(length) < defect_rate

    stream_indices = np.empty(length, dtype=np.int64)
    stream_indices[is_defect] = rng.choice(defect_indices, size=int(is_defect.sum()))
    stream_indices[~is_defect] = rng.choice(normal_indices, size=int((~is_defect).sum()))

    return embeddings[stream_indices], labels[stream_indices], t
