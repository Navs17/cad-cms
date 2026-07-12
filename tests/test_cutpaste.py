"""CutPaste augmentation, dataset labeling, and backbone fine-tune/refreeze."""

import random

import numpy as np
from PIL import Image

from cadcms.cutpaste import CutPasteDataset, cutpaste_augment, finetune_backbone_with_cutpaste
from cadcms.data import MVTecDataset, build_transform
from cadcms.features import ResNetBackbone, get_device


def _make_gradient_image(size: int = 64) -> Image.Image:
    # A non-uniform image so a pasted patch is very likely to change pixels.
    arr = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    arr = np.stack([arr, arr.T, np.zeros_like(arr)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _make_train_dataset(tmp_path, count: int = 10) -> MVTecDataset:
    good_dir = tmp_path / "pill" / "train" / "good"
    good_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(good_dir / f"{i:03d}.png")
    return MVTecDataset(tmp_path, "pill", "train", transform=None)


def test_cutpaste_augment_preserves_size_and_changes_pixels():
    image = _make_gradient_image()
    rng = random.Random(0)

    augmented = cutpaste_augment(image, rng)

    assert augmented.size == image.size
    assert np.array(augmented).shape == np.array(image).shape
    assert not np.array_equal(np.array(augmented), np.array(image))


def test_cutpaste_dataset_yields_both_labels(tmp_path):
    base_dataset = _make_train_dataset(tmp_path, count=5)
    transform = build_transform(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    cutpaste_dataset = CutPasteDataset(base_dataset, transform, seed=0)

    labels = set()
    for i in range(30):
        _, label = cutpaste_dataset[i % len(cutpaste_dataset)]
        labels.add(label)

    assert labels == {0, 1}


def test_finetune_backbone_with_cutpaste_refreezes_afterward(tmp_path):
    train_dataset = _make_train_dataset(tmp_path, count=8)
    transform = build_transform(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    backbone = ResNetBackbone(pretrained=False, layer="penultimate", freeze=True)
    device = get_device()

    finetune_backbone_with_cutpaste(
        backbone,
        train_dataset,
        transform,
        epochs=1,
        lr=1e-3,
        batch_size=4,
        device=device,
        seed=0,
    )

    assert backbone.frozen is True
    assert backbone.training is False
    assert all(not p.requires_grad for p in backbone.encoder.parameters())
