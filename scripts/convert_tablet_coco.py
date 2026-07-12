"""One-off converter: the uploaded "tablet defect detection" dataset is a
Roboflow COCO object-detection export, not MVTec AD format. This script
derives an MVTec-style tablet category from it, using only images with
unambiguous image-level labels:

  - images annotated ONLY with category "no-defect"  -> normal
  - images annotated ONLY with category "defected"    -> defective
  - unannotated images (defect status unknown)         -> discarded

Normal images are split into train/test (80/20 by default, seeded); all
defective images go to test/defect (training is defect-free-only, as with
every other category).

Usage:
    python scripts/convert_tablet_coco.py \\
        [--source "data/MVTec-AD/tablet defect detection.coco.zip"] \\
        [--dest data/MVTec-AD/tablet] [--train-fraction 0.8] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_labels(zf: zipfile.ZipFile, annotations_member: str) -> dict:
    with zf.open(annotations_member) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    image_filenames = {img["id"]: img["file_name"] for img in data["images"]}

    labels_per_image: dict[int, set[str]] = {}
    for ann in data["annotations"]:
        labels_per_image.setdefault(ann["image_id"], set()).add(categories[ann["category_id"]])

    good_ids = [i for i, labels in labels_per_image.items() if labels == {"no-defect"}]
    defect_ids = [i for i, labels in labels_per_image.items() if labels == {"defected"}]
    skipped = len(image_filenames) - len(good_ids) - len(defect_ids)

    return {
        "image_filenames": image_filenames,
        "good_ids": good_ids,
        "defect_ids": defect_ids,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "data" / "MVTec-AD" / "tablet defect detection.coco.zip",
    )
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "MVTec-AD" / "tablet")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with zipfile.ZipFile(args.source) as zf:
        annotations_member = next(n for n in zf.namelist() if n.endswith("_annotations.coco.json"))
        labels = load_labels(zf, annotations_member)

        rng = random.Random(args.seed)
        good_ids = labels["good_ids"][:]
        rng.shuffle(good_ids)
        split_point = int(len(good_ids) * args.train_fraction)
        train_ids, test_good_ids = good_ids[:split_point], good_ids[split_point:]

        train_dir = args.dest / "train" / "good"
        test_good_dir = args.dest / "test" / "good"
        test_defect_dir = args.dest / "test" / "defect"
        for d in (train_dir, test_good_dir, test_defect_dir):
            d.mkdir(parents=True, exist_ok=True)

        def extract(image_id: int, dest_dir: Path, index: int) -> None:
            member = f"train/{labels['image_filenames'][image_id]}"
            suffix = Path(labels["image_filenames"][image_id]).suffix
            with zf.open(member) as src, open(dest_dir / f"{index:04d}{suffix}", "wb") as dst:
                dst.write(src.read())

        for i, image_id in enumerate(train_ids):
            extract(image_id, train_dir, i)
        for i, image_id in enumerate(test_good_ids):
            extract(image_id, test_good_dir, i)
        for i, image_id in enumerate(labels["defect_ids"]):
            extract(image_id, test_defect_dir, i)

    print(f"train/good:  {len(train_ids)}")
    print(f"test/good:   {len(test_good_ids)}")
    print(f"test/defect: {len(labels['defect_ids'])}")
    print(f"discarded (unannotated): {labels['skipped']}")
    print(f"wrote to: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
