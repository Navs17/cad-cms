"""Verify that MVTec AD is downloaded and laid out as expected.

Usage:
    python scripts/check_data.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_category(data_root: Path, category: str) -> list[str]:
    """Return a list of problems found for a single category (empty if OK)."""
    problems = []
    category_dir = data_root / category

    if not category_dir.is_dir():
        return [f"missing category folder: {category_dir}"]

    train_good = category_dir / "train" / "good"
    if not train_good.is_dir():
        problems.append(f"missing {train_good}")
    elif not any(train_good.iterdir()):
        problems.append(f"empty: {train_good}")

    test_dir = category_dir / "test"
    if not test_dir.is_dir():
        problems.append(f"missing {test_dir}")
    else:
        test_good = test_dir / "good"
        if not test_good.is_dir():
            problems.append(f"missing {test_good}")
        defect_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        if not defect_dirs:
            problems.append(f"no defect subfolders found under {test_dir}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="path to a cad-cms config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = REPO_ROOT / config["paths"]["data_root"]
    tasks = config["tasks"]

    print(f"Checking MVTec AD layout under: {data_root}")
    if not data_root.is_dir():
        print(f"\nERROR: data root does not exist: {data_root}")
        print("See data/README.md for where to place the dataset.")
        return 1

    all_ok = True
    for category in tasks:
        problems = check_category(data_root, category)
        if problems:
            all_ok = False
            print(f"\n[{category}] FAILED")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"[{category}] OK")

    if not all_ok:
        print("\nSee data/README.md for the expected MVTec AD folder layout.")
        return 1

    print("\nAll categories present and correctly laid out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
