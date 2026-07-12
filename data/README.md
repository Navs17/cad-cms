# Data

## MVTec AD

This project uses [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad),
a license-gated dataset that must be downloaded manually from the MVTec
website (registration required). It is **not** downloaded automatically by
any script in this repo.

### Where to place it

Download and extract the archive so the layout looks like:

```
data/MVTec-AD/
  pill/
    train/
      good/
        000.png
        001.png
        ...
    test/
      good/
        000.png
        ...
      <defect_type_1>/
        000.png
        ...
      <defect_type_2>/
        ...
    ground_truth/
      <defect_type_1>/
        000_mask.png
        ...
  capsule/
    train/...
    test/...
    ground_truth/...
  bottle/
    train/...
    test/...
    ground_truth/...
```

Each category directory follows the standard MVTec AD layout: a `train/good/`
split containing only defect-free images, a `test/` split with a `good/`
subfolder plus one subfolder per defect type, and a `ground_truth/` folder
with per-defect-type pixel masks (not required for the image-level AUROC
protocol used here, but kept for completeness/future use).

The `data_root` path is set in `configs/default.yaml` (`paths.data_root`,
default `data/MVTec-AD`).

### Verifying the download

```bash
python scripts/check_data.py
```

This confirms that every category listed in `configs/default.yaml` (`tasks`)
is present with the expected `train/good` and `test/*` subfolders.
