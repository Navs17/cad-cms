# cad-cms — Continual Anomaly Detection with a Continuum Memory System

Unsupervised continual anomaly detection for industrial/pharmaceutical visual
inspection. The system learns only from **normal** (defect-free) images, adapts
across a sequence of product categories, and resists both catastrophic
forgetting and slow test-time drift using a multi-timescale memory.

This extends earlier MEng work on continual learning for pharmaceutical pill
defect inspection.

## Method

Three memory levels, inspired by "Towards Continual Adaptation in Industrial
Anomaly Detection" (DNE, ACM MM 2022) and "Nested Learning" (Google Research,
NeurIPS 2025):

- **SLOW** — a frozen ImageNet-pretrained ResNet-18 backbone. Features are the
  penultimate layer, global-average-pooled and L2-normalized. Never updated in
  the core pipeline.
- **MEDIUM** — one Gaussian (mean, shrunk covariance, sample count) per task,
  fit once at the end of each task from that task's normal training images
  (DNE-style). At inference, all task Gaussians are fused into one distribution
  — either by DNE's sample-and-refit or by closed-form moment matching of the
  mixture (configurable) — and scored with Mahalanobis distance.
- **FAST** — an exponential-moving-average mean/covariance updated at
  inference time, but only from samples the model scores confidently normal
  (below a configurable percentile of recent scores). It decays toward the
  fused medium memory (configurable pull-back coefficient) so it can track
  drift (lighting, batch variation) without drifting away permanently.

Final anomaly score:

```
score = w * Mahalanobis(fused medium) + (1 - w) * Mahalanobis(fast)
```

with `w` and the EMA rate set in the config. An optional, flag-gated CutPaste
pseudo-anomaly fine-tuning phase (`cutpaste.enabled: true` in the config) can
lightly adapt the backbone with a 2-class head before task 1 (frozen
afterwards); the default pipeline requires no training at all beyond
computing statistics.

## Task protocol

Domain-incremental sequence over MVTec AD categories (pharma-flavored):
`pill -> capsule -> bottle`, configurable in `configs/default.yaml`. Each task
trains only on that category's defect-free train split. Evaluation after each
stage covers the full test split (normal + defective) of all categories seen
so far, without task identity at inference.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Colab: source .venv/bin/activate
pip install -r requirements.txt
```

GPU is optional: the code auto-detects CUDA and falls back to CPU. Developed
on Windows (CPU/local) and trained on Google Colab (T4 GPU).

### Data

MVTec AD requires a manual, license-gated download — this repo does not fetch
it automatically. See [data/README.md](data/README.md) for where to place it,
then verify with:

```bash
python scripts/check_data.py
```

### Tests

```bash
pytest tests/
```

## Running

Place MVTec AD under `data/` first (see above), then:

```bash
# baselines (a) naive, (b) medium-only, (c) medium+fast
python scripts/run_benchmark.py --config configs/default.yaml

# headline drift experiment: medium-only vs full CMS over a drifting task-1 stream
python scripts/run_drift.py --config configs/default.yaml

# ablation sweeps: shrinkage alpha, fusion method, EMA rate, fusion weight
python scripts/run_ablations.py --config configs/default.yaml
```

`run_benchmark.py` writes per-baseline AUROC tables to `results/auroc_<baseline>.csv`,
an ACC/FM summary to `results/summary.csv`, and per-baseline AUROC-per-task
plots to `results/figures/`. `run_drift.py` writes `results/drift_stream.csv`
and `results/figures/drift_stream.png`. `run_ablations.py` writes one
`results/ablation_<name>.csv` per sweep.

## Repository layout

```
configs/default.yaml   all hyperparameters (tasks, image size, shrinkage,
                        EMA rate, fusion weight, thresholds, seeds, paths)
src/cadcms/
  data.py               MVTec datasets/loaders, drift-transform wrapper
  features.py           backbone wrapper, embedding extraction, disk caching
  memory.py             GaussianMemory, ContinuumMemory, FastMemory (update/fuse/save/load)
  scorer.py             Mahalanobis scoring, score fusion, streaming/thresholding
  train.py              sequential task loop: extract -> fit memory -> eval
  evaluate.py            AUROC, ACC, FM, results writing
  plotting.py            result plots
  cutpaste.py            optional, flag-gated: CutPaste pseudo-anomaly
                          backbone fine-tuning before task 1 (Phase 6)
scripts/
  check_data.py          verifies MVTec AD is laid out correctly
  run_benchmark.py       runs baselines (a) naive, (b) medium-only, (c) medium+fast
  run_drift.py            brightness/blur drift-stream experiment
  run_ablations.py        EMA rate / fusion weight / fusion method / shrinkage sweeps
tests/                    pytest unit tests for memory math
notebooks/colab_run.ipynb thin Colab wrapper (clone, install, mount Drive, run)
```

## Evaluation

- Image-level AUROC per task after each stage.
- **ACC** = mean final AUROC over all tasks.
- **FM** (forgetting measure) = mean over tasks of (best AUROC ever achieved on
  that task minus its final AUROC).
- Baselines: (a) naive single Gaussian overwritten each task, (b) medium-only
  (pure DNE-style), (c) medium + fast (full CMS).
- Ablations: EMA rate, fusion weight `w`, fusion method, shrinkage `alpha`.
- Drift experiment: gradually increasing brightness + Gaussian blur on a
  replayed task-1 test stream, comparing medium-only vs full CMS AUROC over
  the stream (headline experiment).

Results are written as CSV to `results/` and plots to `results/figures/`.

## Status

- [x] Phase 1 — repo scaffold
- [x] Phase 2 — data loading + feature extraction + embedding cache
- [x] Phase 3 — memory + scorer + single-task baseline
- [x] Phase 4 — sequential loop, baselines (a)/(b), ACC/FM
- [x] Phase 5 — fast memory, score fusion, drift experiment, ablations
- [x] Phase 6 (optional) — CutPaste fine-tuning
