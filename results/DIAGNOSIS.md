# FAST memory diagnosis

Investigation into why the FAST memory component (baseline (c), `medium_fast`)
underperforms `medium_only` rather than improving on it, on real data
(pill/capsule/tablet, task 1 = pill for the drift experiment). Covers Phases
A-C: seeding, mechanism diagnosis, and a targeted seeded sweep. Method logic
was changed only in Phase C (sweeping existing config knobs); Phases A and B
added seeding/instrumentation only, per the constraints of this investigation.

## 1. Which benchmark gaps are real after seeding (Phase A)

`scripts/run_benchmark_seeds.py`, 5 seeds (42-46), full re-run of naive /
medium_only / medium_fast:

| baseline | ACC | FM |
|---|---|---|
| naive | 0.7245 &plusmn; 0.0000 | 0.1729 &plusmn; 0.0000 |
| medium_only | 0.8808 &plusmn; 0.0000 | 0.0124 &plusmn; 0.0000 |
| medium_fast | 0.8693 &plusmn; 0.0000 | 0.0084 &plusmn; 0.0000 |

**Standard deviation is exactly 0.0000 for every baseline and metric.** This
is not a rounding artifact -- we traced every use of `seed` through the
codebase before running anything and confirmed empirically after: with the
default `fusion_method: moment_matching`, the main benchmark path has **no
stochastic component**. `ContinuumMemory.rng` is only consumed by
`fuse_sample_refit` (not the default); `get_dataloader` never shuffles;
`FastMemory`'s streaming update is a deterministic threshold rule with no
internal RNG.

**Conclusion:** the retention gap (naive vs. medium_only) and the
accuracy/forgetting gap (medium_only vs. medium_fast) are both **100%
reproducible**, in the strongest sense -- not because they survived noise,
but because there is no noise source in this path to have gotten lucky
against. `medium_only` beating `medium_fast` on ACC (0.8808 vs. 0.8693) is
real. `medium_fast` beating `medium_only` on FM (0.0084 vs. 0.0124) is also
real -- a small but genuine forgetting-resistance improvement on the main
benchmark, worth noting even though it isn't what motivated this
investigation (the main benchmark applies no synthetic drift; it doesn't
exercise the scenario FAST memory is designed for).

## 2. Confirmed mechanism (Phase B)

Hypothesis under test: *drifted-but-normal samples get scored as anomalous,
fail the confidence gate, so FAST never learns the new normal.*

`scripts/diagnose_drift_gate.py` instruments the exact scoring/gating/update
path via an additive observation hook on `score_stream_with_fast_memory`
(`on_sample`, default `None` -- verified with a dedicated test that hooked
and unhooked runs produce byte-identical scores and FAST-memory state, so
this is pure instrumentation, not a behavior change). Run on the real pill
drift stream (500 samples, default hyperparameters: `ema_rate=0.01`,
`confidence_percentile=50`, `pullback_coefficient=0.05`, `fusion_weight=0.5`).

**Aggregate (misleading on its own):**

| | value |
|---|---|
| gate pass rate | 108/500 (21.6%) |
| normal gate recall (fraction of true normals let through) | 31.8% |
| gate purity (fraction of gated-in samples that were normal) | 19.4% |

Read in isolation this looks like a blend of both failure modes the
hypothesis considered (normals rejected *and* defects leaking through). The
windowed breakdown shows the real story is temporal, not steady-state:

| stream region | pass rate | normal recall | purity |
|---|---|---|---|
| t=0.05 (mild drift) | 68% | 100% | 15% |
| t=0.15 | 64% | 100% | 6% |
| t=0.25 | 48% | 100% | 38% |
| t=0.35 | 36% | 56% | 28% |
| **t=0.45 -- 0.95 (moderate-severe drift)** | **0%** | **0%** | **undefined (nothing passes)** |

The gate works reasonably for the first 4 windows (200/500 samples,
recall 100% down to 56%), then **shuts completely for the remaining 6
windows -- 300/500 samples, 60% of the stream.** `fast_mean_dist_from_medium`
plateaus at exactly the same point (0.0162, unchanged for the rest of the
run): once the gate closes, FAST stops adapting entirely and stays frozen.

**Mechanism, confirmed quantitatively** by comparing the mean incoming score
against the mean gate threshold per window:

| t | mean incoming score | mean threshold | gap |
|---|---|---|---|
| 0.05 | 23.8 | 23.2 | 0.66 |
| 0.35 | 27.9 | 21.2 | 6.72 |
| 0.55 | 36.9 | 24.4 | 12.45 |
| 0.75 | 52.9 | 36.1 | 16.77 |
| 0.95 | 64.6 | 51.1 | 13.57 |

The threshold is a percentile of a *trailing* window (up to the last 200
scores). Under sustained monotonic drift, incoming scores climb faster than
the trailing threshold can follow -- the gap widens for most of the stream.
**This is a structural property of percentile-of-recent-scores gating under
a monotonic trend, not a mistunable artifact.** Any fixed
`confidence_percentile` gets outrun the same way; it only changes how long it
takes.

**Secondary confound identified (not the primary driver, but real):** the
drift stream's class composition is skewed toward defects. `pill/test/`
has 26 `good` images against 141 defective images (7 defect-type
subfolders); `DriftStreamDataset` samples uniformly at random from the whole
test split without controlling for class balance, so the resulting stream is
~87% defective, not the mostly-normal composition a real production line
would have. This dilutes `gate_purity` throughout (even a well-functioning
gate admits many defects simply because they're the majority class) but is
distinct from the temporal freeze -- the freeze dominates the accuracy cost,
since it zeroes out adaptation for 60% of the stream regardless of purity.

**Verdict: hypothesis confirmed, and sharper than stated.** It is not "the
gate is generally too strict." It is "the gate works acceptably under mild
drift and then closes entirely and permanently once drift exceeds what the
trailing window can track."

## 3. Targeted seeded sweep (Phase C)

`scripts/run_drift_sweep.py`: `ema_rate` in {0.01, 0.05, 0.1, 0.3} x
`confidence_percentile` in {50, 75, 90}, all 5 seeds, against a per-seed
`medium_only` reference (the drift stream is seed-dependent, so the
reference is recomputed per seed, not fixed).

**medium_only reference: 0.6907 &plusmn; 0.0290** (mean &plusmn; std windowed AUROC
over 5 seeds)

**medium+fast grid (mean &plusmn; std windowed AUROC over 5 seeds):**

| ema_rate \ percentile | 50 | 75 | 90 |
|---|---|---|---|
| 0.01 | 0.6720 &plusmn; 0.026 | 0.6868 &plusmn; 0.025 | 0.6835 &plusmn; 0.023 |
| 0.05 | 0.6682 &plusmn; 0.026 | 0.6808 &plusmn; 0.025 | 0.6707 &plusmn; 0.022 |
| 0.1 | 0.6725 &plusmn; 0.029 | 0.6807 &plusmn; 0.026 | 0.6683 &plusmn; 0.021 |
| 0.3 | 0.6697 &plusmn; 0.033 | 0.6814 &plusmn; 0.032 | 0.6715 &plusmn; 0.020 |

**Every one of the 12 cells sits below the medium_only reference.**
`beats_medium_only_mean` is `False` for all 12, on the mean over 5 seeds --
not a single cell won even once. There is a real, small, internal effect
within the grid (`confidence_percentile=75` is consistently the best choice,
beating both 50 and 90 in every row; `ema_rate` has almost no effect across
the tested range), which is exactly consistent with the Phase B mechanism:
raising the percentile delays gate closure, it does not prevent it, so it
recovers a little AUROC without closing the gap to medium_only.

## 4. Honest conclusion

**Confidence-gated statistical adaptation, as currently specified, does not
help on this drift benchmark, and there is no tuned configuration in the
swept range that changes that.** The forgetting-resistance benefit on the
main benchmark (FM 0.0084 vs 0.0124) is real but modest and doesn't offset
the drift-experiment cost, which is the scenario FAST memory exists for.

The mechanism is now understood, not mysterious: a percentile-of-recent-scores
gate is a trailing indicator. Under a sustained monotonic drift trend, it
falls further and further behind, and beyond a certain drift severity it
closes completely -- at which point FAST memory is a frozen, stale copy of
whatever it last learned, contributing an outdated opinion to every fused
score for the rest of the stream. This is a structural property of the
gating rule's temporal design, not a hyperparameter that happened to be set
wrong.

There's also a deeper, partially unfixable-by-regating limit worth naming
honestly: once drift pushes true-normal samples' scores past where the
*original* defects used to sit, a scalar Mahalanobis-distance gate cannot
losslessly separate "drifted-but-normal" from "genuinely defective" by score
magnitude alone -- the ranking itself degrades. Re-gating fixes can push this
limit back, but probably can't eliminate it entirely; that's an argument for
testing any proposed fix at multiple drift severities, not just confirming
it helps somewhere.

## 5. Proposed fixes (not implemented -- for review before any further work)

1. **Trend-extrapolated (leading) threshold, instead of a trailing
   percentile.** Fit a simple linear trend to the scores in the recent
   window and extrapolate the threshold forward by roughly the window's
   reaction lag, rather than using the raw percentile of past scores as-is.
   This targets the diagnosed mechanism directly: the threshold would track
   *where the distribution is heading*, not *where it was*. Small,
   localized change (replaces `percentile_threshold`'s input in
   `score_stream_with_fast_memory`); testable in isolation by re-running
   `diagnose_drift_gate.py` and checking whether the gate stays open past
   t=0.45.

2. **Explicit drift-regime detection that relaxes the gate under sustained
   trend.** Track the velocity of `recent_scores` (e.g. an EWMA of
   score-to-score deltas); when a sustained upward trend is detected,
   temporarily loosen the gate (raise the effective percentile, or switch to
   gating on relative rank within a local batch rather than an absolute
   percentile) so FAST can keep absorbing *some* signal during drift instead
   of freezing outright. More complex than (1) -- effectively a two-state
   controller (calm vs. drifting) -- but addresses the "gate should notice
   it's being systematically outrun and compensate" failure mode more
   directly than adjusting a single static percentile ever can.

Lower-priority, experimental-design fix worth doing alongside either of the
above: make `DriftStreamDataset` sample with a configurable normal/defect
ratio (or report `gate_purity`/`normal_gate_recall` normalized against the
stream's actual base rate) so future diagnosis isn't confounded by an
accidentally defect-heavy stream inherited from a test split's category
composition.
