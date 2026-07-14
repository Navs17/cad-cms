# Fix 3 results: gate-free slow FAST memory

Tests whether removing the confidence gate entirely -- replacing it with a
very slow EMA rate and a strong pull-back toward the fused medium memory --
fixes the drift failure documented in `results/DIAGNOSIS.md` (the gate closes
completely under sustained monotonic drift, freezing FAST memory). Compares
three conditions across 5 seeds (42-46): `medium_only` (no fast memory,
reference), `gated` (the original, documented failure), `gatefree_slow`
(Fix 3). Full methodology, code, and raw/summary/windowed CSVs are in
`scripts/run_fix3_comparison.py` and `results/fix3_*.csv`.

## Question 1: Does gate-free slow FAST beat medium-only and gated under drift, across seeds?

**Monotonic (color/blur) drift stream** -- mean windowed AUROC over 5 seeds:

| condition | mean &plusmn; std |
|---|---|
| medium_only | 0.6907 &plusmn; 0.0290 |
| gated | 0.6720 &plusmn; 0.0262 |
| **gatefree_slow** | **0.6987 &plusmn; 0.0255** |

Per-seed win counts: gatefree_slow beats gated on **5/5 seeds** and beats
medium_only on **4/5 seeds** (the one exception is seed 44: 0.7133 vs.
0.7151, a 0.0018 gap -- effectively a tie, not a loss).

The windowed breakdown (`results/fix3_drift_monotonic_windowed.csv`) shows
*where* the advantage comes from, and it's mechanistically coherent with the
Phase B diagnosis: gatefree_slow is roughly tied with (occasionally slightly
below) medium_only through the first half of the stream (t=0.05-0.55), then
pulls clearly and consistently ahead for the rest of it -- exactly the
region where the gated version's confidence gate had already shut completely
(0% pass rate for t&ge;0.45, per `DIAGNOSIS.md`):

| t | medium_only | gated | gatefree_slow |
|---|---|---|---|
| 0.65 | 0.656 | 0.637 | 0.668 |
| 0.75 | 0.549 | 0.541 | **0.589** |
| 0.85 | 0.571 | 0.562 | **0.589** |
| 0.95 | 0.521 | 0.514 | **0.583** |

At the worst drift severity (t=0.95), gatefree_slow beats medium_only by
0.062 AUROC and gated by 0.069 -- the largest margin anywhere in the stream,
precisely where the original design failed hardest.

**Contamination (rising-defect-rate) stream** -- mean windowed AUROC over 5 seeds:

| condition | mean &plusmn; std |
|---|---|
| medium_only | 0.8323 &plusmn; 0.0201 |
| gated | 0.8307 &plusmn; 0.0227 |
| **gatefree_slow** | **0.8409 &plusmn; 0.0203** |

Per-seed win counts: gatefree_slow beats **both** medium_only and gated on
**5/5 seeds** -- no exceptions, unlike the monotonic stream's one near-tie.

**Answer: yes, on both streams, consistently across seeds -- not a single
lucky seed or lucky stream.** This is a real, if moderate, improvement: 0.008
AUROC over medium_only on drift, 0.010 on contamination. It is not a
dramatic reversal (nothing here approaches, say, a 0.1+ AUROC swing), but it
is consistent in direction and, on the monotonic stream, concentrated
exactly where the mechanism predicts it should be.

## Question 2: Does it pass the contamination test, or does it normalize defects?

This is the test that most directly targets the risk in removing the gate:
with nothing filtering incoming samples, does FAST memory slowly absorb
defects as "normal" once they become common, causing AUROC to collapse as
the defect rate climbs toward 60%?

Windowed breakdown (`results/fix3_drift_contamination_windowed.csv`), mean
AUROC over 5 seeds per window (t is stream position; actual defect rate at
each t interpolates linearly from 5% at t=0 to 60% at t=1):

| t (~defect rate) | medium_only | gated | gatefree_slow |
|---|---|---|---|
| 0.10 (~10%) | 0.821 | 0.824 | 0.840 |
| 0.30 (~22%) | 0.787 | 0.790 | 0.797 |
| 0.50 (~33%) | 0.865 | 0.850 | **0.868** |
| 0.70 (~44%) | 0.839 | 0.839 | 0.846 |
| **0.90 (~55%)** | 0.850 | 0.851 | **0.853** |

**No collapse anywhere.** gatefree_slow's AUROC does not trend downward as
the defect rate rises -- it is flat-to-slightly-rising across the whole
stream, and it stays at or above both other conditions in every single
window, including the highest-contamination window. If gate-free adaptation
were quietly normalizing defects, this is exactly the curve that would bend
downward toward the right; it doesn't.

**Answer: it passes.** At the contamination levels tested here (up to
~60% defective), gate-free slow FAST does not visibly degrade from
absorbing defects as "normal." The strong pull-back
(`gatefree_pullback_coefficient=0.2`) combined with the very slow EMA rate
(`gatefree_ema_rate=0.005`) appears to be doing its intended job: enough
inertia that a rising share of defective samples in the incoming stream
can't drag the fast mean toward them within the tested range.

One honest limit on this conclusion: 60% is the highest contamination level
tested. This result says gate-free is robust *up to* that level, not that it
is robust at arbitrary contamination -- a stream that goes to 90%+ defective
was not tested, and the mechanism (strong-but-finite pull-back) would
plausibly degrade eventually at a high enough sustained contamination level.

## Question 3: is a gate unnecessary, or does contamination prove a smarter gate is needed?

Neither answer is clean; both parts of the evidence matter and they point in
different directions depending on what's being optimized:

**On drift robustness (both streams tested), gate-free slow FAST is the
better choice.** It beats medium_only and gated consistently across seeds,
it does not exhibit the pathological freeze that made gated fail, and it
passes the contamination stress test with no visible defect-normalization up
to 60% contamination. If the deployment scenario is "the visual/environment
conditions drift, or the defect rate creeps up over time, and the priority
is not losing ground under those conditions," gate-free slow is a clean win
over both alternatives tested.

**On the ordinary (non-drift) continual benchmark, gate-free slow is the
worst of the three on forgetting.** Main benchmark (5 seeds, all std=0.0000 --
this pipeline path has no stochastic component, see `DIAGNOSIS.md` Phase A):

| condition | ACC | FM |
|---|---|---|
| medium_only | 0.8808 | 0.0124 |
| gated | 0.8693 | 0.0084 |
| gatefree_slow | 0.8701 | **0.0183** |

ACC is essentially tied between gated and gatefree_slow (both below
medium_only, consistent with the earlier finding that FAST memory of either
kind costs some accuracy on the plain sequential-task benchmark). But
gatefree_slow's forgetting measure is 2.2x gated's and 1.5x medium_only's --
the worst of the three. Plausible mechanism (not directly instrumented in
this investigation): with no gate, FAST continuously adapts to whatever it's
currently streaming through, including ordinary within-task variation that
was never a problem for gated (which mostly held its ground once initialized
near the medium memory). That continuous drift may make FAST's state more
sensitive to *whichever task was streamed most recently*, which is exactly
what the forgetting measure penalizes.

**Recommendation: don't treat this as "gate-free replaces gated." Treat it
as "the two serve different regimes, and the honest fix is conditional, not
universal."** Concretely:

- If drift/contamination robustness is the priority (the stated motivation
  for FAST memory in the first place -- see `README.md`'s "track drift...
  without drifting away permanently"), gate-free slow is the better default
  and a gate is not necessary to prevent defect-normalization, at least up
  to the contamination levels tested here.
- If forgetting resistance on stable (non-drifting) sequences matters more,
  the original gated design's lower FM is real and gate-free slow gives some
  of it up.
- This doesn't yet motivate the Phase B-proposed "smarter gate" designs
  (trend-extrapolated threshold, drift-regime detection) as *necessary* --
  gate-free slow already solves the drift failure without one, at the
  contamination levels tested. Those designs remain worth trying only if a
  future test pushes contamination past what gate-free's fixed pull-back can
  absorb, or if the FM regression on the plain benchmark turns out to matter
  for the target use case and a smarter gate could recover it without
  reintroducing the freeze failure.

The most direct next step, if it's worth the effort: instrument *why*
gatefree_slow's FM is worse on the plain benchmark (mirroring the Phase B
methodology -- track how far FAST's mean moves stage-to-stage under
gatefree_slow vs. gated on the main benchmark's task sequence, not just the
drift streams) before deciding whether that cost is fixable or inherent to
removing the gate.
