# Benchmark review: 2026-09-19

Saved study: `notebooks/runs/20260918T155841-0c58a2ba` (ignored local artifacts).
24 BBOB functions, dimensions 2/5/10/20/30, 15 instances, two repeats,
four observation-noise settings, and 10,000 objective calls per optimizer:
28,800 optimization runs plus 270 separate xNES stability runs. Master seed: 1234.
The manifest records dependency versions, source hashes, and base revision
`ce52f91540635c57398e04866a19ceeb1f736234`. Python source hashes matched the
run before the subsequent reporting-only ECDF plot correction.

## Optimization quality

The primary recommendation is the distribution mean, assessed without noise.
At the final budget:

| Observation noise | xNES wins / all 3,600 pairs | CMA wins | Median xNES/CMA gap ratio |
| --- | ---: | ---: | ---: |
| Clean | 37.2% | 37.4% | 1.00 |
| Additive | 65.0% | 31.0% | 0.79 |
| Heteroscedastic | 64.5% | 30.9% | 0.76 |
| Outliers | 65.4% | 30.9% | 0.77 |

Wins compare paired log normalized gaps, clipped below at 1e-16. Remaining
pairs are ties or unavailable after numerical failure. Median ratios use only
finite pairs; failures are not quietly counted as successful recommendations.
There are 14,325 finite final pairs out of 14,400.

xNES is broadly competitive on clean problems and has a consistent advantage
under these noise models: typical paired noisy gaps are about 21–24% smaller.
Across all settings it wins 58.0% of pairs, with a median gap ratio of 0.86.
This is not universal superiority: clean results are approximately tied, and
functions 23 and 24 favor CMA in their pooled median log ratios.

The advantage does not consistently grow with budget: pooled median log10 gap
ratios at 100/300/1,000/3,000/10,000 calls are
-0.064/-0.083/-0.057/-0.055/-0.065. Better final noisy solutions are supported;
an increasingly strong long-term advantage is not established by this run.

Instance-bootstrap intervals at 10,000 calls favor xNES/CMA in respectively
66/6 additive, 62/4 heteroscedastic, 63/4 outlier, and 24/28 clean
function–dimension cells (120 cells per setting; 1,000 bootstrap resamples).
These are exploratory per-cell intervals, not multiplicity-corrected claims.

## Stopping and stability

| Recorded outcome | xNES | CMA |
| --- | ---: | ---: |
| Budget exhausted | 12,634 | 5,442 |
| Nonfailure optimizer stop | 1,765 | 8,884 |
| Numerical failure | 1 | 74 |

The xNES failure was `SCALE_COND_MAX` on clean f5, dimension 2, instance 10,
repeat 1. All 74 CMA failures were `Nonfinite clean assessment`: the harness
could not obtain a finite objective at the recommendation. They should not be
described as 74 covariance failures. CMA warnings are dominated by redundant
seed notices from supplying a separately seeded random generator.

Most CMA nonfailure stops begin with `tolstagnation` (5,864), followed by
`tolfun` (1,784) and `tolflatfitness` (506). The harness label `convergence`
means a nonfailure stop, not proof of reaching the optimum. Finite stopped
recommendations carry forward to later checkpoints.

The separate constant/random-score stress experiment produced 42 condition-limit
failures out of 270 runs, all in dimension 2: 30 with initial axis ratio 1e9
and 12 with ratio 1e6. No isotropically initialized stress run failed, but lack
of failure does not mean lack of drift. With constant scores and isotropic
initial scale, median final mean displacement was about 26,536 in dimension 2,
629 in dimension 10, and 215 in dimension 30. Tie handling and uninformative-score
behavior therefore remain open algorithmic issues, separate from learning rates.
No failed run was restarted or repaired.

## Scope of the conclusion

- This compares bare xNES with default pycma, including their native population
  sizes and stopping policies. CMA has no added noise handler or restart strategy.
  Early stopping is part of the observed comparison, not isolated from update quality.
- These are controlled noise overlays on BBOB, not the official noisy COCO suite
  or evidence for every application-specific noise distribution.
- Clean mean assessments and best-clean-sample diagnostics are oracle measurements;
  they do not enter optimizer updates. Only optimizer observations consume the budget.
- Full raw artifacts remain local and ignored. Preserve the saved directory for
  reproducibility; this summary alone cannot recreate every trajectory.

This is a useful fixed baseline for subsequent learning-rate experiments. Keep
the benchmark settings unchanged and distinguish tuning cases from held-out
evaluation before interpreting a tuned improvement.
