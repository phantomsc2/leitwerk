# Notebook experiments

These helpers are research tooling, not part of the installed `leitwerk` API.
Install the `benchmark` extra (and `dev` for checks), then run notebooks from the
repository root or the `notebooks` directory.

- `benchmark.ipynb`: paired bare xNES/default pycma comparisons on BBOB landscapes
  with controlled observation noise, plus separate xNES stability experiments.
  The preview is the default; the full preset contains 28,800 optimization runs
  and 270 stability runs. Set `LOAD_RUN` to inspect saved results without rerunning.
- `workbench.ipynb`: editable, quick comparisons on five translated objectives.
  Equal population sizes and the original half-normal noise are intentional here.
- `hyperparameter_tuning.ipynb`: exploratory learning-rate sweeps. Its case bands
  are not confidence intervals; its results are not held-out benchmark evidence.

The main metric is clean objective gap at the current mean, normalized by the
initial gap. Clean assessments never influence optimizer updates or stopping.
Only complete populations are submitted; unused evaluations are reported.
Every terminal status stops the run without recovery. Numerical failures remain
in comparison denominators; finite-quality summaries explicitly exclude them.
Finite early-stop recommendations carry forward. First target attainment does
not imply sustained convergence. The best evaluated clean gap is an oracle
diagnostic, not a deployable recommendation under noise.

Each run is saved immediately to a unique ignored `notebooks/runs` directory.
The manifest records configuration, dependency versions, revision, and source
hashes; hashes identify uncommitted source content independently of the revision.
Per-run JSON includes seeds, generation diagnostics, warnings, and failure state.
No result directory is deleted or overwritten by a new study.

`make check` includes these helpers and their tests. Benchmark-specific tests
are skipped if the optional benchmark dependencies are unavailable.
