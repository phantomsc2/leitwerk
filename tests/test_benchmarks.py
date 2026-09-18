from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("cma")
pytest.importorskip("cocoex")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from benchmarks.problems import ObservationNoise, Problem, bbob, seed_for  # noqa: E402
from benchmarks.reporting import checkpoint_table, paired_summary, plot_targets, target_table  # noqa: E402
from benchmarks.runner import XNESAdapter, load_runs, run_trial, write_json  # noqa: E402


def problem() -> Problem:
    return Problem("shifted sphere", lambda x: float(np.sum((x - 2) ** 2)), np.zeros(3))


def annotated(result: dict[str, Any], instance: int = 1) -> dict[str, Any]:
    return {**result, "function": 1, "instance": instance, "repeat": 0}


@pytest.mark.parametrize("algorithm", ["xnes", "cma"])
def test_budget_reproducibility_and_clean_assessment_isolation(algorithm: str) -> None:
    calls = []

    def observed(x: np.ndarray) -> float:
        calls.append(x.copy())
        return float(np.sum((x - 2) ** 2))

    a = run_trial(problem(), algorithm, 103, 17, 29, observation=observed)
    b = run_trial(problem(), algorithm, 103, 17, 29, observation=observed)
    assert a["kind"] == b["kind"] == "budget"
    assert a["trajectory"] == b["trajectory"]
    assert a["evaluations"] <= 103
    assert 0 <= a["unused_budget"] < a["population"]
    assert a["evaluations"] % a["population"] == 0
    assert len(calls) == a["evaluations"] + b["evaluations"]
    assert a["population"] == (8 if algorithm == "xnes" else 7)


def test_noise_pairing_is_independent_of_population_and_assessment() -> None:
    a = ObservationNoise("outliers", 5.0, 100, seed_for(1234, 1, 2))
    b = ObservationNoise("outliers", 5.0, 100, seed_for(1234, 1, 2))
    chunks = [range(start, min(start + 7, 100)) for start in range(0, 100, 7)]
    left = [a(3.0, 3.0, i) for i in range(100)]
    right = [b(3.0, 3.0, i) for chunk in chunks for i in chunk]
    assert left == right


class ControlledAdapter(XNESAdapter):
    def __init__(self, stop: str) -> None:
        super().__init__(np.zeros(3), 1.0, 1, population=4)
        self.stop = stop
        self.updates = 0

    def tell(self, candidates: np.ndarray, values: np.ndarray) -> tuple[str, str]:
        self.updates += 1
        warnings.warn("deliberate diagnostic warning", RuntimeWarning, stacklevel=2)
        if self.stop == "exception":
            raise ArithmeticError("deliberate failure")
        if self.stop == "interrupt":
            raise KeyboardInterrupt
        self.optimizer.mean[:] = 1.0
        return self.stop, "controlled stop"


@pytest.mark.parametrize("kind", ["convergence", "numerical_failure", "exception"])
def test_stops_warnings_and_failure_artifacts(kind: str, tmp_path: Path) -> None:
    adapter = ControlledAdapter(kind)
    result = annotated(run_trial(problem(), "xnes", 100, 1, 2, adapter_factory=lambda: adapter))
    assert adapter.updates == 1
    assert result["evaluations"] == 4
    assert result["kind"] == ("numerical_failure" if kind == "exception" else kind)
    assert result["warnings"][0]["message"] == "deliberate diagnostic warning"
    if kind != "convergence":
        assert np.array_equal(result["failure"]["pre_update"]["mean"], np.zeros(3))
        assert len(result["failure"]["scores"]) == 4
        assert result["failure"]["standardized_samples"].shape == (3, 4)
    write_json(tmp_path / "f01-test.json", result)
    loaded = load_runs(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["trajectory"] == result["trajectory"]
    frame = checkpoint_table(loaded, [2, 4, 100])
    assert frame.iloc[0].mean_evaluations == 0
    assert not frame.iloc[0].failed
    assert bool(frame.iloc[-1].failed) == (kind != "convergence")
    if kind == "convergence":
        assert frame.iloc[-1].relative_gap == pytest.approx(0.25)
        assert frame.iloc[-1].kind == "convergence"


def test_user_interrupt_propagates() -> None:
    with pytest.raises(KeyboardInterrupt):
        run_trial(problem(), "xnes", 100, 1, 2, adapter_factory=lambda: ControlledAdapter("interrupt"))


def test_paired_summary_counts_failures_and_bootstraps_instances() -> None:
    results = []
    for instance in (1, 2, 3):
        for algorithm, kind in (("xnes", "numerical_failure"), ("cma", "convergence")):
            results.append(
                annotated(
                    run_trial(problem(), algorithm, 100, 1, 2, adapter_factory=partial(ControlledAdapter, kind)),
                    instance,
                )
            )
    summary = paired_summary(checkpoint_table(results, [2, 100]), bootstrap_samples=10)
    early, late = summary.iloc[0], summary.iloc[1]
    assert early.finite_pairs == 3
    assert early.ci_low == early.ci_high == 0.0
    assert late.xnes_failures == late.cma_wins == late.excluded_pairs == 3
    assert late.finite_pairs == 0
    assert np.isnan(late.median_log10_gap_ratio)
    targets = target_table(results)
    assert len(targets) == 24  # Unreached targets/failures are retained, not filtered out.


def test_bbob_30d_and_nonfinite_objective_failure() -> None:
    p = bbob(1, 30, 1)
    assert p.x0.shape == (30,)
    assert p.gap(p.x0) > 0
    result = run_trial(problem(), "xnes", 100, 1, 2, observation=lambda x: np.nan)
    assert result["kind"] == "numerical_failure"
    assert result["evaluations"] == 1
    assert "Nonfinite objective" in result["reason"]


def test_target_plot_keeps_zero_hits_visible_through_budget() -> None:
    import matplotlib.pyplot as plt

    results = [annotated(run_trial(problem(), algorithm, 1, 1, 2)) for algorithm in ("xnes", "cma")]
    for result in results:
        result["budget"] = 100
    figure = plot_targets(results)
    for line in figure.axes[0].lines:
        assert line.get_xdata()[-1] == 100
        assert np.all(line.get_ydata() == 0)
    plt.close(figure)
