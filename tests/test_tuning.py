from __future__ import annotations

import numpy as np
from benchmarks.runner import PRESETS
from benchmarks.tuning import TuningPlan, candidates, ranking, summarize


def test_tuning_budget_and_holdout() -> None:
    plan = TuningPlan()
    ceiling = sum(int(row["observations"]) for row in plan.budget())
    assert ceiling == 13_094_400
    assert ceiling < PRESETS["slim"].runs * PRESETS["slim"].budget
    used: set[int] = set()
    for panel in (*plan.stages, plan.validation):
        assert used.isdisjoint(panel.instances)
        used.update(panel.instances)


def test_joint_proposals_are_reproducible_and_include_ablation() -> None:
    proposed = candidates(24, 12)
    assert proposed == candidates(24, 12)
    assert proposed != candidates(24, 13)
    assert len(proposed) == 24
    assert proposed["candidate-00"].adaptive
    assert not proposed["candidate-01"].adaptive


def test_score_carries_early_stop_and_penalizes_failure() -> None:
    result = {
        "budget": 300,
        "trajectory": [
            {"evaluations": 0, "relative_gap": 1.0},
            {"evaluations": 50, "relative_gap": 0.001},
        ],
        "kind": "convergence",
        "evaluations": 50,
        "seconds": 0.1,
        "reason": "small update",
    }
    summary = summarize(result)
    assert summary["score"] == 2
    assert summary["relative_gaps"] == [1, 0.001, 0.001]
    failure = summarize({**result, "kind": "numerical_failure"})
    assert failure["score"] == -8
    assert ranking([{"variant": "failure", **failure}, {"variant": "success", **summary}]) == ["success", "failure"]
    assert np.isfinite(failure["score"])
