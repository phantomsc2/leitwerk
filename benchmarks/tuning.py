"""Joint successive-halving search, with a separately frozen held-out comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from leitwerk import XNESLearningRates
from scipy.stats import qmc

from .problems import bbob, seed_for
from .runner import PRESETS, Study, cases, create_run, run_trial, write_json


@dataclass(frozen=True)
class TuningPlan:
    """Observation ceilings include every fresh run, including eliminated candidates."""

    counts: tuple[int, ...] = (24, 8, 3)
    stages: tuple[Study, ...] = (
        Study((1, 3, 5, 7, 8, 10, 12, 15, 16, 19, 21, 24), (2, 10, 30), (1,), budget=400),
        Study(tuple(range(1, 25)), (2, 10, 30), (2,), budget=1000),
        Study(tuple(range(1, 25)), (2, 10, 30), (3,), budget=2000),
    )
    validation: Study = Study(tuple(range(1, 25)), (2, 5, 10, 20, 30), (4, 5), budget=2000)
    seed: int = 20260919

    def budget(self) -> list[dict[str, int | str]]:
        panels = [(f"search-{i}", n, s) for i, (n, s) in enumerate(zip(self.counts, self.stages, strict=True))]
        panels.append(("held-out", 4, self.validation))
        return [
            {"phase": name, "variants": n, "runs": n * s.runs // 2, "observations": n * s.runs // 2 * s.budget}
            for name, n, s in panels
        ]


def candidates(count: int, seed: int) -> dict[str, XNESLearningRates]:
    """Space-filling joint proposals; defaults plus a fixed-rate ablation always enter."""
    defaults = XNESLearningRates()
    proposals = [defaults, replace(defaults, adaptive=False)]
    bounds = {
        "eta_mean": (0.3, 1.0),
        "eta_scale_global": (0.15, 1.0),
        "eta_scale_shape": (0.08, 1.5),
        "signal_threshold": (1.2, 4.0),
        "evidence_half_life": (8.0, 64.0),
        "rate_half_life": (4.0, 32.0),
        "scale_recovery": (1.0, 8.0),
        "max_kl": (0.05, 2.0),
    }
    for point in qmc.LatinHypercube(len(bounds), seed=seed).random(max(count - 2, 0)):
        values: dict[str, Any] = {
            key: float(lo * (hi / lo) ** x) for (key, (lo, hi)), x in zip(bounds.items(), point, strict=True)
        }
        proposals.append(replace(defaults, **values))
    return {f"candidate-{i:02d}": rates for i, rates in enumerate(proposals[:count])}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Equal-weight early/middle/final log improvement of the clean recommendation.

    Numerical failures get the worst score, even if the last valid mean was good.
    Unused observations are not recycled. Early stops carry their last mean forward.
    """
    trajectory = result["trajectory"]
    checkpoints = sorted({max(1, result["budget"] // 10), max(1, result["budget"] // 3), result["budget"]})
    gaps = []
    for checkpoint in checkpoints:
        available = [row for row in trajectory if row["evaluations"] <= checkpoint]
        gaps.append(available[-1]["relative_gap"] if available else 1.0)
    failed = result["kind"] == "numerical_failure"
    score = -8.0 if failed else float(np.mean(-np.log10(np.clip(gaps, 1e-8, 1e8))))
    diagnostics = trajectory[-1] if trajectory else {}
    return {
        "score": score,
        "failed": failed,
        "relative_gap": gaps[-1],
        "checkpoints": checkpoints,
        "relative_gaps": gaps,
        "evaluations": result["evaluations"],
        "seconds": result["seconds"],
        "kind": result["kind"],
        "reason": result["reason"],
        "diagnostics": diagnostics,
        "clipping": {
            block: float(np.mean([row[f"trust_{block}"] < 1 - 1e-10 for row in trajectory[1:]]))
            for block in ("mean", "scale", "shape")
            if len(trajectory) > 1 and f"trust_{block}" in diagnostics
        },
    }


def ranking(rows: list[dict[str, Any]]) -> list[str]:
    names = sorted({row["variant"] for row in rows})
    return sorted(names, key=lambda name: -float(np.mean([r["score"] for r in rows if r["variant"] == name])))


def run_panel(
    study: Study, variants: dict[str, XNESLearningRates | None], directory: Path, phase: str
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    rows = []
    for case in tqdm(list(cases(study)), desc=phase):
        base_seed = seed_for(study.master_seed, case["function"], case["dimension"], case["instance"], case["repeat"])
        for name, rates in variants.items():
            algorithm = "cma" if rates is None else "xnes"
            result = run_trial(
                bbob(case["function"], case["dimension"], case["instance"]),
                algorithm,
                study.budget,
                seed_for(base_seed, int(algorithm == "cma")),
                seed_for(base_seed, 2),
                case["noise"],
                learning_rates=rates,
            )
            row = {**case, "variant": name, **summarize(result)}
            rows.append(row)
            # One compact record per run; retain complete trajectories/state only on failure.
            tag = f"{phase}-{name}-f{case['function']}-d{case['dimension']}-i{case['instance']}"
            tag += f"-r{case['repeat']}-{case['noise']}"
            write_json(directory / f"{tag}.json", row)
            if row["failed"]:
                write_json(directory / f"failure-{tag}.json", result)
    return rows


def run_tuning(plan: TuningPlan | None = None) -> Path:
    """Select only on training panels; freeze the winner before touching validation.

    Outputs use the benchmark's ignored runs directory. No core defaults are rewritten.
    The default ceiling is 13,094,400 observations versus slim's 13,440,000.
    Clean diagnostic assessments are additional calls, as in the benchmark.
    """
    plan = plan or TuningPlan()
    proposals = candidates(plan.counts[0], plan.seed)
    directory = create_run(plan.validation)
    write_json(
        directory / "tuning-plan.json",
        {"plan": asdict(plan), "budget": plan.budget(), "candidates": {k: asdict(v) for k, v in proposals.items()}},
    )
    print(f"Tuning artifacts: {directory}", flush=True)
    active = list(proposals)
    for i, (count, study) in enumerate(zip(plan.counts, plan.stages, strict=True)):
        rows = run_panel(study, {name: proposals[name] for name in active[:count]}, directory, f"search-{i}")
        if all(row["failed"] for row in rows):
            raise RuntimeError(f"All trials failed; inspect {directory} before spending more budget")
        active = ranking(rows)
        write_json(directory / f"ranking-{i}.json", active)
        print(f"Stage {i}: {active}", flush=True)
    winner = proposals[active[0]]
    write_json(directory / "selected.json", {"candidate": active[0], "learning_rates": asdict(winner)})
    variants = {
        "selected": winner,
        "default": XNESLearningRates(),
        "fixed": replace(winner, adaptive=False),
        "cma": None,
    }
    write_json(
        directory / "validation-variants.json", {k: asdict(v) if v is not None else None for k, v in variants.items()}
    )
    run_panel(plan.validation, variants, directory, "held-out")
    write_json(
        directory / "complete.json", {"observations_ceiling": sum(int(r["observations"]) for r in plan.budget())}
    )
    return directory


def load_panel(directory: Path, phase: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob(f"{phase}-*.json"))]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Small integration run, not evidence of superiority")
    args = parser.parse_args()
    plan = TuningPlan()
    if args.smoke:
        panel = Study((1, 8), (2,), (1,), budget=80, noises=("clean", "additive"))
        plan = TuningPlan((3, 2), (panel, replace(panel, instances=(2,))), replace(panel, instances=(4,)))
    print(plan.budget())
    assert sum(int(r["observations"]) for r in plan.budget()) <= PRESETS["slim"].runs * PRESETS["slim"].budget
    run_tuning(plan)
