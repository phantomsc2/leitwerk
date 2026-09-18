"""Budgeted, non-restarting optimizer runs and incremental experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import traceback
import warnings
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
from leitwerk import XNES, XNESLearningRates
from leitwerk.xnes import _default_sample_count

from .problems import NOISE_MODELS, ObservationNoise, Problem, bbob, random_rotation, seed_for

CHECKPOINTS = (100, 300, 1000, 3000, 10000)
TARGETS = (1e-1, 1e-2, 1e-3, 1e-5)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Study:
    functions: tuple[int, ...] = (1, 7, 9, 10, 12, 15, 21, 24)
    dimensions: tuple[int, ...] = (2, 10, 30)
    instances: tuple[int, ...] = (1, 2, 3)
    repeats: int = 1
    budget: int = 1000
    noises: tuple[str, ...] = NOISE_MODELS
    master_seed: int = 1234

    @property
    def runs(self) -> int:
        return 2 * len(self.functions) * len(self.dimensions) * len(self.instances) * self.repeats * len(self.noises)


PRESETS = {
    "preview": Study(),
    "slim": Study(tuple(range(1, 25)), (2, 5, 10, 20, 30), tuple(range(1, 8)), 1, 2000),
    "full": Study(tuple(range(1, 25)), (2, 5, 10, 20, 30), tuple(range(1, 16)), 2, 10000),
}


class Adapter(Protocol):
    population: int

    @property
    def mean(self) -> np.ndarray: ...
    def ask(self) -> np.ndarray: ...
    def tell(self, candidates: np.ndarray, values: np.ndarray) -> tuple[str, str]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def diagnostics(self) -> dict[str, float]: ...


def scale_diagnostics(scale: np.ndarray) -> dict[str, float]:
    if not np.all(np.isfinite(scale)):
        raise FloatingPointError("Nonfinite distribution scale")
    axes = np.linalg.svd(scale, compute_uv=False)
    return {"axis_ratio": float(axes[0] / axes[-1]), "scale_min": float(axes[-1]), "scale_max": float(axes[0])}


class XNESAdapter:
    def __init__(
        self,
        x0: np.ndarray,
        scale: float | np.ndarray,
        seed: int,
        population: int | None = None,
        learning_rates: XNESLearningRates | None = None,
    ) -> None:
        self.optimizer = XNES(x0, scale)
        self.rng = np.random.default_rng(seed)
        self.population = _default_sample_count(population, len(x0))
        self.learning_rates = learning_rates
        self.samples = np.empty((len(x0), 0))

    @property
    def mean(self) -> np.ndarray:
        return self.optimizer.mean

    def ask(self) -> np.ndarray:
        self.samples = self.optimizer.sample(self.population, self.rng)
        return self.optimizer.transform(self.samples).T

    def tell(self, candidates: np.ndarray, values: np.ndarray) -> tuple[str, str]:
        # Exact ties retain the current implementation's stable index order; do not repair the algorithm here.
        status = self.optimizer.update(self.samples, np.argsort(values, kind="stable").tolist(), self.learning_rates)
        kind = "running" if status.is_ok else "convergence" if status.is_completion else "numerical_failure"
        return kind, status.name

    def snapshot(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "scale": self.optimizer.scale.copy(),
            "rng_state": self.rng.bit_generator.state,
            "learning_rates": asdict(self.learning_rates or XNESLearningRates()),
        }

    def diagnostics(self) -> dict[str, float]:
        return {**scale_diagnostics(self.optimizer.scale), "scale_global": self.optimizer.scale_global}


class CMAAdapter:
    def __init__(self, x0: np.ndarray, scale: float, seed: int, population: int | None = None) -> None:
        import cma

        self.rng = np.random.RandomState(seed)
        options = {"seed": seed or 1, "randn": self.rng.randn, "verb_disp": 0, "verb_log": 0}
        if population is not None:
            options["popsize"] = population
        self.optimizer = cma.CMAEvolutionStrategy(x0, scale, options)
        self.population = int(self.optimizer.popsize)

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.optimizer.mean)

    def ask(self) -> np.ndarray:
        return np.asarray(self.optimizer.ask())

    def tell(self, candidates: np.ndarray, values: np.ndarray) -> tuple[str, str]:
        self.optimizer.tell(candidates, values)
        reasons = self.optimizer.stop()
        if not reasons:
            return "running", "OK"
        numerical = "tolconditioncov" in reasons
        return ("numerical_failure" if numerical else "convergence"), repr(dict(reasons))

    def snapshot(self) -> dict[str, Any]:
        es = self.optimizer
        return {
            "mean": self.mean.copy(),
            "covariance": np.array(es.sm.C, copy=True),
            "sigma": float(es.sigma),
            "sigma_vec": np.array(es.sigma_vec.scaling, copy=True),
            "pc": es.pc.copy(),
            "rng_state": self.rng.get_state(),
        }

    def diagnostics(self) -> dict[str, float]:
        es = self.optimizer
        scaling = np.broadcast_to(es.sigma_vec.scaling, self.mean.shape)
        covariance = es.sigma**2 * np.asarray(es.sm.C) * scaling[:, None] * scaling[None, :]
        axes = np.sqrt(np.linalg.eigvalsh(covariance))
        if not np.all(np.isfinite(axes)) or axes[0] <= 0:
            raise FloatingPointError("Nonpositive or nonfinite CMA covariance")
        return {
            "axis_ratio": float(axes[-1] / axes[0]),
            "scale_min": float(axes[0]),
            "scale_max": float(axes[-1]),
            "scale_global": float(np.exp(np.mean(np.log(axes)))),
        }


def run_trial(
    problem: Problem,
    algorithm: str,
    budget: int,
    optimizer_seed: int,
    noise_seed: int,
    noise_model: str = "clean",
    scale: float | np.ndarray = 1.0,
    population: int | None = None,
    learning_rates: XNESLearningRates | None = None,
    observation: Callable[[np.ndarray], float] | None = None,
    adapter_factory: Callable[[], Adapter] | None = None,
) -> dict[str, Any]:
    """Assess the latest fully updated mean; never restart or tell a partial population.

    Clean assessment and optimizer observations have separate call paths. Exceptions become
    failed run records; KeyboardInterrupt/SystemExit propagate. A failed update retains the
    last finite recommendation for diagnosis, but is not a successful recommendation.
    """
    start = perf_counter()
    evaluations = 0
    best_gap: float | None = None
    trajectory: list[dict[str, Any]] = []
    best_trace: list[dict[str, Any]] = []
    warning_events: list[dict[str, Any]] = []
    kind, reason = "running", "OK"
    failure: dict[str, Any] | None = None
    pre_update: dict[str, Any] = {}
    candidates = np.empty((0, len(problem.x0)))
    values: list[float] = []
    adapter: Adapter | None = None
    reference = 1.0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            reference = max(problem.gap(problem.x0), 1e-12)
            noise = ObservationNoise(noise_model, reference, budget, noise_seed)
            if adapter_factory is not None:
                adapter = adapter_factory()
            elif algorithm == "xnes":
                adapter = XNESAdapter(problem.x0, scale, optimizer_seed, population, learning_rates)
            elif algorithm == "cma":
                adapter = CMAAdapter(problem.x0, float(scale), optimizer_seed, population)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            def record() -> None:
                if not np.all(np.isfinite(adapter.mean)):
                    raise FloatingPointError("Nonfinite distribution mean")
                gap = problem.gap(adapter.mean)
                if not np.isfinite(gap):
                    raise FloatingPointError("Nonfinite clean assessment")
                diagnostics = adapter.diagnostics()
                if not all(np.isfinite(value) for value in diagnostics.values()):
                    raise FloatingPointError("Nonfinite distribution diagnostic")
                trajectory.append(
                    {
                        "evaluations": evaluations,
                        "mean_gap": gap,
                        "relative_gap": gap / reference,
                        "best_gap": best_gap,
                        "mean_norm": float(np.linalg.norm(adapter.mean)),
                        "mean_drift": float(np.linalg.norm(adapter.mean - problem.x0)),
                        **diagnostics,
                    }
                )

            record()
            while budget - evaluations >= adapter.population:
                pre_update = adapter.snapshot()
                candidates = np.empty((0, len(problem.x0)))
                values = []
                candidates = adapter.ask()
                if not np.all(np.isfinite(candidates)):
                    raise FloatingPointError("Nonfinite candidate")
                for candidate in candidates:
                    index = evaluations
                    evaluations += 1
                    clean_value = float(problem.clean(candidate))
                    gap = max(clean_value - problem.optimum, 0.0)
                    value = observation(candidate) if observation else noise(clean_value, gap, index)
                    values.append(float(value))
                    if not np.isfinite(clean_value) or not np.isfinite(value):
                        raise FloatingPointError("Nonfinite objective value")
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_trace.append({"evaluations": evaluations, "gap": gap})
                kind, reason = adapter.tell(candidates, np.asarray(values))
                if kind == "numerical_failure":
                    failure = {"pre_update": pre_update, "candidates": candidates, "scores": values}
                    break
                record()
                warning_events.extend(
                    {"evaluations": evaluations, "category": w.category.__name__, "message": str(w.message)}
                    for w in captured
                )
                captured.clear()
                if kind != "running":
                    break
            if kind == "running":
                kind, reason = "budget", "Insufficient budget for another complete population"
        except Exception as exc:
            previous_reason = reason
            kind, reason = "numerical_failure", f"{type(exc).__name__}: {exc}"
            failure = {
                "pre_update": pre_update,
                "candidates": candidates,
                "scores": values,
                "traceback": traceback.format_exc(),
                "update_status": previous_reason,
            }
        warning_events.extend(
            {"evaluations": evaluations, "category": w.category.__name__, "message": str(w.message)} for w in captured
        )
    if failure is not None and isinstance(adapter, XNESAdapter):
        failure["standardized_samples"] = adapter.samples.copy()
        failure["post_update"] = adapter.snapshot()
    return {
        "problem": problem.name,
        "dimension": len(problem.x0),
        "algorithm": algorithm,
        "noise": noise_model,
        "optimizer_seed": optimizer_seed,
        "noise_seed": noise_seed,
        "budget": budget,
        "evaluations": evaluations,
        "unused_budget": budget - evaluations,
        "population": adapter.population if adapter else None,
        "reference_gap": reference,
        "kind": kind,
        "reason": reason,
        "seconds": perf_counter() - start,
        "trajectory": trajectory,
        "best_trace": best_trace,
        "warnings": warning_events,
        "failure": failure,
    }


def cases(config: Study) -> Iterator[dict[str, Any]]:
    for function, dimension, instance, repeat, noise in product(
        config.functions, config.dimensions, config.instances, range(config.repeats), config.noises
    ):
        yield {"function": function, "dimension": dimension, "instance": instance, "repeat": repeat, "noise": noise}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def provenance() -> dict[str, Any]:
    """Record source content independently of commit identity; no subprocesses in notebook kernels."""
    files = sorted(
        [
            *ROOT.glob("leitwerk/**/*.py"),
            *ROOT.glob("benchmarks/**/*.py"),
            *ROOT.glob("notebooks/*.ipynb"),
            ROOT / "pyproject.toml",
        ]
    )
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    git = ROOT / ".git"
    revision: str | None = None
    if git.is_dir():
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            if (git / ref).exists():
                revision = (git / ref).read_text().strip()
            elif (git / "packed-refs").exists():
                revision = next(
                    (
                        line.split()[0]
                        for line in (git / "packed-refs").read_text().splitlines()
                        if line.endswith(" " + ref)
                    ),
                    None,
                )
        else:
            revision = head
    return {
        "revision": revision,
        "source_sha256": hashes,
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("numpy", "scipy", "cma", "coco-experiment", "pandas", "matplotlib")
        },
        "learning_rates": asdict(XNESLearningRates()),
    }


def create_run(config: Study, root: Path = ROOT / "notebooks" / "runs") -> Path:
    directory = root / f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    write_json(directory / "manifest.json", {"config": asdict(config), **provenance()})
    return directory


def run_study(config: Study, directory: Path) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    results = []
    for case in tqdm(list(cases(config)), desc="Paired cases"):
        function, dimension, instance, repeat = (case[k] for k in ("function", "dimension", "instance", "repeat"))
        base_seed = seed_for(config.master_seed, function, dimension, instance, repeat)
        for algorithm_index, algorithm in enumerate(("xnes", "cma")):
            result = run_trial(
                bbob(function, dimension, instance),
                algorithm,
                config.budget,
                seed_for(base_seed, algorithm_index),
                seed_for(base_seed, 2),
                case["noise"],
            )
            result.update(case)
            filename = f"f{function:02d}-d{dimension}-i{instance}-r{repeat}-{case['noise']}-{algorithm}.json"
            write_json(directory / filename, result)
            results.append(result)
    return results


def load_runs(directory: Path, pattern: str = "f*.json") -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob(pattern))]


def run_stability(config: Study, directory: Path) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    results = []
    settings = list(product((2, 10, 30), (1.0, 1e6, 1e9), ("constant", "random"), range(len(config.instances))))
    for dimension, ratio, model, repeat in tqdm(settings, desc="xNES stability"):
        seed = seed_for(config.master_seed, 9000, dimension, int(np.log10(ratio)), repeat)
        rotation = random_rotation(dimension, np.random.default_rng(seed))
        axes = np.geomspace(ratio**-0.5, ratio**0.5, dimension)
        scale = (rotation * axes) @ rotation.T
        problem = Problem("stability", lambda x: float(x @ x), np.zeros(dimension))
        result = run_trial(problem, "xnes", config.budget, seed_for(seed, 0), seed_for(seed, 1), model, scale)
        result.update({"initial_axis_ratio": ratio, "repeat": repeat})
        write_json(directory / f"stability-d{dimension}-a{ratio:g}-{model}-r{repeat}.json", result)
        results.append(result)
    return results
