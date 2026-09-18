"""Recommendation quality, paired comparisons, and uncensored failure reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runner import CHECKPOINTS, TARGETS


def run_table(results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {k: v for k, v in result.items() if k not in {"trajectory", "warnings", "failure", "best_trace"}}
        row["warning_count"] = len(result["warnings"])
        row["warning_messages"] = " | ".join(dict.fromkeys(w["message"] for w in result["warnings"]))
        if result["trajectory"]:
            row.update({f"final_{k}": v for k, v in result["trajectory"][-1].items() if k != "evaluations"})
            for metric in ("axis_ratio", "mean_drift", "scale_max"):
                row[f"max_{metric}"] = max(p[metric] for p in result["trajectory"])
            row["target_hit"] = any(p["relative_gap"] <= TARGETS[-1] for p in result["trajectory"])
        else:
            row["target_hit"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_table(results: Sequence[dict[str, Any]], checkpoints: Sequence[int] = CHECKPOINTS) -> pd.DataFrame:
    """Carry stopped recommendations, not failed ones; never look ahead to a later generation."""
    rows = []
    for result in results:
        for budget in checkpoints:
            if budget > result["budget"]:
                continue
            points = [p for p in result["trajectory"] if p["evaluations"] <= budget]
            failed = result["kind"] == "numerical_failure" and result["evaluations"] <= budget
            point = points[-1] if points else None
            improvements = [p for p in result.get("best_trace", []) if p["evaluations"] <= budget]
            rows.append(
                {
                    **{k: result[k] for k in ("function", "dimension", "instance", "repeat", "noise", "algorithm")},
                    "budget": budget,
                    "failed": failed,
                    "stopped": result["evaluations"] <= budget,
                    "kind": result["kind"] if result["evaluations"] <= budget else "running",
                    "relative_gap": point["relative_gap"] if point and not failed else np.nan,
                    "last_finite_relative_gap": point["relative_gap"] if point else np.nan,
                    "mean_evaluations": point["evaluations"] if point else 0,
                    "best_relative_gap": improvements[-1]["gap"] / result["reference_gap"] if improvements else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_summary(checkpoints: pd.DataFrame, bootstrap_samples: int = 1000, seed: int = 2026) -> pd.DataFrame:
    """Positive log-gap difference favors CMA; numerical failures lose to finite runs.

    Within each function/dimension/noise/budget group, resample whole instances (all repeats)
    for intervals on the median paired difference. Missing/nonfinite pairs are counted explicitly.
    """
    keys = ["function", "dimension", "noise", "budget", "instance", "repeat"]
    pairs = checkpoints.pivot(index=keys, columns="algorithm", values=["relative_gap", "failed"])
    pairs = pairs.reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    groups = ["function", "dimension", "noise", "budget"]
    for identity, group in pairs.groupby(groups, sort=True):
        x = group[("relative_gap", "xnes")].to_numpy(dtype=float)
        c = group[("relative_gap", "cma")].to_numpy(dtype=float)
        fx = group[("failed", "xnes")].to_numpy(dtype=bool)
        fc = group[("failed", "cma")].to_numpy(dtype=bool)
        valid = np.isfinite(x) & np.isfinite(c) & ~fx & ~fc
        delta = np.log10(np.maximum(x[valid], 1e-16)) - np.log10(np.maximum(c[valid], 1e-16))
        labels = group[("instance", "")].to_numpy()[valid]
        instances = np.unique(labels)
        interval = (np.nan, np.nan)
        if len(instances) >= 2:
            clusters = [delta[labels == instance] for instance in instances]
            medians = [
                np.median(np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]))
                for _ in range(bootstrap_samples)
            ]
            interval = tuple(np.quantile(medians, [0.025, 0.975]))
        wins = int(np.sum(~fx & fc)) + int(np.sum(delta < -1e-12))
        losses = int(np.sum(fx & ~fc)) + int(np.sum(delta > 1e-12))
        rows.append(
            {
                **dict(zip(groups, identity, strict=True)),
                "pairs": len(group),
                "finite_pairs": int(valid.sum()),
                "excluded_pairs": int((~valid).sum()),
                "xnes_failures": int(fx.sum()),
                "cma_failures": int(fc.sum()),
                "xnes_wins": wins,
                "cma_wins": losses,
                "ties_or_both_failed": len(group) - wins - losses,
                "median_log10_gap_ratio": float(np.median(delta)) if len(delta) else np.nan,
                "ci_low": interval[0],
                "ci_high": interval[1],
                "instance_clusters": len(instances),
            }
        )
    return pd.DataFrame(rows)


def target_table(results: Sequence[dict[str, Any]], targets: Sequence[float] = TARGETS) -> pd.DataFrame:
    rows = []
    for result in results:
        for target in targets:
            hit = next((p["evaluations"] for p in result["trajectory"] if p["relative_gap"] <= target), np.inf)
            rows.append(
                {
                    **{k: result[k] for k in ("function", "dimension", "noise", "algorithm")},
                    "target": target,
                    "evaluations": hit,
                }
            )
    return pd.DataFrame(rows)


def plot_quality(results: Sequence[dict[str, Any]], function: int | None = None) -> Any:
    selected = [r for r in results if function is None or r["function"] == function]
    budgets = np.unique(np.geomspace(1, max(r["budget"] for r in selected), 60).astype(int)).tolist()
    frame = checkpoint_table(selected, budgets)
    dimensions = sorted(frame.dimension.unique())
    noises = list(dict.fromkeys(frame.noise))
    figure, axes = plt.subplots(
        len(dimensions), len(noises), squeeze=False, figsize=(4 * len(noises), 3 * len(dimensions)), sharex=True
    )
    for i, dimension in enumerate(dimensions):
        for j, noise in enumerate(noises):
            ax = axes[i, j]
            subset = frame[(frame.dimension == dimension) & (frame.noise == noise)]
            for algorithm, color in (("xnes", "tab:red"), ("cma", "tab:blue")):
                group = subset[subset.algorithm == algorithm].groupby("budget")
                median = group.relative_gap.median().clip(lower=1e-16)
                lower = group.relative_gap.quantile(0.25).clip(lower=1e-16)
                upper = group.relative_gap.quantile(0.75).clip(lower=1e-16)
                ax.plot(median.index, median, color=color, label=algorithm)
                ax.fill_between(median.index, lower, upper, color=color, alpha=0.15)
                best = group.best_relative_gap.median().clip(lower=1e-16)
                ax.plot(best.index, best, color=color, linestyle=":", alpha=0.6)
            count = subset[subset.budget == subset.budget.max()].groupby("algorithm").failed.sum().to_dict()
            ax.set(
                title=f"d={dimension}, {noise}\nfailures: {count}",
                xscale="log",
                yscale="log",
                xlabel="objective calls",
                ylabel="gap / initial gap",
            )
            ax.grid(alpha=0.2)
            ax.legend()
    figure.suptitle("Current mean: median + case IQR; dotted: best evaluated (oracle). Finite survivors only.")
    figure.tight_layout()
    return figure


def plot_targets(results: Sequence[dict[str, Any]]) -> Any:
    frame = target_table(results)
    noises = list(dict.fromkeys(frame.noise))
    figure, axes = plt.subplots(1, len(noises), squeeze=False, figsize=(4 * len(noises), 3.5))
    for ax, noise in zip(axes[0], noises, strict=True):
        for algorithm, color in (("xnes", "tab:red"), ("cma", "tab:blue")):
            values = np.sort(frame[(frame.noise == noise) & (frame.algorithm == algorithm)].evaluations.to_numpy())
            finite = values[np.isfinite(values)]
            # Extend the last attained fraction through the budget, including zero-hit runs.
            end = max(r["budget"] for r in results if r["noise"] == noise and r["algorithm"] == algorithm)
            ax.step(
                np.r_[1, np.maximum(finite, 1), end],
                np.r_[0, np.arange(1, len(finite) + 1), len(finite)] / len(values),
                where="post",
                label=algorithm,
                color=color,
            )
        ax.set(title=noise, xscale="log", xlabel="objective calls", ylabel="fraction of run–target pairs", ylim=(0, 1))
        ax.set_xlim(1, max(r["budget"] for r in results))
        ax.grid(alpha=0.2)
        ax.legend()
    figure.suptitle("First attainment by the current mean; all runs remain in the denominator")
    figure.tight_layout()
    return figure


def plot_stability(results: Sequence[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for result in results:
        trace = result["trajectory"]
        color = "tab:orange" if result["noise"] == "constant" else "tab:purple"
        for ax, metric in zip(axes, ("mean_drift", "scale_global", "axis_ratio"), strict=True):
            ax.plot([p["evaluations"] for p in trace], [max(p[metric], 1e-16) for p in trace], color=color, alpha=0.3)
            ax.set(xlabel="objective calls", ylabel=metric, yscale="log")
            ax.grid(alpha=0.2)
    figure.suptitle("xNES: constant scores (orange), random rankings (purple); traces end at the first stop")
    figure.tight_layout()
    return figure
