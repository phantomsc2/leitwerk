import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import SupportsFloat, cast

import matplotlib.pyplot as plt
import numpy as np
from sc2.position import Point2
from sc2.unit import Unit
from scipy.spatial.distance import pdist, squareform

MOVING_AVERAGE_WINDOW = 10


def simulate_combat(units: Sequence[Unit], time_horizon: float) -> Mapping[Unit, float]:
    alliance = np.array([u.owner_id for u in units])
    range_matrix = np.array([[u.air_range if v.is_flying else u.ground_range for v in units] for u in units])
    dps = np.array([[u.calculate_dps_vs_target(v) for v in units] for u in units])
    radius = np.array([u.radius for u in units], dtype=float)
    health = np.array([max(1.0, u.health + u.shield) for u in units], dtype=float)
    speed = np.array([1.4 * u.real_speed for u in units], dtype=float)
    distance = pairwise_distances([u.position for u in units])
    range_projection = range_matrix + radius[:, None] + time_horizon * speed[:, None]
    in_range = distance <= range_projection + radius[None, :]
    is_opponent = alliance[:, None] != alliance[None, :]
    is_target = is_opponent & in_range
    num_targets = np.sum(is_target, axis=1, keepdims=True)
    targeting = np.divide(is_target, num_targets, where=num_targets != 0, out=np.zeros_like(distance, dtype=float))
    fire = dps * targeting / health[None, :]
    losses = np.sum(fire, axis=0)
    attrition = np.sum(fire, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        outcome = np.log(attrition) - np.log(losses)
    outcome_dict = dict(zip(units, outcome, strict=False))
    return outcome_dict


def medoid(points: Sequence[Point2]) -> Point2:
    distances = pairwise_distances(points)
    medoid_index = int(distances.sum(axis=1).argmin())
    return points[medoid_index]


def pairwise_distances(positions: Sequence[Point2]) -> np.ndarray:
    return squareform(pdist(positions), checks=False)


def load_history(output_dir: Path) -> list[dict[str, object]]:
    history_file = output_dir / "history.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history_file.exists():
        return []
    with history_file.open(encoding="utf-8") as f:
        return json.load(f)


def save_history(history: list[dict[str, object]], output_dir: Path) -> None:
    history_file = output_dir / "history.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def moving_average(values: list[float], window: int = MOVING_AVERAGE_WINDOW) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    series = np.asarray(values, dtype=float)
    cumulative = np.cumsum(series)
    trailing = cumulative.copy()
    trailing[window:] -= cumulative[:-window]
    counts = np.minimum(np.arange(1, len(series) + 1), window)
    return trailing / counts


def flatten_numeric_fields(value: object, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if is_dataclass(value):
        for field in fields(value):
            key = f"{prefix}.{field.name}" if prefix else field.name
            result.update(flatten_numeric_fields(getattr(value, field.name), key))
        return result
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric_fields(item, name))
        return result
    return {prefix: float(cast(SupportsFloat, value))}


def result_size(history: list[dict[str, object]]) -> int:
    return max((len(result) for entry in history if isinstance(result := entry.get("result"), list)), default=0)


def parameter_metric_names(history: list[dict[str, object]]) -> list[str]:
    return sorted({key for entry in history for key in entry if key != "result"})


def result_metric_values(history: list[dict[str, object]], objective_index: int) -> np.ndarray:
    values = np.full(len(history), np.nan, dtype=float)
    for idx, entry in enumerate(history):
        result = entry.get("result")
        if isinstance(result, list) and objective_index < len(result):
            values[idx] = float(cast(SupportsFloat, result[objective_index]))
    return values


def parameter_metric_values(history: list[dict[str, object]], metric: str) -> np.ndarray:
    return np.array([float(cast(SupportsFloat, entry.get(metric, np.nan))) for entry in history], dtype=float)


def plot_metric(
    ax: plt.Axes,
    values: np.ndarray,
    ylim: tuple[float, float] | None = None,
) -> None:
    games = np.arange(1, len(values) + 1)
    valid = ~np.isnan(values)
    ax.scatter(games[valid], values[valid], s=18, alpha=0.7)
    if np.any(valid):
        ax.plot(games[valid], moving_average(values[valid].tolist()), linewidth=2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)


def save_plot(history: list[dict[str, object]], output_dir: Path, objective_labels: Sequence[str]) -> None:
    plot_file = output_dir / "plot.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_metrics = list(range(result_size(history)))
    parameter_metrics = parameter_metric_names(history)
    rows = max(len(result_metrics), len(parameter_metrics), 1)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows), sharex=True, squeeze=False)

    for row, ax in enumerate(axes[:, 0]):
        if row >= len(result_metrics):
            ax.set_visible(False)
            continue
        result_metric = result_metrics[row]
        plot_metric(ax, result_metric_values(history, result_metric))
        title = (
            objective_labels[result_metric]
            if result_metric < len(objective_labels)
            else f"objective {result_metric + 1}"
        )
        ax.set_title(title)

    for row, ax in enumerate(axes[:, 1]):
        if row >= len(parameter_metrics):
            ax.set_visible(False)
            continue
        parameter_metric = parameter_metrics[row]
        plot_metric(ax, parameter_metric_values(history, parameter_metric))
        ax.set_title(parameter_metric)

    fig.tight_layout()
    fig.savefig(plot_file)
    plt.close(fig)
