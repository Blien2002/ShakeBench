"""Deterministic Round-7 scorecard aggregation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np


METRICS = (
    "success",
    "ee_tracking_error_rms_m",
    "grip_margin_min",
    "grip_excess",
    "max_grasp_slip_m",
)
DECISION_METRICS = (
    "success",
    "ee_tracking_error_rms_m",
    "grip_margin_min",
    "max_grasp_slip_m",
)


def bootstrap_interval(
    values: Iterable[float], *, seed: int = 0, samples: int = 2000
) -> tuple[float | None, float | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return None, None
    if array.size == 1 or np.all(array == array[0]):
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def metric_summary(values: Iterable[float], *, seed: int = 0) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "ci95": [None, None]}
    low, high = bootstrap_interval(array, seed=seed)
    return {
        "mean": float(array.mean()) if array.size else float("nan"),
        "ci95": [low, high],
    }


def critical_value(points: Iterable[tuple[float, float]]) -> float | None:
    """Linearly interpolate the first success-rate crossing of 0.5."""

    ordered = sorted((float(x), float(y)) for x, y in points)
    for x, y in ordered:
        if y == 0.5:
            return x
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if (y0 - 0.5) * (y1 - 0.5) < 0.0:
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return None


def episode_is_valid(
    episode: Mapping[str, Any],
    *,
    penetration_limit_mm: float,
    gamma_tolerance: float = 1.0e-6,
) -> bool:
    return bool(
        episode.get("scoreable", False)
        and episode.get("support_geometry_valid", False)
        and not episode.get("grasp_assist_used", True)
        and float(episode.get("max_penetration_mm", math.inf)) < penetration_limit_mm
        and math.isclose(
            float(episode.get("gamma_realized", math.nan)),
            float(episode.get("gamma_expected", math.nan)),
            rel_tol=0.0,
            abs_tol=gamma_tolerance,
        )
    )


def _group_success(
    episodes: list[Mapping[str, Any]], key: str
) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for episode in episodes:
        value = episode.get(key)
        if value is not None:
            grouped[float(value)].append(float(bool(episode.get("success", False))))
    return [(value, float(np.mean(success))) for value, success in grouped.items()]


def _critical_summary(
    episodes: list[Mapping[str, Any]], key: str, *, seed: int, samples: int = 2000
) -> dict[str, Any]:
    point = critical_value(_group_success(episodes, key))
    grouped_lists: dict[float, list[float]] = defaultdict(list)
    for episode in episodes:
        value = episode.get(key)
        if value is not None:
            grouped_lists[float(value)].append(
                float(bool(episode.get("success", False)))
            )
    grouped = {
        value: np.asarray(values, np.float64) for value, values in grouped_lists.items()
    }
    if point is None or len(grouped) < 2:
        return {"mean": point, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        points = []
        for value, outcomes in grouped.items():
            resampled = outcomes[rng.integers(0, len(outcomes), size=len(outcomes))]
            points.append((value, float(resampled.mean())))
        estimate = critical_value(points)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return {"mean": point, "ci95": [None, None]}
    low, high = np.quantile(np.asarray(estimates), (0.025, 0.975))
    return {"mean": point, "ci95": [float(low), float(high)]}


def _predictability_summary(
    episodes: list[Mapping[str, Any]], *, seed: int, samples: int = 2000
) -> dict[str, Any]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for episode in episodes:
        value = episode.get("bandwidth_ratio")
        if value is not None:
            grouped[float(value)].append(float(bool(episode.get("success", False))))
    if 0.0 not in grouped or 0.40 not in grouped:
        return {"mean": None, "ci95": [None, None]}
    narrow = np.asarray(grouped[0.0], np.float64)
    wide = np.asarray(grouped[0.40], np.float64)
    point = float(narrow.mean() - wide.mean())
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, np.float64)
    for index in range(samples):
        narrow_sample = narrow[rng.integers(0, len(narrow), size=len(narrow))]
        wide_sample = wide[rng.integers(0, len(wide), size=len(wide))]
        estimates[index] = narrow_sample.mean() - wide_sample.mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {"mean": point, "ci95": [float(low), float(high)]}


def aggregate_episodes(
    episodes: list[Mapping[str, Any]],
    *,
    penetration_limit_mm: float,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    valid = [
        episode
        for episode in episodes
        if episode_is_valid(episode, penetration_limit_mm=penetration_limit_mm)
    ]
    metrics = {
        key: metric_summary(
            [float(episode.get(key, 0.0)) for episode in valid],
            seed=bootstrap_seed + index,
        )
        for index, key in enumerate(METRICS)
    }
    metrics["SR"] = metrics.pop("success")
    metrics["Gamma_c"] = _critical_summary(
        valid, "gamma_target", seed=bootstrap_seed + 101
    )
    metrics["f_c"] = _critical_summary(
        valid, "center_frequency_hz", seed=bootstrap_seed + 102
    )
    metrics["control_freq_c"] = _critical_summary(
        valid, "control_freq", seed=bootstrap_seed + 103
    )
    metrics["predictability_gain"] = _predictability_summary(
        valid, seed=bootstrap_seed + 104
    )
    return {
        "n_valid": len(valid),
        "n_voided": len(episodes) - len(valid),
        "void_rate": (len(episodes) - len(valid)) / len(episodes) if episodes else 0.0,
        "paper_eligible": bool(episodes) and (len(episodes) - len(valid)) / len(episodes) <= 0.05,
        "metrics": metrics,
    }


def paired_bootstrap_difference(
    left: list[Mapping[str, Any]],
    right: list[Mapping[str, Any]],
    key: str,
    *,
    seed: int = 0,
    samples: int = 5000,
) -> dict[str, Any]:
    """Return left-minus-right paired mean and bootstrap interval."""

    identity = lambda episode: (
        episode.get("task_name"),
        episode.get("init_state_index"),
    )
    right_by_id = {identity(episode): episode for episode in right}
    differences = []
    for episode in left:
        other = right_by_id.get(identity(episode))
        if other is None:
            continue
        left_value = float(bool(episode[key])) if key == "success" else float(episode[key])
        right_value = float(bool(other[key])) if key == "success" else float(other[key])
        differences.append(left_value - right_value)
    values = np.asarray(differences, np.float64)
    if values.size == 0:
        return {"mean": None, "ci95": [None, None], "n_pairs": 0}
    if values.size == 1 or np.all(values == values[0]):
        interval = [float(values[0]), float(values[0])]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, values.size, size=(samples, values.size))
        low, high = np.quantile(values[indices].mean(axis=1), (0.025, 0.975))
        interval = [float(low), float(high)]
    return {"mean": float(values.mean()), "ci95": interval, "n_pairs": int(values.size)}


def decision_point(
    phase_card: Mapping[str, Any], reactive_card: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare matched oracle_phase/reactive episodes without unpaired leakage."""

    if phase_card.get("policy_name") != "oracle_phase":
        raise ValueError("phase_card must be an oracle_phase scorecard")
    if reactive_card.get("policy_name") != "oracle_reactive":
        raise ValueError("reactive_card must be an oracle_reactive scorecard")
    phase_limit = float(phase_card["penetration_limit_mm"])
    reactive_limit = float(reactive_card["penetration_limit_mm"])
    phase = [
        episode
        for episode in phase_card.get("episodes", [])
        if episode_is_valid(episode, penetration_limit_mm=phase_limit)
    ]
    reactive = [
        episode
        for episode in reactive_card.get("episodes", [])
        if episode_is_valid(episode, penetration_limit_mm=reactive_limit)
    ]
    differences = {
        key: paired_bootstrap_difference(phase, reactive, key, seed=1000 + index)
        for index, key in enumerate(DECISION_METRICS)
    }
    success_ci = differences["success"]["ci95"]
    protocol_eligible = bool(
        phase_card.get("control_freq") == reactive_card.get("control_freq") == 5
        and phase_card.get("gamma_target") == reactive_card.get("gamma_target") == [0.5]
        and phase_card.get("frequency_scale")
        == reactive_card.get("frequency_scale")
        == [1.0]
    )
    supported = bool(
        protocol_eligible
        and success_ci[0] is not None
        and success_ci[0] > 0.0
        and differences["success"]["n_pairs"] > 0
    )
    n_pairs = differences["success"]["n_pairs"]
    return {
        "phase_policy": "oracle_phase",
        "reactive_policy": "oracle_reactive",
        "differences_phase_minus_reactive": differences,
        "protocol_eligible": protocol_eligible,
        "prediction_supported": supported,
        "decision": (
            "continue"
            if supported
            else ("stop" if protocol_eligible and n_pairs else "pending")
        ),
    }
