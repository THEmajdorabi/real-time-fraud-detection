"""Small, dependency-free Population Stability Index implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.features import FEATURE_NAMES


def build_reference(frame: pd.DataFrame, bins: int = 10) -> dict[str, dict[str, list[float]]]:
    reference: dict[str, dict[str, list[float]]] = {}
    for feature in FEATURE_NAMES:
        values = frame[feature].to_numpy(dtype=float)
        unique = np.unique(values)
        if len(unique) <= bins:
            # Discrete features need one bucket per observed value. Quantiles
            # collapse a binary feature to [-inf, inf], which makes a shift
            # from mostly-zero to mostly-one mathematically invisible.
            midpoints = (unique[:-1] + unique[1:]) / 2
            edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
        else:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=edges)
        proportions = ((counts + 0.5) / (counts.sum() + 0.5 * len(counts))).tolist()
        reference[feature] = {"edges": edges.tolist(), "proportions": proportions}
    return reference


def population_stability_index(values: np.ndarray, reference: dict[str, list[float]]) -> float:
    edges = np.asarray(reference["edges"], dtype=float)
    expected = np.asarray(reference["proportions"], dtype=float)
    counts, _ = np.histogram(values.astype(float), bins=edges)
    actual = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(frame: pd.DataFrame, reference: dict, minimum_rows: int = 50) -> dict:
    if len(frame) < minimum_rows:
        return {
            "status": "insufficient_data",
            "rows": len(frame),
            "minimum_rows": minimum_rows,
            "max_psi": None,
            "features": {},
        }

    scores = {
        feature: population_stability_index(frame[feature].to_numpy(), reference[feature])
        for feature in FEATURE_NAMES
    }
    maximum = max(scores.values())
    status = "drift" if maximum >= 0.25 else "warning" if maximum >= 0.10 else "stable"
    return {
        "status": status,
        "rows": len(frame),
        "max_psi": round(maximum, 4),
        "features": {key: round(value, 4) for key, value in sorted(scores.items())},
    }
