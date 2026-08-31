"""Deterministic synthetic transaction generator.

The generator is intentionally explicit about being synthetic. It creates a
learnable rare-event problem for demonstrating the engineering pipeline; its
scores must never be presented as evidence of performance on bank traffic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_transactions(rows: int = 50_000, seed: int = 2026) -> pd.DataFrame:
    """Generate chronological transactions with mild late-period drift."""
    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)
    event_time = pd.date_range("2025-01-01", periods=rows, freq="2min", tz="UTC")
    late = np.arange(rows) >= int(rows * 0.8)

    amount = rng.lognormal(mean=3.45 + late * 0.10, sigma=1.0, size=rows).clip(1, 12_000)
    customer_age_days = rng.gamma(shape=2.0, scale=350, size=rows).clip(1, 4_000)
    distance = rng.exponential(scale=38 + late * 9, size=rows).clip(0, 6_000)
    minutes_since = rng.exponential(scale=220, size=rows).clip(0.05, 2_000)
    velocity = rng.poisson(lam=0.45 + late * 0.10, size=rows).clip(0, 12)
    merchant_risk = rng.beta(a=1.4, b=7.0, size=rows)
    new_device = rng.binomial(1, 0.11 + late * 0.04, size=rows)
    foreign = rng.binomial(1, 0.07 + late * 0.03, size=rows)
    failed = rng.poisson(lam=0.14 + late * 0.03, size=rows).clip(0, 8)
    hour = event_time.hour.to_numpy()

    logit = (
        -5.1
        + 0.42 * (np.log1p(amount) - 3.5)
        - 0.00035 * customer_age_days
        + 0.004 * np.minimum(distance, 1_000)
        - 0.0025 * np.minimum(minutes_since, 500)
        + 0.42 * velocity
        + 2.3 * merchant_risk
        + 0.95 * new_device
        + 1.25 * foreign
        + 0.55 * failed
        + 1.15 * (new_device * foreign)
        + 0.65 * ((hour <= 4) | (hour >= 23))
    )
    probability = _sigmoid(logit)
    fraud = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:08d}" for i in range(rows)],
            "event_time": event_time,
            "amount": amount.round(2),
            "customer_age_days": customer_age_days.round(1),
            "distance_from_home_km": distance.round(1),
            "minutes_since_last_txn": minutes_since.round(2),
            "transactions_last_10m": velocity,
            "merchant_risk": merchant_risk.round(4),
            "is_new_device": new_device,
            "is_foreign": foreign,
            "failed_attempts_24h": failed,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "fraud": fraud,
        }
    )


def random_live_transaction(seed: int | None = None, suspicious: bool = False) -> dict:
    """Create one plausible request for the dashboard simulator."""
    rng = np.random.default_rng(seed)
    now = datetime.now(UTC)
    if suspicious:
        hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
        now = now.replace(hour=hour)
        amount = float(rng.lognormal(7.0, 0.55))
        distance = float(rng.uniform(600, 4_000))
        minutes = float(rng.uniform(0.05, 2.0))
        velocity = int(rng.integers(4, 10))
        merchant_risk = float(rng.uniform(0.55, 0.98))
        new_device = 1
        foreign = 1
        failed = int(rng.integers(1, 6))
    else:
        amount = float(rng.lognormal(3.5, 0.9))
        distance = float(rng.exponential(35))
        minutes = float(rng.exponential(210))
        velocity = int(rng.poisson(0.4))
        merchant_risk = float(rng.beta(1.4, 7.0))
        new_device = int(rng.binomial(1, 0.1))
        foreign = int(rng.binomial(1, 0.06))
        failed = int(rng.poisson(0.12))

    return {
        "transaction_id": f"live_{rng.integers(10**8, 10**9)}",
        "event_time": now.isoformat(),
        "amount": round(amount, 2),
        "customer_age_days": round(float(rng.gamma(2.0, 350)), 1),
        "distance_from_home_km": round(distance, 1),
        "minutes_since_last_txn": round(minutes, 2),
        "transactions_last_10m": velocity,
        "merchant_risk": round(merchant_risk, 4),
        "is_new_device": new_device,
        "is_foreign": foreign,
        "failed_attempts_24h": failed,
        "hour_sin": float(np.sin(2 * np.pi * now.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * now.hour / 24)),
    }


def shifted_live_batch(rows: int = 500, seed: int = 99) -> pd.DataFrame:
    """Create deliberate drift for tests and the demonstration script."""
    frame = generate_transactions(max(rows, 100), seed=seed).tail(rows).copy()
    frame["amount"] *= 3.0
    frame["is_foreign"] = 1
    frame["is_new_device"] = 1
    frame["event_time"] = [datetime.now(UTC) + timedelta(seconds=i) for i in range(rows)]
    return frame
