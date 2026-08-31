"""Feature definitions shared by training and online scoring."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

FEATURE_NAMES = (
    "amount",
    "customer_age_days",
    "distance_from_home_km",
    "minutes_since_last_txn",
    "transactions_last_10m",
    "merchant_risk",
    "is_new_device",
    "is_foreign",
    "failed_attempts_24h",
    "hour_sin",
    "hour_cos",
)

REASON_LABELS = {
    "amount": "unusual transaction amount",
    "customer_age_days": "new or established customer profile",
    "distance_from_home_km": "unusual distance from home",
    "minutes_since_last_txn": "very short time since the previous transaction",
    "transactions_last_10m": "high recent transaction velocity",
    "merchant_risk": "merchant risk history",
    "is_new_device": "new device",
    "is_foreign": "foreign transaction",
    "failed_attempts_24h": "recent failed payment attempts",
    "hour_sin": "unusual transaction time",
    "hour_cos": "unusual transaction time",
}


def feature_frame(record: Mapping[str, float | int]) -> pd.DataFrame:
    """Create a one-row model frame in the exact training order."""
    return pd.DataFrame([{name: record[name] for name in FEATURE_NAMES}], columns=FEATURE_NAMES)
