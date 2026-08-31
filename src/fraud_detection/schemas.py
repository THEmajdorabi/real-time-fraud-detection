"""Public API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class Decision(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: f"txn_{uuid4().hex[:12]}")
    event_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    amount: float = Field(gt=0, le=1_000_000)
    customer_age_days: float = Field(ge=0, le=20_000)
    distance_from_home_km: float = Field(ge=0, le=50_000)
    minutes_since_last_txn: float = Field(ge=0, le=100_000)
    transactions_last_10m: int = Field(ge=0, le=1_000)
    merchant_risk: float = Field(ge=0, le=1)
    is_new_device: int = Field(ge=0, le=1)
    is_foreign: int = Field(ge=0, le=1)
    failed_attempts_24h: int = Field(ge=0, le=1_000)
    hour_sin: float = Field(ge=-1, le=1)
    hour_cos: float = Field(ge=-1, le=1)


class Reason(BaseModel):
    feature: str
    label: str
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: Decision
    reasons: list[Reason]
    model_version: str
    scored_at: datetime
    latency_ms: float


class Feedback(BaseModel):
    transaction_id: str
    confirmed_fraud: bool
