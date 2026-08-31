"""FastAPI application for online fraud scoring and monitoring."""

from __future__ import annotations

import os
import random
import threading
import time
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from prometheus_client import Counter as PromCounter

from fraud_detection.drift import drift_report
from fraud_detection.features import FEATURE_NAMES
from fraud_detection.generator import generate_transactions, random_live_transaction
from fraud_detection.model import FraudModel, save_model, train_model
from fraud_detection.schemas import Decision, Feedback, ScoreResponse, Transaction

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", ROOT / "artifacts/model.joblib"))
METADATA_PATH = Path(os.environ.get("FRAUD_METADATA_PATH", ROOT / "artifacts/metadata.json"))
RECENT_LIMIT = int(os.environ.get("FRAUD_RECENT_LIMIT", "500"))
UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"

REQUESTS = PromCounter("fraud_scores_total", "Transactions scored", ["decision"])
LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "Online scoring latency",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PROBABILITY = Histogram(
    "fraud_probability",
    "Predicted fraud probability",
    buckets=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
)
DRIFT = Gauge("fraud_max_psi", "Maximum recent Population Stability Index")
FEEDBACK = PromCounter("fraud_feedback_total", "Confirmed outcomes", ["confirmed_fraud"])


class FraudService:
    def __init__(self, model: FraudModel) -> None:
        self.model = model
        self.recent: deque[dict] = deque(maxlen=RECENT_LIMIT)
        self.feedback: dict[str, bool] = {}
        self.lock = threading.Lock()
        self.started = datetime.now(UTC)

    def score(self, transaction: Transaction) -> ScoreResponse:
        started = time.perf_counter()
        record = transaction.model_dump(mode="json")
        probability = self.model.probability(record)
        decision = self.model.decision(probability)
        reasons = self.model.explain(record)
        latency_ms = (time.perf_counter() - started) * 1000
        scored_at = datetime.now(UTC)
        response = ScoreResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(probability, 6),
            decision=decision,
            reasons=reasons,
            model_version=self.model.version,
            scored_at=scored_at,
            latency_ms=round(latency_ms, 3),
        )
        stored = {
            **{name: record[name] for name in FEATURE_NAMES},
            "transaction_id": transaction.transaction_id,
            "event_time": transaction.event_time.isoformat(),
            "fraud_probability": response.fraud_probability,
            "decision": decision.value,
            "reasons": [reason.model_dump() for reason in reasons],
            "latency_ms": response.latency_ms,
        }
        with self.lock:
            self.recent.appendleft(stored)
        REQUESTS.labels(decision=decision.value).inc()
        LATENCY.observe(latency_ms / 1000)
        PROBABILITY.observe(probability)
        return response

    def drift(self) -> dict:
        with self.lock:
            records = list(self.recent)
        frame = pd.DataFrame(records)
        report = drift_report(frame, self.model.metadata["reference"])
        if report["max_psi"] is not None:
            DRIFT.set(report["max_psi"])
        return report

    def stats(self) -> dict:
        with self.lock:
            records = list(self.recent)
            feedback_count = len(self.feedback)
        counts = Counter(record["decision"] for record in records)
        average_probability = (
            sum(record["fraud_probability"] for record in records) / len(records) if records else 0
        )
        return {
            "scored": len(records),
            "decisions": {decision.value: counts[decision.value] for decision in Decision},
            "average_probability": round(average_probability, 4),
            "feedback_received": feedback_count,
            "uptime_seconds": int((datetime.now(UTC) - self.started).total_seconds()),
            "drift": self.drift(),
        }


def _load_service() -> FraudService:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        frame = generate_transactions(rows=5_000, seed=2026)
        model, report = train_model(frame)
        save_model(model, report, MODEL_PATH, METADATA_PATH)
    return FraudService(FraudModel.load(MODEL_PATH, METADATA_PATH))


service = _load_service()
app = FastAPI(
    title="Real-Time Fraud Detection",
    version="0.1.0",
    description="Online fraud scoring with explanations, drift monitoring, and measured latency.",
)


@app.post("/api/score", response_model=ScoreResponse)
def score(transaction: Transaction) -> ScoreResponse:
    return service.score(transaction)


@app.post("/api/simulate", response_model=ScoreResponse)
def simulate(suspicious: bool = Query(default=False)) -> ScoreResponse:
    payload = random_live_transaction(seed=random.randint(1, 10**9), suspicious=suspicious)
    return service.score(Transaction.model_validate(payload))


@app.get("/api/recent")
def recent(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    with service.lock:
        transactions = list(service.recent)[:limit]
    return {"transactions": transactions}


@app.get("/api/stats")
def stats() -> dict:
    return service.stats()


@app.get("/api/drift")
def drift() -> dict:
    return service.drift()


@app.get("/api/model")
def model_info() -> dict:
    metadata = service.model.metadata
    return {
        "model_version": metadata["model_version"],
        "provenance": metadata["provenance"],
        "features": metadata["features"],
        "thresholds": metadata["thresholds"],
        "validation": metadata["validation"],
        "test": metadata["test"],
    }


@app.post("/api/feedback")
def feedback(body: Feedback) -> dict:
    with service.lock:
        if not any(item["transaction_id"] == body.transaction_id for item in service.recent):
            raise HTTPException(
                status_code=404,
                detail="transaction is not in the recent scoring window",
            )
        service.feedback[body.transaction_id] = body.confirmed_fraud
    FEEDBACK.labels(confirmed_fraud=str(body.confirmed_fraud).lower()).inc()
    return {"recorded": True, "transaction_id": body.transaction_id}


@app.get("/healthz")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": service.model.version,
        "model_loaded": True,
        "dataset": service.model.metadata["provenance"]["dataset"],
        "real_bank_data": False,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(UI_PATH)
