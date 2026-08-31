from fastapi.testclient import TestClient

from fraud_detection.api.main import app
from fraud_detection.generator import random_live_transaction

client = TestClient(app)


def test_health_reports_synthetic_provenance() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["real_bank_data"] is False


def test_score_returns_probability_decision_and_reasons() -> None:
    response = client.post("/api/score", json=random_live_transaction(seed=2, suspicious=True))
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"approve", "review", "block"}
    assert body["model_version"]
    assert body["latency_ms"] >= 0


def test_invalid_amount_is_rejected() -> None:
    payload = random_live_transaction(seed=3)
    payload["amount"] = -1
    assert client.post("/api/score", json=payload).status_code == 422


def test_unknown_feedback_is_rejected() -> None:
    response = client.post(
        "/api/feedback", json={"transaction_id": "does-not-exist", "confirmed_fraud": True}
    )
    assert response.status_code == 404


def test_dashboard_and_metrics_are_available() -> None:
    assert client.get("/").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "fraud_scores_total" in metrics.text
