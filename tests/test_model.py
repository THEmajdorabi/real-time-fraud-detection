from fraud_detection.generator import generate_transactions, random_live_transaction
from fraud_detection.model import train_model


def test_training_uses_time_split_and_produces_useful_signal() -> None:
    model, report = train_model(generate_transactions(5_000, seed=8))
    assert report["provenance"]["real_bank_data"] is False
    assert report["test"]["pr_auc"] > report["test"]["fraud_rate"]
    assert report["test"]["roc_auc"] > 0.7
    assert 0 < model.review_threshold < model.block_threshold < 1


def test_suspicious_transaction_scores_above_normal_transaction() -> None:
    model, _ = train_model(generate_transactions(8_000, seed=9))
    normal = random_live_transaction(seed=42, suspicious=False)
    suspicious = random_live_transaction(seed=42, suspicious=True)
    assert model.probability(suspicious) > model.probability(normal)
    assert model.explain(suspicious)
