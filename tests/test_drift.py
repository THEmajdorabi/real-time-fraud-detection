from fraud_detection.drift import build_reference, drift_report
from fraud_detection.features import FEATURE_NAMES
from fraud_detection.generator import generate_transactions, shifted_live_batch


def test_small_sample_does_not_claim_stability() -> None:
    frame = generate_transactions(500, seed=1)
    reference = build_reference(frame[list(FEATURE_NAMES)])
    report = drift_report(frame.head(10), reference)
    assert report["status"] == "insufficient_data"


def test_deliberate_shift_fires_drift_alert() -> None:
    baseline = generate_transactions(2_000, seed=3)
    reference = build_reference(baseline[list(FEATURE_NAMES)])
    shifted = shifted_live_batch(300, seed=4)
    report = drift_report(shifted, reference)
    assert report["status"] == "drift"
    assert report["max_psi"] >= 0.25
