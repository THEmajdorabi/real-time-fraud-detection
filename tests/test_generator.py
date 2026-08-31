from fraud_detection.generator import generate_transactions, random_live_transaction


def test_generator_is_deterministic_and_rare_event() -> None:
    first = generate_transactions(2_000, seed=7)
    second = generate_transactions(2_000, seed=7)
    assert first.equals(second)
    assert 0.005 < first["fraud"].mean() < 0.25
    assert first["event_time"].is_monotonic_increasing


def test_live_payload_respects_schema_ranges() -> None:
    payload = random_live_transaction(seed=1, suspicious=True)
    assert payload["amount"] > 0
    assert 0 <= payload["merchant_risk"] <= 1
    assert payload["is_new_device"] in (0, 1)
