"""Measure in-process scoring latency without HTTP or network noise."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fraud_detection.generator import random_live_transaction
from fraud_detection.model import FraudModel


def percentile(values: list[float], value: float) -> float:
    return round(float(np.percentile(values, value)), 4)


def main() -> None:
    model = FraudModel.load(Path("artifacts/model.joblib"), Path("artifacts/metadata.json"))
    latencies = []
    for index in range(5_000):
        record = random_live_transaction(seed=index, suspicious=index % 10 == 0)
        started = time.perf_counter()
        model.probability(record)
        latencies.append((time.perf_counter() - started) * 1000)

    elapsed_seconds = sum(latencies) / 1000
    report = {
        "requests": len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "model_only_requests_per_second": round(len(latencies) / elapsed_seconds, 1),
        "note": "In-process model timing; HTTP and network latency are intentionally excluded.",
    }
    Path("artifacts/benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
