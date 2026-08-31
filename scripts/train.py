"""Train and evaluate the model on deterministic synthetic transactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_detection.generator import generate_transactions
from fraud_detection.model import save_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    frame = generate_transactions(rows=args.rows, seed=args.seed)
    model, report = train_model(frame)
    save_model(model, report, args.artifacts / "model.joblib", args.artifacts / "metadata.json")
    print(json.dumps({"thresholds": report["thresholds"], "test": report["test"]}, indent=2))
    print("WARNING: the dataset is synthetic; these metrics are not bank-performance claims.")


if __name__ == "__main__":
    main()
