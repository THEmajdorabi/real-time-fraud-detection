"""Training, persistence, prediction, and local explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.drift import build_reference
from fraud_detection.features import FEATURE_NAMES, REASON_LABELS, feature_frame
from fraud_detection.schemas import Decision, Reason


def _choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    max_fpr: float = 0.05,
) -> float:
    """Maximise recall while keeping false positives under a visible limit."""
    candidates = np.linspace(0.05, 0.95, 181)
    choices: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        if fpr <= max_fpr:
            choices.append((recall, -fpr, float(threshold)))
    if choices:
        return max(choices)[2]
    scores = [(f1_score(labels, probabilities >= t, zero_division=0), float(t)) for t in candidates]
    return max(scores)[1]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(len(labels)),
        "fraud_rate": round(float(labels.mean()), 6),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 6),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 6),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


@dataclass
class FraudModel:
    pipeline: Pipeline
    calibrator: LogisticRegression
    review_threshold: float
    block_threshold: float
    version: str
    metadata: dict

    def probability(self, record: dict) -> float:
        raw_score = self.pipeline.decision_function(feature_frame(record)).reshape(-1, 1)
        return float(self.calibrator.predict_proba(raw_score)[0, 1])

    def decision(self, probability: float) -> Decision:
        if probability >= self.block_threshold:
            return Decision.BLOCK
        if probability >= self.review_threshold:
            return Decision.REVIEW
        return Decision.APPROVE

    def explain(self, record: dict, limit: int = 4) -> list[Reason]:
        """Return the strongest positive log-odds contributions.

        This is exact for the deployed linear model: standardised feature value
        multiplied by the learned coefficient. It is not a causal explanation.
        """
        scaler: StandardScaler = self.pipeline.named_steps["scale"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        values = feature_frame(record).to_numpy(dtype=float)[0]
        standardised = (values - scaler.mean_) / scaler.scale_
        contributions = standardised * classifier.coef_[0]
        ranked = np.argsort(contributions)[::-1]
        reasons = []
        seen_labels: set[str] = set()
        for index in ranked:
            contribution = float(contributions[index])
            if contribution <= 0:
                continue
            feature = FEATURE_NAMES[index]
            label = REASON_LABELS[feature]
            if label in seen_labels:
                continue
            reasons.append(
                Reason(
                    feature=feature,
                    label=label,
                    contribution=round(contribution, 4),
                )
            )
            seen_labels.add(label)
            if len(reasons) == limit:
                break
        return reasons

    @classmethod
    def load(cls, model_path: str | Path, metadata_path: str | Path) -> FraudModel:
        payload = joblib.load(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(
            pipeline=payload["pipeline"],
            calibrator=payload["calibrator"],
            review_threshold=float(metadata["thresholds"]["review"]),
            block_threshold=float(metadata["thresholds"]["block"]),
            version=metadata["model_version"],
            metadata=metadata,
        )


def train_model(frame: pd.DataFrame) -> tuple[FraudModel, dict]:
    """Train with chronological train/validation/test splits."""
    ordered = frame.sort_values("event_time").reset_index(drop=True)
    train_end = int(len(ordered) * 0.70)
    validation_end = int(len(ordered) * 0.85)
    train = ordered.iloc[:train_end]
    validation = ordered.iloc[train_end:validation_end]
    test = ordered.iloc[validation_end:]

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    C=0.75,
                    random_state=2026,
                ),
            ),
        ]
    )
    pipeline.fit(train[list(FEATURE_NAMES)], train["fraud"])

    # Class balancing improves the rare-event decision boundary but distorts
    # raw probability values. Fit a one-dimensional Platt calibrator on the
    # validation period so the API's `fraud_probability` is interpretable.
    validation_scores = pipeline.decision_function(validation[list(FEATURE_NAMES)]).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=2026)
    calibrator.fit(validation_scores, validation["fraud"])
    validation_probabilities = calibrator.predict_proba(validation_scores)[:, 1]
    review_threshold = _choose_threshold(validation["fraud"].to_numpy(), validation_probabilities)
    block_threshold = min(0.98, max(0.80, review_threshold + 0.25))
    test_scores = pipeline.decision_function(test[list(FEATURE_NAMES)]).reshape(-1, 1)
    test_probabilities = calibrator.predict_proba(test_scores)[:, 1]

    model_version = datetime.now(UTC).strftime("fraud-calibrated-logreg-%Y%m%dT%H%M%SZ")
    report = {
        "provenance": {
            "dataset": "deterministic synthetic transactions",
            "real_bank_data": False,
            "warning": (
                "Metrics demonstrate the pipeline only and do not estimate "
                "production fraud performance."
            ),
            "split": "chronological 70/15/15",
        },
        "model_version": model_version,
        "features": list(FEATURE_NAMES),
        "thresholds": {"review": round(review_threshold, 6), "block": round(block_threshold, 6)},
        "train_rows": len(train),
        "validation": _metrics(
            validation["fraud"].to_numpy(), validation_probabilities, review_threshold
        ),
        "test": _metrics(test["fraud"].to_numpy(), test_probabilities, review_threshold),
        "reference": build_reference(train[list(FEATURE_NAMES)]),
    }
    model = FraudModel(
        pipeline=pipeline,
        calibrator=calibrator,
        review_threshold=review_threshold,
        block_threshold=block_threshold,
        version=model_version,
        metadata=report,
    )
    return model, report


def save_model(
    model: FraudModel,
    report: dict,
    model_path: str | Path,
    metadata_path: str | Path,
) -> None:
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model.pipeline, "calibrator": model.calibrator}, model_path)
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
