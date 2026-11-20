"""Logistic-regression scoring implemented on NumPy.

Why hand-rolled instead of scikit-learn? The whole model is ~80 lines of linear
algebra, trains in milliseconds, serialises to plain JSON (so it can ship in a
container without a pickle), and every coefficient is inspectable. For a
tabular, low-dimensional anomaly score that is the right trade-off; the weights
are the explanation an operator or auditor asks for.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .features import FEATURES
from .models import FeatureVector


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable form: avoid exp overflow on large |z|.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class LogisticModel:
    """Binary logistic model with standardised inputs.

    ``threshold`` is tuned on the training split to favour precision: a false
    alarm on a producing well costs field time, so the default operating point
    is purposefully conservative.
    """

    feature_names: tuple[str, ...] = FEATURES
    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))
    bias: float = 0.0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))
    scale: np.ndarray = field(default_factory=lambda: np.ones(len(FEATURES)))
    threshold: float = 0.5
    model_version: str = "anomaly-logit-v1"
    trained_rows: int = 0

    # ----------------------------- training ----------------------------- #
    def fit(
        self,
        x: Sequence[Sequence[float]],
        y: Sequence[int],
        epochs: int = 600,
        learning_rate: float = 0.15,
        l2: float = 0.01,
    ) -> "LogisticModel":
        matrix = np.asarray(x, dtype=float)
        labels = np.asarray(y, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0]:
            raise ValueError("x must be 2-D with one row per label")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {matrix.shape[1]}")
        if len(np.unique(labels)) < 2:
            raise ValueError("training data needs both positive and negative examples")

        self.mean = matrix.mean(axis=0)
        raw_scale = matrix.std(axis=0)
        self.scale = np.where(raw_scale > 1e-9, raw_scale, 1.0)
        z = (matrix - self.mean) / self.scale

        weights = np.zeros(self.feature_names.__len__())
        bias = 0.0
        n = z.shape[0]
        for _ in range(epochs):
            probs = _sigmoid(z @ weights + bias)
            error = probs - labels
            grad_w = (z.T @ error) / n + l2 * weights
            grad_b = float(error.mean())
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
        self.weights = weights
        self.bias = bias
        self.trained_rows = int(n)
        self.threshold = self._tune_threshold(z, labels)
        return self

    def _tune_threshold(self, z: np.ndarray, labels: np.ndarray) -> float:
        best_threshold, best_score = 0.5, -1.0
        for candidate in np.linspace(0.05, 0.95, 19):
            probs = _sigmoid(z @ self.weights + self.bias)
            tp = float(np.sum((probs >= candidate) & (labels == 1)))
            fp = float(np.sum((probs >= candidate) & (labels == 0)))
            fn = float(np.sum((probs < candidate) & (labels == 1)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            # Favour precision, but require usable recall.
            score = precision * 0.6 + recall * 0.4
            if score > best_score:
                best_threshold, best_score = float(candidate), score
        return round(best_threshold, 3)

    # ---------------------------- inference ---------------------------- #
    def predict_proba(self, values: Mapping[str, float]) -> float:
        row = np.array([float(values.get(name, 0.0)) for name in self.feature_names])
        z = ((row - self.mean) / self.scale) @ self.weights + self.bias
        return float(_sigmoid(np.array([z]))[0])

    def predict(self, values: Mapping[str, float]) -> int:
        return int(self.predict_proba(values) >= self.threshold)

    def score_features(self, vector: FeatureVector) -> tuple[float, str]:
        score = self.predict_proba(vector.values)
        decision = "flag" if score >= self.threshold else "monitor"
        return score, decision

    def contributions(self, values: Mapping[str, float]) -> dict[str, float]:
        """Per-feature log-odds contribution, for alert explanations."""
        row = np.array([float(values.get(name, 0.0)) for name in self.feature_names])
        z = (row - self.mean) / self.scale
        return {name: float(w * v) for name, w, v in zip(self.feature_names, self.weights, z)}

    # --------------------------- persistence --------------------------- #
    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "mean": [float(m) for m in self.mean],
            "scale": [float(s) for s in self.scale],
            "threshold": float(self.threshold),
            "trained_rows": self.trained_rows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogisticModel":
        return cls(
            feature_names=tuple(payload["feature_names"]),  # type: ignore[arg-type]
            weights=np.asarray(payload["weights"], dtype=float),  # type: ignore[arg-type]
            bias=float(payload["bias"]),  # type: ignore[arg-type]
            mean=np.asarray(payload["mean"], dtype=float),  # type: ignore[arg-type]
            scale=np.asarray(payload["scale"], dtype=float),  # type: ignore[arg-type]
            threshold=float(payload.get("threshold", 0.5)),  # type: ignore[arg-type]
            model_version=str(payload.get("model_version", "anomaly-logit-v1")),
            trained_rows=int(payload.get("trained_rows", 0)),  # type: ignore[arg-type]
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------ metrics ------------------------------ #
def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Rank-based AUC, tie-aware (equivalent to the Mann-Whitney U statistic)."""
    y = np.asarray(list(labels), dtype=float)
    s = np.asarray(list(scores), dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_report(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float]:
    cells = confusion(labels, scores, threshold)
    precision = cells["tp"] / (cells["tp"] + cells["fp"]) if cells["tp"] + cells["fp"] else 0.0
    recall = cells["tp"] / (cells["tp"] + cells["fn"]) if cells["tp"] + cells["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (cells["tp"] + cells["tn"]) / max(1, len(labels))
    brier = float(np.mean((np.asarray(scores, dtype=float) - np.asarray(labels, dtype=float)) ** 2))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "brier": round(brier, 4),
        "roc_auc": round(roc_auc(labels, scores), 4),
        "threshold": round(threshold, 4),
    }


def log_odds(probability: float) -> float:
    clamped = min(max(probability, 1e-9), 1 - 1e-9)
    return math.log(clamped / (1 - clamped))
