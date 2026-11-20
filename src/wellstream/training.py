"""Dataset construction and model training for the anomaly classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .features import FEATURES, FeatureBuilder
from .model import LogisticModel, classification_report
from .models import FeatureVector, Reading
from .windows import TumblingWindower


@dataclass(frozen=True)
class Dataset:
    vectors: list[FeatureVector] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def positives(self) -> int:
        return sum(self.labels)

    def matrix(self) -> list[list[float]]:
        return [vector.as_list(FEATURES) for vector in self.vectors]


def build_dataset(
    readings: Sequence[Reading],
    label_windows: set[tuple[str, datetime]],
    windower: TumblingWindower | None = None,
    features: FeatureBuilder | None = None,
) -> Dataset:
    """Replay readings through the streaming operators to build training rows.

    Using the *same* windower and feature builder as production is the point:
    training/serving skew is the classic reason a model looks good offline and
    useless in the field.
    """
    windower = windower or TumblingWindower()
    features = features or FeatureBuilder()
    vectors: list[FeatureVector] = []
    labels: list[int] = []

    def score_window(window) -> None:
        vector = features.push(window)
        if vector is None:
            return
        vectors.append(vector)
        labels.append(1 if (window.well_id, window.window_start) in label_windows else 0)

    for reading in sorted(readings, key=lambda item: item.ts):
        for window in windower.add(reading):
            score_window(window)
    for window in windower.flush():
        score_window(window)
    return Dataset(vectors=vectors, labels=labels)


def split_by_time(dataset: Dataset, cutoff: datetime) -> tuple[Dataset, Dataset]:
    """Time-based split: never train on the future of the test window."""
    train_vectors: list[FeatureVector] = []
    train_labels: list[int] = []
    test_vectors: list[FeatureVector] = []
    test_labels: list[int] = []
    for vector, label in zip(dataset.vectors, dataset.labels):
        if vector.window_start < cutoff:
            train_vectors.append(vector)
            train_labels.append(label)
        else:
            test_vectors.append(vector)
            test_labels.append(label)
    return Dataset(train_vectors, train_labels), Dataset(test_vectors, test_labels)


@dataclass(frozen=True)
class TrainingReport:
    train_rows: int
    test_rows: int
    train_positives: int
    test_positives: int
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    threshold: float
    model_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_positives": self.train_positives,
            "test_positives": self.test_positives,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


def train(
    train_set: Dataset,
    test_set: Dataset,
    model_version: str = "anomaly-logit-v1",
) -> tuple[LogisticModel, TrainingReport]:
    if train_set.positives == 0:
        raise ValueError("training split contains no anomaly windows")
    model = LogisticModel(model_version=model_version)
    model.fit(
        [vector.as_list(model.feature_names) for vector in train_set.vectors],
        train_set.labels,
    )

    train_scores = [model.predict_proba(vector.values) for vector in train_set.vectors]
    test_scores = [model.predict_proba(vector.values) for vector in test_set.vectors]

    report = TrainingReport(
        train_rows=len(train_set),
        test_rows=len(test_set),
        train_positives=train_set.positives,
        test_positives=test_set.positives,
        train_metrics=classification_report(train_set.labels, train_scores, model.threshold),
        test_metrics=classification_report(test_set.labels, test_scores, model.threshold)
        if test_set.positives
        else {},
        threshold=model.threshold,
        model_version=model.model_version,
    )
    return model, report


def fault_recall(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> float:
    """Share of true anomaly windows that raised a flag."""
    positives = [(label, score) for label, score in zip(labels, scores) if label == 1]
    if not positives:
        return 0.0
    hits = sum(1 for _, score in positives if score >= threshold)
    return hits / len(positives)
