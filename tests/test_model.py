from __future__ import annotations

import pytest
from wellstream.features import FEATURES
from wellstream.model import (
    LogisticModel,
    classification_report,
    confusion,
    roc_auc,
)


def separable_rows(count: int = 60) -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for index in range(count):
        signal = index % 2
        base = 1.2 if signal else -0.8
        rows.append([base + 0.01 * (index % 5) for _ in FEATURES])
        labels.append(signal)
    return rows, labels


def trained_model() -> LogisticModel:
    rows, labels = separable_rows()
    return LogisticModel().fit(rows, labels)


def test_fit_learns_a_separable_signal() -> None:
    model = trained_model()
    positives = [model.predict_proba(dict(zip(FEATURES, row))) for row in separable_rows()[0]]
    labels = separable_rows()[1]
    assert roc_auc(labels, positives) > 0.95
    assert 0.0 < model.threshold < 1.0


def test_threshold_is_conservative_enough_to_limit_false_alarms() -> None:
    model = trained_model()
    rows, labels = separable_rows()
    scores = [model.predict_proba(dict(zip(FEATURES, row))) for row in rows]
    cells = confusion(labels, scores, model.threshold)
    assert cells["fp"] <= cells["tp"]


def test_scores_are_probabilities() -> None:
    model = trained_model()
    score = model.predict_proba(dict(zip(FEATURES, [2.0] * len(FEATURES))))
    assert 0.0 <= score <= 1.0
    assert model.predict(dict(zip(FEATURES, [2.0] * len(FEATURES)))) in {0, 1}


def test_model_round_trips_through_json(tmp_path) -> None:
    model = trained_model()
    path = model.save(tmp_path / "model.json")
    restored = LogisticModel.load(path)
    sample = dict(zip(FEATURES, [0.9] * len(FEATURES)))
    assert restored.predict_proba(sample) == pytest.approx(model.predict_proba(sample), rel=1e-12)
    assert restored.threshold == model.threshold
    assert restored.feature_names == model.feature_names


def test_contributions_are_signed_per_feature() -> None:
    model = trained_model()
    contributions = model.contributions(dict(zip(FEATURES, [2.0] * len(FEATURES))))
    assert set(contributions) == set(FEATURES)
    assert sum(value > 0 for value in contributions.values()) >= 1


def test_training_requires_both_classes() -> None:
    rows, _ = separable_rows()
    with pytest.raises(ValueError, match="both positive and negative"):
        LogisticModel().fit(rows, [0] * len(rows))


def test_feature_width_is_validated() -> None:
    with pytest.raises(ValueError, match="expected"):
        LogisticModel().fit([[1.0, 2.0], [0.5, 0.2]], [0, 1])


def test_auc_matches_hand_computed_value() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_handles_ties() -> None:
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_classification_report_shape() -> None:
    model = trained_model()
    rows, labels = separable_rows()
    scores = [model.predict_proba(dict(zip(FEATURES, row))) for row in rows]
    report = classification_report(labels, scores, model.threshold)
    assert {"precision", "recall", "f1", "roc_auc", "brier", "threshold"} <= set(report)
    assert 0.0 <= report["brier"] <= 1.0
