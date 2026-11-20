from __future__ import annotations

from datetime import datetime, timedelta, timezone

from wellstream.features import FEATURES, FeatureBuilder, explain
from wellstream.models import WindowAggregate
from wellstream.windows import window_readings
from wellstream.models import Reading

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def aggregate(
    index: int,
    flow: float,
    pressure: float = 1_000.0,
    samples: int = 5,
    expected: int = 5,
    temp: float = 150.0,
) -> WindowAggregate:
    start = BASE + timedelta(minutes=5 * index)
    return WindowAggregate(
        well_id="W-0001",
        window_start=start,
        window_end=start + timedelta(minutes=5),
        sample_count=samples,
        expected_samples=expected,
        avg_flow_bbl=flow,
        total_volume_bbl=flow * samples / 1440,
        flow_cv=0.02,
        avg_pressure_psi=pressure,
        max_pressure_psi=pressure * 1.01,
        avg_temp_f=temp,
    )


def test_features_are_withheld_during_warmup() -> None:
    builder = FeatureBuilder(min_baseline=4)
    assert [builder.push(aggregate(i, 100.0)) for i in range(4)] == [None, None, None, None]
    assert builder.push(aggregate(4, 100.0)) is not None


def test_production_drop_shows_up_in_flow_and_decline_features() -> None:
    builder = FeatureBuilder(min_baseline=4)
    for index in range(4):
        builder.push(aggregate(index, 100.0))
    vector = builder.push(aggregate(4, 50.0))
    assert vector is not None
    assert vector.values["flow_drop_ratio"] > 0.45
    assert vector.values["decline_deviation"] > 0.45
    assert abs(vector.values["pressure_z"]) < 0.1


def test_pressure_spike_shows_up_even_without_flow_loss() -> None:
    builder = FeatureBuilder(min_baseline=4)
    for index, pressure in enumerate([1_000.0, 1_010.0, 990.0, 1_005.0]):
        builder.push(aggregate(index, 100.0, pressure=pressure))
    vector = builder.push(aggregate(4, 100.0, pressure=1_400.0))
    assert vector is not None
    assert vector.values["pressure_z"] > 2.0
    assert abs(vector.values["flow_drop_ratio"]) < 0.05


def test_missing_samples_become_a_coverage_gap() -> None:
    builder = FeatureBuilder(min_baseline=2)
    builder.push(aggregate(0, 100.0))
    builder.push(aggregate(1, 100.0))
    vector = builder.push(aggregate(2, 100.0, samples=2, expected=5))
    assert vector is not None
    assert vector.values["coverage_gap"] == 0.6


def test_feature_set_is_stable_and_ordered() -> None:
    builder = FeatureBuilder(min_baseline=2)
    builder.push(aggregate(0, 100.0))
    builder.push(aggregate(1, 100.0))
    vector = builder.push(aggregate(2, 99.0))
    assert vector is not None
    assert tuple(vector.values) == FEATURES
    assert vector.as_list(FEATURES) == [vector.values[name] for name in FEATURES]


def test_explanations_are_human_readable() -> None:
    builder = FeatureBuilder(min_baseline=4)
    for index in range(4):
        builder.push(aggregate(index, 100.0))
    vector = builder.push(aggregate(4, 40.0))
    assert vector is not None
    reasons = explain(vector)
    assert reasons
    assert any("flow_drop_ratio" in reason for reason in reasons)


def test_values_are_clipped_so_one_bad_window_cannot_dominate() -> None:
    builder = FeatureBuilder(min_baseline=4)
    for index in range(4):
        builder.push(aggregate(index, 100.0))
    vector = builder.push(aggregate(4, 0.0))
    assert vector is not None
    assert -3.0 <= vector.values["flow_drop_ratio"] <= 3.0
    assert -3.0 <= vector.values["decline_deviation"] <= 3.0


def test_builder_accepts_streamed_windows() -> None:
    readings = [
        Reading("W-0001", BASE + timedelta(minutes=minute), 1_000.0, 150.0, 100.0)
        for minute in range(60)
    ]
    windows = window_readings(readings, window_seconds=300, allowed_lateness_seconds=0)
    builder = FeatureBuilder(min_baseline=4)
    vectors = [builder.push(window) for window in windows]
    assert len([v for v in vectors if v is not None]) == len(windows) - 4
