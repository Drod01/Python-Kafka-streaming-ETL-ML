from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wellstream.models import Reading
from wellstream.windows import TumblingWindower, floor_to_window, window_readings

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def reading(minute: float, flow: float = 100.0, pressure: float = 1_000.0, well: str = "W-0001") -> Reading:
    return Reading(well, BASE + timedelta(minutes=minute), pressure, 150.0, flow)


def test_floor_to_window_aligns_to_epoch_grid() -> None:
    assert floor_to_window(BASE + timedelta(minutes=7, seconds=31), 300) == BASE + timedelta(minutes=5)


def test_window_closes_only_after_watermark_passes() -> None:
    windower = TumblingWindower(window_seconds=300, allowed_lateness_seconds=0, expected_cadence_seconds=60)
    for minute in range(5):
        assert windower.add(reading(minute)) == []
    closed = windower.add(reading(5))
    assert len(closed) == 1
    window = closed[0]
    assert window.window_start == BASE
    assert window.sample_count == 5
    assert window.avg_flow_bbl == pytest.approx(100.0)
    assert window.total_volume_bbl == pytest.approx(5 * 100 / 1440)
    assert window.coverage_ratio == pytest.approx(1.0)


def test_watermark_holds_window_open_for_stragglers() -> None:
    windower = TumblingWindower(window_seconds=300, allowed_lateness_seconds=120, expected_cadence_seconds=60)
    windower.add(reading(1))
    windower.add(reading(4))
    # Straggler from 00:02 still belongs to the open window.
    assert windower.add(reading(2)) == []
    assert windower.out_of_order == 1
    closed = windower.add(reading(7))
    assert len(closed) == 1
    assert closed[0].sample_count == 3


def test_events_after_close_are_counted_not_applied() -> None:
    windower = TumblingWindower(window_seconds=300, allowed_lateness_seconds=0, expected_cadence_seconds=60)
    for minute in range(5):
        windower.add(reading(minute))
    windower.add(reading(5))  # closes 00:00-00:05
    assert windower.add(reading(2)) == []
    assert windower.late_dropped == 1


def test_windows_are_keyed_per_well() -> None:
    windower = TumblingWindower(window_seconds=300, allowed_lateness_seconds=0, expected_cadence_seconds=60)
    for minute in range(5):
        windower.add(reading(minute, well="W-0001"))
        windower.add(reading(minute, well="W-0002"))
    closed = windower.add(reading(5, well="W-0001"))
    # Both wells' windows for 00:00 close together once the watermark advances.
    assert {window.well_id for window in closed} == {"W-0001", "W-0002"}


def test_flush_closes_partial_windows() -> None:
    windower = TumblingWindower(window_seconds=300, allowed_lateness_seconds=0, expected_cadence_seconds=60)
    for minute in range(3):
        windower.add(reading(minute))
    closed = windower.flush()
    assert len(closed) == 1
    assert closed[0].sample_count == 3
    assert closed[0].coverage_ratio == pytest.approx(0.6)
    assert windower.open_windows == 0


def test_coefficient_of_variation_flags_instability() -> None:
    windower = TumblingWindower(window_seconds=300, allowed_lateness_seconds=0, expected_cadence_seconds=60)
    for minute, flow in enumerate([100.0, 10.0, 100.0, 10.0, 100.0]):
        windower.add(reading(minute, flow=flow))
    window = windower.add(reading(5, flow=100.0))[0]
    assert window.flow_cv > 0.4


def test_batch_helper_matches_streaming_operator() -> None:
    readings = [reading(minute) for minute in range(10)]
    windows = window_readings(readings, window_seconds=300, allowed_lateness_seconds=0)
    assert len(windows) == 2
    assert sum(window.sample_count for window in windows) == 10


def test_invalid_configuration_is_refused() -> None:
    with pytest.raises(ValueError):
        TumblingWindower(window_seconds=0)
