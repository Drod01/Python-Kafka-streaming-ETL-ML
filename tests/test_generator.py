from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from wellstream.generator import (
    DEFAULT_PROFILES,
    build_scenario,
    fault_windows,
    iter_events,
    well_rows,
)
from wellstream.models import Reading

WINDOW = timedelta(seconds=300)


def test_generation_is_deterministic_for_a_seed() -> None:
    first, first_faults = build_scenario(hours=1, seed=42)
    second, second_faults = build_scenario(hours=1, seed=42)
    assert first == second
    assert first_faults == second_faults


def test_different_seeds_produce_different_noise() -> None:
    first, _ = build_scenario(hours=1, seed=1)
    second, _ = build_scenario(hours=1, seed=2)
    assert first[0].flow_bbl != second[0].flow_bbl


def test_every_well_is_represented_and_ordered() -> None:
    readings, _ = build_scenario(hours=1, seed=3)
    assert {reading.well_id for reading in readings} == {profile.well_id for profile in DEFAULT_PROFILES}
    timestamps = [reading.ts for reading in readings]
    assert timestamps == sorted(timestamps)
    assert len(readings) == len(DEFAULT_PROFILES) * 60


def test_fault_windows_align_with_the_tumbling_grid() -> None:
    _, faults = build_scenario(hours=2, seed=4)
    labels = fault_windows(faults, window_seconds=300)
    assert labels
    for _, start in labels:
        assert int(start.timestamp()) % 300 == 0


def test_faulted_windows_drop_the_rate_below_trend() -> None:
    readings, faults = build_scenario(hours=2, seed=5)
    fault = next(spec for spec in faults if spec.flow_factor < 0.5)
    before = [
        reading.flow_bbl
        for reading in readings
        if reading.well_id == fault.well_id
        and fault.window_start - WINDOW <= reading.ts < fault.window_start
    ]
    during = [
        reading.flow_bbl
        for reading in readings
        if reading.well_id == fault.well_id
        and fault.window_start <= reading.ts < fault.window_start + WINDOW
    ]
    assert before and during
    # Natural decline inside five minutes is ~0.01%, so any large gap is the fault.
    assert sum(during) / len(during) < 0.6 * (sum(before) / len(before))


def test_event_contract_round_trips() -> None:
    readings, _ = build_scenario(hours=1, seed=6)
    events = list(iter_events(readings))
    assert Reading.from_event(events[0]) == readings[0]
    assert isinstance(events[0]["ts"], str)


def test_well_rows_match_profiles() -> None:
    rows = well_rows()
    assert len(rows) == len(DEFAULT_PROFILES)
    assert rows[0]["well_id"] == DEFAULT_PROFILES[0].well_id
    assert rows[0]["spud_date"] == DEFAULT_PROFILES[0].spud_date


def test_invalid_configuration_is_refused() -> None:
    with pytest.raises(ValueError):
        build_scenario(hours=0)


def test_generated_timestamps_are_timezone_aware() -> None:
    readings, _ = build_scenario(hours=1, seed=9)
    assert all(reading.ts.tzinfo == timezone.utc for reading in readings)
    assert isinstance(datetime.now(timezone.utc), datetime)
