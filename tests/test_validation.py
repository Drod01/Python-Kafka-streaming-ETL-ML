from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wellstream.models import Reading
from wellstream.validation import (
    Deduplicator,
    RuleSet,
    reading_key,
    split_valid,
    validate_reading,
)

NOW = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


def make_reading(**overrides: object) -> Reading:
    payload = {
        "well_id": "W-0001",
        "ts": NOW,
        "pressure_psi": 1_800.0,
        "temp_f": 148.0,
        "flow_bbl": 640.0,
        "source": "scada",
    }
    payload.update(overrides)
    return Reading(**payload)  # type: ignore[arg-type]


def test_accepts_a_nominal_sample() -> None:
    result = validate_reading(make_reading(), now=NOW)
    assert result.accepted
    assert result.reasons == ()


def test_reports_every_violation_at_once() -> None:
    result = validate_reading(make_reading(pressure_psi=9_999.0, flow_bbl=-5.0), now=NOW)
    assert not result.accepted
    assert any(reason.startswith("pressure_out_of_range") for reason in result.reasons)
    assert any(reason.startswith("negative_flow") for reason in result.reasons)


def test_rejects_non_finite_values() -> None:
    result = validate_reading(make_reading(flow_bbl=float("nan")), now=NOW)
    assert not result.accepted
    assert "non_finite:flow_bbl" in result.reasons


def test_rejects_future_timestamps_beyond_tolerance() -> None:
    future = NOW + timedelta(minutes=30)
    result = validate_reading(make_reading(ts=future), now=NOW)
    assert not result.accepted
    assert any(reason.startswith("timestamp_in_future") for reason in result.reasons)


def test_rejects_bad_well_id() -> None:
    result = validate_reading(make_reading(well_id="bad id!"), now=NOW)
    assert not result.accepted
    assert any(reason.startswith("invalid_well_id") for reason in result.reasons)


def test_naive_timestamp_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Reading("W-0001", datetime(2026, 6, 1, 12), 1_800.0, 148.0, 640.0)


def test_rules_are_configurable_per_basin() -> None:
    rules = RuleSet(max_pressure_psi=1_000.0)
    result = validate_reading(make_reading(pressure_psi=1_800.0), rules=rules, now=NOW)
    assert not result.accepted


def test_deduplicator_bounds_memory_and_keeps_recent_keys() -> None:
    deduper = Deduplicator(maxlen=2)
    assert deduper.seen("a") is False
    assert deduper.seen("b") is False
    assert deduper.seen("a") is True  # recent hit stays
    assert deduper.seen("c") is False  # evicts "b"
    assert len(deduper) == 2
    assert deduper.seen("b") is False  # b was evicted, so it looks new again


def test_reading_key_is_stable_to_the_second() -> None:
    first = make_reading(ts=NOW.replace(microsecond=123))
    second = make_reading(ts=NOW.replace(microsecond=987))
    assert reading_key(first) == reading_key(second)


def test_split_valid_partitions_batch() -> None:
    good = make_reading()
    bad = make_reading(pressure_psi=-1.0)
    accepted, rejected = split_valid([good, bad], now=NOW)
    assert accepted == [good]
    assert rejected[0][0] == bad and rejected[0][1]
