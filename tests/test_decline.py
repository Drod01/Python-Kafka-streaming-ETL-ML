from __future__ import annotations

import math

import pytest
from wellstream.decline import fit_exponential_decline


def arps_series(qi: float, di: float, days: int) -> list[float]:
    import math

    return [qi * math.exp(-di * day) for day in range(days)]


def test_recovers_known_decline_parameters() -> None:
    fit = fit_exponential_decline(arps_series(1_000.0, 0.01, 20), interval_days=1.0)
    assert fit is not None
    assert fit.qi_bpd == pytest.approx(1_000.0, rel=0.01)
    assert fit.di_per_day == pytest.approx(0.01, rel=0.02)
    assert fit.r_squared > 0.99
    assert fit.points == 20


def test_prediction_matches_the_series() -> None:
    fit = fit_exponential_decline(arps_series(800.0, 0.02, 10), interval_days=1.0)
    assert fit is not None
    assert fit.predict(5) == pytest.approx(800.0 * 2.718281828 ** (-0.02 * 5), rel=0.01)


def test_deviation_is_positive_when_under_producing() -> None:
    fit = fit_exponential_decline(arps_series(1_000.0, 0.01, 10), interval_days=1.0)
    assert fit is not None
    assert fit.deviation(observed_bpd=fit.predict(10) * 0.5, day_offset=10) == pytest.approx(0.5, rel=0.01)


def test_flat_series_has_no_decline() -> None:
    fit = fit_exponential_decline([500.0] * 8, interval_days=1.0)
    assert fit is not None
    assert fit.di_per_day == pytest.approx(0.0, abs=1e-9)


def test_insufficient_data_returns_none() -> None:
    assert fit_exponential_decline([100.0, 90.0, 80.0], interval_days=1.0) is None
    assert fit_exponential_decline([], interval_days=1.0) is None


def test_zero_production_points_are_ignored() -> None:
    assert fit_exponential_decline([100.0, 0.0, 0.0, 0.0], interval_days=1.0) is None


def test_window_aligned_intervals_are_supported() -> None:
    # Five-minute windows: interval_days is 5/1440, and Di must scale with it.
    interval = 5 / 1440
    series = [1_200.0 * (0.999) ** step for step in range(12)]
    fit = fit_exponential_decline(series, interval_days=interval)
    assert fit is not None
    assert fit.di_per_day == pytest.approx(-math.log(0.999) / interval, rel=0.05)


def test_window_aligned_negative_slope_is_clamped_to_zero() -> None:
    # A rising rate has no physical decline; the model must not invent one.
    fit = fit_exponential_decline([100.0, 110.0, 121.0, 133.0, 146.0], interval_days=1.0)
    assert fit is not None
    assert fit.di_per_day == pytest.approx(0.0, abs=1e-9)
