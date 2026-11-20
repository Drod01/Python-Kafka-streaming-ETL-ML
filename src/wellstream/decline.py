"""Arps exponential-decline curve fitting.

A decline curve is the cheapest useful *math model* in production engineering:
fit ``q(t) = qi * exp(-Di * t)`` to a well's recent rate history and you get an
expected rate for the next interval. Comparing the expected rate with the
observed one separates "the well is behaving" from "the well fell off its type
curve" - a far stronger anomaly signal than a raw threshold.

The fit is a least-squares line through ``ln(q)`` versus time, which makes it
deterministic, explainable and free of training data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class DeclineFit:
    """Fitted exponential decline curve."""

    qi_bpd: float
    di_per_day: float
    r_squared: float
    rmse_bpd: float
    points: int

    def predict(self, day_offset: float) -> float:
        return self.qi_bpd * math.exp(-self.di_per_day * max(0.0, day_offset))

    def deviation(self, observed_bpd: float, day_offset: float) -> float:
        """Relative miss versus the curve; positive means under-producing."""
        expected = self.predict(day_offset)
        if expected <= 0:
            return 0.0
        return (expected - observed_bpd) / expected


MIN_POINTS = 4


def fit_exponential_decline(
    rates_bpd: Sequence[float],
    interval_days: float = 1.0,
) -> DeclineFit | None:
    """Fit ``qi``/``Di`` to a rate series sampled at ``interval_days``.

    Returns ``None`` when there is not enough signal to fit: fewer than four
    positive rates, zero variance in time, or a flat series (which would imply a
    zero decline rate and an unstable extrapolation).
    """
    pts = [(i * interval_days, float(q)) for i, q in enumerate(rates_bpd) if q > 0]
    if len(pts) < MIN_POINTS or interval_days <= 0:
        return None

    xs = [t for t, _ in pts]
    ys = [math.log(q) for _, q in pts]
    n = len(pts)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    qi = math.exp(intercept)
    di = -slope
    if not math.isfinite(qi) or qi <= 0:
        return None
    # A negative Di means the well is building rate; treat it as no decline so
    # the model cannot claim the well is "expected" to produce more than seen.
    di = max(0.0, di)

    preds = [qi * math.exp(-di * x) for x in xs]
    ss_res = sum((q - p) ** 2 for q, p in zip([q for _, q in pts], preds))
    ss_tot = sum((q - (sum(q for _, q in pts) / n)) ** 2 for q, _ in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / n)
    return DeclineFit(qi_bpd=qi, di_per_day=di, r_squared=r2, rmse_bpd=rmse, points=n)
