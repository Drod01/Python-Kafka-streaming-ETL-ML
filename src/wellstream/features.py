"""Feature engineering: turn a closed window into model inputs.

Every feature is expressed as a *relative* deviation from that well's own recent
baseline. Absolute values do not transfer between wells - a 400 bbl/d well and a
40 bbl/d well look nothing alike in raw terms - but both look the same when a
window produces 30% below its trailing average with a pressure spike.

Features are computed *before* the baseline is updated, so the window being
scored never contaminates its own reference period.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque, Mapping

from .decline import fit_exponential_decline
from .models import FeatureVector, WindowAggregate

FEATURES: tuple[str, ...] = (
    "flow_drop_ratio",
    "pressure_z",
    "flow_cv",
    "coverage_gap",
    "temp_deviation",
    "decline_deviation",
)

FEATURE_DESCRIPTIONS: Mapping[str, str] = {
    "flow_drop_ratio": "window rate vs trailing baseline (positive = under-producing)",
    "pressure_z": "pressure z-score against trailing baseline",
    "flow_cv": "within-window coefficient of variation (instability)",
    "coverage_gap": "share of expected samples missing",
    "temp_deviation": "temperature drift from baseline, in 10F units",
    "decline_deviation": "miss versus fitted Arps decline curve",
}


@dataclass(slots=True)
class _WellHistory:
    flows: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    pressures: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    temps: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    window_seconds: int = 300


class FeatureBuilder:
    """Stateful per-well feature builder.

    ``baseline_windows`` controls the trailing reference period (6 windows of
    5 minutes = 30 minutes by default). Fewer than ``min_baseline`` closed
    windows means features are withheld rather than scored against noise.
    """

    def __init__(self, baseline_windows: int = 12, min_baseline: int = 4) -> None:
        if min_baseline < 2:
            raise ValueError("min_baseline must be >= 2 to compute a spread")
        self.baseline_windows = baseline_windows
        self.min_baseline = min_baseline
        self._history: dict[str, _WellHistory] = defaultdict(_WellHistory)

    def _history_for(self, agg: WindowAggregate) -> _WellHistory:
        hist = self._history[agg.well_id]
        if not hist.flows:
            hist.window_seconds = int(agg.window_hours * 3600)
        return hist

    def push(self, agg: WindowAggregate) -> FeatureVector | None:
        hist = self._history_for(agg)
        if len(hist.flows) < self.min_baseline:
            self._remember(hist, agg)
            return None

        baseline_flow = mean(hist.flows)
        baseline_pressure = mean(hist.pressures)
        pressure_spread = pstdev(hist.pressures) if len(hist.pressures) > 1 else 0.0
        baseline_temp = mean(hist.temps)

        flow_drop = ((baseline_flow - agg.avg_flow_bbl) / baseline_flow) if baseline_flow > 0 else 0.0
        if pressure_spread > 1e-9:
            pressure_z = (agg.avg_pressure_psi - baseline_pressure) / pressure_spread
        else:
            pressure_z = 0.0

        interval_days = max(agg.window_hours / 24.0, 1e-6)
        fit = fit_exponential_decline(list(hist.flows), interval_days=interval_days)
        decline_dev = fit.deviation(agg.avg_flow_bbl, day_offset=interval_days) if fit else 0.0

        values = {
            "flow_drop_ratio": _clip(flow_drop, -3.0, 3.0),
            "pressure_z": _clip(pressure_z, -6.0, 6.0),
            "flow_cv": _clip(agg.flow_cv, 0.0, 2.0),
            "coverage_gap": _clip(1.0 - agg.coverage_ratio, 0.0, 1.0),
            "temp_deviation": _clip((agg.avg_temp_f - baseline_temp) / 10.0, -6.0, 6.0),
            "decline_deviation": _clip(decline_dev, -3.0, 3.0),
        }
        self._remember(hist, agg)
        return FeatureVector(well_id=agg.well_id, window_start=agg.window_start, values=values)

    @staticmethod
    def _remember(hist: _WellHistory, agg: WindowAggregate) -> None:
        hist.flows.append(agg.avg_flow_bbl)
        hist.pressures.append(agg.avg_pressure_psi)
        hist.temps.append(agg.avg_temp_f)

    def warmup_complete(self, well_id: str) -> bool:
        return len(self._history[well_id].flows) >= self.min_baseline


def _clip(value: float, low: float, high: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(low, min(high, value))


def explain(vector: FeatureVector, top_n: int = 3) -> tuple[str, ...]:
    """Human-readable drivers, ordered by magnitude.

    Used for the alert payload so an operator sees "flow 42% below baseline"
    instead of an opaque probability.
    """
    ranked = sorted(vector.values.items(), key=lambda item: abs(item[1]), reverse=True)
    reasons: list[str] = []
    for name, value in ranked[:top_n]:
        if abs(value) < 0.25:
            continue
        reasons.append(f"{name}={value:+.2f} ({FEATURE_DESCRIPTIONS[name]})")
    return tuple(reasons)


def build_features(
    aggregates: list[WindowAggregate],
    baseline_windows: int = 12,
    min_baseline: int = 4,
) -> list[FeatureVector]:
    """Batch helper mirroring the streaming operator's outputs."""
    builder = FeatureBuilder(baseline_windows=baseline_windows, min_baseline=min_baseline)
    out: list[FeatureVector] = []
    for agg in sorted(aggregates, key=lambda a: (a.window_start, a.well_id)):
        vector = builder.push(agg)
        if vector is not None:
            out.append(vector)
    return out


def features_frame(vectors: list[FeatureVector]) -> list[dict[str, float]]:
    return [dict(v.values) for v in vectors]


__all__ = [
    "FEATURES",
    "FEATURE_DESCRIPTIONS",
    "FeatureBuilder",
    "build_features",
    "explain",
    "features_frame",
]
