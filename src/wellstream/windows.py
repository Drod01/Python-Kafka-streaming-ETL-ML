"""Event-time tumbling windows with watermarking.

Production telemetry arrives out of order: a field gateway buffers for a few
seconds when the cellular link drops, then flushes a burst. A processing-time
window would split those bursts across the wrong minute, so windows are keyed by
event time and only close once the watermark (max seen event time minus the
allowed lateness) has passed their end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Iterable

from .models import Reading, WindowAggregate


def floor_to_window(ts: datetime, window_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - epoch % window_seconds, tz=timezone.utc)


@dataclass(slots=True)
class _WindowState:
    well_id: str
    window_start: datetime
    window_end: datetime
    count: int = 0
    flow_sum: float = 0.0
    flow_sq_sum: float = 0.0
    volume_sum: float = 0.0
    pressure_sum: float = 0.0
    pressure_max: float = 0.0
    temp_sum: float = 0.0

    def add(self, reading: Reading, cadence_seconds: float) -> None:
        self.count += 1
        self.flow_sum += reading.flow_bbl
        self.flow_sq_sum += reading.flow_bbl**2
        self.volume_sum += reading.flow_bbl * (cadence_seconds / 86_400.0)
        self.pressure_sum += reading.pressure_psi
        self.pressure_max = max(self.pressure_max, reading.pressure_psi)
        self.temp_sum += reading.temp_f

    def to_aggregate(self, expected_samples: int) -> WindowAggregate:
        # Sample standard deviation, guarded for the single-sample case.
        if self.count > 1:
            variance = max(0.0, (self.flow_sq_sum - self.count * (self.flow_sum / self.count) ** 2) / (self.count - 1))
            stdev = math.sqrt(variance)
        else:
            stdev = 0.0
        avg_flow = self.flow_sum / self.count if self.count else 0.0
        return WindowAggregate(
            well_id=self.well_id,
            window_start=self.window_start,
            window_end=self.window_end,
            sample_count=self.count,
            expected_samples=expected_samples,
            avg_flow_bbl=avg_flow,
            total_volume_bbl=self.volume_sum,
            flow_cv=(stdev / avg_flow) if avg_flow > 0 else 0.0,
            avg_pressure_psi=self.pressure_sum / self.count if self.count else 0.0,
            max_pressure_psi=self.pressure_max,
            avg_temp_f=self.temp_sum / self.count if self.count else 0.0,
        )


class TumblingWindower:
    """Keyed tumbling window operator.

    Parameters
    ----------
    window_seconds:
        Window size. Five minutes is the operational default for well telemetry.
    allowed_lateness_seconds:
        How long to keep a window open for stragglers. Larger values improve
        completeness at the cost of alert latency.
    expected_cadence_seconds:
        Nominal sensor cadence, used to compute the coverage ratio so a window
        that lost half its samples is not mistaken for a real production drop.
    """

    def __init__(
        self,
        window_seconds: int = 300,
        allowed_lateness_seconds: int = 90,
        expected_cadence_seconds: int = 60,
    ) -> None:
        if window_seconds <= 0 or allowed_lateness_seconds < 0 or expected_cadence_seconds <= 0:
            raise ValueError("invalid window configuration")
        self.window_seconds = window_seconds
        self.allowed_lateness = timedelta(seconds=allowed_lateness_seconds)
        self.cadence_seconds = float(expected_cadence_seconds)
        self._states: dict[tuple[str, datetime], _WindowState] = {}
        self._max_event_time: datetime | None = None
        self._watermark: datetime | None = None
        self.closed_windows = 0
        self.late_dropped = 0
        self.out_of_order = 0

    @property
    def watermark(self) -> datetime | None:
        return self._watermark

    @property
    def open_windows(self) -> int:
        return len(self._states)

    def _advance_watermark(self, ts: datetime) -> None:
        if self._max_event_time is None or ts > self._max_event_time:
            self._max_event_time = ts
        candidate = self._max_event_time - self.allowed_lateness
        if self._watermark is None or candidate > self._watermark:
            self._watermark = candidate

    def add(self, reading: Reading) -> list[WindowAggregate]:
        """Add one reading and return any windows that closed as a result."""
        if self._max_event_time is not None and reading.ts < self._max_event_time:
            self.out_of_order += 1

        start = floor_to_window(reading.ts, self.window_seconds)
        end = start + timedelta(seconds=self.window_seconds)
        late = self._watermark is not None and end <= self._watermark

        self._advance_watermark(reading.ts)
        if late:
            # The window's result has already been emitted; count it for
            # observability instead of silently corrupting an emitted aggregate.
            self.late_dropped += 1
            return []

        key = (reading.well_id, start)
        state = self._states.get(key)
        if state is None:
            state = _WindowState(reading.well_id, start, end)
            self._states[key] = state
        state.add(reading, self.cadence_seconds)
        return self._close_ready()

    def _close_ready(self) -> list[WindowAggregate]:
        if self._watermark is None:
            return []
        ready = [
            key
            for key, state in self._states.items()
            if state.window_end <= self._watermark
        ]
        if not ready:
            return []
        ready.sort(key=lambda key: (key[1], key[0]))
        expected = max(1, round(self.window_seconds / self.cadence_seconds))
        closed: list[WindowAggregate] = []
        for key in ready:
            state = self._states.pop(key)
            closed.append(state.to_aggregate(expected))
            self.closed_windows += 1
        return closed

    def flush(self) -> list[WindowAggregate]:
        """Close every open window. Used at end-of-stream for backfills."""
        keys = sorted(self._states, key=lambda key: (key[1], key[0]))
        expected = max(1, round(self.window_seconds / self.cadence_seconds))
        closed = [self._states.pop(key).to_aggregate(expected) for key in keys]
        self.closed_windows += len(closed)
        return closed


def window_readings(
    readings: Iterable[Reading],
    window_seconds: int = 300,
    allowed_lateness_seconds: int = 90,
    expected_cadence_seconds: int = 60,
) -> list[WindowAggregate]:
    """Convenience wrapper for batch replays."""
    windower = TumblingWindower(window_seconds, allowed_lateness_seconds, expected_cadence_seconds)
    out: list[WindowAggregate] = []
    for reading in readings:
        out.extend(windower.add(reading))
    out.extend(windower.flush())
    return out


def summarize_flows(aggregates: Iterable[WindowAggregate]) -> tuple[float, float]:
    """Return (mean, population stdev) of window flow rates."""
    flows = [agg.avg_flow_bbl for agg in aggregates]
    if not flows:
        return 0.0, 0.0
    return mean(flows), (pstdev(flows) if len(flows) > 1 else 0.0)
