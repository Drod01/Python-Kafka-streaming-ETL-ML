"""Immutable domain records shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping


def to_utc(ts: datetime) -> datetime:
    """Normalise a timestamp to UTC, rejecting naive input.

    Naive datetimes are a classic source of stream-processing bugs (windows
    silently shifting by the host timezone), so they are rejected loudly.
    """
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class Reading:
    """One SCADA/telemetry sample for a well."""

    well_id: str
    ts: datetime
    pressure_psi: float
    temp_f: float
    flow_bbl: float
    source: str = "scada"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", to_utc(self.ts))

    @property
    def volume_bbl(self) -> float:
        """Instantaneous rate (bbl/day) is carried in ``flow_bbl``."""
        return self.flow_bbl

    @classmethod
    def from_event(cls, event: Mapping[str, object]) -> "Reading":
        """Decode the JSON event contract used on the Kafka topic."""
        try:
            return cls(
                well_id=str(event["well_id"]),
                ts=datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00")),
                pressure_psi=float(event["pressure_psi"]),
                temp_f=float(event["temp_f"]),
                flow_bbl=float(event["flow_bbl"]),
                source=str(event.get("source", "scada")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed reading event: {exc}") from exc

    def to_event(self) -> dict[str, object]:
        return {
            "well_id": self.well_id,
            "ts": self.ts.isoformat(),
            "pressure_psi": self.pressure_psi,
            "temp_f": self.temp_f,
            "flow_bbl": self.flow_bbl,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class WindowAggregate:
    """A closed tumbling window for a single well."""

    well_id: str
    window_start: datetime
    window_end: datetime
    sample_count: int
    expected_samples: int
    avg_flow_bbl: float
    total_volume_bbl: float
    flow_cv: float
    avg_pressure_psi: float
    max_pressure_psi: float
    avg_temp_f: float

    @property
    def coverage_ratio(self) -> float:
        if self.expected_samples <= 0:
            return 0.0
        return min(1.0, self.sample_count / self.expected_samples)

    @property
    def window_hours(self) -> float:
        return (self.window_end - self.window_start).total_seconds() / 3600.0


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Model input derived from a window plus its trailing baseline."""

    well_id: str
    window_start: datetime
    values: Mapping[str, float]

    def as_list(self, order: tuple[str, ...]) -> list[float]:
        return [self.values[name] for name in order]


@dataclass(frozen=True, slots=True)
class AnomalySignal:
    """Scored window, persisted for the alerting/review workflow."""

    well_id: str
    window_start: datetime
    score: float
    decision: str
    model_version: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
