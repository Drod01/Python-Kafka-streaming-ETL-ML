"""Data-quality gates applied before anything reaches the warehouse.

Two classes of defence live here:

* ``RuleSet`` / ``validate_reading`` - field-level physics and sanity rules.
* ``Deduplicator`` - at-least-once delivery from Kafka means duplicate events
  are expected, not exceptional; the pipeline keeps a bounded memory of recent
  keys so retries are idempotent rather than doubled production.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .models import Reading

WELL_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,31}$")


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Thresholds for a producing well's telemetry.

    Defaults match a Permian/Delaware basin rod-pump installation; override per
    asset type (gas lift, ESP) from configuration rather than editing code.
    """

    min_pressure_psi: float = 0.0
    max_pressure_psi: float = 5_000.0
    min_temp_f: float = -40.0
    max_temp_f: float = 400.0
    max_flow_bpd: float = 50_000.0
    future_tolerance_seconds: int = 300


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:  # pragma: no cover - sugar
        return self.accepted


ACCEPT = ValidationResult(True)


def validate_reading(
    reading: Reading,
    rules: RuleSet | None = None,
    now: datetime | None = None,
) -> ValidationResult:
    """Return every rule violation, not just the first one.

    Surfacing all reasons at once makes the dead-letter payload far more useful
    when a site starts misbehaving: one glance shows whether it is a sensor
    calibration drift or a timestamp/timezone problem.
    """
    rules = rules or RuleSet()
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    if not WELL_ID_PATTERN.match(reading.well_id):
        reasons.append(f"invalid_well_id:{reading.well_id!r}")

    for name, value in (
        ("pressure_psi", reading.pressure_psi),
        ("temp_f", reading.temp_f),
        ("flow_bbl", reading.flow_bbl),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append(f"non_finite:{name}")

    if reasons:
        # Range checks on non-numeric input would raise; stop here.
        return ValidationResult(False, tuple(reasons))

    if not rules.min_pressure_psi <= reading.pressure_psi <= rules.max_pressure_psi:
        reasons.append(f"pressure_out_of_range:{reading.pressure_psi:.1f}")
    if not rules.min_temp_f <= reading.temp_f <= rules.max_temp_f:
        reasons.append(f"temp_out_of_range:{reading.temp_f:.1f}")
    if reading.flow_bbl < 0:
        reasons.append(f"negative_flow:{reading.flow_bbl:.1f}")
    elif reading.flow_bbl > rules.max_flow_bpd:
        reasons.append(f"flow_out_of_range:{reading.flow_bbl:.1f}")

    skew = (reading.ts - now).total_seconds()
    if skew > rules.future_tolerance_seconds:
        reasons.append(f"timestamp_in_future:{skew:.0f}s")

    return ValidationResult(not reasons, tuple(reasons))


class Deduplicator:
    """Bounded, insertion-ordered de-duplication of message keys.

    ``maxlen`` keeps memory flat on a long-running consumer while still catching
    the retry storms that at-least-once delivery actually produces (seconds to
    minutes apart), unlike an unbounded ``set`` which grows until the pod OOMs.
    """

    def __init__(self, maxlen: int = 200_000) -> None:
        self.maxlen = maxlen
        self._seen: OrderedDict[str, None] = OrderedDict()

    def seen(self, key: str) -> bool:
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = None
        if len(self._seen) > self.maxlen:
            self._seen.popitem(last=False)
        return False

    def __len__(self) -> int:
        return len(self._seen)


def dedupe(keys: Iterable[str], maxlen: int = 200_000) -> list[str]:
    """Helper used by tests and batch backfills."""
    deduper = Deduplicator(maxlen=maxlen)
    return [key for key in keys if not deduper.seen(key)]


def reading_key(reading: Reading) -> str:
    """Natural key for a sample: one well, one instant."""
    return f"{reading.well_id}|{reading.ts.replace(microsecond=0).isoformat()}"


def split_valid(
    readings: Iterable[Reading],
    rules: RuleSet | None = None,
    now: datetime | None = None,
    tolerance: timedelta = timedelta(seconds=0),
) -> tuple[list[Reading], list[tuple[Reading, tuple[str, ...]]]]:
    """Partition a batch into accepted rows and rejections with reasons."""
    del tolerance
    accepted: list[Reading] = []
    rejected: list[tuple[Reading, tuple[str, ...]]] = []
    for reading in readings:
        result = validate_reading(reading, rules=rules, now=now)
        if result.accepted:
            accepted.append(reading)
        else:
            rejected.append((reading, result.reasons))
    return accepted, rejected
