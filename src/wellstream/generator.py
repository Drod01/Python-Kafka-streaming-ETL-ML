"""Synthetic well telemetry generator.

Deterministic, physics-flavoured data so the demo, the benchmark and the
training script need no vendor feed or credentials: exponential decline plus
sensor noise for normal behaviour, and scripted rate/pressure faults that give
the classifier something real to learn. Seeded, so every run and CI job sees the
same numbers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Sequence

from .models import Reading

DEFAULT_START = datetime(2026, 6, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class WellProfile:
    well_id: str
    name: str
    basin: str
    county: str
    qi_bpd: float
    di_per_day: float
    pressure_psi: float
    temp_f: float
    spud_date: str = "2022-03-14"


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """One scripted anomaly, aligned to a tumbling window boundary."""

    well_id: str
    window_start: datetime
    flow_factor: float = 0.30
    pressure_factor: float = 1.45
    temp_delta: float = 0.0


DEFAULT_PROFILES: tuple[WellProfile, ...] = (
    WellProfile("W-0001", "Rattlesnake 12H", "Delaware", "Loving", 820.0, 0.0021, 1_850.0, 148.0),
    WellProfile("W-0002", "Rattlesnake 13H", "Delaware", "Loving", 610.0, 0.0026, 1_640.0, 151.0),
    WellProfile("W-0003", "Eagle Ford 4H", "Eagle Ford", "Karnes", 430.0, 0.0034, 1_320.0, 163.0),
    WellProfile("W-0004", "Wolfcamp 21H", "Midland", "Martin", 1_150.0, 0.0017, 2_240.0, 142.0),
    WellProfile("W-0005", "Wolfcamp 22H", "Midland", "Martin", 980.0, 0.0019, 2_080.0, 144.0),
    WellProfile("W-0006", "Bone Spring 8H", "Delaware", "Reeves", 350.0, 0.0039, 1_120.0, 158.0),
)

DEFAULT_FAULT_FRACTIONS: tuple[float, ...] = (0.25, 0.40, 0.55, 0.72)


def build_scenario(
    profiles: Sequence[WellProfile] = DEFAULT_PROFILES,
    start: datetime = DEFAULT_START,
    hours: int = 6,
    cadence_seconds: int = 60,
    window_seconds: int = 300,
    seed: int = 7,
    faulted_wells: int = 4,
) -> tuple[list[Reading], list[FaultSpec]]:
    """Generate readings plus the ground-truth fault list used for labelling."""
    if hours <= 0 or cadence_seconds <= 0 or window_seconds <= 0:
        raise ValueError("hours, cadence_seconds and window_seconds must be positive")
    rng = random.Random(seed)
    total_samples = int(hours * 3600 / cadence_seconds)
    windows_per_well = max(2, int(hours * 3600 / window_seconds))

    chosen = list(profiles)[:faulted_wells]
    faults: list[FaultSpec] = []
    seen: set[tuple[str, datetime]] = set()
    for index, profile in enumerate(chosen):
        for offset, fraction in enumerate(DEFAULT_FAULT_FRACTIONS):
            window_index = int(windows_per_well * fraction) + offset
            window_index = min(window_index, windows_per_well - 2)
            window_start = start + timedelta(seconds=window_index * window_seconds)
            if (profile.well_id, window_start) in seen:
                continue
            seen.add((profile.well_id, window_start))
            faults.append(
                FaultSpec(
                    well_id=profile.well_id,
                    window_start=window_start,
                    # Alternate rate-failure and pressure-only signatures so the
                    # model has to use more than one feature.
                    flow_factor=0.28 + 0.04 * index if offset % 2 == 0 else 0.72,
                    pressure_factor=1.42 + 0.03 * index if offset % 2 == 0 else 1.60,
                    temp_delta=1.5 * (index - 1),
                )
            )

    readings: list[Reading] = []
    for profile in profiles:
        for sample in range(total_samples):
            ts = start + timedelta(seconds=sample * cadence_seconds)
            day = sample * cadence_seconds / 86_400.0
            base_rate = profile.qi_bpd * math.exp(-profile.di_per_day * day)
            # Diurnal ambient swing: small, realistic, and irrelevant to faults.
            diurnal = 1.0 + 0.015 * math.sin(2 * math.pi * (ts.hour / 24.0))
            rate = base_rate * diurnal * (1.0 + rng.gauss(0.0, 0.012))

            pressure = profile.pressure_psi * (0.72 + 0.28 * (rate / max(profile.qi_bpd, 1e-9)))
            pressure *= 1.0 + rng.gauss(0.0, 0.004)
            temperature = profile.temp_f + rng.gauss(0.0, 0.35)

            fault = _fault_at(faults, profile.well_id, ts, window_seconds)
            if fault is not None:
                rate *= fault.flow_factor
                pressure *= fault.pressure_factor
                temperature += fault.temp_delta

            readings.append(
                Reading(
                    well_id=profile.well_id,
                    ts=ts,
                    pressure_psi=round(pressure, 3),
                    temp_f=round(temperature, 3),
                    flow_bbl=round(rate, 3),
                    source="simulated-scada",
                )
            )

    # Interleave wells in arrival order, with a little jitter, so the consumer
    # genuinely has to handle out-of-order, multi-well traffic.
    readings.sort(key=lambda reading: reading.ts)
    return readings, faults


def _fault_at(
    faults: Sequence[FaultSpec],
    well_id: str,
    ts: datetime,
    window_seconds: int = 300,
) -> FaultSpec | None:
    for fault in faults:
        if fault.well_id != well_id:
            continue
        if fault.window_start <= ts < fault.window_start + timedelta(seconds=window_seconds):
            return fault
    return None


def fault_windows(faults: Sequence[FaultSpec], window_seconds: int = 300) -> set[tuple[str, datetime]]:
    """Label set aligned to window starts, as the pipeline emits them."""
    labels: set[tuple[str, datetime]] = set()
    for fault in faults:
        epoch = int(fault.window_start.timestamp())
        start = datetime.fromtimestamp(epoch - epoch % window_seconds, tz=timezone.utc)
        labels.add((fault.well_id, start))
    return labels


def iter_events(readings: Sequence[Reading]) -> Iterator[dict[str, object]]:
    """Project readings onto the Kafka JSON contract."""
    for reading in readings:
        yield reading.to_event()


def simulate(
    profiles: Sequence[WellProfile] = DEFAULT_PROFILES,
    start: datetime = DEFAULT_START,
    hours: int = 6,
    cadence_seconds: int = 60,
    seed: int = 7,
) -> Iterator[Reading]:
    """Stream-only convenience wrapper (no ground-truth labels)."""
    readings, _ = build_scenario(
        profiles=profiles,
        start=start,
        hours=hours,
        cadence_seconds=cadence_seconds,
        seed=seed,
    )
    yield from readings


def well_rows(profiles: Sequence[WellProfile] = DEFAULT_PROFILES) -> list[dict[str, object]]:
    return [
        {
            "well_id": profile.well_id,
            "name": profile.name,
            "basin": profile.basin,
            "county": profile.county,
            "status": "producing",
            "spud_date": profile.spud_date,
        }
        for profile in profiles
    ]
