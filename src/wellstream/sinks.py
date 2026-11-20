"""Warehouse sinks.

Two implementations behind ``WarehouseSink``:

* ``PostgresSink`` - production target, upserts so at-least-once delivery and
  restarts are harmless.
* ``SqliteSink`` - the same contract backed by the standard library, which is
  what the offline demo and the integration tests run against. No server, no
  container, no excuse not to test the SQL path.

Writes are batched: one round trip per batch instead of per window is the
difference between 200 and 20,000 commits a second.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

from .models import AnomalySignal, WindowAggregate
from .schema import POSTGRES_DDL, SQLITE_DDL, split_statements

WINDOW_COLUMNS = (
    "well_id",
    "window_start",
    "window_end",
    "sample_count",
    "expected_samples",
    "avg_flow_bbl",
    "total_volume_bbl",
    "flow_cv",
    "avg_pressure_psi",
    "max_pressure_psi",
    "avg_temp_f",
)


@runtime_checkable
class WarehouseSink(Protocol):
    def ensure_schema(self) -> None: ...

    def write_windows(self, windows: Sequence[WindowAggregate]) -> int: ...

    def write_signals(self, signals: Sequence[AnomalySignal]) -> int: ...

    def close(self) -> None: ...


def _window_tuple(window: WindowAggregate) -> tuple[object, ...]:
    values = [getattr(window, column) for column in WINDOW_COLUMNS]
    return tuple(str(value) if isinstance(value, datetime) else value for value in values)


class PostgresSink:
    """Production sink with idempotent upserts."""

    def __init__(self, dsn: str, batch_size: int = 500) -> None:
        self.dsn = dsn
        self.batch_size = batch_size
        self._conn = None

    def _connect(self):
        if self._conn is None:
            import psycopg  # lazy: not needed for the demo or unit tests

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def ensure_schema(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            for statement in split_statements(POSTGRES_DDL):
                cur.execute(statement)

    def write_windows(self, windows: Sequence[WindowAggregate]) -> int:
        if not windows:
            return 0
        conn = self._connect()
        placeholders = ", ".join(["%s"] * len(WINDOW_COLUMNS))
        updates = ", ".join(
            f"{column} = EXCLUDED.{column}" for column in WINDOW_COLUMNS[3:]
        )
        sql = (
            f"INSERT INTO production_windows ({', '.join(WINDOW_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (well_id, window_start) DO UPDATE SET {updates}"
        )
        with conn.cursor() as cur:
            cur.executemany(sql, [_window_tuple(window) for window in windows])
        return len(windows)

    def write_signals(self, signals: Sequence[AnomalySignal]) -> int:
        if not signals:
            return 0
        conn = self._connect()
        sql = (
            "INSERT INTO anomaly_signals "
            "(well_id, window_start, score, decision, model_version, reasons) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
            "ON CONFLICT (well_id, window_start, model_version) DO UPDATE "
            "SET score = EXCLUDED.score, "
            "    decision = EXCLUDED.decision, "
            "    reasons = EXCLUDED.reasons"
        )
        rows = [
            (
                signal.well_id,
                signal.window_start,
                signal.score,
                signal.decision,
                signal.model_version,
                json.dumps(list(signal.reasons)),
            )
            for signal in signals
        ]
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class SqliteSink:
    """Portable sink used by the demo, benchmarks and integration tests."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode = WAL")

    def ensure_schema(self) -> None:
        self._conn.executescript(SQLITE_DDL)
        self._conn.commit()

    def write_windows(self, windows: Sequence[WindowAggregate]) -> int:
        if not windows:
            return 0
        placeholders = ", ".join(["?"] * len(WINDOW_COLUMNS))
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in WINDOW_COLUMNS[3:]
        )
        sql = (
            f"INSERT INTO production_windows ({', '.join(WINDOW_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (well_id, window_start) DO UPDATE SET {updates}"
        )
        self._conn.executemany(sql, [_window_tuple(window) for window in windows])
        return len(windows)

    def write_signals(self, signals: Sequence[AnomalySignal]) -> int:
        if not signals:
            return 0
        sql = (
            "INSERT INTO anomaly_signals "
            "(well_id, window_start, score, decision, model_version, reasons) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (well_id, window_start, model_version) DO UPDATE SET "
            "score = excluded.score, decision = excluded.decision, reasons = excluded.reasons"
        )
        self._conn.executemany(
            sql,
            [
                (
                    signal.well_id,
                    signal.window_start.isoformat(),
                    signal.score,
                    signal.decision,
                    signal.model_version,
                    json.dumps(list(signal.reasons)),
                )
                for signal in signals
            ],
        )
        return len(signals)

    def commit(self) -> None:
        self._conn.commit()

    def query(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        return list(self._conn.execute(sql, params))

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()


class MultiSink:
    """Fan-out sink: write to Postgres *and* keep a local audit copy."""

    def __init__(self, sinks: Iterable[WarehouseSink]) -> None:
        self.sinks = list(sinks)

    def ensure_schema(self) -> None:
        for sink in self.sinks:
            sink.ensure_schema()

    def write_windows(self, windows: Sequence[WindowAggregate]) -> int:
        return max((sink.write_windows(windows) for sink in self.sinks), default=0)

    def write_signals(self, signals: Sequence[AnomalySignal]) -> int:
        return max((sink.write_signals(signals) for sink in self.sinks), default=0)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def window_record(window: WindowAggregate) -> dict[str, object]:
    record = {key: value for key, value in asdict(window).items()}
    record["coverage_ratio"] = round(window.coverage_ratio, 4)
    return record
