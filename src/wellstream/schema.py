"""Warehouse schema, kept in code so it can be printed for any environment.

``wellstream ddl`` emits this script, so the schema used by tests, the local
compose stack and the deployed warehouse is generated from one source instead of
being copy-pasted into three README snippets.
"""

from __future__ import annotations

POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS wells (
    well_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    basin          TEXT NOT NULL,
    county         TEXT,
    status         TEXT NOT NULL DEFAULT 'producing',
    spud_date      DATE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS production_windows (
    well_id          TEXT NOT NULL REFERENCES wells (well_id) ON DELETE CASCADE,
    window_start     TIMESTAMPTZ NOT NULL,
    window_end       TIMESTAMPTZ NOT NULL,
    sample_count     INTEGER NOT NULL,
    expected_samples INTEGER NOT NULL,
    avg_flow_bbl     DOUBLE PRECISION NOT NULL,
    total_volume_bbl DOUBLE PRECISION NOT NULL,
    flow_cv          DOUBLE PRECISION NOT NULL,
    avg_pressure_psi DOUBLE PRECISION NOT NULL,
    max_pressure_psi DOUBLE PRECISION NOT NULL,
    avg_temp_f       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (well_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_production_windows_start
    ON production_windows (window_start DESC);

CREATE TABLE IF NOT EXISTS anomaly_signals (
    well_id       TEXT NOT NULL,
    window_start  TIMESTAMPTZ NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    decision      TEXT NOT NULL CHECK (decision IN ('monitor', 'flag')),
    model_version TEXT NOT NULL,
    reasons       JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (well_id, window_start, model_version)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_signals_flags
    ON anomaly_signals (window_start DESC)
    WHERE decision = 'flag';

CREATE OR REPLACE VIEW well_daily_rollup AS
SELECT
    well_id,
    date_trunc('day', window_start)                     AS production_day,
    sum(total_volume_bbl)                               AS volume_bbl,
    avg(avg_flow_bbl)                                   AS avg_flow_bpd,
    max(max_pressure_psi)                               AS peak_pressure_psi,
    count(*)                                            AS windows,
    count(*) FILTER (WHERE decision = 'flag')           AS flagged_windows
FROM production_windows
LEFT JOIN anomaly_signals USING (well_id, window_start)
GROUP BY well_id, date_trunc('day', window_start);
""".strip()

SINGLESTORE_REFERENCE = """
-- SingleStore (MemSQL) landing table for the same stream. Columnstore with a
-- sort key gives sub-second scans over billions of rows for KPI queries, while
-- the shard key keeps all rows for a well on one partition.
CREATE TABLE IF NOT EXISTS production_windows (
    well_id          VARCHAR(32) NOT NULL,
    window_start     DATETIME NOT NULL,
    window_end       DATETIME NOT NULL,
    sample_count     INT NOT NULL,
    expected_samples INT NOT NULL,
    avg_flow_bbl     DOUBLE NOT NULL,
    total_volume_bbl DOUBLE NOT NULL,
    flow_cv          DOUBLE NOT NULL,
    avg_pressure_psi DOUBLE NOT NULL,
    max_pressure_psi DOUBLE NOT NULL,
    avg_temp_f       DOUBLE NOT NULL,
    SHARD KEY (well_id),
    SORT KEY (window_start),
    PRIMARY KEY (well_id, window_start)
);
""".strip()

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS wells (
    well_id   TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    basin     TEXT NOT NULL,
    county    TEXT,
    status    TEXT NOT NULL DEFAULT 'producing',
    spud_date TEXT
);

CREATE TABLE IF NOT EXISTS production_windows (
    well_id          TEXT NOT NULL,
    window_start     TEXT NOT NULL,
    window_end       TEXT NOT NULL,
    sample_count     INTEGER NOT NULL,
    expected_samples INTEGER NOT NULL,
    avg_flow_bbl     REAL NOT NULL,
    total_volume_bbl REAL NOT NULL,
    flow_cv          REAL NOT NULL,
    avg_pressure_psi REAL NOT NULL,
    max_pressure_psi REAL NOT NULL,
    avg_temp_f       REAL NOT NULL,
    PRIMARY KEY (well_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_production_windows_start
    ON production_windows (window_start);

CREATE TABLE IF NOT EXISTS anomaly_signals (
    well_id       TEXT NOT NULL,
    window_start  TEXT NOT NULL,
    score         REAL NOT NULL,
    decision      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    reasons       TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (well_id, window_start, model_version)
);
""".strip()


def split_statements(script: str) -> list[str]:
    """Split a DDL script into executable statements.

    Safe here because none of the statements contain a semicolon inside a string
    literal or a ``$$`` block; this is deliberately not a general SQL parser.
    """
    return [statement.strip() for statement in script.split(";") if statement.strip()]
