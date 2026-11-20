"""Environment-driven configuration.

Defaults are safe for local development; every value that differs between
laptops, CI and production is an environment variable, not a code edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True, slots=True)
class Settings:
    kafka_bootstrap: str = "localhost:9092"
    topic: str = "well.telemetry"
    consumer_group: str = "wellstream-processor"
    postgres_dsn: str = "postgresql://wellstream:wellstream@localhost:5432/wellstream"
    window_seconds: int = 300
    allowed_lateness_seconds: int = 90
    cadence_seconds: int = 60
    model_path: str = "models/anomaly_logit.json"
    batch_size: int = 200

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP", cls.kafka_bootstrap),
            topic=os.getenv("KAFKA_TOPIC", cls.topic),
            consumer_group=os.getenv("KAFKA_GROUP_ID", cls.consumer_group),
            postgres_dsn=os.getenv("POSTGRES_DSN", cls.postgres_dsn),
            window_seconds=_int("WINDOW_SECONDS", cls.window_seconds),
            allowed_lateness_seconds=_int("ALLOWED_LATENESS_SECONDS", cls.allowed_lateness_seconds),
            cadence_seconds=_int("CADENCE_SECONDS", cls.cadence_seconds),
            model_path=os.getenv("MODEL_PATH", cls.model_path),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
        )
