"""Wellstream: event-driven well telemetry pipeline with an anomaly model."""

from __future__ import annotations

__version__ = "0.1.0"

from .models import AnomalySignal, FeatureVector, Reading, WindowAggregate  # noqa: E402
from .pipeline import PipelineStats, StreamingPipeline  # noqa: E402
from .windows import TumblingWindower  # noqa: E402

__all__ = [
    "__version__",
    "AnomalySignal",
    "FeatureVector",
    "Reading",
    "WindowAggregate",
    "PipelineStats",
    "StreamingPipeline",
    "TumblingWindower",
]
