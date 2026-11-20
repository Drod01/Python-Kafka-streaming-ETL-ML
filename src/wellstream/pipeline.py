"""The streaming operator: validate, window, featurise, score, persist."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator, Mapping, Sequence

from .broker import InMemoryBroker, MessageBroker
from .features import FeatureBuilder, explain
from .model import LogisticModel
from .models import AnomalySignal, Reading, WindowAggregate
from .sinks import WarehouseSink
from .validation import Deduplicator, RuleSet, reading_key, validate_reading
from .windows import TumblingWindower

FLAG_BATCH_SIZE = 200


@dataclass
class PipelineStats:
    """Counters that become the Prometheus/CloudWatch metrics in production."""

    consumed: int = 0
    malformed: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    late_dropped: int = 0
    windows: int = 0
    scored: int = 0
    flagged: int = 0
    written: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "consumed": self.consumed,
            "malformed": self.malformed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "late_dropped": self.late_dropped,
            "windows": self.windows,
            "scored": self.scored,
            "flagged": self.flagged,
            "written": self.written,
            "top_rejection_reasons": dict(self.rejection_reasons.most_common(5)),
        }


class StreamingPipeline:
    """Composable operator with injected collaborators.

    Every dependency (broker, sink, model, clock, windower) is passed in, which
    is what lets the same class run against Kafka/Postgres in production, an
    in-memory broker plus SQLite in CI, and a replay of a parquet/JSONL backfill
    locally with zero code changes.
    """

    def __init__(
        self,
        sink: WarehouseSink,
        scorer: LogisticModel | None = None,
        rules: RuleSet | None = None,
        windower: TumblingWindower | None = None,
        features: FeatureBuilder | None = None,
        dedupe: Deduplicator | None = None,
        now: datetime | None = None,
        flush_every: int = FLAG_BATCH_SIZE,
    ) -> None:
        self.sink = sink
        self.scorer = scorer
        self.rules = rules or RuleSet()
        self.windower = windower or TumblingWindower()
        self.features = features or FeatureBuilder()
        self.dedupe = dedupe or Deduplicator()
        self.fixed_now = now
        self.flush_every = max(1, flush_every)
        self.stats = PipelineStats()
        self._pending_windows: list[WindowAggregate] = []
        self._pending_signals: list[AnomalySignal] = []

    # ------------------------------- ingest ------------------------------- #
    def consume(self, event: Mapping[str, object]) -> None:
        """Process a single raw event. Never raises on bad input."""
        self.stats.consumed += 1
        try:
            reading = Reading.from_event(event)
        except ValueError:
            self.stats.malformed += 1
            self.stats.rejection_reasons["malformed_event"] += 1
            return

        now = self.fixed_now or datetime.now(timezone.utc)
        result = validate_reading(reading, rules=self.rules, now=now)
        if not result.accepted:
            self.stats.rejected += 1
            for reason in result.reasons:
                self.stats.rejection_reasons[reason.split(":")[0]] += 1
            return

        if self.dedupe.seen(reading_key(reading)):
            self.stats.duplicates += 1
            return

        self.stats.accepted += 1
        for window in self.windower.add(reading):
            self._handle_window(window)

    def _handle_window(self, window: WindowAggregate) -> None:
        self.stats.windows += 1
        self._pending_windows.append(window)

        vector = self.features.push(window)
        if vector is None or self.scorer is None:
            self._maybe_flush()
            return

        score, decision = self.scorer.score_features(vector)
        self.stats.scored += 1
        reasons = explain(vector)
        if decision == "flag":
            self.stats.flagged += 1
        self._pending_signals.append(
            AnomalySignal(
                well_id=window.well_id,
                window_start=window.window_start,
                score=round(score, 6),
                decision=decision,
                model_version=self.scorer.model_version,
                reasons=reasons,
            )
        )
        self._maybe_flush()

    # -------------------------------- flush -------------------------------- #
    def _maybe_flush(self) -> None:
        if len(self._pending_windows) >= self.flush_every or len(self._pending_signals) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        windows = self._pending_windows
        signals = self._pending_signals
        self._pending_windows = []
        self._pending_signals = []
        # Windows first: the signal table is keyed by the window it describes.
        if windows:
            self.sink.write_windows(windows)
            self.stats.written += len(windows)
        if signals:
            self.sink.write_signals(signals)

    def finish(self) -> PipelineStats:
        """Drain the watermark and every buffer. Call once at end-of-stream."""
        for window in self.windower.flush():
            self._handle_window(window)
        self.flush()
        self.stats.late_dropped = self.windower.late_dropped
        return self.stats

    # ------------------------------- runners ------------------------------- #
    def run(self, events: Iterable[Mapping[str, object]]) -> PipelineStats:
        self.sink.ensure_schema()
        for event in events:
            self.consume(event)
        return self.finish()

    def run_stream(self, broker: MessageBroker, topic: str) -> Iterator[PipelineStats]:
        """Long-running consumer; yields stats after every flushed batch."""
        self.sink.ensure_schema()
        for event in broker.subscribe(topic):
            self.consume(event)
            if not self._pending_windows and not self._pending_signals:
                yield self.stats
        yield self.finish()

    def run_memory(self, events: Sequence[Mapping[str, object]], topic: str = "well.telemetry") -> PipelineStats:
        """In-process smoke path used by the demo command and the tests."""
        broker = InMemoryBroker()
        for event in events:
            broker.publish(topic, event)
        broker.close()
        self.sink.ensure_schema()
        stats = self.stats
        for event in broker.subscribe(topic):
            self.consume(event)
        stats.late_dropped = self.windower.late_dropped
        return self.finish()
