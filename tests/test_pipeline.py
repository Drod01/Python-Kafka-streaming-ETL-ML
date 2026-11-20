from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from wellstream.broker import InMemoryBroker
from wellstream.generator import build_scenario, fault_windows, iter_events
from wellstream.models import Reading
from wellstream.pipeline import StreamingPipeline
from wellstream.windows import window_readings
from wellstream.sinks import SqliteSink, window_record
from wellstream.training import build_dataset, split_by_time, train

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def scenario():
    readings, faults = build_scenario(hours=2, seed=11)
    labels = fault_windows(faults)
    dataset = build_dataset(readings, labels)
    cutoff = BASE + timedelta(hours=1)
    train_set, test_set = split_by_time(dataset, cutoff)
    model, report = train(train_set, test_set)
    return readings, faults, labels, model, report


def test_replay_persists_every_window(scenario) -> None:
    readings, _, _, model, _ = scenario
    sink = SqliteSink(":memory:")
    pipeline = StreamingPipeline(sink=sink, scorer=model, flush_every=50)
    stats = pipeline.run(iter_events(readings))

    rows = sink.query("SELECT count(*) AS n FROM production_windows")
    signals = sink.query("SELECT count(*) AS n FROM anomaly_signals")
    assert stats.windows > 0
    assert rows[0]["n"] == stats.windows
    assert stats.written == stats.windows
    assert signals[0]["n"] == stats.scored
    assert stats.consumed == len(readings)
    sink.close()


def test_duplicate_delivery_is_idempotent(scenario) -> None:
    readings, _, _, model, _ = scenario
    sink = SqliteSink(":memory:")
    pipeline = StreamingPipeline(sink=sink, scorer=model)
    pipeline.sink.ensure_schema()
    event = readings[0].to_event()
    pipeline.consume(event)
    pipeline.consume(event)
    assert pipeline.stats.duplicates == 1
    assert pipeline.stats.accepted == 1
    sink.close()


def test_replaying_a_batch_twice_does_not_duplicate_rows(scenario) -> None:
    readings, _, _, model, _ = scenario
    sink = SqliteSink(":memory:")
    events = list(iter_events(readings))
    first = StreamingPipeline(sink=sink, scorer=model).run(events)
    second = StreamingPipeline(sink=sink, scorer=model).run(events)
    rows = sink.query("SELECT count(*) AS n FROM production_windows")[0]["n"]
    assert first.written == second.written == rows
    sink.close()


def test_malformed_events_are_counted_not_raised(scenario) -> None:
    _, _, _, model, _ = scenario
    sink = SqliteSink(":memory:")
    pipeline = StreamingPipeline(sink=sink, scorer=model)
    pipeline.sink.ensure_schema()
    pipeline.consume({"well_id": "W-0001"})
    assert pipeline.stats.malformed == 1
    assert pipeline.stats.consumed == 1
    sink.close()


def test_invalid_readings_are_rejected_with_reasons(scenario) -> None:
    readings, _, _, model, _ = scenario
    sink = SqliteSink(":memory:")
    pipeline = StreamingPipeline(sink=sink, scorer=model)
    pipeline.sink.ensure_schema()
    event = readings[0].to_event()
    event["pressure_psi"] = 99_000.0
    pipeline.consume(event)
    assert pipeline.stats.rejected == 1
    assert pipeline.stats.rejection_reasons["pressure_out_of_range"] == 1
    sink.close()


def test_fault_windows_score_higher_than_normal_windows(scenario) -> None:
    readings, faults, labels, model, _ = scenario
    sink = SqliteSink(":memory:")
    StreamingPipeline(sink=sink, scorer=model).run(iter_events(readings))
    rows = sink.query("SELECT well_id, window_start, score FROM anomaly_signals")
    assert rows

    label_keys = {(well_id, start.isoformat()) for well_id, start in labels}
    fault_scores = [row["score"] for row in rows if (row["well_id"], row["window_start"]) in label_keys]
    normal_scores = [row["score"] for row in rows if (row["well_id"], row["window_start"]) not in label_keys]
    assert fault_scores and normal_scores
    assert sum(fault_scores) / len(fault_scores) > sum(normal_scores) / len(normal_scores)
    sink.close()


def test_held_out_faults_are_detected(scenario) -> None:
    _, _, _, model, report = scenario
    assert report.test_metrics["roc_auc"] >= 0.9
    assert report.test_metrics["recall"] >= 0.5


def test_in_memory_broker_round_trip(scenario) -> None:
    readings, _, _, model, _ = scenario
    broker = InMemoryBroker()
    events = list(iter_events(readings[:120]))
    for event in events:
        broker.publish("well.telemetry", event)
    broker.close()
    sink = SqliteSink(":memory:")
    pipeline = StreamingPipeline(sink=sink, scorer=model)
    pipeline.sink.ensure_schema()
    for event in broker.subscribe("well.telemetry"):
        pipeline.consume(event)
    stats = pipeline.finish()
    assert stats.consumed == len(events)
    sink.close()


def test_window_record_includes_coverage() -> None:
    readings = [
        Reading("W-0001", BASE + timedelta(minutes=minute), 1_800.0, 148.0, 300.0)
        for minute in range(5)
    ]
    windows = window_readings(readings, allowed_lateness_seconds=0)
    record = window_record(windows[0])
    assert record["well_id"] == "W-0001"
    assert record["coverage_ratio"] == 1.0
    assert record["avg_flow_bbl"] == 300.0
