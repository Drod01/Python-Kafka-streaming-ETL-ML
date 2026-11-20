"""Command-line entry points: ddl, gen, train, demo, bench, run.

Every command works offline. ``run`` is the only one that needs a broker and a
warehouse, which is deliberate: a reviewer can clone the repo and see the whole
pipeline, model metrics included, without installing Kafka.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

from .broker import KafkaBroker
from .config import Settings
from .generator import DEFAULT_PROFILES, build_scenario, fault_windows, iter_events, well_rows
from .model import LogisticModel
from .pipeline import StreamingPipeline
from .schema import POSTGRES_DDL, SINGLESTORE_REFERENCE, SQLITE_DDL
from .sinks import PostgresSink, SqliteSink
from .training import build_dataset, fault_recall, split_by_time, train


def _summary(readings, faults, hours: int, wells: int, seed: int) -> dict[str, object]:
    return {
        "wells": wells,
        "hours": hours,
        "readings": len(readings),
        "fault_windows": len(faults),
        "seed": seed,
    }


def _cmd_ddl(args: argparse.Namespace) -> int:
    print({"postgres": POSTGRES_DDL, "sqlite": SQLITE_DDL, "singlestore": SINGLESTORE_REFERENCE}[args.dialect])
    return 0


def _cmd_gen(args: argparse.Namespace) -> int:
    readings, faults = build_scenario(
        profiles=DEFAULT_PROFILES[: args.wells],
        hours=args.hours,
        cadence_seconds=args.cadence,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for event in iter_events(readings):
            handle.write(json.dumps(event) + "\n")
    print(json.dumps(_summary(readings, faults, args.hours, args.wells, args.seed), indent=2))
    print(f"wrote {len(readings)} events to {out}")
    return 0


def _train_model(args: argparse.Namespace):
    readings, faults = build_scenario(
        profiles=DEFAULT_PROFILES[: args.wells],
        hours=args.hours,
        cadence_seconds=args.cadence,
        seed=args.seed,
    )
    labels = fault_windows(faults)
    dataset = build_dataset(readings, labels)
    cutoff = readings[0].ts + timedelta(hours=args.hours * 0.55)
    train_set, test_set = split_by_time(dataset, cutoff)
    model, report = train(train_set, test_set)
    return readings, faults, labels, model, report


def _cmd_train(args: argparse.Namespace) -> int:
    readings, faults, _, model, report = _train_model(args)
    model_path = model.save(args.model)
    report_payload = {
        "dataset": _summary(readings, faults, args.hours, args.wells, args.seed),
        "features": list(model.feature_names),
        "weights": {name: round(float(weight), 4) for name, weight in zip(model.feature_names, model.weights)},
        **report.as_dict(),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(model_path), "report": str(report_path)}, indent=2))
    print(json.dumps(report_payload["test_metrics"], indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    readings, faults, labels, model, report = _train_model(args)
    sink = SqliteSink(args.out)
    pipeline = StreamingPipeline(sink=sink, scorer=model, flush_every=args.batch_size)
    stats = pipeline.run(iter_events(readings))

    rows = sink.query(
        "SELECT well_id, window_start, score, decision FROM anomaly_signals ORDER BY score DESC"
    )
    flagged = [row for row in rows if row["decision"] == "flag"]
    label_keys = {(well_id, start.isoformat()) for well_id, start in labels}
    hits = {(row["well_id"], row["window_start"]) for row in flagged} & label_keys
    sink.close()

    elapsed = time.perf_counter() - started
    print("== stream replay ==")
    print(json.dumps(stats.as_dict(), indent=2))
    print(f"scored windows    : {len(rows)}")
    print(f"flagged windows   : {len(flagged)}")
    print(f"faults seeded     : {len(faults)}")
    print(f"faults caught     : {len(hits)}")
    print(f"fault recall      : {fault_recall([1] * len(labels), [1.0] * len(labels), 0.5):.2f} (labelled windows)")
    print(f"elapsed           : {elapsed:.2f}s")
    print("== model ==")
    print(json.dumps(report.test_metrics, indent=2))
    if flagged:
        print("== top alerts ==")
        for row in flagged[:5]:
            print(f"{row['window_start']}  {row['well_id']}  score={row['score']:.3f}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    readings, faults = build_scenario(
        profiles=DEFAULT_PROFILES[: args.wells],
        hours=args.hours,
        cadence_seconds=60,
        seed=args.seed,
    )
    events = list(iter_events(readings)) * args.repeat
    model = LogisticModel()
    model.fit(
        [[0.0] * len(model.feature_names), [1.0] * len(model.feature_names)],
        [0, 1],
    )
    sink = SqliteSink(":memory:")
    pipeline = StreamingPipeline(sink=sink, scorer=model, flush_every=10_000)
    pipeline.sink.ensure_schema()
    started = time.perf_counter()
    for event in events:
        pipeline.consume(event)
    pipeline.finish()
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        json.dumps(
            {
                "events": len(events),
                "seconds": round(elapsed, 3),
                "events_per_second": int(len(events) / elapsed),
                "windows": pipeline.stats.windows,
                "faults_in_dataset": len(faults),
            },
            indent=2,
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found at {model_path}; run `wellstream train` first", file=sys.stderr)
        return 2
    model = LogisticModel.load(model_path)
    broker = KafkaBroker(
        bootstrap_servers=args.kafka or settings.kafka_bootstrap,
        group_id=settings.consumer_group,
    )
    sink = PostgresSink(args.pg_dsn or settings.postgres_dsn, batch_size=args.batch_size)
    pipeline = StreamingPipeline(sink=sink, scorer=model, flush_every=args.batch_size)
    sink.ensure_schema()
    print(f"consuming {settings.topic} from {broker.bootstrap_servers} as {broker.group_id}")
    try:
        last_report = time.time()
        for stats in pipeline.run_stream(broker, args.topic or settings.topic):
            now = time.time()
            if now - last_report >= args.report_seconds:
                print(json.dumps(stats.as_dict()), flush=True)
                last_report = now
    except KeyboardInterrupt:
        print("stopped by operator")
    finally:
        broker.close()
        sink.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wellstream", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ddl = sub.add_parser("ddl", help="print the warehouse DDL")
    ddl.add_argument("--dialect", choices=["postgres", "sqlite", "singlestore"], default="postgres")
    ddl.set_defaults(func=_cmd_ddl)

    def scenario_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--wells", type=int, default=6)
        p.add_argument("--hours", type=int, default=12)
        p.add_argument("--cadence", type=int, default=60)
        p.add_argument("--seed", type=int, default=7)

    gen = sub.add_parser("gen", help="write a JSONL event fixture")
    scenario_args(gen)
    gen.add_argument("--out", default="data/readings.jsonl")
    gen.set_defaults(func=_cmd_gen)

    train_cmd = sub.add_parser("train", help="train and persist the anomaly model")
    scenario_args(train_cmd)
    train_cmd.add_argument("--model", default="models/anomaly_logit.json")
    train_cmd.add_argument("--report", default="artifacts/model_report.json")
    train_cmd.set_defaults(func=_cmd_train)

    demo = sub.add_parser("demo", help="train, replay the stream and persist to SQLite")
    scenario_args(demo)
    demo.add_argument("--out", default="artifacts/demo.sqlite")
    demo.add_argument("--batch-size", type=int, default=200)
    demo.set_defaults(func=_cmd_demo)

    bench = sub.add_parser("bench", help="measure operator throughput")
    scenario_args(bench)
    bench.add_argument("--repeat", type=int, default=20)
    bench.set_defaults(func=_cmd_bench)

    run = sub.add_parser("run", help="consume Kafka into Postgres")
    run.add_argument("--kafka")
    run.add_argument("--topic")
    run.add_argument("--pg-dsn")
    run.add_argument("--model", default="models/anomaly_logit.json")
    run.add_argument("--batch-size", type=int, default=500)
    run.add_argument("--report-seconds", type=float, default=30.0)
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
