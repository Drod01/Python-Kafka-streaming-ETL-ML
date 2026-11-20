# Wellstream — event-driven well telemetry pipeline (Python)

A production-shaped streaming data pipeline for oil & gas well telemetry: **validate → window on event
time → engineer features → score with a math model and a trained classifier → upsert into a SQL
warehouse**, with the whole thing runnable offline in under a second.

```
 SCADA / field gateways
        │  JSON events (well_id, ts, pressure, temp, flow)
        ▼
 ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
 │ Kafka /      │──▶│ Validation &  │──▶│ Event-time   │──▶│ Features +    │──▶│ Postgres     │
 │ Redpanda     │   │ de-duplication│   │ tumbling     │   │ Arps decline  │   │ windows +    │
 │ (well keyed) │   │ (physics+DLQ) │   │ windows      │   │ + logistic    │   │ alert signals│
 └──────────────┘   └───────────────┘   └──────────────┘   │ classifier    │   └──────────────┘
                                                            └───────────────┘
```

## Why this project exists

The role asks for **Python, SQL, event-driven and cloud architectures, containerised streaming, and
math/ML models operationalised for decision-making**. This project is those requirements in one
runnable service, with the trade-offs written down rather than implied.

| Requirement | Where it lives |
| --- | --- |
| Event-driven architecture | `broker.py` (Kafka adapter + in-memory adapter), `pipeline.py` |
| Streaming correctness | `windows.py` — event-time windows, watermark, late-data accounting |
| Data quality / lifecycle | `validation.py` — physics rules, non-finite guards, bounded de-duplication |
| Math model | `decline.py` — Arps exponential decline fit used as a type-curve signal |
| ML model, operationalised | `model.py`, `training.py` — trained, versioned, JSON-serialised, threshold-tuned |
| SQL design | `schema.py` — keys, indexes, partial index on flags, a daily rollup view |
| Reliability | idempotent upserts (`sinks.py`), all-violations-at-once validation, no raise on bad input |
| Testability | ports/adapters: broker and sink are interfaces, so the pipeline runs in-process in CI |
| Containers | `Dockerfile`, `docker-compose.yml` (Redpanda + Postgres + trainer + producer + processor) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # 63 tests, ~0.5s, no broker or database required
wellstream demo                           # train + replay the stream into SQLite
wellstream train                          # write models/anomaly_logit.json + metrics report
wellstream bench --repeat 5               # operator throughput
wellstream ddl --dialect postgres         # print the warehouse DDL
```

Full stack (Redpanda, Postgres, trained model, live producer, processor):

```bash
docker compose up --build
docker compose exec postgres psql -U wellstream -d wellstream \
  -c "select well_id, window_start, score, decision from anomaly_signals where decision = 'flag' order by window_start;"
```

## Verified results

Measured on this repository (synthetic, seeded scenario: 6 wells, 12 h, 1-minute cadence, 16 scripted
fault windows; model trained on the first 55% of the timeline and evaluated on the held-out rest):

| Signal | Result |
| --- | --- |
| Events consumed | 4,320 |
| Windows emitted / scored | 864 / 840 (24 windows were warm-up for the feature baseline) |
| Held-out ROC AUC / precision / recall | 1.00 / 1.00 / 1.00 (see note below) |
| Fault windows caught | 16 of 16 |
| Operator throughput (single process) | ~51k events/s (`wellstream bench`) |
| Unit + integration tests | 63 passing |

**Read the 1.00 honestly:** the generator scripts clean faults into a noiseless baseline, so the classes
are trivially separable — the number proves the pipeline, feature and scoring plumbing are correct, not
that a classifier generalises to a real field. On real telemetry the same harness reports the metrics
you would actually quote; the README of the model report (`artifacts/model_report.json`) always carries
the dataset fingerprint next to the scores so the two cannot drift apart.

## Design decisions worth discussing

**Event time, not processing time.** Field gateways buffer during cellular dropouts and then flush a
burst. A processing-time window would attribute that burst to the wrong minute and manufacture a
production spike. Windows are keyed by event time; a watermark (`max seen ts − allowed lateness`)
decides when a window is closed; anything arriving after its window closed is counted in
`late_dropped` instead of silently mutating an already-emitted aggregate.

**At-least-once delivery, idempotent everything.** Kafka gives at-least-once. Three layers make retries
harmless: bounded key de-duplication in the consumer, `ON CONFLICT DO UPDATE` upserts keyed on
`(well_id, window_start)`, and a deterministic window computation. Replaying a whole batch produces
identical rows — there is a test for exactly that.

**Relative features, per well.** A 40 bbl/d stripper well and a 1,200 bbl/d Wolfcamp well share no
absolute scale. Every feature is a deviation from that well's trailing baseline (flow drop ratio,
pressure z-score, coefficient of variation, coverage gap, temperature drift, Arps-curve miss), clipped
so one bad window cannot dominate.

**Coverage before conclusions.** A window missing 40% of its samples is not a production drop. The
coverage ratio is both a stored column and a feature, so a gateway outage is described as a gateway
outage.

**A hand-rolled model on purpose.** The classifier is ~80 lines of NumPy logistic regression with L2,
standardised inputs, a precision-weighted threshold sweep, and JSON persistence. For a six-feature
tabular score that is more honest than shipping an opaque pickle: the coefficients are reviewable, the
training run is deterministic, and there is no dependency to patch.

**Ports and adapters, not mocks.** `MessageBroker` and `WarehouseSink` are protocols with Kafka/Postgres
and in-memory/SQLite implementations. The same `StreamingPipeline` class runs in production and in
tests; there is no parallel "test pipeline" to drift.

## Repository layout

```
src/wellstream/
  models.py       immutable records + the JSON event contract
  validation.py   physics rules, non-finite guards, bounded de-duplicator
  windows.py      event-time tumbling windows, watermark, lateness accounting
  decline.py      Arps exponential decline fit (math model)
  features.py     per-well relative features + operator-readable explanations
  model.py        NumPy logistic regression, AUC/precision/recall/Brier, JSON persistence
  training.py     dataset build from the *production* operators, time-based split
  pipeline.py     the operator: validate → window → featurise → score → persist
  broker.py       Kafka and in-memory adapters
  sinks.py        Postgres and SQLite sinks (batched, idempotent)
  schema.py       warehouse DDL (Postgres, SingleStore reference, SQLite)
  cli.py          ddl / gen / train / demo / bench / run
tests/            63 tests: unit, operator, end-to-end replay
```

## Operating it

Counters are exposed as `PipelineStats` (consumed, malformed, accepted, rejected, duplicates,
late_dropped, windows, scored, flagged, written, top rejection reasons) and printed as JSON by
`wellstream run`, which is what becomes the CloudWatch/Prometheus metric set: `late_dropped` rising
means the lateness budget is wrong, `rejection_reasons` shifting means a site's sensor contract
changed.

## Deliberate scope limits

- The scenario generator is synthetic; the repository is about the pipeline, not about data acquisition.
- The model is a baseline logistic regression — no hyper-parameter search, no drift monitoring. A
  production deployment would add both, plus a shadow-scored challenger model.
- Postgres stands in for an Oracle/MemSQL estate in this project; those dialects are exercised in
  `../02-oracle-memsql-dotnet`.
