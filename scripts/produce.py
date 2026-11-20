"""Publish the simulated well telemetry into Kafka for the compose demo.

The generator is deterministic, so `docker compose up` produces a reproducible
stream and the processor's alert count is predictable.

    python scripts/produce.py --rate 120 --hours 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

from kafka import KafkaProducer

from wellstream.generator import build_scenario, iter_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC", "well.telemetry"))
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate", type=float, default=60.0, help="events per second (0 = as fast as possible)")
    parser.add_argument("--loop", action="store_true", help="keep replaying the scenario")
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        acks="all",
        linger_ms=20,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    sent = 0
    try:
        while True:
            readings, _ = build_scenario(hours=args.hours, seed=args.seed)
            for event in iter_events(readings):
                producer.send(args.topic, key=str(event["well_id"]), value=event)
                sent += 1
                if interval:
                    time.sleep(interval)
                if sent % 500 == 0:
                    producer.flush()
                    print(f"published {sent} events", flush=True)
            producer.flush()
            print(f"scenario complete: {sent} events", flush=True)
            if not args.loop:
                return 0
    except KeyboardInterrupt:
        print(f"stopped after {sent} events", flush=True)
        return 0
    finally:
        producer.close()


if __name__ == "__main__":
    raise SystemExit(main())
