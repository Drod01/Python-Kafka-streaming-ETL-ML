"""Broker adapters: one protocol, two implementations.

The pipeline talks to ``MessageBroker`` and never to Kafka directly. That keeps
the operator testable in-process (``InMemoryBroker``) while production uses a
real Kafka/Redpanda cluster (``KafkaBroker``), and it documents the event
contract in one place instead of scattering ``confluent``/``kafka`` imports
through the business logic.
"""

from __future__ import annotations

import json
import queue
from collections import defaultdict
from typing import Iterator, Mapping, Protocol, runtime_checkable


@runtime_checkable
class MessageBroker(Protocol):
    """Minimal publish/subscribe contract used by the pipeline."""

    def publish(self, topic: str, event: Mapping[str, object]) -> None: ...

    def subscribe(self, topic: str) -> Iterator[dict[str, object]]: ...

    def close(self) -> None: ...


_SENTINEL = object()


class InMemoryBroker:
    """Thread-safe in-process broker.

    Messages are JSON round-tripped on publish, exactly like the wire format, so
    a schema mistake shows up in tests instead of in production.
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._subscribers: dict[str, list[queue.Queue[object]]] = defaultdict(list)
        self._closed = False

    def publish(self, topic: str, event: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("broker is closed")
        payload = json.loads(json.dumps(dict(event)))
        for subscriber in list(self._subscribers[topic]):
            subscriber.put(payload)
        self._messages[topic].append(payload)

    def subscribe(self, topic: str) -> Iterator[dict[str, object]]:
        # A closed broker behaves like a log: it replays everything it accepted.
        # That is what makes offline demos and tests deterministic, and it
        # mirrors `warmstream replay` against a drained Kafka topic.
        if self._closed:
            yield from list(self._messages[topic])
            return
        channel: queue.Queue[object] = queue.Queue()
        self._subscribers[topic].append(channel)
        try:
            while True:
                try:
                    item = channel.get(timeout=0.05)
                except queue.Empty:
                    if self._closed:
                        return
                    continue
                if item is _SENTINEL:
                    return
                yield item  # type: ignore[misc]
        finally:
            if channel in self._subscribers[topic]:
                self._subscribers[topic].remove(channel)

    def replay(self, topic: str) -> list[dict[str, object]]:
        """Every message published to a topic, for assertions and backfills."""
        return list(self._messages[topic])

    def close(self) -> None:
        self._closed = True
        for channels in self._subscribers.values():
            for channel in list(channels):
                channel.put(_SENTINEL)


class KafkaBroker:
    """Kafka/Redpanda adapter built on kafka-python.

    Imported lazily so the operator, its tests and the offline demo do not need
    a broker installed. Consumer groups give us horizontal scale; the pipeline
    itself stays idempotent because delivery is at-least-once.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str = "wellstream-processor",
        auto_offset_reset: str = "earliest",
        client_id: str = "wellstream",
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.client_id = client_id
        self._producer = None
        self._consumer = None

    def _get_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer  # imported here on purpose

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                acks="all",
                linger_ms=20,
                key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        return self._producer

    def publish(self, topic: str, event: Mapping[str, object]) -> None:
        producer = self._get_producer()
        # Key by well so all samples for a well land on one partition; that is
        # what makes per-well ordering (and therefore windowing) deterministic.
        producer.send(topic, key=str(event.get("well_id", "")), value=dict(event))
        producer.flush()

    def subscribe(self, topic: str) -> Iterator[dict[str, object]]:
        if self._consumer is None:
            from kafka import KafkaConsumer

            self._consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                client_id=self.client_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
                max_poll_records=500,
            )
        for record in self._consumer:
            yield record.value

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
        if self._producer is not None:
            self._producer.close()
            self._producer = None
