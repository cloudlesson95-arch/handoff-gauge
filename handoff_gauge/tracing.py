"""OpenTelemetry setup: one tracer provider, an in-memory exporter that hands each request
its own spans as JSON, and an optional JSONL file exporter for local runs."""

import json
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Tracer


class TraceCollector(SpanExporter):
    """In-memory exporter that keeps finished spans per trace until the request collects them."""

    def __init__(self, max_traces: int = 256) -> None:
        self._traces: OrderedDict[int, list[ReadableSpan]] = OrderedDict()
        self._max_traces = max_traces
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                self._traces.setdefault(span.context.trace_id, []).append(span)
            while len(self._traces) > self._max_traces:  # traces nobody collected
                self._traces.popitem(last=False)
        return SpanExportResult.SUCCESS

    def pop(self, trace_id: int) -> list[ReadableSpan]:
        with self._lock:
            return self._traces.pop(trace_id, [])

    def shutdown(self) -> None:
        with self._lock:
            self._traces.clear()


class JsonlSpanExporter(SpanExporter):
    """Appends each finished span as one JSON line (used by local scripts)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as f:
            for span in spans:
                record = span_to_dict(span) | {"trace_id": format(span.context.trace_id, "032x")}
                f.write(json.dumps(record) + "\n")
        return SpanExportResult.SUCCESS


_collector = TraceCollector()
_provider = TracerProvider(resource=Resource.create({"service.name": "handoff-gauge"}))
_provider.add_span_processor(SimpleSpanProcessor(_collector))


def tracer() -> Tracer:
    return _provider.get_tracer("handoff_gauge")


def enable_jsonl_export(path: Path) -> None:
    _provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))


def span_to_dict(span: ReadableSpan, origin_ns: int | None = None) -> dict[str, Any]:
    start = span.start_time or 0
    end = span.end_time or start
    origin = start if origin_ns is None else origin_ns
    return {
        "name": span.name,
        "span_id": format(span.context.span_id, "016x"),
        "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_ms": round((start - origin) / 1e6, 3),
        "duration_ms": round((end - start) / 1e6, 3),
        "status": span.status.status_code.name,
        "attributes": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in (span.attributes or {}).items()
        },
    }


def spans_as_json(trace_id: int) -> list[dict[str, Any]]:
    """Collect one trace's spans, ordered by start (parents first on equal timestamps, which
    coarse clocks produce), with times relative to the first span."""
    spans = _collector.pop(trace_id)
    if not spans:
        return []
    parents = {s.context.span_id: (s.parent.span_id if s.parent else None) for s in spans}

    def depth(span_id: int) -> int:
        d = 0
        while (span_id := parents.get(span_id)) is not None:
            d += 1
        return d

    spans.sort(key=lambda s: (s.start_time or 0, depth(s.context.span_id)))
    origin = spans[0].start_time or 0
    return [span_to_dict(span, origin) for span in spans]
