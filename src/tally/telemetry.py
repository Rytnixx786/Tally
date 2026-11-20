"""Client-side OpenTelemetry GenAI spans for every model call (TALLY_SPAN_SOURCE=client).

Cinch, unmodified, does not export spans (docs/CINCH_CHANGES.md F1), so the harness emits the
standard GenAI CLIENT span itself, using the recorded timestamps and Cinch's response headers.
Ids and times are deterministic, so replaying a run produces identical raw.otel_spans rows.

Span tree per decision:
    tally.decision (INTERNAL)            tally.* decision attributes
      └─ chat {model} (CLIENT)           gen_ai.* + tally.* call attributes, one per model call
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import pandas as pd
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import SpanKind, Status, StatusCode, set_span_in_context

from tally import __version__, ids

SEMCONV_VERSION = "gen_ai development, semantic-conventions-genai main @ 2026-11-19"


class _DeterministicIds(IdGenerator):
    def __init__(self) -> None:
        self._local = threading.local()

    def set(self, trace_id: int, span_id: int) -> None:
        self._local.trace_id = trace_id
        self._local.span_id = span_id

    def generate_trace_id(self) -> int:
        return int(self._local.trace_id)

    def generate_span_id(self) -> int:
        return int(self._local.span_id)


def _ns(seconds: float) -> int:
    return round(seconds * 1e9)


def _present(v: Any) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v != "")


def call_attributes(c: pd.Series) -> dict[str, Any]:
    """GenAI semconv attributes plus tally.* (and cinch.* gateway facts). No prompt/response text."""
    a: dict[str, Any] = {
        "gen_ai.operation.name": "chat" if c["provider"] != "sklearn" else "text_classification",
        "gen_ai.provider.name": c["provider"],
        "gen_ai.request.model": c["model"],
        "gen_ai.request.max_tokens": 16,
        "gen_ai.request.temperature": 0.0,
        "gen_ai.usage.input_tokens": int(c["input_tokens"]),
        "gen_ai.usage.output_tokens": int(c["output_tokens"]),
        "gen_ai.usage.cache_read.input_tokens": int(c["cached_input_tokens"]),
        "tally.request_id": c["request_id"],
        "tally.decision_id": c["decision_id"],
        "tally.run_id": c["run_id"],
        "tally.setup": c["setup"],
        "tally.split": c["split"],
        "tally.arm": c["arm"],
        "tally.call_index": int(c["call_index"]),
        "tally.tenant_id": c["tenant_id"],
        "tally.feature": "ticket_triage",
        "tally.prompt_version": c["prompt_version"],
        "tally.validation_passed": bool(c["validation_passed"]),
        "tally.escalation": bool(c["is_escalation"]),
        "tally.reused_response": bool(c["reused_response"]),
        "tally.cache_status": c["cache_status"],
        "tally.http_status": int(c["http_status"]),
        "tally.latency_ms": float(c["latency_ms"]),
    }
    if _present(c.get("model_response")):
        a["gen_ai.response.model"] = c["model_response"]
    if _present(c.get("response_id")):
        a["gen_ai.response.id"] = c["response_id"]
    if _present(c.get("finish_reason")):
        a["gen_ai.response.finish_reasons"] = [c["finish_reason"]]
    if _present(c.get("confidence")):
        a["tally.confidence"] = float(c["confidence"])
    if _present(c.get("predicted_intent")):
        a["tally.predicted_label"] = c["predicted_intent"]
    if _present(c.get("billed_usd")):
        a["cinch.finops.billed_usd"] = float(c["billed_usd"])
    if _present(c.get("server_address")):
        from urllib.parse import urlparse

        u = urlparse(str(c["server_address"]))
        if u.hostname:
            a["server.address"] = u.hostname
        if u.port:
            a["server.port"] = u.port
    if int(c["http_status"]) != 200:
        a["error.type"] = str(int(c["http_status"]))
    return a


def emit(
    calls: pd.DataFrame, decisions: pd.DataFrame, endpoint: str, exporter: SpanExporter | None = None
) -> int:
    """Export one decision span plus one client span per call. Returns the number of spans sent."""
    gen = _DeterministicIds()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "tally-harness",
                "service.version": __version__,
                "tally.semconv": SEMCONV_VERSION,
            }
        ),
        id_generator=gen,
    )
    exp = exporter or OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces", timeout=30)
    processor = BatchSpanProcessor(
        exp, max_queue_size=200_000, max_export_batch_size=2048, schedule_delay_millis=500
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("tally.harness", __version__)
    by_decision = {k: g.sort_values("call_index") for k, g in calls.groupby("decision_id", sort=False)}
    n = 0
    for d in _rows(decisions):
        did = d["decision_id"]
        group = by_decision.get(did)
        if group is None:
            continue
        start = float(group["started_at"].min())
        end = start + float(group["latency_ms"].sum()) / 1000.0
        gen.set(ids.trace_id_int(did), ids.span_id_int(f"{did}:decision"))
        parent = tracer.start_span(
            "tally.decision",
            kind=SpanKind.INTERNAL,
            start_time=_ns(start),
            attributes={
                "tally.decision_id": did,
                "tally.run_id": d["run_id"],
                "tally.setup": d["setup"],
                "tally.split": d["split"],
                "tally.arm": d["arm"],
                "tally.ticket_id": d["ticket_id"],
                "tally.tenant_id": d["tenant_id"],
                "tally.prompt_version": d["prompt_version"],
                "tally.escalated": bool(d["escalated"]),
                "tally.n_calls": int(d["n_calls"]),
            },
        )
        ctx = set_span_in_context(parent)
        t = start
        for c in _rows(group):
            gen.set(ids.trace_id_int(did), ids.span_id_int(f"{did}:{int(c['call_index'])}"))
            span = tracer.start_span(
                f"chat {c['model']}",
                context=ctx,
                kind=SpanKind.CLIENT,
                start_time=_ns(t),
                attributes=call_attributes(c),
            )
            if int(c["http_status"]) != 200:
                span.set_status(Status(StatusCode.ERROR, str(c.get("error") or "")[:200]))
            t += float(c["latency_ms"]) / 1000.0
            span.end(end_time=_ns(t))
            n += 1
        parent.end(end_time=_ns(end))
        n += 1
    provider.force_flush(timeout_millis=120_000)
    provider.shutdown()
    return n


def _rows(df: pd.DataFrame) -> Iterable[pd.Series]:
    for _, r in df.iterrows():
        yield r
