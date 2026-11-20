"""Flatten OTLP trace payloads (JSON or protobuf) into rows for raw.otel_spans."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class SpanRow:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: int | None
    start_time: datetime
    end_time: datetime
    status_code: int | None
    service_name: str | None
    scope_name: str | None
    attributes: dict[str, Any]
    resource: dict[str, Any]


def _ns_to_dt(ns: int | str) -> datetime:
    return datetime.fromtimestamp(int(ns) / 1e9, tz=UTC)


def _any_value(v: Mapping[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_any_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _any_value(kv.get("value", {})) for kv in v["kvlistValue"].get("values", [])}
    return None


def _finite(v: Any) -> Any:
    """JSONB rejects NaN/Infinity; store them as null."""
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    if isinstance(v, list):
        return [_finite(x) for x in v]
    return v


def _attrs(items: list[Mapping[str, Any]] | None) -> dict[str, Any]:
    return {a["key"]: _finite(_any_value(a.get("value", {}))) for a in (items or [])}


def rows_from_json(payload: Mapping[str, Any]) -> Iterator[SpanRow]:
    """OTLP/JSON: ids are lowercase hex, times are (string) unix nanoseconds."""
    for rs in payload.get("resourceSpans", []):
        resource = _attrs((rs.get("resource") or {}).get("attributes"))
        for ss in rs.get("scopeSpans", []):
            scope = (ss.get("scope") or {}).get("name")
            for sp in ss.get("spans", []):
                status = sp.get("status") or {}
                yield SpanRow(
                    trace_id=str(sp["traceId"]).lower(),
                    span_id=str(sp["spanId"]).lower(),
                    parent_span_id=(str(sp["parentSpanId"]).lower() or None)
                    if sp.get("parentSpanId")
                    else None,
                    name=sp.get("name", ""),
                    kind=_kind(sp.get("kind")),
                    start_time=_ns_to_dt(sp["startTimeUnixNano"]),
                    end_time=_ns_to_dt(sp["endTimeUnixNano"]),
                    status_code=_kind(status.get("code")),
                    service_name=resource.get("service.name"),
                    scope_name=scope,
                    attributes=_attrs(sp.get("attributes")),
                    resource=resource,
                )


_KINDS = {
    "SPAN_KIND_UNSPECIFIED": 0,
    "SPAN_KIND_INTERNAL": 1,
    "SPAN_KIND_SERVER": 2,
    "SPAN_KIND_CLIENT": 3,
    "SPAN_KIND_PRODUCER": 4,
    "SPAN_KIND_CONSUMER": 5,
    "STATUS_CODE_UNSET": 0,
    "STATUS_CODE_OK": 1,
    "STATUS_CODE_ERROR": 2,
}


def _kind(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, str):
        return _KINDS.get(v, int(v) if v.isdigit() else None)
    return int(v)


def rows_from_protobuf(body: bytes) -> Iterator[SpanRow]:
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest()
    req.ParseFromString(body)
    for rs in req.resource_spans:
        resource = _attrs(MessageToDict(rs.resource).get("attributes"))
        for ss in rs.scope_spans:
            for sp in ss.spans:
                d = MessageToDict(sp)
                yield SpanRow(
                    trace_id=sp.trace_id.hex(),
                    span_id=sp.span_id.hex(),
                    parent_span_id=sp.parent_span_id.hex() or None,
                    name=sp.name,
                    kind=int(sp.kind),
                    start_time=_ns_to_dt(sp.start_time_unix_nano),
                    end_time=_ns_to_dt(sp.end_time_unix_nano),
                    status_code=int(sp.status.code),
                    service_name=resource.get("service.name"),
                    scope_name=ss.scope.name or None,
                    attributes=_attrs(d.get("attributes")),
                    resource=resource,
                )
