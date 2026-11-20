"""tally-ingest: an OTLP/HTTP trace receiver that writes raw.otel_spans (ADR-002).

The OTel Collector's `otlphttp` exporter posts to {endpoint}/v1/traces. The harness can also post
here directly when no collector is running. Inserts are idempotent on (trace_id, span_id).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from psycopg.types.json import Jsonb

from tally.ingest.otlp import SpanRow, rows_from_json, rows_from_protobuf

INSERT_SQL = """
INSERT INTO raw.otel_spans (trace_id, span_id, parent_span_id, name, kind, start_time, end_time,
                            status_code, service_name, scope_name, attributes, resource)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (trace_id, span_id) DO NOTHING
"""


def insert_rows(conn: psycopg.Connection[Any], rows: Iterable[SpanRow]) -> int:
    data = [
        (
            r.trace_id,
            r.span_id,
            r.parent_span_id,
            r.name,
            r.kind,
            r.start_time,
            r.end_time,
            r.status_code,
            r.service_name,
            r.scope_name,
            Jsonb(r.attributes),
            Jsonb(r.resource),
        )
        for r in rows
    ]
    if not data:
        return 0
    with conn.cursor() as cur:
        cur.executemany(INSERT_SQL, data)
    conn.commit()
    return len(data)


def create_app(dsn: str | None = None) -> FastAPI:
    app = FastAPI(title="tally-ingest", version="0.1.0")
    state: dict[str, Any] = {"dsn": dsn or os.environ.get("TALLY_INGEST_DSN", ""), "conn": None, "spans": 0}

    def conn() -> psycopg.Connection[Any]:
        c = state["conn"]
        if c is None or c.closed:
            c = psycopg.connect(state["dsn"])
            state["conn"] = c
        return c  # type: ignore[no-any-return]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "spans_ingested": state["spans"]}

    @app.post("/v1/traces")
    async def traces(request: Request) -> Response:
        body = await request.body()
        ctype = request.headers.get("content-type", "")
        try:
            if "json" in ctype:
                rows = list(rows_from_json(json.loads(body or b"{}")))
            else:
                rows = list(rows_from_protobuf(body))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid OTLP payload: {exc}") from exc
        n = insert_rows(conn(), rows)
        state["spans"] += n
        if "json" in ctype:
            return Response(content="{}", media_type="application/json")
        return Response(content=b"", media_type="application/x-protobuf")

    return app


app = create_app()
