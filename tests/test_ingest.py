from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tally.ingest import app as ingest_app
from tally.ingest.otlp import rows_from_json
from tests.conftest import pg_dsn

PAYLOAD: dict[str, Any] = {
    "resourceSpans": [
        {
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "cinch-gateway"}}]},
            "scopeSpans": [
                {
                    "scope": {"name": "cinch"},
                    "spans": [
                        {
                            "traceId": "5B8EFFF798038103D269B633813FC60C",
                            "spanId": "EEE19B7EC3C1B174",
                            "parentSpanId": "",
                            "name": "chat Qwen/Qwen2.5-7B-Instruct-AWQ",
                            "kind": "SPAN_KIND_CLIENT",
                            "startTimeUnixNano": "1795112146000000000",
                            "endTimeUnixNano": "1795112146250000000",
                            "status": {"code": "STATUS_CODE_OK"},
                            "attributes": [
                                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "1539"}},
                                {"key": "tally.confidence", "value": {"doubleValue": "NaN"}},
                                {
                                    "key": "gen_ai.response.finish_reasons",
                                    "value": {"arrayValue": {"values": [{"stringValue": "stop"}]}},
                                },
                                {"key": "cinch.cache.hit", "value": {"boolValue": False}},
                            ],
                        }
                    ],
                }
            ],
        }
    ],
}


def test_rows_from_json_types_and_sanitises() -> None:
    (row,) = list(rows_from_json(PAYLOAD))
    assert row.trace_id == "5b8efff798038103d269b633813fc60c" and row.parent_span_id is None
    assert row.kind == 3 and row.status_code == 1
    assert (row.end_time - row.start_time).total_seconds() == pytest.approx(0.25)
    assert row.attributes["gen_ai.usage.input_tokens"] == 1539
    assert row.attributes["tally.confidence"] is None  # NaN is not valid JSONB
    assert row.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert row.service_name == "cinch-gateway" and row.scope_name == "cinch"


def test_http_endpoint_accepts_json_and_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    got: list[Any] = []
    monkeypatch.setattr(ingest_app, "insert_rows", lambda conn, rows: got.extend(rows) or len(got))
    monkeypatch.setattr(ingest_app.psycopg, "connect", lambda dsn: type("C", (), {"closed": False})())
    client = TestClient(ingest_app.create_app("postgresql://unused"))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/v1/traces", json=PAYLOAD)
    assert r.status_code == 200 and len(got) == 1
    bad = client.post(
        "/v1/traces", content=b"not protobuf", headers={"content-type": "application/x-protobuf"}
    )
    assert bad.status_code == 400


@pytest.mark.integration
def test_insert_is_idempotent_against_postgres() -> None:
    dsn = pg_dsn()
    if dsn is None:
        pytest.skip("no Postgres available")
    import psycopg

    payload = {**PAYLOAD}
    span = dict(PAYLOAD["resourceSpans"][0]["scopeSpans"][0]["spans"][0])
    span["traceId"] = uuid.uuid4().hex
    payload = {
        "resourceSpans": [
            {**PAYLOAD["resourceSpans"][0], "scopeSpans": [{"scope": {"name": "t"}, "spans": [span]}]}
        ]
    }
    with psycopg.connect(dsn) as conn:
        ingest_app.insert_rows(conn, rows_from_json(payload))
        ingest_app.insert_rows(conn, rows_from_json(payload))
        n = conn.execute(
            "select count(*) from raw.otel_spans where trace_id = %s", (span["traceId"],)
        ).fetchone()
        conn.execute("delete from raw.otel_spans where trace_id = %s", (span["traceId"],))
        conn.commit()
    assert n is not None and n[0] == 1
