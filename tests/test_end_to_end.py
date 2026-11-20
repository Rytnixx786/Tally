"""Full harness run against mocked Cinch and hosted endpoints, then replay, report and telemetry.

This is the reproduction guarantee in miniature: a replayed run must produce byte-identical decisions,
costs and metrics without touching the network.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tally import harness, report, telemetry
from tally.config import load_cost_model, load_experiment, load_tenants
from tally.ingest.otlp import rows_from_protobuf
from tests.conftest import synthetic_tickets

LABELS = sorted({"track_order", "get_refund", "recover_password"})


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def make_responder(gold: dict[str, str], wrong_rate: float, with_logprobs: bool):  # type: ignore[no-untyped-def]
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        text = user.removeprefix("Message: ").removesuffix("\nLabel:")
        g = gold[text]
        wrong = (_h(text + body["model"]) % 1000) / 1000 < wrong_rate
        label = next(lab for lab in LABELS if lab != g) if wrong else g
        content = json.dumps({"intent": label}) if "response_format" in body else label
        p = 0.55 if wrong else 0.97
        choice: dict = {"message": {"content": content}, "finish_reason": "stop"}  # type: ignore[type-arg]
        if with_logprobs and body.get("logprobs"):
            alt = next(lab for lab in LABELS if lab != label)
            choice["logprobs"] = {
                "content": [
                    {
                        "token": label,
                        "logprob": math.log(p),
                        "top_logprobs": [
                            {"token": label, "logprob": math.log(p)},
                            {"token": alt, "logprob": math.log(1 - p)},
                        ],
                    }
                ]
            }
        return httpx.Response(
            200,
            json={
                "id": f"r-{_h(text)}",
                "model": body["model"],
                "choices": [choice],
                "usage": {"prompt_tokens": 300 + len(text), "completion_tokens": 3},
            },
            headers={"X-FinOps-Request-Cost-USD": "0.00005", "X-Semantic-Cache-Status": "MISS"},
        )

    return respond


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(harness, "RECORDINGS_DIR", tmp_path / "rec")
    monkeypatch.setattr(harness, "HARDWARE_DIR", tmp_path / "rec" / "hardware")
    monkeypatch.setattr(harness, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("CINCH_BASE_URL", "http://cinch.test")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://hosted.test/v1beta/openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return tmp_path


def _opts(mode: str) -> harness.RunOptions:
    return harness.RunOptions(
        "e2e",
        mode,
        load_experiment(),
        load_tenants(),
        load_cost_model(),  # type: ignore[arg-type]
        ("A_tfidf", "B_local", "C_hosted", "D_cascade"),
        allow_spend=True,
    )


def _run(mode: str, tickets: pd.DataFrame) -> harness.Harness:
    h = harness.Harness(_opts(mode), tickets)
    h.run()
    h.write()
    return h


def test_full_run_then_identical_replay(isolated: Path) -> None:
    tickets = synthetic_tickets(per_intent=40)
    gold = dict(zip(tickets["text"], tickets["intent"], strict=True))
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://cinch.test/v1/chat/completions").mock(side_effect=make_responder(gold, 0.2, True))
        mock.post("http://hosted.test/v1beta/openai/chat/completions").mock(
            side_effect=make_responder(gold, 0.05, False)
        )
        live = _run("live", tickets)
    run_dir = isolated / "runs" / "e2e"
    dec_live = pd.read_parquet(run_dir / "decisions.parquet")
    calls_live = pd.read_parquet(run_dir / "calls.parquet")

    # Shape of the study
    prim_test = dec_live[(dec_live["split"] == "test") & (dec_live["arm"] == "primary")]
    assert set(prim_test["setup"]) == {"A_tfidf", "B_local", "C_hosted", "D_cascade"}
    assert prim_test.groupby("setup").size().nunique() == 1
    d = prim_test[prim_test["setup"] == "D_cascade"]
    assert ((d["n_calls"] == 2) == d["escalated"]).all()
    # Cascade cost = B cost + C cost when escalated
    b = prim_test[prim_test["setup"] == "B_local"].set_index("ticket_id")["total_cost_usd"]
    c = prim_test[prim_test["setup"] == "C_hosted"].set_index("ticket_id")["total_cost_usd"]
    for r in d.itertuples():
        assert r.total_cost_usd == pytest.approx(b[r.ticket_id] + (c[r.ticket_id] if r.escalated else 0.0))
    # Selection artifacts exist and never touch test
    assert json.loads((run_dir / "prompt_selection.json").read_text())["split"] == "tune"
    assert json.loads((run_dir / "calibration.json").read_text())["fit_split"] == "tune"
    assert 0.0 <= live.artifacts["tau"]["chosen_tau"] <= 1.0
    assert (calls_live["cost_usd"] >= 0).all()
    assert set(calls_live.loc[calls_live["setup"] == "C_hosted", "cost_method"]) == {"price_book"}

    # Replay: no network (respx not active -> any call would fail), identical outputs
    _run("replay", tickets)
    dec_replay = pd.read_parquet(run_dir / "decisions.parquet")
    calls_replay = pd.read_parquet(run_dir / "calls.parquet")
    pd.testing.assert_frame_equal(dec_live.reset_index(drop=True), dec_replay.reset_index(drop=True))
    pd.testing.assert_frame_equal(
        calls_live.drop(columns=["replayed"]).reset_index(drop=True),
        calls_replay.drop(columns=["replayed"]).reset_index(drop=True),
    )
    assert calls_replay.loc[calls_replay["setup"] != "A_tfidf", "replayed"].all()


def test_hosted_spend_needs_explicit_permission(isolated: Path) -> None:
    tickets = synthetic_tickets(per_intent=40)
    gold = dict(zip(tickets["text"], tickets["intent"], strict=True))
    opts = _opts("live")
    opts.allow_spend = False
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://cinch.test/v1/chat/completions").mock(side_effect=make_responder(gold, 0.2, True))
        route = mock.post("http://hosted.test/v1beta/openai/chat/completions").mock(
            side_effect=make_responder(gold, 0.05, False)
        )
        with pytest.raises(harness.SpendNotAllowedError, match="confirm-spend"):
            harness.Harness(opts, tickets).run()
    assert route.call_count == 0


def test_report_and_telemetry_from_artifacts(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tickets = synthetic_tickets(per_intent=40)
    gold = dict(zip(tickets["text"], tickets["intent"], strict=True))
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://cinch.test/v1/chat/completions").mock(side_effect=make_responder(gold, 0.2, True))
        mock.post("http://hosted.test/v1beta/openai/chat/completions").mock(
            side_effect=make_responder(gold, 0.05, False)
        )
        _run("live", tickets)

    monkeypatch.setattr(report, "RUNS_DIR", isolated / "runs")
    monkeypatch.setattr(report, "DOCS", isolated / "docs")
    monkeypatch.setattr(report, "ASSETS", isolated / "docs" / "assets")
    readme = isolated / "README.md"
    readme.write_text(f"intro\n{report.START}\nold\n{report.END}\noutro\n", encoding="utf-8")
    monkeypatch.setattr(report, "README", readme)
    run_dir = report.build("e2e", source="artifacts")
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["headline"] is not None
    assert set(m["test"]) == {"A_tfidf", "B_local", "C_hosted", "D_cascade"}
    assert m["test"]["D_cascade"]["accuracy"]["lo"] <= m["test"]["D_cascade"]["accuracy"]["point"]
    text = readme.read_text(encoding="utf-8")
    assert "Router achieved" in text and "old" not in text and text.startswith("intro") and "outro" in text
    assert (isolated / "docs" / "assets" / "hero_cost_vs_accuracy.svg").exists()

    # Telemetry: one decision span + one CLIENT span per call, deterministic ids, no content attributes
    calls = pd.read_parquet(run_dir / "calls.parquet")
    dec = pd.read_parquet(run_dir / "decisions.parquet")
    exp1, exp2 = InMemorySpanExporter(), InMemorySpanExporter()
    n = telemetry.emit(calls, dec, "http://unused", exporter=exp1)
    telemetry.emit(calls, dec, "http://unused", exporter=exp2)
    spans1, spans2 = exp1.get_finished_spans(), exp2.get_finished_spans()
    assert n == len(spans1) == len(calls) + len(dec)
    assert [s.context.span_id for s in spans1] == [s.context.span_id for s in spans2]
    client = [s for s in spans1 if s.name.startswith("chat ")]
    assert all(s.attributes["gen_ai.operation.name"] in ("chat", "text_classification") for s in client)
    assert not any(
        k.startswith(("gen_ai.input", "gen_ai.output")) for s in spans1 for k in (s.attributes or {})
    )
    # ... and the ingest decoder reads the exporter's own protobuf encoding back
    rows = list(rows_from_protobuf(encode_spans(spans1[:50]).SerializeToString()))
    assert len(rows) == 50 and all(len(r.trace_id) == 32 and len(r.span_id) == 16 for r in rows)
    assert {r.attributes.get("tally.run_id") for r in rows} == {"e2e"}
