from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest
import respx

from tally.llm.client import (
    ChatClient,
    Endpoint,
    anthropic_payload,
    openai_payload,
    parse_anthropic,
    parse_openai,
)
from tally.llm.logprobs import label_confidence, parse_label
from tally.llm.record_replay import Recorder, ReplayMissError, request_key
from tally.llm.types import CallResult, ChatRequest, TokenLogprob

LABELS = ("track_order", "track_refund", "get_refund", "review")


def req(**kw: object) -> ChatRequest:
    base = {
        "provider": "cinch",
        "model": "m",
        "messages": (("system", "s"), ("user", "u")),
        "max_tokens": 16,
        "labels": LABELS,
        "constraint": "guided_choice",
        "top_logprobs": 5,
    }
    base.update(kw)
    return ChatRequest(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- labels and confidence


@pytest.mark.parametrize(
    ("content", "want"),
    [
        ("track_order", "track_order"),
        (" Track_Order. ", "track_order"),
        ('"review"', "review"),
        ('{"intent": "get_refund"}', "get_refund"),
        ("I think track_order", None),
        ("{bad json", None),
        ('{"label": "review"}', None),
    ],
)
def test_parse_label(content: str, want: str | None) -> None:
    assert parse_label(content, LABELS) == want


def test_confidence_renormalises_over_valid_continuations() -> None:
    # Position 1: "track" (p=.6) vs "get" (p=.2) vs junk "#" (p=.2, invalid -> excluded).
    # Position 2 given "track": "_order" (p=.5) vs "_refund" (p=.3) vs "!!" (invalid).
    toks = [
        TokenLogprob(
            "track", math.log(0.6), (("track", math.log(0.6)), ("get", math.log(0.2)), ("#", math.log(0.2)))
        ),
        TokenLogprob(
            "_order",
            math.log(0.5),
            (("_order", math.log(0.5)), ("_refund", math.log(0.3)), ("!!", math.log(0.2))),
        ),
        TokenLogprob("", 0.0, (("", 0.0),)),
    ]
    got = label_confidence(toks, LABELS, "track_order")
    assert got == pytest.approx((0.6 / 0.8) * (0.5 / 0.8))


def test_confidence_handles_leading_whitespace_and_missing_logprobs() -> None:
    toks = [
        TokenLogprob(" ", 0.0, ()),
        TokenLogprob("review", math.log(0.9), (("review", math.log(0.9)), ("track", math.log(0.1)))),
    ]
    assert label_confidence(toks, LABELS, "review") == pytest.approx(0.9)
    assert label_confidence(None, LABELS, "review") is None
    assert label_confidence(toks, LABELS, None) is None


def test_confidence_is_one_when_constraint_leaves_one_option() -> None:
    toks = [TokenLogprob("review", 0.0, (("review", 0.0), ("#", -9999.0)))]
    assert label_confidence(toks, LABELS, "review") == pytest.approx(1.0)


# ---------------------------------------------------------------- record / replay


def _result(r: ChatRequest, status: int = 200) -> CallResult:
    return CallResult(
        request_key(r),
        r.provider,
        r.model,
        "m",
        "id1",
        "review",
        "stop",
        10,
        1,
        0,
        12.5,
        1.0e9,
        status,
        None if status == 200 else "err",
        token_logprobs=[TokenLogprob("review", -0.1, (("review", -0.1),))],
    )


def test_record_then_replay_roundtrip(tmp_path: Path) -> None:
    r = req()
    rec = Recorder(tmp_path, "live")
    rec.put(_result(r), r)
    again = Recorder(tmp_path, "replay").get(r)
    assert again is not None and again.replayed and again.content == "review"
    assert again.token_logprobs == [TokenLogprob("review", -0.1, (("review", -0.1),))]


def test_transient_failures_are_not_recorded_but_rejections_are(tmp_path: Path) -> None:
    r1, r2 = req(seed=1), req(seed=2)
    rec = Recorder(tmp_path, "live")
    rec.put(_result(r1, 503), r1)
    rec.put(_result(r2, 400), r2)
    fresh = Recorder(tmp_path, "replay")
    assert fresh.get(r1) is None
    assert fresh.get(r2) is not None


def test_request_key_covers_the_experiment_arm() -> None:
    assert request_key(req()) != request_key(req(variant="cinch_cache_on"))
    assert request_key(req()) == request_key(req())


def test_replay_miss_is_an_error(tmp_path: Path) -> None:
    async def go() -> None:
        async with ChatClient(Endpoint("cinch", "http://x", None), Recorder(tmp_path, "replay")) as c:
            await c.chat(req())

    with pytest.raises(ReplayMissError):
        asyncio.run(go())


# ---------------------------------------------------------------- payloads and parsing


def test_payload_constraints() -> None:
    assert openai_payload(req())["guided_choice"] == list(LABELS)
    assert "seed" in openai_payload(req())
    assert "seed" not in openai_payload(req(provider="gemini", constraint="json_schema"))
    assert openai_payload(req(constraint="structured_outputs"))["structured_outputs"] == {
        "choice": list(LABELS)
    }
    rf = openai_payload(req(constraint="json_schema"))["response_format"]
    assert rf["json_schema"]["schema"]["properties"]["intent"]["enum"] == list(LABELS)
    body = openai_payload(req(top_logprobs=0, constraint="none"))
    assert "logprobs" not in body and "guided_choice" not in body
    ap = anthropic_payload(req())
    assert ap["system"] == "s" and ap["messages"] == [{"role": "user", "content": "u"}]


def test_parse_openai_and_anthropic() -> None:
    o = parse_openai(
        {
            "id": "x",
            "model": "m",
            "choices": [
                {
                    "message": {"content": "review"},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "review",
                                "logprob": -0.2,
                                "top_logprobs": [{"token": "review", "logprob": -0.2}],
                            }
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        }
    )
    assert (o["content"], o["input_tokens"], o["cached_input_tokens"]) == ("review", 100, 64)
    assert o["token_logprobs"][0].top == (("review", -0.2),)
    a = parse_anthropic(
        {
            "id": "y",
            "model": "h",
            "content": [{"type": "text", "text": "review"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 2, "cache_read_input_tokens": 10},
        }
    )
    assert (a["content"], a["input_tokens"], a["cached_input_tokens"]) == ("review", 60, 10)


@respx.mock
def test_client_retries_transient_errors_and_reads_gateway_headers(tmp_path: Path) -> None:
    body = {
        "id": "c1",
        "model": "m",
        "choices": [{"message": {"content": "review"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1},
    }
    route = respx.post("http://gw/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.Response(
                200,
                json=body,
                headers={
                    "X-Semantic-Cache-Status": "MISS",
                    "X-FinOps-Request-Cost-USD": "0.000123",
                    "X-Request-ID": "abcd1234",
                },
            ),
        ]
    )

    async def go() -> CallResult:
        async with ChatClient(
            Endpoint("cinch", "http://gw", "k"), Recorder(tmp_path, "live"), max_retries=2
        ) as c:
            return await c.chat(req(), {"X-Tenant-ID": "support"})

    import tally.llm.client as client_mod

    orig = client_mod.asyncio.sleep
    client_mod.asyncio.sleep = lambda *_: orig(0)  # type: ignore[assignment]
    try:
        res = asyncio.run(go())
    finally:
        client_mod.asyncio.sleep = orig  # type: ignore[assignment]
    assert route.call_count == 2
    assert res.ok and res.cache_status == "MISS" and res.billed_usd == pytest.approx(0.000123)
    assert res.gateway_request_id == "abcd1234"
    sent = json.loads(route.calls[-1].request.content)
    assert (
        sent["guided_choice"] == list(LABELS) and route.calls[-1].request.headers["X-Tenant-ID"] == "support"
    )
    assert route.calls[-1].request.headers["Authorization"] == "Bearer k"
    assert Recorder(tmp_path, "replay").get(req()) is not None


@respx.mock
def test_client_records_deterministic_rejection(tmp_path: Path) -> None:
    respx.post("http://gw/v1/chat/completions").mock(
        return_value=httpx.Response(400, text="blocked by guardrails")
    )

    async def go() -> CallResult:
        async with ChatClient(Endpoint("cinch", "http://gw", None), Recorder(tmp_path, "live")) as c:
            return await c.chat(req())

    res = asyncio.run(go())
    assert res.http_status == 400 and not res.ok and "guardrails" in (res.error or "")
