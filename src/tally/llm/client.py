"""Async chat clients: OpenAI-compatible (Cinch/vLLM, OpenAI, Gemini's OpenAI endpoint) and Anthropic.

One function, `chat`, takes a provider-neutral ChatRequest plus per-call ids and headers, goes
through the Recorder (record/replay), retries transient failures, and returns a CallResult.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from tally.llm.record_replay import Recorder, ReplayMissError, request_key
from tally.llm.types import CallResult, ChatRequest, TokenLogprob

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Endpoint:
    provider: str
    base_url: str
    api_key: str | None
    timeout_s: float = 60.0

    @classmethod
    def for_provider(
        cls, provider: str, *, base_url_env: str | None = None, api_key_env: str | None = None
    ) -> Endpoint:
        defaults = {
            "cinch": ("CINCH_BASE_URL", "CINCH_API_KEY", "http://localhost:8181"),
            "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY", "https://api.openai.com"),
            "gemini": (
                "GEMINI_BASE_URL",
                "GEMINI_API_KEY",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            "anthropic": ("ANTHROPIC_API_BASE", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
        }
        url_env, key_env, default_url = defaults[provider]
        return cls(
            provider=provider,
            base_url=os.environ.get(base_url_env or url_env, default_url).rstrip("/"),
            api_key=os.environ.get(api_key_env or key_env) or None,
        )


# ------------------------------------------------------------------ payload builders


def openai_payload(req: ChatRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": req.model,
        "messages": [{"role": r, "content": c} for r, c in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.provider != "gemini":  # Gemini's OpenAI-compatible endpoint rejects `seed` (HTTP 400)
        body["seed"] = req.seed
    if req.top_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = req.top_logprobs
    labels = list(req.labels)
    if req.constraint == "guided_choice":  # vLLM < 0.12
        body["guided_choice"] = labels
    elif req.constraint == "structured_outputs":  # vLLM >= 0.12
        body["structured_outputs"] = {"choice": labels}
    elif req.constraint == "json_schema":  # OpenAI / Gemini structured outputs
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "triage",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"intent": {"type": "string", "enum": labels}},
                    "required": ["intent"],
                    "additionalProperties": False,
                },
            },
        }
    return body


def parse_openai(data: Mapping[str, Any]) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    lp = choice.get("logprobs") or {}
    toks: list[TokenLogprob] | None = None
    if lp.get("content"):
        toks = [
            TokenLogprob(
                token=str(t["token"]),
                logprob=float(t["logprob"]),
                top=tuple((str(x["token"]), float(x["logprob"])) for x in (t.get("top_logprobs") or [])),
            )
            for t in lp["content"]
        ]
    return {
        "content": msg.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "model_response": data.get("model"),
        "response_id": data.get("id"),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cached_input_tokens": int(details.get("cached_tokens") or 0),
        "token_logprobs": toks,
    }


def anthropic_payload(req: ChatRequest) -> dict[str, Any]:
    system = "\n\n".join(c for r, c in req.messages if r == "system")
    msgs = [{"role": r, "content": c} for r, c in req.messages if r != "system"]
    return {
        "model": req.model,
        "system": system,
        "messages": msgs,
        "max_tokens": req.max_tokens,
        "temperature": 0,
    }


def parse_anthropic(data: Mapping[str, Any]) -> dict[str, Any]:
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage") or {}
    return {
        "content": text,
        "finish_reason": data.get("stop_reason"),
        "model_response": data.get("model"),
        "response_id": data.get("id"),
        "input_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cached_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "token_logprobs": None,
    }


def _float_header(headers: Mapping[str, str], name: str) -> float | None:
    v = headers.get(name)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


# ------------------------------------------------------------------ the call


class ChatClient:
    def __init__(self, endpoint: Endpoint, recorder: Recorder, *, max_retries: int = 4) -> None:
        self.endpoint = endpoint
        self.recorder = recorder
        self.max_retries = max_retries
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ChatClient:
        self._http = httpx.AsyncClient(timeout=self.endpoint.timeout_s)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http is not None:
            await self._http.aclose()

    def _url_headers_body(
        self, req: ChatRequest, headers: Mapping[str, str]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        h = {"Content-Type": "application/json", **headers}
        if self.endpoint.provider == "anthropic":
            if self.endpoint.api_key:
                h["x-api-key"] = self.endpoint.api_key
            h["anthropic-version"] = "2023-06-01"
            return f"{self.endpoint.base_url}/v1/messages", h, anthropic_payload(req)
        if self.endpoint.api_key:
            h["Authorization"] = f"Bearer {self.endpoint.api_key}"
        base = self.endpoint.base_url
        path = "/chat/completions" if base.endswith(("/v1", "/openai")) else "/v1/chat/completions"
        return base + path, h, openai_payload(req)

    async def chat(self, req: ChatRequest, headers: Mapping[str, str] | None = None) -> CallResult:
        key = request_key(req)
        if self.recorder.mode in ("replay", "auto"):
            hit = self.recorder.get(req)
            if hit is not None:
                return hit
            if self.recorder.mode == "replay":
                raise ReplayMissError(f"no recording for {req.provider}/{req.model} key={key[:12]}")
        if self._http is None:
            raise RuntimeError("ChatClient must be used as an async context manager")
        url, h, body = self._url_headers_body(req, headers or {})
        result: CallResult | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            t0 = time.perf_counter()
            try:
                resp = await self._http.post(url, headers=h, content=json.dumps(body))
            except httpx.HTTPError as exc:
                result = self._failure(req, key, started, t0, 599, f"{type(exc).__name__}: {exc}")
            else:
                result = self._from_response(req, key, started, t0, resp)
            if result.http_status not in RETRY_STATUSES and result.http_status != 599:
                break
            await asyncio.sleep(min(2**attempt, 20) + 0.1 * attempt)
        assert result is not None
        self.recorder.put(result, req)
        return result

    def _failure(
        self, req: ChatRequest, key: str, started: float, t0: float, status: int, err: str
    ) -> CallResult:
        return CallResult(
            request_key=key,
            provider=req.provider,
            model_requested=req.model,
            model_response=None,
            response_id=None,
            content="",
            finish_reason=None,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            started_at=started,
            http_status=status,
            error=err,
        )

    def _from_response(
        self, req: ChatRequest, key: str, started: float, t0: float, resp: httpx.Response
    ) -> CallResult:
        latency = (time.perf_counter() - t0) * 1000
        hdr = resp.headers
        if resp.status_code != 200:
            r = self._failure(req, key, started, t0, resp.status_code, resp.text[:500])
            r.latency_ms = latency
            r.cache_status = hdr.get("X-Semantic-Cache-Status")
            r.gateway_request_id = hdr.get("X-Request-ID")
            return r
        data = resp.json()
        parsed = parse_anthropic(data) if self.endpoint.provider == "anthropic" else parse_openai(data)
        return CallResult(
            request_key=key,
            provider=req.provider,
            model_requested=req.model,
            latency_ms=latency,
            started_at=started,
            http_status=200,
            cache_status=hdr.get("X-Semantic-Cache-Status"),
            billed_usd=_float_header(hdr, "X-FinOps-Request-Cost-USD"),
            gateway_request_id=hdr.get("X-Request-ID"),
            **parsed,
        )
