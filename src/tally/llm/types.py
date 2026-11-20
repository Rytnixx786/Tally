"""Shared data types for model calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float
    top: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ChatRequest:
    """Provider-neutral request. `key_payload()` defines record/replay identity (no ids, no headers)."""

    provider: str
    model: str
    messages: tuple[tuple[str, str], ...]  # (role, content)
    max_tokens: int
    labels: tuple[str, ...]
    constraint: str  # guided_choice | structured_outputs | json_schema | none
    top_logprobs: int = 0
    temperature: float = 0.0
    seed: int = 0
    variant: str = ""  # experiment arm that changes the response (e.g. gateway cache on/off)

    def key_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": [list(m) for m in self.messages],
            "max_tokens": self.max_tokens,
            "labels": list(self.labels),
            "constraint": self.constraint,
            "top_logprobs": self.top_logprobs,
            "temperature": self.temperature,
            "seed": self.seed,
            "variant": self.variant,
        }


@dataclass
class CallResult:
    request_key: str
    provider: str
    model_requested: str
    model_response: str | None
    response_id: str | None
    content: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_ms: float
    started_at: float  # unix seconds
    http_status: int
    error: str | None = None
    cache_status: str | None = None  # HIT | MISS | BYPASS (gateway header)
    billed_usd: float | None = None  # gateway ledger charge, if reported
    gateway_request_id: str | None = None
    token_logprobs: list[TokenLogprob] | None = None
    replayed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.http_status == 200 and self.error is None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("replayed", None)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CallResult:
        d = dict(d)
        tl = d.pop("token_logprobs", None)
        if tl is not None:
            d["token_logprobs"] = [
                TokenLogprob(t["token"], float(t["logprob"]), tuple((a, float(b)) for a, b in t["top"]))
                for t in tl
            ]
        return cls(**d, replayed=True)
