"""Setups B (local via Cinch) and C (hosted): build requests, run them concurrently, parse labels."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tally.llm.client import ChatClient
from tally.llm.logprobs import label_confidence, parse_label
from tally.llm.types import CallResult, ChatRequest
from tally.prompts import build_messages


@dataclass(frozen=True)
class LlmSpec:
    provider: str  # recorder namespace: cinch | openai | gemini | anthropic
    model: str
    constraint: str
    max_tokens: int
    top_logprobs: int
    variant: str = ""


@dataclass
class LlmOutcome:
    result: CallResult
    label: str | None
    confidence: float | None

    @property
    def valid(self) -> bool:
        return self.result.ok and self.label is not None


def make_request(spec: LlmSpec, system: str, ticket_text: str, labels: Sequence[str]) -> ChatRequest:
    msgs = build_messages(system, ticket_text)
    return ChatRequest(
        provider=spec.provider,
        model=spec.model,
        messages=tuple((m["role"], m["content"]) for m in msgs),
        max_tokens=spec.max_tokens,
        labels=tuple(labels),
        constraint=spec.constraint,
        top_logprobs=spec.top_logprobs,
        variant=spec.variant,
    )


def interpret(result: CallResult, labels: Sequence[str]) -> LlmOutcome:
    if not result.ok:
        return LlmOutcome(result, None, None)
    label = parse_label(result.content, labels)
    conf = label_confidence(result.token_logprobs, labels, label) if label is not None else None
    return LlmOutcome(result, label, conf)


async def run_batch(
    client: ChatClient,
    requests: Sequence[ChatRequest],
    headers: Sequence[dict[str, str]],
    labels: Sequence[str],
    *,
    concurrency: int,
    progress: Callable[[int], None] | None = None,
) -> list[LlmOutcome]:
    sem = asyncio.Semaphore(concurrency)
    out: list[LlmOutcome | None] = [None] * len(requests)

    async def one(i: int) -> None:
        async with sem:
            res = await client.chat(requests[i], headers[i])
        out[i] = interpret(res, labels)
        if progress is not None:
            progress(1)

    await asyncio.gather(*(one(i) for i in range(len(requests))))
    return [o for o in out if o is not None]
