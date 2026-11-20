# Changes Tally Needs From Cinch

Tally never edits the Cinch repository. This file is the spec for a pull request **you** open in `H:\Projects\Cinch` (branch `feat/tally-telemetry`) in Week 1. Follow Cinch's own `AGENTS.md`: plan first, tests with every change, conventional commits, and a benchmark delta if the serving path changes.

Each item below cites the code it touches, as read on 2026-11-19. Re-check the line numbers before you start.

---

## Findings from reading Cinch (why these changes are needed)

| # | What the code does today | Why it breaks Tally |
|---|---|---|
| F1 | `gateway/telemetry.py` builds `OpenTelemetrySpan` objects in memory. `span.finish()` records an end time, and nothing exports the span. There is no OTel SDK, no OTLP exporter, and no collector in `docker-compose.yml`. | Tally's whole pipeline starts from exported spans. |
| F2 | `gateway/app.py` sets `X-Request-ID` (around line 594) **after** the semantic-cache fast path returns (around lines 529-556). | Cache hits have no request id, so they cannot join to outcomes. Cache savings become unmeasurable. |
| F3 | `gateway/semantic_cache.py` is a single global store keyed on the concatenated text of *all* messages, compared with bag-of-words term-frequency cosine at a threshold of 0.95. | (a) The cache is shared across tenants, which leaks answers between tenants. (b) A triage prompt has a long fixed system prompt (the label list and few-shot examples) and a short ticket. The fixed part dominates the term-frequency vector, so two *different* tickets can score above 0.95 and the second ticket gets the first ticket's label. Tally must measure this; see `BUILD_SPEC.md` §7.4. |
| F4 | On a cache hit, `finops.record_usage(...)` charges the tenant `estimated_tokens` prompt + 20 completion tokens. | The ledger bills cache hits as if they ran on the GPU. Tally prices cache hits as near zero, so Tally's cost and Cinch's ledger will disagree unless the hit is flagged. |
| F5 | FinOps prices come from fixed per-token rates (`prompt_rate_per_1k=0.00015`, `completion_rate_per_1k=0.00060` in `gateway/finops.py`). These are hosted-API-style list prices, not the cost of your GPU. | Tally computes local cost from GPU-hours (see `BUILD_SPEC.md` §6). Cinch's number stays useful as "chargeback price" and Tally reports both. |
| F6 | The tenant comes from `X-Tenant-ID` or `body.tenant_id`; the team comes from `X-Team-ID`. There is no prompt-version or feature field. | Tally needs `prompt_version` and `feature` on every span to build `dim_prompt_version` and Telltale's per-feature analysis. |
| F7 | The upstream body forwards all fields except `priority`. | `logprobs` / `top_logprobs` already reach vLLM. No change needed, but add a test that pins this behaviour. |
| F9 | Observed 2026-11-20: Cinch **ignores a client-supplied `X-Request-ID`** and returns its own 8-character id. | The caller cannot pre-assign the join key. Tally sends its own UUID, keeps Cinch's id as `gateway_request_id`, and relies on client spans until C2 lands. |
| F10 | Observed: the prompt compactor is **on by default** and rewrites prompts unless the client sends `X-Prompt-Compaction: false`; the per-IP rate limit defaults to 60 RPM. | A study that forgets the header measures a different prompt than it thinks. Tally sends the header and raises the limits through environment variables. |
| F8 | `CascadeRouter` (`gateway/cascade_router.py`) routes on regex complexity heuristics between 0.5B and 7B tiers. | Not the router Tally studies. Tally's cascade (setup D) lives client-side in Tally so it can escalate to a hosted model Cinch does not serve. Leave Cinch's router alone and disable it for Tally runs. |

---

## Required changes (PR scope)

### C1. Real OpenTelemetry export with GenAI semantic conventions (fixes F1)

- Add `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` (and optionally `opentelemetry-instrumentation-fastapi`).
- Create one span per chat completion. Span kind `CLIENT` for the call to vLLM, as the GenAI conventions specify for model calls. The existing gateway span can become the parent `SERVER` span.
- Span name: `chat {gen_ai.request.model}`.
- The GenAI conventions are still at **Development** status and now live in `open-telemetry/semantic-conventions-genai`. Emit the current names (`gen_ai.provider.name`, not the deprecated `gen_ai.system`; `gen_ai.usage.input_tokens`/`output_tokens`, not `prompt_tokens`/`completion_tokens`), record the semconv version you followed as a resource attribute, and see `RESEARCH.md` §2. Library instrumentations may also need `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`; check each package's README.

| Attribute | Source in Cinch |
|---|---|
| `gen_ai.operation.name` = `chat` | constant |
| `gen_ai.provider.name` = `vllm` (older spec versions call this `gen_ai.system`) | constant |
| `gen_ai.request.model` | `body["model"]` |
| `gen_ai.response.model` | upstream response `model` |
| `gen_ai.response.id` | upstream response `id` |
| `gen_ai.request.max_tokens`, `gen_ai.request.temperature` | body |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | upstream `usage` |
| `gen_ai.response.finish_reasons` | upstream `choices[].finish_reason` |
| `gen_ai.usage.cache_read.input_tokens` | vLLM prefix-cache hit tokens, if the response reports them |
| `server.address`, `server.port` | vLLM upstream host and port |
| `error.type` | Set when the call errors (for example `timeout`, `503`) |

Cinch-specific attributes use a `cinch.` namespace so they never collide with the spec:

| Attribute | Meaning |
|---|---|
| `cinch.request_id` | Same value as the `X-Request-ID` header (C2) |
| `cinch.tenant_id`, `cinch.team_id` | From the headers |
| `cinch.prompt_version`, `cinch.feature` | New headers `X-Prompt-Version`, `X-Feature` (C4) |
| `cinch.cache.status` | `HIT` / `MISS` / `BYPASS` |
| `cinch.cache.similarity` | Best cosine score |
| `cinch.cache.source_request_id` | On a hit, the request id of the stored response |
| `cinch.queue.wait_ms`, `cinch.ttft_ms` | Existing timings |
| `cinch.finops.billed_usd` | What the ledger charged (F4/F5) |
| `cinch.http.status_code` | Including 402, 429, 400 and 503 outcomes |

- Do **not** capture prompt or completion text by default. Honour the standard switch `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` (default `NO_CONTENT`; other values `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT`), which is the variable the GenAI conventions use [Tally RESEARCH §2]. Tally does not need text in spans; the eval harness keeps its own record of inputs and outputs.
- Exporter endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://otel-collector:4318`). If the endpoint is unset, telemetry becomes a no-op and serving continues. Exporting must never fail a request: use `BatchSpanProcessor`.
- Test: an in-memory span exporter (`InMemorySpanExporter`) asserts the attribute set for a normal request, a cache hit, a 402 and a 429.

### C2. Request id on every response, set before any early return (fixes F2)

- Generate `req_id` at the top of the handler, or accept a client-supplied `X-Request-ID` if it is a valid UUID. Tally supplies its own so the eval harness knows the id before the response arrives.
- Return `X-Request-ID` on cache hits, 402, 429, 400 and 503 responses too.
- Test: cache-hit response includes `X-Request-ID` and the span carries the same value.

### C3. Tenant-scoped, prompt-aware semantic cache (fixes F3 and F4)

- Cache key namespace = `(tenant_id, sha256(system prompt), model)`. Similarity runs only over the **last user message** within that namespace.
- New header `X-Cache-Control: no-store` / `bypass` so experiments can turn the cache off per request.
- On a hit, record `billed_usd` as 0, or as a configurable `cache_hit_price_usd`, and tag the ledger row as a cache hit.
- Test: two different tickets with the same long system prompt do **not** hit each other. A ticket from tenant A never returns tenant B's cached answer.
- Keep the old behaviour behind a flag (`SEMANTIC_CACHE_LEGACY_KEYING=true`) so Tally can measure the false-hit rate before and after. That before/after measurement is a strong case-study finding.

### C4. Attribution headers (fixes F6)

- Accept `X-Prompt-Version` (string, for example `triage-v2`) and `X-Feature` (for example `ticket_triage`, `sql_agent`, `chat_assistant`). Pass both to the span. Never forward them upstream.

### C5. Collector service in Docker Compose

- Add `otel/opentelemetry-collector-contrib` to `docker-compose.yml` under a `telemetry` profile, so the default stack is unchanged. Tally ships its own compose file with the collector and Postgres; Cinch only needs the endpoint env var. Pick one owner and document it. **Recommendation:** Tally owns the collector and Cinch only exports.

### C6. vLLM version decision (affects Tally's confidence signal)

Cinch pins `vllm/vllm-openai:v0.6.3.post1`. Two facts matter [RESEARCH §5]:
- Constrained choice is `guided_choice` on 0.6.x, but newer vLLM uses `structured_outputs: {"choice": [...]}`. The `guided_*` fields were removed in v0.12.0.
- Newer vLLM has `--logprobs-mode` (default `raw_logprobs`, i.e. *before* logits processors such as the structured-output mask) and `--max-logprobs` (default 20).

Choose one:
1. **Upgrade vLLM in Cinch** (recommended if the benchmarks still reproduce on the 3060 Ti). Start it with `--logprobs-mode processed_logprobs`, so returned logprobs reflect the constrained distribution, and re-run Cinch's benchmark suite. This is its own Cinch PR, with a benchmark delta.
2. **Stay on 0.6.3.** Tally then sends `guided_choice` and validates empirically, using a recorded response, whether the logprobs reflect the masked distribution. If they don't, compute confidence by renormalising over the label set (Tally does this anyway).

Tally's client hides the difference behind one function (`llm/cinch_client.py::constrain_to_choices`), so the decision does not leak into the harness.

### C7. Pin the pass-through behaviour (F7)

- Test that `logprobs: true, top_logprobs: 5` and vLLM's structured-output fields reach the upstream request untouched, and that the response's `choices[0].logprobs` comes back untouched.

---

## Acceptance checklist for the Cinch PR

- [ ] `pytest` green, coverage not lower than before
- [ ] Span attribute tests for: success, cache hit, 402, 429, upstream 503
- [ ] Cross-tenant cache isolation test
- [ ] Same-system-prompt false-hit regression test
- [ ] Benchmark delta reported in the PR description (export overhead on p50/p95; expect < 1 ms with the batch processor)
- [ ] `docs/design-decisions.md` in Cinch gains an entry "Telemetry export for cost-per-outcome analytics"

## If you cannot change Cinch in time

Tally's eval harness can emit the same spans **client-side** with the OTel SDK around each Cinch call, using response headers for cache status and cost. Implement this as `tally.telemetry.client_spans` behind `TALLY_SPAN_SOURCE=client`. It is a fallback, not the goal: the portfolio story is stronger when the gateway itself emits standard telemetry.
