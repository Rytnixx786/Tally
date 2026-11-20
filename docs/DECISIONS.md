# Tally Architecture Decision Records

Format: context, decision, alternatives, consequences. Append new ADRs; never rewrite an accepted one. Supersede it with a new ADR instead.

---

## ADR-001: Postgres + dbt as the single warehouse
- **Status**: Accepted (2026-11-19)
- **Context**: Tally's marts are read by Lexicon (text-to-SQL) and Telltale (joins). Volume is small: tens of thousands of spans per run.
- **Decision**: Postgres 16 as the one warehouse, with dbt-postgres for modelling.
- **Alternatives**: ClickHouse is faster at scale and has an OTel exporter, but it adds a second engine and Lexicon would have to target a less common SQL dialect. DuckDB is great for local analysis, but it is single-writer and has no role-based read-only access for Lexicon.
- **Consequences**: Lexicon gets standard Postgres SQL and roles. Scale is not a portfolio claim here.

## ADR-002: OTel Collector → custom ingest service → Postgres
- **Status**: Accepted (2026-11-19)
- **Context**: Collector contrib has no Postgres exporter (RESEARCH §3).
- **Decision**: The collector exports OTLP/HTTP to `tally-ingest` (FastAPI), which flattens spans into `raw.otel_spans` with a JSONB attribute column. dbt staging types the columns.
- **Alternatives**: the `fileexporter` plus a batch loader (simpler, not live); an SDK-level Postgres `SpanExporter` in each app (skips the collector, so every app couples to Postgres).
- **Consequences**: One extra ~150-line service, fully tested with recorded OTLP payloads. The collector still strips content attributes and batches.

## ADR-003: The cascade lives in Tally, not in Cinch
- **Status**: Accepted (2026-11-19)
- **Context**: Cinch's `CascadeRouter` routes 0.5B ↔ 7B on regex heuristics and cannot call a hosted provider.
- **Decision**: Setup D runs client-side in Tally's harness. Cinch's router is disabled for Tally runs.
- **Consequences**: The study does not depend on Cinch features. Moving a calibrated cascade into Cinch later becomes a natural follow-up PR, and a good "what's next".

## ADR-004: Calibrated confidence (temperature scaling), not raw logprob
- **Status**: Accepted (2026-11-19)
- **Context**: Instruction-tuned models are overconfident. Label token lengths differ, which biases sequence logprobs.
- **Decision**: Renormalise the label probability over the label set, then fit one temperature on `tune`. τ applies to the calibrated probability.
- **Consequences**: τ becomes interpretable ("escalate when < 70% likely correct"). The reliability diagram goes into the case study.

## ADR-005: Cluster-based splitting
- **Status**: Accepted (2026-11-19)
- **Context**: Bitext is template-generated, so near-paraphrases would leak across a random split.
- **Decision**: Cluster near-duplicates (MinHash with Jaccard ≥ 0.8 on normalised text), then do a stratified split by cluster.
- **Consequences**: Test accuracy will be lower and more honest. Report the leaky random split too, as a side note, so readers see the effect.

## ADR-006: Misroute penalty in ROI
- **Status**: Accepted (2026-11-19)
- **Decision**: ROI subtracts wrong triages × penalty minutes × wage.
- **Consequences**: Cheap, inaccurate setups stop looking profitable. Break-even accuracy becomes a reportable number.

## ADR-007: Tenants are category mixes, and `internal-docs` is dropped
- **Status**: Accepted (2026-11-19)
- **Context**: Correctness must be checkable. Internal-docs Q&A has no gold labels.
- **Decision**: Tenants `support`, `billing` and `sales` sample Bitext categories with different weights. `finance-ops` (Banking77) is a stretch goal.
- **Consequences**: Per-tenant cost-per-correct differences come from traffic mix and budgets, which is exactly what a FinOps reader cares about.

## ADR-008: Record/replay for reproducibility
- **Status**: Accepted (2026-11-19)
- **Decision**: Every external response is recorded. `make reproduce` defaults to replay, so a reviewer needs no GPU and no keys.
- **Consequences**: Live runs need `TALLY_MODE=live` and the spend flag. Replay files are published as release assets when large.

## ADR-009: Hosted model for setup C is `gpt-4o-mini`
- **Status**: Accepted (2026-11-20)
- **Decision**: `gpt-4o-mini` through the OpenAI chat-completions API with a strict JSON-schema enum over the 27 intents. List price $0.15 / $0.60 per 1M input / output tokens, checked 2026-11-19.
- **Why**: It is non-reasoning (no hidden reasoning tokens to account for) and supports strict structured output. Its list price equals the per-token rates Cinch's FinOps ledger already charges, so "billed" and "true" cost can be compared on the same scale. `gemini-2.5-flash-lite` ($0.10 / $0.40) and Claude Haiku 5.5 ($0.10 / $0.50) are supported by the adapter as alternatives (`config/experiment.yaml: setups.C_hosted`).
- **Consequence**: Setup C and the cascade need an API key and an explicit `--confirm-spend`. The harness prints the estimate first (about $2 for tune + test with the chosen prompt).

## ADR-010: τ selection objective (pre-registered)
- **Status**: Accepted (2026-11-20), before any C or D run on any split.
- **Decision**: Choose τ on **tune** as the value with the lowest cost per correct, subject to tune accuracy ≥ 0.98 × C's tune accuracy. If no τ meets the floor, take the most accurate τ and say so. The grid is 0.00–1.00 in steps of 0.02, applied to the temperature-calibrated confidence.
- **Encoded in**: `config/experiment.yaml: setups.D_cascade` and `src/tally/setups/cascade.py::choose_tau`.

## ADR-011: Stay on vLLM 0.6.3 (Cinch's pinned version) and use `guided_choice`
- **Status**: Accepted (2026-11-20)
- **Context**: CINCH_CHANGES C6 option 2. A live request showed that vLLM 0.6.3 with outlines guided decoding returns logprobs **after** the choice mask: blocked tokens come back at −9999, and valid alternatives keep real values.
- **Decision**: Keep the image Cinch benchmarks with, and renormalise over valid label continuations anyway (`llm/logprobs.py`), which is correct for both raw and processed logprobs.
- **Consequence**: Guided decoding is CPU-bound in this vLLM version, so throughput is lower than unconstrained generation. That measured throughput is what the cost model uses.

## ADR-012: Drop the dynamic few-shot prompt (v2)
- **Status**: Accepted (2026-11-20)
- **Why**: Per-ticket nearest-neighbour examples change the prompt prefix on every request, which defeats vLLM prefix caching and provider prompt caching, the main cost lever for a fixed-prefix workload. The static few-shot prompt (v3) keeps a cacheable prefix. Two candidates (v1 zero-shot, v3 static few-shot) are compared on tune.

## ADR-013: Client-side GenAI spans until Cinch exports its own
- **Status**: Accepted (2026-11-20)
- **Decision**: The harness emits one `tally.decision` span and one GenAI `CLIENT` span per call, with deterministic trace and span ids derived from the decision id, and the recorded timestamps. Spans go over OTLP to the collector, then to tally-ingest, then to Postgres. A replay therefore re-creates identical `raw.otel_spans` rows.
- **Consequence**: The pipeline is the production one (OTLP → collector → warehouse). Only the emitter moves into Cinch later (C1).

## ADR-014: Run Cinch unmodified, with environment overrides
- **Status**: Accepted (2026-11-20)
- **Decision**: Use Cinch's published image with `RATE_LIMIT_RPM` raised, `COMPRESSOR_ENABLED=false`, `CASCADE_ROUTING_ENABLED=false`, `SHADOW_REPLAYER_ENABLED=false`, and the semantic cache off except for the cache arm. Every request also sends `X-Prompt-Compaction: false`.
- **Why**: Tally must not edit Cinch, and the study must measure the prompt it thinks it sends.

## ADR-015: Ship run `main` with setups A and B; C and D deferred
- **Status**: Accepted (2026-11-20, owner's decision)
- **Context**: The hosted model was switched to `gemini-2.5-flash-lite` (ADR-009 amended). On the free tier, that model allows 20 requests per day (HTTP 429, quota `generate_content_free_tier_requests`, observed 2026-11-20), and the study needs about 10,700 hosted calls. The owner chose not to enable billing for now.
- **Decision**: `config/experiment.yaml` sets `C_hosted.enabled: false` and `D_cascade.enabled: false`. The harness, adapter, cascade, calibration, τ selection, report sections and dbt tests for C and D all exist and are covered by the end-to-end test (mocked endpoints). The README headline and case study render the A/B story and say plainly that C and D have not run.
- **To complete**: enable billing on the Google project, set both flags to `true`, then run `tally run --mode auto --confirm-spend` (prints the estimate first, about $1.70), `tally publish`, `tally dbt build` and `tally report`. Recorded responses for A and B are replayed, so only the hosted calls are new.

## ADR-016: Duplicate ticket texts share one recorded response
- **Status**: Accepted (2026-11-20)
- **Context**: The record key is the request content. Bitext has exact duplicate texts within a split, so they produce the same key. In a live pass, vLLM batching can answer two identical requests slightly differently; the recorder keeps the first response, and a replay serves it to both.
- **Decision**: The **replayed** run is the reported one. It is deterministic: `tally reproduce` rebuilt `metrics.json` byte-identically on 2026-11-20. It can differ from the first live pass by a few tickets (tune accuracy for the chosen prompt was 91.1% live and 91.0% replayed).
