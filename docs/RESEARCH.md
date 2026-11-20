# Tally Research Notes

Checked on 2026-11-19. **[V]** = verified on the linked source that day. **[U]** = from memory or a secondary source; confirm before you rely on it. Prices and model names change; re-check §6 the week you run setup C, and update `price_checked_on` in `config/cost_model.yaml`.

---

## §1 Datasets

**Bitext customer support** [V]: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- 26,872 rows, one `train` split. Columns: `flags`, `instruction`, `category`, `intent`, `response`. License CDLA-Sharing-1.0.
- 27 intents of about 950–1,000 rows each:
  - delivery_options, delivery_period
  - check_refund_policy, get_refund, track_refund
  - recover_password
  - check_cancellation_fee
  - check_payment_methods, payment_issue
  - contact_customer_service, contact_human_agent
  - track_order, place_order, cancel_order, change_order
  - get_invoice, check_invoice
  - edit_account, create_account, delete_account, switch_account, registration_problems
  - complaint, review
  - set_up_shipping_address, change_shipping_address
  - newsletter_subscription
- 11 categories **in the data**:

  | Category | Rows |
  |---|---|
  | ACCOUNT | 5,986 |
  | ORDER | 3,988 |
  | REFUND | 2,992 |
  | CONTACT, INVOICE, PAYMENT, FEEDBACK, DELIVERY | about 2k each |
  | SHIPPING | 1,970 |
  | SUBSCRIPTION | 999 |
  | CANCEL | 950 |

  The dataset card names differ (CANCELLATION_FEE, NEWSLETTER, SHIPPING_ADDRESS). Trust the data.
- `flags` are language-variation tags (colloquial, typos, offensive and so on), not labels. Use them as evaluation slices.
- Implication: the data is template-generated and balanced, so expect a strong TF-IDF baseline. The design in `BUILD_SPEC.md` §0 plans for that.

**Banking77** (stretch) [U]: 13,083 real online-banking queries across 77 intents, from PolyAI (`PolyAI/banking77` on Hugging Face). Confirm the license and row counts before use.

## §2 OpenTelemetry GenAI semantic conventions

- Status **Development**; nothing in `gen_ai.*` is stable [V]. The conventions moved to https://github.com/open-telemetry/semantic-conventions-genai. Inference spans: `docs/gen-ai/client-inference.md` in that repo.
- Span name `{gen_ai.operation.name} {gen_ai.request.model}` [V].
- Required: `gen_ai.operation.name`, `gen_ai.provider.name`.
- Conditionally required: `error.type`, `gen_ai.conversation.id`, `gen_ai.request.model`, `server.port`.
- Recommended: `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`, `server.address`.
- Opt-in: `gen_ai.input.messages`, `gen_ai.output.messages` [V].
- Deprecated: `gen_ai.system` → `gen_ai.provider.name`; `gen_ai.usage.prompt_tokens` / `completion_tokens` → `input_tokens` / `output_tokens` [V]. Registry: https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/
- Content capture: `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`, default `NO_CONTENT`; other values `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT` [V].
- `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` switches library instrumentations to the latest experimental GenAI names [V via secondary sources, e.g. https://github.com/strands-agents/sdk-python/issues/877]. Behaviour differs per package.

## §3 Spans into Postgres

- The collector-contrib exporters include no PostgreSQL or SQL exporter [V]: https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/exporter. Available ones include `clickhouseexporter`, `fileexporter` and `kafkaexporter`. `postgresqlreceiver` scrapes Postgres metrics; it is not a sink.
- Decision: collector → `otlphttp` → small `tally-ingest` service → Postgres (ADR-002).

## §4 dbt

- `dbt-postgres` 1.11.0 (2026-07-16) [V]: https://pypi.org/project/dbt-postgres/
- Staging naming `stg_[source]__[entity]s`, materialised as views, one folder per source [V]: https://docs.getdbt.com/best-practices/how-we-structure/2-staging
- `int_` / `fct_` / `dim_` naming for intermediate and marts [U, community convention].
- Model contracts (`contract: enforced: true`) enforce column names and types at build time [U]. Verify against your pinned dbt version.
- dbt ≥ 1.12 declares semantic models and metrics under `models:` [V]. This matters if Lexicon reuses metric definitions.

## §5 vLLM

- Chat completions accept `logprobs: true` + `top_logprobs: N` [V]: https://docs.vllm.ai/en/v0.16.0/api/vllm/entrypoints/openai/chat_completion/protocol/
- `--max-logprobs` defaults to 20 [V].
- `--logprobs-mode` defaults to `raw_logprobs`, meaning *before* logits processors. `processed_logprobs` gives values after them [V]: https://docs.vllm.ai/en/latest/configuration/engine_args/
- Structured outputs: `structured_outputs: {"choice": [...]}` (also `regex`, `json`, `grammar`, `structural_tag`). The old `guided_*` fields were **removed in v0.12.0** [V]: https://docs.vllm.ai/en/latest/features/structured_outputs.html. Cinch pins v0.6.3, which uses `guided_choice`.
- Inference, not documented: the structured-output mask works as a logits processor, so raw logprobs can put mass on tokens the constraint blocked. Hence the renormalisation in Tally.

## §6 Hosted model prices (standard list, $ per 1M input / output tokens) [V, 2026-11-19]

| Provider | Model | Price |
|---|---|---|
| OpenAI ([pricing](https://developers.openai.com/api/docs/pricing)) | gpt-5-nano | 0.05 / 0.40 |
| | gpt-4o-mini | 0.15 / 0.60 |
| | gpt-5-mini | 0.25 / 2.00 |
| | gpt-4.1-mini | 0.40 / 1.60 |
| Google ([pricing](https://aistudio.google.com/docs/pricing)) | gemini-2.5-flash-lite | 0.10 / 0.40 |
| | gemini-3.1-flash-lite | 0.25 / 1.50 |
| | gemini-2.5-flash | 0.30 / 2.50 (deprecated; routes to 3.5-flash-lite from 2026-12-01) |
| Anthropic ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)) | Claude Haiku 5.5 | 0.10 / 0.50 (≤ 100k prompt tokens) |
| | Claude Haiku 4.5 | 1 / 5 |

Notes:
- Newer Anthropic models use a tokenizer that yields about 30% more tokens for the same text [V]. Compare dollars per decision, never tokens.
- The pricing pages also list newer model names this researcher could not independently confirm. Check the page itself before you choose.

## §7 Cascades and routing: prior work to cite in the case study

- **FrugalGPT** (Chen, Zaharia, Zou 2023), https://arxiv.org/abs/2305.05176 [V]: a learned cascade matched GPT-4 at up to 98% lower cost. Savings vary a lot by dataset.
- **RouteLLM** (LMSYS 2024), https://arxiv.org/abs/2406.18665 [V]: routers trained on preference data cut cost by over 85% on MT-Bench at about 95% of GPT-4's quality, with smaller savings on MMLU and GSM8K.
- **AutoMix** (NeurIPS 2024), https://arxiv.org/abs/2310.12963 [V]: small-model self-verification with a POMDP router.
- **When does confidence-based cascade deferral suffice?** https://arxiv.org/abs/2307.02764 [V]: the theory of the exact rule Tally uses.
- **Rational Tuning of LLM Cascades** (2025), https://arxiv.org/abs/2501.09345 [V]: calibrated per-model confidence plus a probabilistic cascade model. Supports calibrating before thresholding.
- **Gatekeeper** (2025), https://arxiv.org/abs/2502.19335 [V]: fine-tunes the small model for better deferral. A "future work" pointer.
- Length bias in sequence logprob confidence: https://arxiv.org/abs/2404.10136 [V].

How Tally differs from that work: it is a reproducible, **cost-per-correct** accounting on self-hosted hardware, with real GPU amortisation, gateway telemetry and a misroute-penalised ROI. Most papers report API-price savings only.

## §8 GPU cost inputs

- RTX 3060 Ti: 200 W total graphics power (Founders Edition); about 192 W average in gaming-load reviews [V]: https://www.tomshardware.com/reviews/nvidia-geforce-rtx-3060-ti-founders-edition-review/8. Inference draw is often lower [U]. **Measure it.**
- RunPod on-demand, community / secure cloud, $ per hour [V, https://www.runpod.io/pricing]: RTX 3090 0.22 / 0.50; RTX A5000 0.16 / 0.27; L4 0.44 / 0.69; RTX 4090 0.34 / 0.89.
- Electricity: use your own tariff (Bangladesh residential slab rates, converted to USD at a stated rate and date). Cite the tariff source in `cost_model.yaml`.

## §9 Things in your own repos that Tally relies on (read-only)

- Cinch telemetry, cache and FinOps behaviour: see `CINCH_CHANGES.md`, "Findings".
- `Nexus_Researcher/NEXUS_R_Main/backend/app/core/cost.py` already has a per-provider `COST_TABLE` with `estimate_cost()`; unknown models fail open to $0. Tally's cost model should fail **closed**: an unknown model raises an error. Mention the contrast in interviews.

## §10 Observed while building (2026-11-20)

- Gemini API free tier for `gemini-2.5-flash-lite`: **20 generate-content requests per day** (HTTP 429, quota `generativelanguage.googleapis.com/generate_content_free_tier_requests`, limit 20). Billing must be enabled for a study-sized run (ADR-015).
- Gemini's OpenAI-compatible endpoint (`/v1beta/openai/chat/completions`) rejects the `seed` field with HTTP 400 ("Unknown name \"seed\""). It accepts strict `response_format: json_schema` and returns pretty-printed JSON (about 13 completion tokens for `{"intent": ...}`), so `max_tokens: 16` truncated the object; Tally uses 64.
- vLLM 0.6.3.post1 with outlines `guided_choice` returns **post-mask** logprobs (blocked tokens at −9999). It is CPU-bound on this machine: GPU utilisation around 74% with the vLLM process at about 175% CPU, at about 3 to 13 requests per second depending on prefix-cache warmth.
- Cinch, unmodified: ignores client `X-Request-ID`, compresses prompts by default, limits each client IP to 60 requests per minute, and its semantic cache (whole-prompt term-frequency cosine ≥ 0.95, shared across tenants) answered 99.6% of triage requests from cache with wrong labels for most. See `RESULTS.md`.
