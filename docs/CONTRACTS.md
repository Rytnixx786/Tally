# Cross-Project Contracts (v1)

Tally owns this file. Lexicon and Telltale depend on it and link here instead of copying it. Change it only with a version bump and a note in each dependent repo's `docs/DECISIONS.md`.

Tally enforces the warehouse half with **dbt model contracts** (`config: contract: {enforced: true}`), so a column rename fails `dbt build` before it can break Lexicon.

---

## 1. Identifiers

| Field | Format | Who creates it | Who carries it |
|---|---|---|---|
| `request_id` | UUIDv4 string, lowercase | The **caller** (Tally harness, Lexicon agent, the Telltale SDK host app) creates it before calling Cinch | Sent as `X-Request-ID`; Cinch echoes it and puts it on the span as `cinch.request_id`; Telltale events carry it; Tally's `fct_requests.request_id` is the join key |
| `trace_id` | W3C `traceparent` | Caller or Cinch | OTel spans |
| `tenant_id` | lowercase slug `[a-z0-9-]{2,32}` | Config | `X-Tenant-ID` header |
| `feature` | lowercase snake case | Caller | `X-Feature` header |
| `prompt_version` | `<feature>.<variant>.v<N>`, for example `ticket_triage.fewshot.v2` | Caller; registered in `config/prompts/` | `X-Prompt-Version` header |
| `prompt_hash` | sha256 of the rendered template with placeholders unfilled, first 12 hex chars | Tally prompt registry | `dim_prompt_version.prompt_hash` |

**Apps that do not call through Cinch** (MindStack calls Ollama directly; MaaSheba calls Groq and Gemini) emit their own GenAI client span to Tally's collector, with the same ids under the `tally.` prefix: `tally.request_id`, `tally.tenant_id`, `tally.feature`, `tally.prompt_version`. Tally's `stg_otel__spans` coalesces `cinch.request_id` and `tally.request_id` (likewise for tenant, feature and prompt version). Providers without a price-book entry make dbt fail (`price_book_covers_all_models` test). They are never silently costed at $0.

Rule: one `request_id` per **model call**. A cascade that calls B and then C produces two requests that share a `decision_id` (Tally-side) and a `trace_id`.

## 2. Tenant registry

| `tenant_id` | Owner project | Purpose | Notes |
|---|---|---|---|
| `support` | Tally | High-volume L1 triage mix | Bitext categories ORDER, DELIVERY, SHIPPING, CONTACT |
| `billing` | Tally | Money-related triage | PAYMENT, REFUND, INVOICE, CANCEL |
| `sales` | Tally | Account and retention triage | ACCOUNT, SUBSCRIPTION, FEEDBACK |
| `finance-ops` | Tally (stretch) | Second, harder workload | Banking77, see `BUILD_SPEC.md` §3.4 |
| `lexicon` | Lexicon | The SQL agent's own LLM calls | Lets Tally report Lexicon's cost per question |
| `telltale-demo` | Telltale | Demo app traffic | Instrumented demo / fallback app |
| `masheba` / `mindstack` | Telltale | Real product traffic, if instrumented | Only after consent and privacy review |

Category names above are the values found **in the data** (11 categories: ACCOUNT, ORDER, REFUND, CONTACT, INVOICE, PAYMENT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL). The dataset card lists different names (CANCELLATION_FEE, NEWSLETTER, SHIPPING_ADDRESS); always trust the data. Each tenant samples its categories with weights (not exclusively), so every tenant sees some tickets from every category. The final mapping is generated in Week 1 from the actual Bitext category list and committed as `config/tenants.yaml`. This table is the intent; the YAML file is the authority.

## 3. Warehouse layout

- One Postgres 16 instance, database `tally`.
- Schemas:
  - `raw`: append-only landing tables (`raw.otel_spans`, `raw.eval_outcomes`, `raw.prompt_registry`, `raw.price_book`)
  - `staging`, `intermediate`: dbt internals, no external readers
  - `marts`: **the public contract**. Lexicon and Telltale read only from here
  - `telltale_raw`, `telltale_marts`: owned by Telltale's own dbt project, in the same instance
- Roles:
  - `tally_writer`: ingest and dbt
  - `lexicon_ro`: `USAGE` on `marts`, `SELECT` only, `default_transaction_read_only = on`, `statement_timeout = '5s'`
  - `telltale_rw`: owns the `telltale_*` schemas; read-only on `marts`

## 4. Public mart tables (`marts` schema)

Column types are Postgres types. Names marked † are the grain. The authoritative, enforced definition is `dbt/models/marts/_marts.yml`; this page is the readable summary. `fct_outcomes` also carries `ticket_id`, `model_mix`, `predicted_category`, `gold_category`, `valid`, `escalated`, `any_cache_hit`, `confidence`, `calibrated_confidence` and `flags`.

### `marts.fct_requests`: one row per model call
| column | type | meaning |
|---|---|---|
| request_id † | uuid | Join key across all projects |
| run_id | text | Harness run (NULL for live traffic) |
| arm | text | `primary` (reported configuration), `prompt_selection` (non-chosen prompts on tune), `cache_on` (gateway semantic cache enabled) |
| call_index | int | Position of the call within its decision |
| decision_id | uuid | Groups the calls made for one ticket or question (cascade = 2 calls) |
| trace_id | text | OTel trace id |
| requested_at | timestamptz | Span start |
| tenant_id | text | FK → dim_tenant |
| feature | text | `ticket_triage`, `sql_agent`, … |
| prompt_version | text | FK → dim_prompt_version |
| setup | text | `A_tfidf`, `B_local`, `C_hosted`, `D_cascade` (Tally runs only; NULL for other traffic) |
| provider | text | `vllm`, `openai`, `google`, `anthropic`, `sklearn` |
| model | text | Response model name |
| input_tokens | int | |
| output_tokens | int | |
| latency_ms | numeric | End to end |
| ttft_ms | numeric | NULL when not streamed |
| cache_status | text | `HIT`, `MISS`, `BYPASS` |
| http_status | int | |
| is_escalated | boolean | This call happened because a cascade escalated |
| confidence | numeric | Label probability from logprobs (NULL if unavailable) |
| validation_passed | boolean | Output parsed and matched the label set |
| cached_input_tokens | int | Prompt tokens served from a provider prompt cache |
| reused_response | boolean | The cascade reused the identical recorded response of B or C |
| cost_usd | numeric(14,10) | Tally's cost model (see BUILD_SPEC §6) |
| billed_usd | numeric(14,10) | What Cinch's ledger charged, for comparison |
| split | text | `dev`, `tune`, `test`, or NULL for live traffic |

### `marts.fct_outcomes`: one row per decision
| column | type | meaning |
|---|---|---|
| decision_id † | uuid | |
| decided_at | timestamptz | When the final answer was produced |
| run_id | text | Tally harness run id (NULL for live traffic) |
| final_request_id | uuid | The call whose answer was used |
| tenant_id, feature, prompt_version, setup, split | | Same as above |
| predicted_label | text | |
| gold_label | text | NULL for live traffic until an outcome arrives |
| is_correct | boolean | `predicted_label = gold_label`, or the live outcome signal |
| outcome_source | text | `gold_label`, `human_review`, `telltale_signal` |
| total_cost_usd | numeric(12,8) | Sum over all calls in the decision |
| total_latency_ms | numeric | |
| n_calls | int | |

### `marts.dim_tenant`
`tenant_id` †, `display_name`, `owner_project`, `monthly_budget_usd`, `traffic_mix` (jsonb), `created_at`

### `marts.dim_prompt_version`
`prompt_version` †, `feature`, `prompt_hash`, `template_path`, `n_fewshot`, `description`, `created_at`

### `marts.mart_cost_per_outcome`: one row per (day, run, tenant, feature, setup, model mix, prompt_version, split, arm)
`day`, `run_id`, `arm`, `tenant_id`, `feature`, `setup`, `model`, `prompt_version`, `split`, `n_decisions`, `n_correct`, `accuracy`, `total_cost_usd`, `cost_per_correct_usd` (NULL when n_correct = 0), `cache_hit_rate`, `escalation_rate`, `p50_latency_ms`, `p95_latency_ms`

Metric definitions live in one place, `Tally/dbt/models/marts/_marts.yml`, and Lexicon's context YAML copies them **verbatim** (Lexicon has a test that diffs them).

## 5. Business rules every consumer must respect

1. `cost_per_correct_usd = total_cost_usd / n_correct` over the **same** set of decisions. Never average per-row ratios.
2. A cascade decision's cost includes both calls, even when the local answer was discarded.
3. Cache hits count as decisions, with `cost_usd` taken from the cost model's `cache_hit_cost_usd` (near zero), not from the Cinch ledger.
4. Headline numbers use `split = 'test'` only. `tune` is for τ selection; `dev` is for prompt work.
5. Requests with `http_status <> 200` count as failed decisions (`is_correct = false`), never as missing.

## 6. Telltale event contract (summary)

Telltale owns the full schema in `Telltale/docs/BUILD_SPEC.md` §4. The join contract is: every AI-related event carries `request_id` (and `decision_id` when known), so `telltale_marts.*` can join to `marts.fct_requests` for cost.

## 7. ARES gate contract (summary)

Lexicon writes eval results in the format ARES expects (see `Lexicon/docs/BUILD_SPEC.md` §9). Tally can also be gated by ARES on the cascade's test accuracy if you choose. This is optional for Tally.
