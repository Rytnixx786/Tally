-- Model-call spans (CLIENT) joined to Tally's per-call cost. A call without a cost row is a bug,
-- caught by assert_every_call_is_costed.
with spans as (
    select * from {{ ref('stg_otel__spans') }}
    where span_kind = 'CLIENT' and request_id is not null
),
costs as (
    select * from {{ ref('stg_harness__call_costs') }}
)
select
    s.request_id,
    s.decision_id,
    s.trace_id,
    s.run_id,
    s.start_time                                   as requested_at,
    s.tenant_id,
    s.feature,
    s.prompt_version,
    s.setup,
    s.split,
    s.arm,
    s.call_index,
    s.provider,
    coalesce(s.response_model, s.request_model)    as model,
    s.input_tokens,
    s.output_tokens,
    s.cached_input_tokens,
    coalesce(s.latency_ms, s.duration_ms)          as latency_ms,
    s.cache_status,
    s.http_status,
    s.is_escalation,
    s.reused_response,
    s.confidence,
    coalesce(s.validation_passed, false)           as validation_passed,
    s.predicted_label,
    c.cost_usd,
    c.cost_method,
    s.billed_usd
from spans s
left join costs c using (request_id)
