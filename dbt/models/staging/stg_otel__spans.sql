-- One row per span, with GenAI semantic-convention and tally.* attributes typed into columns.
-- request_id coalesces Tally's and Cinch's attribute so gateway-emitted spans join the same way (CONTRACTS §1).
with src as (
    select * from {{ source('raw', 'otel_spans') }}
)
select
    trace_id,
    span_id,
    parent_span_id,
    name                                                            as span_name,
    case kind when 1 then 'INTERNAL' when 2 then 'SERVER' when 3 then 'CLIENT' else 'OTHER' end as span_kind,
    start_time,
    end_time,
    extract(epoch from (end_time - start_time)) * 1000.0            as duration_ms,
    status_code,
    service_name,
    coalesce(attributes ->> 'tally.request_id', attributes ->> 'cinch.request_id')::uuid as request_id,
    (attributes ->> 'tally.decision_id')::uuid                      as decision_id,
    attributes ->> 'tally.run_id'                                   as run_id,
    attributes ->> 'tally.setup'                                    as setup,
    attributes ->> 'tally.split'                                    as split,
    attributes ->> 'tally.arm'                                      as arm,
    (attributes ->> 'tally.call_index')::int                        as call_index,
    coalesce(attributes ->> 'tally.tenant_id', attributes ->> 'cinch.tenant_id')        as tenant_id,
    coalesce(attributes ->> 'tally.feature', attributes ->> 'cinch.feature')            as feature,
    coalesce(attributes ->> 'tally.prompt_version', attributes ->> 'cinch.prompt_version') as prompt_version,
    attributes ->> 'gen_ai.operation.name'                          as operation_name,
    attributes ->> 'gen_ai.provider.name'                           as provider,
    attributes ->> 'gen_ai.request.model'                           as request_model,
    attributes ->> 'gen_ai.response.model'                          as response_model,
    (attributes ->> 'gen_ai.usage.input_tokens')::int               as input_tokens,
    (attributes ->> 'gen_ai.usage.output_tokens')::int              as output_tokens,
    coalesce((attributes ->> 'gen_ai.usage.cache_read.input_tokens')::int, 0) as cached_input_tokens,
    coalesce(attributes ->> 'tally.cache_status', attributes ->> 'cinch.cache.status', 'BYPASS') as cache_status,
    coalesce((attributes ->> 'tally.http_status')::int, (attributes ->> 'cinch.http.status_code')::int, 200) as http_status,
    (attributes ->> 'tally.latency_ms')::double precision           as latency_ms,
    (attributes ->> 'tally.confidence')::double precision           as confidence,
    (attributes ->> 'tally.validation_passed')::boolean             as validation_passed,
    coalesce((attributes ->> 'tally.escalation')::boolean, false)   as is_escalation,
    coalesce((attributes ->> 'tally.reused_response')::boolean, false) as reused_response,
    attributes ->> 'tally.predicted_label'                          as predicted_label,
    (attributes ->> 'cinch.finops.billed_usd')::double precision    as billed_usd,
    attributes ->> 'error.type'                                     as error_type
from src
