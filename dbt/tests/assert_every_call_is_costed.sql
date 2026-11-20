-- Every model call must have a cost from Tally's cost model; nothing is silently costed at $0 or NULL.
select request_id, provider, model from {{ ref('int_model_calls') }} where cost_usd is null
