-- One row per decision: harness outcome + aggregates over its model calls.
with calls as (
    select * from {{ ref('int_model_calls') }}
),
agg as (
    select
        decision_id,
        count(*)                                        as n_calls,
        sum(cost_usd)                                   as total_cost_usd,
        sum(latency_ms)                                 as total_latency_ms,
        bool_and(http_status = 200)                     as all_calls_ok,
        string_agg(distinct model, ' + ' order by model) as model_mix,
        bool_or(cache_status = 'HIT')                   as any_cache_hit
    from calls
    group by decision_id
)
select
    o.*,
    a.n_calls,
    a.total_cost_usd,
    a.total_latency_ms,
    a.all_calls_ok,
    a.model_mix,
    a.any_cache_hit
from {{ ref('stg_harness__eval_outcomes') }} o
join agg a using (decision_id)
