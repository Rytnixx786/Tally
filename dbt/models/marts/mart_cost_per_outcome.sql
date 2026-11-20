-- Rule 1: cost_per_correct = sum(cost) / count(correct) over the SAME rows; never an average of ratios.
select
    (decided_at at time zone 'UTC')::date                                     as day,
    run_id,
    tenant_id,
    feature,
    setup,
    model_mix                                                                 as model,
    prompt_version,
    split,
    arm,
    count(*)::int                                                             as n_decisions,
    count(*) filter (where is_correct)::int                                   as n_correct,
    avg(is_correct::int)::numeric(8, 6)                                       as accuracy,
    sum(total_cost_usd)::numeric(16, 10)                                      as total_cost_usd,
    (sum(total_cost_usd) / nullif(count(*) filter (where is_correct), 0))::numeric(16, 10) as cost_per_correct_usd,
    avg(any_cache_hit::int)::numeric(8, 6)                                    as cache_hit_rate,
    avg(escalated::int)::numeric(8, 6)                                        as escalation_rate,
    (percentile_cont(0.5) within group (order by total_latency_ms))::numeric(14, 3)  as p50_latency_ms,
    (percentile_cont(0.95) within group (order by total_latency_ms))::numeric(14, 3) as p95_latency_ms
from {{ ref('fct_outcomes') }}
group by 1, 2, 3, 4, 5, 6, 7, 8, 9
