-- The mart's cost per correct must equal sum(cost)/count(correct) recomputed from fct_outcomes.
with recompute as (
    select (decided_at at time zone 'UTC')::date as day, run_id, tenant_id, setup, model_mix as model,
           prompt_version, split, arm,
           sum(total_cost_usd) / nullif(count(*) filter (where is_correct), 0) as cpc
    from {{ ref('fct_outcomes') }}
    group by 1, 2, 3, 4, 5, 6, 7, 8
)
select m.*
from {{ ref('mart_cost_per_outcome') }} m
join recompute r using (day, run_id, tenant_id, setup, model, prompt_version, split, arm)
where abs(coalesce(m.cost_per_correct_usd, -1) - coalesce(r.cpc, -1)) > 1e-9
