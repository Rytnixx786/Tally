-- 100% join rate between harness outcomes and exported spans (a dropped span would bias cost down).
select e.decision_id
from {{ ref('stg_harness__eval_outcomes') }} e
left join {{ ref('fct_outcomes') }} o using (decision_id)
where o.decision_id is null
