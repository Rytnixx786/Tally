-- Spans and outcomes must agree: each decision has exactly n_calls request rows, and its final request exists.
select o.decision_id
from {{ ref('fct_outcomes') }} o
left join (select decision_id, count(*) as n from {{ ref('fct_requests') }} group by 1) r using (decision_id)
left join {{ ref('fct_requests') }} f on f.request_id = o.final_request_id
where r.n is distinct from o.n_calls or f.request_id is null
