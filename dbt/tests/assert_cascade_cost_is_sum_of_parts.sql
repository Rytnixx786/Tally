-- A cascade decision pays for the local call and, when escalated, the hosted call too (CONTRACTS §5 rule 2).
select o.decision_id
from {{ ref('fct_outcomes') }} o
where o.setup = 'D_cascade'
  and ((o.escalated and o.n_calls <> 2) or (not o.escalated and o.n_calls <> 1))
