select request_id, run_id, cost_usd::numeric(14, 10) as cost_usd, cost_method
from {{ source('raw', 'call_costs') }}
