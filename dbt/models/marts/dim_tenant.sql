select
    tenant_id,
    display_name,
    owner_project,
    monthly_budget_usd::numeric(12, 2) as monthly_budget_usd,
    traffic_mix
from {{ ref('stg_harness__tenants') }}
