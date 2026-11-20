select tenant_id, display_name, owner_project, monthly_budget_usd, traffic_mix
from {{ source('raw', 'tenants') }}
