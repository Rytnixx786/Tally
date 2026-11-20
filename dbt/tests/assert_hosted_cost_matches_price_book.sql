-- Hosted calls: cost = uncached input x price + cached input x cached price + output x price (fails closed if a hosted model has no price-book entry).
select c.request_id, c.provider, c.model, c.cost_usd, p.input_per_mtok
from {{ ref('int_model_calls') }} c
left join {{ ref('stg_harness__price_book') }} p
  on p.provider = c.provider and c.model like p.model || '%'
where c.provider not in ('vllm', 'sklearn')
  and c.http_status = 200
  and (p.provider is null
       or abs(c.cost_usd - ((c.input_tokens - c.cached_input_tokens) * p.input_per_mtok
                            + c.cached_input_tokens * p.cached_input_per_mtok
                            + c.output_tokens * p.output_per_mtok) / 1e6) > 1e-9)
