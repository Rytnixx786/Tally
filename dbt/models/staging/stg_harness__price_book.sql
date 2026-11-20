-- Latest checked price per (provider, model).
select distinct on (provider, model)
    provider, model, input_per_mtok, output_per_mtok,
    coalesce(cached_input_per_mtok, input_per_mtok) as cached_input_per_mtok, checked_on, source
from {{ source('raw', 'price_book') }}
order by provider, model, checked_on desc
