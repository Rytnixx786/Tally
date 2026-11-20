select
    prompt_version,
    feature,
    prompt_hash,
    template_path,
    n_fewshot,
    description,
    created_at
from {{ ref('stg_harness__prompt_registry') }}
