select prompt_version, feature, prompt_hash, template_path, n_fewshot, description, created_at
from {{ source('raw', 'prompt_registry') }}
