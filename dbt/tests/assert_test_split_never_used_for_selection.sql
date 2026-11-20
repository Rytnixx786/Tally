-- Leakage guard: prompt-selection runs only ever happen on the tune split.
select decision_id from {{ ref('fct_outcomes') }} where arm = 'prompt_selection' and split <> 'tune'
