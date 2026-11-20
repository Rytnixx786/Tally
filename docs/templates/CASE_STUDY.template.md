{# Rendered by `tally report`. Narrative sentences that depend on a result are conditional, so the text stays
   true whichever way the numbers come out. Edit this template, never docs/CASE_STUDY.md. #}
{% set t = m.test %}
{% set A = t.get("A_tfidf") %}{% set B = t.get("B_local") %}{% set C = t.get("C_hosted") %}{% set D = t.get("D_cascade") %}
# Case study: what does a correct support-ticket triage cost?

*Generated from run `{{ m.run.run_id }}` by `tally report`. Every number traces to `marts.fct_outcomes` or to `reports/runs/{{ m.run.run_id }}/`.*

## The headline

{{ headline }}

![Cost vs accuracy](assets/hero_cost_vs_accuracy.png)

## 1. The problem

Teams compare LLM options by price per token. Price per token ignores the thing a business buys: a correct answer. A model that is half the price per token but misroutes one ticket in five costs more once someone re-handles those tickets. Tally measures **cost per correct triage** on held-out data, and asks whether a confidence-based cascade (local model first, hosted model when unsure) beats either model alone.

## 2. Setup

- **Data**: Bitext customer support, 26,872 tickets, 27 intents. 60/20/20 split by near-duplicate cluster (22,037 clusters), stratified by intent. Test n = {{ "{:,}".format(m.paired_n) }}.
- **A**: TF-IDF + logistic regression, trained on {{ "{:,}".format(16124) }} dev tickets (C = {{ m.run.a_chosen_c }}, chosen on tune).
- **B**: Qwen2.5-7B-Instruct AWQ on one RTX 3060 Ti (vLLM 0.6.3, behind Cinch), guided choice over the 27 labels, prompt `{{ m.run.chosen_prompt }}` (chosen on tune).
- **C**: `{{ exp.setups.C_hosted.model }}` ({{ exp.setups.C_hosted.provider }}), strict JSON-schema enum, same prompt.
- **D**: B first; escalate to C when temperature-scaled confidence < τ or B's output is invalid.
- **Cost model**: local GPU = amortised card + measured power, divided by measured throughput at {{ "%.0f" | format(m.run.utilisation * 100) }}% utilisation; hosted = list price per token, with cached prompt tokens at the cached price; TF-IDF = CPU seconds.

## 3. Results

{{ table }}

{% if A and B %}
## 4. The baseline everyone skips
{% if A.accuracy.point >= B.accuracy.point %}
The TF-IDF baseline **beat the 7B model**: {{ pct(A.accuracy) }} vs {{ pct(B.accuracy) }}, at a cost per correct triage {{ "%.0f" | format(B.cost_per_correct_usd.point / A.cost_per_correct_usd.point) }}× lower. On a balanced, template-generated dataset with 16k labelled examples, a linear model over n-grams is hard to beat. That is the point of running it: an LLM is not the default answer for a closed-label task with plenty of labels.
{% else %}
The 7B model beat the TF-IDF baseline: {{ pct(B.accuracy) }} vs {{ pct(A.accuracy) }}. The baseline still costs {{ "%.0f" | format(B.cost_per_correct_usd.point / A.cost_per_correct_usd.point) }}× less per correct triage.
{% endif %}
{% if m.learning_curve %}
The learning curve shows where that changes. With {{ m.learning_curve[0].per_intent | int }} labels per intent the baseline reaches {{ pct(m.learning_curve[0].accuracy) }}; with {{ m.learning_curve[-2].per_intent | int }} per intent, {{ pct(m.learning_curve[-2].accuracy) }}. The prompted local model needs **no** labelled training data (only {{ exp.prompts.fewshot_per_intent }} examples per intent in the prompt) and reaches {{ pct(B.accuracy.point) }}. When labels are scarce, or when the label set changes faster than you can relabel, the LLM earns its cost; when you have thousands of clean labels for a fixed taxonomy, it does not.

![Learning curve](assets/learning_curve.png)
{% endif %}
{% endif %}

{% if C and D %}
## 5. The router

- Raw logprob confidence from the guided-decoding model was {{ "over" if m.calibration.tune.ece_raw > 0.02 else "nearly" }}confident (tune ECE {{ "%.3f" | format(m.calibration.tune.ece_raw) }}). One temperature, fitted on tune (T = {{ "%.2f" | format(m.calibration.temperature) }}), brought test ECE from {{ "%.3f" | format(m.calibration.test.ece_raw) }} to {{ "%.3f" | format(m.calibration.test.ece_calibrated) }}.
- τ = {{ m.run.chosen_tau }} was chosen on tune by the pre-registered rule ({{ m.run.tau_rule }}). On test the cascade escalated {{ pct(D.escalation_rate) }} of tickets.
- Accuracy: D {{ pct(D.accuracy) }} vs C {{ pct(C.accuracy) }} vs B {{ pct(B.accuracy) }}. Cost per 1,000 correct: D {{ usd(D.cost_per_correct_usd, 1000) }}, C {{ usd(C.cost_per_correct_usd, 1000) }}, B {{ usd(B.cost_per_correct_usd, 1000) }}.
- The oracle router (escalate exactly when B is wrong) would reach {{ pct(m.oracle.test.accuracy) }} at a {{ pct(m.oracle.test.escalation_rate) }} escalation rate. The gap between D and the oracle is what a better confidence signal could still win.

![Reliability](assets/reliability.png)
![Tau sweep](assets/tau_sweep.png)
{% endif %}

{% if m.cache %}
## 6. The cache finding

Cinch's semantic cache compares the **whole prompt** (system prompt plus ticket) by term-frequency cosine, at a 0.95 threshold, in one store shared by all tenants. A triage prompt is about 1,500 tokens of fixed instructions and examples plus a ticket of about 10 words, so two different tickets look almost identical to it.

With the cache on, {{ pct(m.cache.hit_rate) }} of test tickets were cache hits, and only {{ pct(m.cache.hit_precision) }} of those hits returned the right label for the ticket being asked. Accuracy fell from {{ pct(m.cache.accuracy_cache_off) }} to {{ pct(m.cache.accuracy_cache_on) }}; {{ "{:,}".format(m.cache.hits_wrong_but_cache_off_right) }} tickets the model answered correctly without the cache came back wrong with it. Cost fell from {{ usd(m.cache.cost_cache_off_usd) }} to {{ usd(m.cache.cost_cache_on_usd) }} for the split, which is a saving on paper and a loss in practice.

The fix is specified in `docs/CINCH_CHANGES.md` (C3): key the cache by tenant and system-prompt hash, and compare only the user message. Headline numbers in this study use the cache-off run.
{% endif %}

## 7. ROI, and how much it depends on assumptions

Base case: {{ roi.minutes_saved_per_correct }} minutes saved per correct triage, ${{ roi.wage_usd_per_hour }}/hour, {{ roi.misroute_penalty_minutes }} minutes lost per misroute, {{ "{:,}".format(roi.tickets_per_month) }} tickets a month.

| Setup | Net USD / month | AI cost / month | Break-even accuracy |
|---|---|---|---|
{% for s, r in m.roi.items() %}
| {{ labels[s] }} | {{ "{:,.0f}".format(r.net_usd_per_month) }} | {{ "{:,.2f}".format(r.ai_cost_usd) }} | {{ pct(r.break_even_accuracy) }} |
{% endfor %}

At these volumes the AI bill is small next to the labour terms for every setup. The decision turns on accuracy (each misroute costs {{ roi.misroute_penalty_minutes }} minutes of an agent) far more than on price per token. The 27-case sensitivity grid in `docs/RESULTS.md` shows the ranking under cheaper and more expensive labour.

## 8. Limitations

- Bitext is synthetic and template-generated; real tickets are longer, noisier and multi-intent. The cluster split removes near-duplicate leakage but cannot make the task realistic.
- Public data may be in either model's training set. Neither model was probed for memorisation of Bitext.
- One local GPU at {{ "%.0f" | format(m.run.utilisation * 100) }}% assumed utilisation; one hosted model at list price on 2026-11-19. Utilisation is the cost-model input that moves local cost the most.
- Prompts were tuned on the local model only; the hosted model might do better with its own prompt.
- Latency for the cascade adds the two calls; a production router could start the hosted call speculatively.

## 9. What I would do next

1. Move the calibrated cascade into Cinch (it already has a cascade stage, driven by regex today) and emit the spans there (`docs/CINCH_CHANGES.md` C1).
2. Fix the semantic cache keying (C3) and re-run the cache arm to measure the honest saving.
3. Repeat the study on real tickets with a changing taxonomy, where the learning curve says LLMs should win.
4. Feed Telltale's live signals (re-opened tickets, agent re-routes) into `fct_outcomes` as `outcome_source = 'telltale_signal'`, so cost per correct can be measured in production, not only on a labelled split.
