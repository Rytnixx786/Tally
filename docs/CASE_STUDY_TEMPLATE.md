# Case Study: Cost per Correct Triage

> Fill this in during Week 5. Every number comes from `make report` (marts or `reports/runs/<run_id>/`); cite the run id. Apply stop-slop before publishing.

**Headline**: Router achieved **{X}%** (95% CI {a}–{b}) of the hosted model's accuracy at **{Y}%** (CI {c}–{d}) of its cost per correct triage. Run `{run_id}`.

## 1. Problem (3–4 sentences)
Why cost per token misleads, and what "correct" means here.

## 2. Setup
- Data: Bitext (26,872 rows, 27 intents), cluster split 60/20/20, test n = {n}.
- Setups A–D (one line each). Hardware: RTX 3060 Ti, vLLM {ver}, Qwen2.5-7B-Instruct-AWQ. Hosted model: {model} at ${in}/${out} per 1M tokens (checked {date}).
- Cost model: owned GPU, amortised over {years} years at {util}% utilisation and {W} W measured. Alternative: cloud GPU-hour at ${rate}.

## 3. Results
Hero chart. Table: accuracy, macro-F1, cost per 1k tickets, cost per correct, p95 latency, each with a CI. McNemar results.

## 4. What the TF-IDF baseline taught me
Learning curve. When an LLM is worth it.

## 5. Calibration and the router
Reliability diagram before and after. τ frontier, chosen τ, oracle gap, which intents escalate.

## 6. The cache finding
Hit rate, hit precision, false hits, net dollars saved. Legacy vs fixed keying.

## 7. ROI and its assumptions
Sensitivity grid and break-even accuracy. The three assumptions that move it most.

## 8. Limitations
Synthetic, template-generated data; possible contamination ({probe result}); a single GPU; a single hosted model; prices as of {date}.

## 9. What I'd do next
For example: move the calibrated cascade into Cinch; learned routers (RouteLLM-style); live outcome signals from Telltale.
