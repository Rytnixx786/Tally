# Tally Roadmap: 5 weeks (portfolio weeks 1–5)

> Progress as of 2026-11-20: built end to end; A, B and the cache arm measured; C and D pending billing (ADR-015).

Each task is one checkpoint: plan, build, test, stop and report (see `AGENTS.md`). Tick the boxes as you go. Each week ends with an **exit check**; do not start the next week until it passes, or until you have written down why you are moving on anyway.

---

## Week 1: Foundations and the Cinch PR

**Goal**: data is split and sealed, the warehouse is running, and spans flow end to end for one request.

- [x] **T1.1 Repo scaffold.** `uv init`, `pyproject.toml`, ruff, mypy, pytest and pre-commit; Makefile skeleton; CI workflow running lint + test.
- [x] **T1.2 Data + splits.**
  - `tally data` downloads Bitext, asserts the 27 intents and 11 categories, and clusters near-duplicates (ADR-005).
  - It writes the 60/20/20 split and `data/splits/manifest.json`.
  - Tests: label sets, split sizes, no cluster spans two splits, reproducible hash.
- [x] **T1.3 Config models.** Pydantic models for `experiment.yaml`, `tenants.yaml`, `cost_model.yaml` and `roi.yaml`, with sample files and tests.
- [x] **T1.4 Warehouse.** docker-compose with Postgres and the collector; `raw` schema DDL; roles (`tally_writer`, `lexicon_ro`).
- [ ] **T1.5 Cinch PR**, in the Cinch repo, following Cinch's own AGENTS.md. Items C1–C4 and C7 from `CINCH_CHANGES.md`, plus the vLLM decision (C6). If it is not merged by Friday, switch on the client-span fallback and continue. *(not done: Tally never edits Cinch; the client-span fallback is in use, ADR-013)*
- [x] **T1.6 Decisions.** Fill in ADR-009 (hosted model choice) and ADR-010 (τ objective).

**Exit check**: one curl through Cinch with `X-Request-ID` produces one row in `raw.otel_spans` with all required `gen_ai.*` attributes. The split manifest is committed.

## Week 2: Ingest, setups A and B

- [x] **T2.1 tally-ingest.** OTLP/HTTP JSON and protobuf → `raw.otel_spans`; idempotent. Tests use a recorded payload.
- [x] **T2.2 Record/replay layer** with a miss-is-an-error replay mode, plus the spend guard (`--confirm-spend`).
- [x] **T2.3 Setup A.** TF-IDF + LR, tuned on `tune`, with the learning curve (10/25/50/100/all per intent) and client-side spans.
- [x] **T2.4 Prompt registry.** Jinja templates v1, v2 and v3 → `raw.prompt_registry` with hashes.
- [x] **T2.5 Setup B.** Cinch client: choice constraint, logprobs and label renormalisation. Run on `dev` for prompt iteration, then pick a prompt on `tune`.
- [x] **T2.6 Throughput + power measurement.** Run B at the concurrency you will report, logging `nvidia-smi` power every 500 ms. Save results to `reports/runs/<id>/hardware.json`.

**Exit check**: A and B have tune-split accuracy with CIs in `reports/runs/`, and the spans for both are in Postgres.

## Week 3: Setups C and D, calibration, dbt

- [ ] **T3.1 Setup C** adapter. Run on `tune` (spend flag, estimate printed first). *(adapter built and tested; not run, ADR-015)*
- [ ] **T3.2 Calibration.** Temperature scaling on `tune`; ECE and reliability diagram. *(built and tested; runs with C)*
- [ ] **T3.3 Setup D.** τ sweep on `tune`; choose τ with the pre-registered objective (ADR-010); add the oracle router. *(built and tested; runs with C)*
- [x] **T3.4 dbt staging + intermediate.** `stg_*`, `int_model_calls` with the cost model, `int_decisions`.
- [x] **T3.5 dbt marts with contracts.** Build all five public marts plus tests, matching `CONTRACTS.md` exactly.
- [x] **T3.6 Leakage test** in CI.

**Exit check**: `dbt build` is green, and `mart_cost_per_outcome` shows all four setups on `tune`.

## Week 4: The sealed test run, cache study, dashboard

- [ ] **T4.1 Test-split run.** Run every setup once, recorded, with the run manifest (model versions, vLLM version, prices, τ, git sha). Do not iterate after this. If you find a bug, fix it, rerun everything, and note it in the case study. *(done for A, B and the cache arm; C and D pending, ADR-015)*
- [x] **T4.2 Stats.** Bootstrap CIs, McNemar with Holm correction, and the headline-sentence generator.
- [x] **T4.3 Cache arms.** Cache on and off; hit precision; net savings. Use legacy and fixed keying if both are available.
- [x] **T4.4 Dashboard.** Overview, hero chart, cost per correct, calibration, cache, escalations.
- [x] **T4.5 ROI page.** Sensitivity grid, tornado chart, break-even accuracy.

**Exit check**: the hero chart is rendered from marts, and the headline sentence is generated with real numbers.

## Week 5: Reproducibility, case study, polish

- [x] **T5.1 `make reproduce`** from a fresh clone on a second machine, or in a clean container, in replay mode. Fix whatever breaks.
- [ ] **T5.2 CI reproduction smoke test** on the 200-ticket fixture. *(workflow written in .github/workflows/ci.yml; not executed on GitHub, nothing pushed)*
- [x] **T5.3 CASE_STUDY.md**, following the template. Apply stop-slop.
- [x] **T5.4 README.** Headline, hero chart, quickstart, architecture diagram, results table with CIs, limitations, and links to Lexicon and Telltale.
- [ ] **T5.5 Stretch** (only if everything above is done): the Banking77 `finance-ops` tenant. *(not started)*
- [x] **T5.6 Handoff to Lexicon.** `lexicon_ro` verified, `manifest.json` published, warehouse snapshot exported (`pg_dump` of `marts`) for Lexicon's frozen eval.

**Exit check**: every box in BUILD_SPEC §12 is ticked.

---

## If you fall behind

Cut in this order: Banking77 → the ROI tornado chart → the legacy-cache comparison → the prompt v3 variant. Never cut: the cluster split, the CIs, replay reproducibility, or the honest baseline.
