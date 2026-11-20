# AGENTS.md: Tally

Rules for any AI coding agent (Claude Code, Gemini CLI / Antigravity, Codex, Cursor) working in this repository. `CLAUDE.md` and `GEMINI.md` import this file. If they disagree with it, this file wins.

## Project in one line
Tally measures **cost per correct outcome** for LLM traffic served through Cinch, and compares four setups (TF-IDF baseline, local Qwen via Cinch, a hosted API model, a confidence-based cascade) on held-out support-ticket triage.

Read `OBJECTIVE.md` first, then the section of `docs/BUILD_SPEC.md` that covers your task.

## Hard boundaries
1. **Only edit files inside `H:\Projects\Tally`.** Every other folder under `H:\Projects` (Cinch, Ares, Nexus_Researcher, MaaSheba, the portfolio, all of them) is read-only. Reading them for context is encouraged.
2. Changes Tally needs in Cinch go into `docs/CINCH_CHANGES.md` as a spec. Never patch Cinch from this repo.
3. Never commit secrets. API keys live in `.env` (gitignored); `.env.example` lists the names only.
4. Never spend real money without saying so first. Any command that calls a paid API prints the estimated cost and requires `--confirm-spend` or `TALLY_ALLOW_SPEND=1`. Default runs use replay mode.
5. Never run destructive commands (`rm -rf`, `DROP SCHEMA`, `docker volume rm`, force-push) without explicit approval.

## Non-negotiable process
1. **Plan before code.** Each task starts with a short plan (files, tests, risks). Wait for approval on anything that adds a dependency, changes a public mart column, or changes the experiment design.
2. **Stop-and-report checkpoints.** After each ROADMAP task, stop. Report what changed, show test and lint output, and wait for review. Do not chain tasks.
3. **No silent scope expansion.** If a task needs a new service, a schema change or a new dependency that is not in the plan, flag it.
4. **Read existing skills before reinventing.** Check `H:\Projects\SKILLS_Main_Added` for relevant guidance, for example `Agent-Skills-for-Context-Engineering-main/skills/evaluation` and `advanced-evaluation`, and `stop-slop-main` for README prose.

## Experiment integrity (the rules that make the numbers worth anything)
- The **test split is sealed**. Prompts, few-shot examples, τ and any other tuning use `dev` or `tune` only. `tests/test_leakage.py` enforces this. Never weaken that test.
- Splits are generated once from a fixed seed, and their row ids and sha256 are committed to `data/splits/manifest.json`. Regenerating with a different result fails CI.
- Every headline number carries a bootstrap 95% CI (≥ 2,000 resamples, seeded). Setup-vs-setup accuracy claims carry a McNemar p-value.
- Temperature 0 for all LLM setups. Record the model revision or snapshot, the vLLM version and the hosted model version string in the run manifest.
- Report the TF-IDF baseline honestly, even if it beats the LLMs.
- Every number in the README, the case study or the dashboard must trace to a query on a `marts.*` table or a file in `reports/runs/<run_id>/`. No hand-typed numbers.

## Engineering standards
- Python 3.12, dependencies managed with `uv` (`pyproject.toml` + `uv.lock`). Pin versions.
- Layout: `src/tally/` (package), `dbt/` (dbt project), `dashboard/` (Streamlit), `config/` (YAML), `tests/`, `scripts/`, `reports/`.
- Lint: `ruff check` and `ruff format`. Types: `mypy --strict` on `src/tally`.
- Tests: `pytest`, ≥ 85% line coverage on `src/tally`. Every bug fix ships with the test that would have caught it. dbt models ship with `dbt test` coverage: `unique` + `not_null` on every grain, `relationships` on every FK, and `accepted_values` on enums.
- Config over constants. Prices, wages, GPU assumptions, τ grids and tenant mixes live in `config/*.yaml`, validated with pydantic.
- Telemetry follows OTel GenAI semantic conventions. Do not invent `gen_ai.*` attributes; Cinch-specific ones use the `cinch.` prefix and Tally-specific ones use `tally.`.
- No prompt or completion text in spans. The eval harness stores inputs and outputs in its own run artifacts.

## Commands (keep this list current)
The `tally` CLI is canonical (it works on Windows without make); `Makefile` targets wrap it.
```
uv sync --all-extras                       # make setup
docker compose up -d --build postgres tally-ingest otel-collector   # make up
docker compose --profile model up -d       # make up-model: vLLM + Cinch (GPU)
uv run tally data                          # download (pinned), cluster, verify sealed split
uv run tally run --mode replay             # all setups from recordings (default; no spend)
uv run tally run --mode auto               # record what is missing (hosted needs --confirm-spend)
uv run tally publish                       # raw.* + OTLP spans -> collector -> ingest -> Postgres
uv run tally dbt build                     # models + contracts + data tests
uv run tally report                        # metrics.json, figures, docs/RESULTS.md, CASE_STUDY.md, README block
uv run tally dashboard                     # Streamlit on :8501 (read-only role)
uv run tally reproduce                     # data -> run (replay) -> publish -> dbt -> report
uv run pytest -m "not integration"         # make test (coverage gate 85%)
uv run ruff check src tests dashboard && uv run mypy   # make lint
```

## Git
- Conventional commits: `feat:`, `fix:`, `exp:` (experiment runs), `dbt:`, `docs:`, `chore:`, `test:`.
- Feature branches. No direct commits to `main`, even solo.
- PR descriptions state what changed, why, and the metric delta if it touches the harness, the cost model or the dbt marts.
- Agents do not `git push` unless asked.

## Writing standard (README, case study, docs)
- No filler, no "blazing fast", no "revolutionary". Numbers or nothing.
- State assumptions next to the numbers that depend on them.
- Run prose through the `stop-slop` rules before publishing.
