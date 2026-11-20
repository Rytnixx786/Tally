# GEMINI.md: Tally

@./AGENTS.md

The import above is the full rule set (Gemini CLI resolves `@file` imports). If your tool does not resolve imports (some Antigravity setups), read `AGENTS.md` manually before doing anything else.

## Context
Tally measures cost per correct outcome for LLM traffic served through Cinch.
Read `OBJECTIVE.md`, then the `docs/BUILD_SPEC.md` section for the task, then the current week in `docs/ROADMAP.md`.

## Hard boundaries
- Write **only** inside `H:\Projects\Tally`. All sibling folders under `H:\Projects` are read-only.
- No secrets in commits. No paid API calls without the explicit spend flag. No destructive commands without approval.

## Antigravity / Gemini workflow
1. Produce an **Implementation Plan artifact** for each ROADMAP task and wait for explicit approval before editing.
2. Produce a **Task List artifact** and keep it current.
3. After the task, produce a **Walkthrough artifact**: files changed, commands run, test + lint output, and any metric delta.
4. One task per checkpoint. Do not chain tasks.
5. Flag (do not absorb) any new dependency, schema change or service.
