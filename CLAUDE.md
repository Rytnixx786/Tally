# CLAUDE.md: Tally

@AGENTS.md

The import above is the full rule set. This file only adds Claude Code specifics.

## Context
Tally measures cost per correct outcome for LLM traffic served through Cinch.
Start every session by reading `OBJECTIVE.md`, then the `docs/BUILD_SPEC.md` section for the current task, then the current week in `docs/ROADMAP.md`.

## Hard boundaries (repeated here in case the import is not loaded)
- Edit files **only** inside `H:\Projects\Tally`. Every other folder under `H:\Projects` is read-only reference material.
- Never commit secrets, never spend paid-API money without an explicit flag, never run destructive commands without approval.
- Plan first, one ROADMAP task per checkpoint, and stop to report with test and lint output.

## Claude Code workflow
- Use plan mode for any task that touches the experiment design, a public schema, or adds a dependency.
- Keep the session task list in sync with the ROADMAP task you are on.
- Prefer subagents for wide read-only searches across the sibling projects (Cinch, Ares, Tally, ...) so their file dumps stay out of the main context.
- Before writing README or case-study prose, apply `H:\Projects\SKILLS_Main_Added\stop-slop-main\SKILL.md`.
- When a decision is made, append an ADR to `docs/DECISIONS.md` in the same change.
- Update the "Commands" section of `AGENTS.md` whenever a Make target changes.
