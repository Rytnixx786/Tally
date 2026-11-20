"""`tally report`: metrics.json, figures, docs/RESULTS.md, docs/CASE_STUDY.md and the README headline.

Every number in those documents is rendered from metrics computed here. Nothing is typed by hand.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, StrictUndefined

from tally import figures, metrics
from tally.config import ROOT, SETUP_ORDER, load_cost_model, load_experiment, load_roi
from tally.paths import RUNS_DIR

DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
README = ROOT / "README.md"
START, END = "<!-- tally:results:start -->", "<!-- tally:results:end -->"

OUTCOME_SQL = "select * from marts.fct_outcomes where run_id = %s"


def load_decisions(run_id: str, source: str) -> pd.DataFrame:
    if source == "artifacts":
        df = pd.read_parquet(RUNS_DIR / run_id / "decisions.parquet")
        calls = pd.read_parquet(RUNS_DIR / run_id / "calls.parquet")
        hit = calls.groupby("decision_id")["cache_status"].agg(lambda s: bool((s == "HIT").any()))
        df["any_cache_hit"] = df["decision_id"].map(hit).fillna(False)
        return df
    from tally.warehouse import read_sql

    df = read_sql(OUTCOME_SQL, params=[run_id])
    if df.empty:
        raise RuntimeError(
            f"no rows for run {run_id!r} in marts.fct_outcomes; run `tally publish` and `tally dbt build`"
        )
    return df


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(e: dict[str, float] | float | None, nd: int = 1) -> str:
    if e is None:
        return "n/a"
    if isinstance(e, dict):
        return f"{e['point']:.{nd}%} ({e['lo']:.{nd}%}–{e['hi']:.{nd}%})"
    return f"{e:.{nd}%}"


def usd(e: dict[str, float] | float | None, scale: float = 1.0, sig: int = 3) -> str:
    if e is None:
        return "n/a"
    if isinstance(e, dict):
        return f"${e['point'] * scale:.{sig}g} (${e['lo'] * scale:.{sig}g}–${e['hi'] * scale:.{sig}g})"
    return f"${e * scale:.{sig}g}"


def ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f} ms" if v < 1 else f"{v:,.0f} ms"


def headline_sentence(m: dict[str, Any]) -> str:
    h = m.get("headline")
    if h:
        a, c = h["accuracy_ratio"], h["cpc_ratio"]
        return (
            f"Router achieved **{a['point']:.1%}** (95% CI {a['lo']:.1%}–{a['hi']:.1%}) of the hosted model's accuracy "
            f"at **{c['point']:.1%}** (95% CI {c['lo']:.1%}–{c['hi']:.1%}) of its cost per correct triage."
        )
    t = m["test"]
    if "B_local" in t and "A_tfidf" in t:
        return (
            f"Hosted setup C and the cascade D have not run yet. On the held-out test split, the local "
            f"model reached **{t['B_local']['accuracy']['point']:.1%}** accuracy at "
            f"**${t['B_local']['cost_per_correct_usd']['point'] * 1000:.3g} per 1,000 correct triages**, while the "
            f"no-LLM baseline reached **{t['A_tfidf']['accuracy']['point']:.1%}** at "
            f"**${t['A_tfidf']['cost_per_correct_usd']['point'] * 1000:.2g} per 1,000**."
        )
    return "Results pending."


def results_table(m: dict[str, Any]) -> str:
    rows = [
        "| Setup | Accuracy (95% CI) | Macro-F1 | Cost per 1k tickets | Cost per 1k correct | p95 latency | Escalated |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in SETUP_ORDER:
        r = m["test"].get(s)
        if r is None:
            rows.append(f"| {figures.LABELS[s]} | not run | | | | | |")
            continue
        rows.append(
            f"| {figures.LABELS[s]} | {pct(r['accuracy'])} | {r['macro_f1']:.3f} | {usd(r['cost_per_1k_tickets_usd'])} "
            f"| {usd(r['cost_per_correct_usd'], 1000)} | {ms(r['p95_latency_ms'])} | "
            f"{pct(r['escalation_rate']) if s == 'D_cascade' else '–'} |"
        )
    return "\n".join(rows)


def build(run_id: str, source: str = "warehouse") -> Path:
    exp, roi_cfg, cost_cfg = load_experiment(), load_roi(), load_cost_model()
    run_dir = RUNS_DIR / run_id
    df = load_decisions(run_id, source)
    m = metrics.compute(df, exp.stats, roi_cfg)
    extras = {
        k: _json(run_dir / f"{k}.json")
        for k in ("a_selection", "prompt_selection", "calibration", "tau", "learning_curve", "manifest")
    }
    m["run"] = {
        "run_id": run_id,
        "source": source,
        "chosen_prompt": (extras["manifest"] or {}).get("chosen_prompt"),
        "chosen_tau": (extras["tau"] or {}).get("chosen_tau"),
        "tau_rule": (extras["tau"] or {}).get("rule"),
        "temperature": (extras["calibration"] or {}).get("temperature"),
        "a_chosen_c": (extras["a_selection"] or {}).get("chosen_c"),
        "cost_method": cost_cfg.method,
        "utilisation": cost_cfg.local_gpu.utilisation,
    }
    m["learning_curve"] = extras["learning_curve"]
    if extras["calibration"]:
        m["calibration"] = {k: extras["calibration"][k] for k in ("temperature",)} | {
            s: {k: extras["calibration"][s][k] for k in ("n", "ece_raw", "ece_calibrated")}
            for s in ("tune", "test")
        }
    if extras["tau"]:
        m["oracle"] = extras["tau"]["oracle"]
    m = metrics.round_floats(m)
    out_metrics = run_dir / "metrics.json"
    out_metrics.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig_dir = run_dir / "figures"
    made = figures.hero(m, extras["tau"], fig_dir)
    if extras["learning_curve"]:
        made += figures.learning_curve(extras["learning_curve"], m, fig_dir)
    if extras["calibration"]:
        made += figures.reliability(extras["calibration"], fig_dir)
    if extras["tau"]:
        made += figures.tau_curves(extras["tau"], fig_dir)
    made += figures.flag_slices(m, fig_dir)
    made += figures.tenant_cpc(m, fig_dir)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copyfile(p, ASSETS / p.name)

    ctx = {
        "m": m,
        "headline": headline_sentence(m),
        "table": results_table(m),
        "pct": pct,
        "usd": usd,
        "ms": ms,
        "labels": figures.LABELS,
        "order": SETUP_ORDER,
        "cost": cost_cfg,
        "roi": roi_cfg,
        "exp": exp,
    }
    env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    for template, target in (
        ("RESULTS.template.md", "RESULTS.md"),
        ("CASE_STUDY.template.md", "CASE_STUDY.md"),
    ):
        src = DOCS / "templates" / template
        if src.exists():
            (DOCS / target).write_text(
                env.from_string(src.read_text(encoding="utf-8")).render(**ctx) + "\n", encoding="utf-8"
            )
    update_readme(ctx)
    return run_dir


def findings(m: dict[str, Any]) -> list[str]:
    """Short, generated statements for the README. Each one is conditional on what was measured."""
    out: list[str] = []
    t = m["test"]
    a, b = t.get("A_tfidf"), t.get("B_local")
    if a and b:
        better = "beat" if a["accuracy"]["point"] > b["accuracy"]["point"] else "trailed"
        ratio = b["cost_per_correct_usd"]["point"] / a["cost_per_correct_usd"]["point"]
        out.append(
            f"The no-LLM baseline {better} the 7B model on accuracy ({a['accuracy']['point']:.1%} vs "
            f"{b['accuracy']['point']:.1%}) and was {ratio:,.0f}x cheaper per correct triage."
        )
    lc = m.get("learning_curve") or []
    if lc and b:
        target = b["accuracy"]["point"]
        below = [r for r in lc if r["per_intent"] > 0 and r["accuracy"] <= target]
        above = [r for r in lc if r["per_intent"] > 0 and r["accuracy"] > target]
        if below and above:
            lo, hi = below[-1], above[0]
            out.append(
                f"Labels decide it: with {int(lo['per_intent'])} labelled examples per intent the baseline scores "
                f"{lo['accuracy']:.1%}, below the prompted model's {target:.1%}; with {int(hi['per_intent'])} per "
                f"intent it scores {hi['accuracy']:.1%}."
            )
    c = m.get("cache")
    if c and c.get("hit_precision") is not None:
        out.append(
            f"Cinch's semantic cache, as shipped, answered {c['hit_rate']:.1%} of tickets from cache and only "
            f"{c['hit_precision']:.1%} of those hits carried the right label: accuracy fell from "
            f"{c['accuracy_cache_off']:.1%} to {c['accuracy_cache_on']:.1%} (fix specified in docs/CINCH_CHANGES.md, C3)."
        )
    pv = m.get("prompt_versions_tune") or {}
    if len(pv) > 1:
        best, worst = max(pv, key=lambda k: pv[k]), min(pv, key=lambda k: pv[k])
        out.append(
            f"Two labelled examples per intent in the prompt lifted the local model from {pv[worst]:.1%} to "
            f"{pv[best]:.1%} on tune (`{best}` chosen)."
        )
    if "C_hosted" not in t:
        out.append(
            "Hosted setup C and the cascade D are implemented and tested but not yet run (ADR-015); "
            "`tally run --mode auto --confirm-spend` completes them."
        )
    return out


def update_readme(ctx: dict[str, Any]) -> None:
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return
    m = ctx["m"]
    bullets = "\n".join(f"- {b}" for b in findings(m))
    block = (
        f"{START}\n> {ctx['headline']}\n>\n> Test split, n = {m['paired_n']:,} tickets per setup, run `{m['run']['run_id']}`. "
        f"Every number below is generated by `tally report` from the warehouse.\n\n{ctx['table']}\n\n"
        f"**Key findings** (generated):\n\n{bullets}\n{END}"
    )
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    README.write_text(text, encoding="utf-8")
