"""Tally dashboard. Reads the public marts with the read-only `lexicon_ro` role, plus run artifacts.

tally dashboard            # or: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tally import cost
from tally.config import ROOT, SETUP_ORDER, load_cost_model, load_roi
from tally.figures import COLORS, LABELS, SHORT
from tally.paths import RUNS_DIR

st.set_page_config(page_title="Tally: cost per correct outcome", layout="wide")
SYMBOLS = {"A_tfidf": "square", "B_local": "circle", "C_hosted": "triangle-up", "D_cascade": "diamond"}
INK, INK_2 = "#0b0b0b", "#52514e"


@st.cache_data(ttl=60)
def q(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    from tally.warehouse import read_sql

    return read_sql(sql, role="lexicon", params=list(params))


def runs() -> list[str]:
    try:
        return list(q("select distinct run_id from marts.fct_outcomes order by 1")["run_id"])
    except Exception as exc:  # pragma: no cover - UI path
        st.error(
            f"Cannot reach the warehouse ({exc}). Start it with `docker compose up -d` and run `tally publish`."
        )
        st.stop()
        return []


def artifact(run_id: str, name: str) -> Any:
    p = RUNS_DIR / run_id / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def layout(fig: go.Figure, title: str, x: str, y: str, logx: bool = False) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 16, "color": INK}},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": INK_2},
        xaxis={"title": x, "type": "log" if logx else "linear", "gridcolor": "#e4e3df"},
        yaxis={"title": y, "gridcolor": "#e4e3df"},
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=460,
    )
    return fig


# ------------------------------------------------------------------ sidebar
all_runs = runs()
run_id = st.sidebar.selectbox("Run", all_runs, index=all_runs.index("main") if "main" in all_runs else 0)
split = st.sidebar.radio(
    "Split", ["test", "tune"], help="Headline numbers use test. Tune is where prompts and τ were chosen."
)
metrics: dict[str, Any] = (
    json.loads((RUNS_DIR / run_id / "metrics.json").read_text())
    if (RUNS_DIR / run_id / "metrics.json").exists()
    else {}
)

outcomes = q(
    """select setup, tenant_id, prompt_version, arm, is_correct, total_cost_usd::float as cost,
              total_latency_ms::float as latency_ms, escalated, any_cache_hit, decided_at, gold_label, flags
       from marts.fct_outcomes where run_id = %s and split = %s""",
    (run_id, split),
)
primary = outcomes[outcomes["arm"] == "primary"]
setups = [s for s in SETUP_ORDER if s in set(primary["setup"])]

st.title("Tally: cost per correct triage")
st.caption(
    f"Run `{run_id}` · split `{split}` · read-only role `lexicon_ro` · every number is a query on `marts.*`."
)

tabs = st.tabs(
    [
        "Overview",
        "Cost vs accuracy",
        "Cost per correct",
        "Calibration & router",
        "Cache",
        "Escalations",
        "ROI",
        "Case study",
    ]
)

# ------------------------------------------------------------------ overview
with tabs[0]:
    if metrics.get("headline") and split == "test":
        h = metrics["headline"]
        st.info(
            f"Router achieved **{h['accuracy_ratio']['point']:.1%}** of the hosted model's accuracy at "
            f"**{h['cpc_ratio']['point']:.1%}** of its cost per correct triage (test split, 95% CIs on the case-study page)."
        )
    cols = st.columns(max(len(setups), 1))
    for col, s in zip(cols, setups, strict=False):
        d = primary[primary["setup"] == s]
        acc = d["is_correct"].mean()
        cpc = d["cost"].sum() / max(d["is_correct"].sum(), 1)
        col.metric(SHORT[s], f"{acc:.1%} accurate", f"${cpc * 1000:.3g} per 1k correct", delta_color="off")
    table = (
        primary.groupby("setup")
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            total_cost_usd=("cost", "sum"),
            p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
            escalation_rate=("escalated", "mean"),
        )
        .reindex(setups)
    )
    table["cost_per_1k_correct_usd"] = (
        primary.groupby("setup")
        .apply(lambda d: d["cost"].sum() / d["is_correct"].sum() * 1000)
        .reindex(setups)
    )
    st.dataframe(
        table.style.format(
            {
                "accuracy": "{:.2%}",
                "total_cost_usd": "${:.5f}",
                "cost_per_1k_correct_usd": "${:.4g}",
                "p95_latency_ms": "{:.0f}",
                "escalation_rate": "{:.1%}",
            }
        ),
        width="stretch",
    )

# ------------------------------------------------------------------ hero
with tabs[1]:
    fig = go.Figure()
    tau = artifact(run_id, "tau")
    if tau and split == "test" and "D_cascade" in setups:
        n = int((primary["setup"] == "D_cascade").sum())
        pts = [p for p in tau["test_sweep"] if p["total_cost"] > 0]
        fig.add_trace(
            go.Scatter(
                x=[p["total_cost"] / n * 1000 for p in pts],
                y=[p["accuracy"] for p in pts],
                mode="lines",
                line={"color": COLORS["D_cascade"], "width": 2},
                opacity=0.55,
                name="D at every τ",
                customdata=[p["tau"] for p in pts],
                hovertemplate="τ=%{customdata:.2f}<br>$%{x:.4g} per 1k<br>%{y:.1%}<extra></extra>",
            )
        )
    for s in setups:
        d = primary[primary["setup"] == s]
        x, y = d["cost"].mean() * 1000, d["is_correct"].mean()
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[SHORT[s]],
                textposition="bottom right",
                marker={
                    "color": COLORS[s],
                    "size": 14,
                    "symbol": SYMBOLS[s],
                    "line": {"color": "#fcfcfb", "width": 2},
                },
                name=LABELS[s],
                hovertemplate=f"{LABELS[s]}<br>$%{{x:.4g}} per 1k tickets<br>%{{y:.2%}}<extra></extra>",
            )
        )
    st.plotly_chart(
        layout(
            fig, "Cost vs accuracy (hover for values)", "Cost per 1,000 tickets (USD, log)", "Accuracy", True
        ),
        width="stretch",
    )
    img = ROOT / "docs" / "assets" / "hero_cost_vs_accuracy.png"
    if img.exists():
        with st.expander("Static hero chart with 95% CIs (README version)"):
            st.image(str(img))

# ------------------------------------------------------------------ cost per correct
with tabs[2]:
    by = st.radio("Break down by", ["tenant_id", "prompt_version"], horizontal=True)
    src = outcomes if by == "prompt_version" else primary
    g = (
        src.groupby(["setup", by])
        .agg(cost=("cost", "sum"), correct=("is_correct", "sum"), n=("is_correct", "size"))
        .reset_index()
    )
    g["cpc_1k"] = g["cost"] / g["correct"].clip(lower=1) * 1000
    fig = go.Figure()
    for s in setups:
        gs = g[g["setup"] == s]
        fig.add_trace(
            go.Bar(
                x=gs[by],
                y=gs["cpc_1k"],
                name=LABELS[s],
                marker_color=COLORS[s],
                hovertemplate="%{x}<br>$%{y:.4g} per 1k correct<extra>" + SHORT[s] + "</extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(
        layout(fig, f"Cost per 1,000 correct triages by {by}", by, "USD (log)").update_yaxes(type="log"),
        width="stretch",
    )
    st.dataframe(g, width="stretch")

# ------------------------------------------------------------------ calibration
with tabs[3]:
    cal = artifact(run_id, "calibration")
    if cal:
        c1, c2 = st.columns(2)
        for col, key, name in (
            (c1, "reliability_raw", "raw"),
            (c2, "reliability_calibrated", f"calibrated (T={cal['temperature']:.2f})"),
        ):
            bins = [b for b in cal[split][key] if b["n"]]
            fig = go.Figure(
                [
                    go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": INK_2}, name="perfect"
                    ),
                    go.Scatter(
                        x=[b["mean_confidence"] for b in bins],
                        y=[b["accuracy"] for b in bins],
                        mode="lines+markers",
                        marker={"size": 9, "color": COLORS["B_local"]},
                        line={"width": 2, "color": COLORS["B_local"]},
                        customdata=[b["n"] for b in bins],
                        name="B local",
                        hovertemplate="conf %{x:.2f}<br>acc %{y:.2f}<br>n=%{customdata}<extra></extra>",
                    ),
                ]
            )
            col.plotly_chart(
                layout(fig, f"Reliability, {name}", "Confidence", "Observed accuracy"),
                width="stretch",
            )
        if tau:
            sw = tau[f"{split}_sweep"]
            fig = go.Figure(
                [
                    go.Scatter(
                        x=[p["tau"] for p in sw],
                        y=[p["accuracy"] for p in sw],
                        mode="lines",
                        name="accuracy",
                        line={"color": COLORS["D_cascade"], "width": 2},
                    )
                ]
            )
            fig.add_vline(
                x=tau["chosen_tau"], line_color=INK_2, annotation_text=f"chosen τ={tau['chosen_tau']:.2f}"
            )
            st.plotly_chart(layout(fig, "Cascade accuracy by τ", "τ", "Accuracy"), width="stretch")
            fig = go.Figure(
                [
                    go.Scatter(
                        x=[p["tau"] for p in sw],
                        y=[p["escalation_rate"] for p in sw],
                        mode="lines",
                        name="escalation rate",
                        line={"color": COLORS["D_cascade"], "width": 2},
                    )
                ]
            )
            st.plotly_chart(
                layout(fig, "Share of tickets escalated by τ", "τ", "Escalation rate"),
                width="stretch",
            )
            st.caption(f"Selection rule (pre-registered): {tau['rule']}")
    else:
        st.info("Calibration appears once setups C and D have run.")

# ------------------------------------------------------------------ cache
with tabs[4]:
    cache = metrics.get("cache")
    if cache:
        c = st.columns(4)
        c[0].metric("Hit rate", f"{cache['hit_rate']:.1%}")
        c[1].metric(
            "Hit precision", f"{cache['hit_precision']:.1%}" if cache["hit_precision"] is not None else "n/a"
        )
        c[2].metric(
            "Accuracy, cache on vs off",
            f"{cache['accuracy_cache_on']:.1%}",
            f"{cache['accuracy_cache_on'] - cache['accuracy_cache_off']:+.1%}",
        )
        c[3].metric(
            "Hits that were wrong (but right without cache)", f"{cache['hits_wrong_but_cache_off_right']:,}"
        )
        st.caption(
            "Cinch's semantic cache compares the whole prompt (system + ticket) with term-frequency cosine ≥ 0.95. "
            "With a long fixed triage prompt, different tickets look alike, so a hit can return another ticket's label."
        )
    else:
        st.info(
            "No cache arm in this run. Run `tally run --cache-arm` with Cinch started with CINCH_SEMANTIC_CACHE=true."
        )

# ------------------------------------------------------------------ escalations
with tabs[5]:
    d = primary[primary["setup"] == "D_cascade"]
    if d.empty:
        st.info("The cascade has not run yet.")
    else:
        t = (
            d.assign(bucket=pd.to_datetime(d["decided_at"]).dt.floor("5min"))
            .groupby("bucket")["escalated"]
            .mean()
            .reset_index()
        )
        fig = go.Figure(
            [
                go.Scatter(
                    x=t["bucket"],
                    y=t["escalated"],
                    mode="lines+markers",
                    line={"color": COLORS["D_cascade"], "width": 2},
                    marker={"size": 8},
                    name="escalation rate",
                )
            ]
        )
        st.plotly_chart(
            layout(fig, "Escalation rate over time (5-minute buckets)", "time", "Escalation rate"),
            width="stretch",
        )
        by_intent = d.groupby("gold_label")["escalated"].mean().sort_values(ascending=False).reset_index()
        fig = go.Figure(
            [
                go.Bar(
                    x=by_intent["escalated"],
                    y=by_intent["gold_label"],
                    orientation="h",
                    marker_color=COLORS["D_cascade"],
                )
            ]
        )
        st.plotly_chart(
            layout(fig, "Which intents the local model is unsure about", "Escalation rate", "").update_layout(
                height=700
            ),
            width="stretch",
        )

# ------------------------------------------------------------------ ROI
with tabs[6]:
    roi_cfg, cost_cfg = load_roi(), load_cost_model()
    st.write(
        "ROI = correct × minutes saved × wage − wrong × misroute minutes × wage − AI cost. **All three labour inputs are assumptions.**"
    )
    c = st.columns(4)
    minutes = c[0].slider(
        "Minutes saved per correct triage", 0.0, 5.0, roi_cfg.minutes_saved_per_correct, 0.25
    )
    wage = c[1].slider("Wage, USD/hour (fully loaded)", 1.0, 40.0, roi_cfg.wage_usd_per_hour, 0.5)
    mis = c[2].slider("Misroute penalty, minutes", 0.0, 15.0, roi_cfg.misroute_penalty_minutes, 0.5)
    volume = c[3].number_input("Tickets per month", 1000, 10_000_000, roi_cfg.tickets_per_month, 1000)
    rows = []
    for s in setups:
        d = primary[primary["setup"] == s]
        acc, per_ticket = d["is_correct"].mean(), d["cost"].mean()
        r = cost.roi(
            volume * acc,
            volume * (1 - acc),
            volume * per_ticket,
            minutes_saved=minutes,
            wage_per_hour=wage,
            misroute_minutes=mis,
        )
        rows.append(
            {
                "setup": LABELS[s],
                "accuracy": acc,
                "AI cost / month": r.ai_cost_usd,
                "labour saved": r.labour_saved_usd,
                "misroute cost": r.misroute_cost_usd,
                "net / month": r.net_usd,
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                "accuracy": "{:.2%}",
                "AI cost / month": "${:,.2f}",
                "labour saved": "${:,.0f}",
                "misroute cost": "${:,.0f}",
                "net / month": "${:,.0f}",
            }
        ),
        width="stretch",
    )
    st.caption(
        f"Local GPU cost method: {cost_cfg.method}, utilisation {cost_cfg.local_gpu.utilisation:.0%} "
        "(config/cost_model.yaml). The report's sensitivity grid covers 27 combinations."
    )

# ------------------------------------------------------------------ case study
with tabs[7]:
    p = ROOT / "docs" / "CASE_STUDY.md"
    if p.exists():
        # Images in the markdown use repo-relative paths; strip them and render the PNGs below instead.
        st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", p.read_text(encoding="utf-8")))
        for img in ("hero_cost_vs_accuracy", "learning_curve", "reliability", "tau_sweep", "slices_by_flag"):
            f = ROOT / "docs" / "assets" / f"{img}.png"
            if f.exists():
                st.image(str(f), caption=img.replace("_", " "))
    else:
        st.info("Run `tally report` to generate the case study.")
