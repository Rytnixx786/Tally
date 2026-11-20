"""Static figures for the README and case study (matplotlib, light surface).

Palette: the dataviz skill's validated categorical slots 1-4 in fixed order (A, B, C, D), checked
with validate_palette.js (adjacent CVD ΔE >= 9.1). Two slots are below 3:1 contrast, so every
series is direct-labelled and also has its own marker shape (identity is never colour-alone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"A_tfidf": "#2a78d6", "B_local": "#eb6834", "C_hosted": "#1baf7a", "D_cascade": "#eda100"}
MARKERS = {"A_tfidf": "s", "B_local": "o", "C_hosted": "^", "D_cascade": "D"}
LABELS = {
    "A_tfidf": "A  TF-IDF + LR (no LLM)",
    "B_local": "B  Qwen2.5-7B local via Cinch",
    "C_hosted": "C  hosted API model",
    "D_cascade": "D  cascade B→C",
}
SHORT = {"A_tfidf": "A TF-IDF", "B_local": "B local", "C_hosted": "C hosted", "D_cascade": "D cascade"}


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w: float = 8.0, h: float = 5.0, ncols: int = 1) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(1, ncols, figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig: Figure, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    for p in paths:
        fig.savefig(
            p, bbox_inches="tight", facecolor=SURFACE, metadata={"Date": None} if p.suffix == ".svg" else None
        )
    plt.close(fig)
    return paths


def hero(metrics: dict[str, Any], tau: dict[str, Any] | None, out: Path) -> list[Path]:
    """Cost per 1,000 tickets (log x) vs accuracy, 95% CIs on both axes, tau frontier for D."""
    fig, ax = _fig(8.6, 5.4)
    _style(ax)
    test = metrics["test"]
    if tau and "test_sweep" in tau and "D_cascade" in test:
        pts = [p for p in tau["test_sweep"] if p["total_cost"] > 0]
        n = test["D_cascade"]["n"]
        ax.plot(
            [p["total_cost"] / n * 1000 for p in pts],
            [p["accuracy"] for p in pts],
            color=COLORS["D_cascade"],
            linewidth=2,
            alpha=0.55,
            zorder=2,
            label="D at every τ (test)",
        )
        o = tau.get("oracle", {}).get("test")
        if o:
            ax.scatter(
                [o["total_cost"] / n * 1000],
                [o["accuracy"]],
                s=70,
                facecolors="none",
                edgecolors=INK_2,
                linewidths=1.5,
                zorder=4,
            )
            ax.annotate(
                "oracle router (ceiling)",
                (o["total_cost"] / n * 1000, o["accuracy"]),
                xytext=(8, 6),
                textcoords="offset points",
                color=INK_2,
                fontsize=8.5,
            )
    for s, m in test.items():
        x, y = m["cost_per_1k_tickets_usd"], m["accuracy"]
        ax.errorbar(
            x["point"],
            y["point"],
            xerr=[[x["point"] - x["lo"]], [x["hi"] - x["point"]]],
            yerr=[[y["point"] - y["lo"]], [y["hi"] - y["point"]]],
            fmt=MARKERS[s],
            color=COLORS[s],
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            elinewidth=1.4,
            capsize=3,
            zorder=5,
            label=LABELS[s],
        )
        ax.annotate(
            f"{SHORT[s]}\n{y['point']:.1%} · ${x['point']:.3g}/1k",
            (x["point"], y["point"]),
            xytext=(10, -14),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 tickets (USD, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Intent accuracy, held-out test split", color=INK, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Where each setup lands: cost vs accuracy (95% CIs)", loc="left", color=INK, fontsize=12)
    ax.margins(x=0.12, y=0.12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center left")
    return _save(fig, out, "hero_cost_vs_accuracy")


def learning_curve(curve: list[dict[str, float]], metrics: dict[str, Any], out: Path) -> list[Path]:
    fig, ax = _fig(7.6, 4.6)
    _style(ax)
    xs = [r["n_train"] for r in curve]
    ax.plot(
        xs,
        [r["accuracy"] for r in curve],
        marker=MARKERS["A_tfidf"],
        color=COLORS["A_tfidf"],
        linewidth=2,
        markersize=8,
        label="A TF-IDF + LR, trained on n examples",
    )
    for r in curve:
        lab = "all" if r["per_intent"] == 0 else f"{int(r['per_intent'])}/intent"
        ax.annotate(
            lab,
            (r["n_train"], r["accuracy"]),
            xytext=(4, -14),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    for s in ("B_local", "C_hosted"):
        if s in metrics["test"]:
            acc = metrics["test"][s]["accuracy"]["point"]
            ax.axhline(acc, color=COLORS[s], linewidth=2, linestyle="--")
            ax.annotate(
                f"{SHORT[s]} (no training labels): {acc:.1%}",
                (xs[0], acc),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=8.5,
                color=INK,
            )
    ax.margins(y=0.1)
    ax.set_xscale("log")
    ax.set_xlabel("Labelled training tickets (log scale)", color=INK)
    ax.set_ylabel("Test accuracy", color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(
        "How many labels does the baseline need to beat a prompted LLM?", loc="left", color=INK, fontsize=12
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    return _save(fig, out, "learning_curve")


def reliability(calib: dict[str, Any], out: Path) -> list[Path]:
    fig, axes = _fig(9.0, 4.2, ncols=2)
    for ax, key, title in (
        (axes[0], "reliability_raw", "Raw logprob confidence"),
        (axes[1], "reliability_calibrated", f"Temperature-scaled (T = {calib['temperature']:.2f})"),
    ):
        _style(ax)
        bins = [b for b in calib["test"][key] if b["n"] > 0]
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=":")
        ax.plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            markersize=8,
            color=COLORS["B_local"],
            linewidth=2,
        )
        ece_key = "ece_raw" if key == "reliability_raw" else "ece_calibrated"
        ax.set_title(f"{title}\nECE {calib['test'][ece_key]:.3f} (test)", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Confidence", color=INK)
        ax.set_ylabel("Observed accuracy", color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    return _save(fig, out, "reliability")


def tau_curves(tau: dict[str, Any], out: Path) -> list[Path]:
    fig, axes = _fig(9.0, 4.0, ncols=2)
    for ax, key, title in (
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "escalation_rate", "Escalation rate"),
    ):
        _style(ax)
        for split, ls in (("tune_sweep", "-"), ("test_sweep", "--")):
            pts = tau[split]
            ax.plot(
                [p["tau"] for p in pts],
                [p[key] for p in pts],
                color=COLORS["D_cascade"],
                linewidth=2,
                linestyle=ls,
                label=split.split("_")[0],
            )
        ax.axvline(tau["chosen_tau"], color=INK_2, linewidth=1)
        ax.annotate(
            f"chosen τ = {tau['chosen_tau']:.2f} (on tune)",
            (tau["chosen_tau"], 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("τ (escalate when calibrated confidence < τ)", color=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, out, "tau_sweep")


def flag_slices(metrics: dict[str, Any], out: Path) -> list[Path]:
    """Dot plot: accuracy per language-variation slice, one marker per setup."""
    flags = metrics["slices"]["flags"]
    setups = list(flags)
    names = sorted({n for s in setups for n in flags[s]})
    fig, ax = _fig(7.6, 0.55 * len(names) + 1.6)
    _style(ax)
    for i, n in enumerate(names):
        for s in setups:
            if n in flags[s]:
                ax.scatter(
                    flags[s][n]["accuracy"],
                    i,
                    marker=MARKERS[s],
                    color=COLORS[s],
                    s=60,
                    zorder=3,
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    label=LABELS[s] if i == 0 else None,
                )
    ax.set_yticks(range(len(names)), [f"{n} (n={flags[setups[0]].get(n, {}).get('n', 0)})" for n in names])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Test accuracy within slice", color=INK)
    ax.set_title(
        "Where each setup breaks: Bitext language-variation slices", loc="left", color=INK, fontsize=12
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)
    return _save(fig, out, "slices_by_flag")


def tenant_cpc(metrics: dict[str, Any], out: Path) -> list[Path]:
    ten = metrics["slices"]["tenant"]
    setups = [s for s in ten if s != "A_tfidf"]
    tenants = sorted({t for s in setups for t in ten[s]})
    fig, ax = _fig(7.6, 0.6 * len(tenants) + 1.8)
    _style(ax)
    for i, t in enumerate(tenants):
        for s in setups:
            v = ten[s].get(t, {}).get("cost_per_correct_usd")
            if v:
                ax.scatter(
                    v * 1000,
                    i,
                    marker=MARKERS[s],
                    color=COLORS[s],
                    s=60,
                    zorder=3,
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    label=LABELS[s] if i == 0 else None,
                )
    ax.set_yticks(range(len(tenants)), tenants)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 correct triages (USD, log scale)", color=INK)
    ax.set_title("Cost per correct triage by tenant", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)
    return _save(fig, out, "tenant_cost_per_correct")
