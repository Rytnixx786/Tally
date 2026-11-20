"""Study metrics computed from a decisions frame (from marts or from run artifacts, identically)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tally import cost, stats
from tally.config import SETUP_ORDER, RoiCfg, StatsCfg

FLAG_SLICES = {
    "Z": "typos",
    "Q": "colloquial",
    "W": "offensive",
    "K": "keyword-only",
    "M": "morphological",
    "L": "lexical variation",
}
REQUIRED = [
    "setup",
    "split",
    "arm",
    "ticket_id",
    "tenant_id",
    "is_correct",
    "total_cost_usd",
    "total_latency_ms",
    "escalated",
    "valid",
    "gold_intent",
    "gold_category",
    "predicted_intent",
    "predicted_category",
    "flags",
]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Accept fct_outcomes (gold_label/predicted_label) or artifact column names."""
    out = df.rename(columns={"gold_label": "gold_intent", "predicted_label": "predicted_intent"}).copy()
    missing = set(REQUIRED) - set(out.columns)
    if missing:
        raise ValueError(f"decisions frame is missing {sorted(missing)}")
    out["is_correct"] = out["is_correct"].astype(bool)
    out["escalated"] = out["escalated"].astype(bool)
    out["valid"] = out["valid"].astype(bool)
    out["total_cost_usd"] = out["total_cost_usd"].astype(float)
    out["total_latency_ms"] = out["total_latency_ms"].astype(float)
    return out


def primary(df: pd.DataFrame, split: str = "test") -> dict[str, pd.DataFrame]:
    """Per-setup frames on the reported configuration, sorted by ticket and restricted to tickets that
    every present setup answered (paired comparisons need the same tickets)."""
    p = df[(df["split"] == split) & (df["arm"] == "primary")]
    setups = [s for s in SETUP_ORDER if s in set(p["setup"])]
    common: set[str] | None = None
    for s in setups:
        ids = set(p.loc[p["setup"] == s, "ticket_id"])
        common = ids if common is None else common & ids
    keep = sorted(common or set())
    return {
        s: p[(p["setup"] == s) & (p["ticket_id"].isin(keep))].sort_values("ticket_id").reset_index(drop=True)
        for s in setups
    }


def _cols(d: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "correct": d["is_correct"].to_numpy(dtype=float),
        "cost": d["total_cost_usd"].to_numpy(dtype=float),
    }


def _boot(
    cols: dict[str, np.ndarray], fn: Callable[[Mapping[str, np.ndarray]], float], sc: StatsCfg
) -> stats.Estimate:
    return stats.paired_bootstrap(
        cols, fn, resamples=sc.bootstrap_resamples, seed=sc.bootstrap_seed, level=sc.ci_level
    )


def setup_metrics(d: pd.DataFrame, sc: StatsCfg) -> dict[str, Any]:
    cols = _cols(d)
    acc = _boot(cols, lambda c: stats.accuracy(c["correct"]), sc)
    cpc = _boot(cols, lambda c: stats.cost_per_correct(c["cost"], c["correct"]), sc)
    per_1k = _boot(cols, lambda c: float(np.mean(c["cost"]) * 1000), sc)
    pred = d["predicted_intent"].fillna("__invalid__")
    pred_cat = d["predicted_category"].fillna("__invalid__")
    return {
        "n": len(d),
        "accuracy": acc.as_dict(),
        "macro_f1": float(
            f1_score(d["gold_intent"], pred, average="macro", labels=sorted(d["gold_intent"].unique()))
        ),
        "category_accuracy": float(np.mean(pred_cat == d["gold_category"])),
        "cost_per_correct_usd": cpc.as_dict(),
        "cost_per_1k_tickets_usd": per_1k.as_dict(),
        "total_cost_usd": float(d["total_cost_usd"].sum()),
        "p50_latency_ms": float(np.percentile(d["total_latency_ms"], 50)),
        "p95_latency_ms": float(np.percentile(d["total_latency_ms"], 95)),
        "escalation_rate": float(d["escalated"].mean()),
        "invalid_rate": float(1 - d["valid"].mean()),
    }


def pairwise(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    raw: dict[str, stats.McNemar] = {}
    for a, b in combinations(frames, 2):
        raw[f"{a} vs {b}"] = stats.mcnemar_exact(
            frames[a]["is_correct"].to_numpy(bool), frames[b]["is_correct"].to_numpy(bool)
        )
    adj = stats.holm({k: v.p_value for k, v in raw.items()})
    return {
        k: {"first_only_correct": v.b, "second_only_correct": v.c, "p_value": v.p_value, "p_holm": adj[k]}
        for k, v in raw.items()
    }


def headline(
    frames: dict[str, pd.DataFrame], sc: StatsCfg, num: str = "D_cascade", den: str = "C_hosted"
) -> dict[str, Any] | None:
    if num not in frames or den not in frames:
        return None
    cols = {f"{k}_{s}": v for s in (num, den) for k, v in _cols(frames[s]).items()}
    acc_ratio = _boot(
        cols, lambda c: stats.accuracy(c[f"correct_{num}"]) / stats.accuracy(c[f"correct_{den}"]), sc
    )
    cpc_ratio = _boot(
        cols,
        lambda c: (
            stats.cost_per_correct(c[f"cost_{num}"], c[f"correct_{num}"])
            / stats.cost_per_correct(c[f"cost_{den}"], c[f"correct_{den}"])
        ),
        sc,
    )
    return {
        "numerator": num,
        "denominator": den,
        "accuracy_ratio": acc_ratio.as_dict(),
        "cpc_ratio": cpc_ratio.as_dict(),
    }


def slices(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {"tenant": {}, "flags": {}, "intent": {}}
    for s, d in frames.items():
        g = d.groupby("tenant_id").agg(
            n=("is_correct", "size"), correct=("is_correct", "sum"), cost=("total_cost_usd", "sum")
        )
        out["tenant"][s] = {
            t: {
                "n": int(r.n),
                "accuracy": float(r.correct / r.n),
                "cost_per_correct_usd": float(r.cost / r.correct) if r.correct else None,
            }
            for t, r in g.iterrows()
        }
        out["flags"][s] = {}
        for letter, name in FLAG_SLICES.items():
            m = d["flags"].fillna("").str.contains(letter, regex=False)
            if m.sum():
                out["flags"][s][name] = {"n": int(m.sum()), "accuracy": float(d.loc[m, "is_correct"].mean())}
        out["intent"][s] = {k: float(v) for k, v in d.groupby("gold_intent")["is_correct"].mean().items()}
    return out


def roi_table(results: dict[str, Any], cfg: RoiCfg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s, m in results.items():
        acc = m["accuracy"]["point"]
        per_ticket = m["cost_per_1k_tickets_usd"]["point"] / 1000
        r = cost.roi_per_month(acc, per_ticket, cfg)
        out[s] = {
            "net_usd_per_month": r.net_usd,
            "labour_saved_usd": r.labour_saved_usd,
            "misroute_cost_usd": r.misroute_cost_usd,
            "ai_cost_usd": r.ai_cost_usd,
            "break_even_accuracy": cost.break_even_accuracy(per_ticket, cfg),
            "sensitivity": cost.sensitivity_grid(acc, per_ticket, cfg),
        }
    return out


def cache_effect(df: pd.DataFrame) -> dict[str, Any] | None:
    on = df[(df["split"] == "test") & (df["arm"] == "cache_on") & (df["setup"] == "B_local")]
    off = df[(df["split"] == "test") & (df["arm"] == "primary") & (df["setup"] == "B_local")]
    if on.empty or off.empty:
        return None
    m = on.merge(off[["ticket_id", "is_correct", "total_cost_usd"]], on="ticket_id", suffixes=("", "_off"))
    hit = m["any_cache_hit"].astype(bool) if "any_cache_hit" in m else pd.Series(False, index=m.index)
    hits = m[hit]
    return {
        "n": len(m),
        "hit_rate": float(hit.mean()),
        "hit_precision": float(hits["is_correct"].mean()) if len(hits) else None,
        "accuracy_cache_on": float(m["is_correct"].mean()),
        "accuracy_cache_off": float(m["is_correct_off"].mean()),
        "hits_wrong_but_cache_off_right": int(((~hits["is_correct"]) & hits["is_correct_off"]).sum()),
        "cost_cache_on_usd": float(m["total_cost_usd"].sum()),
        "cost_cache_off_usd": float(m["total_cost_usd_off"].sum()),
    }


def compute(df: pd.DataFrame, sc: StatsCfg, roi_cfg: RoiCfg) -> dict[str, Any]:
    df = normalise(df)
    frames = primary(df)
    results = {s: setup_metrics(d, sc) for s, d in frames.items()}
    tune_frames = primary(df, "tune")
    return {
        "test": results,
        "tune": {s: {"n": len(d), "accuracy": float(d["is_correct"].mean())} for s, d in tune_frames.items()},
        "paired_n": len(next(iter(frames.values()))) if frames else 0,
        "pairwise_mcnemar": pairwise(frames),
        "headline": headline(frames, sc),
        "local_vs_baseline": headline(frames, sc, "B_local", "A_tfidf"),
        "slices": slices(frames),
        "roi": roi_table(results, roi_cfg),
        "cache": cache_effect(df),
        "prompt_versions_tune": {
            pv: float(g["is_correct"].mean())
            for pv, g in df[
                (df["split"] == "tune") & (df["setup"] == "B_local") & (df["arm"] != "cache_on")
            ].groupby("prompt_version")
        }
        if "prompt_version" in df
        else {},
    }


def round_floats(obj: Any, sig: int = 7) -> Any:
    """Round to significant figures (tiny costs keep their precision); NaN/inf become null."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, sig) for v in obj]
    return obj
