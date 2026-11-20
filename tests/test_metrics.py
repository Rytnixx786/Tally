from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tally import metrics


def frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    acc = {"A_tfidf": 0.9, "B_local": 0.85, "C_hosted": 0.95, "D_cascade": 0.93}
    unit = {"A_tfidf": 1e-9, "B_local": 3e-6, "C_hosted": 2e-4, "D_cascade": 5e-5}
    for s in acc:
        for i in range(n):
            ok = rng.uniform() < acc[s]
            rows.append(
                {
                    "setup": s,
                    "split": "test",
                    "arm": "primary",
                    "ticket_id": f"t{i}",
                    "tenant_id": ["support", "billing"][i % 2],
                    "is_correct": ok,
                    "total_cost_usd": unit[s],
                    "total_latency_ms": 100.0 + i,
                    "escalated": s == "D_cascade" and i % 4 == 0,
                    "valid": True,
                    "gold_intent": ["a", "b", "c"][i % 3],
                    "gold_category": "X",
                    "predicted_intent": ["a", "b", "c"][i % 3] if ok else "zzz",
                    "predicted_category": "X" if ok else "Y",
                    "flags": "BLZ" if i % 3 == 0 else "BL",
                    "any_cache_hit": False,
                }
            )
    return pd.DataFrame(rows)


def test_compute_headline_and_tables(exp_cfg, roi_cfg) -> None:  # type: ignore[no-untyped-def]
    m = metrics.compute(frame(), exp_cfg.stats, roi_cfg)
    assert set(m["test"]) == {"A_tfidf", "B_local", "C_hosted", "D_cascade"}
    h = m["headline"]
    assert h["accuracy_ratio"]["lo"] <= h["accuracy_ratio"]["point"] <= h["accuracy_ratio"]["hi"]
    assert h["cpc_ratio"]["point"] == pytest.approx(
        m["test"]["D_cascade"]["cost_per_correct_usd"]["point"]
        / m["test"]["C_hosted"]["cost_per_correct_usd"]["point"]
    )
    assert len(m["pairwise_mcnemar"]) == 6
    assert "typos" in m["slices"]["flags"]["B_local"]
    assert m["roi"]["C_hosted"]["break_even_accuracy"] < 1
    assert m["cache"] is None


def test_headline_absent_without_hosted(exp_cfg, roi_cfg) -> None:  # type: ignore[no-untyped-def]
    df = frame()
    m = metrics.compute(df[df["setup"].isin(["A_tfidf", "B_local"])], exp_cfg.stats, roi_cfg)
    assert m["headline"] is None and m["local_vs_baseline"] is not None


def test_primary_restricts_to_common_tickets() -> None:
    df = frame(50)
    df = df[~((df["setup"] == "C_hosted") & (df["ticket_id"] == "t3"))]
    frames = metrics.primary(metrics.normalise(df))
    assert {len(f) for f in frames.values()} == {49}


def test_normalise_accepts_mart_column_names() -> None:
    df = frame(5).rename(columns={"gold_intent": "gold_label", "predicted_intent": "predicted_label"})
    assert "gold_intent" in metrics.normalise(df)
    with pytest.raises(ValueError, match="missing"):
        metrics.normalise(df.drop(columns=["flags"]))


def test_cache_effect_counts_false_hits() -> None:
    off = frame(20)
    off = off[off["setup"] == "B_local"].assign(is_correct=True)
    on = off.assign(
        arm="cache_on",
        any_cache_hit=[i % 2 == 0 for i in range(len(off))],
        is_correct=[i % 2 == 1 for i in range(len(off))],
    )
    c = metrics.cache_effect(metrics.normalise(pd.concat([off, on])))
    assert c is not None
    assert c["hit_rate"] == 0.5 and c["hit_precision"] == 0.0 and c["hits_wrong_but_cache_off_right"] == 10


def test_round_floats_keeps_tiny_values_and_nulls_nan() -> None:
    assert metrics.round_floats({"a": 1.23456789e-9, "b": float("nan"), "c": [0.123456789]}) == {
        "a": 1.234568e-9,
        "b": None,
        "c": [0.1234568],
    }
