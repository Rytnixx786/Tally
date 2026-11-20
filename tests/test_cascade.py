from __future__ import annotations

import numpy as np
import pytest

from tally.setups import cascade


def inputs(n: int = 200, seed: int = 0) -> cascade.CascadeInputs:
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0, 1, n)
    b_correct = rng.uniform(size=n) < conf  # confidence is informative
    return cascade.CascadeInputs(
        b_correct=b_correct,
        b_valid=rng.uniform(size=n) > 0.02,
        b_conf_cal=conf,
        b_cost=np.full(n, 1e-6),
        b_latency_ms=np.full(n, 100.0),
        c_correct=rng.uniform(size=n) < 0.9,
        c_cost=np.full(n, 2e-4),
        c_latency_ms=np.full(n, 400.0),
    )


def test_tau_zero_escalates_only_invalid_and_tau_above_one_escalates_all() -> None:
    inp = inputs()
    assert np.array_equal(cascade.escalate_mask(inp, 0.0), ~inp.b_valid)
    assert cascade.escalate_mask(inp, 1.01).all()


def test_cascade_pays_for_both_calls_on_escalation() -> None:
    inp = inputs()
    p = cascade.evaluate(inp, 0.5)
    esc = cascade.escalate_mask(inp, 0.5)
    assert p.total_cost == pytest.approx(inp.b_cost.sum() + inp.c_cost[esc].sum())
    assert p.escalation_rate == pytest.approx(esc.mean())


def test_nan_confidence_always_escalates() -> None:
    inp = inputs(10)
    conf = inp.b_conf_cal.copy()
    conf[0] = np.nan
    inp2 = cascade.CascadeInputs(**{**inp.__dict__, "b_conf_cal": conf})
    assert cascade.escalate_mask(inp2, 1e-9)[0]


def test_choose_tau_respects_accuracy_floor_and_minimises_cpc() -> None:
    inp = inputs(2000)
    pts = cascade.sweep(inp, 0.05)
    c_acc = float(inp.c_correct.mean())
    best, why = cascade.choose_tau(pts, c_acc, 0.9)
    assert best.accuracy >= 0.9 * c_acc
    feasible = [p for p in pts if p.accuracy >= 0.9 * c_acc]
    assert best.cost_per_correct == min(p.cost_per_correct for p in feasible)
    assert "s.t." in why


def test_choose_tau_falls_back_when_floor_unreachable() -> None:
    inp = inputs()
    best, why = cascade.choose_tau(cascade.sweep(inp, 0.1), 1.0, 1.0)
    assert "no tau" in why
    assert best.accuracy == max(p.accuracy for p in cascade.sweep(inp, 0.1))


def test_oracle_dominates_every_threshold() -> None:
    inp = inputs(1000)
    o = cascade.oracle(inp)
    assert all(o.accuracy >= p.accuracy - 1e-12 for p in cascade.sweep(inp, 0.02))


def test_inputs_length_check() -> None:
    inp = inputs(5)
    with pytest.raises(ValueError, match="wrong length"):
        cascade.CascadeInputs(**{**inp.__dict__, "c_cost": np.ones(4)})
