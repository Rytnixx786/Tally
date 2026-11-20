from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tally import calibration, cost, stats
from tally.config import CostModelCfg, RoiCfg

# ---------------------------------------------------------------- cost model


def test_gpu_hourly_owned_is_amortisation_plus_power(cost_cfg: CostModelCfg) -> None:
    g = cost_cfg.local_gpu
    want = (
        g.purchase_usd / (g.lifetime_years * 8760)
        + (g.board_power_w + g.system_overhead_w) / 1000 * g.electricity_usd_per_kwh
    )
    assert cost.gpu_hourly_usd(cost_cfg) == pytest.approx(want)
    # measured power overrides the configured board power
    assert cost.gpu_hourly_usd(cost_cfg, measured_power_w=g.board_power_w + 100) > want


def test_hand_computed_local_cost(cost_cfg: CostModelCfg) -> None:
    cfg = cost_cfg.model_copy(
        update={
            "local_gpu": cost_cfg.local_gpu.model_copy(
                update={
                    "purchase_usd": 876.0,
                    "lifetime_years": 1.0,
                    "board_power_w": 200.0,
                    "system_overhead_w": 0.0,
                    "electricity_usd_per_kwh": 0.5,
                    "utilisation": 0.5,
                }
            )
        }
    )
    # hourly = 876/8760 + 0.2 kW * 0.5 = 0.1 + 0.1 = 0.2 USD; 3600 req/h at 50% util -> 0.2/1800
    tp = cost.Throughput(n_requests=3600, wall_seconds=3600, concurrency=8)
    assert cost.local_cost_per_request_mean(cfg, tp) == pytest.approx(0.2 / 1800)


def test_cloud_method_ignores_utilisation(cost_cfg: CostModelCfg) -> None:
    tp = cost.Throughput(1000, 100.0, 8)
    a = cost.local_cost_per_request_mean(cost_cfg, tp, method="cloud_gpu_hour", utilisation=0.1)
    b = cost.local_cost_per_request_mean(cost_cfg, tp, method="cloud_gpu_hour", utilisation=0.9)
    assert a == b == pytest.approx(cost_cfg.local_gpu.alt_cloud_gpu_usd_per_hour / 36000)


@settings(max_examples=50)
@given(st.floats(0.05, 0.95), st.floats(0.01, 0.04))
def test_more_utilisation_never_costs_more(cost_cfg: CostModelCfg, u: float, du: float) -> None:  # type: ignore[no-untyped-def]
    tp = cost.Throughput(500, 60.0, 8)
    assert cost.local_cost_per_request_mean(
        cost_cfg, tp, utilisation=u + du
    ) <= cost.local_cost_per_request_mean(cost_cfg, tp, utilisation=u)


def test_throughput_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="positive wall time"):
        _ = cost.Throughput(0, 10.0, 1).requests_per_hour


@given(st.floats(0, 100), st.lists(st.floats(0, 1000), min_size=1, max_size=50))
def test_apportion_preserves_total(total: float, weights: list[float]) -> None:
    parts = cost.apportion(total, weights)
    assert sum(parts) == pytest.approx(total, rel=1e-9, abs=1e-9)
    assert all(p >= 0 for p in parts)


def test_hosted_cost_and_fail_closed(cost_cfg: CostModelCfg) -> None:
    assert cost.hosted_cost(cost_cfg, "openai", "gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    # half the prompt from the provider cache at half price
    assert cost.hosted_cost(cost_cfg, "openai", "gpt-4o-mini", 1_000_000, 0, 500_000) == pytest.approx(0.1125)
    # cached tokens never exceed input tokens; models without a cached price bill cached tokens in full
    assert cost.hosted_cost(cost_cfg, "openai", "gpt-4o-mini", 100, 0, 10_000) == pytest.approx(
        100 * 0.075 / 1e6
    )
    assert cost.hosted_cost(cost_cfg, "openai", "gpt-5-nano", 1_000_000, 0, 999) == pytest.approx(0.05)
    with pytest.raises(KeyError):
        cost.hosted_cost(cost_cfg, "openai", "unpriced", 1, 1)


def test_roi_and_break_even(roi_cfg: RoiCfg) -> None:
    be = cost.break_even_accuracy(0.001, roi_cfg)
    assert 0 < be < 1
    assert cost.roi_per_month(be, 0.001, roi_cfg).net_usd == pytest.approx(0, abs=1e-6)
    assert cost.roi_per_month(be + 0.05, 0.001, roi_cfg).net_usd > 0
    grid = cost.sensitivity_grid(0.9, 0.001, roi_cfg)
    assert len(grid) == 27 and {"net_usd_per_month", "wage_usd_per_hour"} <= set(grid[0])


def test_misroute_penalty_makes_cheap_wrong_setups_unprofitable(roi_cfg: RoiCfg) -> None:
    good = cost.roi_per_month(0.95, 0.0002, roi_cfg).net_usd
    cheap_bad = cost.roi_per_month(0.40, 0.0, roi_cfg).net_usd
    assert good > 0 > cheap_bad


# ---------------------------------------------------------------- calibration


def test_temperature_fixes_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.4, 0.95, 4000)
    correct = rng.uniform(size=4000) < true_p
    stated = calibration.apply_temperature(true_p, 0.4)  # sharpen -> overconfident
    t = calibration.fit_temperature(stated, correct)
    assert t > 1.5
    assert calibration.ece(calibration.apply_temperature(stated, t), correct) < calibration.ece(
        stated, correct
    )


def test_reliability_bins_cover_all_points_and_degenerate_temperature() -> None:
    p = np.array([0.05, 0.5, 0.51, 0.99, 1.0])
    y = np.array([False, True, False, True, True])
    bins = calibration.reliability(p, y, bins=10)
    assert sum(b.n for b in bins) == len(p)
    assert calibration.fit_temperature(np.array([0.9, 0.8]), np.array([True, True])) == 1.0
    with pytest.raises(ValueError, match="empty"):
        calibration.fit_temperature(np.array([]), np.array([], dtype=bool))


def test_ece_zero_when_perfectly_calibrated() -> None:
    p = np.repeat([0.25, 0.75], 400)
    y = np.concatenate([np.arange(400) < 100, np.arange(400) < 300])
    assert calibration.ece(p, y, bins=4) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- statistics


def test_bootstrap_ci_brackets_point_and_is_seeded() -> None:
    rng = np.random.default_rng(1)
    cols = {"correct": (rng.uniform(size=500) < 0.8).astype(float), "cost": rng.uniform(0, 1e-3, 500)}
    a = stats.paired_bootstrap(cols, lambda c: stats.cost_per_correct(c["cost"], c["correct"]), resamples=500)
    b = stats.paired_bootstrap(cols, lambda c: stats.cost_per_correct(c["cost"], c["correct"]), resamples=500)
    assert a == b
    assert a.lo <= a.point <= a.hi


def test_cost_per_correct_is_ratio_of_sums_not_mean_of_ratios() -> None:
    cost_ = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert stats.cost_per_correct(cost_, correct) == 2.0
    assert np.isnan(stats.cost_per_correct(cost_, np.zeros(4)))


def test_bootstrap_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        stats.paired_bootstrap({"a": np.ones(3), "b": np.ones(4)}, lambda c: 0.0)


def test_mcnemar_exact_known_value() -> None:
    first = np.array([True] * 10 + [False] * 2 + [True] * 50)
    second = np.array([False] * 10 + [True] * 2 + [True] * 50)
    r = stats.mcnemar_exact(first, second)
    assert (r.b, r.c) == (10, 2)
    assert r.p_value == pytest.approx(0.0386, abs=1e-3)
    assert stats.mcnemar_exact(first, first).p_value == 1.0


def test_holm_is_monotone_and_capped() -> None:
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.09) and adj["b"] == pytest.approx(0.09)
    assert adj["d"] == pytest.approx(0.5)
    assert all(v <= 1 for v in stats.holm({"x": 0.9, "y": 0.8}).values())


def test_wilson_interval() -> None:
    e = stats.wilson(80, 100)
    assert e.lo < 0.8 < e.hi and e.lo > 0.7
    assert np.isnan(stats.wilson(0, 0).point)
