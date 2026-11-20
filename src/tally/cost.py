"""Cost model (BUILD_SPEC §6) and ROI (§7.5). The only place cost formulas live.

Local GPU cost per request is a share of the GPU's effective hourly cost:
    hourly        = purchase / (lifetime_years * 8760) + (board_W + overhead_W)/1000 * tariff    [owned]
                  = alt_cloud_gpu_usd_per_hour                                                  [cloud]
    per request   = hourly / (requests_per_hour_at_measured_concurrency * utilisation)
Utilisation < 1 charges idle hours to the paid work (fully loaded cost). The run total is split
across requests in proportion to their token count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

from tally.config import CostModelCfg, RoiCfg

HOURS_PER_YEAR = 8760.0
Method = Literal["owned_gpu", "cloud_gpu_hour"]


@dataclass(frozen=True)
class Throughput:
    """Measured during a live run and recorded next to the responses."""

    n_requests: int
    wall_seconds: float
    concurrency: int
    mean_power_w: float | None = None

    @property
    def requests_per_hour(self) -> float:
        if self.wall_seconds <= 0 or self.n_requests <= 0:
            raise ValueError("throughput needs positive wall time and request count")
        return self.n_requests / self.wall_seconds * 3600.0


def gpu_hourly_usd(
    cfg: CostModelCfg, method: Method | None = None, measured_power_w: float | None = None
) -> float:
    m = method or cfg.method
    g = cfg.local_gpu
    if m == "cloud_gpu_hour":
        return g.alt_cloud_gpu_usd_per_hour
    amortised = g.purchase_usd / (g.lifetime_years * HOURS_PER_YEAR)
    board = measured_power_w if measured_power_w is not None else g.board_power_w
    power = (board + g.system_overhead_w) / 1000.0 * g.electricity_usd_per_kwh
    return amortised + power


def local_cost_per_request_mean(
    cfg: CostModelCfg, tp: Throughput, method: Method | None = None, utilisation: float | None = None
) -> float:
    m = method or cfg.method
    util = (
        1.0
        if m == "cloud_gpu_hour"
        else (utilisation if utilisation is not None else cfg.local_gpu.utilisation)
    )
    return gpu_hourly_usd(cfg, m, tp.mean_power_w) / (tp.requests_per_hour * util)


def apportion(total: float, weights: Sequence[float]) -> list[float]:
    """Split a total across items in proportion to weights (equal split if all weights are zero)."""
    s = float(sum(weights))
    if not weights:
        return []
    if s <= 0:
        return [total / len(weights)] * len(weights)
    return [total * (w / s) for w in weights]  # normalise first: avoids underflow with subnormal weights


def hosted_cost(
    cfg: CostModelCfg,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """List-price cost. Cached prompt tokens use the cached price when the price book has one; ignoring the
    discount would overstate the hosted model's cost and flatter the cascade."""
    price = cfg.price_for(provider, model)  # raises on unknown models: fail closed
    cached = min(max(cached_input_tokens, 0), input_tokens)
    cached_rate = (
        price.cached_input_per_mtok if price.cached_input_per_mtok is not None else price.input_per_mtok
    )
    return (
        (input_tokens - cached) * price.input_per_mtok
        + cached * cached_rate
        + output_tokens * price.output_per_mtok
    ) / 1e6


def cpu_cost(cfg: CostModelCfg, seconds: float) -> float:
    return cfg.cpu.usd_per_hour * seconds / 3600.0


# ------------------------------------------------------------------ ROI


@dataclass(frozen=True)
class RoiResult:
    labour_saved_usd: float
    misroute_cost_usd: float
    ai_cost_usd: float

    @property
    def net_usd(self) -> float:
        return self.labour_saved_usd - self.misroute_cost_usd - self.ai_cost_usd


def roi(
    n_correct: float,
    n_wrong: float,
    ai_cost_usd: float,
    *,
    minutes_saved: float,
    wage_per_hour: float,
    misroute_minutes: float,
) -> RoiResult:
    per_min = wage_per_hour / 60.0
    return RoiResult(
        labour_saved_usd=n_correct * minutes_saved * per_min,
        misroute_cost_usd=n_wrong * misroute_minutes * per_min,
        ai_cost_usd=ai_cost_usd,
    )


def roi_per_month(accuracy: float, cost_per_ticket: float, cfg: RoiCfg, **overrides: float) -> RoiResult:
    n = cfg.tickets_per_month
    return roi(
        n * accuracy,
        n * (1 - accuracy),
        n * cost_per_ticket,
        minutes_saved=overrides.get("minutes_saved", cfg.minutes_saved_per_correct),
        wage_per_hour=overrides.get("wage_per_hour", cfg.wage_usd_per_hour),
        misroute_minutes=overrides.get("misroute_minutes", cfg.misroute_penalty_minutes),
    )


def break_even_accuracy(cost_per_ticket: float, cfg: RoiCfg) -> float:
    """Accuracy at which net ROI is zero: a*s - (1-a)*m - c/w = 0 (per minute of wage)."""
    per_min = cfg.wage_usd_per_hour / 60.0
    s, m = cfg.minutes_saved_per_correct, cfg.misroute_penalty_minutes
    if per_min == 0 or s + m == 0:
        return float("nan")
    return (m + cost_per_ticket / per_min) / (s + m)


def sensitivity_grid(accuracy: float, cost_per_ticket: float, cfg: RoiCfg) -> list[dict[str, float]]:
    s = cfg.sensitivity
    rows = []
    for ms, wage, mis in product(
        s.minutes_saved_per_correct, s.wage_usd_per_hour, s.misroute_penalty_minutes
    ):
        r = roi_per_month(
            accuracy, cost_per_ticket, cfg, minutes_saved=ms, wage_per_hour=wage, misroute_minutes=mis
        )
        rows.append(
            {
                "minutes_saved_per_correct": ms,
                "wage_usd_per_hour": wage,
                "misroute_penalty_minutes": mis,
                "net_usd_per_month": r.net_usd,
            }
        )
    return rows
