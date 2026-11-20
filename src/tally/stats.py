"""Uncertainty for every headline number: paired bootstrap over tickets, exact McNemar, Holm."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def accuracy(correct: FloatArray) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def cost_per_correct(cost: FloatArray, correct: FloatArray) -> float:
    """Ratio of sums over the same decisions (CONTRACTS §5 rule 1). NaN when nothing is correct."""
    n = float(np.sum(correct))
    return float(np.sum(cost) / n) if n > 0 else float("nan")


def paired_bootstrap(
    columns: Mapping[str, FloatArray],
    statistic: Callable[[Mapping[str, FloatArray]], float],
    *,
    resamples: int = 2000,
    seed: int = 7,
    level: float = 0.95,
) -> Estimate:
    """Resample ticket indices once per replicate and apply the same indices to every column, so
    ratios and between-setup comparisons keep their pairing."""
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length (one row per ticket)")
    n = lengths.pop()
    point = statistic(columns)
    if n == 0:
        return Estimate(point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    reps = np.empty(resamples)
    for r in range(resamples):
        idx = rng.integers(0, n, n)
        reps[r] = statistic({k: v[idx] for k, v in columns.items()})
    reps = reps[np.isfinite(reps)]
    alpha = (1 - level) / 2
    lo, hi = np.quantile(reps, [alpha, 1 - alpha]) if len(reps) else (float("nan"), float("nan"))
    return Estimate(point, float(lo), float(hi))


@dataclass(frozen=True)
class McNemar:
    b: int  # first correct, second wrong
    c: int  # first wrong, second correct
    p_value: float


def mcnemar_exact(first_correct: NDArray[np.bool_], second_correct: NDArray[np.bool_]) -> McNemar:
    b = int(np.sum(first_correct & ~second_correct))
    c = int(np.sum(~first_correct & second_correct))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return McNemar(b=b, c=c, p_value=p)


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted


def wilson(successes: int, n: int, level: float = 0.95) -> Estimate:
    from scipy.stats import norm

    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"))
    z = float(norm.ppf(1 - (1 - level) / 2))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Estimate(p, float(centre - half), float(centre + half))
