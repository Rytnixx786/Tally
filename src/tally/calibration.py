"""Confidence calibration (ADR-004): binary temperature scaling, ECE and reliability bins.

We calibrate P(prediction is correct) from the model's label confidence p:
    p_T = sigmoid(logit(p) / T)
T is fitted on the tune split by minimising the negative log-likelihood of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

_EPS = 1e-6
FloatArray = NDArray[np.float64]


def _logit(p: FloatArray) -> FloatArray:
    q = np.clip(p, _EPS, 1 - _EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def apply_temperature(p: FloatArray, temperature: float) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-_logit(p) / temperature)), dtype=np.float64)


def nll(p: FloatArray, correct: NDArray[np.bool_]) -> float:
    q = np.clip(p, _EPS, 1 - _EPS)
    y = correct.astype(np.float64)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def fit_temperature(p: FloatArray, correct: NDArray[np.bool_]) -> float:
    if len(p) == 0:
        raise ValueError("cannot calibrate on an empty set")
    if correct.all() or not correct.any():
        return 1.0  # degenerate: nothing to learn
    res = minimize_scalar(
        lambda log_t: nll(apply_temperature(p, float(np.exp(log_t))), correct),
        bounds=(np.log(0.05), np.log(50.0)),
        method="bounded",
    )
    return float(np.exp(res.x))


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float


def reliability(p: FloatArray, correct: NDArray[np.bool_], bins: int = 15) -> list[ReliabilityBin]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    out: list[ReliabilityBin] = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                n=n,
                mean_confidence=float(p[m].mean()) if n else float("nan"),
                accuracy=float(correct[m].mean()) if n else float("nan"),
            )
        )
    return out


def ece(p: FloatArray, correct: NDArray[np.bool_], bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    total = len(p)
    if total == 0:
        return float("nan")
    return float(
        sum(b.n / total * abs(b.accuracy - b.mean_confidence) for b in reliability(p, correct, bins) if b.n)
    )
