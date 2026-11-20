"""Setup D: local first, escalate to hosted when calibrated confidence < tau or output is invalid.

Everything here is pure array maths over per-ticket results of B and C on the same tickets, so the
tau sweep, the chosen operating point and the oracle bound are exact and cheap to recompute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CascadeInputs:
    b_correct: BoolArray
    b_valid: BoolArray
    b_conf_cal: FloatArray  # calibrated P(correct); NaN where unavailable
    b_cost: FloatArray
    b_latency_ms: FloatArray
    c_correct: BoolArray
    c_cost: FloatArray
    c_latency_ms: FloatArray

    def __post_init__(self) -> None:
        n = len(self.b_correct)
        for name in (
            "b_valid",
            "b_conf_cal",
            "b_cost",
            "b_latency_ms",
            "c_correct",
            "c_cost",
            "c_latency_ms",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has the wrong length")


def escalate_mask(inp: CascadeInputs, tau: float) -> BoolArray:
    conf = np.nan_to_num(inp.b_conf_cal, nan=0.0)
    return np.asarray(~inp.b_valid | (conf < tau), dtype=np.bool_)


@dataclass(frozen=True)
class CascadePoint:
    tau: float
    accuracy: float
    total_cost: float
    cost_per_correct: float
    escalation_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "tau": self.tau,
            "accuracy": self.accuracy,
            "total_cost": self.total_cost,
            "cost_per_correct": self.cost_per_correct,
            "escalation_rate": self.escalation_rate,
        }


def evaluate(inp: CascadeInputs, tau: float) -> CascadePoint:
    esc = escalate_mask(inp, tau)
    correct = np.where(esc, inp.c_correct, inp.b_correct)
    cost = inp.b_cost + np.where(esc, inp.c_cost, 0.0)
    n_correct = float(correct.sum())
    return CascadePoint(
        tau=float(tau),
        accuracy=float(correct.mean()),
        total_cost=float(cost.sum()),
        cost_per_correct=float(cost.sum() / n_correct) if n_correct else float("nan"),
        escalation_rate=float(esc.mean()),
    )


def sweep(inp: CascadeInputs, step: float) -> list[CascadePoint]:
    taus = np.round(np.arange(0.0, 1.0 + step / 2, step), 6)
    return [evaluate(inp, float(t)) for t in taus]


def choose_tau(
    points: list[CascadePoint], c_accuracy: float, min_relative_accuracy: float
) -> tuple[CascadePoint, str]:
    """Pre-registered objective (ADR-010): lowest cost per correct with accuracy >= r x C's accuracy.
    Falls back to the most accurate point if no tau meets the constraint (and says so)."""
    floor = min_relative_accuracy * c_accuracy
    feasible = [p for p in points if p.accuracy >= floor and np.isfinite(p.cost_per_correct)]
    if feasible:
        best = min(feasible, key=lambda p: (p.cost_per_correct, -p.accuracy, p.tau))
        return best, f"min cost per correct s.t. accuracy >= {min_relative_accuracy:.2f} x C ({floor:.4f})"
    best = max(points, key=lambda p: (p.accuracy, -p.cost_per_correct))
    return best, "no tau met the accuracy floor; chose the most accurate point"


def oracle(inp: CascadeInputs) -> CascadePoint:
    """Escalate exactly when B is wrong: the ceiling for any router over these two models."""
    esc = ~inp.b_correct
    correct = np.where(esc, inp.c_correct, inp.b_correct)
    cost = inp.b_cost + np.where(esc, inp.c_cost, 0.0)
    n_correct = float(correct.sum())
    return CascadePoint(
        tau=float("nan"),
        accuracy=float(correct.mean()),
        total_cost=float(cost.sum()),
        cost_per_correct=float(cost.sum() / n_correct) if n_correct else float("nan"),
        escalation_rate=float(esc.mean()),
    )
