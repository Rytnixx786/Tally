"""Deterministic ticket -> tenant assignment from category weights (ADR-007)."""

from __future__ import annotations

import hashlib

import pandas as pd

from tally.config import TenantsCfg


def _unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_tenant(ticket_id: str, category: str, cfg: TenantsCfg) -> str:
    names = sorted(cfg.tenants)
    weights = [cfg.tenants[n].weights.get(category, 0.0) for n in names]
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"no tenant has a positive weight for category {category!r}")
    u = _unit_interval(f"{cfg.seed}:{ticket_id}") * total
    acc = 0.0
    for name, w in zip(names, weights, strict=True):
        acc += w
        if u < acc:
            return name
    return names[-1]


def assign(df: pd.DataFrame, cfg: TenantsCfg) -> pd.Series:
    return pd.Series(
        [assign_tenant(t, c, cfg) for t, c in zip(df["ticket_id"], df["category"], strict=True)],
        index=df.index,
        name="tenant_id",
    )
